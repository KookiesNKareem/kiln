//! Consistency checks over results (05 §3.8 `kiln trace validate`, 03 §8, 06 §6.3). Every finding is a
//! structured diagnostic; an empty list means the result is consistent.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;

use crate::interval::{Corner, Interval};
use crate::provenance::Tier;
use crate::result::{EvalResult, Status};
use crate::sim::{CheckStatus, EnergyBreakdown, SimResult};
use crate::{RESULT_SCHEMA, SIM_SCHEMA, SUM_REL_TOL};

fn close(a: f64, b: f64) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a - b).abs() <= SUM_REL_TOL * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

fn finite(out: &mut Vec<Diagnostic>, path: &str, x: f64) {
    if !x.is_finite() {
        out.push(Diagnostic::error("E-TRACE-NUMBER", format!("non-finite value {x}")).at(path));
    }
}

fn nonneg(out: &mut Vec<Diagnostic>, path: &str, x: f64) {
    if x < 0.0 {
        out.push(
            Diagnostic::error(
                "E-TRACE-NEGATIVE",
                format!("negative physical quantity {x}"),
            )
            .at(path),
        );
    }
}

fn quantity(out: &mut Vec<Diagnostic>, path: &str, x: f64) {
    finite(out, path, x);
    nonneg(out, path, x);
}

fn energy(out: &mut Vec<Diagnostic>, path: &str, e: &EnergyBreakdown) {
    for (name, x) in [
        ("compute_j", e.compute_j),
        ("nmp_j", e.nmp_j),
        ("static_j", e.static_j),
        ("conversion_j", e.conversion_j),
        ("padding_j", e.padding_j),
        ("total_j", e.total_j),
    ] {
        quantity(out, &format!("{path}.{name}"), x);
    }
    for (k, x) in e.memory_j.iter().chain(&e.link_j) {
        quantity(out, &format!("{path}[{k}]"), *x);
    }
}

fn interval(out: &mut Vec<Diagnostic>, path: &str, i: &Interval) {
    if !i.is_valid() {
        out.push(
            Diagnostic::error(
                "E-TRACE-INTERVAL",
                format!("invalid interval [{}, {}, {}]", i.low, i.central, i.high),
            )
            .at(path)
            .hint("intervals are finite and ordered low <= central <= high"),
        );
    }
}

pub fn check_sim(r: &SimResult) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let p = format!("sim[{}:{:?}]", r.phase, r.corner).to_lowercase();
    if r.schema != SIM_SCHEMA {
        out.push(
            Diagnostic::error(
                "E-TRACE-SCHEMA",
                format!("schema {:?}, expected {SIM_SCHEMA:?}", r.schema),
            )
            .at(&p),
        );
    }
    for (name, x) in [
        ("makespan_s", r.makespan_s),
        ("t_a0_s", r.t_a0_s),
        ("t_a2_s", r.t_a2_s),
        ("power.avg_w", r.power.avg_w),
        ("power.peak_windowed_w", r.power.peak_windowed_w),
        ("power.cap_w", r.power.cap_w.unwrap_or(0.0)),
    ] {
        quantity(&mut out, &format!("{p}.{name}"), x);
    }
    energy(&mut out, &format!("{p}.energy"), &r.energy);
    let avg_ok = if r.makespan_s > 0.0 {
        close(r.power.avg_w, r.energy.total_j / r.makespan_s)
    } else {
        r.power.avg_w == 0.0 && r.energy.total_j == 0.0
    };
    if !avg_ok {
        out.push(
            Diagnostic::error(
                "E-TRACE-POWER",
                format!(
                    "average power {} W, energy {} J and makespan {} s are inconsistent",
                    r.power.avg_w, r.energy.total_j, r.makespan_s
                ),
            )
            .at(format!("{p}.power.avg_w")),
        );
    }
    if r.makespan_s < r.t_a0_s * (1.0 - SUM_REL_TOL) {
        out.push(
            Diagnostic::error(
                "E-FLOOR-I15",
                format!(
                    "makespan {} s below roofline A0 {} s",
                    r.makespan_s, r.t_a0_s
                ),
            )
            .at(format!("{p}.makespan_s")),
        );
    }
    if r.makespan_s < r.t_a2_s * (1.0 - SUM_REL_TOL) {
        let code = if r.tier == Tier::B {
            "E-FLOOR-I9"
        } else {
            "E-TRACE-TIER-A"
        };
        out.push(
            Diagnostic::error(
                code,
                format!("makespan {} s below floor A2 {} s", r.makespan_s, r.t_a2_s),
            )
            .at(format!("{p}.makespan_s")),
        );
    }
    if !close(r.energy.total_j, r.energy.component_sum()) {
        out.push(
            Diagnostic::error(
                "E-FLOOR-I6",
                format!(
                    "energy total {} J != sum of components {} J",
                    r.energy.total_j,
                    r.energy.component_sum()
                ),
            )
            .at(format!("{p}.energy")),
        );
    }
    let attributed = r.bottleneck.attributed_s();
    if !close(attributed, r.makespan_s) {
        out.push(
            Diagnostic::error(
                "E-TRACE-ATTRIBUTION",
                format!(
                    "time_by_binding sums to {attributed} s, makespan is {} s",
                    r.makespan_s
                ),
            )
            .at(format!("{p}.bottleneck.time_by_binding")),
        );
    }
    for o in &r.ops {
        let op = format!("{p}.ops[{}]", o.op);
        if !(o.start_s >= 0.0
            && o.start_s <= o.end_s
            && o.end_s <= r.makespan_s * (1.0 + SUM_REL_TOL))
        {
            out.push(
                Diagnostic::error(
                    "E-TRACE-SPAN",
                    format!(
                        "op span [{}, {}] s outside [0, makespan {}]",
                        o.start_s, o.end_s, r.makespan_s
                    ),
                )
                .at(&op),
            );
        }
        if o.macs_issued < o.macs_useful {
            out.push(
                Diagnostic::error(
                    "E-FLOOR-I5",
                    format!("issued MACs {} < useful {}", o.macs_issued, o.macs_useful),
                )
                .at(&op),
            );
        }
        let window = o
            .group
            .and_then(|g| r.groups.iter().find(|x| x.group == g))
            .map_or(o.time_s(), |g| (g.end_s - g.start_s).max(o.time_s()));
        for (k, f) in o.floors.iter().enumerate() {
            let path = format!("{op}.floors[{k}]");
            quantity(&mut out, &path, f.seconds);
            if f.seconds > window * (1.0 + SUM_REL_TOL) {
                out.push(
                    Diagnostic::error(
                        "E-FLOOR-I15",
                        format!(
                            "{:?} floor {} s above op time {window} s",
                            f.kind, f.seconds
                        ),
                    )
                    .at(path),
                );
            }
        }
        energy(&mut out, &format!("{op}.energy"), &o.energy);
        if !close(o.energy.total_j, o.energy.component_sum()) {
            out.push(
                Diagnostic::error("E-FLOOR-I6", "op energy total != sum of components")
                    .at(format!("{op}.energy")),
            );
        }
    }
    for g in &r.groups {
        if !(g.start_s >= 0.0
            && g.start_s <= g.end_s
            && g.end_s <= r.makespan_s * (1.0 + SUM_REL_TOL))
        {
            out.push(
                Diagnostic::error(
                    "E-TRACE-SPAN",
                    format!(
                        "group span [{}, {}] s outside [0, makespan {}]",
                        g.start_s, g.end_s, r.makespan_s
                    ),
                )
                .at(format!("{p}.groups[{}]", g.group)),
            );
        }
    }
    for res in &r.resources {
        let path = format!("{p}.resources[{}]", res.resource);
        for (name, x) in [
            ("busy_s", res.busy_s),
            ("stall_s", res.stall_s),
            ("bytes", res.bytes),
            ("energy_j", res.energy_j),
        ] {
            quantity(&mut out, &format!("{path}.{name}"), x);
        }
        let util_ok = if r.makespan_s > 0.0 {
            close(res.utilization, res.busy_s / r.makespan_s)
        } else {
            res.utilization == 0.0
        };
        if !util_ok
            || !(0.0..=1.0 + SUM_REL_TOL).contains(&res.utilization)
            || res.busy_s > r.makespan_s * (1.0 + SUM_REL_TOL)
        {
            out.push(
                Diagnostic::error(
                    "E-FLOOR-I3",
                    format!(
                        "busy {} s, utilization {} over makespan {} s",
                        res.busy_s, res.utilization, r.makespan_s
                    ),
                )
                .at(path),
            );
        }
    }
    for c in r.invariants.failures() {
        let mut d = Diagnostic::error(
            c.id.code(),
            if c.message.is_empty() {
                "invariant failed"
            } else {
                &c.message
            },
        );
        if let Some(path) = &c.path {
            d = d.at(path);
        }
        out.push(d);
    }
    out
}

/// Consistency checks over a `.kiln` trace (05 §3.8): table references, sorted unique resource paths, op
/// envelopes, span order and lane overlap, limiter shares, per-op energy attribution, floorplan geometry and
/// headline intervals.
pub fn check_trace(t: &crate::trace::Trace) -> Vec<Diagnostic> {
    use crate::trace::NONE_U32;
    let mut out = Vec::new();
    let nres = t.resources.len() as u32;
    let nops = t.ops.len() as u32;
    let bad = |code: &str, msg: String| Diagnostic::error(code, msg);
    for w in t.resources.windows(2) {
        if w[0].path >= w[1].path {
            out.push(bad(
                "E-TRACE-ORDER",
                format!("resources not sorted by unique path at {:?}", w[1].path),
            ));
        }
    }
    for (i, r) in t.resources.iter().enumerate() {
        if r.parent.is_some_and(|p| p >= nres || p as usize == i) || r.chip >= nres {
            out.push(
                bad(
                    "E-TRACE-REF",
                    format!("resource {} has a bad parent or chip index", r.path),
                )
                .at(&r.path),
            );
        }
    }
    let envelope = |t0: i64, t1: i64| t0 >= 0 && t1 >= t0;
    for (i, o) in t.ops.iter().enumerate() {
        if !envelope(o.t_start, o.t_end) {
            out.push(
                bad(
                    "E-TRACE-SPAN",
                    format!(
                        "op {i} {} has envelope [{}, {}]: negative or ends before it starts",
                        o.path, o.t_start, o.t_end
                    ),
                )
                .at(&o.path),
            );
        }
        if !o.flops.is_finite() || o.flops < 0.0 || !o.energy_j.is_finite() || o.energy_j < 0.0 {
            out.push(
                bad(
                    "E-TRACE-NUMBER",
                    format!("op {} has non-finite flops or energy", o.path),
                )
                .at(&o.path),
            );
        }
    }
    for g in &t.groups {
        if !envelope(g.t_start, g.t_end) {
            out.push(bad(
                "E-TRACE-SPAN",
                format!(
                    "group {} has envelope [{}, {}]: negative or ends before it starts",
                    g.group, g.t_start, g.t_end
                ),
            ));
        }
    }
    for c in &t.collectives {
        if !envelope(c.t_start, c.t_end) {
            out.push(bad(
                "E-TRACE-SPAN",
                format!(
                    "collective {} has envelope [{}, {}]: negative or ends before it starts",
                    c.name, c.t_start, c.t_end
                ),
            ));
        }
    }
    let mut prev: Option<&crate::trace::SpanRow> = None;
    for s in &t.spans {
        if s.resource >= nres || (s.op != NONE_U32 && s.op >= nops) {
            out.push(bad(
                "E-TRACE-REF",
                format!(
                    "span references resource {} / op {} out of range",
                    s.resource, s.op
                ),
            ));
            continue;
        }
        if s.dur < 0 || s.t_start < 0 {
            out.push(bad(
                "E-TRACE-SPAN",
                format!(
                    "negative span start or duration on {}",
                    t.resources[s.resource as usize].path
                ),
            ));
        }
        let Some(end) = s.t_start.checked_add(s.dur) else {
            out.push(bad(
                "E-TRACE-SPAN",
                format!(
                    "span end overflows on {}",
                    t.resources[s.resource as usize].path
                ),
            ));
            continue;
        };
        if s.op != NONE_U32 {
            let o = &t.ops[s.op as usize];
            if s.t_start < o.t_start || end > o.t_end {
                out.push(
                    bad(
                        "E-TRACE-SPAN",
                        format!("span of {} lies outside the op envelope", o.path),
                    )
                    .at(&o.path),
                );
            }
        }
        if let Some(p) = prev {
            let key = |x: &crate::trace::SpanRow| (x.resource, x.lane, x.t_start);
            if key(p) > key(s) {
                out.push(bad(
                    "E-TRACE-ORDER",
                    "spans not sorted by (resource, lane, t_start)".into(),
                ));
            } else if (p.resource, p.lane) == (s.resource, s.lane) && p.t_start + p.dur > s.t_start
            {
                out.push(bad(
                    "E-TRACE-SPAN",
                    format!(
                        "overlapping spans in lane {} of {}",
                        s.lane, t.resources[s.resource as usize].path
                    ),
                ));
            }
        }
        prev = Some(s);
    }
    let mut rates: BTreeMap<u32, Vec<(i64, f64)>> = BTreeMap::new();
    for s in &t.spans {
        let bw = t
            .resources
            .get(s.resource as usize)
            .and_then(|r| r.peak_bw_bps);
        if !(s.bytes.is_finite() && s.bytes >= 0.0) {
            out.push(bad(
                "E-TRACE-NUMBER",
                format!("span bytes {} on resource {}", s.bytes, s.resource),
            ));
            continue;
        }
        // Tier A estimated spans carry their op's total bytes, not traffic on the span's resource.
        let estimated = s.flags & crate::trace::span_flags::ESTIMATED != 0;
        if estimated || bw.is_none() || s.bytes == 0.0 || s.dur < 0 {
            continue;
        }
        let Some(end) = s.t_start.checked_add(s.dur) else {
            continue;
        };
        if s.dur == 0 {
            out.push(bad(
                "E-TRACE-BANDWIDTH",
                format!(
                    "zero-length span moves {} B on resource {}",
                    s.bytes, s.resource
                ),
            ));
            continue;
        }
        let rate = s.bytes / s.dur as f64;
        let e = rates.entry(s.resource).or_default();
        e.push((s.t_start, rate));
        e.push((end, -rate));
    }
    for (r, mut ev) in rates {
        let res = &t.resources[r as usize];
        let cap = res.peak_bw_bps.unwrap_or(0.0) * t.tick_s();
        ev.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut cur = 0.0;
        let mut peak = 0.0f64;
        for (_, d) in ev {
            cur += d;
            peak = peak.max(cur);
        }
        if peak
            .partial_cmp(&(cap * (1.0 + SUM_REL_TOL)))
            .is_none_or(|o| o.is_gt())
        {
            out.push(
                bad(
                    "E-TRACE-BANDWIDTH",
                    format!(
                        "spans on {} move {} B/s, above peak_bw_bps {}",
                        res.path,
                        peak / t.tick_s(),
                        res.peak_bw_bps.unwrap_or(0.0)
                    ),
                )
                .at(&res.path),
            );
        }
    }
    let mut share = vec![0.0; t.ops.len()];
    let mut has = vec![false; t.ops.len()];
    for l in &t.limiters {
        if l.op >= nops || l.resource.is_some_and(|r| r >= nres) {
            out.push(bad(
                "E-TRACE-REF",
                format!("limiter references op {} out of range", l.op),
            ));
            continue;
        }
        share[l.op as usize] += l.share;
        has[l.op as usize] = true;
    }
    for (i, (s, h)) in share.iter().zip(&has).enumerate() {
        if *h && !close(*s, 1.0) {
            out.push(bad(
                "E-TRACE-ATTRIBUTION",
                format!("limiter shares of op {} sum to {s}", t.ops[i].path),
            ));
        }
    }
    let mut e = vec![0.0; t.ops.len()];
    let mut seen = vec![false; t.ops.len()];
    for a in &t.aggregates_op_resource {
        if a.op >= nops || a.resource >= nres {
            out.push(bad(
                "E-TRACE-REF",
                format!(
                    "aggregates_op_resource references op {} / resource {} out of range",
                    a.op, a.resource
                ),
            ));
            continue;
        }
        if !a.energy_j.is_finite() || a.energy_j < 0.0 {
            out.push(bad(
                "E-TRACE-NUMBER",
                format!(
                    "aggregates_op_resource energy {} J of op {} / resource {} is not a finite nonnegative number",
                    a.energy_j, a.op, a.resource
                ),
            ));
        }
        e[a.op as usize] += a.energy_j;
        seen[a.op as usize] = true;
    }
    for (i, o) in t.ops.iter().enumerate() {
        let d = (e[i] - o.energy_j).abs();
        if seen[i] && (d.is_nan() || d > SUM_REL_TOL * o.energy_j.abs().max(1e-30) * 10.0) {
            out.push(
                bad(
                    "E-TRACE-ENERGY",
                    format!(
                        "energy components of {} sum to {} J, op total {} J",
                        o.path, e[i], o.energy_j
                    ),
                )
                .at(&o.path),
            );
        }
    }
    for f in &t.floorplan {
        if f.resource >= nres
            || ![f.x_um, f.y_um, f.w_um, f.h_um]
                .iter()
                .all(|x| x.is_finite())
            || f.w_um < 0.0
            || f.h_um < 0.0
        {
            out.push(bad(
                "E-TRACE-FLOORPLAN",
                format!("bad floorplan rect for resource {}", f.resource),
            ));
        }
    }
    let h = &t.manifest.headline;
    for (name, i) in [
        ("latency_s", &h.latency_s),
        ("tokens_per_s", &h.tokens_per_s),
        ("energy_j", &h.energy_j),
        ("power_w", &h.power_w),
        ("area_mm2", &h.area_mm2),
        ("score", &h.score),
    ] {
        if let Some(i) = i {
            interval(&mut out, &format!("headline.{name}"), i);
            if name != "score" {
                nonneg(&mut out, &format!("headline.{name}"), i.low);
            }
        }
    }
    out
}

pub fn check_result(r: &EvalResult) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if r.schema != RESULT_SCHEMA {
        out.push(Diagnostic::error(
            "E-TRACE-SCHEMA",
            format!("schema {:?}, expected {RESULT_SCHEMA:?}", r.schema),
        ));
    }
    finite(&mut out, "score", r.score);
    if r.status != Status::Ok && r.score > 0.0 {
        out.push(
            Diagnostic::error(
                "E-TRACE-SCORE",
                format!("status {:?} with positive score {}", r.status, r.score),
            )
            .at("score")
            .hint("score is 0 (or negative in graded mode) unless status is ok"),
        );
    }
    if let Some(i) = &r.score_interval {
        interval(&mut out, "score_interval", i);
        if i.central != r.score && r.status == Status::Ok && r.score != i.low {
            out.push(
                Diagnostic::warning(
                    "W-TRACE-SCORE-BASIS",
                    "score equals neither score_interval.central nor .low",
                )
                .at("score"),
            );
        }
    }
    if let Some(rs) = &r.score_realistic {
        finite(&mut out, "score_realistic.score", rs.score);
        interval(&mut out, "score_realistic.interval", &rs.interval);
    }
    if let Some(sc) = &r.score_components {
        for (ph, i) in &sc.phases {
            interval(&mut out, &format!("score_components.phases[{ph}]"), i);
        }
    }
    for ph in &r.phases {
        let p = format!("phases[{}]", ph.phase);
        for (name, i) in [
            ("time_s", &ph.time_s),
            ("tokens_per_s", &ph.tokens_per_s),
            ("energy_j", &ph.energy_j),
            ("tokens_per_j", &ph.tokens_per_j),
            ("avg_power_w", &ph.avg_power_w),
            ("clock_hz", &ph.clock_hz),
        ] {
            interval(&mut out, &format!("{p}.{name}"), i);
            nonneg(&mut out, &format!("{p}.{name}"), i.low);
        }
        if r.status == Status::Ok && r.score != 0.0 && !ph.scope.is_scored() {
            out.push(
                Diagnostic::error(
                    "E-TRACE-SCOPE",
                    "a scored result contains an isolated-op phase (scope: op)",
                )
                .at(format!("{p}.scope"))
                .hint("scores come from whole steps (03 §4.9)"),
            );
        }
        if let Some(f) = ph
            .floors
            .iter()
            .find(|f| f.seconds > ph.time_s.central * (1.0 + SUM_REL_TOL))
        {
            out.push(
                Diagnostic::error(
                    "E-FLOOR-I15",
                    format!(
                        "{:?} floor {} s above phase time {} s",
                        f.kind, f.seconds, ph.time_s.central
                    ),
                )
                .at(format!("{p}.floors")),
            );
        }
        if let Some(c) = r
            .sim
            .iter()
            .find(|c| c.phase == ph.phase && c.corner == Corner::Central)
        {
            for (name, summary, central) in [
                ("time_s", ph.time_s.central, c.makespan_s),
                ("energy_j", ph.energy_j.central, c.energy.total_j),
                ("avg_power_w", ph.avg_power_w.central, c.power.avg_w),
            ] {
                if !close(summary, central) {
                    out.push(
                        Diagnostic::error(
                            "E-TRACE-PHASE",
                            format!(
                                "phase {name} central {summary} != its central simulation's {central}"
                            ),
                        )
                        .at(format!("{p}.{name}")),
                    );
                }
            }
        }
        for (k, x) in &ph.bound_breakdown {
            if !(x.is_finite() && (-1e-6..=1.0 + 1e-6).contains(x)) {
                out.push(
                    Diagnostic::error(
                        "E-TRACE-ATTRIBUTION",
                        format!("bound_breakdown fraction {x} outside [0, 1]"),
                    )
                    .at(format!("{p}.bound_breakdown[{k}]")),
                );
            }
        }
        let bound: f64 = ph.bound_breakdown.values().sum();
        if !ph.bound_breakdown.is_empty() && (bound - 1.0).abs() > 1e-6 {
            out.push(
                Diagnostic::error(
                    "E-TRACE-ATTRIBUTION",
                    format!("bound_breakdown sums to {bound}, expected 1"),
                )
                .at(format!("{p}.bound_breakdown")),
            );
        }
    }
    let failed = r.invariants.as_ref().is_some_and(|i| !i.passed())
        || r.sim.iter().any(|s| {
            s.invariants
                .checks
                .iter()
                .any(|c| c.status == CheckStatus::Fail)
        });
    if failed && r.status != Status::FloorViolation {
        out.push(
            Diagnostic::error(
                "E-TRACE-STATUS",
                format!("invariant failure but status {:?}", r.status),
            )
            .at("status")
            .hint("any invariant failure sets status = floor_violation (06 §6.3)"),
        );
    }
    for s in &r.sim {
        out.extend(check_sim(s));
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use kiln_ir::common::{Id, content_hash};
    use proptest::prelude::*;

    use super::*;
    use crate::interval::{Corner, IntervalMethod};
    use crate::provenance::{Provenance, TrustLevel};
    use crate::result::*;
    use crate::sim::*;

    fn id(s: &str) -> Id {
        Id::new(s).unwrap()
    }

    pub(crate) fn provenance() -> Provenance {
        Provenance {
            kiln_version: crate::KILN_VERSION.into(),
            git_hash: "0123abcd".into(),
            design_hash: "hw1-00000000000000000000000000000000".into(),
            workload_hash: "wl1-00000000000000000000000000000000".into(),
            calibration_hash: "cal1-null".into(),
            calibration_id: Some("null".into()),
            mapping_hash: None,
            options_hash: None,
            tier: Tier::A,
            seeds: vec![0],
            trust_level: TrustLevel::Uncalibrated,
            window: Some(3),
            chunk_bytes: None,
            flags: BTreeMap::new(),
        }
    }

    pub(crate) fn sim(corner: Corner, makespan: f64) -> SimResult {
        let energy = EnergyBreakdown {
            compute_j: 3.0,
            memory_j: BTreeMap::from([("hbm".into(), 1.5)]),
            static_j: 0.5,
            ..Default::default()
        }
        .with_total();
        SimResult {
            schema: SIM_SCHEMA.into(),
            provenance: provenance(),
            phase: id("decode_b1"),
            tier: Tier::A,
            scope: Scope::Step,
            corner,
            makespan_s: makespan,
            t_a0_s: makespan * 0.5,
            t_a2_s: makespan * 0.9,
            clocks: vec![ClockSample {
                domain: id("chip0.core"),
                t_s: 0.0,
                hz: 1.41e9,
            }],
            energy: energy.clone(),
            power: PowerSummary {
                avg_w: energy.total_j / makespan,
                peak_windowed_w: 400.0,
                cap_w: Some(400.0),
                throttled: false,
            },
            ops: vec![OpResult {
                op: id("layers.block.qkv"),
                layer: Some(0),
                start_s: 0.0,
                end_s: makespan,
                binding: Binding::Dram {
                    resource: id("chip0.hbm.ch0"),
                },
                runner_up: Some(RunnerUp {
                    binding: Binding::Overhead {
                        overhead: OverheadKind::Launch,
                    },
                    ratio: 0.2,
                }),
                macs_useful: 100,
                macs_issued: 128,
                bytes_by_level: vec![LevelBytes {
                    level: "hbm".into(),
                    bytes: 4096,
                }],
                energy,
                target: Target::Host,
                host_vs_nmp_s: None,
                group: Some(0),
                floors: vec![Floor {
                    kind: FloorKind::MemoryLevel,
                    path: Some("chip0.hbm".into()),
                    seconds: makespan * 0.5,
                }],
            }],
            resources: vec![ResourceResult {
                resource: id("chip0.hbm.ch0"),
                kind: ResourceKind::DramChannel,
                busy_s: makespan * 0.8,
                stall_s: 0.0,
                bytes: 4096.0,
                macs: 0,
                energy_j: 1.5,
                utilization: 0.8,
                peak_queue: None,
                class: None,
            }],
            groups: vec![],
            collectives: vec![],
            bottleneck: Bottleneck {
                time_by_binding: BTreeMap::from([
                    (BindingClass::Dram, makespan * 0.75),
                    (BindingClass::Overhead, makespan * 0.25),
                ]),
                top_resources: vec![TopResource {
                    resource: id("chip0.hbm.ch0"),
                    utilization: 0.8,
                    shadow_price: 1.0,
                }],
                slack: vec![],
                summary: "decode_b1: 75% of time bound by chip0.hbm".into(),
            },
            invariants: InvariantReport {
                checks: vec![InvariantCheck {
                    id: InvariantId::I1,
                    status: CheckStatus::Pass,
                    margin: Some(0.1),
                    value: None,
                    limit: None,
                    unit: Some("s".into()),
                    path: None,
                    message: String::new(),
                }],
            },
            cost_model: CostModelSummary::default(),
            calibration: CalibrationContribution {
                set_hash: "cal1-null".into(),
                ..Default::default()
            },
            trace: None,
        }
    }

    pub(crate) fn result() -> EvalResult {
        let time = Interval::new(0.012, 0.0125, 0.014).unwrap();
        EvalResult {
            schema: RESULT_SCHEMA.into(),
            status: Status::Ok,
            score: 1.05,
            score_interval: Some(Interval::new(0.95, 1.05, 1.1).unwrap()),
            score_components: None,
            score_realistic: None,
            interval: IntervalInfo {
                method: IntervalMethod::Corners,
                ..Default::default()
            },
            stage_reached: Stage::S3,
            tier: Some(Tier::A),
            phases: vec![PhaseResult {
                phase: id("decode_b1"),
                scope: Scope::Step,
                time_s: time,
                tokens_per_s: time.recip_scaled(1.0),
                energy_j: Interval::new(4.0, 5.0, 6.0).unwrap(),
                tokens_per_j: Interval::new(0.16, 0.2, 0.25).unwrap(),
                avg_power_w: Interval::new(380.0, 400.0, 410.0).unwrap(),
                clock_hz: Interval::new(1.2e9, 1.3e9, 1.41e9).unwrap(),
                floors: vec![Floor {
                    kind: FloorKind::Compute,
                    path: None,
                    seconds: 0.001,
                }],
                roofline_frac: 0.8,
                bound_breakdown: BTreeMap::from([
                    ("compute".into(), 0.25),
                    ("mem:chip0.hbm".into(), 0.75),
                ]),
                per_layer: vec![],
                trusted: false,
            }],
            ops: vec![],
            physical: None,
            features: BTreeMap::from([
                ("power_w".into(), Feature::Scalar(400.0)),
                ("energy_split".into(), Feature::Vector(vec![0.5, 0.4, 0.1])),
            ]),
            violations: vec![],
            errors: vec![],
            warnings: vec![],
            audit: Audit::default(),
            trace: None,
            provenance: provenance(),
            timing: Timing::default(),
            calibration: None,
            invariants: None,
            sim: vec![sim(Corner::Central, 0.0125)],
        }
    }

    #[test]
    fn consistent_result_passes() {
        assert_eq!(check_result(&result()), vec![]);
    }

    #[test]
    fn detects_inconsistencies() {
        let mut r = result();
        r.status = Status::Invalid;
        r.phases[0].time_s.low = 1.0;
        r.sim[0]
            .bottleneck
            .time_by_binding
            .insert(BindingClass::Link, 1.0);
        r.sim[0].makespan_s = r.sim[0].t_a0_s * 0.5;
        r.sim[0].invariants.checks[0].status = CheckStatus::Fail;
        let codes: Vec<_> = check_result(&r).into_iter().map(|d| d.code).collect();
        for c in [
            "E-TRACE-SCORE",
            "E-TRACE-INTERVAL",
            "E-TRACE-STATUS",
            "E-TRACE-ATTRIBUTION",
            "E-FLOOR-I15",
            "E-FLOOR-I1",
        ] {
            assert!(codes.iter().any(|x| x == c), "missing {c} in {codes:?}");
        }
    }

    #[test]
    fn timing_does_not_affect_deterministic_hash() {
        let a = result();
        let mut b = a.clone();
        b.timing.stages_s.insert(Stage::S3, 0.042);
        assert_eq!(a.deterministic_hash(), b.deterministic_hash());
        b.score = 1.06;
        assert_ne!(a.deterministic_hash(), b.deterministic_hash());
    }

    #[test]
    fn result_json_round_trip() {
        let r = result();
        let back: EvalResult = serde_json::from_str(&r.canonical_json()).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn golden_result() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/result_min.json");
        let json = result().canonical_json() + "\n";
        if std::env::var_os("KILN_UPDATE_GOLDEN").is_some() {
            std::fs::write(path, &json).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(path).unwrap(),
            json,
            "set KILN_UPDATE_GOLDEN=1 to update"
        );
    }

    fn codes(d: &[Diagnostic]) -> Vec<&str> {
        d.iter().map(|d| d.code.as_str()).collect()
    }

    #[test]
    fn average_power_matches_energy_over_time() {
        let mut s = sim(Corner::Central, 1.0);
        assert!(check_sim(&s).is_empty());
        s.power.avg_w = 1.0;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-POWER"));
        s.power.avg_w = f64::NAN;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-NUMBER"));
        let mut z = sim(Corner::Central, 1.0);
        z.makespan_s = 0.0;
        z.power.avg_w = 1.0;
        assert!(codes(&check_sim(&z)).contains(&"E-TRACE-POWER"));
    }

    #[test]
    fn negative_physical_quantities_are_rejected() {
        let mut r = result();
        r.phases[0].energy_j = Interval::point(-1.0);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-NEGATIVE"));
        let mut r = result();
        r.phases[0].clock_hz = Interval::new(-1.0, 1.3e9, 1.41e9).unwrap();
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-NEGATIVE"));
        let mut s = sim(Corner::Central, 1.0);
        s.energy.compute_j = -0.5;
        s.energy = s.energy.with_total();
        s.power.avg_w = s.energy.total_j;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-NEGATIVE"));
    }

    #[test]
    fn central_floor_checked_against_central_time() {
        let mut r = result();
        r.sim.clear();
        r.phases[0].time_s = Interval::new(1.0, 2.0, 10.0).unwrap();
        r.phases[0].floors[0].seconds = 5.0;
        assert!(codes(&check_result(&r)).contains(&"E-FLOOR-I15"));
    }

    fn span_trace(op_end: i64, spans: &[(i64, i64)]) -> crate::trace::Trace {
        let mut t = crate::trace::Trace::empty(crate::container::Manifest::new(
            crate::provenance::TraceLevel::Summary,
            provenance(),
        ));
        t.resources.push(crate::trace::ResourceRow {
            path: "chip0".into(),
            kind: 0,
            class: 0,
            parent: None,
            chip: 0,
            array_id: None,
            array_pos: None,
            mem_level: None,
            capacity_b: None,
            peak_bw_bps: None,
            peak_flops: vec![],
            lanes: 1,
        });
        t.ops.push(crate::trace::OpRow {
            path: "op".into(),
            family: "op".into(),
            kind: 0,
            phase: 0,
            layer: None,
            flops: 0.0,
            precision: 0,
            bytes_by_level: vec![],
            link_bytes: 0.0,
            t_start: 0,
            t_end: op_end,
            chips: vec![0],
            mapping: 0,
            macs_useful: 0,
            macs_issued: 0,
            target: 0,
            host_vs_nmp_s: None,
            group: 0,
            energy_j: 0.0,
            time_low_s: None,
            time_high_s: None,
        });
        t.spans = spans
            .iter()
            .map(|&(t_start, dur)| crate::trace::SpanRow {
                resource: 0,
                lane: 0,
                kind: 0,
                flags: 0,
                op: 0,
                task: 0,
                slice: 0,
                t_start,
                dur,
                bytes: 0.0,
                energy_j: 0.0,
            })
            .collect();
        t
    }

    #[test]
    fn span_endpoint_overflow_is_invalid() {
        assert!(check_trace(&span_trace(10, &[(0, 4), (4, 6)])).is_empty());
        let t = span_trace(i64::MAX, &[(i64::MAX - 5, 10)]);
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-SPAN"));
        let t = span_trace(i64::MAX, &[(i64::MAX - 20, i64::MAX), (i64::MAX - 10, 1)]);
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-SPAN"));
    }

    #[test]
    fn overflowing_energy_sum_fails_i6() {
        let mut s = sim(Corner::Central, 1.0);
        s.energy.compute_j = 1e308;
        s.energy.memory_j.insert("hbm".into(), 1e308);
        s.energy.total_j = 1.0;
        s.power.avg_w = 1.0;
        assert!(codes(&check_sim(&s)).contains(&"E-FLOOR-I6"));
    }

    #[test]
    fn resource_quantities_are_validated() {
        let check = |f: fn(&mut ResourceResult), code: &str| {
            let mut s = sim(Corner::Central, 1.0);
            f(&mut s.resources[0]);
            assert!(codes(&check_sim(&s)).contains(&code), "{code}");
        };
        check(|r| r.busy_s = -1.0, "E-TRACE-NEGATIVE");
        check(|r| r.busy_s = f64::NAN, "E-TRACE-NUMBER");
        check(|r| r.stall_s = -1.0, "E-TRACE-NEGATIVE");
        check(|r| r.bytes = -100.0, "E-TRACE-NEGATIVE");
        check(|r| r.energy_j = -1.0, "E-TRACE-NEGATIVE");
        check(|r| r.utilization = 0.5, "E-FLOOR-I3");
        let mut z = sim(Corner::Central, 1.0);
        z.makespan_s = 0.0;
        z.t_a0_s = 0.0;
        z.t_a2_s = 0.0;
        z.power.avg_w = 0.0;
        z.ops.clear();
        z.bottleneck.time_by_binding.clear();
        z.resources[0].busy_s = 0.0;
        assert!(codes(&check_sim(&z)).contains(&"E-FLOOR-I3"));
        z.resources[0].utilization = 0.0;
        assert!(!codes(&check_sim(&z)).contains(&"E-FLOOR-I3"));
    }

    #[test]
    fn negative_or_overflowing_envelopes_are_invalid() {
        let mut t = span_trace(10, &[]);
        t.ops[0].t_start = i64::MIN;
        t.ops[0].t_end = i64::MAX;
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-SPAN"));
        let mut t = span_trace(10, &[]);
        t.groups.push(crate::trace::GroupRow {
            group: 0,
            phase: 0,
            ops: vec![0],
            kind: 0,
            binding: 0,
            t_start: -5,
            t_end: 10,
            bubble_s: 0.0,
            exposed_overhead_s: 0.0,
        });
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-SPAN"));
        let mut t = span_trace(10, &[]);
        t.collectives.push(crate::trace::CollectiveRow {
            collective: 0,
            name: "ar".into(),
            phase: 0,
            op: 0,
            algorithm: "ring".into(),
            group_chips: vec![0],
            steps: 1,
            bytes: 0.0,
            t_start: 4,
            t_end: 2,
            link_bytes_by_tier: vec![],
        });
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-SPAN"));
    }

    #[test]
    fn spans_respect_peak_bandwidth() {
        let bw_trace = |spans: &[(i64, i64)], bytes: f64| {
            let mut t = span_trace(2_000_000_000_000, spans);
            t.manifest.tick_s = 1e-12;
            t.resources[0].peak_bw_bps = Some(1e9);
            t.resources[0].lanes = 2;
            for (i, s) in t.spans.iter_mut().enumerate() {
                s.bytes = bytes;
                s.lane = i as u16;
            }
            t
        };
        let sec = 1_000_000_000_000;
        assert!(check_trace(&bw_trace(&[(0, sec)], 1e9)).is_empty());
        assert!(codes(&check_trace(&bw_trace(&[(0, sec)], 1e12))).contains(&"E-TRACE-BANDWIDTH"));
        let two = bw_trace(&[(0, sec), (0, sec)], 0.75e9);
        assert!(codes(&check_trace(&two)).contains(&"E-TRACE-BANDWIDTH"));
        let staggered = bw_trace(&[(0, sec), (sec, sec)], 0.75e9);
        assert!(check_trace(&staggered).is_empty());
        let mut zero = bw_trace(&[(0, sec)], 1.0);
        zero.spans[0].dur = 0;
        assert!(codes(&check_trace(&zero)).contains(&"E-TRACE-BANDWIDTH"));
        let mut estimated = bw_trace(&[(0, sec)], 1e12);
        estimated.spans[0].flags = crate::trace::span_flags::ESTIMATED;
        assert!(check_trace(&estimated).is_empty());
    }

    #[test]
    fn phase_summaries_match_their_central_simulation() {
        let mut r = result();
        r.phases[0].time_s = Interval::point(100.0);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-PHASE"));
        let mut r = result();
        r.phases[0].energy_j = Interval::point(50.0);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-PHASE"));
        let mut r = result();
        r.phases[0].avg_power_w = Interval::point(1.0);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-PHASE"));
        let mut r = result();
        r.sim[0].corner = Corner::Low;
        r.phases[0].time_s = Interval::new(0.01, 100.0, 200.0).unwrap();
        assert!(!codes(&check_result(&r)).contains(&"E-TRACE-PHASE"));
    }

    fn binding() -> impl Strategy<Value = Binding> {
        let res = (0u32..64).prop_map(|i| id(&format!("chip0.unit{i}")));
        prop_oneof![
            res.clone()
                .prop_map(|resource| Binding::Compute { resource }),
            res.clone().prop_map(|resource| Binding::Dram { resource }),
            res.prop_map(|resource| Binding::Link { resource }),
            Just(Binding::Dependency),
            Just(Binding::PipelineBubble),
            Just(Binding::Overhead {
                overhead: OverheadKind::Gap
            }),
        ]
    }

    proptest! {
        #[test]
        fn sim_round_trip_and_hash(
            makespan in 1e-6f64..10.0,
            fracs in proptest::collection::vec((0.0f64..1.0, binding(), 0u64..1 << 40), 0..24),
        ) {
            let mut s = sim(Corner::Low, makespan);
            s.ops = fracs.iter().enumerate().map(|(i, (f, b, macs))| OpResult {
                op: id(&format!("layers.op{i}")),
                start_s: 0.0,
                end_s: makespan * f,
                binding: b.clone(),
                macs_useful: *macs,
                macs_issued: *macs,
                floors: vec![Floor { kind: FloorKind::Compute, path: None, seconds: makespan * f }],
                ..s.ops[0].clone()
            }).collect();
            let json = serde_json::to_string(&s).unwrap();
            let back: SimResult = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(&back, &s);
            let v = serde_json::to_value(&s).unwrap();
            prop_assert_eq!(content_hash("x-", &v), content_hash("x-", &serde_json::to_value(&back).unwrap()));
            prop_assert!(check_sim(&s).is_empty());
        }
    }

    #[test]
    fn op_floors_bound_op_time() {
        let mut r = result();
        r.sim[0].ops[0].floors[0].seconds = 1.0;
        assert!(codes(&check_result(&r)).contains(&"E-FLOOR-I15"));
        let mut s = sim(Corner::Central, 1.0);
        s.ops[0].floors[0].seconds = f64::NAN;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-NUMBER"));
        s.ops[0].floors[0].seconds = -1.0;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-NEGATIVE"));
        let mut s = sim(Corner::Central, 1.0);
        s.ops[0].end_s = 0.25;
        s.ops[0].floors[0].seconds = 0.5;
        assert!(codes(&check_sim(&s)).contains(&"E-FLOOR-I15"));
        s.groups.push(GroupResult {
            group: 0,
            kind: GroupKind::Fused,
            ops: vec![s.ops[0].op.clone()],
            start_s: 0.0,
            end_s: 1.0,
            binding: Binding::Dependency,
            bubble_s: 0.0,
            exposed_overhead_s: 0.0,
        });
        assert!(check_sim(&s).is_empty(), "{:?}", check_sim(&s));
        s.groups[0].kind = GroupKind::Single;
        assert!(check_sim(&s).is_empty(), "{:?}", check_sim(&s));
        s.groups[0].end_s = 2.0;
        assert!(codes(&check_sim(&s)).contains(&"E-TRACE-SPAN"));
        s.groups[0].end_s = 0.25;
        assert!(codes(&check_sim(&s)).contains(&"E-FLOOR-I15"));
    }

    #[test]
    fn zero_makespan_has_zero_energy() {
        let mut z = sim(Corner::Central, 1.0);
        z.makespan_s = 0.0;
        z.t_a0_s = 0.0;
        z.t_a2_s = 0.0;
        z.power.avg_w = 0.0;
        z.ops.clear();
        z.resources.clear();
        z.bottleneck.time_by_binding.clear();
        assert!(codes(&check_sim(&z)).contains(&"E-TRACE-POWER"));
        z.energy = EnergyBreakdown::default().with_total();
        assert!(check_sim(&z).is_empty(), "{:?}", check_sim(&z));
    }

    #[test]
    fn nan_aggregate_energy_is_rejected() {
        let mut t = span_trace(10, &[]);
        t.ops[0].energy_j = 1.0;
        let row = |energy_j| crate::trace::AggOpResourceRow {
            op: 0,
            resource: 0,
            time_s: 0.0,
            energy_component: 0,
            energy_j,
        };
        t.aggregates_op_resource = vec![row(1.0)];
        assert!(check_trace(&t).is_empty(), "{:?}", check_trace(&t));
        t.aggregates_op_resource = vec![row(f64::NAN)];
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-NUMBER"));
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-ENERGY"));
        t.aggregates_op_resource = vec![row(2.0), row(-1.0)];
        assert!(codes(&check_trace(&t)).contains(&"E-TRACE-NUMBER"));
    }

    #[test]
    fn bound_fractions_lie_in_unit_interval() {
        let mut r = result();
        r.phases[0].bound_breakdown =
            BTreeMap::from([("compute".into(), -1.0), ("dram".into(), 2.0)]);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-ATTRIBUTION"));
        r.phases[0].bound_breakdown = BTreeMap::from([("compute".into(), f64::NAN)]);
        assert!(codes(&check_result(&r)).contains(&"E-TRACE-ATTRIBUTION"));
    }
}
