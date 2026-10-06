//! 06 §6.4 fitness: matched-envelope ratio vs the simulated baseline, per-corner intervals, invalid scoring,
//! and the claim interval rule (06 §6.6). Both sides of every ratio are simulations under the same calibration
//! set, tier, seeds and kiln version (08 §F scoring basis); measured data validates the simulator and never
//! enters a score. `score` runs both sides under one software stack (`kiln_ideal` by default); the realistic
//! score runs each under its own execution model's default stack (08 §F software stack in scoring).

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_trace::Interval;
use kiln_trace::result::{
    Aggregation, EvalResult, Feature, PhysicalSummary, RealisticScore, ResultError,
    ScoreComponents, Stage, Status,
};
use kiln_trace::{Corner, IntervalMethod};

use crate::options::{
    Envelope, FitnessKind, IntervalBasis, InvalidScore, Options, STACK_IDEAL, STACK_OWN,
};

pub const RETICLE_MM2: f64 = 858.0;
pub const BASELINE_CODE: &str = "E-FIT-0001";
pub const NO_PHASES_CODE: &str = "E-FIT-0002";
pub const PHASE_CODE: &str = "E-FIT-0003";
pub const UNCHECKED_CODE: &str = "W-ENV-0001";
pub const ASYM_CODE: &str = "E-CAL-ASYM";
pub const CLAIM_CODE: &str = "E-AUDIT-CLAIM";
/// Design-derived features (`HwSummary`): mem stack interface bandwidth (bytes/s) and capacity (bytes).
pub const OFFCHIP_BW: &str = "offchip_bw";
pub const OFFCHIP_BYTES: &str = "offchip_bytes";
/// Relative slack on the off-chip limits (exact design arithmetic, so only float noise).
pub const OFFCHIP_TOL: f64 = 1e-9;

fn env_error(
    code: &str,
    msg: String,
    path: &str,
    hint: String,
    value: f64,
    limit: f64,
    unit: &str,
) -> ResultError {
    ResultError {
        value: Some(value),
        limit: Some(limit),
        unit: Some(unit.into()),
        section: Some("06 §6.4".into()),
        ..Diagnostic::error(code, msg).at(path).hint(hint).into()
    }
}

fn total_die(p: &PhysicalSummary) -> Interval {
    p.die_mm2
        .values()
        .fold(Interval::point(0.0), |a, d| Interval {
            low: a.low + d.low,
            central: a.central + d.central,
            high: a.high + d.high,
        })
}

fn scalar(r: &EvalResult, key: &str) -> Option<f64> {
    match r.features.get(key) {
        Some(Feature::Scalar(x)) => Some(*x),
        _ => None,
    }
}

/// Limits from `fitness.envelope`, else from the baseline (06 §6.4 matched envelope): physical limits from its
/// `physical` result (not under `explicit_envelope`), off-chip limits from its design summary under every
/// enveloped kind, since off-chip memory is bought, not designed. An explicit `null` off-chip limit stays unset.
fn limits(opts: &Options, base: &EvalResult) -> Envelope {
    let e = opts.fitness.envelope.clone().unwrap_or_default();
    let from_base = opts.fitness.kind != FitnessKind::ExplicitEnvelope;
    let base_phys = base.physical.as_ref();
    let b = |f: &dyn Fn(&PhysicalSummary) -> f64| base_phys.filter(|_| from_base).map(f);
    let off = |l: Option<Option<f64>>, key| Some(l.unwrap_or_else(|| scalar(base, key)));
    Envelope {
        die_mm2: e.die_mm2.or_else(|| b(&|p| total_die(p).central)),
        power_w: e.power_w.or_else(|| b(&|p| p.tdp_w)),
        node: e
            .node
            .or_else(|| base_phys.filter(|_| from_base).map(|p| p.node.clone())),
        package_mm2: e.package_mm2.or_else(|| b(&|p| p.package_mm2.central)),
        hbm_shoreline_mm: e
            .hbm_shoreline_mm
            .or_else(|| b(&|p| p.hbm_shoreline_used_mm)),
        reticle_mm2: Some(e.reticle_mm2.unwrap_or(RETICLE_MM2)),
        offchip_bw: off(e.offchip_bw, OFFCHIP_BW),
        offchip_bytes: off(e.offchip_bytes, OFFCHIP_BYTES),
    }
}

/// Off-chip memory violations: the candidate's design-derived mem stack interface bandwidth and capacity against
/// the limits. Exact (no corners), so they need no physical result. Near-memory (PIM) internal bandwidth is
/// inside the stack and not counted.
pub fn offchip_violations(r: &EvalResult, lim: &Envelope, baseline: &str) -> Vec<ResultError> {
    let src = format!("envelope of baseline {baseline}");
    let over = |key, l: Option<Option<f64>>| {
        let l = l.flatten()?;
        let v = scalar(r, key)?;
        (v > l * (1.0 + OFFCHIP_TOL)).then_some((v, l))
    };
    let mut out = Vec::new();
    if let Some((v, l)) = over(OFFCHIP_BW, lim.offchip_bw) {
        out.push(env_error(
            "E-ENV-0007",
            format!(
                "off-chip memory bandwidth {:.1} GB/s exceeds {:.1} GB/s ({src})",
                v / 1e9,
                l / 1e9
            ),
            "features.offchip_bw",
            format!(
                "off-chip memory is bought, not designed: keep the baseline's mem_stacks (count x io_width_bits x \
                 pin_rate_bits_per_s / 8 <= {:.1} GB/s), i.e. cut >= {:.1} GB/s by lowering pin_rate_bits_per_s or \
                 removing stacks, and win with on-chip structure (reuse, SRAM, dataflow) instead",
                l / 1e9,
                (v - l) / 1e9
            ),
            v,
            l,
            "B/s",
        ));
    }
    if let Some((v, l)) = over(OFFCHIP_BYTES, lim.offchip_bytes) {
        let gib = |x: f64| x / f64::from(1u32 << 30);
        out.push(env_error(
            "E-ENV-0008",
            format!(
                "off-chip memory capacity {:.2} GiB exceeds {:.2} GiB ({src})",
                gib(v),
                gib(l)
            ),
            "features.offchip_bytes",
            format!(
                "off-chip memory is bought, not designed: keep the baseline's stack count and per-stack capacity \
                 (<= {:.2} GiB in total), i.e. remove stacks or cut capacity by >= {:.2} GiB",
                gib(l),
                gib(v - l)
            ),
            v,
            l,
            "B",
        ));
    }
    out
}

/// Envelope violations at the central corner (claims check the pessimistic corner, 06 §6.6).
pub fn envelope_violations(
    p: &PhysicalSummary,
    lim: &Envelope,
    baseline: &str,
) -> Vec<ResultError> {
    let mut v = Vec::new();
    let src = format!("envelope of baseline {baseline}");
    let die = total_die(p).central;
    if let Some(l) = lim.die_mm2.filter(|l| die > *l) {
        v.push(env_error(
            "E-ENV-0001",
            format!("total die area {die:.1} mm^2 exceeds {l:.1} mm^2 ({src})"),
            "physical.die_mm2",
            format!(
                "remove or shrink blocks to cut >= {:.1} mm^2 of die area",
                die - l
            ),
            die,
            l,
            "mm^2",
        ));
    }
    if let Some(l) = lim.reticle_mm2 {
        for (die_id, a) in p.die_mm2.iter().filter(|(_, a)| a.central > l) {
            v.push(env_error(
                "E-ENV-0002",
                format!(
                    "die {die_id} is {:.1} mm^2, above the {l:.0} mm^2 reticle limit",
                    a.central
                ),
                &format!("physical.die_mm2.{die_id}"),
                format!(
                    "split {die_id} into chiplets or cut >= {:.1} mm^2",
                    a.central - l
                ),
                a.central,
                l,
                "mm^2",
            ));
        }
    }
    if let Some(l) = lim.power_w.filter(|l| p.tdp_w > *l) {
        v.push(env_error(
            "E-ENV-0003",
            format!("TDP {:.1} W exceeds {l:.1} W ({src})", p.tdp_w),
            "physical.tdp_w",
            format!("lower the power cap / TDP by >= {:.1} W", p.tdp_w - l),
            p.tdp_w,
            l,
            "W",
        ));
    }
    if let Some(l) = lim.node.as_ref().filter(|l| **l != p.node) {
        v.push(ResultError {
            section: Some("06 §6.4".into()),
            ..Diagnostic::error(
                "E-ENV-0004",
                format!("process node {} differs from {l} ({src})", p.node),
            )
            .at("physical.node")
            .hint(format!("use technology node {l}"))
            .into()
        });
    }
    if let Some(l) = lim.package_mm2.filter(|l| p.package_mm2.central > *l) {
        v.push(env_error(
            "E-ENV-0005",
            format!(
                "package area {:.1} mm^2 exceeds {l:.1} mm^2 ({src})",
                p.package_mm2.central
            ),
            "physical.package_mm2",
            format!(
                "cut >= {:.1} mm^2 of package area",
                p.package_mm2.central - l
            ),
            p.package_mm2.central,
            l,
            "mm^2",
        ));
    }
    if let Some(l) = lim
        .hbm_shoreline_mm
        .filter(|l| p.hbm_shoreline_used_mm > *l)
    {
        v.push(env_error(
            "E-ENV-0006",
            format!(
                "HBM shoreline {:.1} mm exceeds {l:.1} mm ({src})",
                p.hbm_shoreline_used_mm
            ),
            "physical.hbm_shoreline_used_mm",
            "remove HBM stacks or use fewer, larger stacks".into(),
            p.hbm_shoreline_used_mm,
            l,
            "mm",
        ));
    }
    v
}

/// `graded`: `-(1 + sum of normalized violation magnitudes)`; magnitude 1 where no value/limit is known.
pub fn invalid_score(r: &EvalResult, mode: InvalidScore) -> f64 {
    match mode {
        InvalidScore::Zero => 0.0,
        InvalidScore::Graded => {
            let errs = if r.violations.is_empty() {
                &r.errors
            } else {
                &r.violations
            };
            let m: f64 = errs
                .iter()
                .map(|e| match (e.value, e.limit) {
                    (Some(v), Some(l)) if l != 0.0 => ((v - l) / l.abs()).max(0.0),
                    _ => 1.0,
                })
                .sum();
            -(1.0 + m)
        }
    }
}

fn aggregate(ratios: &[(f64, f64)], agg: Aggregation) -> f64 {
    let wsum: f64 = ratios.iter().map(|(_, w)| w).sum();
    if ratios.is_empty() || wsum == 0.0 {
        return 0.0;
    }
    match agg {
        Aggregation::Geomean => (ratios
            .iter()
            .map(|(r, w)| w * r.max(f64::MIN_POSITIVE).ln())
            .sum::<f64>()
            / wsum)
            .exp(),
        Aggregation::Min => ratios
            .iter()
            .filter(|(_, w)| *w > 0.0)
            .map(|(r, _)| *r)
            .fold(f64::INFINITY, f64::min),
        Aggregation::WeightedHarmonic => {
            wsum / ratios
                .iter()
                .map(|(r, w)| w / r.max(f64::MIN_POSITIVE))
                .sum::<f64>()
        }
    }
}

fn phase_metric(
    r: &EvalResult,
    kind: FitnessKind,
    phase: &kiln_trace::result::PhaseResult,
) -> Interval {
    match kind {
        FitnessKind::PerfPerWatt => phase.tokens_per_j,
        FitnessKind::PerfPerArea => {
            let a = r
                .physical
                .as_ref()
                .map_or(Interval::point(f64::NAN), total_die);
            Interval::from_corners(
                phase.tokens_per_s.central / a.central,
                phase.tokens_per_s.low / a.high,
                phase.tokens_per_s.high / a.low,
            )
        }
        _ => phase.tokens_per_s,
    }
}

fn needs_envelope(kind: FitnessKind) -> bool {
    kind != FitnessKind::BaselineRelative
}

fn fail(r: &mut EvalResult, status: Status, err: ResultError, opts: &Options) {
    r.status = status;
    r.errors.push(err);
    r.score = invalid_score(r, opts.invalid_score());
    r.score_interval = None;
}

/// Sets `score`, `score_interval`, `score_components` (and `status = envelope` on violations).
pub fn apply(r: &mut EvalResult, baseline: Option<(&str, &EvalResult)>, opts: &Options) {
    r.score_interval = None;
    r.score_components = None;
    if r.status != Status::Ok {
        r.score = invalid_score(r, opts.invalid_score());
        return;
    }
    let Some((bid, base)) = baseline else {
        r.score = 0.0;
        return;
    };
    if base.status != Status::Ok {
        let err = Diagnostic::error(
            BASELINE_CODE,
            format!("baseline {bid} did not evaluate: status {:?}", base.status),
        )
        .at("options.fitness.baseline")
        .hint("the baseline must evaluate ok on the same workload set; check `kiln eval` on it");
        return fail(r, Status::InternalError, err.into(), opts);
    }
    if let Some(why) = basis_mismatch(r, base) {
        let err = Diagnostic::error(
            ASYM_CODE,
            format!("baseline {bid} is not comparable: {why}"),
        )
        .at("options.fitness.baseline")
        .hint("scores compare the simulated candidate with the baseline simulated under the same calibration set, tier, seeds and kiln version (06 §3.1 rule 3, 08 §F)");
        return fail(r, Status::InternalError, err.into(), opts);
    }
    if let Some(why) = stack_mismatch(r, base, opts) {
        let err = Diagnostic::error(
            ASYM_CODE,
            format!("baseline {bid} is not comparable: {why}"),
        )
        .at("options.stack")
        .hint("score both sides under one stack (`kiln_ideal` by default); `own` is the declared realistic comparison (08 §F)");
        return fail(r, Status::InternalError, err.into(), opts);
    }
    let kind = opts.fitness.kind;
    if needs_envelope(kind) {
        let lim = limits(opts, base);
        let mut v = offchip_violations(r, &lim, bid);
        match &r.physical {
            Some(p) => v.extend(envelope_violations(p, &lim, bid)),
            None => r.warnings.push(
                Diagnostic::warning(
                    UNCHECKED_CODE,
                    "no physical result; the physical envelope (area, power, node, shoreline) was not checked",
                )
                .at("physical")
                .into(),
            ),
        }
        if !v.is_empty() {
            r.violations.extend(v);
            r.status = Status::Envelope;
            r.score = invalid_score(r, opts.invalid_score());
            return;
        }
    }
    let mut phases = BTreeMap::new();
    let mut weights = BTreeMap::new();
    for p in &r.phases {
        let Some(b) = base.phase(p.phase.as_str()) else {
            continue;
        };
        let denom = phase_metric(base, kind, b).central;
        let c = phase_metric(r, kind, p);
        if !(denom.is_finite() && denom > 0.0) {
            continue;
        }
        if !c.is_valid() {
            let err = Diagnostic::error(
                PHASE_CODE,
                format!("phase {} has an invalid metric interval [{}, {}, {}]", p.phase, c.low, c.central, c.high),
            )
            .at(format!("phases[{}]", p.phase))
            .hint("a phase whose metric (or one of its corners) failed to evaluate cannot drop out of the score");
            return fail(r, Status::InternalError, err.into(), opts);
        }
        let w = opts
            .fitness
            .phase_weights
            .get(p.phase.as_str())
            .copied()
            .unwrap_or(1.0);
        phases.insert(
            p.phase.to_string(),
            Interval::from_corners(c.central / denom, c.low / denom, c.high / denom),
        );
        weights.insert(p.phase.to_string(), w);
    }
    if phases.is_empty() {
        let err = Diagnostic::error(
            NO_PHASES_CODE,
            "no phase is shared with the baseline result",
        )
        .at("phases")
        .hint("candidate and baseline must be evaluated on the same workload set");
        return fail(r, Status::InternalError, err.into(), opts);
    }
    let agg = opts.fitness.aggregation;
    let at = |f: fn(&Interval) -> f64| {
        let v: Vec<(f64, f64)> = phases.iter().map(|(k, i)| (f(i), weights[k])).collect();
        aggregate(&v, agg)
    };
    let si = Interval::from_corners(at(|i| i.central), at(|i| i.low), at(|i| i.high));
    r.score = match opts.fitness.interval_basis {
        IntervalBasis::Central => si.central,
        IntervalBasis::Low => si.low,
    };
    if kind == FitnessKind::Pareto {
        let ratio = |k| {
            let v: Vec<(f64, f64)> = r
                .phases
                .iter()
                .filter_map(|p| {
                    let b = base.phase(p.phase.as_str())?;
                    let d = phase_metric(base, k, b).central;
                    (d > 0.0).then(|| (phase_metric(r, k, p).central / d, 1.0))
                })
                .collect();
            aggregate(&v, agg)
        };
        let (die, power) = r
            .physical
            .as_ref()
            .map_or((f64::NAN, f64::NAN), |p| (total_die(p).central, p.tdp_w));
        r.features.insert(
            "pareto".into(),
            Feature::Vector(vec![
                si.central,
                ratio(FitnessKind::PerfPerWatt),
                -die,
                -power,
            ]),
        );
    }
    r.score_interval = Some(si);
    r.score_components = Some(ScoreComponents {
        phases,
        weights,
        aggregation: agg,
        baseline_id: bid.into(),
        baseline_hash: base.provenance.design_hash.clone(),
        candidate_stack: stack_label(r),
        baseline_stack: stack_label(base),
    });
    let ratio = opts.audit.suspicion_ratio;
    if si.central > ratio {
        r.audit.status = kiln_trace::result::AuditStatus::Pending;
        r.audit.reasons.push(format!(
            "suspicion: score {:.3} > {ratio} x baseline {bid}; tier B audit is not available in M1",
            si.central
        ));
    }
}

/// Why `base` cannot be the denominator of `r`'s score, if it cannot: it must be a simulated result (tier set,
/// S3 or later) under the same calibration set, tier, seeds and kiln build. A measured value has no such
/// provenance, so no path divides by one.
pub fn basis_mismatch(r: &EvalResult, base: &EvalResult) -> Option<String> {
    let (a, b) = (&r.provenance, &base.provenance);
    if base.tier.is_none() || base.stage_reached < Stage::S3 {
        return Some("it is not a simulated whole-step result".into());
    }
    let diffs: Vec<&str> = [
        (
            "calibration set",
            a.calibration_hash != b.calibration_hash || a.calibration_id != b.calibration_id,
        ),
        ("tier", a.tier != b.tier || r.tier != base.tier),
        ("seeds", a.seeds != b.seeds),
        (
            "kiln build",
            a.kiln_version != b.kiln_version || a.git_hash != b.git_hash,
        ),
    ]
    .into_iter()
    .filter_map(|(k, d)| d.then_some(k))
    .collect();
    (!diffs.is_empty()).then(|| format!("{} differ from the candidate's", diffs.join(", ")))
}

fn stack_label(r: &EvalResult) -> Option<String> {
    r.provenance.flags.get("stack").cloned()
}

/// Why the two sides of a same-stack score ran under different software stacks, if they did. `own` declares
/// the realistic comparison, whose sides differ by design.
pub fn stack_mismatch(r: &EvalResult, base: &EvalResult, opts: &Options) -> Option<String> {
    if opts.stack == STACK_OWN {
        return None;
    }
    let (a, b) = (stack_label(r), stack_label(base));
    let show = |x: &Option<String>| x.clone().unwrap_or_else(|| "none".into());
    (a != b).then(|| {
        format!(
            "software stack {} differs from the candidate's {}",
            show(&b),
            show(&a)
        )
    })
}

/// The realistic score of `own` (the candidate evaluated with `stack = own`) against `baseline` (the baseline
/// under `stack = own`): formed exactly as [`apply`] forms `score`, or the error that prevented it.
pub fn realistic(
    own: &EvalResult,
    baseline: (&str, &EvalResult),
    opts: &Options,
) -> Result<RealisticScore, Box<ResultError>> {
    let o = opts.with_stack(STACK_OWN);
    let mut r = own.clone();
    apply(&mut r, Some(baseline), &o);
    match (r.status, r.score_interval) {
        (Status::Ok, Some(interval)) => Ok(RealisticScore {
            score: r.score,
            interval,
            candidate_stack: stack_label(own).unwrap_or_default(),
            baseline_stack: stack_label(baseline.1).unwrap_or_default(),
        }),
        (status, _) => Err(Box::new(
            r.violations
                .into_iter()
                .chain(r.errors)
                .next()
                .unwrap_or_else(|| {
                    Diagnostic::error(
                        BASELINE_CODE,
                        format!("realistic score not formed: status {status:?}"),
                    )
                    .at("score_realistic")
                    .into()
                }),
        )),
    }
}

/// `p` at its pessimistic corner: high area, high power (06 §6.6 claim rule).
fn pessimistic(p: &PhysicalSummary) -> PhysicalSummary {
    let hi = |i: &Interval| Interval::point(i.high.max(i.central));
    PhysicalSummary {
        die_mm2: p.die_mm2.iter().map(|(k, v)| (k.clone(), hi(v))).collect(),
        package_mm2: hi(&p.package_mm2),
        peak_power_w: hi(&p.peak_power_w),
        tdp_w: p.tdp_w.max(p.peak_power_w.high),
        ..p.clone()
    }
}

/// The interval part of the claim rule (06 §6.6): `score_interval.low` (the candidate's pessimistic corner
/// over the simulated baseline's central value, as [`apply`] forms it) >= 1.0 under `kiln_ideal` on both
/// sides AND `score_realistic.interval.low` >= 1.0 (each side under its own default stack, 08 §F), at a
/// matched envelope, with `corners` intervals, the envelope satisfied at the candidate's pessimistic corner and
/// no corner flip at `low`. Empty when the rule holds; tier B, seeds, held-out and trust gates are checked
/// elsewhere.
pub fn claim_interval_rule(
    r: &EvalResult,
    baseline: (&str, &EvalResult),
    opts: &Options,
) -> Vec<ResultError> {
    let (bid, base) = baseline;
    let mut out = vec![];
    let mut no = |msg: String, path: &str, hint: &str| {
        out.push(ResultError {
            section: Some("06 §6.6".into()),
            ..Diagnostic::error(CLAIM_CODE, msg)
                .at(path)
                .hint(hint)
                .into()
        });
    };
    if r.status != Status::Ok {
        no(
            format!("status is {:?}", r.status),
            "status",
            "only ok results can be claimed",
        );
        return out;
    }
    if let Some(why) = basis_mismatch(r, base) {
        no(
            format!("baseline {bid}: {why}"),
            "options.fitness.baseline",
            "re-evaluate both under identical options",
        );
    }
    if opts.stack != STACK_IDEAL {
        no(
            format!("score uses stack {:?}", opts.stack),
            "options.stack",
            "claims compare both sides under kiln_ideal and report the realistic score next to it (08 §F)",
        );
    } else if let Some(why) = stack_mismatch(r, base, opts) {
        no(
            format!("baseline {bid}: {why}"),
            "options.stack",
            "re-evaluate both under the same stack",
        );
    }
    match &r.score_realistic {
        Some(rs) if rs.interval.low >= 1.0 => {}
        Some(rs) => no(
            format!(
                "realistic low score {:.4} < 1.0 against {bid} ({} vs {})",
                rs.interval.low, rs.candidate_stack, rs.baseline_stack
            ),
            "score_realistic.interval.low",
            "a claim must also win with each design under its own software stack",
        ),
        None => no(
            "no realistic score".into(),
            "score_realistic",
            "evaluate through the session, which reports the realistic score next to score",
        ),
    }
    if opts.fitness.kind == FitnessKind::BaselineRelative {
        no(
            "baseline_relative is diagnostic only".into(),
            "options.fitness.kind",
            "claims are at a matched envelope",
        );
    }
    if opts.interval != IntervalMethod::Corners {
        no(
            format!("interval is {:?}", opts.interval),
            "options.interval",
            "claims use corners (03 §9.1)",
        );
    }
    match &r.score_interval {
        Some(si) if si.low >= 1.0 => {}
        Some(si) => no(
            format!(
                "low score {:.4} < 1.0 against the simulated baseline {bid} (central)",
                si.low
            ),
            "score_interval.low",
            "the pessimistic corner must still beat the baseline",
        ),
        None => no(
            "no score interval".into(),
            "score_interval",
            "score against a baseline first",
        ),
    }
    if let Some(f) = r
        .interval
        .corner_flips
        .iter()
        .find(|f| f.corner == Corner::Low)
    {
        no(
            format!("corner flip at low: {}", f.what),
            "interval.corner_flips",
            "a discrete outcome flips inside the range",
        );
    }
    if opts.fitness.kind != FitnessKind::BaselineRelative
        && let Some(p) = &r.physical
    {
        for v in envelope_violations(&pessimistic(p), &limits(opts, base), bid) {
            no(
                format!("pessimistic corner: {}", v.diag.message),
                &v.diag.path.clone().unwrap_or_default(),
                "the envelope must hold at high area and power",
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::result_with;
    use serde_json::json;

    fn opts(v: serde_json::Value) -> Options {
        Options::from_value(&v).unwrap()
    }

    /// A realistic score whose low corner is `low` (central 10% above it).
    fn realistic_at(low: f64) -> RealisticScore {
        RealisticScore {
            score: 1.1 * low,
            interval: Interval::from_corners(1.1 * low, low, 1.2 * low),
            candidate_stack: "xla_tpu_fused@stk1-x".into(),
            baseline_stack: "pytorch_cuda_graph_sdpa@stk1-y".into(),
        }
    }

    fn with_stack(mut r: EvalResult, label: &str) -> EvalResult {
        r.provenance.flags.insert("stack".into(), label.into());
        r
    }

    #[test]
    fn matched_envelope_ratio_and_interval() {
        let base = result_with(&[("decode_b1", 80.0), ("decode_b8", 400.0)], 800.0, 400.0);
        let mut cand = result_with(&[("decode_b1", 160.0), ("decode_b8", 400.0)], 700.0, 350.0);
        apply(&mut cand, Some(("a100", &base)), &Options::default());
        assert_eq!(cand.status, Status::Ok);
        assert!((cand.score - 2f64.sqrt()).abs() < 1e-12);
        let si = cand.score_interval.unwrap();
        assert!(si.low < si.central && si.central < si.high);
        assert_eq!(cand.score_components.as_ref().unwrap().phases.len(), 2);
        assert_eq!(cand.audit.status, kiln_trace::result::AuditStatus::Pending);

        let mut same = base.clone();
        apply(&mut same, Some(("a100", &base)), &Options::default());
        assert_eq!(same.score, 1.0);
        assert_eq!(same.audit.status, kiln_trace::result::AuditStatus::NotRun);
    }

    #[test]
    fn envelope_violation_zero_and_graded() {
        let base = result_with(&[("decode_b1", 80.0)], 800.0, 400.0);
        let mut cand = result_with(&[("decode_b1", 800.0)], 1000.0, 500.0);
        apply(&mut cand, Some(("a100", &base)), &Options::default());
        assert_eq!(cand.status, Status::Envelope);
        assert_eq!(cand.score, 0.0);
        let codes: Vec<_> = cand
            .violations
            .iter()
            .map(|v| v.diag.code.as_str())
            .collect();
        assert_eq!(codes, ["E-ENV-0001", "E-ENV-0002", "E-ENV-0003"]);
        assert!(
            cand.violations[0]
                .diag
                .hint
                .as_ref()
                .unwrap()
                .contains("200.0 mm^2")
        );

        let mut cand = result_with(&[("decode_b1", 800.0)], 1000.0, 500.0);
        apply(
            &mut cand,
            Some(("a100", &base)),
            &opts(json!({"invalid_score": "graded"})),
        );
        assert!(cand.score < -1.0);

        let mut cand = result_with(&[("decode_b1", 800.0)], 1000.0, 500.0);
        apply(
            &mut cand,
            Some(("a100", &base)),
            &opts(json!({"fitness": {"kind": "baseline_relative"}})),
        );
        assert_eq!(cand.status, Status::Ok);
        assert!((cand.score - 10.0).abs() < 1e-9);
    }

    fn with_offchip(mut r: EvalResult, bw: f64, bytes: f64) -> EvalResult {
        r.features.insert(OFFCHIP_BW.into(), Feature::Scalar(bw));
        r.features
            .insert(OFFCHIP_BYTES.into(), Feature::Scalar(bytes));
        r
    }

    #[test]
    fn offchip_limits_default_override_and_null() {
        let gib = 2f64.powi(30);
        let base = with_offchip(
            result_with(&[("d", 100.0)], 800.0, 400.0),
            1555.2e9,
            40.0 * gib,
        );
        let cand = |bw: f64, bytes: f64| {
            with_offchip(result_with(&[("d", 150.0)], 700.0, 300.0), bw, bytes)
        };
        let run = |c: EvalResult, o: &Options| {
            let mut c = c;
            apply(&mut c, Some(("a100", &base)), o);
            c
        };
        let codes = |c: &EvalResult| -> Vec<String> {
            c.violations.iter().map(|v| v.diag.code.clone()).collect()
        };
        let d = Options::default();
        // Float noise inside the 1e-9 relative tolerance is not a violation.
        assert_eq!(
            run(cand(1555.2e9 * (1.0 + 1e-12), 40.0 * gib), &d).status,
            Status::Ok
        );
        let c = run(cand(1638e9, 32.0 * gib), &d);
        assert_eq!(
            (c.status, codes(&c), c.score),
            (Status::Envelope, vec!["E-ENV-0007".to_string()], 0.0)
        );
        let c = run(cand(1555.2e9, 48.0 * gib), &d);
        assert_eq!(codes(&c), ["E-ENV-0008"]);
        assert!(
            c.violations[0]
                .diag
                .hint
                .as_ref()
                .unwrap()
                .contains("8.00 GiB")
        );
        // Every enveloped kind pins off-chip memory; baseline_relative does not.
        for kind in ["perf_per_watt", "perf_per_area", "pareto"] {
            let o = opts(json!({"fitness": {"kind": kind}}));
            assert_eq!(
                run(cand(3e12, 80.0 * gib), &o).violations.len(),
                2,
                "{kind}"
            );
        }
        let rel = opts(json!({"fitness": {"kind": "baseline_relative"}}));
        assert_eq!(run(cand(3e12, 80.0 * gib), &rel).status, Status::Ok);

        let o = opts(json!({"fitness": {"envelope": {"offchip_bw": null}}}));
        let e = o.fitness.envelope.as_ref().unwrap();
        assert_eq!((e.offchip_bw, e.offchip_bytes), (Some(None), None));
        let back = Options::from_value(&serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(back, o);
        assert_ne!(o.hash(), Options::default().hash());
        assert_eq!(codes(&run(cand(3e12, 80.0 * gib), &o)), ["E-ENV-0008"]);
        let o = opts(json!({"fitness": {"kind": "explicit_envelope",
            "envelope": {"offchip_bw": 4e12, "offchip_bytes": 96.0 * gib}}}));
        assert_eq!(run(cand(3e12, 80.0 * gib), &o).status, Status::Ok);
        assert_eq!(codes(&run(cand(5e12, 80.0 * gib), &o)), ["E-ENV-0007"]);
    }

    #[test]
    fn low_basis_min_aggregation_and_weights() {
        let base = result_with(&[("a", 100.0), ("b", 100.0)], 800.0, 400.0);
        let mut cand = result_with(&[("a", 200.0), ("b", 50.0)], 800.0, 400.0);
        let o = opts(json!({"fitness": {"aggregation": "min", "interval_basis": "low"}}));
        apply(&mut cand, Some(("x", &base)), &o);
        assert!(cand.score < 0.5);
        let mut cand = result_with(&[("a", 200.0), ("b", 50.0)], 800.0, 400.0);
        let o = opts(json!({"fitness": {"phase_weights": {"b": 0.0}}}));
        apply(&mut cand, Some(("x", &base)), &o);
        assert!((cand.score - 2.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_candidate_phase_fails_instead_of_dropping_out() {
        let base = result_with(&[("a", 100.0), ("b", 100.0)], 800.0, 400.0);
        let mut cand = result_with(&[("a", 200.0), ("b", 50.0)], 800.0, 400.0);
        cand.phases[1].tokens_per_s = Interval::from_corners(50.0, f64::NAN, 60.0);
        apply(&mut cand, Some(("x", &base)), &opts(json!({})));
        assert_ne!(cand.status, Status::Ok);
        assert!(cand.score <= 0.0);
    }

    #[test]
    fn identical_design_scores_one_for_every_kind_and_basis() {
        let base = result_with(&[("decode_b1", 80.0), ("prefill_b1", 3000.0)], 800.0, 400.0);
        for kind in [
            "matched_envelope",
            "explicit_envelope",
            "perf_per_watt",
            "perf_per_area",
            "pareto",
            "baseline_relative",
        ] {
            for basis in ["central", "low"] {
                let mut f = json!({"kind": kind, "interval_basis": basis});
                if kind == "explicit_envelope" {
                    f["envelope"] = json!({"die_mm2": 800.0, "power_w": 400.0});
                }
                let o = opts(json!({"fitness": f}));
                let mut same = base.clone();
                apply(&mut same, Some(("a100", &base)), &o);
                assert_eq!(same.status, Status::Ok, "{kind}: {:?}", same.errors);
                let si = same.score_interval.unwrap();
                assert_eq!(si.central, 1.0, "{kind}");
                assert!(si.low < 1.0 && si.high > 1.0, "{kind}");
                // `low` is the candidate's pessimistic corner over the baseline's central value: a design never
                // claims to beat itself.
                let want = if basis == "central" { 1.0 } else { si.low };
                assert_eq!(same.score, want, "{kind} {basis}");
                assert!(!claim_interval_rule(&same, ("a100", &base), &o).is_empty());
            }
        }
        let mut point = base.clone();
        for p in &mut point.phases {
            p.tokens_per_s = Interval::point(p.tokens_per_s.central);
        }
        let o = opts(json!({"fitness": {"interval_basis": "low"}}));
        let mut same = point.clone();
        apply(&mut same, Some(("a100", &point)), &o);
        assert_eq!((same.score, same.score_interval.unwrap().low), (1.0, 1.0));
    }

    #[test]
    fn baseline_must_share_the_simulation_basis() {
        let base = result_with(&[("decode_b1", 80.0)], 800.0, 400.0);
        let mut measured = base.clone();
        measured.tier = None;
        let mut other_cal = base.clone();
        other_cal.provenance.calibration_hash = "cal1-other".into();
        let mut seeds = base.clone();
        seeds.provenance.seeds = vec![1];
        let mut tier_b = base.clone();
        tier_b.provenance.tier = kiln_trace::Tier::B;
        for (b, why) in [
            (&measured, "simulated"),
            (&other_cal, "calibration set"),
            (&seeds, "seeds"),
            (&tier_b, "tier"),
        ] {
            let mut cand = result_with(&[("decode_b1", 160.0)], 700.0, 350.0);
            apply(&mut cand, Some(("a100", b)), &Options::default());
            assert_eq!(cand.status, Status::InternalError);
            assert_eq!(cand.errors[0].diag.code, ASYM_CODE);
            assert!(
                cand.errors[0].diag.message.contains(why),
                "{}",
                cand.errors[0].diag.message
            );
            assert_eq!((cand.score, cand.score_interval), (0.0, None));
        }
    }

    #[test]
    fn claim_rule_low_corner_against_simulated_central() {
        let base = result_with(&[("d", 100.0)], 800.0, 400.0);
        let corners = opts(json!({"interval": "corners"}));
        let scored = |tps: f64, die: f64, o: &Options| {
            let mut c = result_with(&[("d", tps)], die, 300.0);
            c.physical.as_mut().unwrap().package_mm2 = Interval::point(1800.0);
            apply(&mut c, Some(("a100", &base)), o);
            c
        };
        let codes = |c: &EvalResult, o: &Options| claim_interval_rule(c, ("a100", &base), o).len();
        let mut win = scored(120.0, 700.0, &corners);
        win.score_realistic = Some(realistic_at(1.05));
        assert!((win.score_interval.unwrap().low - 1.08).abs() < 1e-12);
        assert_eq!(codes(&win, &corners), 0);
        let with_realistic = |mut c: EvalResult| {
            c.score_realistic = win.score_realistic.clone();
            c
        };
        assert_eq!(
            codes(&with_realistic(scored(110.0, 700.0, &corners)), &corners),
            1
        );
        let sens = Options::default();
        assert_eq!(
            codes(&with_realistic(scored(120.0, 700.0, &sens)), &sens),
            1
        );
        let rel = opts(json!({"interval": "corners", "fitness": {"kind": "baseline_relative"}}));
        assert_eq!(codes(&with_realistic(scored(120.0, 700.0, &rel)), &rel), 1);
        // 750 mm^2 central fits the 800 mm^2 envelope; its high corner (825 mm^2) does not.
        let big = scored(120.0, 750.0, &corners);
        assert_eq!(big.status, Status::Ok);
        let errs = claim_interval_rule(&big, ("a100", &base), &corners);
        assert!(
            errs.iter()
                .any(|e| e.diag.message.contains("pessimistic corner")),
            "{errs:?}"
        );
        let mut flip = win.clone();
        flip.interval
            .corner_flips
            .push(kiln_trace::result::CornerFlip {
                corner: Corner::Low,
                what: "infeasible".into(),
                code: None,
            });
        assert_eq!(codes(&flip, &corners), 1);
        let mut asym = base.clone();
        asym.provenance.calibration_hash = "cal1-other".into();
        assert_eq!(
            claim_interval_rule(&win, ("a100", &asym), &corners).len(),
            1
        );
    }

    #[test]
    fn mismatched_stacks_are_refused_unless_declared_realistic() {
        let base = with_stack(
            result_with(&[("d", 100.0)], 800.0, 400.0),
            "pytorch_cuda_graph_sdpa@stk1-y",
        );
        let cand = with_stack(
            result_with(&[("d", 150.0)], 700.0, 300.0),
            "kiln_ideal@stk1-z",
        );
        let mut c = cand.clone();
        apply(&mut c, Some(("a100", &base)), &Options::default());
        assert_eq!(c.status, Status::InternalError);
        assert_eq!(c.errors[0].diag.code, ASYM_CODE);
        assert!(
            c.errors[0].diag.message.contains("software stack"),
            "{:?}",
            c.errors
        );
        assert_eq!((c.score, c.score_interval), (0.0, None));
        let corners = opts(json!({"interval": "corners"}));
        let mut won = cand.clone();
        won.score_interval = Some(Interval::from_corners(1.5, 1.35, 1.65));
        won.score_realistic = Some(realistic_at(1.2));
        let errs = claim_interval_rule(&won, ("a100", &base), &corners);
        assert!(
            errs.iter()
                .any(|e| e.diag.message.contains("software stack")),
            "{errs:?}"
        );

        // `own` declares the realistic comparison: differing stacks are its point.
        let own = opts(json!({"stack": "own"}));
        let mut c = cand.clone();
        apply(&mut c, Some(("a100", &base)), &own);
        assert_eq!(c.status, Status::Ok, "{:?}", c.errors);
        let sc = c.score_components.clone().unwrap();
        assert_eq!(sc.candidate_stack.as_deref(), Some("kiln_ideal@stk1-z"));
        assert_eq!(
            sc.baseline_stack.as_deref(),
            Some("pytorch_cuda_graph_sdpa@stk1-y")
        );
        let rs = realistic(&cand, ("a100", &base), &Options::default()).unwrap();
        assert!((rs.score - 1.5).abs() < 1e-12);
        assert_eq!(rs.baseline_stack, "pytorch_cuda_graph_sdpa@stk1-y");
        let errs = claim_interval_rule(
            &c,
            ("a100", &base),
            &opts(json!({"interval": "corners", "stack": "own"})),
        );
        assert!(
            errs.iter()
                .any(|e| e.diag.path.as_deref() == Some("options.stack")),
            "{errs:?}"
        );
    }

    #[test]
    fn identical_design_realistic_score_is_one() {
        let base = with_stack(
            result_with(&[("decode_b1", 80.0), ("prefill_b1", 3000.0)], 800.0, 400.0),
            "pytorch_cuda_graph_sdpa@stk1-y",
        );
        let rs = realistic(&base, ("a100", &base), &Options::default()).unwrap();
        assert_eq!(rs.score, 1.0);
        assert_eq!(rs.interval.central, 1.0);
        assert!(rs.interval.low < 1.0);
        assert_eq!(rs.candidate_stack, rs.baseline_stack);
        let mut bad = base.clone();
        bad.status = Status::Infeasible;
        assert!(realistic(&bad, ("a100", &base), &Options::default()).is_err());
    }

    #[test]
    fn claim_requires_winning_under_both_stacks() {
        let base = result_with(&[("d", 100.0)], 800.0, 400.0);
        let corners = opts(json!({"interval": "corners"}));
        let mut c = result_with(&[("d", 120.0)], 700.0, 300.0);
        c.physical.as_mut().unwrap().package_mm2 = Interval::point(1800.0);
        apply(&mut c, Some(("a100", &base)), &corners);
        assert!(c.score_interval.unwrap().low >= 1.0);
        let claim = |rs: Option<RealisticScore>| {
            let mut x = c.clone();
            x.score_realistic = rs;
            claim_interval_rule(&x, ("a100", &base), &corners)
        };
        assert!(claim(Some(realistic_at(1.01))).is_empty());
        let lost = claim(Some(realistic_at(0.95)));
        assert_eq!(lost.len(), 1, "{lost:?}");
        assert_eq!(
            lost[0].diag.path.as_deref(),
            Some("score_realistic.interval.low")
        );
        let missing = claim(None);
        assert_eq!(missing[0].diag.path.as_deref(), Some("score_realistic"));
        // Winning only under the realistic stacks is not a claim either.
        let mut ideal_loss = result_with(&[("d", 105.0)], 700.0, 300.0);
        ideal_loss.physical.as_mut().unwrap().package_mm2 = Interval::point(1800.0);
        apply(&mut ideal_loss, Some(("a100", &base)), &corners);
        ideal_loss.score_realistic = Some(realistic_at(1.3));
        let errs = claim_interval_rule(&ideal_loss, ("a100", &base), &corners);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert_eq!(errs[0].diag.path.as_deref(), Some("score_interval.low"));
    }
}
