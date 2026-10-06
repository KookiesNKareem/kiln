//! Derived analysis over a trace (05 §3.8), shared by the views, the CLI and `kiln-py`: per-resource stats
//! with channel traffic attributed to memory endpoints, roofline points, time breakdowns and two-run alignment.

use std::collections::BTreeMap;

use crate::trace::{NONE_U32, Trace};

/// Per-resource aggregate over the selected phases.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResStat {
    /// Busy fraction of the run time; `None` when the engine reported nothing for the resource.
    pub util: Option<f64>,
    pub busy_s: f64,
    pub energy_j: f64,
    pub bytes: f64,
    pub flops: f64,
}

/// Per-op compute and reference-level memory bounds in seconds (FLOPs at the peak ceiling, bytes across the
/// reference boundary at its bandwidth).
fn op_bounds(t: &Trace) -> Vec<(f64, f64)> {
    let peak = t
        .ceilings
        .iter()
        .filter(|c| c.kind == 0)
        .map(|c| c.value)
        .fold(0.0, f64::max);
    let level = reference_level(t);
    let bw = t
        .ceilings
        .iter()
        .find(|c| c.kind == 1 && c.level == level)
        .map(|c| c.value);
    t.ops
        .iter()
        .map(|o| {
            let c = if peak > 0.0 { o.flops / peak } else { 0.0 };
            let m = match (level, bw) {
                (Some(l), Some(bw)) if bw > 0.0 => boundary_bytes(t, o, l) / bw,
                _ => 0.0,
            };
            (c, m)
        })
        .collect()
}

/// Bytes across the boundary of memory level `l` (see [`roofline_points`]).
pub fn boundary_bytes(t: &Trace, o: &crate::trace::OpRow, l: u8) -> f64 {
    let at = |l: u8| o.bytes_by_level.get(usize::from(l)).copied().unwrap_or(0.0);
    at(l) + inner_level(t, l).map_or(0.0, at)
}

/// Time attributed to each op: its execution group's duration split over the group's ops by the group's
/// binding term (FLOPs for a compute-bound group, reference-level bytes for a memory-bound one, the op
/// envelopes otherwise). Tier A op envelopes do not span their group's traffic (a fused GEMV's envelope ends
/// long before its weights finish streaming), while group times tile the scheduled window, so the bottleneck
/// and compare views sum these.
pub fn op_times(t: &Trace) -> Vec<f64> {
    let tick = t.tick_s();
    let own: Vec<f64> = t.ops.iter().map(|o| o.time_s(tick)).collect();
    let bounds = op_bounds(t);
    let mut out = own.clone();
    for g in &t.groups {
        let ids: Vec<usize> = g
            .ops
            .iter()
            .map(|&o| o as usize)
            .filter(|&o| o < out.len())
            .collect();
        let dur = (g.t_end - g.t_start) as f64 * tick;
        if ids.is_empty() || dur <= 0.0 {
            continue;
        }
        let weight = |i: usize| match t.binding_name(g.binding) {
            "compute" => bounds[i].0,
            "dram" | "port" => bounds[i].1,
            _ => own[i],
        };
        let w: Vec<f64> = ids.iter().map(|&i| weight(i)).collect();
        let ws: f64 = w.iter().sum();
        let os: f64 = ids.iter().map(|&i| own[i]).sum();
        for (k, &i) in ids.iter().enumerate() {
            out[i] = if ws > 0.0 {
                dur * w[k] / ws
            } else if os > 0.0 {
                dur * own[i] / os
            } else {
                dur / ids.len() as f64
            };
        }
    }
    out
}

pub fn phase_code(t: &Trace, phase: Option<&str>) -> Option<u8> {
    phase.and_then(|p| t.phases.iter().find(|x| x.id == p).map(|x| x.phase))
}

/// Run time of the selected phases (their makespans).
pub fn run_time(t: &Trace, phase: Option<u8>) -> f64 {
    t.phases
        .iter()
        .filter(|p| phase.is_none_or(|c| c == p.phase))
        .map(|p| p.makespan_s)
        .sum()
}

/// Stats per resource row. Memories without their own aggregates take the busiest channel touching them
/// (`a-b` link/port resources), so on-chip SRAMs are colored by their port traffic.
pub fn resource_stats(t: &Trace, phase: Option<u8>) -> Vec<ResStat> {
    let n = t.resources.len();
    let mut st = vec![ResStat::default(); n];
    let total = run_time(t, phase);
    let mut busy = vec![0.0; n];
    let mut seen = vec![false; n];
    for a in &t.aggregates_resource {
        if phase.is_some_and(|p| p != a.phase) || a.resource as usize >= n {
            continue;
        }
        let i = a.resource as usize;
        seen[i] = true;
        busy[i] += a.busy_s;
        st[i].busy_s += a.busy_s;
        st[i].energy_j += a.energy_dyn_j + a.energy_leak_j;
        st[i].bytes += a.bytes;
        st[i].flops += a.flops;
    }
    for i in 0..n {
        if seen[i] {
            st[i].util = Some(if total > 0.0 {
                (busy[i] / total).min(1.0)
            } else {
                0.0
            });
        }
    }
    // Channel endpoints.
    let channel =
        crate::trace::code(&t.manifest.enums, "resources.kind", "channel").map(|c| c as u16);
    let mut extra: Vec<ResStat> = vec![ResStat::default(); n];
    let mut has_extra = vec![false; n];
    for (i, r) in t.resources.iter().enumerate() {
        if Some(r.kind) != channel || !seen[i] {
            continue;
        }
        for (k, _) in r.path.match_indices('-') {
            for end in [&r.path[..k], &r.path[k + 1..]] {
                if let Some(e) = t.resource_by_path(end) {
                    let e = e as usize;
                    if seen[e] {
                        continue;
                    }
                    has_extra[e] = true;
                    let x = &mut extra[e];
                    x.util = Some(x.util.unwrap_or(0.0).max(st[i].util.unwrap_or(0.0)));
                    x.energy_j += st[i].energy_j;
                    x.bytes += st[i].bytes;
                    x.busy_s = x.busy_s.max(st[i].busy_s);
                }
            }
        }
    }
    for i in 0..n {
        if has_extra[i] {
            st[i] = extra[i];
        }
    }
    st
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoofPoint {
    /// Op index, or the first op of an aggregated family.
    pub op: u32,
    pub label: String,
    pub phase: u8,
    pub kind: u16,
    pub flops: f64,
    pub bytes: f64,
    pub time_s: f64,
    /// Attained FLOP/s at the slow / fast corners.
    pub attained_low: Option<f64>,
    pub attained_high: Option<f64>,
    pub binding: Option<u8>,
    pub count: u32,
}

impl RoofPoint {
    pub fn intensity(&self) -> f64 {
        self.flops / self.bytes
    }

    pub fn attained(&self) -> f64 {
        self.flops / self.time_s
    }
}

/// The reference memory level of the roofline: the outermost level with a bandwidth ceiling.
pub fn reference_level(t: &Trace) -> Option<u8> {
    t.ceilings
        .iter()
        .filter(|c| c.kind == 1)
        .filter_map(|c| c.level)
        .max()
}

/// The next level inward of `l` among the levels with a bandwidth ceiling.
pub fn inner_level(t: &Trace, l: u8) -> Option<u8> {
    t.ceilings
        .iter()
        .filter(|c| c.kind == 1)
        .filter_map(|c| c.level)
        .filter(|&x| x < l)
        .max()
}

/// Roofline points, one per execution group (03's unit of time: the ops of a fused group share it), labelled
/// by the group's largest-FLOP op, plus one per ungrouped op; `aggregate` merges points by label family across
/// layers. 03 records bytes *delivered into* each memory level, so the traffic across the boundary of `level` is
/// what was delivered into it (writes) plus what was delivered into the next level inward (reads from it).
pub fn roofline_points(
    t: &Trace,
    phase: Option<u8>,
    level: Option<u8>,
    aggregate: bool,
) -> Vec<RoofPoint> {
    let tick = t.tick_s();
    let bind = t.binding_of_ops();
    let bytes_of = |o: &crate::trace::OpRow| match level {
        Some(l) => boundary_bytes(t, o, l),
        None => o.bytes_by_level.iter().sum(),
    };
    // (member ops, duration, binding) per unit.
    let mut units: Vec<(Vec<usize>, f64, Option<u8>)> = vec![];
    let mut grouped = vec![false; t.ops.len()];
    for g in &t.groups {
        let ids: Vec<usize> = g
            .ops
            .iter()
            .map(|&o| o as usize)
            .filter(|&o| o < t.ops.len())
            .collect();
        if ids.is_empty() {
            continue;
        }
        ids.iter().for_each(|&i| grouped[i] = true);
        units.push((ids, (g.t_end - g.t_start) as f64 * tick, Some(g.binding)));
    }
    for (i, o) in t.ops.iter().enumerate() {
        if !grouped[i] {
            units.push((vec![i], o.time_s(tick), bind[i].map(|l| l.binding)));
        }
    }
    let mut pts: Vec<RoofPoint> = vec![];
    let mut fam: BTreeMap<(u8, String), usize> = BTreeMap::new();
    let mut corner: Vec<(f64, f64)> = vec![];
    for (ids, dur, binding) in units {
        let main = *ids
            .iter()
            .max_by(|&&a, &&b| t.ops[a].flops.total_cmp(&t.ops[b].flops).then(b.cmp(&a)))
            .expect("non-empty");
        let o = &t.ops[main];
        if phase.is_some_and(|p| p != o.phase) {
            continue;
        }
        let flops: f64 = ids.iter().map(|&i| t.ops[i].flops).sum();
        let bytes: f64 = ids.iter().map(|&i| bytes_of(&t.ops[i])).sum();
        if flops <= 0.0 || bytes <= 0.0 || dur <= 0.0 {
            continue;
        }
        // Corner times scale the unit duration by the members' envelope ratio.
        let own: f64 = ids.iter().map(|&i| t.ops[i].time_s(tick)).sum();
        let ratio = |f: fn(&crate::trace::OpRow) -> Option<f64>| -> Option<f64> {
            let s: Option<f64> = ids.iter().map(|&i| f(&t.ops[i])).sum();
            s.filter(|_| own > 0.0).map(|s| dur * s / own)
        };
        let (slow, fast) = (ratio(|o| o.time_high_s), ratio(|o| o.time_low_s));
        let label = if aggregate {
            o.family.clone()
        } else {
            o.path.clone()
        };
        if aggregate && let Some(&k) = fam.get(&(o.phase, label.clone())) {
            let q = &mut pts[k];
            q.flops += flops;
            q.bytes += bytes;
            q.time_s += dur;
            q.count += 1;
            corner[k].0 += slow.unwrap_or(dur);
            corner[k].1 += fast.unwrap_or(dur);
            continue;
        }
        fam.insert((o.phase, label.clone()), pts.len());
        corner.push((slow.unwrap_or(dur), fast.unwrap_or(dur)));
        pts.push(RoofPoint {
            op: main as u32,
            label,
            phase: o.phase,
            kind: o.kind,
            flops,
            bytes,
            time_s: dur,
            attained_low: slow.map(|s| flops / s),
            attained_high: fast.map(|f| flops / f),
            binding,
            count: 1,
        });
    }
    if aggregate {
        for (k, p) in pts.iter_mut().enumerate() {
            if p.attained_low.is_some() {
                p.attained_low = Some(p.flops / corner[k].0);
                p.attained_high = Some(p.flops / corner[k].1);
            }
        }
    }
    pts
}

/// Time per binding class over the selected phases (sums to the makespans, 03 §10).
pub fn time_by_binding(t: &Trace, phase: Option<u8>) -> BTreeMap<u8, f64> {
    let mut m = BTreeMap::new();
    for b in &t.bottleneck {
        if let (0, true, Some(c), Some(x)) = (
            b.section,
            phase.is_none_or(|p| p == b.phase),
            b.binding,
            b.time_s,
        ) {
            *m.entry(c).or_insert(0.0) += x;
        }
    }
    m
}

/// Scheduled (group-attributed) time per op family split by the binding class of each op (bottleneck bars).
pub fn family_breakdown(t: &Trace, phase: Option<u8>) -> Vec<(String, BTreeMap<u8, f64>)> {
    let times = op_times(t);
    let bind = t.binding_of_ops();
    let mut m: BTreeMap<&str, BTreeMap<u8, f64>> = BTreeMap::new();
    for (i, o) in t.ops.iter().enumerate() {
        if phase.is_some_and(|p| p != o.phase) {
            continue;
        }
        let b = bind[i].map_or(u8::MAX, |l| l.binding);
        *m.entry(o.family.as_str())
            .or_default()
            .entry(b)
            .or_insert(0.0) += times[i];
    }
    let mut v: Vec<(String, BTreeMap<u8, f64>)> =
        m.into_iter().map(|(k, x)| (k.to_string(), x)).collect();
    v.sort_by(|a, b| {
        let (sa, sb): (f64, f64) = (a.1.values().sum(), b.1.values().sum());
        sb.total_cmp(&sa).then(a.0.cmp(&b.0))
    });
    v
}

/// Upper bound on time recoverable per limiter term (05 §6.5): for each op, its attributed time times
/// `1 - runner-up/binding` (the op would drop to its runner-up term), grouped by (binding class, resource).
/// Returns `(binding, resource, seconds, ops)`, largest first.
pub fn recoverable(t: &Trace, phase: Option<u8>) -> Vec<(u8, Option<u32>, f64, u32)> {
    let times = op_times(t);
    let mut ratio: BTreeMap<u32, f64> = BTreeMap::new();
    for l in t.limiters.iter().filter(|l| l.rank == 1) {
        ratio.insert(l.op, l.attained_frac.clamp(0.0, 1.0));
    }
    let mut m: BTreeMap<(u8, u32), (f64, u32)> = BTreeMap::new();
    for b in t.limiters.iter().filter(|l| l.rank == 0) {
        let Some(o) = t.ops.get(b.op as usize) else {
            continue;
        };
        if phase.is_some_and(|p| o.phase != p) {
            continue;
        }
        let gain = times[b.op as usize] * (1.0 - ratio.get(&b.op).copied().unwrap_or(0.0));
        let e = m
            .entry((b.binding, b.resource.unwrap_or(NONE_U32)))
            .or_default();
        e.0 += gain.max(0.0);
        e.1 += 1;
    }
    let mut v: Vec<(u8, Option<u32>, f64, u32)> = m
        .into_iter()
        .map(|((c, r), (g, n))| (c, (r != NONE_U32).then_some(r), g, n))
        .collect();
    v.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    v
}

/// One op-family row of a two-run comparison (05 §6.6).
#[derive(Clone, Debug, PartialEq)]
pub struct OpDelta {
    pub key: String,
    pub a_s: Option<f64>,
    pub b_s: Option<f64>,
    pub a_binding: Option<u8>,
    pub b_binding: Option<u8>,
}

impl OpDelta {
    pub fn delta(&self) -> f64 {
        self.b_s.unwrap_or(0.0) - self.a_s.unwrap_or(0.0)
    }
}

/// Aligns two traces by `phase/op family` (op paths match when the workloads match, 05 §6.6 rule 1).
/// Returns rows sorted by |delta| and whether the workload hashes matched.
pub fn align(a: &Trace, b: &Trace) -> (Vec<OpDelta>, bool) {
    fn sums(t: &Trace) -> BTreeMap<String, (f64, BTreeMap<u8, f64>)> {
        let times = op_times(t);
        let bind = t.binding_of_ops();
        let mut m: BTreeMap<String, (f64, BTreeMap<u8, f64>)> = BTreeMap::new();
        for (i, o) in t.ops.iter().enumerate() {
            let k = format!("{}/{}", t.phase_name(o.phase), o.family);
            let e = m.entry(k).or_default();
            let dt = times[i];
            e.0 += dt;
            if let Some(l) = bind[i] {
                *e.1.entry(l.binding).or_insert(0.0) += dt;
            }
        }
        m
    }
    let dom = |m: &BTreeMap<u8, f64>| {
        m.iter()
            .max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(x.0)))
            .map(|x| *x.0)
    };
    let (sa, sb) = (sums(a), sums(b));
    let mut keys: Vec<&String> = sa.keys().chain(sb.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut rows: Vec<OpDelta> = keys
        .into_iter()
        .map(|k| OpDelta {
            key: k.clone(),
            a_s: sa.get(k).map(|x| x.0),
            b_s: sb.get(k).map(|x| x.0),
            a_binding: sa.get(k).and_then(|x| dom(&x.1)),
            b_binding: sb.get(k).and_then(|x| dom(&x.1)),
        })
        .collect();
    rows.sort_by(|x, y| {
        y.delta()
            .abs()
            .total_cmp(&x.delta().abs())
            .then(x.key.cmp(&y.key))
    });
    (
        rows,
        a.manifest.provenance.workload_hash == b.manifest.provenance.workload_hash,
    )
}
