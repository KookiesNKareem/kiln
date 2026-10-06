//! Mapping search (03 §2.3-§2.4): mode choice, spatial candidates, ragged tile classes, loop-order enumeration
//! with LOMA-style bottom-up allocation, double-buffer and capacity-share choices.

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{Precision, PrecisionKind, PrecisionSpec};

use crate::model::{ClassEval, MAX_DIMS};
use crate::nest::{dim_class, err};
use crate::types::*;

/// Loop prime factors per class are coarsened to at most this many (LOMA's `lpf_limit`).
pub(crate) const LPF_LIMIT: usize = 6;
/// Ragged dims beyond this many are padded instead of split (2^k tile classes).
const MAX_RAGGED: usize = 3;

fn base(p: Precision) -> Precision {
    use Precision::*;
    match p {
        Mxfp8E4m3 => Fp8E4m3,
        Mxfp8E5m2 => Fp8E5m2,
        Mxfp6E3m2 => Fp6E3m2,
        Mxfp6E2m3 => Fp6E2m3,
        Mxfp4 | Nvfp4 => Fp4E2m1,
        Mxint8 => Int8,
        p => p.compute(),
    }
}

fn matches(mode: PrecisionSpec, op: PrecisionSpec) -> Option<bool> {
    if mode.precision == op.precision {
        Some(true)
    } else if mode.precision == base(op.precision) && !matches!(mode.precision.kind(), PrecisionKind::Mx) {
        Some(false)
    } else {
        None
    }
}

/// `acc` holds every value an accumulator of precision `required` holds (one rule with kiln-wl's convert choice).
pub use kiln_wl::convert::accumulates;

/// Exact-precision mode if one exists, else a mode on the element type of MX/block/scaled operands (scales then
/// cost a vector pass, 03 §2.7) whose accumulator holds the nest's required one. Among equals, the fastest. Never a
/// silent upcast.
pub(crate) fn select_mode(unit: &UnitTemplate, nest: &OpNest) -> Result<usize, Diagnostic> {
    let pick = |r: OperandRole| nest.operands.iter().find(|o| o.role == r && !o.is_output).map(|o| o.dtype);
    let inputs: Vec<PrecisionSpec> = nest.operands.iter().filter(|o| !o.is_output).map(|o| o.dtype).collect();
    let (pa, pb) = match nest.kind {
        NestKind::Contraction => {
            let a = pick(OperandRole::A).or(inputs.first().copied());
            let b = pick(OperandRole::B).or(inputs.get(1).copied()).or(a);
            (a, b)
        }
        _ => {
            let x = inputs.first().copied().or_else(|| nest.operands.first().map(|o| o.dtype));
            (x, x)
        }
    };
    let (Some(pa), Some(pb)) = (pa, pb) else {
        return Err(err("E-COST-NEST", "nest has no operands"));
    };
    // Exact operand match, then throughput, then the widest accumulator, then the mode text: never declaration order.
    let mut best: Option<(usize, (bool, f64, u32, String))> = None;
    for (i, m) in unit.modes.iter().enumerate() {
        let (Some(ea), Some(eb)) = (matches(m.a, pa), matches(m.b, pb)) else { continue };
        if nest.accum.is_some_and(|r| !accumulates(m.acc.precision, r)) {
            continue;
        }
        let key = (ea && eb, m.macs_per_cycle, m.acc.precision.element_bits(), format!("{}*{}+{}", m.a, m.b, m.acc));
        if best.as_ref().is_none_or(|(_, k)| key.0.cmp(&k.0).then(key.1.total_cmp(&k.1)).then(key.2.cmp(&k.2)).then(k.3.cmp(&key.3)).is_gt()) {
            best = Some((i, key));
        }
    }
    best.map(|b| b.0).ok_or_else(|| {
        let acc = nest.accum.map_or_else(String::new, |r| format!(" accumulating in {r}"));
        err("E-COST-UNMAPPABLE", format!("unit {} has no precision mode for {pa} x {pb}{acc}", unit.name))
            .at(unit.name.clone())
            .hint("add a matching precision mode to the unit, or quantize the operands (upcast is a mapping choice, not implicit)")
    })
}

fn allowed(unit: &UnitTemplate, nest: &OpNest, a: usize, d: usize) -> bool {
    let al = &unit.axes[a].allowed;
    al.is_empty() || al.iter().any(|x| *x == nest.dims[d].name || Some(x.as_str()) == dim_class(nest, d))
}

fn largest_divisor_le(s: u64, cap: u64) -> u64 {
    (1..=cap.min(s)).rev().find(|&x| s.is_multiple_of(x)).unwrap_or(1)
}

fn unroll_options(s: u64, size: u64) -> Vec<u64> {
    let mut v = vec![s.min(size), largest_divisor_le(s, size)];
    for k in [2, 4] {
        let x = size / k;
        if x >= 1 && x <= s {
            v.push(x);
        }
    }
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.dedup();
    v
}

struct SpatialEnum<'a> {
    unit: &'a UnitTemplate,
    nest: &'a OpNest,
    array: f64,
    points: f64,
    unroll: [u64; MAX_DIMS],
    cur: Vec<Vec<(DimIx, u64)>>,
    out: Vec<(f64, u64, SpatialMapping)>,
}

impl SpatialEnum<'_> {
    /// Binds dims to axis `a` with `room` lanes left on it. On a reduction axis, a second dim joins once the
    /// first is unrolled completely (coalescing contracting dims: a contraction over `(h, d)` with `d = 128`
    /// fills a 256-row array with two heads, as XLA's dot over flattened contracting dims does).
    fn rec(&mut self, a: usize, room: u64) {
        let nd = self.nest.dims.len();
        if a == self.unit.axes.len() {
            let temporal: f64 = (0..nd).map(|d| self.nest.dims[d].size.div_ceil(self.unroll[d]) as f64).product();
            let util = self.points / (temporal * self.array);
            let unroll = &self.unroll;
            let demand: u64 = self
                .nest
                .operands
                .iter()
                .filter(|o| !o.is_output)
                .map(|o| {
                    o.axes
                        .iter()
                        .map(|ax| ax.terms.iter().map(|&(d, c)| c.unsigned_abs() * (unroll[d] - 1)).sum::<u64>() + 1)
                        .product::<u64>()
                })
                .sum();
            self.out.push((util, demand, SpatialMapping { axes: self.cur.clone() }));
            return;
        }
        let next = self.unit.axes.get(a + 1).map_or(1, |x| u64::from(x.size));
        self.rec(a + 1, next);
        let first = match self.cur[a].as_slice() {
            [] => None,
            &[(d0, u0)] if u0 == self.nest.dims[d0].size && room >= 2 && self.unit.axes[a].allowed.iter().any(|x| x == "k") => Some(d0),
            _ => return,
        };
        for d in 0..nd {
            if !allowed(self.unit, self.nest, a, d) || first == Some(d) {
                continue;
            }
            let remaining = self.nest.dims[d].size.div_ceil(self.unroll[d]);
            if remaining <= 1 {
                continue;
            }
            for u in unroll_options(remaining, room).into_iter().filter(|&u| u > 1) {
                // Two complete unrolls on one axis are the same mapping in either order.
                if first.is_some_and(|d0| u == remaining && d < d0) {
                    continue;
                }
                self.unroll[d] *= u;
                self.cur[a].push((d, u));
                self.rec(a, room / u);
                self.cur[a].pop();
                self.unroll[d] /= u;
            }
        }
    }
}

/// Spatial candidates ranked by spatial utilization, then lower per-cycle input demand (03 §2.3).
pub(crate) fn spatial_candidates(unit: &UnitTemplate, nest: &OpNest, k: usize) -> Vec<SpatialMapping> {
    let mut e = SpatialEnum {
        unit,
        nest,
        array: unit.axes.iter().map(|a| f64::from(a.size)).product(),
        points: nest.box_points() as f64,
        unroll: [1; MAX_DIMS],
        cur: vec![vec![]; unit.axes.len()],
        out: vec![],
    };
    e.rec(0, unit.axes.first().map_or(1, |x| u64::from(x.size)));
    let mut out = e.out;
    // Coalesced bindings only enter when they fill the array better than every single-dim binding.
    let coalesced = |m: &SpatialMapping| m.axes.iter().any(|x| x.len() > 1);
    let single = out.iter().filter(|c| !coalesced(&c.2)).map(|c| c.0).fold(0.0, f64::max);
    out.retain(|c| !coalesced(&c.2) || c.0 > single * (1.0 + 1e-12));
    let best = out.iter().map(|c| c.0).fold(0.0, f64::max);
    out.retain(|c| c.0 >= 0.5 * best);
    out.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    out.truncate(k.max(1));
    out.into_iter().map(|c| c.2).collect()
}

/// Tile classes of a spatial mapping: ragged dims bound on one axis split into full + remainder (03 §2.3).
pub(crate) fn classes(nest: &OpNest, sp: &SpatialMapping, policy: RaggedPolicy) -> Vec<(Vec<u64>, SpatialMapping)> {
    let nd = nest.dims.len();
    let sizes: Vec<u64> = nest.dims.iter().map(|d| d.size).collect();
    if policy == RaggedPolicy::Pad {
        return vec![(sizes, sp.clone())];
    }
    let mut unroll = vec![1u64; nd];
    let mut nbind = vec![0usize; nd];
    for binds in &sp.axes {
        for &(d, u) in binds {
            unroll[d] *= u;
            nbind[d] += 1;
        }
    }
    let ragged: Vec<usize> = (0..nd)
        .filter(|&d| nbind[d] == 1 && sizes[d] > unroll[d] && !sizes[d].is_multiple_of(unroll[d]))
        .take(MAX_RAGGED)
        .collect();
    let mut out = vec![];
    for mask in 0..(1u32 << ragged.len()) {
        let mut sz = sizes.clone();
        let mut s = sp.clone();
        for (i, &d) in ragged.iter().enumerate() {
            let rem = sizes[d] % unroll[d];
            if mask & (1 << i) != 0 {
                sz[d] = rem;
                for binds in &mut s.axes {
                    for b in binds.iter_mut().filter(|b| b.0 == d) {
                        b.1 = rem;
                    }
                }
            } else {
                sz[d] = sizes[d] - rem;
            }
        }
        out.push((sz, s));
    }
    out
}

fn factorize(mut n: u64) -> Vec<u64> {
    let mut f = vec![];
    let mut p = 2;
    while p * p <= n {
        while n.is_multiple_of(p) {
            f.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// `x`'s prime factors coarsened to at most `n` items as [`loop_items`] does (smallest two merged first).
pub(crate) fn coarsen(x: u64, n: usize) -> Vec<u64> {
    let mut f = factorize(x);
    while f.len() > n.max(1) {
        f.sort_unstable();
        let y = f.remove(0);
        f[0] *= y;
    }
    f
}

/// Loop items per dim (prime factors, coarsened to at most `LPF_LIMIT` in total).
pub(crate) fn loop_items(t: &[u64]) -> Vec<TemporalLoop> {
    let mut per: Vec<Vec<u64>> = t.iter().map(|&x| factorize(x)).collect();
    while per.iter().map(Vec::len).sum::<usize>() > LPF_LIMIT {
        let Some(d) = (0..per.len()).filter(|&d| per[d].len() > 1).max_by_key(|&d| (per[d].len(), std::cmp::Reverse(d))) else {
            break;
        };
        per[d].sort_unstable();
        let x = per[d].remove(0);
        per[d][0] *= x;
    }
    let mut items = vec![];
    for (d, fs) in per.into_iter().enumerate() {
        for f in fs {
            items.push(TemporalLoop { dim: d, factor: f });
        }
    }
    items.sort();
    items
}

pub(crate) fn objective_value(obj: Objective, cycles: f64, energy: f64, floor_c: f64, floor_e: f64) -> f64 {
    match obj {
        Objective::Latency => cycles,
        Objective::Energy => energy,
        Objective::Edp => cycles * energy,
        Objective::Weighted { w_lat_milli, w_e_milli } => {
            f64::from(w_lat_milli) * cycles / floor_c.max(1.0) + f64::from(w_e_milli) * energy / floor_e.max(f64::MIN_POSITIVE)
        }
    }
}

pub(crate) fn better(obj: Objective, a: &ClassEval, b: &ClassEval, fc: f64, fe: f64) -> std::cmp::Ordering {
    let va = objective_value(obj, a.total_f, a.energy.total_j, fc, fe);
    let vb = objective_value(obj, b.total_f, b.energy.total_j, fc, fe);
    let tie = match obj {
        Objective::Energy => a.total_f.total_cmp(&b.total_f),
        _ => a.energy.total_j.total_cmp(&b.energy.total_j),
    };
    va.total_cmp(&vb).then(tie)
}

