//! Intra-unit cost interface (03 §2). [`RooflineCost`] is the M1 stand-in for `kiln-cost`: structural
//! array-shape quantization and capacity-limited tile reuse, no loop-order search. `kiln-cost` plugs in
//! through [`UnitCostModel`] once its search lands.

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::{ComputeKind, Geometry, PrecisionMode};
use kiln_ir::op_class::{OpClass, SpecialFn};
use kiln_ir::precision::Precision;
use kiln_ir::wl::KernelClass;

use crate::geom::Slice;
use crate::program::{DimRole, POp, Program};

pub struct NestQuery<'a> {
    pub prog: &'a Program,
    pub op: &'a POp,
    pub slice: &'a Slice,
    /// Live points of `slice` ([`crate::geom::slice_points`]).
    pub points: u128,
    pub hw: &'a HwModel,
    pub unit: usize,
    /// Per-unit capacity share of each private chain level, feed memory first.
    pub level_caps: &'a [u64],
    /// Memory instance (`MemIx`) of each private chain level, as `level_caps`.
    pub level_mems: &'a [usize],
    /// Read bandwidth (B/s) this unit gets from each shared on-chip chain level `(MemIx, B/s)`.
    pub level_bw: &'a [(usize, f64)],
    /// Per operand, the memory (`MemIx`) an input is read from before the op or an output must end at; None (or
    /// past the end) for the top of the unit's chain.
    pub residency: &'a [Option<usize>],
    /// Units ganged with `unit` (itself included): the slice runs on all of them.
    pub gang: u32,
    /// Candidate scoring: a cheaper search is acceptable.
    pub quick: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NestCost {
    pub cycles: f64,
    pub fill_cycles: f64,
    pub useful_macs: u128,
    pub issued_macs: u128,
    pub vec_ops: u128,
    /// Of `vec_ops`, conversions (convert class rate). A contraction's vector work runs beside its MAC unit.
    pub cvt_ops: u128,
    pub transc_ops: u128,
    /// Bytes crossing the feed boundary per operand (index = kernel operand).
    pub feed_bytes: Vec<f64>,
    /// Re-read multiplier per private level (outer index, as `level_caps`) and operand (>= 1).
    pub reread: Vec<Vec<f64>>,
    /// Bytes the unit reads back through its feed per operand: an output's partial sums.
    pub readback: Vec<f64>,
    /// Partial-sum traffic of the output above the feed: per memory level (entity), the bytes written up into it
    /// from the level below and read back down from it, beyond the final result's one pass.
    pub spill: Vec<(String, f64, f64)>,
    pub mode: String,
    pub bits: (u32, u32),
    /// Cycles on each special-function unit next to the slice's vector unit(s) ([`special_units`]), at that
    /// unit's clock: the transcendentals they implement, run alongside the vector work in `cycles`.
    pub special_cycles: f64,
}

/// Special-function units (01 §5.4) a vector unit hands its transcendentals to: enabled, not near-memory,
/// declaring op class `transcendental`, fed from the vector unit's own feed memory (v5e's EUP next to the VPU on
/// the vregs, an SMSP's MUFU next to its ALUs on the register file).
pub fn special_units(hw: &HwModel, unit: usize) -> Vec<usize> {
    let mems: Vec<usize> = hw.units[unit].feeds.values().map(|f| f.mem).collect();
    (0..hw.units.len())
        .filter(|&u| {
            let x = &hw.units[u];
            matches!(x.spec.kind, ComputeKind::Special(_))
                && hw.nodes[x.node].enabled
                && x.near.is_none()
                && x.ops.contains(&OpClass::Transcendental)
                && x.feeds.values().any(|f| mems.contains(&f.mem))
        })
        .collect()
}

/// Special functions that implement each transcendental body term, preferred first, with the vector multiplies
/// each needs (exp via exp2 scales by log2(e), log via log2 by ln 2).
fn special_fns(field: usize) -> &'static [(SpecialFn, u16)] {
    use SpecialFn::*;
    match field {
        0 => &[(Exp, 0), (Exp2, 1)],
        1 => &[(Log, 0), (Log2, 1)],
        2 => &[(Recip, 0)],
        3 => &[(Rsqrt, 0)],
        4 => &[(Tanh, 0)],
        5 => &[(Erf, 0)],
        _ => &[(Sin, 0), (Cos, 0)],
    }
}

/// Per point, for the transcendentals of `body` that the special units `specials` (one vector unit's, or one gang
/// member's) implement: their count, the cycles they take on each special unit when `gang` vector units share
/// the slice, and the extra vector multiplies they need there. The rest stay on the vector unit.
fn special_terms(hw: &HwModel, specials: &[usize], body: &kiln_ir::wl::ScalarBody, dt: Precision, gang: f64) -> (f64, f64, f64) {
    let terms = [body.exp, body.log, body.rcp, body.rsqrt, body.tanh, body.erf, body.sin_cos];
    let rate = |u: usize, f: SpecialFn| -> f64 {
        let x = &hw.units[u];
        let ComputeKind::Special(sp) = &x.spec.kind else { return 0.0 };
        if !sp.functions.contains(&f) {
            return 0.0;
        }
        let modes = elem_modes(hw, u);
        let Some((_, r)) = modes.iter().copied().find(|(p, _)| *p == dt).or_else(|| modes.iter().copied().find(|(p, _)| *p == Precision::Fp32)) else {
            return 0.0;
        };
        f64::from(sp.lanes) * r * sp.fn_rates.get(&f).copied().unwrap_or(1.0)
    };
    let (mut n_sup, mut cycles, mut muls) = (0.0, 0.0, 0.0);
    for (i, &n) in terms.iter().enumerate().filter(|(_, n)| **n > 0) {
        let best = special_fns(i).iter().map(|&(f, m)| (specials.iter().map(|&u| rate(u, f)).sum::<f64>(), m)).find(|(r, _)| *r > 0.0);
        if let Some((r, m)) = best {
            n_sup += f64::from(n);
            cycles += f64::from(n) / (r * gang);
            muls += f64::from(n * m);
        }
    }
    (n_sup, cycles, muls)
}

fn nominal_hz(hw: &HwModel, unit: usize) -> f64 {
    hw.units[unit].clock.map_or(1.0e9, |c| hw.clocks[c].spec.freq.0)
}

/// Costs one slice of an op on one unit. A cost may depend on the slice's position only through its live point
/// count and its operands' footprint sizes (masked or param segments; box segments: not at all), and through its
/// start within floor-division and scale blocks: the lowerer memoizes costs on extents, segment and those.
pub trait UnitCostModel: Send + Sync {
    fn name(&self) -> &str;
    fn cost(&self, q: &NestQuery) -> Result<NestCost, Diagnostic>;

    /// A cheap cost no field of which (`cycles`, `fill_cycles`, `feed_bytes`, `reread`) exceeds that of
    /// [`UnitCostModel::cost`] for the same query, with the same shape; lets the mapper skip split candidates
    /// whose lower-bound estimate cannot win. `None`: no such bound (the mapper costs every candidate).
    fn cost_floor(&self, _q: &NestQuery) -> Option<Result<NestCost, Diagnostic>> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RooflineCost;

/// Bytes of `elems` elements of operand `oi`, block scales included.
fn bytes_of(prog: &Program, op: &POp, oi: usize, elems: u128) -> f64 {
    prog.tensors[op.operands[oi].tensor].bytes(elems) as f64
}

struct Mnk {
    b: f64,
    m: f64,
    n: f64,
    k: f64,
}

fn mnk(op: &POp, s: &Slice) -> Mnk {
    let mut r = Mnk { b: 1.0, m: 1.0, n: 1.0, k: 1.0 };
    if let Some(mr) = &op.mac {
        for (d, role) in mr.dims.iter().enumerate() {
            let e = s.extent(d) as f64;
            match role {
                DimRole::Batch => r.b *= e,
                DimRole::M | DimRole::Other => r.m *= e,
                DimRole::N => r.n *= e,
                DimRole::K => r.k *= e,
            }
        }
    }
    r
}

fn ceil(a: f64, b: f64) -> f64 {
    (a / b.max(1.0)).ceil()
}

fn elem_modes(hw: &HwModel, unit: usize) -> Vec<(Precision, f64)> {
    hw.units[unit]
        .spec
        .precisions
        .iter()
        .filter_map(|m| match m {
            PrecisionMode::Elem { dtype, rate } => Some((dtype.precision, *rate)),
            _ => None,
        })
        .collect()
}

/// The element mode unit `unit` runs vector work over `dtypes` in: the fastest declared mode that holds every one
/// of them (exactly or by a lossless widening), exact matches first. An error when it declares none, or no lanes.
pub(crate) fn vector_mode(hw: &HwModel, unit: usize, dtypes: &[Precision]) -> Result<(Precision, f64), Diagnostic> {
    let u = &hw.units[unit];
    let holds = |p: Precision, d: Precision| kiln_cost::accumulates(p, d) || kiln_wl::convert::widens(p, &kiln_ir::wl::ElemType::plain(d));
    elem_modes(hw, unit)
        .into_iter()
        .filter(|&(p, r)| r > 0.0 && r.is_finite() && u.spec.kind.base_ops_per_cycle() > 0 && dtypes.iter().all(|&d| holds(p, d)))
        .max_by(|x, y| dtypes.contains(&x.0).cmp(&dtypes.contains(&y.0)).then(x.1.total_cmp(&y.1)))
        .ok_or_else(|| {
            let names: Vec<&str> = dtypes.iter().map(|d| d.name()).collect();
            Diagnostic::error("E-MAP-PREC-002", format!("{} has no element mode for {}", hw.nodes[u.node].path, names.join(" and ")))
                .hint("declare an element mode that holds the precision (e.g. \"fp32@1\") or map the work elsewhere")
        })
}

/// Precisions vector work over `op` must hold: its operands' but the index operands of indirect accesses; over
/// float data only the floats (integer operands such as positions or an argmax result take the integer path).
pub fn vector_dtypes(prog: &Program, op: &POp) -> Vec<Precision> {
    let via: Vec<&str> = op
        .kernel
        .operands
        .iter()
        .flat_map(|o| &o.index)
        .filter_map(|i| match i {
            kiln_ir::wl::IndexExpr::Indirect { via, .. } => Some(via.as_str()),
            _ => None,
        })
        .collect();
    let mut v: Vec<Precision> = vec![];
    for (o, k) in op.operands.iter().zip(&op.kernel.operands) {
        let p = prog.tensors[o.tensor].dtype.scalar.compute();
        if !via.contains(&k.tensor.as_str()) && !v.contains(&p) {
            v.push(p);
        }
    }
    if v.iter().any(|p| p.is_float()) {
        v.retain(|p| p.is_float());
    }
    if v.is_empty() {
        v.push(Precision::Fp32);
    }
    v
}

impl UnitCostModel for RooflineCost {
    fn name(&self) -> &str {
        "roofline-m1"
    }

    fn cost(&self, q: &NestQuery) -> Result<NestCost, Diagnostic> {
        let (op, s, hw) = (q.op, q.slice, q.hw);
        let ui = &hw.units[q.unit];
        let points = q.points as f64;
        let density = if s.box_points() == 0 { 0.0 } else { points / s.box_points() as f64 };
        let fp = |oi: usize| {
            let t = &q.prog.tensors[op.operands[oi].tensor];
            let b = crate::geom::footprint(op, oi, s, &t.shape);
            bytes_of(q.prog, op, oi, b.elems())
        };
        let nops = op.operands.len();
        let mut feed_bytes: Vec<f64> = (0..nops).map(fp).collect();
        let mut reread = vec![vec![1.0; nops]; q.level_caps.len()];
        let body = op.body();
        if op.class() == KernelClass::Contraction {
            let mr = op.mac.as_ref().ok_or_else(|| Diagnostic::error("E-MAP-OP-003", "contraction without operand roles"))?;
            let prec = |oi: usize| kiln_wl::convert::operand_spec(&q.prog.tensors[op.operands[oi].tensor].dtype).precision;
            let (pa, pb) = (prec(mr.a), prec(mr.b));
            let exact = |a: Precision, b: Precision| a == pa && b == pb;
            let mode = ui
                .spec
                .precisions
                .iter()
                .filter_map(|m| match m {
                    PrecisionMode::Mac { a, b, acc, rate, .. }
                        if kiln_wl::convert::accepts(a.precision, pa)
                            && kiln_wl::convert::accepts(b.precision, pb)
                            && op.kernel.accum.is_none_or(|r| kiln_cost::accumulates(acc.precision, r)) =>
                    {
                        Some((m.to_string(), *rate, acc.precision.element_bits(), exact(a.precision, b.precision)))
                    }
                    _ => None,
                })
                .max_by(|x, y| x.3.cmp(&y.3).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2)).then(y.0.cmp(&x.0)))
                .ok_or_else(|| {
                    Diagnostic::error("E-MAP-PREC-001", format!("{} has no MAC mode for {}x{}", hw.nodes[ui.node].path, pa.name(), pb.name()))
                        .at(op.id.clone())
                        .hint("add the precision mode to the unit or cast the operands explicitly")
                })?;
            let rate = mode.1;
            let d = mnk(op, s);
            let m_eff = d.m * density;
            let (cycles, fill, issued, a_mult, b_mult) = match &ui.spec.kind {
                ComputeKind::Matrix(mx) => match &mx.geometry {
                    Geometry::Systolic { rows, cols } => {
                        let (r, c) = (f64::from(*rows), f64::from(*cols));
                        let tiles = d.b * ceil(d.k, r) * ceil(d.n, c);
                        (tiles * m_eff.max(r) / rate, r + c, tiles * m_eff.max(r) * r * c, ceil(d.n, c), 1.0)
                    }
                    Geometry::Mma { m, n, k } => {
                        // An instruction spans 256 bits of k per row (kiln-cost `MacMode::k_pack`).
                        let pack = f64::from((16 / pa.element_bits().max(pb.element_bits()).max(1)).max(1));
                        let (mm, nn, kk) = (f64::from(*m), f64::from(*n), f64::from(*k) * pack);
                        let it = d.b * ceil(m_eff, mm) * ceil(d.n, nn) * ceil(d.k, kk);
                        (it * pack / rate, 0.0, it * mm * nn * kk, ceil(d.n, nn), ceil(d.m, mm))
                    }
                    Geometry::OuterProduct { rows, cols } => {
                        let (r, c) = (f64::from(*rows), f64::from(*cols));
                        let it = d.b * ceil(m_eff, r) * ceil(d.n, c) * d.k;
                        (it / rate, 0.0, it * r * c, ceil(d.n, c), ceil(d.m, r))
                    }
                    Geometry::Spatial { .. } => {
                        let macs = mx.geometry.macs_per_cycle() as f64;
                        let it = (d.b * m_eff * d.n * d.k / macs).ceil();
                        (it / rate, 0.0, it * macs, 1.0, 1.0)
                    }
                },
                k => {
                    let macs = k.ops_per_cycle(&PrecisionMode::Mac {
                        a: pa.into(),
                        b: pb.into(),
                        acc: Precision::Fp32.into(),
                        out: None,
                        rate,
                    });
                    let it = (d.b * m_eff * d.n * d.k / macs.max(1.0)).ceil();
                    (it, 0.0, it * macs, 1.0, 1.0)
                }
            };
            feed_bytes[mr.a] *= a_mult;
            feed_bytes[mr.b] *= b_mult;
            let acc_b = op.kernel.accum.map_or(4.0, |p| f64::from(p.element_bits()) / 8.0);
            let elem_b = |oi: usize| q.prog.tensors[op.operands[oi].tensor].dtype.elem_bits() as f64 / 8.0;
            let (ba, bb) = (elem_b(mr.a), elem_b(mr.b));
            for (li, &cap) in q.level_caps.iter().enumerate() {
                let cap = cap as f64;
                if fp(mr.a) + fp(mr.b) + fp(mr.out) <= cap {
                    continue;
                }
                let t = if li == 0 { (cap / (2.0 * acc_b)).sqrt() } else { cap / (2.0 * 64.0 * (ba + bb)) }.floor().max(1.0);
                reread[li][mr.a] = ceil(d.n, t.min(d.n));
                reread[li][mr.b] = ceil(d.m, t.min(d.m));
            }
            let useful = points as u128 * u128::from(body.mac);
            return Ok(NestCost {
                cycles: (cycles / f64::from(q.gang.max(1))).ceil(),
                fill_cycles: fill + ui.spec.pipeline.fill.map_or(0.0, |c| c.0),
                useful_macs: useful,
                issued_macs: (issued.round() as u128).max(useful),
                vec_ops: 0,
                cvt_ops: 0,
                transc_ops: 0,
                readback: vec![0.0; nops],
                spill: vec![],
                feed_bytes,
                reread,
                mode: mode.0,
                bits: (pa.element_bits(), pb.element_bits()),
                special_cycles: 0.0,
            });
        }
        let lanes = ui.spec.kind.base_ops_per_cycle() as f64;
        let (mdt, rate) = vector_mode(hw, q.unit, &vector_dtypes(q.prog, op)).map_err(|d| d.at(op.id.clone()))?;
        let kr = |c: OpClass| ui.spec.kind.class_rate(c);
        // A ganged slice runs on every member (the SM's four ALUs), as a contraction's does.
        let gang = f64::from(q.gang.max(1));
        let per = lanes * rate * gang;
        let transc = points * f64::from(body.transcendental());
        let specials = if body.transcendental() > 0 { special_units(hw, q.unit) } else { vec![] };
        let (n_sup, sfu_pt, muls) = if specials.is_empty() { (0.0, 0.0, 0.0) } else { special_terms(hw, &specials, body, mdt, gang) };
        let moves = match op.class() {
            KernelClass::Gather | KernelClass::Scatter => points / kr(OpClass::GatherScatter),
            KernelClass::Layout => points / kr(OpClass::Permute),
            _ => 0.0,
        };
        let vec = points * f64::from(body.vector());
        let cvt = points * f64::from(body.cvt);
        let base = (vec / kr(OpClass::Elementwise) + (transc - points * n_sup) / kr(OpClass::Transcendental) + cvt / kr(OpClass::Convert) + moves) / per;
        // Transcendentals the special units implement go to them in the share that balances both units (the
        // compiler may leave some on a wide vector unit: v6e's bf16 VPU emulates exp as fast as its EUP).
        let on_vec = points * n_sup / kr(OpClass::Transcendental) / per;
        let mul = points * muls / kr(OpClass::Elementwise) / per;
        let sfu = points * sfu_pt;
        let ratio = if specials.is_empty() { 1.0 } else { nominal_hz(hw, q.unit) / nominal_hz(hw, specials[0]) };
        let share = if on_vec > mul && sfu > 0.0 { ((base + on_vec) / (sfu * ratio + on_vec - mul)).min(1.0) } else { 0.0 };
        let cycles = base + (1.0 - share) * on_vec + share * mul;
        let vec = vec + share * points * muls;
        Ok(NestCost {
            cycles: cycles.ceil(),
            fill_cycles: ui.spec.pipeline.fill.map_or(0.0, |c| c.0),
            useful_macs: 0,
            issued_macs: 0,
            vec_ops: (vec + cvt) as u128,
            cvt_ops: cvt as u128,
            transc_ops: transc as u128,
            readback: vec![0.0; nops],
            spill: vec![],
            feed_bytes,
            reread,
            mode: format!("{}@{rate}", mdt.name()),
            bits: (mdt.element_bits(), mdt.element_bits()),
            special_cycles: (share * sfu).ceil(),
        })
    }
}

/// Template level the unit reads (or writes) operands of `role` through its feed: the first level of its chain
/// past unit-local buffers.
fn feed_level(hw: &HwModel, unit: &kiln_cost::UnitTemplate, role: Option<kiln_ir::hw::compute::OperandRole>) -> Option<usize> {
    let r = role?;
    let c = unit.chains.iter().find(|c| c.role == r)?;
    c.levels.iter().copied().find(|&l| unit.levels[l].mem.is_some_and(|m| !hw.memories[m].is_local()))
}

/// A unit template shared by every unit of one entity at given private-level capacities.
#[derive(Debug)]
struct Tpl {
    unit: kiln_cost::UnitTemplate,
    hash: String,
    /// Entity of each level's memory (matches query levels to template levels across sibling units).
    level_entity: Vec<Option<String>>,
}

/// Design, unit entity, gang, private capacities, shared read bandwidths.
type TplKey = (String, String, u32, Vec<u64>, Vec<(usize, u64)>);

/// kiln-cost errors meaning it cannot represent the kernel or the unit (not that the slice does not fit or has
/// no precision mode): only these fall back to [`RooflineCost`].
const UNSUPPORTED: [&str; 5] = ["E-COST-NEST", "E-COST-UNIT", "E-COST-CLOCK", "E-COST-GANG", "E-COST-CHAIN"];

/// The mapper needs latency: the search stops once a mapping reaches the nest's latency floor. Candidate
/// scoring keeps the two best-ranked spatial unrollings and a small loop-order budget; committed ops get the
/// full search.
const SEARCH: kiln_cost::CostOptions = kiln_cost::CostOptions {
    ragged: kiln_cost::RaggedPolicy::Split,
    budget: kiln_cost::SearchBudget { top_k_spatial: 8, max_evals_per_spatial: 20_000, stop_at_floor: true },
    zigzag_compat: false,
};
const QUICK: kiln_cost::CostOptions = kiln_cost::CostOptions {
    budget: kiln_cost::SearchBudget { top_k_spatial: 2, max_evals_per_spatial: 300, stop_at_floor: true },
    ..SEARCH
};

/// `kiln-cost` (03 §2 loop-nest search) behind [`UnitCostModel`]. One template per design, unit entity and private
/// capacity share (units of an entity are interchangeable, as kiln-map's own cost cache assumes); entries are
/// memoized by exact tile shape. Masked tiles are costed as their bounding box with work, cycles and feed
/// traffic scaled by the live-point density (fully masked blocks skipped, as fused attention kernels do).
/// Kernels or units kiln-cost cannot express fall back to [`RooflineCost`] (counted); infeasible tiles, residency
/// and precision errors propagate.
#[derive(Debug, Default)]
pub struct KilnCost {
    templates: std::sync::Mutex<std::collections::BTreeMap<TplKey, Result<std::sync::Arc<Tpl>, Diagnostic>>>,
    cache: kiln_cost::CostCache,
    pub fallbacks: std::sync::atomic::AtomicU64,
}

impl KilnCost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers from `cache` (e.g. a file-backed [`kiln_cost::CostCache::open`] shared across an evolution run).
    pub fn with_cache(cache: kiln_cost::CostCache) -> Self {
        Self { cache, ..Self::default() }
    }

    /// Answers from a cache backed by the file at `path` ([`kiln_cost::CostCache::open`]; at most `max_bytes`,
    /// e.g. [`kiln_cost::CostCache::DEFAULT_FILE_BYTES`], written back by [`KilnCost::save_cache`]).
    pub fn with_cache_file(path: impl Into<std::path::PathBuf>, max_bytes: u64) -> Self {
        Self::with_cache(kiln_cost::CostCache::open(path, max_bytes))
    }

    /// Writes the cache back to its file, if it has one.
    pub fn save_cache(&self) -> std::io::Result<()> {
        self.cache.save()
    }

    pub fn cache_stats(&self) -> kiln_cost::CacheStats {
        self.cache.stats()
    }

    fn template(&self, q: &NestQuery) -> Result<std::sync::Arc<Tpl>, Diagnostic> {
        let hw = q.hw;
        let entity = |m: usize| hw.nodes[hw.memories[m].node].entity.clone();
        let key = (
            hw.design_hash.clone(),
            hw.nodes[hw.units[q.unit].node].entity.clone(),
            q.gang,
            q.level_caps.to_vec(),
            q.level_bw.iter().map(|&(m, b)| (m, b.to_bits())).collect(),
        );
        let mut m = self.templates.lock().expect("template cache");
        m.entry(key)
            .or_insert_with(|| {
                let opts = kiln_cost::TemplateOptions { gang: q.gang.max(1), ..Default::default() };
                let mut unit = kiln_cost::UnitTemplate::from_hw(hw, q.unit, &opts)?;
                let level_entity: Vec<Option<String>> = unit.levels.iter().map(|l| l.mem.map(entity)).collect();
                let axes: Vec<u64> = unit.axes.iter().map(|a| u64::from(a.size)).collect();
                for (&mem, &cap) in q.level_mems.iter().zip(q.level_caps) {
                    let e = Some(entity(mem));
                    for (l, le) in unit.levels.iter_mut().zip(&level_entity) {
                        if *le == e {
                            let copies: u64 = l.instance_axes.iter().map(|&a| axes[a]).product();
                            l.capacity_bytes = l.capacity_bytes.min(cap * copies);
                        }
                    }
                }
                for &(mem, bw) in q.level_bw {
                    let e = Some(entity(mem));
                    let limit = bw / unit.clock_hz;
                    for (l, le) in unit.levels.iter_mut().zip(&level_entity) {
                        if *le != e {
                            continue;
                        }
                        let reads = |p: &kiln_cost::MemPort| p.dir != kiln_cost::PortDir::Write;
                        let total: f64 = l.ports.iter().filter(|p| reads(p)).map(|p| p.bytes_per_cycle).sum();
                        if total > limit {
                            l.ports.iter_mut().filter(|p| reads(p)).for_each(|p| p.bytes_per_cycle *= limit / total);
                        }
                    }
                }
                let hash = unit.hash();
                Ok(std::sync::Arc::new(Tpl { unit, hash, level_entity }))
            })
            .clone()
    }

    fn query(&self, q: &NestQuery) -> Result<NestCost, Diagnostic> {
        let (op, s) = (q.op, q.slice);
        let roles: Option<Vec<kiln_ir::hw::compute::OperandRole>> = op.mac.as_ref().map(|m| {
            use kiln_ir::hw::compute::OperandRole as R;
            (0..op.operands.len()).map(|i| if i == m.a { R::A } else if i == m.b { R::B } else if i == m.out { R::O } else { R::In }).collect()
        });
        let (t, tile) = self.tile(q)?;
        let e = self.cache.query_hashed(
            &kiln_cost::CostQuery { unit: &t.unit, nest: &tile, objective: kiln_cost::Objective::Latency, options: if q.quick { QUICK } else { SEARCH } },
            &t.hash,
        )?;
        let (points, boxp) = (q.points, s.box_points().max(1));
        let density = points as f64 / boxp as f64;
        let scale = |x: u64| (u128::from(x) * points).div_ceil(boxp);
        let nops = op.operands.len();
        let level_of = |mem: usize| {
            let e = Some(q.hw.nodes[q.hw.memories[mem].node].entity.clone());
            t.level_entity.iter().position(|x| *x == e)
        };
        let fp: Vec<f64> = (0..nops)
            .map(|oi| {
                let tt = &q.prog.tensors[op.operands[oi].tensor];
                bytes_of(q.prog, op, oi, crate::geom::footprint(op, oi, s, &tt.shape).elems())
            })
            .collect();
        let mut feed_bytes = fp.clone();
        let mut readback = vec![0.0; nops];
        let mut spill = vec![];
        let mut reread = vec![vec![1.0; nops]; q.level_mems.len()];
        let first = q.level_mems.first().and_then(|&m| level_of(m));
        // Accesses index streams; an input's MX/block scale stream counts toward that input.
        let owner = tile.stream_operands();
        let of = |a: &kiln_cost::LevelAccess, l: usize, oi: usize| a.level == l && owner.get(a.operand) == Some(&oi);
        let dir = |l: usize, oi: usize, d: usize| e.accesses.iter().filter(|a| of(a, l, oi)).map(|a| a.dir_bytes[d]).sum::<u64>() as f64 * density;
        let (to_low, from_low) = (0, 3);
        for oi in 0..nops {
            let role = roles.as_deref().map(|r| r[oi]);
            let Some(l0) = feed_level(q.hw, &t.unit, role).or(first) else { continue };
            if !op.operands[oi].access.writes() {
                let b: u64 = e.accesses.iter().filter(|a| of(a, l0, oi)).map(|a| a.read_bytes).sum();
                if b > 0 {
                    feed_bytes[oi] = b as f64 * density;
                }
                continue;
            }
            if dir(l0, oi, from_low) > 0.0 {
                feed_bytes[oi] = dir(l0, oi, from_low);
            }
            readback[oi] = dir(l0, oi, to_low);
            let chain = role.and_then(|r| t.unit.chains.iter().find(|c| c.role == r)).map_or(&[][..], |c| &c.levels[..]);
            for &l in chain.iter().skip_while(|&&l| l != l0).skip(1) {
                let (up, down) = ((dir(l, oi, from_low) - fp[oi]).max(0.0), dir(l, oi, to_low));
                if let (Some(ent), true) = (&t.level_entity[l], up + down > 0.0) {
                    spill.push((ent.clone(), up, down));
                }
            }
        }
        for (li, &m) in q.level_mems.iter().enumerate() {
            let Some(l) = level_of(m) else { continue };
            for oi in (0..nops).filter(|&oi| !op.operands[oi].access.writes()) {
                let w: u64 = e.accesses.iter().filter(|a| of(a, l, oi)).map(|a| a.write_bytes).sum();
                if w > 0 && fp[oi] > 0.0 {
                    reread[li][oi] = (w as f64 * density / fp[oi]).max(1.0);
                }
            }
        }
        let mode = t.unit.modes.get(e.mode);
        Ok(NestCost {
            cycles: (e.cycles as f64 * density).ceil(),
            fill_cycles: 0.0,
            useful_macs: scale(e.useful_macs),
            issued_macs: scale(e.issued_macs.max(e.useful_macs)),
            vec_ops: scale(e.vector_ops + e.conversion_ops),
            cvt_ops: scale(e.conversion_ops),
            transc_ops: 0,
            feed_bytes,
            reread,
            readback,
            spill,
            mode: mode.map_or_else(String::new, |m| format!("{}*{}+{}", m.a, m.b, m.acc)),
            bits: mode.map_or((16, 16), |m| (m.a.precision.element_bits(), m.b.precision.element_bits())),
            special_cycles: 0.0,
        })
    }
}

impl KilnCost {
    /// Lower bound of [`UnitCostModel::cost`]: per field, the smaller of the roofline fallback's value and a
    /// bound on kiln-cost's (its latency floor for cycles, compulsory traffic for feed bytes and re-reads, no
    /// fill), so it holds whichever of the two answers.
    fn floor(&self, q: &NestQuery) -> Result<NestCost, Diagnostic> {
        let roof = RooflineCost.cost(q)?;
        let nops = q.op.operands.len();
        let fp = self.footprints(q);
        let density = q.points as f64 / q.slice.box_points().max(1) as f64;
        let mut c = NestCost { fill_cycles: 0.0, feed_bytes: vec![0.0; nops], reread: vec![vec![1.0; nops]; q.level_mems.len()], ..roof.clone() };
        if let Ok((cycles, acc, l0, levels)) = self.floor_parts(q) {
            c.cycles = c.cycles.min((cycles as f64 * density).ceil());
            for (oi, fb) in c.feed_bytes.iter_mut().enumerate() {
                *fb = l0[oi].map_or(fp[oi], |l| {
                    let (r, w) = acc[oi][l];
                    fp[oi].min(if q.op.operands[oi].access.writes() { w } else { r } as f64 * density)
                });
            }
            for (li, l) in levels.iter().enumerate() {
                let Some(l) = *l else { continue };
                for oi in (0..nops).filter(|&oi| !q.op.operands[oi].access.writes()) {
                    let w = acc[oi][l].1;
                    if w > 0 && fp[oi] > 0.0 {
                        c.reread[li][oi] = (w as f64 * density / fp[oi]).max(1.0);
                    }
                }
            }
        }
        for (x, r) in c.feed_bytes.iter_mut().zip(&roof.feed_bytes) {
            *x = x.min(*r);
        }
        for (row, rr) in c.reread.iter_mut().zip(&roof.reread) {
            for (x, r) in row.iter_mut().zip(rr) {
                *x = x.min(*r);
            }
        }
        Ok(c)
    }

    fn footprints(&self, q: &NestQuery) -> Vec<f64> {
        (0..q.op.operands.len())
            .map(|oi| {
                let tt = &q.prog.tensors[q.op.operands[oi].tensor];
                bytes_of(q.prog, q.op, oi, crate::geom::footprint(q.op, oi, q.slice, &tt.shape).elems())
            })
            .collect()
    }

    /// kiln-cost latency floor, access floors, template level of the feed memory and of each private level.
    #[allow(clippy::type_complexity)]
    fn floor_parts(&self, q: &NestQuery) -> Result<(u64, Vec<Vec<(u64, u64)>>, Vec<Option<usize>>, Vec<Option<usize>>), Diagnostic> {
        let (t, tile) = self.tile(q)?;
        let level_of = |mem: usize| {
            let e = Some(q.hw.nodes[q.hw.memories[mem].node].entity.clone());
            t.level_entity.iter().position(|x| *x == e)
        };
        let first = q.level_mems.first().and_then(|&m| level_of(m));
        let l0 = tile.operands.iter().map(|o| feed_level(q.hw, &t.unit, Some(o.role)).or(first)).collect();
        let levels = q.level_mems.iter().map(|&m| level_of(m)).collect();
        Ok((kiln_cost::latency_floor(&t.unit, &tile)?, kiln_cost::access_floors(&t.unit, &tile)?, l0, levels))
    }

    fn tile(&self, q: &NestQuery) -> Result<(std::sync::Arc<Tpl>, kiln_cost::OpNest), Diagnostic> {
        use kiln_ir::precision::PrecisionSpec;
        let (op, s) = (q.op, q.slice);
        let seg = &op.segs[s.seg as usize];
        let mut k = op.kernel.clone();
        for (d, dim) in k.dims.iter_mut().enumerate() {
            dim.extent = seg.ext[d];
        }
        k.domain = kiln_ir::wl::Domain::Box;
        let dtypes: Vec<PrecisionSpec> = op.operands.iter().map(|o| kiln_wl::convert::operand_spec(&q.prog.tensors[o.tensor].dtype)).collect();
        let roles: Option<Vec<kiln_ir::hw::compute::OperandRole>> = op.mac.as_ref().map(|m| {
            use kiln_ir::hw::compute::OperandRole as R;
            (0..op.operands.len()).map(|i| if i == m.a { R::A } else if i == m.b { R::B } else if i == m.out { R::O } else { R::In }).collect()
        });
        let mut nest = kiln_cost::OpNest::from_kernel(&k, &dtypes, roles.as_deref())?;
        let t = self.template(q)?;
        for (o, r) in nest.operands.iter_mut().zip(q.residency) {
            let Some(m) = *r else { continue };
            let e = Some(q.hw.nodes[q.hw.memories[m].node].entity.clone());
            let level = t.unit.chains.iter().find(|c| c.role == o.role).and_then(|c| c.levels.iter().copied().find(|&l| t.level_entity[l] == e));
            if o.is_output {
                o.sink = level;
            } else {
                o.source = level;
            }
        }
        let sizes: Vec<u64> = (0..s.lo.len()).map(|d| s.extent(d)).collect();
        Ok((t, nest.positioned(&s.lo).tile(&sizes, None)))
    }
}

impl UnitCostModel for KilnCost {
    fn name(&self) -> &str {
        "kiln-cost"
    }

    fn cost_floor(&self, q: &NestQuery) -> Option<Result<NestCost, Diagnostic>> {
        Some(if q.op.class() != KernelClass::Contraction { RooflineCost.cost(q) } else { self.floor(q) })
    }

    fn cost(&self, q: &NestQuery) -> Result<NestCost, Diagnostic> {
        if q.op.class() != KernelClass::Contraction {
            return RooflineCost.cost(q);
        }
        self.query(q).or_else(|e| {
            if !UNSUPPORTED.contains(&e.code.as_str()) {
                return Err(e);
            }
            self.fallbacks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            RooflineCost.cost(q)
        })
    }
}
