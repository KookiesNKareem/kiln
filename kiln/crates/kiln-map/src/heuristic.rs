//! Default deterministic mapper (03 §3.2-3.7): tensor homes, per-op split candidates scored by a local
//! Tier A estimate, one execution group per workload node (or per contraction with its fused elementwise
//! neighbours, when the software stack fuses them) with a barrier after it.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::{ComputeKind, Geometry};
use kiln_ir::hw::types::ExecModel;
use kiln_ir::wl::{DimKind, KernelClass};

use crate::cost::UnitCostModel;
use crate::geom::{TBox, aligned_parts, even_parts};
use crate::hwview::{GroupIx, HwView, Pool};
use crate::lower::Lowerer;
use crate::mapping::{
    ExecGroup, GroupKind, Home, LaunchKind, Lifetime, MAPPING_VERSION, Mapping, OpPlacement, RoutingPolicy, SplitAxis,
    Target, TensorPlacement, TensorRegion, UnitSet,
};
use crate::program::{DimRole, POp, Program, ProgramKind};

#[derive(Clone, Debug, PartialEq)]
pub struct MapOptions {
    pub seed: u64,
    pub max_candidates: usize,
    /// Overrides the design's `exec_model` (launch kind of every group).
    pub exec_model: Option<ExecModel>,
    pub workload_hash: String,
    /// Activations whose footprint is at most this fraction of the shared on-chip level stay on chip.
    pub onchip_activation_fraction: f64,
    /// On unit sets larger than this, candidates are pre-screened by compute balance and only the best
    /// `prescreen_keep` (plus the inherited split) are lowered for scoring.
    pub prescreen_units: usize,
    pub prescreen_keep: usize,
    /// Skip split candidates whose estimate from [`UnitCostModel::cost_floor`] costs cannot beat the best so
    /// far (same choice, fewer cost queries); `false` scores every candidate.
    pub bound_candidates: bool,
    /// The software stack fuses elementwise nodes into the adjacent contraction's group (whole steps only;
    /// [`kiln_wl::stack::Stack::fuse_elementwise`]).
    pub fuse_elementwise: bool,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            max_candidates: 8,
            exec_model: None,
            workload_hash: String::new(),
            onchip_activation_fraction: 0.25,
            prescreen_units: 64,
            prescreen_keep: 5,
            bound_candidates: true,
            fuse_elementwise: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MapReport {
    pub warnings: Vec<Diagnostic>,
    /// Model state that does not fit its home: `(group, needed, capacity)`.
    pub capacity_overflow: Option<(String, u128, u64)>,
    pub candidates_scored: u64,
}

pub fn launch_kind(m: ExecModel) -> LaunchKind {
    match m {
        ExecModel::HostLaunched => LaunchKind::HostLaunch,
        ExecModel::DeviceQueued => LaunchKind::DeviceQueued,
        ExecModel::StaticDataflow => LaunchKind::StaticProgram,
    }
}

fn whole(shape: &[u64]) -> TensorRegion {
    TensorRegion { lo: vec![0; shape.len()], hi: shape.to_vec() }
}

/// Homes (03 §3.4): model state and graph I/O off chip; fused temps private; other activations on the
/// shared on-chip level when small enough and while the activations live there at once fit it (next to the model
/// state on a design without off-chip memory), else off chip; isolated programs keep everything off chip.
pub fn place_tensors(prog: &Program, view: &HwView, opts: &MapOptions, report: &mut MapReport) -> Result<IndexMap<String, TensorPlacement>, Diagnostic> {
    let top = view.offchip.or_else(|| view.shared_onchip()).ok_or_else(|| Diagnostic::error("E-MAP-CAP-002", "design has no memory every unit can reach"))?;
    let shared = view.shared_onchip();
    let mut out = IndexMap::new();
    let mut resident: u128 = 0;
    let ranges = prog.live_ranges();
    let mut live = vec![0u128; prog.ops.len()];
    let state: u128 = prog.tensors.iter().filter(|t| t.alias_of.is_none() && t.model_state()).map(|t| t.footprint()).sum();
    let free = shared.map_or(0, |s| u128::from(view.groups[s].capacity).saturating_sub(if s == top { prog.resident_bytes.map_or(state, |r| r.max(state)) } else { 0 }));
    for (i, t) in prog.tensors.iter().enumerate() {
        if t.alias_of.is_some() {
            continue;
        }
        let home = |g: GroupIx, lifetime: Lifetime| TensorPlacement {
            home: vec![Home { region: whole(&t.shape), mems: view.group_paths(g) }],
            interleave_granule_b: None,
            lifetime,
        };
        let p = if t.temp && prog.kind == ProgramKind::WholeStep {
            TensorPlacement { home: vec![], interleave_granule_b: None, lifetime: Lifetime::Private }
        } else if t.model_state() {
            resident += t.footprint();
            home(top, Lifetime::Resident)
        } else if prog.kind == ProgramKind::Isolated || !matches!(t.class, kiln_ir::wl::TensorClass::Activation) {
            home(top, Lifetime::Streamed)
        } else {
            let fp = t.footprint();
            let r = ranges[i].map_or(0..0, |(a, b)| a..b + 1);
            match shared {
                Some(s) if (fp as f64) <= view.groups[s].capacity as f64 * opts.onchip_activation_fraction && live[r.clone()].iter().all(|&x| x + fp <= free) => {
                    live[r].iter_mut().for_each(|x| *x += fp);
                    home(s, Lifetime::Streamed)
                }
                _ => home(top, Lifetime::Spilled { to: view.group_paths(top) }),
            }
        };
        out.insert(t.id.clone(), p);
    }
    let cap = view.groups[top].capacity;
    let resident = prog.resident_bytes.map_or(resident, |r| r.max(resident));
    if resident > u128::from(cap) {
        let g = view.groups[top].name.clone();
        report.warnings.push(
            Diagnostic::warning("W-MAP-CAP-001", format!("resident model state {resident} B exceeds {g} capacity {cap} B"))
                .at(g.clone())
                .hint("the whole step does not fit; it is scored at scope `layer` (02 §12.5)"),
        );
        report.capacity_overflow = Some((g, resident, cap));
    }
    Ok(out)
}

fn granule(view: &HwView, units: &[usize], role: DimRole) -> u64 {
    let Some(&u) = units.first() else { return 1 };
    match &view.hw.units[view.units[u].unit].spec.kind {
        ComputeKind::Matrix(m) => match (&m.geometry, role) {
            (Geometry::Systolic { cols, .. }, DimRole::N) => u64::from(*cols),
            (Geometry::Mma { n, .. }, DimRole::N) => u64::from(*n),
            (Geometry::Mma { m, .. }, DimRole::M) => u64::from(*m),
            (Geometry::OuterProduct { rows, .. }, DimRole::M) => u64::from(*rows),
            (Geometry::OuterProduct { cols, .. }, DimRole::N) => u64::from(*cols),
            _ => 1,
        },
        _ => 1,
    }
}

/// Split candidates (03 §3.3): greedy fills of up to `n` parts over splittable dims in several orders.
pub fn candidates(op: &POp, view: &HwView, units: &[usize], max: usize) -> Vec<Vec<SplitAxis>> {
    let n = units.len() as u64;
    let k = &op.kernel;
    let outs = op.output_dims();
    let uniform = |d: usize| op.segs.iter().all(|s| s.ext[d] == op.segs[0].ext[d]);
    let ext = |d: usize| op.segs[0].ext[d];
    let role = |d: usize| op.mac.as_ref().map_or(DimRole::Other, |m| m.dims[d]);
    let par: Vec<usize> = (0..k.dims.len()).filter(|&d| outs[d] && k.dims[d].kind == DimKind::Parallel && uniform(d) && ext(d) > 1).collect();
    let red: Vec<usize> = (0..k.dims.len()).filter(|&d| !outs[d] && uniform(d) && ext(d) > 1 && k.class != KernelClass::Gather).collect();
    let mut orders: Vec<Vec<usize>> = vec![par.clone()];
    let mut by_ext = par.clone();
    by_ext.sort_by_key(|&d| (std::cmp::Reverse(ext(d)), d));
    orders.push(by_ext.clone());
    for &d in &par {
        let mut o = vec![d];
        o.extend(par.iter().copied().filter(|&x| x != d));
        orders.push(o);
    }
    let par_total: u64 = par.iter().map(|&d| ext(d)).product::<u64>().max(1);
    if par_total < n / 2 {
        for &r in &red {
            let mut o = by_ext.clone();
            o.push(r);
            orders.push(o);
        }
    }
    let mut out: Vec<Vec<SplitAxis>> = vec![];
    let push = |s: Vec<SplitAxis>, out: &mut Vec<Vec<SplitAxis>>| {
        if !out.contains(&s) && out.len() < max {
            out.push(s);
        }
    };
    let fill = |order: &[usize], first: Option<u64>| -> Vec<SplitAxis> {
        let mut left = n;
        let mut v = vec![];
        for (i, &d) in order.iter().enumerate() {
            if left <= 1 {
                break;
            }
            let want = if i == 0 { first.unwrap_or(left) } else { left };
            let p = want.min(ext(d)).min(left);
            if p <= 1 {
                continue;
            }
            left /= p;
            v.push(SplitAxis { dim: k.dims[d].name.clone(), parts: aligned_parts(ext(d), p, granule(view, units, role(d))) });
        }
        v
    };
    // Traffic-balanced 2-D splits first: for parallel dims a, b with pa * pb = n, operand reloads scale as
    // ext_a * pb + ext_b * pa, minimized near pa = sqrt(n * ext_a / ext_b).
    for (i, &a) in par.iter().enumerate() {
        for &b in &par[i + 1..] {
            let want = (n as f64 * ext(a) as f64 / ext(b) as f64).sqrt();
            let best = (1..=n).filter(|p| n.is_multiple_of(*p) && *p <= ext(a) && n / p <= ext(b)).min_by(|x, y| {
                let d = |p: u64| ((p as f64).ln() - want.ln()).abs();
                d(*x).total_cmp(&d(*y)).then(x.cmp(y))
            });
            if let Some(pa) = best.filter(|&p| p > 1 && p < n) {
                push(fill(&[a, b], Some(pa)), &mut out);
            }
        }
    }
    for o in &orders {
        push(fill(o, None), &mut out);
    }
    for (i, &a) in par.iter().enumerate() {
        for &b in &par[i + 1..] {
            for p1 in std::iter::successors(Some(2u64), |x| (x * 2 <= n).then_some(x * 2)) {
                if p1 <= ext(a) {
                    push(fill(&[a, b], Some(p1)), &mut out);
                    push(fill(&[b, a], Some(p1.min(ext(b)))), &mut out);
                }
            }
        }
    }
    if out.is_empty() {
        out.push(vec![]);
    }
    for c in &mut out {
        for ax in c.iter_mut() {
            let d = op.dim_ix(&ax.dim).expect("dim");
            if ax.parts.iter().sum::<u64>() != ext(d) {
                ax.parts = even_parts(ext(d), ax.parts.len() as u64);
            }
        }
    }
    out
}

/// Registry precisions of a contraction's `a` and `b` operands (`mxfp4`, `fp8_e4m3_pt`, ... for scaled types).
pub fn mac_precisions(prog: &Program, op: &POp) -> (kiln_ir::precision::Precision, kiln_ir::precision::Precision) {
    let m = op.mac.as_ref().expect("contraction roles");
    let p = |i: usize| kiln_wl::convert::operand_spec(&prog.tensors[op.operands[i].tensor].dtype).precision;
    (p(m.a), p(m.b))
}

/// The unit has a MAC mode for these operand precisions: the same names, or the element type of MX/scaled
/// operands (03 §2.7: never silently upcast; kiln-wl inserts explicit converts otherwise).
fn supports(view: &HwView, u: usize, a: kiln_ir::precision::Precision, b: kiln_ir::precision::Precision) -> bool {
    use kiln_wl::convert::accepts;
    crate::cost::require_matmul(&view.hw, view.units[u].unit).is_ok()
        && view.hw.units[view.units[u].unit].spec.precisions.iter().any(|m| {
        matches!(m, kiln_ir::hw::compute::PrecisionMode::Mac { a: x, b: y, .. } if accepts(x.precision, a) && accepts(y.precision, b))
    })
}

/// Row-major slices assigned round-robin over the unit set (adjacent slices share private memories); fewer
/// slices than units spread evenly over the set, so they span its dies, partitions and memory ports instead
/// of filling its first units (64 slices on ember's 254 tiles reach all four chiplets).
pub fn placement(op: &POp, set: u32, split: Vec<SplitAxis>, n_units: usize) -> OpPlacement {
    let per_seg: usize = split.iter().map(|s| s.parts.len()).product();
    let total = per_seg * op.segs.len();
    let unit = |i: usize| if total < n_units { i * n_units / total } else { i % n_units };
    OpPlacement { target: Target::Units { set }, split, slice_to_unit: (0..total).map(|i| unit(i) as u32).collect() }
}

/// Shape signature: kernel structure with tensor names replaced by operand position, plus each operand's
/// shape, dtype and placement, plus the inherited split. Equal signatures get equal splits.
fn op_signature(prog: &Program, op: &POp, tensors: &IndexMap<String, TensorPlacement>, prev: &Option<Vec<SplitAxis>>) -> String {
    let mut k = op.kernel.clone();
    k.id.clear();
    for (i, o) in k.operands.iter_mut().enumerate() {
        o.tensor = kiln_ir::common::Id::new(format!("t{i}")).expect("id");
    }
    let mut s = serde_json::to_string(&k).unwrap_or_default();
    for o in &op.operands {
        let t = &prog.tensors[o.tensor];
        let root = &prog.tensors[prog.root(o.tensor)];
        let place = tensors.get(&root.id).map(|p| (format!("{:?}", p.lifetime), p.home.first().and_then(|h| h.mems.first().cloned())));
        s.push_str(&format!("|{:?}{:?}{:?}{place:?}", t.shape, t.dtype, t.class));
    }
    s.push_str(&format!("|{prev:?}"));
    s
}

fn unit_peak(view: &HwView, u: usize) -> f64 {
    let ui = &view.units[u];
    let spec = &view.hw.units[ui.unit].spec;
    spec.precisions.iter().map(|m| spec.kind.ops_per_cycle(m)).fold(0.0, f64::max) * view.clock_hz(ui.clock)
}

/// Units by capability (peak ops/s, then the capacity of each chain level), so slice assignment over a mixed
/// pool does not depend on declaration order or names; equal keys keep expansion order.
fn canonical_order(view: &HwView, mut units: Vec<usize>) -> Vec<usize> {
    let key = |u: usize| {
        let ui = &view.units[u];
        (std::cmp::Reverse(unit_peak(view, u).to_bits()), ui.chain.iter().map(|&g| std::cmp::Reverse(view.groups[g].capacity)).collect::<Vec<_>>())
    };
    units.sort_by_cached_key(|&u| key(u));
    units
}

/// The fastest `k` units of a canonically ordered pool, `k` maximizing `k * peak(k-th)`: slices are equal and
/// assigned round-robin, so an op over a mixed pool runs at the pace of its slowest unit (on TPU v6e, one
/// SparseCore tile next to the TensorCore VPU makes every vector op about twice as slow as the VPU alone).
/// Homogeneous pools are kept whole.
fn balanced_prefix(view: &HwView, units: Vec<usize>) -> Vec<usize> {
    let rate = |u: usize| unit_peak(view, u) * view.units[u].members.len().max(1) as f64;
    let mut best = (0usize, 0.0f64);
    for (i, &u) in units.iter().enumerate() {
        let r = (i + 1) as f64 * rate(u);
        if r > best.1 * (1.0 + 1e-12) {
            best = (i + 1, r);
        }
    }
    let mut units = units;
    if best.1 > 0.0 {
        units.truncate(best.0);
    }
    units
}

/// Elementwise node: no op is a contraction, collective or opaque kernel (norms, RoPE with its table
/// gather, activations, residual adds, in-place cache appends); the others anchor their own group.
fn elementwise(prog: &Program, n: usize) -> bool {
    let ops = &prog.ops[prog.nodes[n].ops.clone()];
    !ops.is_empty() && ops.iter().all(|o| !matches!(o.class(), KernelClass::Contraction | KernelClass::Collective | KernelClass::Opaque))
}

/// Nodes per execution group, in program order: one node per group, or with `fuse` (03 §3.6) each
/// elementwise node joins the group before it (output fusion), and one that opens an iteration (or the
/// program) joins the group after it (operand fusion). Groups never span iterations.
fn node_groups(prog: &Program, fuse: bool) -> Vec<Vec<usize>> {
    if !fuse {
        return (0..prog.nodes.len()).map(|n| vec![n]).collect();
    }
    let mut out: Vec<Vec<usize>> = vec![];
    // The last group holds only elementwise nodes that wait for the next anchor.
    let mut open_prologue = false;
    for n in 0..prog.nodes.len() {
        let it = prog.nodes[n].iteration;
        let same = out.last().and_then(|g| g.first()).is_some_and(|&f| prog.nodes[f].iteration == it);
        let ew = elementwise(prog, n);
        match out.last_mut() {
            Some(g) if same && (ew || open_prologue) => {
                g.push(n);
                open_prologue &= ew;
            }
            _ => {
                out.push(vec![n]);
                open_prologue = ew;
            }
        }
    }
    out
}

/// Errors that rule out one split candidate (its tiles do not fit, or it needs a unit or mode the design lacks) rather
/// than the op: scoring skips the candidate.
const UNMAPPABLE_CANDIDATE: [&str; 3] = ["E-COST-INFEASIBLE", "E-MAP-OP-004", "E-MAP-PREC-002"];

/// The default heuristic mapping and its report.
pub fn heuristic(prog: &Program, view: &HwView, cost: &dyn UnitCostModel, opts: &MapOptions) -> Result<(Mapping, MapReport), Diagnostic> {
    heuristic_lowered(prog, view, cost, opts).map(|(m, r, _)| (m, r))
}

/// As [`heuristic`], also returning the task graph built while mapping (identical to `lower(mapping)`).
pub fn heuristic_lowered(
    prog: &Program,
    view: &HwView,
    cost: &dyn UnitCostModel,
    opts: &MapOptions,
) -> Result<(Mapping, MapReport, crate::lower::TaskGraph), Diagnostic> {
    let mut report = MapReport::default();
    let path = |i: usize| view.units[i].path.clone();
    let reach = |u: &usize| view.offchip.or_else(|| view.shared_onchip()).is_none_or(|g| view.units[*u].chain.contains(&g));
    let mac: Vec<usize> = canonical_order(view, view.pool(Pool::Mac).into_iter().filter(reach).collect());
    let all_vec: Vec<usize> = canonical_order(view, view.pool(Pool::Vector).into_iter().filter(reach).collect());
    let vec = balanced_prefix(view, all_vec.clone());
    let unit_sets = vec![UnitSet { name: "vector".into(), units: vec.iter().map(|&i| path(i)).collect() }];
    let tensors = place_tensors(prog, view, opts, &mut report)?;
    let exec = opts.exec_model.unwrap_or(view.hw.exec_model);
    let groups: Vec<ExecGroup> = node_groups(prog, opts.fuse_elementwise && prog.kind == ProgramKind::WholeStep)
        .into_iter()
        .map(|nodes| {
            let ops: Vec<String> = nodes.iter().flat_map(|&n| prog.ops[prog.nodes[n].ops.clone()].iter().map(|o| o.id.clone())).collect();
            let temps: Vec<String> = nodes
                .iter()
                .flat_map(|&n| prog.ops[prog.nodes[n].ops.clone()].iter())
                .flat_map(|o| o.operands.iter().map(|x| x.tensor))
                .filter(|&t| prog.tensors[t].temp)
                .map(|t| prog.tensors[t].id.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            ExecGroup {
                kind: if ops.len() > 1 { GroupKind::Fused { on_chip: temps } } else { GroupKind::Single },
                ops,
                launch: launch_kind(exec),
                barrier_after: true,
            }
        })
        .collect();
    let mut mapping = Mapping {
        version: MAPPING_VERSION,
        design_hash: view.hw.design_hash.clone(),
        workload_hash: opts.workload_hash.clone(),
        unit_sets,
        ops: IndexMap::new(),
        tensors,
        groups: groups.clone(),
        routing: RoutingPolicy::Ecmp,
        routes: IndexMap::new(),
        priorities: None,
    };
    let mut l = Lowerer::new(prog, view, &mapping, cost)?;
    let mut sets: Vec<Vec<usize>> = vec![vec.clone()];
    let mut memo: BTreeMap<String, Vec<SplitAxis>> = BTreeMap::new();
    for g in &groups {
        let first = g.ops.first().and_then(|o| prog.op(o));
        let iteration = first.and_then(|i| prog.iteration_of_op(i));
        let label = first.map_or_else(String::new, |i| prog.nodes[prog.ops[i].node].path.clone());
        l.begin_group(label, g, iteration);
        let mut prev: Option<Vec<SplitAxis>> = None;
        for o in &g.ops {
            let i = prog.op(o).expect("program op");
            let op = &prog.ops[i];
            if !prog.placed(i) {
                continue;
            }
            let (name, eligible): (String, Vec<usize>) = if op.class() == KernelClass::Contraction {
                let (pa, pb) = mac_precisions(prog, op);
                (format!("mac.{}x{}", pa.name(), pb.name()), balanced_prefix(view, mac.iter().copied().filter(|&u| supports(view, u, pa, pb)).collect()))
            } else {
                // Vector work runs in a mode holding every operand precision (01 §6); units without one cannot take it.
                let dts = crate::cost::vector_dtypes(prog, op);
                let ok = balanced_prefix(view, all_vec.iter().copied().filter(|&u| crate::cost::vector_unit_mode(&view.hw, view.units[u].unit, prog, op).is_ok()).collect());
                let names: Vec<&str> = dts.iter().map(|p| p.name()).collect();
                (if ok == vec { "vector".into() } else { format!("vector.{}", names.join("+")) }, ok)
            };
            let set = match mapping.unit_sets.iter().position(|s| s.name == name) {
                Some(s) => s,
                None => {
                    mapping.unit_sets.push(UnitSet { name, units: eligible.iter().map(|&i| path(i)).collect() });
                    sets.push(eligible);
                    mapping.unit_sets.len() - 1
                }
            };
            let (set, units) = (set as u32, &sets[set]);
            if units.is_empty() {
                return Err(Diagnostic::error("E-MAP-OP-004", format!("no unit can run {:?} kernels", op.class())).at(o.clone()));
            }
            let sig = op_signature(prog, op, &mapping.tensors, &prev);
            let chosen = match memo.get(&sig) {
                Some(s) => s.clone(),
                None => {
                    let mut cands = vec![];
                    if let Some(p) = &prev {
                        let inherited: Vec<SplitAxis> = p.iter().filter(|a| op.dim_ix(&a.dim).is_some_and(|d| op.segs.iter().all(|s| s.ext[d] == a.parts.iter().sum::<u64>()))).cloned().collect();
                        cands.push(inherited);
                    }
                    for c in candidates(op, view, units, opts.max_candidates) {
                        if !cands.contains(&c) {
                            cands.push(c);
                        }
                    }
                    if units.len() > opts.prescreen_units && cands.len() > opts.prescreen_keep {
                        let mut pre: Vec<(f64, usize)> = Vec::with_capacity(cands.len());
                        for (ci, c) in cands.iter().enumerate() {
                            pre.push((l.compute_estimate(i, &placement(op, set, c.clone(), units.len()), units)?, ci));
                        }
                        pre.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        let keep: std::collections::BTreeSet<usize> = pre.iter().take(opts.prescreen_keep).map(|x| x.1).chain([0]).collect();
                        cands = cands.into_iter().enumerate().filter(|(ci, _)| keep.contains(ci)).map(|x| x.1).collect();
                    }
                    let mut best: Option<(f64, f64, usize)> = None;
                    let mut infeasible = None;
                    for (ci, c) in cands.iter().enumerate() {
                        let p = placement(op, set, c.clone(), units.len());
                        // A candidate wins below `b * (1 - 1e-12)`, or within `1e-12` of `b` on less summed resource load (ties
                        // do not fall to candidate order, which hardware nothing uses could reorder); its estimate is
                        // never below the one from floor costs, so a floor estimate above `b * (1 + 1e-12)` cannot win.
                        if let Some((b, _, _)) = best
                            && opts.bound_candidates
                        {
                            l.floor = true;
                            let lb = l.lower_op(i, &p, units, false);
                            l.floor = false;
                            match lb {
                                Ok(lb) if lb > b * (1.0 + 1e-12) => continue,
                                Err(e) if UNMAPPABLE_CANDIDATE.contains(&e.code.as_str()) => continue,
                                Err(e) => return Err(e),
                                Ok(_) => {}
                            }
                        }
                        let est = match l.lower_op(i, &p, units, false) {
                            Ok(e) => e,
                            Err(e) if UNMAPPABLE_CANDIDATE.contains(&e.code.as_str()) => {
                                infeasible.get_or_insert(e);
                                continue;
                            }
                            Err(e) => return Err(e),
                        };
                        report.candidates_scored += 1;
                        if best.is_none_or(|(b, bl, _)| est < b * (1.0 - 1e-12) || (est <= b * (1.0 + 1e-12) && l.load < bl * (1.0 - 1e-9))) {
                            best = Some((est, l.load, ci));
                        }
                    }
                    if let (None, Some(e)) = (best, infeasible) {
                        return Err(e);
                    }
                    let c = cands.swap_remove(best.map_or(0, |b| b.2));
                    memo.insert(sig, c.clone());
                    c
                }
            };
            let p = placement(op, set, chosen.clone(), units.len());
            l.lower_op(i, &p, units, true)?;
            mapping.ops.insert(o.clone(), p);
            prev = Some(chosen);
        }
        l.end_group();
    }
    Ok((mapping, report, l.finish()))
}

/// Region helper for tests and moves: the whole-tensor box.
pub fn whole_box(shape: &[u64]) -> TBox {
    TBox::new(&vec![0; shape.len()], &shape.iter().map(|&x| x as i64).collect::<Vec<_>>(), shape)
}
