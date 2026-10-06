//! Serializable mapping (03 §3.1): explicit uneven splits, slice-to-unit assignment, tensor homes per memory
//! instance, explicit routes, execution groups. JSON-canonical; validated before use.

use indexmap::IndexMap;
use kiln_ir::common::{Diagnostic, content_hash};
use serde::{Deserialize, Serialize};

use crate::hwview::HwView;
use crate::program::Program;

pub const MAPPING_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mapping {
    pub version: u32,
    pub design_hash: String,
    pub workload_hash: String,
    /// Named unit sets; placements refer to them by index (03 `UnitSetId`).
    pub unit_sets: Vec<UnitSet>,
    pub ops: IndexMap<String, OpPlacement>,
    pub tensors: IndexMap<String, TensorPlacement>,
    pub groups: Vec<ExecGroup>,
    pub routing: RoutingPolicy,
    /// Explicit per-transfer overrides, keyed by transfer id (`<op>.o<operand>.s<slice>.h<hop>`).
    #[serde(default)]
    pub routes: IndexMap<String, Route>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priorities: Option<Vec<(String, i32)>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitSet {
    pub name: String,
    pub units: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Target {
    Units { set: u32 },
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitAxis {
    pub dim: String,
    /// Explicit sizes, summing to the dim's extent in every segment the split applies to.
    pub parts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpPlacement {
    pub target: Target,
    pub split: Vec<SplitAxis>,
    /// Position within the target unit set per slice (row-major over `split`, segments outermost).
    pub slice_to_unit: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRegion {
    pub lo: Vec<u64>,
    pub hi: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Home {
    pub region: TensorRegion,
    /// Memory instances the region is interleaved over.
    pub mems: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Lifetime {
    /// Lives in its home for the whole step (weights, KV cache).
    Resident,
    /// Written to its home by the producer, read by consumers.
    Streamed,
    /// Kept in the producing unit's memory and consumed there (fused intermediate).
    Private,
    Spilled { to: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPlacement {
    pub home: Vec<Home>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interleave_granule_b: Option<u64>,
    pub lifetime: Lifetime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchKind {
    HostLaunch,
    DeviceQueued,
    StaticProgram,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GroupKind {
    Single,
    Fused { on_chip: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecGroup {
    pub ops: Vec<String>,
    pub kind: GroupKind,
    pub launch: LaunchKind,
    pub barrier_after: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Route {
    Path { links: Vec<String> },
    Split { paths: Vec<(Vec<String>, f64)> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPolicy {
    /// Equal-hop paths split in proportion to bandwidth (03 `Ecmp`), lowest link index first on ties.
    Ecmp,
    DimensionOrder,
}

/// Why `homes` do not partition a tensor of `shape` into in-bounds, pairwise disjoint regions; None if they do.
fn partition_error(shape: &[u64], homes: &[Home]) -> Option<String> {
    for h in homes {
        let r = &h.region;
        if r.lo.len() != shape.len() || r.hi.len() != shape.len() {
            return Some(format!("have rank {} for a rank-{} tensor", r.lo.len().max(r.hi.len()), shape.len()));
        }
        if r.lo.iter().zip(&r.hi).zip(shape).any(|((l, h), s)| l >= h || h > s) {
            return Some(format!("include an empty or out-of-bounds region {:?}..{:?} of {shape:?}", r.lo, r.hi));
        }
    }
    for (i, a) in homes.iter().enumerate() {
        for b in &homes[i + 1..] {
            if (0..shape.len()).all(|d| a.region.lo[d] < b.region.hi[d] && b.region.lo[d] < a.region.hi[d]) {
                return Some(format!("overlap ({:?}..{:?} and {:?}..{:?})", a.region.lo, a.region.hi, b.region.lo, b.region.hi));
            }
        }
    }
    let covered: u128 = homes.iter().map(|h| h.region.lo.iter().zip(&h.region.hi).map(|(l, h)| u128::from(h - l)).product::<u128>()).sum();
    let full: u128 = shape.iter().map(|&d| u128::from(d)).product();
    (covered != full).then(|| format!("cover {covered} of {full} elements"))
}

impl Mapping {
    /// Every tensor a placed op reads or writes has a placement; a private one is read only after an op of the
    /// same barrier-free span has written it (lowering finds no home to load it from otherwise).
    fn validate_uses(&self, prog: &Program) -> Vec<Diagnostic> {
        let mut out = vec![];
        let mut produced: Vec<usize> = vec![];
        for g in &self.groups {
            for o in &g.ops {
                let Some(i) = prog.op(o).filter(|&i| prog.placed(i)) else { continue };
                let op = &prog.ops[i];
                for (oi, x) in op.operands.iter().enumerate() {
                    let root = prog.root(prog.converted_from(i, oi).map_or(x.tensor, |c| c.0));
                    let t = &prog.tensors[root];
                    match self.tensors.get(&t.id) {
                        None => out.push(Diagnostic::error("E-MAP-VAL-016", format!("{} uses tensor {} which has no placement", op.id, t.id)).at(t.id.clone())),
                        Some(tp) if tp.lifetime == Lifetime::Private && x.access.reads() && !produced.contains(&root) => out.push(
                            Diagnostic::error("E-MAP-VAL-017", format!("{} reads private tensor {} before any op of its span writes it", op.id, t.id))
                                .at(t.id.clone())
                                .hint("give the tensor a home, or fuse its producer into the same span"),
                        ),
                        _ => {}
                    }
                }
                produced.extend(op.operands.iter().filter(|x| x.access.writes()).map(|x| prog.root(x.tensor)));
            }
            if g.barrier_after {
                produced.clear();
            }
        }
        out
    }

    pub fn hash(&self) -> String {
        content_hash("map1-", &serde_json::to_value(self).expect("mapping serializes"))
    }

    pub fn set_of(&self, op: &str) -> Option<&UnitSet> {
        match self.ops.get(op)?.target {
            Target::Units { set } => self.unit_sets.get(set as usize),
            Target::Host => None,
        }
    }

    /// Number of slices a placement produces for `op` (every segment is split the same way).
    pub fn slices(&self, prog: &Program, op: usize) -> usize {
        let Some(p) = self.ops.get(&prog.ops[op].id) else { return 0 };
        prog.ops[op].segs.len() * p.split.iter().map(|s| s.parts.len()).product::<usize>()
    }

    /// Structured validation (03 §3.1): every op placed, splits partition dims exactly, units exist and serve
    /// the op's pool, regions partition each homed tensor, every used tensor is placed (private ones produced
    /// before use), groups cover ops once and in program order.
    pub fn validate(&self, prog: &Program, view: &HwView) -> Vec<Diagnostic> {
        let mut out = vec![];
        let err = |code: &str, msg: String, at: &str| Diagnostic::error(code, msg).at(at.to_string());
        for set in &self.unit_sets {
            for u in &set.units {
                if view.unit_by_path(u).is_none() {
                    out.push(err("E-MAP-VAL-001", format!("unit {u} is not an enabled unit with feeds"), &set.name));
                }
            }
        }
        for (i, op) in prog.ops.iter().enumerate() {
            if !prog.placed(i) {
                continue;
            }
            let Some(p) = self.ops.get(&op.id) else {
                out.push(err("E-MAP-VAL-002", "op has no placement".into(), &op.id));
                continue;
            };
            let set = match p.target {
                Target::Units { set } => match self.unit_sets.get(set as usize) {
                    Some(s) => s,
                    None => {
                        out.push(err("E-MAP-VAL-003", format!("unit set {set} does not exist"), &op.id));
                        continue;
                    }
                },
                Target::Host => {
                    out.push(err("E-MAP-VAL-004", "host execution is unmodelled in v0".into(), &op.id).hint("target a unit set"));
                    continue;
                }
            };
            for sp in &p.split {
                let Some(d) = op.dim_ix(&sp.dim) else {
                    out.push(err("E-MAP-VAL-005", format!("split names unknown dim {}", sp.dim), &op.id));
                    continue;
                };
                for (si, seg) in op.segs.iter().enumerate() {
                    if sp.parts.iter().sum::<u64>() != seg.ext[d] || sp.parts.contains(&0) {
                        out.push(
                            err("E-MAP-VAL-006", format!("split of {} into {:?} does not partition extent {} (segment {si})", sp.dim, sp.parts, seg.ext[d]), &op.id)
                                .hint("parts must be positive and sum to the dim extent"),
                        );
                    }
                }
            }
            let n = self.slices(prog, i);
            if p.slice_to_unit.len() != n {
                out.push(err("E-MAP-VAL-007", format!("{} slices but {} unit assignments", n, p.slice_to_unit.len()), &op.id));
            }
            if p.slice_to_unit.iter().any(|&u| u as usize >= set.units.len()) {
                out.push(err("E-MAP-VAL-008", "slice assigned past the end of its unit set".into(), &op.id));
            }
            let want = if op.class() == kiln_ir::wl::KernelClass::Contraction { crate::hwview::Pool::Mac } else { crate::hwview::Pool::Vector };
            if set.units.is_empty() {
                out.push(err("E-MAP-VAL-015", format!("unit set {} is empty", set.name), &op.id).hint("no unit supports this op's class and precision"));
            }
            if set.units.iter().filter_map(|u| view.unit_by_path(u)).any(|u| view.units[u].pool != want) {
                out.push(err("E-MAP-VAL-009", format!("unit set {} mixes units that cannot run a {:?} kernel", set.name, op.class()), &op.id));
            }
        }
        for (id, tp) in &self.tensors {
            let Some(t) = prog.tensor(id) else {
                out.push(err("E-MAP-VAL-010", "placement for unknown tensor".into(), id));
                continue;
            };
            if tp.lifetime == Lifetime::Private {
                continue;
            }
            if let Some(why) = partition_error(&prog.tensors[t].shape, &tp.home) {
                out.push(err("E-MAP-VAL-011", format!("home regions {why}"), id).hint("regions must be in bounds, disjoint and cover the tensor"));
            }
            if tp.home.iter().any(|h| view.group_by_paths(&h.mems).is_none()) {
                out.push(err("E-MAP-VAL-011", "home names unknown memories".into(), id));
            }
        }
        out.extend(self.validate_uses(prog));
        // Lowering routes every transfer over the view's ECMP profiles: other choices would be scored as ECMP.
        if self.routing != RoutingPolicy::Ecmp {
            out.push(err("E-MAP-ROUTE-002", format!("routing policy {:?} is unmodelled in v0", self.routing), "routing").hint("use ecmp"));
        }
        for id in self.routes.keys() {
            out.push(err("E-MAP-ROUTE-002", "explicit transfer routes are unmodelled in v0".into(), id).hint("drop the route; transfers use ECMP"));
        }
        let mut seen = vec![false; prog.ops.len()];
        let mut last = 0usize;
        for g in &self.groups {
            for o in &g.ops {
                match prog.op(o) {
                    Some(i) if !seen[i] && i >= last => {
                        seen[i] = true;
                        last = i;
                    }
                    _ => out.push(err("E-MAP-VAL-012", format!("group lists {o} twice, out of order, or unknown"), o)),
                }
            }
        }
        if let Some(i) = seen.iter().position(|s| !s) {
            out.push(err("E-MAP-VAL-013", "op is in no execution group".into(), &prog.ops[i].id));
        }
        for w in self.groups.windows(2) {
            let it = |g: &ExecGroup| g.ops.first().and_then(|o| prog.op(o)).map(|i| prog.iteration_of_op(i));
            if it(&w[0]) != it(&w[1]) && !w[0].barrier_after {
                out.push(err("E-MAP-VAL-014", "whole-step iterations must end at a barrier in v0".into(), w[0].ops.last().map_or("", String::as_str)));
            }
        }
        out
    }
}
