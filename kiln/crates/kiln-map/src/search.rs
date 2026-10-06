//! Shared action space and seeded searches (03 §3.8). Every search is deterministic given its seed:
//! candidates are generated in a fixed order, shuffled by a SplitMix64 stream, evaluated (possibly in
//! parallel by the caller) and reduced in candidate-index order.

use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::hwview::HwView;
use crate::mapping::{ExecGroup, GroupKind, Mapping, Route, SplitAxis, Target, TensorPlacement};
use crate::program::Program;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "move")]
pub enum Move {
    Resplit { op: String, split: Vec<SplitAxis> },
    Reassign { op: String, slice: u32, unit: u32 },
    Rehome { tensor: String, home: TensorPlacement },
    Reroute { transfer: String, route: Route },
    ToggleFusion { a: String, b: String },
    SetGroupKind { group: u32, kind: GroupKind },
    SetTarget { op: String, target: Target },
    SetPriority { sel: String, prio: i32 },
    SetBarrier { group: u32, barrier_after: bool },
}

fn bad(code: &str, msg: impl Into<String>, at: &str) -> Diagnostic {
    Diagnostic::error(code, msg).at(at.to_string())
}

/// Applies a move in place; the result still needs [`Mapping::validate`].
pub fn apply(m: &mut Mapping, prog: &Program, mv: &Move) -> Result<(), Diagnostic> {
    match mv {
        Move::Resplit { op, split } => {
            let i = prog.op(op).ok_or_else(|| bad("E-MAP-MOVE-001", "unknown op", op))?;
            let n_units = m.set_of(op).map_or(1, |s| s.units.len());
            let p = m.ops.get_mut(op).ok_or_else(|| bad("E-MAP-MOVE-001", "op has no placement", op))?;
            p.split.clone_from(split);
            let total = prog.ops[i].segs.len() * split.iter().map(|s| s.parts.len()).product::<usize>();
            p.slice_to_unit = (0..total).map(|s| (s % n_units) as u32).collect();
        }
        Move::Reassign { op, slice, unit } => {
            let p = m.ops.get_mut(op).ok_or_else(|| bad("E-MAP-MOVE-001", "op has no placement", op))?;
            *p.slice_to_unit.get_mut(*slice as usize).ok_or_else(|| bad("E-MAP-MOVE-002", "slice out of range", op))? = *unit;
        }
        Move::Rehome { tensor, home } => {
            *m.tensors.get_mut(tensor).ok_or_else(|| bad("E-MAP-MOVE-003", "unknown tensor", tensor))? = home.clone();
        }
        Move::Reroute { transfer, route } => {
            m.routes.insert(transfer.clone(), route.clone());
        }
        Move::ToggleFusion { a, b } => {
            let ga = m.groups.iter().position(|g| g.ops.contains(a)).ok_or_else(|| bad("E-MAP-MOVE-004", "op in no group", a))?;
            let gb = m.groups.iter().position(|g| g.ops.contains(b)).ok_or_else(|| bad("E-MAP-MOVE-004", "op in no group", b))?;
            if ga == gb {
                let g = &mut m.groups[ga];
                let at = g.ops.iter().position(|o| o == b).expect("member");
                let tail: Vec<String> = g.ops.split_off(at);
                let ng = ExecGroup { ops: tail, kind: GroupKind::Single, launch: g.launch, barrier_after: g.barrier_after };
                g.barrier_after = true;
                if g.ops.len() == 1 {
                    g.kind = GroupKind::Single;
                }
                m.groups.insert(ga + 1, ng);
            } else if gb == ga + 1 {
                let next = m.groups.remove(gb);
                let g = &mut m.groups[ga];
                g.ops.extend(next.ops);
                g.barrier_after = next.barrier_after;
                if let GroupKind::Single = g.kind {
                    g.kind = GroupKind::Fused { on_chip: vec![] };
                }
            } else {
                return Err(bad("E-MAP-MOVE-005", "only adjacent groups can fuse", a));
            }
        }
        Move::SetGroupKind { group, kind } => {
            m.groups.get_mut(*group as usize).ok_or_else(|| bad("E-MAP-MOVE-006", "no such group", ""))?.kind = kind.clone();
        }
        Move::SetBarrier { group, barrier_after } => {
            m.groups.get_mut(*group as usize).ok_or_else(|| bad("E-MAP-MOVE-006", "no such group", ""))?.barrier_after = *barrier_after;
        }
        Move::SetTarget { op, target } => {
            m.ops.get_mut(op).ok_or_else(|| bad("E-MAP-MOVE-001", "op has no placement", op))?.target = target.clone();
        }
        Move::SetPriority { sel, prio } => {
            let p = m.priorities.get_or_insert_with(Vec::new);
            p.retain(|(s, _)| s != sel);
            p.push((sel.clone(), *prio));
        }
    }
    Ok(())
}

/// SplitMix64: tiny, seedable, platform-independent.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub time_s: f64,
    /// Per-op time, used to rank which ops moves should touch first (binding-first move generation).
    pub op_times: Vec<(String, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchOutcome {
    pub best: Mapping,
    pub score: f64,
    pub history: Vec<(Move, f64)>,
    pub evaluated: u64,
}

type Child = (Mapping, Score, Vec<(Move, f64)>);

pub trait MappingSearch: Send + Sync {
    fn search(&self, prog: &Program, view: &HwView, init: Mapping, eval: &dyn Fn(&Mapping) -> Option<Score>, seed: u64) -> SearchOutcome;
}

/// Beam over split choices of the slowest ops (03 §3.8 `BeamSearch`).
#[derive(Clone, Debug)]
pub struct BeamSearch {
    pub width: usize,
    pub depth: usize,
    pub ops_per_step: usize,
    pub moves_per_op: usize,
}

impl Default for BeamSearch {
    fn default() -> Self {
        Self { width: 2, depth: 2, ops_per_step: 3, moves_per_op: 6 }
    }
}

impl BeamSearch {
    /// Children as move batches: one resplit applied to every window instance of the same kernel.
    fn moves(&self, prog: &Program, view: &HwView, m: &Mapping, s: &Score, rng: &mut Rng) -> Vec<Vec<Move>> {
        let mut ranked = s.op_times.clone();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out = vec![];
        let mut seen_sig = std::collections::BTreeSet::new();
        for (op, _) in ranked {
            if seen_sig.len() >= self.ops_per_step {
                break;
            }
            let Some(i) = prog.op(&op) else { continue };
            let pop = &prog.ops[i];
            let base = prog.nodes[pop.node].base.clone();
            if !seen_sig.insert(format!("{base}/{}", pop.kernel.id)) {
                continue;
            }
            let units: Vec<usize> = m.set_of(&op).map_or_else(Vec::new, |s| s.units.iter().filter_map(|p| view.unit_by_path(p)).collect());
            let cur = m.ops.get(&op).map(|p| p.split.clone()).unwrap_or_default();
            let mut cands: Vec<Vec<SplitAxis>> = crate::heuristic::candidates(pop, view, &units, 64).into_iter().filter(|c| *c != cur).collect();
            rng.shuffle(&mut cands);
            for c in cands.into_iter().take(self.moves_per_op) {
                out.push(
                    prog.ops
                        .iter()
                        .filter(|o| o.kernel.id == pop.kernel.id && prog.nodes[o.node].base == base)
                        .map(|o| Move::Resplit { op: o.id.clone(), split: c.clone() })
                        .collect(),
                );
            }
        }
        out
    }
}

impl MappingSearch for BeamSearch {
    fn search(&self, prog: &Program, view: &HwView, init: Mapping, eval: &dyn Fn(&Mapping) -> Option<Score>, seed: u64) -> SearchOutcome {
        let mut rng = Rng::new(seed);
        let mut evaluated = 1u64;
        let Some(s0) = eval(&init) else {
            return SearchOutcome { best: init, score: f64::INFINITY, history: vec![], evaluated };
        };
        let mut beam: Vec<Child> = vec![(init.clone(), s0.clone(), vec![])];
        let (mut best, mut best_s, mut best_h) = (init, s0.time_s, vec![]);
        for _ in 0..self.depth {
            let mut children = vec![];
            for (m, s, h) in &beam {
                for batch in self.moves(prog, view, m, s, &mut rng) {
                    let mut child = m.clone();
                    if batch.is_empty() || !batch.iter().all(|b| apply(&mut child, prog, b).is_ok()) || !child.validate(prog, view).is_empty() {
                        continue;
                    }
                    evaluated += 1;
                    if let Some(cs) = eval(&child) {
                        let mut hh = h.clone();
                        hh.push((batch[0].clone(), cs.time_s));
                        children.push((child, cs, hh));
                    }
                }
            }
            if children.is_empty() {
                break;
            }
            children.sort_by(|a, b| a.1.time_s.total_cmp(&b.1.time_s));
            children.truncate(self.width);
            if children[0].1.time_s < best_s {
                best = children[0].0.clone();
                best_s = children[0].1.time_s;
                best_h = children[0].2.clone();
            }
            beam = children;
        }
        SearchOutcome { best, score: best_s, history: best_h, evaluated }
    }
}
