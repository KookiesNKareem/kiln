//! The op graph the mapper and engines work on: kernels of a lowered workload with program-wide tensor
//! identities, instantiated over a whole-step window (03 §4.9) or as isolated single ops (calibration).

use std::ops::Range;

use indexmap::IndexMap;
use kiln_ir::bench::{BenchKind, BenchOp};
use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::precision::Precision;
use kiln_ir::wl::{
    Access, DiffConstraint, DimKind, Domain, ElemType, IndexExpr, Kernel, KernelClass, LoopDim, Model, Op, Operand,
    ScalarBody, TensorClass, TypeInfo,
};
use kiln_wl::{LoweredGraph, LoweredNode};
use serde::{Deserialize, Serialize};

use crate::MapError;

#[derive(Clone, Debug, PartialEq)]
pub struct PTensor {
    pub id: String,
    pub shape: Vec<u64>,
    pub dtype: ElemType,
    pub class: TensorClass,
    /// Node-internal (fusable) intermediate.
    pub temp: bool,
    /// Layout view of another tensor (no data movement of its own).
    pub alias_of: Option<usize>,
}

impl PTensor {
    pub fn info(&self) -> TypeInfo {
        TypeInfo::new(self.shape.clone(), self.dtype, self.class)
    }

    pub fn bytes(&self, elems: u128) -> u128 {
        self.dtype.bytes(elems, &self.shape)
    }

    pub fn footprint(&self) -> u128 {
        self.bytes(self.shape.iter().map(|&d| u128::from(d)).product())
    }

    pub fn model_state(&self) -> bool {
        self.class.is_model_level()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct POperand {
    pub tensor: usize,
    pub access: Access,
    pub index: Vec<IndexExpr>,
}

/// One segment of a kernel's domain as a box with optional difference constraints and index-map params.
#[derive(Clone, Debug, PartialEq)]
pub struct Seg {
    pub ext: Vec<u64>,
    pub params: Vec<(String, i64)>,
    pub cons: Vec<DiffConstraint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimRole {
    Batch,
    M,
    N,
    K,
    Other,
}

/// Operand roles of a contraction: `a` streams, `b` is the stationary/weight side, `out` accumulates.
#[derive(Clone, Debug, PartialEq)]
pub struct MacRoles {
    pub a: usize,
    pub b: usize,
    pub out: usize,
    pub dims: Vec<DimRole>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct POp {
    pub id: String,
    pub node: usize,
    pub kernel: Kernel,
    pub operands: Vec<POperand>,
    pub segs: Vec<Seg>,
    pub mac: Option<MacRoles>,
}

impl POp {
    pub fn class(&self) -> KernelClass {
        self.kernel.class
    }

    pub fn body(&self) -> &ScalarBody {
        &self.kernel.body
    }

    pub fn dim_names(&self) -> impl Iterator<Item = &str> {
        self.kernel.dims.iter().map(|d| d.name.as_str())
    }

    pub fn dim_ix(&self, name: &str) -> Option<usize> {
        self.kernel.dims.iter().position(|d| d.name == name)
    }

    /// Dims that index the op's output (parallel w.r.t. the result).
    pub fn output_dims(&self) -> Vec<bool> {
        let mut v = vec![false; self.kernel.dims.len()];
        for o in self.operands.iter().filter(|o| o.access.writes()) {
            let mut ds = Vec::new();
            o.index.iter().for_each(|e| e.dims(&mut ds));
            for d in ds {
                if let Some(i) = self.dim_ix(&d) {
                    v[i] = true;
                }
            }
        }
        v
    }

    pub fn is_layout(&self) -> bool {
        self.kernel.class == KernelClass::Layout
    }

    /// Exact domain points (all segments).
    pub fn points(&self) -> u128 {
        self.kernel.points().unwrap_or(0)
    }

    pub fn useful_macs(&self) -> u128 {
        if self.kernel.class == KernelClass::Contraction { self.points() * u128::from(self.kernel.body.mac) } else { 0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PNode {
    pub path: String,
    /// Path without the iteration suffix (equal for every window instance of a body node).
    pub base: String,
    pub op_name: String,
    pub role: Option<String>,
    pub iteration: Option<u32>,
    pub ops: Range<usize>,
    /// The node's input and output tensors in node order (software-stack recipes address them by position).
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramKind {
    /// Prologue, `window` repeat iterations, epilogue (03 §4.9).
    WholeStep,
    /// Every node in isolation: operands cold off-chip, results written back (calibration only).
    Isolated,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub kind: ProgramKind,
    pub tensors: Vec<PTensor>,
    pub ops: Vec<POp>,
    pub nodes: Vec<PNode>,
    /// Iterations lowered (`w`) and the repeat count `L` they stand for; `None` without a repeat.
    pub window: Option<(u32, u64)>,
    /// Resident model state of the whole model (all layers), when known (02 `StepStats::resident`).
    pub resident_bytes: Option<u128>,
    /// Active sequences of the lowered step (software-stack `tokens_per_seq` conditions), when known.
    pub seqs: Option<u64>,
    op_index: std::collections::BTreeMap<String, usize>,
    sigs: Vec<u32>,
    /// Per op: the `(contraction, operand)` a convert op is fused into (03 §3.6 operand fusion).
    fused: Vec<Option<(u32, u32)>>,
}

impl Program {
    pub fn with_resident(mut self, bytes: u128) -> Program {
        self.resident_bytes = Some(bytes);
        self
    }

    pub fn tensor(&self, id: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.id == id)
    }

    /// Follows layout views to the tensor that owns the data.
    pub fn root(&self, mut t: usize) -> usize {
        while let Some(a) = self.tensors[t].alias_of {
            t = a;
        }
        t
    }

    pub fn op(&self, id: &str) -> Option<usize> {
        self.op_index.get(id).copied()
    }

    /// Interned shape signature, equal for every window instance of the same body kernel.
    pub fn signature(&self, op: usize) -> u32 {
        self.sigs[op]
    }

    /// The `(contraction, operand)` a convert op is fused into: kiln-wl's explicit convert of a contraction
    /// operand (a map kernel of `cvt`/`mul` reading one tensor into a node temp that only that contraction
    /// reads, with the same index map). The contraction reads the source and converts each tile it loads on
    /// the vector unit beside its MAC unit, as a dequantizing GEMM kernel (or XLA's convert fusion) does; the
    /// convert op itself gets no placement.
    pub fn fused_into(&self, op: usize) -> Option<(usize, usize)> {
        self.fused[op].map(|(c, o)| (c as usize, o as usize))
    }

    /// The source a fused convert reads for operand `oi` of contraction `op`, with the convert op.
    pub fn converted_from(&self, op: usize, oi: usize) -> Option<(usize, usize)> {
        let t = self.ops[op].operands[oi].tensor;
        let r = self.nodes[self.ops[op].node].ops.clone();
        r.into_iter().find(|&c| self.fused[c] == Some((op as u32, oi as u32))).map(|c| (self.ops[c].operands[0].tensor, c)).filter(|_| self.tensors[t].temp)
    }

    /// Per tensor, by root: the program-order ops `(first, last)` it is live over, from the first op touching it to
    /// the last; model state over the whole program. None for views and unused tensors.
    pub fn live_ranges(&self) -> Vec<Option<(usize, usize)>> {
        let mut r: Vec<Option<(usize, usize)>> = vec![None; self.tensors.len()];
        for (i, op) in self.ops.iter().enumerate() {
            for o in &op.operands {
                let x = &mut r[self.root(o.tensor)];
                *x = Some(x.map_or((i, i), |(a, _)| (a, i)));
            }
        }
        let last = self.ops.len().saturating_sub(1);
        for (t, x) in r.iter_mut().enumerate() {
            if self.tensors[t].alias_of.is_none() && self.tensors[t].model_state() {
                *x = Some((0, last));
            }
        }
        r
    }

    /// Ops with a placement of their own (not layout views, not fused converts).
    pub fn placed(&self, op: usize) -> bool {
        !self.ops[op].is_layout() && self.fused[op].is_none()
    }

    /// Ops in node order whose node belongs to `iteration` (`None` = prologue/epilogue).
    pub fn iteration_of_op(&self, op: usize) -> Option<u32> {
        self.nodes[self.ops[op].node].iteration
    }

    /// Whole-step window of `w` iterations of the first top-level repeat (all of them when `L <= w`).
    pub fn whole_step(model: &Model, lg: &LoweredGraph, w: u32) -> Result<Program, MapError> {
        let repeat = model.graphs.get(&model.entry.forward).and_then(|g| {
            g.nodes.iter().find_map(|n| match &n.op {
                Op::Repeat(r) => Some((n.id.clone(), r.clone())),
                _ => None,
            })
        });
        let mut b = Builder::new(ProgramKind::WholeStep);
        let Some((rid, rep)) = repeat else {
            for n in &lg.nodes {
                b.node(n, None, &|_, o| (o.root.as_str().to_string(), o.class))?;
            }
            return Ok(Program { seqs: lg.seqs, ..b.finish(None) });
        };
        let prefix = format!("{rid}.");
        let body: Vec<&LoweredNode> = lg.nodes.iter().filter(|n| n.path.starts_with(&prefix)).collect();
        let count = body.first().map_or(1, |n| n.multiplicity);
        let w = u64::from(w).min(count).max(1) as u32;
        let first_body = lg.nodes.iter().position(|n| n.path.starts_with(&prefix)).unwrap_or(lg.nodes.len());
        let carry_out: Vec<(Id, Id)> = rep.carry.iter().map(|c| (c.out.clone(), c.yield_.clone())).collect();
        let outer = |local: &Id, o: &kiln_wl::graph::Origin| -> (String, TensorClass) {
            match carry_out.iter().find(|(out, _)| out == local) {
                Some((_, y)) => (format!("{y}.i{}", w - 1), o.class),
                None => (o.root.as_str().to_string(), o.class),
            }
        };
        for n in &lg.nodes[..first_body] {
            b.node(n, None, &outer)?;
        }
        for i in 0..w {
            let name = |local: &Id, o: &kiln_wl::graph::Origin| -> (String, TensorClass) {
                if let Some(c) = rep.carry.iter().find(|c| &c.param == local) {
                    return (if i == 0 { c.init.as_str().to_string() } else { format!("{}.i{}", c.yield_, i - 1) }, o.class);
                }
                // A yield takes its carry's out origin (kiln-wl): only the last iteration's is that out.
                if let Some(c) = rep.carry.iter().find(|c| &c.yield_ == local || o.root == c.out) {
                    return (format!("{}.i{i}", c.yield_), if i + 1 == w { o.class } else { TensorClass::Activation });
                }
                if o.stacked || (!o.class.is_model_level() && o.class != TensorClass::Input) {
                    (format!("{}.i{i}", o.root), o.class)
                } else {
                    (o.root.as_str().to_string(), o.class)
                }
            };
            for n in &body {
                b.node(n, Some(i), &name)?;
            }
        }
        for n in lg.nodes[first_body..].iter().filter(|n| !n.path.starts_with(&prefix)) {
            b.node(n, None, &outer)?;
        }
        Ok(Program { seqs: lg.seqs, ..b.finish(Some((w, count))) })
    }

    /// Each lowered node once, as an isolated op (02 §12.6 `isolated{cold}`); multiplicities are not expanded.
    pub fn isolated(lg: &LoweredGraph) -> Result<Program, MapError> {
        let mut b = Builder::new(ProgramKind::Isolated);
        for n in &lg.nodes {
            b.node(n, None, &|_, o| (o.root.as_str().to_string(), o.class))?;
        }
        Ok(Program { seqs: lg.seqs, ..b.finish(None) })
    }

    /// A single-kernel program for a GEMM-family bench descriptor (06 §4.2), operands cold off-chip.
    pub fn bench_op(op: &BenchOp) -> Result<Program, MapError> {
        let unsupported = || kiln_ir::common::Diagnostic::error("E-MAP-OP-001", format!("bench kind {:?} has no kernel builder", op.kind));
        if !op.kind.is_contraction() {
            return Err(unsupported());
        }
        let dim = |d: &str| op.dim(d).unwrap_or(1);
        let (bt, m, n, k) = (op.dim("batch"), dim("m"), dim("n"), dim("k"));
        let dt = |o: &str| -> Result<ElemType, MapError> {
            let name = op.operands.get(o).map(|x| x.dtype.as_str()).unwrap_or("bf16");
            Precision::from_name(name)
                .map(ElemType::from)
                .ok_or_else(|| kiln_ir::common::Diagnostic::error("E-MAP-OP-001", format!("unknown dtype {name:?}")))
        };
        let batched = op.kind == BenchKind::Bmm;
        let weight_class = if batched { TensorClass::Input } else { TensorClass::Weight };
        let mut dims = vec![];
        if let (true, Some(b)) = (batched, bt) {
            dims.push(LoopDim { name: "b".into(), extent: b, kind: DimKind::Parallel });
        }
        for (name, e, kind) in [("m", m, DimKind::Parallel), ("n", n, DimKind::Parallel), ("k", k, DimKind::Reduction)] {
            dims.push(LoopDim { name: name.into(), extent: e, kind });
        }
        let with_b = |v: &[&str]| -> Vec<String> {
            let mut out: Vec<String> = if batched && bt.is_some() { vec!["b".into()] } else { vec![] };
            out.extend(v.iter().map(|s| s.to_string()));
            out
        };
        let b_layout: &[&str] = if op.kind == BenchKind::Linear { &["n", "k"] } else { &["k", "n"] };
        let specs = [
            ("a", with_b(&["m", "k"]), TensorClass::Input, Access::Read),
            ("b", with_b(b_layout), weight_class, Access::Read),
            ("out", with_b(&["m", "n"]), TensorClass::Output, Access::Write),
        ];
        let mut b = Builder::new(ProgramKind::Isolated);
        let mut operands = vec![];
        for (name, ds, class, access) in &specs {
            let shape = ds.iter().map(|d| dims.iter().find(|x| &x.name == d).map_or(1, |x| x.extent)).collect();
            b.tensor(name, shape, dt(name)?, *class, false);
            let idx: Vec<&str> = ds.iter().map(String::as_str).collect();
            operands.push(Operand::ident(&Id::new(*name).expect("id"), *access, &idx));
        }
        let kernel = Kernel {
            id: format!("{}.k0", op.legacy_name().unwrap_or_else(|| "bench".into())),
            dims,
            domain: Domain::Box,
            operands,
            body: ScalarBody { mac: 1, ..ScalarBody::default() },
            combine: Some(kiln_ir::wl::Combiner::Sum),
            accum: Some(Precision::Fp32),
            class: KernelClass::Contraction,
            opaque_cost: None,
        };
        let res: IndexMap<String, usize> = ["a", "b", "out"].iter().map(|n| (n.to_string(), b.tensors[*n])).collect();
        let name = kernel.id.trim_end_matches(".k0").to_string();
        let io = (vec![b.tensors["a"], b.tensors["b"]], vec![b.tensors["out"]]);
        b.push_node(name.clone(), name, "einsum".into(), None, None, vec![kernel], &res, io)?;
        Ok(b.finish(None))
    }
}

type Namer<'a> = &'a dyn Fn(&Id, &kiln_wl::graph::Origin) -> (String, TensorClass);

struct Builder {
    kind: ProgramKind,
    tensors: IndexMap<String, usize>,
    list: Vec<PTensor>,
    ops: Vec<POp>,
    nodes: Vec<PNode>,
}

impl Builder {
    fn new(kind: ProgramKind) -> Self {
        Self { kind, tensors: IndexMap::new(), list: vec![], ops: vec![], nodes: vec![] }
    }

    fn tensor(&mut self, id: &str, shape: Vec<u64>, dtype: ElemType, class: TensorClass, temp: bool) -> usize {
        if let Some(&i) = self.tensors.get(id) {
            return i;
        }
        let i = self.list.len();
        self.list.push(PTensor { id: id.into(), shape, dtype, class, temp, alias_of: None });
        self.tensors.insert(id.into(), i);
        i
    }

    fn node(&mut self, n: &LoweredNode, iteration: Option<u32>, name: Namer) -> Result<(), MapError> {
        let mut res: IndexMap<String, usize> = IndexMap::new();
        let ids = n.node.inputs.iter().zip(&n.inputs).chain(n.node.outputs.iter().zip(&n.outputs));
        for (local, ti) in ids {
            let origin = n.origins.iter().find(|(l, _)| l == local).map(|(_, o)| o);
            let (pid, class) = origin.map_or_else(|| (local.as_str().to_string(), ti.class), |o| name(local, o));
            let t = self.tensor(&pid, ti.shape.clone(), ti.dtype, class, false);
            res.insert(local.as_str().to_string(), t);
        }
        let suffix = iteration.map_or(String::new(), |i| format!(".i{i}"));
        for (local, ti) in &n.lowered.temps {
            let t = self.tensor(&format!("{}.{}{suffix}", n.path, local), ti.shape.clone(), ti.dtype, TensorClass::Activation, true);
            res.insert(local.as_str().to_string(), t);
        }
        let layout = n.lowered.kernels.iter().all(|k| k.class == KernelClass::Layout);
        if layout {
            let src = n.node.inputs.first().map(|i| res[i.as_str()]);
            for o in &n.node.outputs {
                let t = res[o.as_str()];
                if Some(t) != src {
                    self.list[t].alias_of = src;
                }
            }
        }
        let path = format!("{}{suffix}", n.path);
        let io = (n.node.inputs.iter().map(|i| res[i.as_str()]).collect(), n.node.outputs.iter().map(|o| res[o.as_str()]).collect());
        self.push_node(path, n.path.clone(), n.node.op.name().into(), n.node.role.clone(), iteration, n.lowered.kernels.clone(), &res, io)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_node(
        &mut self,
        path: String,
        base: String,
        op_name: String,
        role: Option<String>,
        iteration: Option<u32>,
        kernels: Vec<Kernel>,
        res: &IndexMap<String, usize>,
        (inputs, outputs): (Vec<usize>, Vec<usize>),
    ) -> Result<(), MapError> {
        let node = self.nodes.len();
        let start = self.ops.len();
        for k in kernels {
            let operands = k
                .operands
                .iter()
                .map(|o| {
                    let tensor = *res.get(o.tensor.as_str()).ok_or_else(|| {
                        kiln_ir::common::Diagnostic::error("E-MAP-OP-002", format!("kernel {} names unknown tensor {}", k.id, o.tensor)).at(&path)
                    })?;
                    let rank = o.index.len().max(self.list[tensor].shape.len());
                    if rank > crate::geom::MAX_RANK {
                        return Err(kiln_ir::common::Diagnostic::error(
                            "E-MAP-OP-005",
                            format!("kernel {} operand {} has rank {rank}; footprints support at most {}", k.id, o.tensor, crate::geom::MAX_RANK),
                        )
                        .at(&path));
                    }
                    Ok(POperand { tensor, access: o.access, index: o.index.clone() })
                })
                .collect::<Result<Vec<_>, MapError>>()?;
            let segs = segments(&k);
            let mac = mac_roles(&k, &operands, &self.list);
            let id = format!("{path}.{}", k.id.rsplit('.').next().unwrap_or(&k.id));
            self.ops.push(POp { id, node, kernel: k, operands, segs, mac });
        }
        self.nodes.push(PNode { path, base, op_name, role, iteration, ops: start..self.ops.len(), inputs, outputs });
        Ok(())
    }

    fn finish(self, window: Option<(u32, u64)>) -> Program {
        let op_index = self.ops.iter().enumerate().map(|(i, o)| (o.id.clone(), i)).collect();
        let mut interned: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        let sigs = self
            .ops
            .iter()
            .map(|o| {
                let key = format!("{}/{}/{:?}", self.nodes[o.node].base, o.kernel.id, o.segs.iter().map(|s| &s.ext).collect::<Vec<_>>());
                let n = interned.len() as u32;
                *interned.entry(key).or_insert(n)
            })
            .collect();
        let fused = fused_converts(&self.ops, &self.nodes, &self.list);
        Program { kind: self.kind, tensors: self.list, ops: self.ops, nodes: self.nodes, window, resident_bytes: None, seqs: None, op_index, sigs, fused }
    }
}

/// Converts kiln-wl inserted before contractions ([`Program::fused_into`]).
fn fused_converts(ops: &[POp], nodes: &[PNode], tensors: &[PTensor]) -> Vec<Option<(u32, u32)>> {
    let mut out = vec![None; ops.len()];
    for n in nodes {
        for c in n.ops.clone() {
            let op = &ops[c];
            let b = op.body();
            let convert = op.class() == KernelClass::Map
                && b.cvt > 0
                && b.vector() == b.mul
                && b.transcendental() == 0
                && op.operands.len() == 2
                && op.operands[0].access == Access::Read
                && op.operands[1].access == Access::Write
                && tensors[op.operands[1].tensor].temp;
            if !convert {
                continue;
            }
            let (t, index) = (op.operands[1].tensor, &op.operands[1].index);
            let readers: Vec<(usize, usize)> = n
                .ops
                .clone()
                .filter(|&x| x != c)
                .flat_map(|x| ops[x].operands.iter().enumerate().filter(|(_, o)| o.tensor == t).map(move |(oi, _)| (x, oi)))
                .collect();
            if let [(x, oi)] = readers[..]
                && x > c
                && ops[x].class() == KernelClass::Contraction
                && ops[x].operands[oi].access == Access::Read
                && ops[x].operands[oi].index == *index
            {
                out[c] = Some((x as u32, oi as u32));
            }
        }
    }
    out
}

fn segments(k: &Kernel) -> Vec<Seg> {
    let base: Vec<u64> = k.dims.iter().map(|d| d.extent).collect();
    match &k.domain {
        Domain::Box => vec![Seg { ext: base, params: vec![], cons: vec![] }],
        Domain::Constrained(cs) => vec![Seg { ext: base, params: vec![], cons: cs.clone() }],
        Domain::Segmented { segments, .. } => segments
            .iter()
            .map(|s| Seg {
                ext: k
                    .dims
                    .iter()
                    .map(|d| s.extents.iter().find(|(n, _)| *n == d.name).map_or(d.extent, |x| x.1))
                    .collect(),
                params: s.params.clone(),
                cons: s.constraints.clone(),
            })
            .collect(),
    }
}

fn mac_roles(k: &Kernel, ops: &[POperand], tensors: &[PTensor]) -> Option<MacRoles> {
    if k.class != KernelClass::Contraction {
        return None;
    }
    let reads: Vec<usize> = (0..ops.len()).filter(|&i| ops[i].access == Access::Read).collect();
    let out = (0..ops.len()).find(|&i| ops[i].access.writes())?;
    let (mut a, mut b) = (*reads.first()?, *reads.get(1)?);
    let weightish = |i: usize| tensors[ops[i].tensor].model_state();
    if weightish(a) && !weightish(b) {
        std::mem::swap(&mut a, &mut b);
    }
    let dset = |i: usize| {
        let mut v = Vec::new();
        ops[i].index.iter().for_each(|e| e.dims(&mut v));
        v
    };
    let (da, db, dout) = (dset(a), dset(b), dset(out));
    let dims = k
        .dims
        .iter()
        .map(|d| match (da.contains(&d.name), db.contains(&d.name), dout.contains(&d.name)) {
            (true, true, true) => DimRole::Batch,
            (true, false, true) => DimRole::M,
            (false, true, true) => DimRole::N,
            (true, true, false) => DimRole::K,
            _ => DimRole::Other,
        })
        .collect();
    Some(MacRoles { a, b, out, dims })
}

/// Bound points for convenience in tests and reports.
pub fn kernel_points(k: &Kernel) -> Result<u128, Diagnostic> {
    k.points()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `out[d0..d6] = in[d0..d6]` over a rank-7 box `[1, 1, 1, 1, 1, 1, 1024]`.
    #[test]
    fn tensors_beyond_the_footprint_rank_are_rejected() {
        let names: Vec<String> = (0..7).map(|d| format!("d{d}")).collect();
        let idx: Vec<&str> = names.iter().map(String::as_str).collect();
        let shape = vec![1, 1, 1, 1, 1, 1, 1024];
        let mut b = Builder::new(ProgramKind::Isolated);
        for t in ["x", "y"] {
            b.tensor(t, shape.clone(), ElemType::from(Precision::Bf16), TensorClass::Activation, false);
        }
        let kernel = Kernel {
            id: "copy.k0".into(),
            dims: names.iter().zip(&shape).map(|(n, &e)| LoopDim { name: n.clone(), extent: e, kind: DimKind::Parallel }).collect(),
            domain: Domain::Box,
            operands: vec![
                Operand::ident(&Id::new("x").unwrap(), Access::Read, &idx),
                Operand::ident(&Id::new("y").unwrap(), Access::Write, &idx),
            ],
            body: ScalarBody { add: 1, ..ScalarBody::default() },
            combine: None,
            accum: None,
            class: KernelClass::Map,
            opaque_cost: None,
        };
        let res: IndexMap<String, usize> = ["x", "y"].iter().map(|n| (n.to_string(), b.tensors[*n])).collect();
        let io = (vec![b.tensors["x"]], vec![b.tensors["y"]]);
        let e = b.push_node("copy".into(), "copy".into(), "copy".into(), None, None, vec![kernel], &res, io).expect_err("rank 7");
        assert_eq!(e.code, "E-MAP-OP-005");
    }
}
