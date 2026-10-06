//! `kiln bench export` data path (06 §4.2): bound nodes/kernels as `BenchOp` descriptors with content keys and
//! legacy harness keys (02 §13.2 `key_for`).

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::wl::*;
use serde::{Deserialize, Serialize};

pub use kiln_ir::bench::{BenchKind, BenchOp, BenchOperand, Residency, dtype_name};

use crate::graph::{LoweredGraph, LoweredNode};

/// Where and how often a descriptor occurs; not part of its key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occurrence {
    /// Suite member or harness `<phase>/<op>`; set when the export is added to a manifest.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub workload: String,
    /// IR op name (`einsum`, `attention`, ...).
    pub ir_op: String,
    pub residency: Residency,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub weight: bool,
    pub count: u64,
    pub flops: u128,
    pub min_bytes: u128,
}

/// One exported op: the canonical descriptor plus its occurrence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchExport {
    pub op: BenchOp,
    pub at: Occurrence,
}

impl BenchExport {
    pub fn key(&self) -> String {
        self.op.key()
    }

    pub fn dim(&self, d: &str) -> u64 {
        self.op.dim(d).unwrap_or(1)
    }

    /// Rewrites a contraction's dims (keeping kind and dtypes) and recomputes its dense FLOPs and compulsory bytes.
    pub fn with_gemm_dims(mut self, batch: u64, m: u64, n: u64, k: u64) -> Self {
        let dt = |o: &str| self.op.operands[o].dtype.as_str();
        self.op = BenchOp::contraction(
            (self.op.kind == BenchKind::Bmm).then_some(batch),
            m,
            n,
            k,
            self.op.kind == BenchKind::Linear,
            [dt("a"), dt("b"), dt("out")],
        );
        self.at.flops = self.op.flops().expect("contraction");
        self.at.min_bytes = self.op.min_bytes().expect("contraction dtypes parse");
        self
    }
}

fn type_of<'a>(n: &'a LoweredNode, t: &Id) -> Option<&'a TypeInfo> {
    n.node
        .inputs
        .iter()
        .zip(&n.inputs)
        .chain(n.node.outputs.iter().zip(&n.outputs))
        .chain(n.lowered.temps.iter().map(|(i, ti)| (i, ti)))
        .find(|(i, _)| *i == t)
        .map(|(_, ti)| ti)
}

/// GEMM-family descriptors of one contraction kernel (one per segment of a segmented domain). The export is the
/// dense contraction over the segment's extents; causal or windowed masking is not part of an unfused BMM.
pub fn contraction_ops(n: &LoweredNode, k: &Kernel, residency: Residency) -> Vec<BenchExport> {
    let reads: Vec<&Operand> = k
        .operands
        .iter()
        .filter(|o| o.access == Access::Read)
        .collect();
    let Some(out) = k.operands.iter().find(|o| o.access.writes()) else {
        return vec![];
    };
    let [a, b] = reads.as_slice() else {
        return vec![];
    };
    let (Some(ta), Some(tb), Some(to)) = (
        type_of(n, &a.tensor),
        type_of(n, &b.tensor),
        type_of(n, &out.tensor),
    ) else {
        return vec![];
    };
    let (mut a, mut b, mut ta, mut tb) = (*a, *b, ta, tb);
    if ta.class == TensorClass::Weight && tb.class != TensorClass::Weight {
        (a, b, ta, tb) = (b, a, tb, ta);
    }
    let (da, db, dout) = (a.dims(), b.dims(), out.dims());
    let weight = tb.class == TensorClass::Weight;
    let dtypes = [
        dtype_name(&ta.dtype),
        dtype_name(&tb.dtype),
        dtype_name(&to.dtype),
    ];
    let segments: Vec<Vec<(String, u64)>> = match &k.domain {
        Domain::Segmented { segments, .. } => segments.iter().map(|s| s.extents.clone()).collect(),
        _ => vec![vec![]],
    };
    let mut ops = Vec::new();
    for ext in segments {
        let extent = |d: &LoopDim| {
            ext.iter()
                .find(|(n, _)| *n == d.name)
                .map_or(d.extent, |x| x.1)
        };
        let (mut bt, mut m, mut nn, mut kk) = (1u64, 1u64, 1u64, 1u64);
        let (mut n_pos, mut k_pos) = (Vec::new(), Vec::new());
        for d in &k.dims {
            let (ia, ib, io) = (
                da.contains(&d.name),
                db.contains(&d.name),
                dout.contains(&d.name),
            );
            let e = extent(d);
            match (ia, ib, io) {
                (true, true, true) => bt *= e,
                (true, false, true) => m *= e,
                (false, true, true) => {
                    nn *= e;
                    n_pos.push(db.iter().position(|x| *x == d.name));
                }
                (true, true, false) => {
                    kk *= e;
                    k_pos.push(db.iter().position(|x| *x == d.name));
                }
                _ => return vec![],
            }
        }
        let shared = weight && bt == 1;
        let nk = n_pos.iter().max() < k_pos.iter().min();
        let op = BenchOp::contraction(
            (!shared).then_some(bt),
            m,
            nn,
            kk,
            shared && nk,
            [&dtypes[0], &dtypes[1], &dtypes[2]].map(String::as_str),
        );
        let at = Occurrence {
            workload: String::new(),
            ir_op: n.node.op.name().into(),
            residency,
            source: k.id.clone(),
            role: n.node.role.clone(),
            weight,
            count: n.multiplicity,
            flops: 0,
            min_bytes: 0,
        };
        ops.push(BenchExport { op, at }.with_gemm_dims(bt, m, nn, kk));
    }
    ops
}

fn node_kind(op: &Op) -> Option<BenchKind> {
    Some(match op {
        Op::Layout(_) | Op::Call(_) | Op::Repeat(_) => return None,
        Op::RmsNorm(_) | Op::LayerNorm(_) => BenchKind::Rmsnorm,
        Op::Softmax(_) => BenchKind::Softmax,
        Op::Map(_) | Op::Act(_) | Op::GatedAct(_) | Op::Rope(_) => BenchKind::Elementwise,
        Op::Collective(_) | Op::SendRecv(_) => BenchKind::Collective,
        _ => BenchKind::Other,
    })
}

/// Bench descriptors of every costed node: contraction kernels as GEMM-family ops (attention exports its two
/// unfused BMMs), other nodes as one op each with positional operands `in0..`, `out0..`.
pub fn export(lg: &LoweredGraph, residency: Residency) -> Vec<BenchExport> {
    let mut out = Vec::new();
    for n in &lg.nodes {
        let contractions: Vec<&Kernel> = n
            .lowered
            .kernels
            .iter()
            .filter(|k| k.class == KernelClass::Contraction)
            .collect();
        if !contractions.is_empty() {
            contractions
                .into_iter()
                .for_each(|k| out.extend(contraction_ops(n, k, residency)));
            continue;
        }
        let Some(kind) = node_kind(&n.node.op) else {
            continue;
        };
        let operand = |t: &TypeInfo| BenchOperand {
            shape: t.shape.clone(),
            ..BenchOperand::new(dtype_name(&t.dtype), Some("row_major"))
        };
        let mut op = BenchOp::new(kind);
        op.op = Some(n.node.op.name().into());
        op.operands = (n
            .inputs
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("in{i}"), operand(t))))
        .chain(
            n.outputs
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("out{i}"), operand(t))),
        )
        .collect();
        out.push(BenchExport {
            op,
            at: Occurrence {
                workload: String::new(),
                ir_op: n.node.op.name().into(),
                residency,
                source: n.path.clone(),
                role: n.node.role.clone(),
                weight: false,
                count: n.multiplicity,
                flops: n.cost.flops_mm,
                min_bytes: n.cost.bytes_in + n.cost.bytes_out,
            },
        });
    }
    out
}

/// Lowers one snapshot scenario of a document and exports it.
pub fn export_scenario(
    model: &Model,
    scenario: &Scenario,
) -> Result<Vec<BenchExport>, Vec<Diagnostic>> {
    let (_, lg, _) = crate::evaluate_snapshot(model, scenario)?;
    Ok(export(&lg, Residency::from_eval(scenario.eval_mode)))
}

pub const MANIFEST_SCHEMA: &str = "kiln.bench/1";

/// `kiln bench export` output: unique descriptors in first-occurrence order, each with every place it occurs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchManifest {
    pub schema: String,
    pub suite: String,
    pub ops: Vec<ManifestOp>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestOp {
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_name: Option<String>,
    pub op: BenchOp,
    pub uses: Vec<Occurrence>,
}

impl BenchManifest {
    pub fn new(suite: impl Into<String>) -> Self {
        Self {
            schema: MANIFEST_SCHEMA.into(),
            suite: suite.into(),
            ops: vec![],
        }
    }

    pub fn add(&mut self, workload: &str, e: BenchExport) {
        let key = e.key();
        let used = Occurrence {
            workload: workload.into(),
            ..e.at
        };
        match self.ops.iter_mut().find(|o| o.key == key) {
            Some(o) => o.uses.push(used),
            None => self.ops.push(ManifestOp {
                key,
                legacy_name: e.op.legacy_name(),
                op: e.op,
                uses: vec![used],
            }),
        }
    }

    pub fn get(&self, key: &str) -> Option<&ManifestOp> {
        self.ops.iter().find(|o| o.key == key)
    }
}

/// Manifest of a suite (02 §11.4), of one workload `<preset>:<scenario>`, or of `legacy`: exactly the ops the
/// legacy harness runs (the op list plus the `_linear` twin of every weight GEMM).
pub fn manifest(suite: &str) -> Result<BenchManifest, Vec<Diagnostic>> {
    if suite == "legacy" {
        return crate::legacy::harness_manifest();
    }
    let members = if suite.contains(':') {
        vec![crate::zoo::workload(suite).map_err(|e| vec![e])?]
    } else {
        crate::zoo::suite(suite).map_err(|e| vec![e])?
    };
    let mut m = BenchManifest::new(suite);
    for member in members {
        for e in export_scenario(member.model(), member.scenario())? {
            m.add(&member.name, e);
        }
    }
    Ok(m)
}
