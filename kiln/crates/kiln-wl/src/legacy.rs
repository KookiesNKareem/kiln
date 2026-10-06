//! Harness compatibility (02 §13.1): reproduces `calibration/oplist.json` from the workload IR.

use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::bench::{BenchExport, BenchKind, BenchManifest, Residency, export};
use crate::zoo::{gemm_model, suite};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyRow {
    pub suite: String,
    pub phase: String,
    pub op: String,
    pub key: String,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub batch: u64,
    pub weight: bool,
    pub count: u64,
    pub tokens: u64,
    pub flops: u128,
    pub min_bytes: u128,
}

fn harness_op(b: &BenchExport) -> Option<&'static str> {
    Some(match b.at.role.as_deref()? {
        "attn.qkv" => "qkv",
        "attn.o" => "o_proj",
        "mlp.gate" | "mlp.up" => "ffn_gate_up",
        "mlp.down" => "ffn_down",
        "lm_head" => "lm_head",
        "attn.core" if b.at.source.ends_with(".k0") => "attn_score",
        "attn.core" => "attn_av",
        _ => return None,
    })
}

/// One op-list row before flattening: the IR export it came from, with merged counts.
struct HarnessOp {
    suite: &'static str,
    phase: String,
    op: String,
    tokens: u64,
    e: BenchExport,
}

impl HarnessOp {
    fn row(&self) -> LegacyRow {
        let e = &self.e;
        LegacyRow {
            suite: self.suite.into(),
            phase: self.phase.clone(),
            op: self.op.clone(),
            key: e.op.legacy_key().expect("GEMM-family op"),
            m: e.dim("m"),
            n: e.dim("n"),
            k: e.dim("k"),
            batch: e.dim("batch"),
            weight: e.at.weight,
            count: e.at.count,
            tokens: self.tokens,
            flops: e.at.flops,
            min_bytes: e.at.min_bytes,
        }
    }
}

/// Ops of one harness-compat member. The harness folded the prefill batch into `count` for non-weight ops
/// (exact only for B = 1) and merged gate and up into one op with count 2L.
fn member_ops(m: &crate::zoo::SuiteMember) -> Result<Vec<HarnessOp>, Vec<Diagnostic>> {
    let (b, lg, _) = crate::evaluate_snapshot(m.model(), m.scenario())?;
    let prefill = b.seqs.max_q() > 1;
    let mut ops: Vec<HarnessOp> = Vec::new();
    for e in export(&lg, Residency::ColdDram) {
        let Some(name) = harness_op(&e) else {
            continue;
        };
        let e = if prefill && e.op.kind == BenchKind::Bmm {
            let count = e.at.count * e.dim("batch");
            let (m_, n_, k_) = (e.dim("m"), e.dim("n"), e.dim("k"));
            let mut e = e.with_gemm_dims(1, m_, n_, k_);
            e.at.count = count;
            e
        } else {
            e
        };
        match ops.iter_mut().find(|h| h.op == name) {
            Some(h) => h.e.at.count += e.at.count,
            None => ops.push(HarnessOp {
                suite: "all",
                phase: m.scenario.to_string(),
                op: name.into(),
                tokens: b.seqs.tokens(),
                e,
            }),
        }
    }
    Ok(ops)
}

fn harness_ops() -> Result<Vec<HarnessOp>, Vec<Diagnostic>> {
    let mut ops = Vec::new();
    for m in suite("legacy").map_err(|e| vec![e])? {
        ops.extend(member_ops(&m)?);
    }
    for m in suite("smoke").map_err(|e| vec![e])? {
        ops.extend(single_gemm(
            m.model(),
            m.scenario(),
            "smoke",
            "smoke",
            &m.name,
        )?);
    }
    for (mm, n, k) in [(1024, 1024, 1024), (4096, 4096, 4096), (16, 8192, 8192)] {
        let key = format!("gemm_{mm}_{n}_{k}");
        let s = crate::zoo::whole_step(
            kiln_ir::wl::PhaseKind::Custom,
            kiln_ir::wl::SeqBatch::uniform(mm, 1, 1),
        );
        ops.extend(single_gemm(
            &gemm_model(mm, n, k),
            &s,
            "make_gemm",
            "gemm",
            &key,
        )?);
    }
    Ok(ops)
}

/// All 33 rows of `calibration/oplist.json`, in its order.
pub fn oplist() -> Result<Vec<LegacyRow>, Vec<Diagnostic>> {
    Ok(harness_ops()?.iter().map(HarnessOp::row).collect())
}

/// The descriptors `gpu_bench.py` measures: every op-list row as exported from the IR, plus, for weight GEMMs,
/// the twin in the other weight layout (it times both `torch.matmul` on `[k,n]` and `F.linear` on `[n,k]`).
/// Uses are labelled `<phase>/<op>` as in the harness output.
pub fn harness_manifest() -> Result<BenchManifest, Vec<Diagnostic>> {
    let mut m = BenchManifest::new("legacy");
    for h in harness_ops()? {
        let label = format!("{}/{}", h.phase, h.op);
        let twin = h.e.op.transposed().map(|op| BenchExport {
            op,
            at: h.e.at.clone(),
        });
        m.add(&label, h.e);
        if let Some(t) = twin {
            m.add(&label, t);
        }
    }
    Ok(m)
}

fn single_gemm(
    model: &kiln_ir::wl::Model,
    s: &kiln_ir::wl::Scenario,
    suite: &'static str,
    phase: &str,
    op: &str,
) -> Result<Vec<HarnessOp>, Vec<Diagnostic>> {
    let (_, lg, _) = crate::evaluate_snapshot(model, s)?;
    Ok(export(&lg, Residency::ColdDram)
        .into_iter()
        .map(|e| HarnessOp {
            suite,
            phase: phase.into(),
            op: op.into(),
            tokens: e.dim("m"),
            e,
        })
        .collect())
}
