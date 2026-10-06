//! Software-stack kernel recipes (`kiln.stack/1`, files in `kiln/stacks/`): how the software that runs a step
//! splits each workload node into device kernels, and the bytes each unfused kernel moves. Part of the
//! execution model: candidate and baseline are scored under the same recipe (08 §F).
//!
//! A rule matches a node by op name, optional roles and shape conditions. Its `primary` kernel is the one the
//! mapped task graph already models; every other kernel is charged on top (launch class plus its own traffic).
//! A node without a matching rule is one primary kernel. A `tiles` rule describes the library's tile quantization
//! of a node's contractions: the rows it issues per tile, padded rows included.

use kiln_ir::common::{Diagnostic, content_hash};
use kiln_ir::hw::types::ExecModel;
use kiln_ir::wl::{TensorClass, TypeInfo};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "kiln.stack/1";
pub const CODE: &str = "E-WL-STACK-001";

const BUILTIN: [(&str, &str); 3] = [
    ("pytorch_cuda_graph_sdpa", include_str!("../../../stacks/pytorch_cuda_graph_sdpa.json5")),
    ("xla_tpu_fused", include_str!("../../../stacks/xla_tpu_fused.json5")),
    ("kiln_ideal", include_str!("../../../stacks/kiln_ideal.json5")),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stack {
    pub schema: String,
    pub id: String,
    /// Execution model the stack runs under (documentation; the default per model is [`default_for`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec_model: Option<ExecModel>,
    pub description: String,
    #[serde(default)]
    pub sources: Vec<String>,
    /// Unfused intermediates whose largest access is at most this fraction of the shared on-chip level are
    /// served from it; larger ones round-trip off chip.
    pub onchip_fraction: f64,
    /// The compiler fuses elementwise nodes (norms, RoPE, activations, residual adds, cache appends) into the
    /// fusion of the neighbouring contraction (03 §3.6): the mapper groups them with it and their tiles stream,
    /// so their vector work overlaps the contraction's instead of running as its own kernel.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fuse_elementwise: bool,
    /// The stack has no mixed-input GEMM kernels: a low-precision operand is converted to the other side's
    /// precision before every contraction even where the design declares a mixed mode (bf16 x fp8), as an
    /// explicit convert kernel ([`crate::convert`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dequantize: bool,
    #[serde(default)]
    pub rules: Vec<Rule>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiles: Vec<TileRule>,
}

/// Library tile quantization of a node's contractions (cuBLAS/CUTLASS CTA tiles, flash-attention query blocks):
/// the kernels issue `rows` (a product of shape terms) in tiles of `tile` rows, so the padded rows of the last
/// tile are issued MACs on top of the useful ones. The first matching rule applies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileRule {
    pub op: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
    #[serde(default)]
    pub when: When,
    pub rows: Vec<RowTerm>,
    pub tile: u64,
}

/// Shape terms whose product is the rows a library kernel tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowTerm {
    /// Leading dim of the first input.
    Tokens,
    /// [`NodeShape`] tokens per active sequence.
    TokensPerSeq,
    /// Query heads (dim 1 of the first input) per KV head (dim 2 of the first KV-cache input): grouped-query
    /// attention kernels that fold a KV head's query heads into the query rows of a tile.
    QueryHeadsPerKvHead,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    /// IR op name (`rms_norm`, `rope`, `einsum`, ...).
    pub op: String,
    /// Node roles the rule applies to (empty: any).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
    #[serde(default)]
    pub when: When,
    pub kernels: Vec<KernelDecl>,
}

/// Shape conditions. `tokens` is the leading dim of the first input; `tokens_per_seq` divides it by the step's
/// active sequences (`N`), or, when unknown, by the leading dim (slots) of the first KV-cache input (1 without one).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct When {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_min: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_seq_min: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_seq_max: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelKind {
    #[default]
    Kernel,
    Memset,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelDecl {
    pub name: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub primary: bool,
    #[serde(default)]
    pub kind: KernelKind,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub count: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<Access>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<Access>,
}

fn one() -> u32 {
    1
}

fn is_one(x: &u32) -> bool {
    *x == 1
}

/// Bytes of a node tensor (`in<i>` / `out<i>`), at its own dtype or `dtype`, scaled by `frac`; `rows` keeps
/// one element per row (the reduction result of the last dim).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Access {
    pub of: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default = "unit", skip_serializing_if = "is_unit")]
    pub frac: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rows: bool,
}

fn unit() -> f64 {
    1.0
}

fn is_unit(x: &f64) -> bool {
    *x == 1.0
}

/// One non-primary kernel of a node, with the bytes it reads and writes.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCharge {
    pub name: String,
    pub kind: KernelKind,
    pub read_b: f64,
    pub write_b: f64,
    /// Largest single access (decides on-chip vs off-chip service).
    pub footprint_b: f64,
}

impl KernelCharge {
    pub fn bytes(&self) -> f64 {
        self.read_b + self.write_b
    }
}

/// A charged kernel bound to an execution group of a task graph, with the memory level serving its traffic.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupKernel {
    pub group: u32,
    pub name: String,
    pub kind: KernelKind,
    pub bytes: f64,
    /// Of `bytes`, those written.
    pub wbytes: f64,
    /// Served by the shared on-chip level (else off chip).
    pub onchip: bool,
}

/// The node a recipe expands: op name, role, input and output tensors in node order.
pub struct NodeShape<'a> {
    pub op: &'a str,
    pub role: Option<&'a str>,
    pub inputs: &'a [TypeInfo],
    pub outputs: &'a [TypeInfo],
    /// Active sequences of the step (`N`); without it, the slots of the first KV-cache input stand in.
    pub seqs: Option<u64>,
}

impl NodeShape<'_> {
    fn tokens(&self) -> u64 {
        self.inputs.first().and_then(|t| t.shape.first()).copied().unwrap_or(1)
    }

    fn tokens_per_seq(&self) -> u64 {
        let seqs = self.seqs.unwrap_or_else(|| {
            self.inputs.iter().find(|t| t.class == TensorClass::KvCache).and_then(|t| t.shape.first()).copied().unwrap_or(1)
        });
        self.tokens() / seqs.max(1)
    }
}

fn dtype_bytes(name: &str) -> Option<f64> {
    Some(match name {
        "fp32" | "int32" | "uint32" => 4.0,
        "bf16" | "fp16" | "int16" => 2.0,
        "fp8" | "int8" | "uint8" => 1.0,
        _ => return None,
    })
}

impl RowTerm {
    fn value(self, n: &NodeShape) -> Option<u64> {
        match self {
            Self::Tokens => Some(n.tokens()),
            Self::TokensPerSeq => Some(n.tokens_per_seq()),
            Self::QueryHeadsPerKvHead => {
                let q = n.inputs.first()?.shape.get(1).copied()?;
                let kv = n.inputs.iter().find(|t| t.class == TensorClass::KvCache)?.shape.get(2).copied()?;
                (kv > 0 && q.is_multiple_of(kv)).then(|| q / kv)
            }
        }
    }
}

impl When {
    fn holds(&self, n: &NodeShape) -> bool {
        let (t, s) = (n.tokens(), n.tokens_per_seq());
        self.tokens_min.is_none_or(|m| t >= m)
            && self.tokens_max.is_none_or(|m| t <= m)
            && self.tokens_per_seq_min.is_none_or(|m| s >= m)
            && self.tokens_per_seq_max.is_none_or(|m| s <= m)
    }
}

impl Access {
    fn bytes(&self, n: &NodeShape) -> Result<f64, Diagnostic> {
        let bad = || Diagnostic::error(CODE, format!("access {:?} names no tensor of {} node", self.of, n.op));
        let t = match (self.of.strip_prefix("in"), self.of.strip_prefix("out")) {
            (Some(i), _) => n.inputs.get(i.parse::<usize>().map_err(|_| bad())?),
            (_, Some(i)) => n.outputs.get(i.parse::<usize>().map_err(|_| bad())?),
            _ => None,
        }
        .ok_or_else(bad)?;
        let mut elems = t.numel() as f64;
        if self.rows {
            elems /= t.shape.last().copied().unwrap_or(1).max(1) as f64;
        }
        let per = match &self.dtype {
            Some(d) => dtype_bytes(d).ok_or_else(|| Diagnostic::error(CODE, format!("unknown dtype {d:?}")))?,
            None => t.footprint() as f64 / (t.numel().max(1) as f64),
        };
        Ok(elems * per * self.frac)
    }
}

impl Stack {
    pub fn parse(text: &str) -> Result<Stack, Diagnostic> {
        let v = kiln_ir::hw::author::parse_json5(text)?;
        let s: Stack = serde_json::from_value(v).map_err(|e| Diagnostic::error(CODE, format!("invalid stack recipe: {e}")))?;
        s.validate()?;
        Ok(s)
    }

    /// A built-in recipe by id, or a recipe file.
    pub fn load(spec: &str) -> Result<Stack, Diagnostic> {
        if let Some((_, text)) = BUILTIN.iter().find(|(id, _)| *id == spec) {
            return Stack::parse(text);
        }
        let text = std::fs::read_to_string(spec)
            .map_err(|e| Diagnostic::error(CODE, format!("{spec}: {e}")).hint(format!("built-in stacks: {}", builtin_ids().join(", "))))?;
        Stack::parse(&text)
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        let err = |m: String| Err(Diagnostic::error(CODE, m).at(self.id.clone()));
        if self.schema != SCHEMA {
            return err(format!("schema {:?}, expected {SCHEMA}", self.schema));
        }
        if !(0.0..=1.0).contains(&self.onchip_fraction) {
            return err(format!("onchip_fraction {} outside [0, 1]", self.onchip_fraction));
        }
        for t in &self.tiles {
            if t.tile == 0 || t.rows.is_empty() {
                return err(format!("tile rule for {}: tile must be >= 1 over at least one row term", t.op));
            }
        }
        for r in &self.rules {
            if r.kernels.iter().filter(|k| k.primary).count() != 1 {
                return err(format!("rule for {} needs exactly one primary kernel", r.op));
            }
            for k in &r.kernels {
                if k.count == 0 || (k.primary && (k.count != 1 || !k.reads.is_empty() || !k.writes.is_empty())) {
                    return err(format!("kernel {} of {}: count must be >= 1; a primary kernel is single and has no traffic of its own", k.name, r.op));
                }
                for a in k.reads.iter().chain(&k.writes) {
                    if !(a.frac.is_finite() && a.frac >= 0.0) || a.dtype.as_deref().is_some_and(|d| dtype_bytes(d).is_none()) {
                        return err(format!("kernel {} of {}: bad access {:?}", k.name, r.op, a.of));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn hash(&self) -> String {
        content_hash("stk1-", &serde_json::to_value(self).expect("stack serializes"))
    }

    /// `id@hash` for provenance.
    pub fn label(&self) -> String {
        format!("{}@{}", self.id, self.hash())
    }

    pub fn rule(&self, n: &NodeShape) -> Option<&Rule> {
        self.rules.iter().find(|r| r.op == n.op && (r.roles.is_empty() || n.role.is_some_and(|x| r.roles.iter().any(|y| y == x))) && r.when.holds(n))
    }

    /// Issued over useful MACs of a node's contractions under the first matching tile rule: rows rounded up to
    /// whole tiles (1 without a rule).
    pub fn row_padding(&self, n: &NodeShape) -> Result<f64, Diagnostic> {
        let Some(t) = self.tiles.iter().find(|t| t.op == n.op && (t.roles.is_empty() || n.role.is_some_and(|x| t.roles.iter().any(|y| y == x))) && t.when.holds(n)) else {
            return Ok(1.0);
        };
        let rows = t.rows.iter().try_fold(1u64, |a, r| r.value(n).map(|v| a.saturating_mul(v))).ok_or_else(|| {
            Diagnostic::error(CODE, format!("tile rule for {}: a row term {:?} is not defined on this node", t.op, t.rows)).at(self.id.clone())
        })?;
        Ok(if rows == 0 { 1.0 } else { (rows.div_ceil(t.tile) * t.tile) as f64 / rows as f64 })
    }

    /// The kernels a node costs beyond its primary one (empty without a matching rule).
    pub fn extra_kernels(&self, n: &NodeShape) -> Result<Vec<KernelCharge>, Diagnostic> {
        let Some(r) = self.rule(n) else { return Ok(vec![]) };
        let mut out = vec![];
        for k in r.kernels.iter().filter(|k| !k.primary) {
            let rd: Vec<f64> = k.reads.iter().map(|a| a.bytes(n)).collect::<Result<_, _>>()?;
            let wr: Vec<f64> = k.writes.iter().map(|a| a.bytes(n)).collect::<Result<_, _>>()?;
            let c = KernelCharge {
                name: k.name.clone(),
                kind: k.kind,
                read_b: rd.iter().sum(),
                write_b: wr.iter().sum(),
                footprint_b: rd.iter().chain(&wr).copied().fold(0.0, f64::max),
            };
            out.extend(std::iter::repeat_n(c, k.count as usize));
        }
        Ok(out)
    }
}

pub fn builtin_ids() -> Vec<&'static str> {
    BUILTIN.iter().map(|b| b.0).collect()
}

/// The recipe a design gets by default: the stack its execution model is measured with.
pub fn default_for(m: ExecModel) -> &'static str {
    match m {
        ExecModel::HostLaunched => "pytorch_cuda_graph_sdpa",
        ExecModel::StaticDataflow => "xla_tpu_fused",
        ExecModel::DeviceQueued => "kiln_ideal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_ir::wl::ElemType;

    fn act(shape: &[u64]) -> TypeInfo {
        TypeInfo::new(shape.to_vec(), ElemType::BF16, TensorClass::Activation)
    }

    #[test]
    fn builtins_parse_and_hash() {
        for id in builtin_ids() {
            let s = Stack::load(id).unwrap();
            assert_eq!(s.id, id);
            assert!(s.hash().starts_with("stk1-"));
        }
        for m in [ExecModel::HostLaunched, ExecModel::StaticDataflow, ExecModel::DeviceQueued] {
            assert!(Stack::load(default_for(m)).is_ok());
        }
    }

    #[test]
    fn pytorch_rmsnorm_round_trips_fp32() {
        let s = Stack::load("pytorch_cuda_graph_sdpa").unwrap();
        let (x, w) = (act(&[2048, 4096]), act(&[4096]));
        let ins = [x.clone(), w];
        let n = NodeShape { op: "rms_norm", role: None, inputs: &ins, outputs: std::slice::from_ref(&x), seqs: None };
        let k = s.extra_kernels(&n).unwrap();
        assert_eq!(k.len(), 7);
        let full16 = 2048.0 * 4096.0 * 2.0;
        assert_eq!(k[0].read_b, full16);
        assert_eq!(k[0].write_b, 2.0 * full16);
        assert_eq!(k[1].bytes(), 4.0 * full16);
        assert_eq!(k[2].write_b, 2048.0 * 4.0);
    }

    #[test]
    fn conditions_select_rules() {
        let s = Stack::load("pytorch_cuda_graph_sdpa").unwrap();
        let cache = |b: u64| TypeInfo::new(vec![b, 2048, 8, 128], ElemType::BF16, TensorClass::KvCache);
        let attn = |t: u64, b: u64| {
            let ins = [act(&[t, 32, 128]), cache(b), cache(b)];
            let outs = [act(&[t, 32, 128])];
            let n = NodeShape { op: "attention", role: Some("attn.core"), inputs: &ins, outputs: &outs, seqs: None };
            s.extra_kernels(&n).unwrap().into_iter().map(|k| k.name).collect::<Vec<_>>()
        };
        assert_eq!(attn(8, 8), ["splitkv_combine"]);
        assert!(attn(32, 32).is_empty());
        assert_eq!(attn(2048, 1), ["memset"]);
        let gemm = |t: u64, role: &str| {
            let ins = [act(&[t, 4096]), act(&[4096, 4096])];
            let outs = [act(&[t, 4096])];
            s.extra_kernels(&NodeShape { op: "einsum", role: Some(role), inputs: &ins, outputs: &outs, seqs: None }).unwrap().len()
        };
        assert_eq!((gemm(8, "attn.o"), gemm(32, "attn.o"), gemm(32, "mlp.down"), gemm(1, "attn.qkv")), (1, 0, 1, 0));
        let ideal = Stack::load("kiln_ideal").unwrap();
        assert!(ideal.rules.is_empty());
    }

    #[test]
    fn active_sequences_not_cache_slots_select_rules() {
        let s = Stack::load("pytorch_cuda_graph_sdpa").unwrap();
        let attn = |slots: u64, seqs: Option<u64>| {
            let cache = TypeInfo::new(vec![slots, 16, 8, 128], ElemType::BF16, TensorClass::KvCache);
            let ins = [act(&[16, 32, 128]), cache.clone(), cache];
            let outs = [act(&[16, 32, 128])];
            let n = NodeShape { op: "attention", role: Some("attn.core"), inputs: &ins, outputs: &outs, seqs };
            s.extra_kernels(&n).unwrap().into_iter().map(|k| k.name).collect::<Vec<_>>()
        };
        assert_eq!(attn(1, Some(1)), ["memset"]);
        assert_eq!(attn(16, Some(1)), attn(1, Some(1)));
        assert_eq!(attn(16, None), attn(16, Some(16)));
    }

    #[test]
    fn library_tiles_pad_rows() {
        let s = Stack::load("pytorch_cuda_graph_sdpa").unwrap();
        let cache = |b: u64| TypeInfo::new(vec![b, 2048, 8, 128], ElemType::BF16, TensorClass::KvCache);
        let attn = |t: u64, b: u64| {
            let ins = [act(&[t, 32, 128]), cache(b), cache(b)];
            let outs = [act(&[t, 32, 128])];
            s.row_padding(&NodeShape { op: "attention", role: Some("attn.core"), inputs: &ins, outputs: &outs, seqs: Some(b) }).unwrap()
        };
        // Flash folds the 4 query heads of a KV head into the rows of a 64-row (split-KV) or 128-row tile.
        assert_eq!((attn(1, 1), attn(8, 8), attn(32, 32)), (16.0, 16.0, 32.0));
        assert_eq!(attn(2000, 1), 2048.0 / 2000.0);
        let gemm = |t: u64| {
            let ins = [act(&[t, 4096]), act(&[4096, 4096])];
            let outs = [act(&[t, 4096])];
            s.row_padding(&NodeShape { op: "einsum", role: Some("attn.qkv"), inputs: &ins, outputs: &outs, seqs: None }).unwrap()
        };
        assert_eq!((gemm(1), gemm(32), gemm(64), gemm(2048), gemm(2050)), (64.0, 2.0, 1.0, 1.0, 2176.0 / 2050.0));
        for id in ["kiln_ideal", "xla_tpu_fused"] {
            let ins = [act(&[32, 4096]), act(&[4096, 4096])];
            let outs = [act(&[32, 4096])];
            assert_eq!(Stack::load(id).unwrap().row_padding(&NodeShape { op: "einsum", role: None, inputs: &ins, outputs: &outs, seqs: None }).unwrap(), 1.0);
        }
    }

    #[test]
    fn rejects_bad_recipes() {
        let two = r#"{schema: "kiln.stack/1", id: "x", description: "", onchip_fraction: 0.25,
            rules: [{op: "rope", kernels: [{name: "a", primary: true}, {name: "b", primary: true}]}]}"#;
        assert!(Stack::parse(two).is_err());
        let unknown = r#"{schema: "kiln.stack/1", id: "x", description: "", onchip_fraction: 0.25, extra: 1}"#;
        assert!(Stack::parse(unknown).is_err());
    }
}
