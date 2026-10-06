//! Nodes and operator attributes (02 §5, §6).

use serde::{Deserialize, Serialize};

use super::Meta;
use super::dim::DimExpr;
use super::dtype::ElemType;
use super::kernel::CostHint;
use crate::common::Id;
use crate::precision::Precision;

/// Closed role vocabulary (02 §7.5).
pub const ROLES: &[&str] = &[
    "embed",
    "attn.norm",
    "attn.qkv",
    "attn.q",
    "attn.k",
    "attn.v",
    "attn.q_down",
    "attn.q_up",
    "attn.kv_down",
    "attn.rope",
    "attn.kv_append",
    "attn.core",
    "attn.o",
    "mlp.norm",
    "mlp.gate_up",
    "mlp.gate",
    "mlp.up",
    "mlp.act",
    "mlp.down",
    "moe.router",
    "moe.route",
    "moe.dispatch",
    "moe.expert.gate_up",
    "moe.expert.act",
    "moe.expert.down",
    "moe.combine",
    "moe.shared.gate_up",
    "moe.shared.act",
    "moe.shared.down",
    "residual",
    "final_norm",
    "logits_select",
    "lm_head",
    "sample",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: Id,
    #[serde(flatten)]
    pub op: Op,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub hints: Hints,
    #[serde(default)]
    pub calib: Option<CalibRef>,
    #[serde(default, skip_serializing_if = "Meta::is_empty")]
    pub meta: Meta,
}

impl Node {
    pub fn new(id: &str, op: Op, inputs: &[&str], outputs: &[&str]) -> Self {
        let ids = |v: &[&str]| v.iter().map(|s| Id::new(*s).expect("valid id")).collect();
        Self {
            id: Id::new(id).expect("valid id"),
            op,
            inputs: ids(inputs),
            outputs: ids(outputs),
            role: None,
            hints: Hints::default(),
            calib: None,
            meta: Meta::default(),
        }
    }

    pub fn role(mut self, role: &str) -> Self {
        self.role = Some(role.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    Einsum(EinsumAttrs),
    Map(MapAttrs),
    Reduce(ReduceAttrs),
    Gather(GatherAttrs),
    Scatter(ScatterAttrs),
    Layout(LayoutAttrs),
    Attention(AttnAttrs),
    Mla(MlaAttrs),
    Softmax(SoftmaxAttrs),
    RmsNorm(RmsNormAttrs),
    LayerNorm(LayerNormAttrs),
    Rope(RopeAttrs),
    GatedAct(GatedActAttrs),
    Act(ActAttrs),
    Embedding(Empty),
    LogitsSelect(LogitsSelectAttrs),
    KvAppend(KvAppendAttrs),
    MoeRoute(MoeRouteAttrs),
    MoeDispatch(MoeDispatchAttrs),
    GroupedEinsum(EinsumAttrs),
    MoeCombine(Empty),
    TopK(TopKAttrs),
    Sample(SampleAttrs),
    Quantize(QuantizeAttrs),
    Dequantize(DequantizeAttrs),
    Collective(CollectiveAttrs),
    SendRecv(SendRecvAttrs),
    Opaque(OpaqueAttrs),
    Call(CallAttrs),
    Repeat(RepeatAttrs),
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Einsum(_) => "einsum",
            Self::Map(_) => "map",
            Self::Reduce(_) => "reduce",
            Self::Gather(_) => "gather",
            Self::Scatter(_) => "scatter",
            Self::Layout(_) => "layout",
            Self::Attention(_) => "attention",
            Self::Mla(_) => "mla",
            Self::Softmax(_) => "softmax",
            Self::RmsNorm(_) => "rms_norm",
            Self::LayerNorm(_) => "layer_norm",
            Self::Rope(_) => "rope",
            Self::GatedAct(_) => "gated_act",
            Self::Act(_) => "act",
            Self::Embedding(_) => "embedding",
            Self::LogitsSelect(_) => "logits_select",
            Self::KvAppend(_) => "kv_append",
            Self::MoeRoute(_) => "moe_route",
            Self::MoeDispatch(_) => "moe_dispatch",
            Self::GroupedEinsum(_) => "grouped_einsum",
            Self::MoeCombine(_) => "moe_combine",
            Self::TopK(_) => "top_k",
            Self::Sample(_) => "sample",
            Self::Quantize(_) => "quantize",
            Self::Dequantize(_) => "dequantize",
            Self::Collective(_) => "collective",
            Self::SendRecv(_) => "send_recv",
            Self::Opaque(_) => "opaque",
            Self::Call(_) => "call",
            Self::Repeat(_) => "repeat",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Empty {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EinsumAttrs {
    pub eq: String,
    #[serde(default)]
    pub accum: Option<Precision>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapFn {
    Add,
    Sub,
    Mul,
    Div,
    Silu,
    GeluTanh,
    GeluErf,
    Relu,
    Sigmoid,
    Exp,
    Cast,
    Scale,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FnSpec {
    One(MapFn),
    Steps(Vec<MapFn>),
}

impl FnSpec {
    pub fn steps(&self) -> &[MapFn] {
        match self {
            Self::One(f) => std::slice::from_ref(f),
            Self::Steps(v) => v,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapAttrs {
    #[serde(rename = "fn")]
    pub func: FnSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
    Mean,
    Sumsq,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceAttrs {
    pub axes: Vec<i32>,
    pub combiner: ReduceKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherAttrs {
    pub axis: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScatterCombine {
    Sum,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScatterAttrs {
    pub axis: i32,
    #[serde(default)]
    pub combine: Option<ScatterCombine>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    Reshape,
    Transpose,
    Slice,
    Concat,
    Split,
    Broadcast,
    Pad,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutAttrs {
    pub kind: LayoutKind,
    #[serde(default)]
    pub perm: Option<Vec<u8>>,
    #[serde(default)]
    pub axis: Option<i32>,
    #[serde(default)]
    pub start: Option<DimExpr>,
    #[serde(default)]
    pub len: Option<DimExpr>,
    #[serde(default)]
    pub stride: Option<DimExpr>,
    #[serde(default)]
    pub sizes: Option<Vec<DimExpr>>,
    /// Per-output row-major views applied after the op (e.g. `[T, 6144] -> [T, 32, 128]`).
    #[serde(default)]
    pub reshape: Option<Vec<Vec<DimExpr>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mask {
    None,
    Causal,
    SlidingWindow { window: u32 },
    Chunked { chunk: u32 },
    Explicit { tensor: Id, density: f64 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttnImpl {
    #[default]
    Auto,
    Fused,
    Unfused,
    Paged,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttnAttrs {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    #[serde(default)]
    pub v_head_dim: Option<u32>,
    #[serde(default)]
    pub scale: Option<f64>,
    pub mask: Mask,
    pub seqs: String,
    #[serde(default)]
    pub softcap: Option<f64>,
    #[serde(default)]
    pub sinks: bool,
    #[serde(default)]
    pub impl_hint: AttnImpl,
}

impl AttnAttrs {
    pub fn dv(&self) -> u32 {
        self.v_head_dim.unwrap_or(self.head_dim)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlaMode {
    Naive,
    Absorbed,
    #[default]
    Auto,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlaAttrs {
    pub n_heads: u32,
    #[serde(default)]
    pub q_lora_rank: Option<u32>,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub mask: Mask,
    pub seqs: String,
    #[serde(default)]
    pub mode: MlaMode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxAttrs {
    pub axis: i32,
    #[serde(default)]
    pub scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RmsNormAttrs {
    pub eps: f64,
    #[serde(default)]
    pub fused_residual: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormAttrs {
    pub eps: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeStyle {
    #[default]
    Half,
    Interleaved,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScaling {
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_pos: u64,
    },
    Yarn {
        factor: f64,
        original_max_pos: u64,
        beta_fast: f64,
        beta_slow: f64,
    },
}

/// Inputs: `x_1..x_m, pos, table`; outputs `y_1..y_m` (q and k rotated by one node).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeAttrs {
    pub theta: f64,
    pub rotary_dim: u32,
    #[serde(default)]
    pub style: RopeStyle,
    #[serde(default)]
    pub scaling: Option<RopeScaling>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateLayout {
    ConcatHalves,
    Interleaved,
    TwoInputs,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatedActAttrs {
    #[serde(rename = "fn")]
    pub func: MapFn,
    pub layout: GateLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActAttrs {
    #[serde(rename = "fn")]
    pub func: MapFn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Which {
    Last,
    All,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogitsSelectAttrs {
    pub which: Which,
    pub seqs: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvAppendAttrs {
    pub seqs: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scoring {
    Softmax,
    Sigmoid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupLimit {
    pub n_group: u32,
    pub topk_group: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoeRouteAttrs {
    pub n_experts: u32,
    pub top_k: u32,
    pub scoring: Scoring,
    pub norm_topk: bool,
    pub softmax_after_topk: bool,
    #[serde(default)]
    pub group_limited: Option<GroupLimit>,
    #[serde(default)]
    pub bias_correction: bool,
    #[serde(default)]
    pub routed_scaling: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropPolicy {
    DropOverflow,
    NoDrop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchLayout {
    CapacityPadded,
    Ragged,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeDispatchAttrs {
    pub n_experts: u32,
    pub top_k: u32,
    /// Exact rational, e.g. `"5/4"`; `None` = dropless.
    #[serde(default)]
    pub capacity_factor: Option<DimExpr>,
    pub drop_policy: DropPolicy,
    pub layout: DispatchLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopKAttrs {
    pub k: u32,
    #[serde(default)]
    pub sorted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Greedy,
    TopK { k: u32 },
    TopP { p: f64 },
    MinP { p: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampleAttrs {
    pub strategy: Strategy,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmaxFrom {
    #[default]
    Dynamic,
    Calibrated,
}

/// The target type is the output tensor's dtype; `target`, when given, must equal it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizeAttrs {
    #[serde(default)]
    pub target: Option<ElemType>,
    #[serde(default)]
    pub amax_from: AmaxFrom,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DequantizeAttrs {
    #[serde(default)]
    pub target: Option<ElemType>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    AllToAllV,
    Broadcast,
    Reduce,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    MeshAxes(Vec<String>),
    Explicit(Vec<Vec<u32>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Avg,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaggedSpec {
    pub send_bytes: Vec<u64>,
    pub recv_bytes: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveAttrs {
    pub kind: CollKind,
    pub group: Group,
    #[serde(default)]
    pub reduce: Option<ReduceOp>,
    #[serde(default)]
    pub axis: Option<i32>,
    #[serde(default)]
    pub root: Option<u32>,
    #[serde(default)]
    pub algo_hint: Option<String>,
    #[serde(default)]
    pub ragged: Option<RaggedSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Peer {
    Shift { axis: String, delta: i32 },
    Rank(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendRecvAttrs {
    pub peer: Peer,
    pub tag: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueAttrs {
    pub cost: CostHint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallAttrs {
    pub graph: Id,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Carry {
    pub init: Id,
    pub param: Id,
    #[serde(rename = "yield")]
    pub yield_: Id,
    pub out: Id,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stacked {
    pub outer: Id,
    pub param: Id,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatAttrs {
    pub body: Id,
    pub count: DimExpr,
    pub carry: Vec<Carry>,
    pub stacked: Vec<Stacked>,
    #[serde(default)]
    pub broadcast: Vec<(Id, Id)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusePolicy {
    #[default]
    Auto,
    Prefer,
    Avoid,
    Barrier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Overlap {
    Allow,
    RequireExposed,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Hints {
    pub fuse: FusePolicy,
    pub fuse_group: Option<Id>,
    pub impl_hint: Option<String>,
    pub overlap: Option<Overlap>,
}

fn graph_cold() -> String {
    "graph_cold".into()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibRef {
    pub set: String,
    pub key: String,
    #[serde(default)]
    pub kernel: Option<String>,
    #[serde(default = "graph_cold")]
    pub timing: String,
}
