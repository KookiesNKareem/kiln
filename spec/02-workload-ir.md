# 02 Workload IR (spec v0)

Owner: workload section. Follows `00-overview.md` (units, ids, determinism, structured errors, provenance).
This section defines what a workload *is* in kiln: tensors, operators, graphs, phases, scenarios, parallel
plans, frontends, hashing, metric semantics, and how today's harness suite and calibration measurements map
onto it. Inference only: training is out of scope (§10).

## 0. Key decisions (summary)

| # | Decision | Why |
|---|---|---|
| D1 | Every op, named or core, lowers to one or more **affine kernels** (iteration space + per-operand index maps + scalar body). Named ops keep their identity and carry closed-form cost hints. | Mapper and cost model reason about loop nests generically; Tier A uses closed forms; a test asserts both agree. |
| D2 | Iteration domains are boxes intersected with **difference constraints** (`Σ c_i·d_i ≤ rhs`, at most 2 dims per constraint, coefficients ±1). | Exactly expresses causal, sliding-window and per-segment masks, so FLOPs are mask-aware with exact closed forms. Fixes the 2x prefill-attention overcount in the harness. |
| D3 | One forward graph serves prefill, decode and mixed (chunked/continuous) steps. A step is parametrized by a **sequence batch** (`segments: [{count, q_len, kv_len}]`). | No duplicated graphs; continuous batching is just another binding. |
| D4 | Layer repetition is a `repeat` node over a body graph with **stacked** tensors (weights, KV cache) and **carried** tensors (residual stream). N copies are never materialized. | Compact IR for LLM authoring; engines may extrapolate steady state. |
| D5 | Workload graphs are **logical (single-device semantics)**. A hardware-agnostic, deterministic **partition pass** in the workload layer turns (logical graph, `ParallelPlan`) into SPMD per-stage graphs with explicit collectives. The **mapper** (03) chooses the plan, maps mesh to chips, chooses collective algorithms, and may apply semantics-preserving collective rewrites. | Which tensor is communicated, over which mesh axes, with which collective kind, is pure sharding algebra (GSPMD). Keeping it out of 03 makes it testable without hardware and reusable during plan search. Details in §9.1. |
| D6 | Precision is a property of each tensor (`ElemType` = scalar type + scaling), not of the op. Ops declare accumulation dtype. The IR never forces explicit dequantize before a matmul; the mapper lowers onto whatever datapaths 01 declares. | Per-tensor precision (requirement 4); MX/NVFP4/int4-group are first-class. |
| D7 | New crate `kiln-wl` (workload transforms: bind, lower, partition, zoo, ONNX import, scenario expansion). `kiln-ir` keeps pure data types + serde + structural validation. | `kiln-ir` stays small and fast to compile and is shared by hardware IR; transforms are large and churn. `kiln-map` depends on `kiln-wl`. |
| D8 | Metrics follow vLLM `benchmark_serving` definitions for inference (TTFT, TPOT, ITL, output tokens/s) and report MFU in two conventions (kiln mask-aware and dense PaLM-style 2N+4LHQT per token). | Comparability with measured and published numbers. |
| D9 | Designs and baselines are scored on **whole steps** (full decode step, full prefill step) simulated end to end; isolated per-op evaluation exists for calibration only (§12.5). | Owner ruling (08 §C); per-op sums miss inter-op reuse, overlap and launch effects that dominate decode. |

## 1. Document structure

A workload file is JSON (canonical), RON or YAML. Top-level object:

```rust
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadDoc {
    pub kiln_workload: SchemaVersion,          // "0.1"
    pub id: Id,                                // [a-z0-9_.-]+
    pub model: Model,                          // or {"zoo": {...}} sugar, see §11.3
    #[serde(default)] pub scenarios: IndexMap<Id, Scenario>,
    #[serde(default)] pub plans: IndexMap<Id, ParallelPlan>,
    #[serde(default)] pub meta: Meta,          // free-form, excluded from hashes
}

pub struct Model {
    pub symbols: IndexMap<SymName, SymbolDecl>,   // §2
    pub graphs: IndexMap<Id, Graph>,              // §7; exactly one has `entry: true` per phase kind
    pub entry: EntryPoints,                       // {"forward": "llama"}
    pub tensors: IndexMap<Id, TensorDecl>,        // model-level persistent tensors (weights, KV, constants)
    #[serde(default)] pub routing: Option<RoutingDefaults>, // MoE stats defaults, §5.6
}
```

An evaluation request to the engine is `(design, workload_doc, scenario_id, plan_id?)`. If `plan_id` is
absent and the design has more than one chip, the mapper searches plans (03).

Persistent tensors (class `weight`, `kv_cache`, `constant`) are declared at model level so the
resident set is visible without walking graphs. Activations are declared inside the graph that produces them.

## 2. Symbols and dimension expressions

### 2.1 Symbols

```rust
pub struct SymbolDecl {
    pub kind: SymKind,                 // size | segments
    #[serde(default)] pub default: Option<DimExprSrc>,
    #[serde(default)] pub min: Option<u64>,
    #[serde(default)] pub max: Option<u64>,
    #[serde(default)] pub divisible_by: Option<u64>,
    #[serde(default)] pub doc: Option<String>,
}
pub enum SymKind { Size, Segments }
```

Reserved symbols every inference forward graph has (the zoo and frontends emit them):

| Symbol | Kind | Meaning | Derived from |
|---|---|---|---|
| `seqs` | segments | the sequence batch of this step (§8.1) | scenario |
| `T` | size | tokens processed this step = Σ count·q_len | `seqs` (derived, never bound directly) |
| `N` | size | sequences this step = Σ count | `seqs` (derived) |
| `kv_cap` | size | KV-cache capacity per sequence slot (footprint only) | scenario; default max kv_len |
| `slots` | size | KV-cache sequence slots allocated | scenario; default N |

Model hyperparameters (e.g. `L`, `E`) are ordinary size symbols with defaults, so one graph can be
re-instantiated at a different depth without editing it.

### 2.2 Dimension expressions

Shapes are arrays whose entries are integers or expression strings. Grammar (whitespace-insensitive):

```
expr   := term (('+'|'-') term)*
term   := factor (('*'|'/') factor)*          // '/' is exact rational division
factor := INT | RATIONAL | SYM | '(' expr ')'
        | 'ceil(' expr ')' | 'floor(' expr ')' | 'min(' expr ',' expr ')' | 'max(' expr ',' expr ')'
        | 'ceil_div(' expr ',' expr ')' | 'sum(' SEGSYM '.' FIELD ')' | 'max(' SEGSYM '.' FIELD ')'
RATIONAL := INT '/' INT  |  DECIMAL                // 1.25 parses as 5/4 exactly
```

```rust
pub enum DimExpr {
    Const(u64), Sym(SymIdx), Rat(i64, u64),
    Add(Box<DimExpr>, Box<DimExpr>), Sub(..), Mul(..), Div(..),
    Ceil(Box<DimExpr>), Floor(Box<DimExpr>), Min(..), Max(..),
    SegSum(SymIdx, SegField), SegMax(SymIdx, SegField),   // SegField = count | q_len | kv_len | count_x_q | count_x_kv
}
```

Invariants: evaluation uses exact rationals (`num-rational` over i128); a shape entry must evaluate to a
non-negative integer after binding, else error `E-WL-DIM-001` naming the tensor and the expression. Expressions
are canonicalized (§11.5) before hashing (sorted sums/products, constants folded).

### 2.3 Binding

Binding resolves every symbol to a value, producing a `BoundModel` with all shapes as `u64`. Sources, in
priority order: (1) explicit `bindings` in the scenario phase instance, (2) values produced by scenario
expansion (`seqs`, `kv_cap`, `slots`), (3) symbol `default`. Unbound symbol at evaluation: `E-WL-SYM-001`
with the list of graphs that use it. `min/max/divisible_by` violations: `E-WL-SYM-002`. Binding is
memoized by `(graph_hash, binding_hash)`.

## 3. Tensor model

### 3.1 Declaration

```rust
pub struct TensorDecl {
    pub shape: Vec<DimExprSrc>,              // logical shape, row-major semantics
    pub dtype: ElemType,                     // §3.2
    pub class: TensorClass,                  // §3.4
    #[serde(default)] pub stack: Option<DimExprSrc>, // stacked by repeat count (§7.3); shape excludes it
    #[serde(default)] pub layout: Option<LayoutSpec>, // §3.3; default row_major, unpinned
    #[serde(default)] pub sparsity: Option<Sparsity>, // §3.5
    #[serde(default)] pub alias_of: Option<Id>,       // in-place version of another tensor (§3.6)
    #[serde(default)] pub init: Option<InitHint>,     // zeros | random | file(hash); frontends only
    #[serde(default = "yes")] pub upcast_ok: bool,    // may run on a wider datapath mode (01 §6.2), see §5.7
    #[serde(default)] pub meta: Meta,
}
```

Footprint of a tensor in bytes (used for capacity checks):
`ceil(numel · elem_bits / 8) + scale_bytes(scaling)`, multiplied by the bound `stack` value when present.
`numel` uses bound shapes (`kv_cap`, `slots` for KV cache).

### 3.2 Element types and scaling

```rust
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    F32, Tf32, Bf16, F16,
    F8E4M3, F8E5M2,            // OCP FP8 (E4M3 "fn" variant: no inf, single NaN)
    F6E3M2, F6E2M3, F4E2M1,    // OCP MX element types
    E8M0,                      // MX shared exponent (scale only)
    I64, I32, I16, I8, U8, I4, U4, Bool,
}

pub struct ElemType { pub scalar: ScalarType, pub scaling: Scaling }

pub enum Scaling {
    None,
    PerTensor { scale: ScalarType },                                 // e.g. fp8 with fp32 per-tensor scale
    PerAxis   { axis: i32, scale: ScalarType },                      // per-channel
    Block     { axis: i32, block: u32, scale: ScalarType,
                #[serde(default)] zero_point: Option<ScalarType>,    // asymmetric int (AWQ/GPTQ)
                #[serde(default)] tensor_scale: Option<ScalarType> },// second level (NVFP4)
}
```

Serde accepts a string shorthand for `ElemType`; the canonical form is always the expanded struct. Shorthand names
are entries of the single `Precision` registry in `kiln-ir` (01 §6.1, owned by 01; 00 decision 4); this table
defines their `ElemType` expansion.

| Shorthand | scalar | scaling | storage bits/elem |
|---|---|---|---|
| `fp32` `bf16` `fp16` `tf32` | as named | none | 32 / 16 / 16 / 32 (tf32 stored as 32) |
| `fp8_e4m3`, `fp8_e5m2` | F8* | none (scale carried by op attr or `PerTensor`) | 8 |
| `fp8_e4m3_pt` | F8E4M3 | PerTensor{F32} | 8 (+4 B per tensor) |
| `mxfp8_e4m3`, `mxfp8_e5m2` | F8* | Block{axis:-1, block:32, scale:E8M0} | 8.25 |
| `mxfp6_e3m2`, `mxfp6_e2m3` | F6* | Block{-1, 32, E8M0} | 6.25 |
| `mxfp4` | F4E2M1 | Block{-1, 32, E8M0} | 4.25 |
| `mxint8` | I8 | Block{-1, 32, E8M0} | 8.25 |
| `nvfp4` | F4E2M1 | Block{-1, 16, F8E4M3, tensor_scale: F32} | 4.5 |
| `int32`, `int16`, `int8`, `int4`, `uint8`, `uint4`, `int64`, `bool` | I32, I16, I8, I4, U8, U4, I64, Bool | none | 32, 16, 8, 4, 8, 4, 64, 8 |
| `int8_pc` | I8 | PerAxis{0, F32} | 8 (+ per-row) |
| `int4_g128` | I4 | Block{-1, 128, Bf16, zero_point: I4} | 4.156 |

Rules:
- `axis` is the **blocking axis** in logical dims; negative indexes from the end. For matmul operands it
  must be the contraction axis (`E-WL-DT-003` otherwise), matching OCP MX v1.0 and hardware MX datapaths.
  The zoo emits weights as `[n, k]` (nn.Linear layout) so `axis: -1` is the contraction axis.
- A dim blocked by `block` must be divisible by it after binding, or the last block is padded (padding
  bytes are counted; `W-WL-DT-004` warning only in the analysis report, not an error).
- Scale storage bytes = `ceil(numel / block) · bits(scale) / 8` (+ `bits(zero_point)` likewise, + tensor scale).
- Sub-byte types pack little-endian along the innermost **layout** axis; footprint uses exact bits, rounded
  up to a byte per tensor.
- Accumulation dtype is an op attribute (`accum`), default `fp32` for float inputs and `i32` for int inputs.
  Output dtype is the output tensor's `dtype`.

### 3.3 Layout

```rust
pub struct LayoutSpec { pub layout: Layout, #[serde(default)] pub pinned: bool }
pub enum Layout {
    RowMajor,
    Permuted { order: Vec<u8> },                         // major→minor logical axes
    Tiled    { order: Vec<u8>, tiles: Vec<(u8, DimExprSrc)> }, // e.g. [[0,128],[1,128]]
    Paged    { token_axis: u8, page_tokens: u32 },       // KV cache pages (vLLM-style)
}
```

Semantics: layout is the *storage* arrangement in the tensor's home memory. `pinned: true` means the mapper
must read/write it in this layout at its home level (true by default for `weight` and `kv_cache` imported
from frontends, because their storage format is given); unpinned tensors' layouts are mapper choices.
Layout ops (`reshape`, `transpose`, ...) in the graph are logical; whether they cost data movement is decided
by the mapper given chosen layouts.

### 3.4 Lifetime classes

| `class` | Persistence | Produced by | Counted in |
|---|---|---|---|
| `weight` | resident across all phases | graph input (model-level) | resident set, weight bytes |
| `kv_cache` | resident across phases of one request | `kv_append` (in place) | resident set, KV bytes |
| `activation` | within a phase | any op | live-set |
| `input` | per phase, from host | graph input (token ids, positions) | host I/O |
| `output` | per phase, to host | graph output (sampled ids, logits) | host I/O |
| `constant` | resident, small tables (RoPE cos/sin) | none | resident set |

Invariants: `weight`, `kv_cache`, `constant` are model-level tensors. An `activation` must have exactly one
producer in its graph. A `weight` is never written by a node. Violations: `E-WL-CLS-001`.

### 3.5 Sparsity annotations (optional)

```rust
pub enum Sparsity {
    Structured   { n: u32, m: u32, axis: i32 },     // n nonzeros per m along axis (2:4 → n=2,m=4)
    Block        { block: Vec<u32>, density: f64 }, // fraction of nonzero blocks
    Unstructured { density: f64 },
    Dynamic      { density: DimExprSrc, source: Id },// e.g. activation sparsity measured or MoE-induced
}
```

Semantics: annotations are *facts about values*, not instructions. Useful FLOPs and bytes in metrics are
always computed **dense** (so MFU stays comparable); the mapper (03) may exploit sparsity only on datapaths
01 declares sparse-capable, and the saving appears as reduced executed work in HFU and energy.
Compressed storage bytes (values + metadata) for `Structured` follow 01's declared metadata format;
default 2:4 metadata = 2 bits per kept element.

### 3.6 Aliasing and in-place updates

SSA everywhere, with explicit versions: an in-place update produces a new tensor id with `alias_of` set to
the updated tensor. Both share storage; the alias chain must be linear (no two versions live with different
contents: `E-WL-ALIAS-001`). Used by `kv_append`.

## 4. Kernels: the generic compute representation

Every op lowers (in `kiln-wl::lower`) to a short list of kernels. This is what `kiln-cost` maps.

```rust
pub struct Kernel {
    pub id: String,                    // "<node_id>.k<i>", e.g. "attn.k0"
    pub dims: Vec<LoopDim>,            // iteration space; order is NOT a schedule
    pub domain: Domain,
    pub operands: Vec<Operand>,
    pub body: ScalarBody,              // per iteration point
    pub combine: Option<Combiner>,     // for reduction dims: sum | max | min | prod | online_softmax
    pub accum: Option<ScalarType>,
    pub class: KernelClass,            // contraction | map | reduce | gather | scatter | layout | collective | opaque
}

pub struct LoopDim { pub name: String, pub extent: DimExpr, pub kind: DimKind }
pub enum DimKind { Parallel, Reduction }

pub enum Domain {
    Box,
    Constrained(Vec<DiffConstraint>),
    Segmented { seg: SymIdx, seg_dim: String, per_segment: Vec<DiffConstraint> }, // see §5.4
}
/// Σ coeff_i · dim_i ≤ rhs, with ≤ 2 terms and coeff ∈ {-1, +1}.
pub struct DiffConstraint { pub terms: Vec<(i8, String)>, pub rhs: DimExpr }

pub struct Operand { pub tensor: Id, pub role: Role, pub index: Vec<IndexExpr> }
pub enum Role { Read, Write, ReadWrite }
pub enum IndexExpr {
    Affine { terms: Vec<(i64, String)>, offset: DimExpr },
    FloorDiv { inner: Box<IndexExpr>, by: u64 },   // MX block scales: k / 32
    Indirect { via: Id, index: Vec<IndexExpr> },   // gather/scatter: value of tensor `via` at index
}

#[derive(Default)]
pub struct ScalarBody {          // counts per iteration point, all u16
    pub mac: u16,                // multiply-accumulate (contractions)
    pub add: u16, pub mul: u16, pub fma: u16, pub max: u16, pub cmp: u16, pub select: u16,
    pub exp: u16, pub log: u16, pub rcp: u16, pub rsqrt: u16, pub tanh: u16, pub erf: u16, pub sin_cos: u16,
    pub cvt: u16,                // dtype conversions (incl. quantize rounding)
}
```

### 4.1 Counting rules (normative)

- `points(kernel)` = number of integer points in the domain (exact; §4.3).
- **Useful matmul FLOPs** = `2 · mac · points` over `contraction` kernels, excluding padding (capacity
  padding, uneven-shard padding).
- **Vector ops** = `points · (add+mul+fma+max+cmp+select)`; **transcendentals** = `points · (exp+log+rcp+rsqrt+tanh+erf+sin_cos)`.
  Reported separately; *not* in MFU.
- **Executed FLOPs** (HFU numerator) come from the mapper (03): tile-granular, including partially masked
  tiles, padding.
- **Compulsory bytes** of a node = bytes of distinct elements read from tensors whose home is outside the node
  plus bytes written that leave it, evaluated on the domain (so causal attention reads all K/V, a gather reads
  only gathered rows).

### 4.2 Lowering to hardware op classes (01 §6.3)

02 owns the op to `OpClass` lowering; 01 owns capabilities. Each kernel's work is split into class demands:

| Kernel content | 01 `OpClass` | Amount |
|---|---|---|
| `contraction` kernel `mac` | `matmul` | points · mac (MACs) |
| `add, mul, fma, max, cmp, select` | `elementwise` | points · count |
| `exp, log, rcp, rsqrt, tanh, erf, sin_cos` | `transcendental` | points · count |
| `reduce` kernel combine along reduction dims | `reduction` | points (elements reduced) |
| `cvt`, `quantize`, `dequantize`, upcast inserted by mapper | `convert` | points · cvt |
| `layout` kernel materialized by the mapper | `permute` | elements moved |
| `gather`/`scatter` kernels, `embedding`, `kv_append`, paged indirection | `gather_scatter` | rows · row elements |
| `top_k`, `sample` (non-greedy), MoE route top-k | `sort_topk` | elements |
| cumulative ops (top-p threshold, SSM scans) | `scan` | elements |
| collective reduction arithmetic | `collective_reduce` | (n−1)/n · elements |

`softmax`, norms and `attention` never reach 01 as composites in v0; they arrive as the classes above via their
kernels. Convolution (`conv`) is reserved for non-LLM imports (ViT patch embedding is an einsum after a
`layout` im2col view).

### 4.3 Exact point counting

For `Box`: product of extents. For `Constrained`/`Segmented`: constraints touch at most two dims each, and
all constraint pairs in v0 share the same two dims (validated, `E-WL-DOM-001` otherwise). Count is
`(product of unconstrained extents) · C(a, b)` where `C` counts lattice points of a convex polygon over the two
constrained dims, computed in closed form by clipping row ranges: for each value of the outer dim the inner
range is an interval with affine bounds, so the sum is a sum of at most 3 arithmetic series pieces
(breakpoints where clipping switches). Implemented exactly in i128. Examples:

| Mask (q in [0,Q), j in [0,K), past P = K − Q) | Constraints | Points |
|---|---|---|
| none | – | Q·K |
| causal | j − q ≤ P | Q·P + Q(Q+1)/2 |
| sliding window W (attend last W incl. self) | j − q ≤ P, q − j ≤ W − 1 − P | Σ_q min(W, P+q+1) |

## 5. Operator set

### 5.1 Node shape

```rust
pub struct Node {
    pub id: Id,
    #[serde(flatten)] pub op: Op,          // tagged by "op"
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
    #[serde(default)] pub role: Option<Role>,     // §7.5 vocabulary, e.g. "attn.qkv"
    #[serde(default)] pub hints: Hints,           // §7.4
    #[serde(default)] pub calib: Option<CalibRef>,// §13.2
    #[serde(default)] pub meta: Meta,             // source location etc., not hashed
}

#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op { Einsum(EinsumAttrs), Map(MapAttrs), Reduce(ReduceAttrs), Gather(..), Scatter(..),
              Layout(LayoutOp), Attention(AttnAttrs), Mla(MlaAttrs), Softmax(..), RmsNorm(..), LayerNorm(..),
              Rope(..), GatedAct(..), Act(..), Embedding(..), LogitsSelect(..), KvAppend(..),
              MoeRoute(..), MoeDispatch(..), GroupedEinsum(..), MoeCombine(..), TopK(..), Sample(..),
              Quantize(..), Dequantize(..), Collective(..), SendRecv(..),
              Opaque(..), Call(CallAttrs), Repeat(RepeatAttrs) }
```

Each `Op` variant implements:

```rust
pub trait OpSemantics {
    fn infer(&self, inputs: &[TypeInfo]) -> Result<Vec<TypeInfo>, IrError>;  // shapes + dtypes
    fn lower(&self, b: &BoundCtx) -> Vec<Kernel>;                            // §4
    fn cost_hint(&self, b: &BoundCtx) -> CostHint;                           // closed form for Tier A
    fn shard_rule(&self) -> ShardRule;                                       // §9.4
}
pub struct CostHint { pub flops_mm: u128, pub vec_ops: u128, pub transc: u128,
                      pub bytes_in: u128, pub bytes_out: u128, pub weight_bytes: u128 }
```

Invariant (tested for every op on a shape corpus): `cost_hint` equals the counts derived from `lower` under
§4.1 exactly. Any documented exception is listed in the op's row below.

### 5.2 Core ops

| Op | Semantics | Attributes | Lowering |
|---|---|---|---|
| `einsum` | pure contraction `out = Σ_red Π inputs` (2 inputs in v0) | `eq` (e.g. `"td,nd->tn"`), `accum` | 1 contraction kernel, dims = letters, reduction = letters absent from output, `mac:1` |
| `map` | elementwise n-ary with numpy broadcasting | `fn`: list of scalar steps or a named fn (`add`,`mul`,`sub`,`div`,`silu`,`gelu_tanh`,`gelu_erf`,`relu`,`sigmoid`,`exp`,`cast`,`scale`) | 1 map kernel; body counts from table §5.9 |
| `reduce` | reduce along axes | `axes`, `combiner` (`sum`,`max`,`min`,`mean`,`sumsq`) | 1 reduce kernel |
| `gather` | `out[i..] = src[idx[i..], ..]` | `axis` | gather kernel with `Indirect` |
| `scatter` | inverse of gather, optional `combine: sum` | `axis`, `combine` | scatter kernel |
| `layout` | `reshape`, `transpose{perm}`, `slice{axis,start,len,stride}`, `concat{axis}`, `split{axis,sizes}`, `broadcast`, `pad` | per kind | layout kernel (0 FLOPs; bytes only if the mapper materializes it) |
| `opaque` | escape hatch for imports | required `cost: CostHint`, shapes | single opaque kernel; mapper treats as roofline-only; `W-WL-OPQ-001` warning in report |

`einsum` restrictions: letters are dims of the iteration space; each operand's index is the identity map on
its letters (general affine maps are only reachable through named-op lowering); repeated letters within one
operand (diagonals) are rejected (`E-WL-EIN-001`). A batched matmul is `"bmk,bkn->bmn"`, a linear layer with
nn.Linear weight is `"tk,nk->tn"`.

### 5.3 Named ops: semantics and cost hints

Notation: T tokens, N sequences, d = d_model, H query heads, Hkv KV heads, g = H/Hkv, Dh head dim,
Dv value head dim (= Dh unless stated), V vocab, P_attn = Σ over segments of masked (q, j) points (§4.3).

| Op | Inputs → outputs | Key attrs | Useful FLOPs (flops_mm) | Vector / transcendental per unit | Lowering |
|---|---|---|---|---|---|
| `rms_norm` | x[T,d], w[d] → y[T,d] | `eps`, `fused_residual: bool` (then also takes r, emits r' = x + r) | 0 | per elem: 1 mul (sq) + 1 add (sum) + 2 mul; per row: 1 rsqrt, 1 add(eps) | reduce(sumsq) + map |
| `layer_norm` | x, w, b → y | `eps` | 0 | per elem: 2 add + 3 mul; per row: 1 rsqrt | 2 reduces + map |
| `rope` | x[T,h,Dh], pos[T], table → y | `theta`, `rotary_dim` (≤ Dh), `style: half|interleaved`, `scaling` (llama3 / yarn params, affects table only) | 0 | per rotated elem: 2 mul + 1 add; table gather per (t, rotary_dim/2) | gather(table by pos) + map |
| `act` | x → y | `fn` | 0 | §5.9 | map |
| `gated_act` | x[T,2F] (or gate, up) → y[T,F] | `fn` (silu/gelu), `layout: concat_halves|interleaved|two_inputs` | 0 | per out elem: fn + 1 mul | map |
| `softmax` | x → y | `axis`, `scale`, `mask` (optional, same as attention) | 0 | per elem: 1 max, 1 add(sub), 1 exp, 1 add, 1 mul; per row: 1 rcp | reduce(max) + map/reduce(exp,sum) + map |
| `embedding` | ids[T] int, table[V,d] → x[T,d] | – | 0 | – ; bytes = T·d·elem (gathered rows only) | gather |
| `logits_select` | h[T,d] → h_sel[N',d] | `which: last|all` (last = last q position per sequence; `all` used for speculative verify) | 0 | – | gather with Indirect over a segment-derived index |
| `kv_append` | cache_k, cache_v, k[T,Hkv,Dh], v → cache_k', cache_v' | `seqs` | 0 | – ; bytes written = T·Hkv·(Dh+Dv)·elem(cache) (+ quantize if cache dtype differs) | scatter (in place) |
| `attention` | q[T,H,Dh], k_cache, v_cache → o[T,H,Dv] | §5.4 | 2·H·(Dh+Dv)·P_attn/… see §5.4 | per point: 1 max, 1 sub, 1 exp, 1 add; per (row): 1 rcp | fused or unfused, §5.4 |
| `mla` | §5.5 | | | | |
| `moe_route`, `moe_dispatch`, `grouped_einsum`, `moe_combine` | §5.6 | | | | |
| `top_k` | x[R,C] → vals[R,k], idx[R,k] | `k`, `sorted` | 0 | per elem: ⌈log2 k⌉+1 cmp (cost hint, heap model) | reduce with custom combiner `topk(k)` |
| `sample` | logits[N,V] → ids[N] | `strategy: greedy | top_k{k} | top_p{p} | min_p{p}`, `temperature` | 0 | greedy: V cmp per row; top_k: top_k + softmax over k; top_p: softmax over V + sort-free threshold model: 3 passes over V | composite |
| `quantize` | x → x_q (+ scales materialized in x_q's storage) | target `ElemType`; `amax_from: dynamic | calibrated` | 0 | dynamic block: per elem 1 max(abs), 1 mul, 1 cvt; per block: 1 log2-floor (counted as 1 cvt) | reduce(max over block) + map |
| `dequantize` | x_q → x | target dtype | 0 | per elem: 1 mul, 1 cvt | map with FloorDiv scale access |

### 5.4 Attention

```rust
pub struct AttnAttrs {
    pub n_heads: u32, pub n_kv_heads: u32,          // GQA: g = H/Hkv; MQA: Hkv = 1; MHA: Hkv = H
    pub head_dim: u32, #[serde(default)] pub v_head_dim: Option<u32>,
    pub scale: Option<f64>,                         // default 1/sqrt(head_dim)
    pub mask: Mask,
    pub seqs: SymName,                              // "seqs"
    #[serde(default)] pub softcap: Option<f64>,     // adds 1 tanh + 2 mul per point
    #[serde(default)] pub sinks: bool,              // learned sink logits (1 extra column per row)
    #[serde(default)] pub impl_hint: AttnImpl,      // auto | fused | unfused | paged
}
pub enum Mask { None, Causal, SlidingWindow { window: u32 }, Chunked { chunk: u32 },
                Explicit { tensor: Id, density: f64 } }
```

**Indexing over a sequence batch.** Tokens of a step are packed: segment s contributes `count_s` sequences,
each with `q_len_s` query tokens, attending `kv_len_s` positions (the cache *including* the tokens appended
this step, so `past_s = kv_len_s − q_len_s ≥ 0`). For a query at local index q in [0, q_len) its absolute
position is `past + q`. The iteration space of the score kernel is `(seq, h, q, j, e)` with `Segmented`
domain: per segment, `q < q_len`, `j < kv_len`, plus mask constraints:

| Mask | Constraint on (q, j) |
|---|---|
| `causal` | j − q ≤ past |
| `sliding_window{W}` | j − q ≤ past and q − j ≤ W − 1 − past |
| `chunked{C}` | causal and same chunk (not a difference constraint; counted by closed form per chunk, documented exception) |
| `explicit` | Box domain, points scaled by `density` (approximate, flagged in report) |

`P_attn = Σ_s count_s · H · points_s(q, j)` (heads included). Then:

- Useful `flops_mm = 2 · P_attn · (Dh + Dv)` (QK^T and PV).
- Softmax per point: 1 max, 1 sub, 1 exp, 1 add; plus per (seq, h, q) row 1 rcp and Dv muls.
- Compulsory bytes: Q, O once; for each sequence and KV head, K and V rows `[0, kv_len)` restricted to the
  union of mask-reachable j (sliding window reads only W rows). GQA: K/V are read per KV head, not per
  query head; the g query heads sharing a KV head are a parallel dim the mapper may fold (as the harness did).

**Lowerings** (both are always available; `impl_hint` only biases the mapper; `fused`/`unfused` set by a
user pins it):

- `unfused`: k0 score `einsum`-like contraction over e<Dh producing S[seq,h,q,j] (domain-masked);
  k1 softmax over j (reduce max, map exp, reduce sum, map scale); k2 PV contraction over j. S and P are
  activations the mapper must place.
- `fused` (flash-style): one kernel group with dims `(seq, h, q, j, e_qk, e_v)` and combiner
  `online_softmax`: S/P are never materialized; the mapper chooses (q-tile, j-tile) and pays a per
  (row, j-tile) rescale cost of `Dv` muls + 1 exp + 1 max, which is mapping-dependent and therefore part of
  executed, not useful, work.
- `paged`: same as fused with K/V `Layout::Paged` indirection; adds per-page index reads.

Decode-specific: when `max(q_len) == 1` the score kernel has no q extent; FLOP count reduces to
`2·N·H·kv_len·(Dh+Dv)` for uniform segments. Split-KV (flash-decoding) is a mapper choice: splitting the j
reduction across tiles adds a cross-tile combine of `(H, Dv + 2)` per sequence per split.

### 5.5 MLA (DeepSeek-style multi-head latent attention)

```rust
pub struct MlaAttrs {
    pub n_heads: u32, pub q_lora_rank: Option<u32>, pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32, pub qk_rope_head_dim: u32, pub v_head_dim: u32,
    pub mask: Mask, pub seqs: SymName,
    pub mode: MlaMode,          // naive | absorbed | auto (default: absorbed when max q_len ≤ 16, else naive)
}
```

KV cache per token per layer: `c_kv[kv_lora_rank]` + `k_rope[qk_rope_head_dim]` (576 elems for V3), shared
across heads (Hkv = 1 in cache terms). The op has inputs (h_norm, weights `w_dq, w_uq, w_dkv, w_uk, w_uv, w_o`,
cache) and lowers to explicit einsums:

- `naive`: up-project cache to per-head K (nope+rope) and V, then standard attention with
  Dh = nope + rope, Dv = v_head_dim. Per-point FLOPs `2·(Dh + Dv)`; up-projection FLOPs scale with kv_len.
- `absorbed`: fold `w_uk` into the query (`q_lat = q_nope · w_uk`, per head nope→kv_lora_rank) and `w_uv`
  into the output; attention runs in latent space with Dh = kv_lora_rank + rope (576), Dv = kv_lora_rank (512),
  MQA-shaped (all heads share one latent KV). Per-point FLOPs `2·(576 + 512)` per head.

Both lowerings are semantically equal; useful FLOPs for MFU are taken from the **naive** form (convention,
so MFU does not depend on the mapper's choice); executed FLOPs reflect the chosen mode.

### 5.6 Mixture of experts

```rust
pub struct MoeRouteAttrs {
    pub n_experts: u32, pub top_k: u32,
    pub scoring: Scoring,                    // softmax | sigmoid
    pub norm_topk: bool,                     // renormalize selected weights
    pub softmax_after_topk: bool,            // Mixtral: topk on logits, softmax over the k
    #[serde(default)] pub group_limited: Option<GroupLimit>, // DeepSeek: {n_group, topk_group}
    #[serde(default)] pub bias_correction: bool,             // aux-loss-free bias (DeepSeek-V3)
    #[serde(default)] pub routed_scaling: Option<f64>,
}
pub struct MoeDispatchAttrs {
    pub n_experts: u32, pub top_k: u32,
    pub capacity_factor: Option<Rational>,   // None = dropless (ragged)
    pub drop_policy: DropPolicy,             // drop_overflow | no_drop
    pub layout: DispatchLayout,              // capacity_padded [E, C, d] | ragged [Σ t_e, d] + offsets
}
```

Graph pattern (zoo-emitted): `einsum(router) → moe_route → moe_dispatch → grouped_einsum(gate_up)
→ gated_act → grouped_einsum(down) → moe_combine`, plus optional shared-expert dense FFN on all tokens.

- `moe_route`: hn[T,d] logits from a preceding einsum ([T,E]) → `idx[T,k] i32`, `wts[T,k]`. Cost: top_k over E
  per token, softmax or sigmoid per element.
- `moe_dispatch`: permutes token rows into per-expert order. Bytes: `T·k·d·elem` moved (+ index tensors).
- `grouped_einsum`: `eq` like `"etd,end->etn"` where `t` is a **ragged dim per expert**: `t_e` tokens for
  expert e. The weight is a tensor stacked by expert (`[E, n, d]`). Iteration space is `Σ_e t_e × n × d`.
- `moe_combine`: weighted scatter-add back to `[T,d]`: per (token, k) d mul + d add.

**Routing statistics are scenario data, not graph data.** Ragged expert loads are described by a
`RoutingModel` (in scenario, defaulting to model-level `routing`):

```rust
pub enum RoutingModel {
    Uniform,                                  // t_e = T·k/E exactly (fractional allowed for expected values)
    Zipf { s: f64, seed: u64 },               // expert popularity ~ rank^-s, deterministic permutation
    Histogram { per_layer: bool, loads: Vec<Vec<f64>> }, // measured fractions, sums to 1 per layer
    WorstCase,                                // all T·k assignments to min(E, T·k) experts, greedy
}
```

Derived per MoE layer: `mean_load = T·k/E`, `max_load`, and with capacity `C = ceil(cf·T·k/E)`:
`executed rows per expert = C` (padded layout) or `t_e` (ragged), `dropped = Σ_e max(0, t_e − C)` when
`drop_overflow`. Useful FLOPs use **kept** assignments `Σ_e min(t_e, C)` (or `T·k` dropless). The mapper uses
per-expert loads; under expert parallelism the slowest rank sets the time, so imbalance matters (§9.6).

### 5.7 Quantization and MX scale handling

- A contraction may take operands of any `ElemType`. If 01 declares a native datapath mode for the operand pair
  (e.g. `mxfp4 × mxfp8 → fp32`), the mapper uses it and scale handling is part of the datapath. If not, and every
  operand has `upcast_ok: true` (the default), the mapper runs it on the cheapest wider mode and inserts a
  `dequantize` (convert class) as an executed, non-useful cost. This is the only case where the mapper inserts
  compute ops. With `upcast_ok: false` (used when the experiment is about native low-precision throughput), no
  matching mode is a mapping error naming the tensor. MFU for an upcast kernel is still priced at the peak of
  the *declared* operand dtypes, so emulation shows up as low MFU, not as a flattering one.
- Activations quantized at runtime appear as explicit `quantize` nodes (e.g. MXFP8 activations before an
  MXFP8 GEMM). `amax_from: dynamic` costs a block max-reduction; `calibrated` costs only the cast.
- Scales are not separate graph tensors: they are part of the tensor's storage (`Scaling`), and kernels
  access them via `FloorDiv` index maps, so bytes and access patterns are exact.
- KV cache dtype is independent (fp8 KV cache: `kv_cache` tensors with `fp8_e4m3_pt`; `kv_append` includes
  the quantize cost).

### 5.8 Layout and view ops

`layout` ops are free unless the mapper materializes them. `split` of a fused qkv output into q/k/v and
`reshape` of `[T, H·Dh]` ↔ `[T, H, Dh]` are always views on row-major data (validated as such; no bytes).
`transpose` of a pinned-layout tensor in its home memory is a real copy.

### 5.9 Scalar function body table

| fn | body per element |
|---|---|
| add, sub | add 1 |
| mul, scale | mul 1 |
| silu | exp 1, add 1, rcp 1, mul 1 |
| sigmoid | exp 1, add 1, rcp 1 |
| gelu_tanh | tanh 1, mul 4, add 2 (fma counted as mul+add) |
| gelu_erf | erf 1, mul 2, add 1 |
| relu | max 1 |
| exp | exp 1 |
| cast | cvt 1 |

Hardware (01) maps each scalar class to unit costs; a fused `silu·mul` is one map kernel with summed bodies.

## 6. Collectives

```rust
pub struct CollectiveAttrs {
    pub kind: CollKind,                       // all_reduce | all_gather | reduce_scatter | all_to_all
                                              // | all_to_all_v | broadcast | reduce
    pub group: Group,
    #[serde(default)] pub reduce: Option<ReduceOp>,    // sum | max | min | avg
    #[serde(default)] pub axis: Option<i32>,           // concat/scatter axis for AG/RS/A2A
    #[serde(default)] pub root: Option<u32>,           // broadcast/reduce
    #[serde(default)] pub algo_hint: Option<String>,   // "ring" | "tree" | "2d" ..., mapper may ignore
    #[serde(default)] pub ragged: Option<RaggedSpec>,  // all_to_all_v sizes from routing stats
}
pub enum Group {
    MeshAxes(Vec<AxisName>),       // ranks differing only in these mesh coordinates form one group
    Explicit(Vec<Vec<u32>>),       // imported graphs (torch process groups); ranks = flat mesh index
}
pub struct SendRecvAttrs { pub peer: Peer, pub tag: u32 }  // op: send_recv; input sent, output received
pub enum Peer { Shift { axis: AxisName, delta: i32 }, Rank(u32) }
```

Per-rank shapes are the node's input/output tensor shapes (local shards). Let n = group size and S = bytes of
the **full logical** tensor (AR: the per-rank buffer; AG: output; RS: input; A2A: per-rank buffer).

| Kind | in (per rank) | out (per rank) | Ideal per-rank send bytes | nccl-tests busbw factor |
|---|---|---|---|---|
| all_reduce | S | S | 2(n−1)/n · S | 2(n−1)/n |
| all_gather | S/n | S | (n−1)/n · S | (n−1)/n |
| reduce_scatter | S | S/n | (n−1)/n · S | (n−1)/n |
| all_to_all | S | S | (n−1)/n · S | (n−1)/n |
| all_to_all_v | Σ sends | Σ recvs | from `ragged` | – |
| broadcast | S (root) | S | S (root) | 1 |
| send_recv | S | S | S | – |

"Ideal" bytes are a floor (checked per 00), not the algorithm: algorithm choice, topology, chunking and
overlap belong to 03. Reduction arithmetic (`(n−1)/n · S/elem` adds per rank) is counted as vector ops.
Collectives are `KernelClass::Collective` with no FLOPs in MFU.

## 7. Graph

### 7.1 Structure

```rust
pub struct Graph {
    pub params: Vec<Id>,            // ordered inputs (activations/inputs passed in, or model-level tensor refs)
    pub results: Vec<Id>,
    pub tensors: IndexMap<Id, TensorDecl>,   // graph-local tensors (activations)
    pub nodes: Vec<Node>,           // any order in the file; canonical order = topo sort, ties by id
    #[serde(default)] pub regions: IndexMap<Id, Region>,  // named node sets: fusion groups, stages
}
pub struct Region { pub nodes: Vec<Id>, pub kind: RegionKind } // fusion | pipeline_stage | user
```

Tensor references resolve in order: graph-local, then graph params, then model-level tensors.

### 7.2 Invariants (validated at load, structured errors)

1. Acyclic per graph; the only iteration is `repeat` (`E-WL-DAG-001` with the cycle path).
2. SSA: every non-param tensor has exactly one producer; in-place versions use `alias_of` (§3.6).
3. Shapes/dtypes: `infer` of every node matches declared output decls (`E-WL-SHAPE-001`, with expected vs found).
4. Ids unique in scope; dotted paths for nested references: `layers.body.attn` (repeat id, body, node).
5. All model-level tensors referenced by at least one graph (unused: warning in report).
6. No dangling results; every graph result produced.

### 7.3 Repetition (`repeat`)

```rust
pub struct RepeatAttrs {
    pub body: Id,                    // graph id
    pub count: DimExprSrc,           // e.g. "L"
    pub carry: Vec<Carry>,           // loop-carried: {init, param, yield, out}
    pub stacked: Vec<Stacked>,       // per-iteration slice: {outer: model tensor with stack=count, param}
    #[serde(default)] pub broadcast: Vec<(Id, Id)>,  // same tensor every iteration (outer, param), e.g. rope table
}
```

Semantics: iteration i binds each stacked `param` to slice i of `outer`, carried params to the previous
iteration's `yield` (or `init` for i = 0), runs `body`, and after the last iteration exposes `out`.
In-place stacked tensors (KV cache) yield their `alias_of` version back into slice i.

**Steady-state contract**: iterations of a repeat are identical except for the slice index. Engines (03)
may evaluate one or a few iterations and extrapolate `count`; the IR guarantees nothing in the body depends on
the iteration index. Heterogeneous stacks are expressed as consecutive repeats (DeepSeek: 3 dense + 58 MoE)
or a body containing multiple sub-layers (alternating local/global attention: body = 2 layers, count L/2).
The partition pass splits repeats at pipeline-stage boundaries (§9.5).

`call` invokes a graph once (no stacking); used for shared sub-graphs (MTP head, draft model).

### 7.4 Fusion and scheduling hints

```rust
#[derive(Default)]
pub struct Hints {
    pub fuse: FusePolicy,            // auto (default) | prefer | avoid | barrier
    pub fuse_group: Option<Id>,      // nodes sharing a group: mapper should try fusing them as one
    pub impl_hint: Option<String>,   // op-specific, e.g. attention "fused"
    pub overlap: Option<Overlap>,    // for collectives: allow | require_exposed (for ablations)
}
```

Hints never change semantics. `prefer`/`avoid` bias the mapper's search; `barrier` is binding: outputs are
written to the tensor's home memory (the outermost memory level holding activations, per 01) before any
consumer starts, and no fusion across it. The scenario-level `eval_mode: isolated` (§12.6) applies `barrier`
to every node, reproducing per-op measurements.

**Collective asynchrony** (03 §7 credits overlap only when 02 declares it). Every collective is asynchronous
with respect to nodes that do not depend on its output: it may overlap any node not on a data path to or from
it. This follows from the DAG and needs no flag. `overlap: allow` (default) additionally permits 03's
decomposition rewrites (chunking a collective so its pieces overlap the producing or consuming einsum, §9.1).
`overlap: require_exposed` forbids overlap of that collective with anything (for ablations and for matching
measured runs that did not overlap).

### 7.4.1 Software-stack kernel recipes (`kiln.stack/1`, 08 §F)

The workload graph says what is computed; a **stack recipe** says how the software that runs a step splits each
node into device kernels. It is part of the execution model (03 §4.4), not of the workload: the hash of a
workload never includes it, and candidate and baseline are scored under the same recipe. Recipes live in
`kiln/stacks/*.json5` (`kiln-wl::stack`); built-ins `pytorch_cuda_graph_sdpa` (default for `host_launched`),
`xla_tpu_fused` (default for `static_dataflow`) and `kiln_ideal` (default for `device_queued`, and what a
novel design's compiler is assumed to achieve). `SimOptions.stack` overrides the default; every result records
the recipe as `provenance.flags.stack = <id>@stk1-<hash>`.

- A rule matches a node by `op` (§5 name), optional `roles` (§7.5) and `when` (`tokens_min/max`, the leading
  dim of input 0; `tokens_per_seq_min/max`, divided by the slots of the first KV-cache input); first match wins.
  A node without a matching rule is one kernel.
- `kernels` lists the kernels in issue order. Exactly one is `primary`: the kernel the mapped task graph
  already models (its compute and traffic). Every other kernel (`count` copies, `kind: kernel | memset`) is
  charged on top with its own `reads` / `writes`: `{of: "in<i>" | "out<i>", dtype?, frac?, rows?}` gives
  `elements x bytes(dtype or the tensor's own) x frac`, with `rows` keeping one element per row (a last-dim
  reduction). A kernel with no accesses is launch-only (its bytes are in the primary's traffic).
- `onchip_fraction`: an extra kernel whose largest access is at most this fraction of the shared on-chip level
  is served there, otherwise off chip (same rule as activation homes, 03 §3.4). Isolated programs keep all of
  it off chip.
- `fuse_elementwise` (default false): the compiler fuses elementwise nodes (no contraction, collective or opaque
  op: norms, RoPE, activations, residual adds, cache appends) into the adjacent contraction's fusion. The mapper
  then groups each with the contraction before it (or after it, when it opens an iteration) and edges between
  the fused nodes stream (03 §3.6). Set on `xla_tpu_fused`; whole steps only.
- Structure only: recipes carry no times. Library heuristics that change the kernel sequence with shape
  (flash split-KV, cuBLAS split-K) are rules whose conditions bracket the measured points.

### 7.5 Role vocabulary

`role` is a dotted tag from a closed vocabulary (extensible by schema version):
`embed, attn.norm, attn.qkv, attn.q, attn.k, attn.v, attn.q_down, attn.q_up, attn.kv_down, attn.rope,
attn.kv_append, attn.core, attn.o, mlp.norm, mlp.gate_up, mlp.gate, mlp.up, mlp.act, mlp.down, moe.router,
moe.route, moe.dispatch, moe.expert.gate_up, moe.expert.act, moe.expert.down, moe.combine, moe.shared.gate_up,
moe.shared.act, moe.shared.down, residual, final_norm, logits_select, lm_head, sample`. Roles drive parallelism templates (§9.3), calibration key derivation
(§13.2), harness mapping (§13.1) and visualizer grouping (05). Frontends infer roles by pattern (§11.1) and
leave them empty when unsure; nothing requires roles except templates.

## 8. Phases and scenarios

### 8.1 Sequence batch

```rust
pub struct SeqBatch { pub segments: Vec<Segment> }
pub struct Segment { pub count: u64, pub q_len: u64, pub kv_len: u64, #[serde(default)] pub tag: Option<String> }
```

Invariants: `count ≥ 1`, `q_len ≥ 1`, `kv_len ≥ q_len` (kv includes this step's tokens). Canonical order:
sorted by `(q_len, kv_len, tag)` with equal entries merged (sum counts). `T = Σ count·q_len`, `N = Σ count`.

| Step kind | Segments |
|---|---|
| prefill, batch B, prompt S, no prefix cache | `[{B, S, S}]` |
| prefill with cached prefix of length p | `[{B, S−p, S}]` |
| decode, batch B, attending kv | `[{B, 1, kv}]` |
| chunked-prefill mixed step | `[{1, c, p+c}, {B_d, 1, kv_i}...]` |
| speculative verify of γ draft tokens | `[{B, γ+1, kv+γ}]` with `logits_select{which: all}` |

### 8.2 Phase instances

```rust
pub struct PhaseInstance {
    pub kind: PhaseKind,           // prefill | decode | mixed | custom
    pub entry: Id,                 // graph id (forward)
    pub seqs: SeqBatch,
    pub bindings: IndexMap<SymName, u64>,
    pub multiplicity: Rational,    // how many times this instance occurs in the scenario (for aggregation)
    pub index: Option<u64>,        // decode step index i, for ITL series
}
```

### 8.3 Scenarios

```rust
pub struct Scenario {
    pub mode: ScenarioMode,
    #[serde(default)] pub bindings: IndexMap<SymName, u64>,   // global overrides (e.g. L for a 2-layer smoke run)
    #[serde(default)] pub routing: Option<RoutingModel>,
    #[serde(default)] pub eval_mode: EvalMode,                // whole_graph (default) | isolated{cache, launch}
    #[serde(default)] pub mfu_ref_dtype: Option<ElemType>,    // default bf16
    #[serde(default)] pub host: HostModel,                    // per-step host overhead s (default 0), §12
    #[serde(default)] pub scope: ScoringScope,                // step (default) | layer, §12.5
}

pub enum ScenarioMode {
    /// One step, what the harness measured. No TTFT/TPOT, just step time + per-op breakdown.
    Snapshot { kind: PhaseKind, seqs: SeqBatch },
    /// Static batch: B requests arrive at t=0, prefill together, decode together.
    Static { batch: u64, prompt_len: u64, gen_len: u64,        // gen_len counts the first token
             #[serde(default)] prefix_cached: u64,
             #[serde(default)] decode_sampling: DecodeSampling },
    /// Continuous batching serving simulation (deterministic, seeded).
    Continuous(ContinuousSpec),
}

pub enum DecodeSampling { Exact, Points { n: u32, max_rel_err: f64 } } // default Points{n:3, max_rel_err:0.005}

pub struct ContinuousSpec {
    pub classes: Vec<RequestClass>,                // {weight, prompt: LenDist, output: LenDist, prefix_cached: LenDist}
    pub arrival: Arrival,                          // closed_loop{concurrency} | poisson{rate_per_s} | all_at_zero
    pub n_requests: u64,
    pub max_num_seqs: u64, pub max_batched_tokens: u64,
    pub chunked_prefill: bool,
    pub policy: SchedPolicy,                       // vllm_v1 (default): decode-first, then FCFS prefill chunks
    pub kv_bucket: u64,                            // kv_len quantization for step-time memoization, default 64
    pub warmup_requests: u64, pub seed: u64,
}
pub enum LenDist { Fixed(u64), Uniform { lo: u64, hi: u64 }, Normal { mean: f64, std: f64, lo: u64, hi: u64 },
                   Empirical { name: String, hash: String, values: Vec<u64> } }
```

### 8.4 Scenario expansion (`kiln-wl::expand`)

| Mode | Instances | Aggregation |
|---|---|---|
| Snapshot | 1 | step time |
| Static | prefill `[{B, S−p, S}]` ×1; decode steps i = 1..G−1 with `[{B, 1, S+i}]` | §12 formulas |
| Continuous | produced online by the scheduler (below) | per-request timelines |

`DecodeSampling::Points{n}`: the engine evaluates decode steps at n kv lengths spread evenly over
`[S+1, S+G−1]` (always including both ends), fits a piecewise-linear model, and integrates. It then evaluates
one extra midpoint; if its residual exceeds `max_rel_err`, it doubles n until it holds or n reaches `G−1`
(exact). The final n and residual go in the result provenance. `Exact` evaluates every step (dedup by
binding hash).

Continuous mode: the scheduler is a deterministic discrete-event loop in `kiln-wl::serve` that forms each
step's `SeqBatch` under the policy and capacity limits (max_num_seqs, max_batched_tokens, KV capacity from
the design's memory after weights, per 03's capacity query), asks the engine for `step_time(SeqBatch)`
(memoized on the batch with kv_len rounded up to `kv_bucket`), and advances time. Preemption is not modeled
in v0: admission waits for KV capacity. KV capacity exhaustion at admission of a single request is an error
`E-WL-SCN-003` (request cannot fit).

## 9. Parallelism

### 9.1 Who produces collectives (decision D5)

Three layers:

1. **IR (this section)** defines collective op types (§6) and accepts explicit collectives in imported
   graphs (already-parallel torch programs, Chakra traces).
2. **Partition pass** (`kiln-wl::partition`, hardware-agnostic, deterministic, pure function of
   `(BoundModel, ParallelPlan)`) assigns shardings, propagates them, inserts the collectives required by
   sharding algebra, splits pipeline stages, and emits a `PartitionedProgram`.
3. **Mapper (03)** chooses the `ParallelPlan` (or validates a given one), maps logical mesh coordinates to
   physical chips, chooses collective algorithms and chunking, schedules overlap, and may apply
   **semantics-preserving rewrites** only: AR → RS + AG, hierarchical decomposition across mesh levels,
   collective-matmul fusion (AG/RS overlapped with the adjacent einsum), and merging of adjacent
   collectives over the same group. It never adds or removes communication volume except through these.

Justification: (a) what must be communicated is fully determined by shardings (GSPMD), independent of
hardware, so it is checkable without a design (FLOP conservation: Σ over ranks of useful FLOPs equals logical
FLOPs; Megatron TP must yield exactly 2 all-reduces per layer forward); (b) plan search in 03 calls
`partition` many times, so it must be cheap and shared, not reimplemented; (c) imported parallel programs
and kiln-partitioned ones share one output type; (d) collective *cost* is entirely hardware-dependent and
stays in 03. Cost of this choice: novel parallelisms must be expressible as shardings over mesh axes;
PartitionSpec over arbitrary named axes covers TP, 2D-TP, SP, CP, DP, EP (including EP over a product
of axes), which is sufficient for v0.

### 9.2 Mesh and plan

```rust
pub struct ParallelPlan {
    pub mesh: IndexMap<AxisName, u32>,           // logical axes, e.g. {"dp":2, "pp":2, "tp":4}; product = ranks used
    #[serde(default)] pub template: Option<Template>,  // §9.3
    #[serde(default)] pub shardings: Vec<ShardAnnot>,  // explicit, override template
    #[serde(default)] pub pipeline: Option<PipelinePlan>,
    #[serde(default)] pub allow_uneven: bool,    // pad uneven splits (padding = executed, not useful)
}
pub struct ShardAnnot { pub tensor: TensorPath, pub spec: ShardSpec }
/// One entry per logical dim: replicated (empty) or sharded over one or more mesh axes (major→minor).
pub struct ShardSpec { pub dims: Vec<Vec<AxisName>>, #[serde(default)] pub partial: Vec<AxisName> }
```

`partial: [ax]` marks a tensor as holding partial sums over `ax` (pending reduction). Mesh axes not mentioned
in a spec mean replication. The mapping of the logical mesh onto chips is 03's choice, constrained by 01's
topology (e.g. keep `tp` within a package).

### 9.3 Templates

Templates expand into `ShardAnnot`s on weights/caches by role, then propagation fills activations.

```rust
pub struct Template {
    pub tp: Option<AxisName>, pub sp: bool,       // Megatron TP (+ sequence parallel over the same axis)
    pub dp: Option<AxisName>,
    pub pp: Option<AxisName>,
    pub ep: Option<Vec<AxisName>>,                // expert dim sharded over these axes (may reuse dp/tp)
    pub cp: Option<AxisName>,                     // context parallel (ring attention) over sequence
    pub vocab_parallel: bool,                     // shard embedding/lm_head over tp
    pub attn_dp: bool,                            // DeepSeek-style: attention data-parallel, MoE expert-parallel
}
```

Megatron TP rules (axis `tp`, size n): `attn.qkv`, `mlp.gate_up` (and `gate`, `up`): shard output dim
(heads must divide: H % n = 0 and Hkv % n = 0, else KV heads are replicated when n > Hkv, recorded as
executed-not-useful duplication); `attn.o`, `mlp.down`: shard input dim → output `partial[tp]`; norms,
residuals replicated (or sharded over sequence when `sp`); `lm_head`/`embed` vocab-sharded when
`vocab_parallel`. KV cache sharded over KV heads like `attn.k`.

### 9.4 Propagation and collective insertion

Each op provides a `ShardRule`: given input shardings, the output sharding (or a set of admissible
alternatives with resharding costs left to 03). Propagation runs producer-to-consumer then
consumer-to-producer to fixpoint over the topologically ordered graph (deterministic: ties broken by node id, then preferring the producer's sharding).
At each edge where the producer's spec differs from the consumer's required spec, the pass inserts:

| Producer → required | Inserted collective (over axes `A`) |
|---|---|
| `partial[A]` → replicated | `all_reduce(A)` |
| `partial[A]` → sharded dim i over A | `reduce_scatter(A, axis=i)` |
| sharded dim i over A → replicated | `all_gather(A, axis=i)` |
| sharded dim i over A → sharded dim j over A | `all_to_all(A, i→j)` |
| replicated → sharded | local slice (no communication) |
| different mesh axes | composition of the above, minimal by a fixed rule order (reduce first, then gather, then a2a) |

Einsum rule: a contraction letter sharded on both operands over the same axes yields `partial` output; a
letter sharded on only one operand forces gathering it on that operand (the pass picks the operand with fewer
bytes to gather, ties by input order). Attention: heads shardable (q and KV heads together); sequence
shardable only under `cp` (ring attention: inserts `send_recv` ring over `cp` of K/V blocks, `cp−1` steps,
each `kv_len/cp` rows; causal load balancing by zig-zag chunk assignment).

### 9.5 Pipeline parallelism

```rust
pub struct PipelinePlan {
    pub axis: AxisName,
    pub split: StageSplit,                       // even | explicit{layers_per_stage: Vec<u32>} | balanced (by cost hint FLOPs + weight bytes)
    pub microbatches: u32,
    pub schedule: PpSchedule,                    // gpipe | interleaved{virtual: u32} | inference_rr
    #[serde(default)] pub embed_stage: u32, #[serde(default)] pub head_stage: Option<u32>, // default last
}
```

The pass splits `repeat` nodes at stage boundaries into per-stage repeats with reduced counts, inserts
`send_recv(Shift{pp, +1})` of the carried activation, and splits each phase instance's `SeqBatch` into
`microbatches` sub-batches (by sequence; error `E-WL-PP-002` if N < microbatches). Scheduling timelines (bubbles, interleaving) are computed by 03/kiln-sim from the schedule
enum; the IR fixes only stage contents and the schedule name.

### 9.6 Expert parallelism

With `ep` over axes A (size n): experts sharded `E/n` per rank (`E % n = 0` required); `moe_dispatch`
and `moe_combine` become `all_to_all_v(A)` with ragged sizes from the routing model: rank r sends to rank r'
the assignments of its local tokens routed to experts on r'. Under `Uniform`, per-rank send bytes =
`(T_local·k)·(n−1)/n·d·elem`. Per-rank expert load = Σ of its experts' `t_e`; the partitioned program records
per-rank loads so 03 can time the slowest rank. `attn_dp`: attention is data-parallel over A (each rank holds
different sequences, full attention weights), so `T_local = T/n` entering the MoE.

### 9.7 Output type (interface to 03)

```rust
pub struct PartitionedProgram {
    pub plan_hash: Hash, pub mesh: IndexMap<AxisName, u32>,
    pub stages: Vec<StageProgram>,          // len = pp size (1 if no PP)
    pub pipeline: Option<PipelinePlan>,
    pub per_rank_variation: Vec<RankVariation>, // e.g. MoE expert loads per rank; empty when SPMD-uniform
}
pub struct StageProgram {
    pub stage: u32,
    pub graph: BoundGraph,                  // local shapes, collectives inserted, repeats kept
    pub layers: std::ops::Range<u32>,
    pub resident_bytes_per_rank: ResidentBreakdown, // weights, kv, constants, after sharding
}
```

Functions `kiln-wl` exposes to 03 (all deterministic, no hardware input):

| Function | Purpose | Target cost |
|---|---|---|
| `bind(&Model, &Bindings) -> BoundModel` | symbols to integers | < 0.2 ms per layer body |
| `lower(&BoundGraph) -> KernelGraph` | kernels per node, cached per node hash | < 0.5 ms per layer body |
| `partition(&BoundModel, &ParallelPlan) -> PartitionedProgram` | shardings + collectives | < 1 ms per layer body |
| `expand(&Scenario, &Model) -> Vec<PhaseInstance>` | scenario to instances | < 1 ms (Static/Snapshot) |
| `serve::Scheduler` | continuous batching loop, calls back `step_time` | O(steps) |
| `stats(&KernelGraph) -> GraphStats` | useful FLOPs, bytes, resident set for floors/metrics | < 0.2 ms |

These budgets keep the workload layer under 5% of Tier A's 50 ms per (design, layer).

## 10. Training (out of scope)

Training is out of scope (owner ruling, 08 §C): no backward, optimizer, recompute, sharded-optimizer or train
scenario is specified or implemented. The IR stays extensible to training later through schema-version
additions (op variants, tensor classes, a phase kind and a scenario mode).

## 11. Frontends, canonicalization, hashing

### 11.1 torch.export / FX (Python, `kiln-py` package `kiln.frontend`)

Entry point: `kiln.frontend.export(module, example_args, dynamic_shapes, *, roles="infer", allow_opaque=False,
weights_dtype=None) -> dict` (canonical JSON object; never touches the network or the GPU; runs on meta tensors
via `torch.device("meta")` / FakeTensor so it is cheap on the user's laptop).

Pipeline:
1. `torch.export.export(module, args, dynamic_shapes=..., strict=False)`.
2. `ep.run_decompositions(table)` with a table that **keeps** `aten.linear`, `aten.mm/bmm/addmm`,
   `aten.scaled_dot_product_attention` and its flash/efficient/cudnn variants, `aten.rms_norm`,
   `aten.layer_norm`, `aten.embedding`, `aten._softmax`, `aten.topk`, `_c10d_functional.*` collectives;
   decomposes the rest to Core ATen.
3. Walk nodes and map:

| ATen | kiln |
|---|---|
| linear / mm / addmm / bmm / matmul | `einsum` (bias → `map add`) |
| scaled_dot_product_attention (+ variants) | `attention` (is_causal → `causal`; attn_mask → `explicit` with density measured on the example input or 1.0; `enable_gqa` or detected `repeat_kv` expand-reshape before SDPA → GQA fold) |
| pow·mean·rsqrt·mul chain (HF LlamaRMSNorm), rms_norm | `rms_norm` (pattern match) |
| rotate_half pattern (cat of negated halves, mul cos/sin, add) | `rope` |
| silu(a)·b | `gated_act` |
| embedding | `embedding` |
| index_copy / index_put / slice_scatter into a buffer | `kv_append` (buffer marked `kv_cache`; HF `StaticCache` is the supported export path) |
| softmax, topk, sort, argmax, multinomial | `softmax`, `top_k`, `sample` |
| view/reshape/permute/transpose/expand/cat/split/slice | `layout` |
| elementwise | `map` |
| `_c10d_functional.all_reduce/all_gather_into_tensor/reduce_scatter_tensor/all_to_all_single` | `collective` with `Group::Explicit` |
| anything else | error `E-WL-FE-001` listing op, source location and the nearest supported op; with `allow_opaque`, `opaque` with FLOPs from `torch.utils.flop_counter` and bytes from shapes |

4. Symbolic dims: `torch.export.Dim("seq", min, max)` becomes a kiln symbol of the same name with min/max.
   The frontend then rewrites token-count dims to `T` and adds `seqs` when it can identify the attention
   pattern; otherwise symbols stay as exported and the scenario binds them by name.
5. Parameters → model-level `weight` tensors (dtype from the parameter or `weights_dtype` override; quantized
   checkpoints via torchao tensor subclasses map to `ElemType` scalings). Buffers → `constant` or `kv_cache`.
6. Repeat detection: consecutive structurally identical subgraphs (same node signature modulo parameter
   identity) whose parameters differ only by layer index are folded into one `repeat` with stacked weights.
   If folding fails the graph is emitted flat (valid, just large).
7. Roles: inferred from module paths (`model.layers.*.self_attn.q_proj` → `attn.q`, etc.) via a per-family
   regex table (llama, mixtral, qwen2/3, deepseek_v3, gpt-neox); unmatched → no role.

FX fallback (`torch.fx.symbolic_trace` + ShapeProp) uses the same node mapper for models export cannot
handle.

### 11.2 ONNX import (Rust, `kiln-wl::onnx`)

`prost`-generated types from `onnx.proto3`; supports opsets 17–23 and the `com.microsoft` domain subset below.
`dim_param` strings become symbols; initializers become `weight` tensors (raw data not loaded, only shapes and
dtypes; external data never read).

| ONNX | kiln |
|---|---|
| MatMul, Gemm, MatMulNBits (ms) | `einsum` (MatMulNBits → `int4_g{block}` weight) |
| Attention (opset 23), GroupQueryAttention / MultiHeadAttention (ms) | `attention` |
| RMSNormalization (23), SimplifiedLayerNormalization / SkipSimplifiedLayerNormalization (ms), LayerNormalization | `rms_norm` / `layer_norm` (Skip* → `fused_residual`) |
| RotaryEmbedding (23 and ms) | `rope` |
| Softmax, TopK, Gather, ScatterND, elementwise, Reshape/Transpose/Slice/Concat/Split | as torch |
| QuantizeLinear / DequantizeLinear (block_size, opset 21+) | `quantize` / `dequantize`, folding DQ→MatMul into a quantized weight `ElemType` |
| other | `E-WL-FE-001` / `opaque` |

Opset-23 op availability to be re-verified against the ONNX changelog at implementation time (open Q8).

### 11.3 Model zoo (Rust, `kiln-wl::zoo`)

```rust
pub struct TransformerConfig {
    pub family: Family,                     // llama | mixtral | deepseek_v3 | custom
    pub d_model: u32, pub n_layers: u32, pub vocab: u32,
    pub attn: AttnConfig,                   // Gqa{heads, kv_heads, head_dim} | Mla{..MlaAttrs dims..}
    pub mlp: MlpConfig,                     // Dense{d_ff, act} | Moe{n_experts, top_k, d_ff_expert, n_shared, d_ff_shared,
                                            //   scoring, norm_topk, softmax_after_topk, group_limited, capacity_factor}
    pub first_dense_layers: u32,            // DeepSeek first_k_dense_replace
    pub dense_d_ff: Option<u32>,
    pub norm: NormKind, pub norm_eps: f64, pub rope: RopeConfig, pub tie_embeddings: bool,
    pub mask: Mask,                         // causal default; sliding window for some families
    pub dtypes: DtypeConfig,                // weights, activations, kv_cache, accum, lm_head; per-role overrides
    pub fuse_qkv: bool, pub fuse_gate_up: bool, // default true
    pub mtp_layers: u32,                    // DeepSeek multi-token prediction heads (call graph); default 0
    pub harness_compat: bool,               // §13.1
}
```

Presets (values from the public HF configs; tests pin parameter counts):

| Preset | d | L | H / Hkv / Dh | FFN | Vocab | Other | Params (check) |
|---|---|---|---|---|---|---|---|
| `llama3_8b` | 4096 | 32 | 32 / 8 / 128 | 14336 SwiGLU | 128256 | θ=500000, eps 1e-5, untied | 8,030,261,248 |
| `llama3_70b` | 8192 | 80 | 64 / 8 / 128 | 28672 | 128256 | θ=500000 | ≈70.55e9 |
| `mixtral_8x7b` | 4096 | 32 | 32 / 8 / 128 | 8 experts × 14336, top-2, softmax after top-k | 32000 | θ=1e6 | ≈46.7e9 |
| `deepseek_v3` | 7168 | 61 | MLA: 128 heads, q_lora 1536, kv_lora 512, nope 128, rope 64, v 128 | 3 dense (18432), then 256 routed × 2048 + 1 shared, top-8, sigmoid, group-limited 8/4, scaling 2.5 | 129280 | MTP 1 | ≈671e9 (+MTP) |

Sugar in `WorkloadDoc.model`: `{"zoo": {"preset": "llama3_8b", "overrides": {"n_layers": 2}}}` expands at
load to the full `Model`; the expanded form (not the sugar) is what gets hashed, so a zoo change that alters
the graph changes the hash.

### 11.4 Built-in workload sets

Named `"<preset>:<scenario>"` (the form 06's `Session.evaluate` accepts), generated by `kiln-wl::zoo::suite(name)`.

| Set | Members | Use |
|---|---|---|
| `legacy` | `llama3_8b:{prefill_b1, decode_b1, decode_b8, decode_b32}` with `harness_compat: true`, `eval_mode: isolated{cold}`, Snapshot scenarios (prefill seq 2048; decode kv 2048) | Identical shapes, counts, FLOPs and bytes to `harness/workloads.py` (§13.1); calibration |
| `standard` | same four snapshots, compat off, `whole_graph`, `scope: step` | Default whole-step scoring (§12.5; 06 "standard suite") |
| `evolve` | `standard` + `llama3_8b:static_b8_p1024_g256` + `mixtral_8x7b:{prefill_b1, decode_b32}` (`scope: layer`) + `llama3_8b:tp4_prefill_b1` (§14.3) | Evolution-loop score set |
| `heldout` | `llama3_70b:tp8_prefill_b1`, `mixtral_8x7b:moe_layer_t1024` (§14.2, `scope: layer`), `llama3_8b:decode_b1_kv32768`, `llama3_8b:decode_b128_kv2048` (`scope: layer`), `vit_l16:fwd_b32` (encoder: `mask: none`, `layer_norm`, `gelu_erf`, no KV cache, 1 segment `{32, 197, 197}`) | Never shown to the loop (06 §6.6) |
| `smoke` | `gemm_1024`, `gemm_16_8192` (single `einsum` graphs) | Tests |

Scoring is whole-step (§12.5): `L` keeps the model's depth and 03 evaluates the repeat by steady-state
extrapolation with explicit boundary iterations (03 §4.9). Members marked `scope: layer` are those whose
whole-step resident set exceeds the single-chip baseline's memory (Mixtral, batch-128 decode); they are scored
on one steady-state layer in whole-graph context. The per-layer breakdown reports the steady-state iteration
plus a `1/L_model` share of non-repeated nodes (embedding, final norm, head, sample), so per-layer numbers sum
to the whole step exactly. `vit_l16` is a zoo preset (`family: vit`, d 1024, L 24, 16 heads, MLP 4096, patch
embedding as an einsum over a `layout` im2col view). Random workloads for property tests (06's
`arb_workload`) are generated by `kiln-wl::arb` (proptest strategies over this IR), not `kiln-ir`.

### 11.5 Canonicalization and hashing

Canonical form (computed after load + zoo expansion, before binding):
1. All defaults materialized; shorthand dtypes expanded; DimExprs canonicalized (flattened, sums and products
   sorted by a total order on terms, constants folded, rationals reduced).
2. Nodes in canonical topological order (Kahn's algorithm, ready set ordered by id); tensors and maps sorted by id.
3. `meta` fields and `doc` strings removed.
4. Serialized as JSON per RFC 8785 (JCS): sorted keys, no whitespace, integers as integers, floats in shortest
   round-trip form; floats that are exact integers are rejected where a field is integral.

Hash = sha256 of the canonical bytes (00 decision 9; matches 01 and 06), rendered as the truncated 32-hex sha256 `"wl1-" + hex(sha256)[..32]` (as 01 §17 `hw1-`). Distinct hashes:

| Hash | Over |
|---|---|
| `model_hash` | symbols, graphs, model-level tensors |
| `scenario_hash` | one scenario |
| `plan_hash` | one plan |
| `binding_hash` | a phase instance's `SeqBatch` + bindings |
| `workload_hash` | H(model_hash, scenario_hash, plan_hash or "none") (the value in provenance per 00) |

Node-level `node_hash` (op + attrs + bound input/output types, ids excluded) keys lowering caches and the
mapper's per-node memo, so identical layers in different models share work.

## 12. Metrics semantics

All times are in seconds of simulated device time. Unless stated, metrics are for a static scenario in which
all B requests arrive at t = 0 and the batch is formed instantly. `t_prefill` is the time of the prefill
instance (embedding through `sample`); `t_dec(i)` is decode step i (i = 1..G−1), each including
`sample` and the host term `host.per_step_s`.

### 12.1 Inference (Static)

| Metric | Definition |
|---|---|
| TTFT | `t_prefill` (same for all B requests) |
| ITL series | `[t_dec(1), …, t_dec(G−1)]` |
| TPOT | `Σ_i t_dec(i) / (G−1)`; equals vLLM's `(e2e − TTFT)/(output_tokens − 1)`. Undefined for G = 1. |
| E2E latency | `TTFT + Σ_i t_dec(i)` |
| Output throughput (headline "tokens/s") | `B·G / E2E` |
| Decode throughput | `B / TPOT` |
| Prefill throughput | `B·(S − prefix_cached) / TTFT` |
| Total token throughput | `B·(S − prefix_cached + G) / E2E` |
| Per-chip variants | each throughput ÷ number of chips used by the plan (mesh product) |

With PP for inference, `t_prefill` is the latency of the full batch through all stages (microbatched per the
schedule), and decode steps are timed in steady state with `microbatches` request groups in flight; 03 reports
the per-step critical path. With DP (replicas), throughputs multiply by `dp`, latencies do not change.

### 12.2 Inference (Continuous)

Per request r: `TTFT_r = t_first_token_r − t_arrival_r` (includes queueing and chunked prefill steps);
`TPOT_r = (t_last_r − t_first_r)/(out_r − 1)`; ITL = gaps between consecutive token times of a request.
Reported: mean, median, p90, p99 (nearest-rank on the sorted list, excluding `warmup_requests`), request
throughput, output token throughput `Σ out_r / (t_end − t_start)` over the measured window. These match
vLLM `benchmark_serving.py` field names (`mean_ttft_ms` etc., converted at display).

### 12.3 Snapshot

`step_time` and per-node breakdown only. Tokens/s for a decode snapshot = `N / step_time`, labeled
"decode-step throughput" so it is never confused with output throughput.

### 12.4 Utilization

- **MFU (kiln, default)** = `(Σ_k useful_flops_k / peak(dtype_k)) / (wall_time · n_chips)` summed over
  contraction kernels k, where `peak(dtype_k)` is 01's dense peak FLOP/s per chip for the operand-dtype pair of
  kernel k. Mask-aware (causal halves attention), excludes padding. For a single-dtype model this
  reduces to `useful_flops / (time · n_chips · peak)`.
- **MFU (ref)** = same with all kernels priced at `peak(mfu_ref_dtype)` (default bf16): comparable to
  published numbers that quote "bf16 MFU".
- **MFU (dense, PaLM-style)** = `(2·N_nonembed + 4·L·H·Dh·S)·tokens / (time · n_chips · peak_ref)` with S the
  attended context length (PaLM appendix B forward terms; attention counted dense, not causal-halved). Always
  reported beside the mask-aware headline so published dense numbers compare like for like.
- **HFU** = executed FLOPs (from 03, includes masked partial tiles, padding, MLA mode) / (time ·
  chips · peak).
- **MBU** (decode) = `(weight bytes read + KV bytes read + KV bytes written) per step / (t_dec · Σ chips' outermost
  memory bandwidth)`, with bytes from §4.1 compulsory bytes; reports how close decode is to the HBM floor.

### 12.5 Scoring basis (whole step)

Designs and baselines are scored on whole steps (owner ruling, 08 §C). With `scope: step` (default) every
phase instance (a decode step, a prefill step, a mixed step) is simulated end to end in `whole_graph` mode,
embedding through `sample` over all `L` layers, so inter-op and inter-layer reuse, overlap, fusion and launch
terms are part of the scored time (03 §4.9). `scope: layer` scores one steady-state layer from the same
whole-step evaluation (boundary effects with its neighbours included) plus a `1/L` share of non-repeated nodes;
it is used only where §11.4 says so. `eval_mode: isolated` (§12.6) and per-op results are for calibration and
diagnosis only and never produce a score (06 §6.4). Every time, throughput, power and energy metric above is
reported as an interval (low / central / high, 03 §9.1, 06 §6.3).

### 12.6 Comparison modes for measured per-op data

`eval_mode: isolated { cache: cold | warm, launch: bool }` (calibration only, §12.5) simulates each node
alone: inputs start in the outermost memory, outputs written back, nothing resident between nodes. `launch: true` adds the exposed host
launch term (`t_launch`, 03 §4.4); in-queue terms (`t_min_kernel`, `t_gap`) always apply, matching 06 §2.5's
`overheads_included` for graph modes. Correspondence with measurement timing modes (methods owned by 06 §4.3):

| kiln eval_mode | Measurement field | Bench semantics |
|---|---|---|
| isolated{cold, launch=false} (default for calibration) | `graph_cold` | CUDA graph over rotating operand copies, L2 cold; in-graph launch and gap included, no host launch |
| isolated{warm, launch=false} | `graph_unflushed` | CUDA graph, same operands, L2 warm |
| isolated{warm, launch=true} | `unflushed` | eager launches, warm |
| isolated{cold, launch=true} | `flushed` | eager launches, L2 flushed before each |
| whole_graph, `scope: step` (scoring) | `sequence` | whole step captured as one CUDA graph / one jitted function (06 §2.5, §4.2) |

## 13. Mapping from today's harness

### 13.1 Op-by-op correspondence

`llama3_8b` with `harness_compat: true` sets `fuse_gate_up: false`, attention `mask: none` (dense, as the
harness computes it) and `impl_hint: unfused`, and the scenario uses `eval_mode: isolated{cold}`. In that
configuration kiln must reproduce `calibration/oplist.json`'s `flops` and `min_bytes` for every row exactly
(regression test in 06). Without compat, the differences are intentional and listed.

| Harness op (llm.py) | Harness shape | kiln node (role) | Difference without compat |
|---|---|---|---|
| `qkv` gemm T×6144×4096 ×L | `gemm_{T}_6144_4096` | `einsum "td,nd->tn"` (`attn.qkv`), weight [6144,4096] | Weight layout is nn.Linear `[n,k]`, which is the `_linear` measurement variant |
| `attn_score` bmm (B·Hkv) × (g·q) × kv × 128 | prefill `bmm_1_8192_2048_128` ×256 | `attention` k0 (`attn.core`) | causal mask: useful FLOPs halve (68.7e9 → 34.4e9 per layer at S=2048); softmax and mask now costed |
| `attn_av` | `bmm_1_8192_128_2048` | `attention` k2 | same; fused impl removes S/P round trips |
| `o_proj` T×4096×4096 | `gemm_{T}_4096_4096` | `einsum` (`attn.o`) | none |
| `ffn_gate_up` 2× T×14336×4096 | `gemm_{T}_14336_4096` ×2L | `einsum` T×28672×4096 (`mlp.gate_up`) | fused into one GEMM (compat: two nodes `mlp.gate`, `mlp.up`) |
| `ffn_down` T×4096×14336 | `gemm_{T}_4096_14336` | `einsum` (`mlp.down`) | none |
| `lm_head` B×128256×4096 | `gemm_{B}_128256_4096` | `logits_select{last}` + `einsum` (`lm_head`) | none (harness already used N rows) |
| (absent) | – | `embedding`, 2× `rms_norm`, `rope`, `kv_append`, `gated_act`, 2 residual `map add`, `final_norm`, `sample` | now costed; adds ~T·d·(small) bytes per op, decisive for small-batch decode on designs with weak vector units |
| phase sum `Σ count · t_op` | – | whole-step mapping with fusion/overlap (§12.5) | only isolated mode (calibration) sums nodes |
| `resident_bytes` | params + KV | `stats().resident_set` | adds norms (+0.27 MB) |

The harness prefill folded the batch into `count` (only exact for B = 1) and treated attention per KV head
because Stream could not express the batched form; kiln has no such restriction.

### 13.2 Calibration references

```rust
pub struct CalibRef {
    pub set: String,            // calibration set id, e.g. "a100_2026-10-04" (registry owned by 06)
    pub key: String,            // record key within the set
    #[serde(default)] pub kernel: Option<String>,  // which kernel of the node, e.g. "k0"
    #[serde(default = "graph_cold")] pub timing: String, // measurement field (§12.6)
}
```

Nodes rarely carry `calib` explicitly. Instead `kiln-wl::calib::key_for(kernel) -> Vec<String>` derives
candidate keys from bound shapes, in the harness's naming:

| Kernel shape | Derived key(s) |
|---|---|
| contraction, one operand `weight` in `[k,n]` layout, M rows | `gemm_{M}_{n}_{k}` |
| same with weight `[n,k]` (nn.Linear) | `gemm_{M}_{n}_{k}_linear` (falls back to `gemm_{M}_{n}_{k}`) |
| contraction, no weight operand, batch b | `bmm_{b}_{m}_{n}_{k}` |
| attention decode unfused k0 / k2, GQA folded | `bmm_{N·Hkv}_{g}_{kv}_{Dh}` / `bmm_{N·Hkv}_{g}_{Dh}_{kv}` |
| prefill attention per KV head (B = 1) | `bmm_1_{g·S}_{S}_{Dh}` / `bmm_1_{g·S}_{Dh}_{S}` |

Set formats differ: A100 sets (`calibration/measurements/a100_*.json`) key records by `name` (= key above) with
timing sub-objects `flushed`, `unflushed`, `graph_unflushed`, `graph_cold` (each `median_us`, `min_us`, ...);
TPU sets (`tpuv5e_*.json`) key by `name = "{phase}/{op}"` with `best_s`, `single_median_s`, `pipelined_s`.
06's calibration registry provides per-set adapters (key scheme + timing field → seconds) and an alias table
from `(phase, harness op)` to kiln `(scenario snapshot, role, kernel)`, generated from `oplist.json`, so TPU
records resolve too. GEMM sweep records (`sweep`) resolve by the same `gemm_*` keys. Calibration *use* (fitting
constants, error reporting) is 06's; this section only guarantees every kernel can name its measured
counterpart when one exists.

## 14. Worked examples

### 14.1 Llama-3-8B decode step, batch 8, kv_len 2048

Abbreviated (weights for k/v, gate/up shown fused; repeated decls elided with `…`), precise in everything shown.

```json
{
  "kiln_workload": "0.1",
  "id": "llama3_8b",
  "model": {
    "symbols": {
      "seqs": {"kind": "segments"},
      "T": {"kind": "size"}, "N": {"kind": "size"},
      "L": {"kind": "size", "default": 32},
      "kv_cap": {"kind": "size"}, "slots": {"kind": "size"}
    },
    "entry": {"forward": "fwd"},
    "tensors": {
      "w_embed":   {"shape": [128256, 4096], "dtype": "bf16", "class": "weight"},
      "w_attn_norm": {"shape": [4096], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_qkv":     {"shape": [6144, 4096], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_o":       {"shape": [4096, 4096], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_mlp_norm": {"shape": [4096], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_gate_up": {"shape": [28672, 4096], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_down":    {"shape": [4096, 14336], "dtype": "bf16", "class": "weight", "stack": "L"},
      "w_final_norm": {"shape": [4096], "dtype": "bf16", "class": "weight"},
      "w_lm":      {"shape": [128256, 4096], "dtype": "bf16", "class": "weight"},
      "rope_tab":  {"shape": ["kv_cap", 64, 2], "dtype": "fp32", "class": "constant"},
      "kv_k": {"shape": ["slots", "kv_cap", 8, 128], "dtype": "bf16", "class": "kv_cache", "stack": "L"},
      "kv_v": {"shape": ["slots", "kv_cap", 8, 128], "dtype": "bf16", "class": "kv_cache", "stack": "L"}
    },
    "graphs": {
      "fwd": {
        "params": ["ids", "pos"], "results": ["next"],
        "tensors": {
          "ids": {"shape": ["T"], "dtype": "int32", "class": "input"},
          "pos": {"shape": ["T"], "dtype": "int32", "class": "input"},
          "h0": {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "hL": {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "hf": {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "hs": {"shape": ["N", 4096], "dtype": "bf16", "class": "activation"},
          "logits": {"shape": ["N", 128256], "dtype": "fp32", "class": "activation"},
          "next": {"shape": ["N"], "dtype": "int32", "class": "output"}
        },
        "nodes": [
          {"id": "embed", "op": "embedding", "inputs": ["ids", "w_embed"], "outputs": ["h0"], "role": "embed"},
          {"id": "layers", "op": "repeat", "inputs": ["h0", "pos"], "outputs": ["hL"],
           "body": "block", "count": "L",
           "carry": [{"init": "h0", "param": "x", "yield": "y", "out": "hL"}],
           "broadcast": [["pos", "pos"], ["rope_tab", "rope_tab"]],
           "stacked": [{"outer": "w_attn_norm", "param": "wn1"}, {"outer": "w_qkv", "param": "wqkv"},
                       {"outer": "w_o", "param": "wo"}, {"outer": "w_mlp_norm", "param": "wn2"},
                       {"outer": "w_gate_up", "param": "wgu"}, {"outer": "w_down", "param": "wd"},
                       {"outer": "kv_k", "param": "ck"}, {"outer": "kv_v", "param": "cv"}]},
          {"id": "final_norm", "op": "rms_norm", "eps": 1e-5, "inputs": ["hL", "w_final_norm"], "outputs": ["hf"], "role": "final_norm"},
          {"id": "select", "op": "logits_select", "which": "last", "seqs": "seqs", "inputs": ["hf"], "outputs": ["hs"], "role": "logits_select"},
          {"id": "lm_head", "op": "einsum", "eq": "nd,vd->nv", "accum": "fp32", "inputs": ["hs", "w_lm"], "outputs": ["logits"], "role": "lm_head"},
          {"id": "sample", "op": "sample", "strategy": "greedy", "inputs": ["logits"], "outputs": ["next"], "role": "sample"}
        ]
      },
      "block": {
        "params": ["x", "pos", "rope_tab", "wn1", "wqkv", "wo", "wn2", "wgu", "wd", "ck", "cv"],
        "results": ["y", "ck1", "cv1"],
        "tensors": {
          "xn":  {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "qkv": {"shape": ["T", 6144], "dtype": "bf16", "class": "activation"},
          "q":   {"shape": ["T", 32, 128], "dtype": "bf16", "class": "activation"},
          "k":   {"shape": ["T", 8, 128], "dtype": "bf16", "class": "activation"},
          "v":   {"shape": ["T", 8, 128], "dtype": "bf16", "class": "activation"},
          "qr":  {"shape": ["T", 32, 128], "dtype": "bf16", "class": "activation"},
          "kr":  {"shape": ["T", 8, 128], "dtype": "bf16", "class": "activation"},
          "ck1": {"shape": ["slots", "kv_cap", 8, 128], "dtype": "bf16", "class": "kv_cache", "alias_of": "ck"},
          "cv1": {"shape": ["slots", "kv_cap", 8, 128], "dtype": "bf16", "class": "kv_cache", "alias_of": "cv"},
          "o":   {"shape": ["T", 32, 128], "dtype": "bf16", "class": "activation"},
          "a":   {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "h":   {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "hn":  {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "gu":  {"shape": ["T", 28672], "dtype": "bf16", "class": "activation"},
          "m":   {"shape": ["T", 14336], "dtype": "bf16", "class": "activation"},
          "dn":  {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"},
          "y":   {"shape": ["T", 4096], "dtype": "bf16", "class": "activation"}
        },
        "nodes": [
          {"id": "attn_norm", "op": "rms_norm", "eps": 1e-5, "inputs": ["x", "wn1"], "outputs": ["xn"], "role": "attn.norm"},
          {"id": "qkv", "op": "einsum", "eq": "td,nd->tn", "inputs": ["xn", "wqkv"], "outputs": ["qkv"], "role": "attn.qkv"},
          {"id": "split", "op": "layout", "kind": "split", "axis": 1, "sizes": [4096, 1024, 1024],
           "reshape": [["T", 32, 128], ["T", 8, 128], ["T", 8, 128]], "inputs": ["qkv"], "outputs": ["q", "k", "v"]},
          {"id": "rope", "op": "rope", "theta": 500000.0, "rotary_dim": 128, "style": "half",
           "scaling": {"llama3": {"factor": 8, "low_freq_factor": 1, "high_freq_factor": 4, "original_max_pos": 8192}},
           "inputs": ["q", "k", "pos", "rope_tab"], "outputs": ["qr", "kr"], "role": "attn.rope"},
          {"id": "kv_append", "op": "kv_append", "seqs": "seqs", "inputs": ["ck", "cv", "kr", "v"], "outputs": ["ck1", "cv1"], "role": "attn.kv_append"},
          {"id": "attn", "op": "attention", "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "mask": "causal",
           "seqs": "seqs", "impl_hint": "auto", "inputs": ["qr", "ck1", "cv1"], "outputs": ["o"], "role": "attn.core"},
          {"id": "o_proj", "op": "einsum", "eq": "thd,nhd->tn", "inputs": ["o", "wo"], "outputs": ["a"], "role": "attn.o"},
          {"id": "res1", "op": "map", "fn": "add", "inputs": ["x", "a"], "outputs": ["h"], "role": "residual"},
          {"id": "mlp_norm", "op": "rms_norm", "eps": 1e-5, "inputs": ["h", "wn2"], "outputs": ["hn"], "role": "mlp.norm"},
          {"id": "gate_up", "op": "einsum", "eq": "td,nd->tn", "inputs": ["hn", "wgu"], "outputs": ["gu"], "role": "mlp.gate_up"},
          {"id": "act", "op": "gated_act", "fn": "silu", "layout": "concat_halves", "inputs": ["gu"], "outputs": ["m"], "role": "mlp.act"},
          {"id": "down", "op": "einsum", "eq": "tf,nf->tn", "inputs": ["m", "wd"], "outputs": ["dn"], "role": "mlp.down"},
          {"id": "res2", "op": "map", "fn": "add", "inputs": ["h", "dn"], "outputs": ["y"], "role": "residual"}
        ]
      }
    }
  },
  "scenarios": {
    "decode_b8_kv2048": {"mode": {"snapshot": {"kind": "decode",
                          "seqs": {"segments": [{"count": 8, "q_len": 1, "kv_len": 2048}]}}},
                         "bindings": {"kv_cap": 2048}}
  }
}
```

Note `o_proj` contracts over `(h, d)` with `wo` viewed as `[4096, 32, 128]` (a free view of the stored
`[4096, 4096]`; validation accepts a declared-shape/einsum-shape mismatch only when it is a row-major split of a
trailing dim, `E-WL-EIN-002` otherwise).

Bound values: `T = 8`, `N = 8`, `slots = 8`, `kv_cap = 2048`, past = 2047 per sequence.

Check values (asserted by tests; bf16 = 2 B):

| Quantity | Per layer | × 32 layers + head |
|---|---|---|
| qkv FLOPs `2·8·4096·6144` | 402,653,184 | |
| attention FLOPs `2·8·32·2048·(128+128)` (decode: causal mask is a no-op) | 268,435,456 | |
| o_proj FLOPs | 268,435,456 | |
| gate_up FLOPs `2·8·4096·28672` | 1,879,048,192 | |
| down FLOPs | 939,524,096 | |
| layer useful FLOPs | 3,758,096,384 | 120,259,084,288 |
| lm_head FLOPs `2·8·4096·128256` | | 8,405,385,216 |
| **step useful FLOPs** | | **128,664,469,504** |
| weight bytes | 436,207,616 (+16,384 norms) | 13,959,168,000 + lm_head 1,050,673,152 + final norm 8,192 = 15,009,849,344 |
| KV read `2·8·2048·8·128·2` | 67,108,864 | 2,147,483,648 |
| KV written `2·8·8·128·2` | 32,768 | 1,048,576 |
| softmax points `8·32·2048` | 524,288 | 16,777,216 |
| resident set (weights incl. embed + KV at kv_cap) | | 16,060,522,496 + 2,147,483,648 = 18,208,006,144 |

Compulsory bytes ≈ 17.16e9 B per step (embedding table excluded: only 8 rows read). On an A100 with the
measured 1.427e12 B/s read bandwidth (`peaks.read_sum_1GiB`), the floor is ≈ 12.0 ms per step, i.e.
≈ 665 decode-step tokens/s at batch 8.

### 14.2 One MoE layer (Mixtral-8x7B block, T = 1024 tokens, uniform routing)

Body nodes after `attn` (identical to 14.1 except vocab 32000 and θ = 1e6):

```json
[
 {"id": "mlp_norm", "op": "rms_norm", "eps": 1e-5, "inputs": ["h", "wn2"], "outputs": ["hn"], "role": "mlp.norm"},
 {"id": "router", "op": "einsum", "eq": "td,ed->te", "accum": "fp32", "inputs": ["hn", "w_router"], "outputs": ["rl"], "role": "moe.router"},
 {"id": "route", "op": "moe_route", "n_experts": 8, "top_k": 2, "scoring": "softmax", "softmax_after_topk": true,
  "norm_topk": false, "inputs": ["rl"], "outputs": ["idx", "wts"], "role": "moe.route"},
 {"id": "dispatch", "op": "moe_dispatch", "n_experts": 8, "top_k": 2, "capacity_factor": "5/4",
  "drop_policy": "drop_overflow", "layout": "capacity_padded", "inputs": ["hn", "idx"], "outputs": ["xe"], "role": "moe.dispatch"},
 {"id": "e_gate_up", "op": "grouped_einsum", "eq": "ecd,end->ecn", "inputs": ["xe", "w_e_gu"], "outputs": ["gue"], "role": "moe.expert.gate_up"},
 {"id": "e_act", "op": "gated_act", "fn": "silu", "layout": "concat_halves", "inputs": ["gue"], "outputs": ["me"], "role": "moe.expert.act"},
 {"id": "e_down", "op": "grouped_einsum", "eq": "ecf,enf->ecn", "inputs": ["me", "w_e_down"], "outputs": ["ye"], "role": "moe.expert.down"},
 {"id": "combine", "op": "moe_combine", "inputs": ["ye", "idx", "wts"], "outputs": ["dn"], "role": "moe.combine"},
 {"id": "res2", "op": "map", "fn": "add", "inputs": ["h", "dn"], "outputs": ["y"], "role": "residual"}
]
```

Tensors: `w_router [8,4096]`, `w_e_gu [8, 28672, 4096]` and `w_e_down [8, 4096, 14336]` (both `stack: L`, expert
axis 0), `xe [8, C, 4096]` with `C = ceil(5/4 · T·2/8)`.

With T = 1024, uniform: mean load 256, C = 320. Useful FLOPs: router `2·1024·4096·8` = 67,108,864; experts
`T·k·(2·4096·28672 + 2·14336·4096)` = 2048 × 352,321,536 = 721,554,505,728. Executed (capacity-padded)
expert FLOPs = `8·320·352,321,536` = 901,943,132,160 (HFU only; 1.25×). Expert weight bytes per layer
2,818,572,288. With `Zipf{s:1.0}` the max load exceeds C for the top expert and `dropped > 0` is reported; with
`layout: ragged` executed equals useful.

Under `ep: ["ep"]` with ep = 8 (one expert per chip), attention data-parallel (`attn_dp`), T_local = 128:
`dispatch`/`combine` become `all_to_all_v(ep)`; uniform per-rank send = `128·2·(7/8)·4096·2` = 1,835,008 B each
way; per-rank expert rows = 256 (uniform) and the slowest rank's load is reported in `per_rank_variation`.

### 14.3 Tensor-parallel Llama-3-8B layer on 4 chips (prefill B = 1, S = 2048)

Plan:

```json
{"mesh": {"tp": 4}, "template": {"tp": "tp", "sp": false, "vocab_parallel": true}}
```

Partition output for one `block` iteration (per rank, local shapes):

| Node | Local op | Sharding |
|---|---|---|
| attn_norm | rms_norm [2048,4096] | replicated |
| qkv | einsum [2048,4096]×[1536,4096] → [2048,1536] | w_qkv rows sharded `tp` (8 q heads + 2 k + 2 v heads per rank; zoo orders fused rows head-grouped so the shard is contiguous) |
| rope, kv_append, attn | 8 q heads, 2 kv heads | heads over `tp`; KV cache `[slots, kv_cap, 2, 128]` per rank |
| o_proj | einsum [2048,8,128]×[4096,8,128] → [2048,4096] `partial[tp]` | w_o input dim sharded |
| **ar1** | `collective all_reduce(tp, sum)` [2048,4096] bf16 | inserted: partial → replicated |
| res1, mlp_norm | replicated | |
| gate_up | [2048,4096]×[7168,4096] → [2048,7168] | output dim sharded (gate and up halves each split, so `concat_halves` stays local) |
| act | [2048,7168] → [2048,3584] | |
| down | [2048,3584]×[4096,3584] → [2048,4096] `partial[tp]` | input dim sharded |
| **ar2** | `all_reduce(tp, sum)` [2048,4096] | inserted |
| res2 | replicated | |

Per rank per layer: weight bytes 109,051,904 (= 436,207,616 / 4); useful FLOPs (causal) = (proj FLOPs
`2·2048·4096·(6144+4096+28672+14336)` = 893,353,197,568 + attention 34,376,515,584) / 4 = 231,932,428,288.
Each all-reduce: S = 2048·4096·2 = 16,777,216 B; ideal per-rank send = 2·(3/4)·S = 25,165,824 B; two per layer
(64 per forward). Conservation test: Σ_ranks useful FLOPs = logical FLOPs; collectives inserted = exactly 2 per
layer.

With `sp: true`: norms and residuals run on `[512, 4096]` sequence shards; `ar1`/`ar2` become
`reduce_scatter(tp, axis 0)` after o_proj/down and `all_gather(tp, axis 0)` before qkv/gate_up; total bytes
equal, activation memory for norms/residuals ÷ 4. Vocab-parallel head: `lm_head` local [1, 32064] logits,
`sample` greedy inserts `all_reduce(tp, max)` over a (value, index) pair (8 B per row) instead of gathering logits.

## 15. Validation error codes (workload)

| Code | Meaning | Hint provided |
|---|---|---|
| E-WL-SYM-001/002 | unbound symbol / constraint violated | where used; which scenario field binds it |
| E-WL-DIM-001 | non-integral or negative bound dim | the expression and symbol values |
| E-WL-DT-001..003, W-WL-DT-004 | unknown dtype / bad scaling / blocking axis not contraction axis / block padding | nearest valid shorthand |
| E-WL-DT-005 | a contraction reads a scaled element type with no registry name that no MAC mode runs and none widens into losslessly (its scales would be dropped, §5.7) | a registry scaled type, or a design with a bf16/fp32 mode it widens into |
| E-WL-SHAPE-001 | inferred shape ≠ declared | expected vs found, per dim |
| E-WL-EIN-001/002 | invalid einsum / non-view reshape in einsum | rewrite suggestion |
| E-WL-DAG-001, E-WL-ALIAS-001, E-WL-CLS-001 | cycle / bad alias chain / class misuse | path |
| E-WL-DOM-001 | unsupported domain constraint set | which constraints |
| E-WL-PP-001/002, E-WL-TP-001 | stage split invalid / too few sequences for microbatches / indivisible head or dim sharding | divisors that would work |
| E-WL-SCN-001..003 | invalid scenario / segment invariants / request cannot fit KV | capacity numbers |
| E-WL-FE-001 | unsupported frontend op | `allow_opaque` or nearest op |
| W-WL-OPQ-001 | opaque node present (warning, not error) | – |

`E-WL-*` codes are errors and `W-WL-*` warnings (namespace per 06 §6.5). Errors are the 00 structured form `{code, message, path, hint}` where `path` is the dotted entity path
(`graphs.block.nodes.attn`).

## 16. Cross-section dependencies (assumptions)

**01 hardware IR** (read; consistent with its §6.2-6.3): 02 owns the `upcast_ok` tensor flag and the op to `OpClass` lowering (§4.2). 01 must provide: (a) per-chip dense peak FLOP/s per operand-dtype pair and accumulate dtype
(for MFU and native-datapath checks), including MX/NVFP4/int4 pairs and sparse-capable flags; (b) per-scalar-class
vector/transcendental throughputs (maps `ScalarBody` fields); (c) the memory level that is the "home" of
activations, weights and KV cache, and its capacity and bandwidth (for MBU, isolated mode, capacity); (d) chip
count and inter-chip topology so 03 can map logical mesh axes; (e) a per-launch overhead constant (isolated mode).

**03 mapping engine** consumes `BoundGraph`/`KernelGraph`/`PartitionedProgram` and must: map kernels including
`Segmented` domains and `online_softmax` combiners; report executed FLOPs (HFU) and per-node time; choose MLA mode
and attention impl; implement steady-state extrapolation of `repeat`; time pipeline schedules; time collectives
(algorithms per 07's ASTRA-sim port); answer a KV-capacity query for continuous mode; respect `barrier` and
`eval_mode: isolated`; evaluate whole steps with steady-state extrapolation (§12.5, 03 §4.9). It must not change communication volume beyond §9.1's rewrite list. (`partition` emits SPMD stage graphs with collective nodes already placed; 03 §3.2 step 2 and §7 instantiate them per chip, per 00 decision 5.)

**04 physical model**: none directly; energy per `ScalarBody` class and per byte moved come via 01/03.

**05 visualizer**: uses node ids, dotted paths, roles and repeat iteration index to label trace spans; collective
nodes carry group axes for link highlighting.

**06 validation/API**: workload names and sets it references (`llama3_8b:decode_b8`, standard/evolve/heldout suites) are defined in §11.4; owns the calibration-set registry and adapters (§13.2), the harness-compat regression test
(§13.1), metric reporting, and the Python `evaluate(design, workload, scenario, plan)` surface; result provenance
includes `workload_hash`, `scenario_hash`, `plan_hash`, and decode-sampling n/residual.

**07 prior art**: Chakra ET import (adopted there) is mapped as a fourth frontend: COMP nodes → `opaque` with
their recorded FLOPs/bytes (or named ops when the node name matches an ATen op in §11.1), COMM nodes →
`collective` with `Group::Explicit`. Lower fidelity, flagged in the report.

## 17. Open questions

1. **Collective ownership**: resolved by 00 decision 5 (propagation and insertion stay in `kiln-wl`).
2. **Decode sampling default** (`Points{n:3, 0.5%}`): is piecewise-linear over kv_len safe for designs with
   capacity cliffs (KV spilling from SRAM to HBM mid-generation)? The residual check catches smooth curvature,
   not a step between sample points; maybe force a point at each memory-capacity boundary that 03 reports.
3. **Default timing field**: resolved by owner ruling (08 §C): `graph_cold` per-op for calibration; whole-step
   `sequence` measurements anchor scoring (§12.5).
4. **Default MFU convention**: kiln mask-aware is the headline; PaLM-style shown beside it. Some published
   inference MFU numbers count dense attention; reviewers may expect that.
5. **MoE routing default**: `Uniform` flatters EP designs. Should the default be a mild `Zipf` or a measured
   histogram (needs real Mixtral/DeepSeek routing traces we do not have yet)?
6. **Continuous batching fidelity**: we model a vLLM-v1-like policy without preemption or prefix-cache hits
   across requests. Enough for design ranking, or should prefix caching be in v0?
7. **Speculative decoding / MTP**: segments express verify steps, but draft-model scheduling and acceptance-rate
   modeling are not specified. Add a `Speculative{draft, gamma, accept_rate}` scenario mode?
8. **ONNX opset 23 ops** (Attention, RotaryEmbedding, RMSNormalization) and `com.microsoft` op attributes must be
   re-verified against current ONNX/ORT docs before implementing the importer.
9. **Uneven sharding** (e.g. TP = 3 on 14336 or H = 32): v0 errors unless `allow_uneven`; padding semantics are
   simple but the mapper may prefer non-SPMD splits. Keep SPMD-only for v0?
10. **Crate split**: resolved; `kiln-wl` is in 00's crate table.
11. **Chunked mask** and **explicit masks** are approximate or special-cased in point counting. Document-masked
    packing: closed, training is out of scope (08 §C).
12. **Opaque nodes** in evolution runs: should any opaque node make a workload ineligible for Tier A scoring, to
    avoid designs being rewarded for roofline-only gaps?
