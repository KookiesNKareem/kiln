//! Public data types of the intra-unit cost model (03 §2): unit template, op nest, mapping, result.

use serde::{Deserialize, Serialize};

use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{Precision, PrecisionSpec};

pub type LevelIx = usize;
pub type DimIx = usize;
pub type OperandIx = usize;

// ------------------------------------------------------------------------------------------- unit template

/// Self-contained description of one compute unit (or a gang of identical units sharing upper levels) plus the
/// memory hierarchy it feeds from (03 §2.2). Its content hash is the cache key component, so it carries every
/// number the model reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitTemplate {
    pub name: String,
    /// Unit clock; every per-cycle quantity below is in cycles of this clock.
    pub clock_hz: f64,
    pub axes: Vec<SpatialAxis>,
    pub modes: Vec<MacMode>,
    /// Every memory level any operand chain uses; index = `LevelIx`.
    pub levels: Vec<MemLevel>,
    /// Per operand role, the levels it stages through, innermost (closest to the array) first.
    pub chains: Vec<OperandChain>,
    pub pipeline: PipelineCycles,
    /// Precision partial sums are held at between reduction steps when the output does not stay in the array.
    /// None => the mode's `acc`.
    pub psum_precision: Option<PrecisionSpec>,
    /// Final down-conversion to the output dtype is free (fused in the drain path).
    pub fused_down_conversion: bool,
    /// Energy of a clock-gated (padding) MAC as a fraction of `e_mac` (03 §2.6; assumed default 0.1).
    pub e_mac_idle_ratio: f64,
    /// Energy per vector op for conversions / MX scale application (J).
    pub e_vector_op_j: f64,
    pub energy_source: EnergySource,
    /// Per input role, the longest run of temporal iterations an operand stays in the array between reads from
    /// level 0 (01 `operand_run`); roles not listed are unlimited.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub operand_run: Vec<(OperandRole, u64)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialAxis {
    pub name: String,
    pub size: u32,
    /// Loop-dim names allowed on this axis (02 canonical names such as `m`, `n`, `k`); empty = any dim.
    pub allowed: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MacMode {
    pub a: PrecisionSpec,
    pub b: PrecisionSpec,
    pub acc: PrecisionSpec,
    pub out: Option<PrecisionSpec>,
    /// MACs per cycle of the whole array in this mode (`ComputeKind::ops_per_cycle`).
    pub macs_per_cycle: f64,
    pub e_mac_j: f64,
    /// Scales of MX/block operands are applied in-array (03 §2.7).
    pub mx_native: bool,
    /// Elements of k one instruction spans per element of the unit's `k` axis: MMA instructions cover 256 bits
    /// of k per row (PTX mma/wgmma: k16 for 16-bit operands, k32 for 8-bit, k64 for 4-bit), so narrow modes
    /// accumulate twice or four times as many products per accumulator access.
    #[serde(default = "one_u32")]
    pub k_pack: u32,
}

fn one_u32() -> u32 {
    1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDir {
    Read,
    Write,
    ReadWrite,
}

/// The four data-movement directions at a level (ZigZag naming): `ToLow` reads that refill the level below,
/// `FromHigh` writes that fill this level from above, `ToHigh` reads that write back upward, `FromLow` writes
/// arriving from below (outputs / partial sums).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    ToLow,
    FromHigh,
    ToHigh,
    FromLow,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemPort {
    pub dir: PortDir,
    /// Bandwidth in bytes per unit cycle, per instance of the level (`MemLevel::instance_axes`).
    pub bytes_per_cycle: f64,
    /// `(role, direction)` pairs routed to this port; empty = every pair compatible with `dir`.
    pub serves: Vec<(OperandRole, Direction)>,
    /// Independent sub-ports splitting `bytes_per_cycle` evenly (01 §7 port `count`, banks, instances). An
    /// access occupies whole sub-port cycles, so concurrent accesses of different operands share the lanes
    /// instead of each rounding up to a full port cycle.
    #[serde(default = "one_lane")]
    pub lanes: u32,
}

fn one_lane() -> u32 {
    1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemLevel {
    pub name: String,
    /// Expanded memory instance this level stands for (None for synthetic levels).
    pub mem: Option<usize>,
    /// Total capacity over all instances of the level.
    pub capacity_bytes: u64,
    pub ports: Vec<MemPort>,
    /// The level supports double buffering (the mapping chooses per operand).
    pub double_buffer: bool,
    /// Spatial axes along which the level has separate instances (per-PE registers); data irrelevant to such
    /// an axis is replicated in each instance. Empty = one instance shared by the whole array.
    pub instance_axes: Vec<usize>,
    pub e_read_j_per_b: f64,
    pub e_write_j_per_b: f64,
    pub latency_cycles: u64,
    /// The level is the boundary to resources outside the unit; its bandwidth is `bw_assumed` (03 §2.5) and
    /// kiln-sim re-checks it against the real shared resources.
    pub external: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperandChain {
    pub role: OperandRole,
    pub levels: Vec<LevelIx>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineCycles {
    pub fill: u64,
    pub drain: u64,
    pub issue_overhead: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergySource {
    /// Placeholder constants (`energy::PLACEHOLDER`) until kiln-phys (04) supplies per-access energies.
    Uncalibrated,
    /// Every energy constant was supplied by the caller (kiln-phys, or an oracle's table).
    Supplied,
}

/// Choices `UnitTemplate::from_hw` cannot infer from the design alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemplateOptions {
    /// Gang this many identical sibling units sharing every non-local level as one spatial axis (e.g. the 4
    /// tensor cores of an A100 SM). 1 = the unit alone.
    pub gang: u32,
    /// Memory instance (`MemIx`) each chain ends at (the home of the operands); None = the deepest memory
    /// reachable from the feed memory.
    pub home: Option<usize>,
    /// Bandwidth of the boundary into the outermost level in bytes/s (03 §2.5 `bw_assumed`); None = the home
    /// memory's structural bandwidth.
    pub bw_assumed: Option<f64>,
}

impl Default for TemplateOptions {
    fn default() -> Self {
        Self { gang: 1, home: None, bw_assumed: None }
    }
}

// ------------------------------------------------------------------------------------------------- op nest

/// One affine kernel (or a tile of it) as seen by one unit (03 §2.2 `OpNest`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpNest {
    pub kind: NestKind,
    pub dims: Vec<NestDim>,
    pub operands: Vec<NestOperand>,
    /// MACs per iteration point (1 for GEMM, 0 for pure data movement).
    pub macs_per_point: u32,
    /// Non-MAC work per point (vector unit work; reported, not executed on the array).
    pub vector_ops_per_point: u32,
    /// Exact iteration points of the (possibly mask-constrained) domain; None = product of dim sizes.
    pub points: Option<u64>,
    /// Accumulator precision the kernel requires (`Kernel::accum`); a mode accumulating narrower is never chosen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accum: Option<Precision>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NestKind {
    Contraction,
    Map,
    Reduce,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopKind {
    Parallel,
    Reduction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestDim {
    /// 02 canonical loop-dim name; only used to match `SpatialAxis::allowed` and for display.
    pub name: String,
    pub size: u64,
    pub kind: LoopKind,
}

/// One tensor axis as an affine combination of loop dims plus a constant, optionally floor-divided (MX scale
/// index `k / 32`): `floor((sum c * d + offset) / div)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisExpr {
    pub terms: Vec<(DimIx, i64)>,
    pub div: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub offset: i64,
}

fn is_zero(x: &i64) -> bool {
    *x == 0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NestOperand {
    pub tensor: String,
    /// Role used to pick the unit's operand chain: `A`, `B` inputs, `O` output (`In`/`Out` for vector units).
    pub role: OperandRole,
    pub axes: Vec<AxisExpr>,
    pub dtype: PrecisionSpec,
    pub is_output: bool,
    /// Level the operand lives at before the op (index into `UnitTemplate::levels`); None = outermost level
    /// of its chain.
    pub source: Option<LevelIx>,
    /// Outputs only: level the result must end at; None = outermost level of its chain.
    pub sink: Option<LevelIx>,
    /// Tensor axis carrying MX/block scales; None = derived (the last axis indexed by a reduction dim).
    pub block_axis: Option<usize>,
}

// ------------------------------------------------------------------------------------------------- mapping

/// Per array axis, the `(dim, unroll)` pairs bound to it; `prod(unroll) <= axis size` (03 §2.3).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SpatialMapping {
    pub axes: Vec<Vec<(DimIx, u64)>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TemporalLoop {
    pub dim: DimIx,
    pub factor: u64,
}

/// LOMA-style temporal nest (03 §2.4) plus the memory allocation of every operand stream.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TemporalMapping {
    /// Innermost first; per dim the factors multiply to the class's temporal size.
    pub loops: Vec<TemporalLoop>,
    /// Per operand stream (nest operands, then MX scale streams in operand order), per chain level
    /// (innermost first): number of loops (counted from the innermost) whose tile lives at or below that
    /// level. Non-decreasing; the outermost entry equals `loops.len()`.
    pub alloc: Vec<Vec<usize>>,
    /// Per operand stream, per chain level: double buffered.
    pub double_buffer: Vec<Vec<bool>>,
}

/// One tile class: the dim sizes it covers and how many times it repeats (03 §2.3 ragged splits).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileClass {
    pub sizes: Vec<u64>,
    pub count: u64,
    pub spatial: SpatialMapping,
    pub temporal: TemporalMapping,
}

/// Complete serializable intra-unit mapping.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Mapping {
    pub classes: Vec<TileClass>,
}

// ------------------------------------------------------------------------------------------------- queries

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Latency,
    Energy,
    Edp,
    /// `w_lat * cycles_norm + w_e * energy_norm`, weights in thousandths (quantized for exact keys).
    Weighted { w_lat_milli: u32, w_e_milli: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RaggedPolicy {
    /// Full and remainder tile classes costed separately (03 §2.3, default).
    Split,
    /// Pad every dim to a multiple of its spatial unroll (ZigZag behavior).
    Pad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SearchBudget {
    /// Spatial candidates kept after ranking (03 §2.3 `K_s`).
    pub top_k_spatial: usize,
    /// Evaluated loop orders per spatial candidate (03 §2.4 node budget).
    pub max_evals_per_spatial: u64,
    /// Latency objective: stop as soon as a mapping's issue + stall cycles reach the latency floor (compute
    /// cycles, and compulsory bytes over each level's ports); later mappings could improve only startup and
    /// fill/drain, or tie (the energy tie-break is skipped).
    pub stop_at_floor: bool,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self { top_k_spatial: 8, max_evals_per_spatial: 20_000, stop_at_floor: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CostOptions {
    pub ragged: RaggedPolicy,
    pub budget: SearchBudget,
    /// ZigZag-compatible accounting (06 §2.3 `cost_model = zigzag_compat`): padded nest, ZigZag's stall and
    /// onload/offload rules. Used only by the differential harness.
    pub zigzag_compat: bool,
}

impl Default for CostOptions {
    fn default() -> Self {
        Self { ragged: RaggedPolicy::Split, budget: SearchBudget::default(), zigzag_compat: false }
    }
}

/// One cost-model request: unit, kernel tile (with residency in its operands), objective.
#[derive(Clone, Copy, Debug)]
pub struct CostQuery<'a> {
    pub unit: &'a UnitTemplate,
    pub nest: &'a OpNest,
    pub objective: Objective,
    pub options: CostOptions,
}

// -------------------------------------------------------------------------------------------------- result

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LevelAccess {
    pub level: LevelIx,
    /// Operand stream (nest operand index, or `operands.len() + i` for the i-th MX scale stream).
    pub operand: OperandIx,
    /// Element counts per direction (exact integers). For outputs, `to_high`/`from_low` include partial sums.
    pub to_low: u64,
    pub from_high: u64,
    pub to_high: u64,
    pub from_low: u64,
    /// Bytes read from / written to this level (accumulator width for partial sums).
    pub read_bytes: u64,
    pub write_bytes: u64,
    /// The same bytes per direction: `[to_low, from_high, to_high, from_low]`.
    #[serde(default)]
    pub dir_bytes: [u64; 4],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyBreakdown {
    pub mac_j: f64,
    pub idle_mac_j: f64,
    pub vector_j: f64,
    pub conversion_j: f64,
    /// Per level (index = `LevelIx`): read + write energy.
    pub levels_j: Vec<f64>,
    pub total_j: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limiter {
    Compute,
    /// Bandwidth of a port of a level (stalls dominate).
    Port { level: LevelIx, port: usize },
    /// Onload/offload and pipeline fill/drain dominate.
    FillDrain,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Floors {
    /// `ceil(useful_macs / macs_per_cycle)` in the chosen mode.
    pub compute_cycles: u64,
    /// Per level: compulsory bytes (each distinct element of operands that cross the level once).
    pub compulsory_bytes: Vec<u64>,
    /// Per level: compulsory bytes / total port bandwidth of the level, cycles.
    pub bandwidth_cycles: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStats {
    pub spatial_candidates: u64,
    pub evaluated: u64,
    pub pruned: u64,
    pub truncated: bool,
}

/// Result of one query (03 §2.8 `CostResult`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostEntry {
    /// `issue + stall + fill_drain` unit cycles (03 §2.5).
    pub cycles: u64,
    pub latency_s: f64,
    pub issue_cycles: u64,
    pub stall_cycles: u64,
    pub fill_drain_cycles: u64,
    pub useful_macs: u64,
    pub issued_macs: u64,
    pub vector_ops: u64,
    pub conversion_ops: u64,
    /// Index of the `MacMode` used.
    pub mode: usize,
    pub accesses: Vec<LevelAccess>,
    pub energy: EnergyBreakdown,
    pub energy_source: EnergySource,
    /// `useful_macs / issued_macs`.
    pub spatial_util: f64,
    /// `useful_macs / (cycles * macs_per_cycle)`.
    pub utilization: f64,
    pub limiter: Limiter,
    pub floors: Floors,
    pub mapping: Mapping,
    pub search: SearchStats,
}

pub type CostResult = CostEntry;
