//! 03 §10 `SimResult`: one engine run of one phase at one corner. Entities are referenced by their
//! dotted IR paths, so this crate does not depend on the hardware or workload IR types.

use std::collections::BTreeMap;

use kiln_ir::common::Id;
use serde::{Deserialize, Serialize};

use crate::interval::Corner;
use crate::provenance::{Provenance, Tier, TraceLevel};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Step,
    Layer,
    Op,
}

impl Scope {
    /// Only whole-step and layer results may enter a score (03 §4.9).
    pub fn is_scored(self) -> bool {
        self != Scope::Op
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimResult {
    pub schema: String,
    pub provenance: Provenance,
    pub phase: Id,
    pub tier: Tier,
    pub scope: Scope,
    pub corner: Corner,
    pub makespan_s: f64,
    pub t_a0_s: f64,
    pub t_a2_s: f64,
    #[serde(default)]
    pub clocks: Vec<ClockSample>,
    pub energy: EnergyBreakdown,
    pub power: PowerSummary,
    #[serde(default)]
    pub ops: Vec<OpResult>,
    #[serde(default)]
    pub resources: Vec<ResourceResult>,
    #[serde(default)]
    pub groups: Vec<GroupResult>,
    #[serde(default)]
    pub collectives: Vec<CollectiveResult>,
    pub bottleneck: Bottleneck,
    pub invariants: InvariantReport,
    #[serde(default)]
    pub cost_model: CostModelSummary,
    pub calibration: CalibrationContribution,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceRef>,
}

impl SimResult {
    pub fn links(&self) -> impl Iterator<Item = &ResourceResult> {
        self.resources
            .iter()
            .filter(|r| r.kind == ResourceKind::Link)
    }

    pub fn op(&self, id: &str) -> Option<&OpResult> {
        self.ops.iter().find(|o| o.op.as_str() == id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClockSample {
    pub domain: Id,
    pub t_s: f64,
    pub hz: f64,
}

/// Energy split (03 §10). Memory and link terms are keyed by memory level and link class names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyBreakdown {
    pub compute_j: f64,
    #[serde(default)]
    pub memory_j: BTreeMap<String, f64>,
    #[serde(default)]
    pub link_j: BTreeMap<String, f64>,
    #[serde(default)]
    pub nmp_j: f64,
    #[serde(default)]
    pub static_j: f64,
    #[serde(default)]
    pub conversion_j: f64,
    #[serde(default)]
    pub padding_j: f64,
    pub total_j: f64,
}

impl EnergyBreakdown {
    /// Sum of the components in the fixed order that I6 requires.
    pub fn component_sum(&self) -> f64 {
        let mut s = self.compute_j;
        s += self.memory_j.values().sum::<f64>();
        s += self.link_j.values().sum::<f64>();
        s + self.nmp_j + self.static_j + self.conversion_j + self.padding_j
    }

    pub fn with_total(mut self) -> Self {
        self.total_j = self.component_sum();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerSummary {
    pub avg_w: f64,
    pub peak_windowed_w: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap_w: Option<f64>,
    pub throttled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpResult {
    pub op: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    pub start_s: f64,
    pub end_s: f64,
    pub binding: Binding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_up: Option<RunnerUp>,
    pub macs_useful: u64,
    pub macs_issued: u64,
    #[serde(default)]
    pub bytes_by_level: Vec<LevelBytes>,
    pub energy: EnergyBreakdown,
    pub target: Target,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_vs_nmp_s: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub floors: Vec<Floor>,
}

impl OpResult {
    pub fn time_s(&self) -> f64 {
        self.end_s - self.start_s
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunnerUp {
    pub binding: Binding,
    /// Runner-up term time over binding term time, in [0, 1].
    pub ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelBytes {
    pub level: String,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Host,
    Nmp,
}

/// 03 §10 `Binding`, with resources named by path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Binding {
    Compute { resource: Id },
    Link { resource: Id },
    MemPort { resource: Id },
    Dram { resource: Id },
    Nmp { resource: Id },
    Dependency,
    Overhead { overhead: OverheadKind },
    Contention { resource: Id },
    PipelineBubble,
}

impl Binding {
    pub fn class(&self) -> BindingClass {
        match self {
            Binding::Compute { .. } => BindingClass::Compute,
            Binding::Link { .. } => BindingClass::Link,
            Binding::MemPort { .. } => BindingClass::Port,
            Binding::Dram { .. } => BindingClass::Dram,
            Binding::Nmp { .. } => BindingClass::Nmp,
            Binding::Dependency => BindingClass::Dependency,
            Binding::Overhead { .. } => BindingClass::Overhead,
            Binding::Contention { .. } => BindingClass::Contention,
            Binding::PipelineBubble => BindingClass::PipelineBubble,
        }
    }

    pub fn resource(&self) -> Option<&Id> {
        match self {
            Binding::Compute { resource }
            | Binding::Link { resource }
            | Binding::MemPort { resource }
            | Binding::Dram { resource }
            | Binding::Nmp { resource }
            | Binding::Contention { resource } => Some(resource),
            _ => None,
        }
    }
}

/// Declaration order is the Tier A tie-break order of 03 §10 (Compute < Dram < Link < Port < Dependency <
/// Overhead); the classes 03 leaves unordered follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingClass {
    Compute,
    Dram,
    Link,
    Port,
    Dependency,
    Overhead,
    Nmp,
    Contention,
    PipelineBubble,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverheadKind {
    Launch,
    MinKernel,
    Gap,
    Dispatch,
    Program,
    Sync,
    CollectiveSetup,
    ModeSwitch,
}

/// 03 §4.1 resource kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    ComputeUnit,
    Link,
    MemPort,
    Bank,
    DramChannel,
    NmpSite,
    Sequencer,
    DmaEngine,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceResult {
    pub resource: Id,
    pub kind: ResourceKind,
    pub busy_s: f64,
    #[serde(default)]
    pub stall_s: f64,
    #[serde(default)]
    pub bytes: f64,
    #[serde(default)]
    pub macs: u64,
    pub energy_j: f64,
    pub utilization: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_queue: Option<f64>,
    /// Link class (e.g. `noc`, `nvlink`) for links; keys `EnergyBreakdown::link_j`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    Single,
    Fused,
    Pipelined,
    LayerByLayer,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupResult {
    pub group: u32,
    pub kind: GroupKind,
    pub ops: Vec<Id>,
    pub start_s: f64,
    pub end_s: f64,
    pub binding: Binding,
    #[serde(default)]
    pub bubble_s: f64,
    #[serde(default)]
    pub exposed_overhead_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectiveResult {
    pub collective: Id,
    pub op: Id,
    pub algorithm: String,
    pub chips: Vec<Id>,
    pub steps: u32,
    pub bytes: f64,
    pub start_s: f64,
    pub end_s: f64,
    #[serde(default)]
    pub link_bytes_by_tier: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Critical-path (B) or per-group (A) decomposition; sums to the makespan.
    pub time_by_binding: BTreeMap<BindingClass, f64>,
    #[serde(default)]
    pub top_resources: Vec<TopResource>,
    #[serde(default)]
    pub slack: Vec<ResourceSlack>,
    /// Filled by `kiln_sim::explain_run` (00 decision 7).
    #[serde(default)]
    pub summary: String,
}

impl Bottleneck {
    pub fn attributed_s(&self) -> f64 {
        self.time_by_binding.values().sum()
    }

    pub fn dominant(&self) -> Option<(BindingClass, f64)> {
        self.time_by_binding
            .iter()
            .fold(None, |best, (&c, &t)| match best {
                Some((_, bt)) if bt >= t => best,
                _ => Some((c, t)),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopResource {
    pub resource: Id,
    pub utilization: f64,
    pub shadow_price: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceSlack {
    pub resource: Id,
    pub slack: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorKind {
    Compute,
    MemoryLevel,
    Link,
    Collective,
    Roofline,
}

/// A physical lower bound on time (06 §6.3 `floor_s` per floor kind); `path` names the level, link or unit set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Floor {
    pub kind: FloorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub seconds: f64,
}

/// 03 §8 invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InvariantId {
    I1,
    I2,
    I3,
    I4,
    I5,
    I6,
    I7,
    I8,
    I9,
    I10,
    I11,
    I12,
    I13,
    I14,
    I15,
}

impl InvariantId {
    pub fn code(self) -> String {
        format!("E-FLOOR-{self:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvariantCheck {
    pub id: InvariantId,
    pub status: CheckStatus,
    /// Signed distance from the limit in `unit`; negative on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InvariantReport {
    pub checks: Vec<InvariantCheck>,
}

impl InvariantReport {
    pub fn failures(&self) -> impl Iterator<Item = &InvariantCheck> {
        self.checks.iter().filter(|c| c.status == CheckStatus::Fail)
    }

    pub fn passed(&self) -> bool {
        self.failures().next().is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostModelSummary {
    #[serde(default)]
    pub cache_hits: u64,
    #[serde(default)]
    pub cache_misses: u64,
    #[serde(default)]
    pub truncated_searches: u64,
}

/// Dual reporting (03 §9 principle 5).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationContribution {
    pub set_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncalibrated_makespan_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncalibrated_energy_j: Option<f64>,
    #[serde(default)]
    pub params: Vec<ParamContribution>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extrapolated: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamContribution {
    pub name: String,
    /// Mechanism key (06 §3.6), e.g. `{"dram_kind": "hbm2e"}`.
    pub key: BTreeMap<String, String>,
    pub delta_makespan_s: f64,
    pub delta_energy_j: f64,
}

/// Handle to a persisted `.kiln` trace (06 §6.3 `trace`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRef {
    pub id: String,
    pub tier: Tier,
    pub level: TraceLevel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewer_url: Option<String>,
}
