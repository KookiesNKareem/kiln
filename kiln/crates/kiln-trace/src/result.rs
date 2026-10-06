//! 06 §6.3 `kiln.result/1`: the evaluation result returned by `kiln-py` and `kiln eval`.

use std::collections::BTreeMap;

use kiln_ir::common::{Diagnostic, Id, canonical_json, content_hash};
use serde::{Deserialize, Serialize};

use crate::interval::{Corner, Interval, IntervalMethod};
use crate::provenance::{Provenance, Tier};
use crate::sim::{
    BindingClass, Floor, InvariantReport, ParamContribution, Scope, SimResult, TraceRef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Invalid,
    Envelope,
    Infeasible,
    FloorViolation,
    Pruned,
    Timeout,
    InternalError,
}

impl Status {
    /// 06 §7 exit code for a single evaluation with this status.
    pub fn exit_code(self) -> u8 {
        match self {
            Status::Ok => 0,
            Status::Invalid | Status::Envelope | Status::Infeasible | Status::Pruned => 1,
            Status::Timeout => 6,
            Status::FloorViolation => 7,
            Status::InternalError => 70,
        }
    }
}

/// Cascade stages (06 §6.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stage {
    S0,
    S1,
    S2,
    S3,
    S4,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalResult {
    pub schema: String,
    pub status: Status,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_interval: Option<Interval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_components: Option<ScoreComponents>,
    /// Realistic-stack score (08 §F): each side under its own execution model's default software stack.
    /// Reported next to `score`, which compares both sides under one stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_realistic: Option<RealisticScore>,
    #[serde(default)]
    pub interval: IntervalInfo,
    pub stage_reached: Stage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub phases: Vec<PhaseResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<OpSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical: Option<PhysicalSummary>,
    #[serde(default)]
    pub features: BTreeMap<String, Feature>,
    #[serde(default)]
    pub violations: Vec<ResultError>,
    #[serde(default)]
    pub errors: Vec<ResultError>,
    #[serde(default)]
    pub warnings: Vec<ResultError>,
    #[serde(default)]
    pub audit: Audit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceRef>,
    pub provenance: Provenance,
    /// Wall-clock data; excluded from `deterministic_hash`.
    #[serde(default)]
    pub timing: Timing,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invariants: Option<InvariantReport>,
    /// 03's `SimResult` per phase and corner, unmodified (`trace >= summary`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sim: Vec<SimResult>,
}

impl EvalResult {
    pub fn phase(&self, id: &str) -> Option<&PhaseResult> {
        self.phases.iter().find(|p| p.phase.as_str() == id)
    }

    pub fn sim_for(&self, phase: &str, corner: Corner) -> Option<&SimResult> {
        self.sim
            .iter()
            .find(|s| s.phase.as_str() == phase && s.corner == corner)
    }

    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("EvalResult serializes")
    }

    pub fn canonical_json(&self) -> String {
        canonical_json(&self.to_value())
    }

    /// Hash of everything but wall-clock `timing`, for the 06 §5.4 determinism checks.
    pub fn deterministic_hash(&self) -> String {
        let mut v = self.to_value();
        if let Some(o) = v.as_object_mut() {
            o.remove("timing");
        }
        content_hash("res1-", &v)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    Geomean,
    Min,
    WeightedHarmonic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreComponents {
    pub phases: BTreeMap<String, Interval>,
    pub weights: BTreeMap<String, f64>,
    pub aggregation: Aggregation,
    pub baseline_id: String,
    pub baseline_hash: String,
    /// Software stack labels (`id@hash`) of the two sides (08 §F); equal unless the score is the realistic one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_stack: Option<String>,
}

/// The realistic score (06 §6.4, 08 §F): the candidate under its execution model's default stack over the
/// baseline under its own (PyTorch for host-launched GPUs, XLA for TPUs), formed like `score`/`score_interval`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealisticScore {
    pub score: f64,
    pub interval: Interval,
    pub candidate_stack: String,
    pub baseline_stack: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IntervalInfo {
    pub method: IntervalMethod,
    #[serde(default)]
    pub corner_flips: Vec<CornerFlip>,
    #[serde(default)]
    pub drivers: Vec<IntervalDriver>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_remapped: Option<f64>,
}

/// A discrete outcome that differs between corners (03 §9.1 (b)); a flip at `low` blocks claims.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CornerFlip {
    pub corner: Corner,
    pub what: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntervalDriver {
    pub param: String,
    pub key: BTreeMap<String, String>,
    /// Fraction of the interval width this parameter accounts for.
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseResult {
    pub phase: Id,
    pub scope: Scope,
    pub time_s: Interval,
    pub tokens_per_s: Interval,
    pub energy_j: Interval,
    pub tokens_per_j: Interval,
    pub avg_power_w: Interval,
    pub clock_hz: Interval,
    #[serde(default)]
    pub floors: Vec<Floor>,
    pub roofline_frac: f64,
    /// Fractions of time bound by `compute`, each memory level (`mem:<path>`), `link`, `overhead`; sums to 1.
    #[serde(default)]
    pub bound_breakdown: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_layer: Vec<LayerTime>,
    pub trusted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerTime {
    pub layer: u32,
    pub time_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpSummary {
    pub phase: Id,
    pub op: Id,
    pub time_s: f64,
    pub energy_j: f64,
    pub bound: BindingClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_resource: Option<Id>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub floors: Vec<Floor>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalSummary {
    pub die_mm2: BTreeMap<String, Interval>,
    pub package_mm2: Interval,
    pub peak_power_w: Interval,
    pub power_density_max_w_mm2: Interval,
    pub hbm_shoreline_used_mm: f64,
    pub hbm_shoreline_available_mm: f64,
    pub tdp_w: f64,
    pub node: String,
    /// Envelope margins (positive = inside) at the central and the pessimistic corner.
    #[serde(default)]
    pub margins_central: BTreeMap<String, f64>,
    #[serde(default)]
    pub margins_pessimistic: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Feature {
    Scalar(f64),
    Vector(Vec<f64>),
}

/// 06 §6.5 error object: a structured diagnostic plus optional numbers and the owning spec section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultError {
    #[serde(flatten)]
    pub diag: Diagnostic,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
}

impl From<Diagnostic> for ResultError {
    fn from(diag: Diagnostic) -> Self {
        Self {
            diag,
            value: None,
            limit: None,
            unit: None,
            section: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditStatus {
    #[default]
    NotRun,
    Pending,
    Passed,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub status: AuditStatus,
    #[serde(default)]
    pub reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_b_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heldout_gap: Option<f64>,
    #[serde(default)]
    pub extrapolated_components: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_report: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    #[serde(default)]
    pub stages_s: BTreeMap<Stage, f64>,
    #[serde(default)]
    pub cache_hits: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    #[serde(default)]
    pub uncalibrated_time_s: BTreeMap<String, f64>,
    #[serde(default)]
    pub contributions: BTreeMap<String, Vec<ParamContribution>>,
    #[serde(default)]
    pub extrapolated: Vec<String>,
}
