//! Measurement sessions, schema `kiln.meas/1` (06 §4.3-§4.5).

pub mod legacy;
pub mod runner;
pub mod store;

use std::collections::{BTreeMap, BTreeSet};

use kiln_ir::common::{Diagnostic, canonical_json, content_hash};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use kiln_ir::bench::{BenchKind, BenchOp, Residency};

pub const MEAS_SCHEMA: &str = "kiln.meas/1";
pub const SESSION_HASH_PREFIX: &str = "meas1-";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasSession {
    pub schema: String,
    pub session_id: String,
    pub device: Device,
    pub clocks_power: ClocksPower,
    pub software: BTreeMap<String, String>,
    pub host: Host,
    pub method: Method,
    pub timing_modes: BTreeMap<String, TimingMode>,
    pub canonical_mode: String,
    pub evidence_grade: EvidenceGrade,
    pub quality: Quality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy: Option<LegacySource>,
    pub records: Vec<MeasRecord>,
    /// Count-weighted occurrences of measured ops in workload phases (06 §2.5 `sum` form).
    #[serde(default)]
    pub uses: Vec<OpUse>,
    /// `meas1-` + sha256 over the canonical JSON of the session without this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    Nvidia,
    Google,
    Amd,
    Other,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub vendor: Vendor,
    pub sku: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pci_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vbios: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_bus_width_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_capability: Option<String>,
    pub chip_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    /// Vendor datasheet peaks as recorded by the runner (not measurements).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_mem_bw_bps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_bf16_flops: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClocksPower {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_max_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_max_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_observed_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_observed_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_limit_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_default_limit_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_max_limit_w: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ecc_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistence_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mig_mode: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Host {
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Method {
    pub runner: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kiln_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iters: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flush_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub settings: BTreeMap<String, Value>,
}

/// Overheads a timing mode's numbers include (06 §2.5 `overheads_included`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Overhead {
    HostDispatch,
    KernelLaunch,
    InterKernelGap,
    L2Warm,
}

/// How calls were issued; separates modes with equal overhead sets but different launch paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchPath {
    EagerStream,
    GraphReplay,
    HostBlocking,
    HostPipelined,
    DeviceLoop,
    Derived,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimingMode {
    pub description: String,
    pub launch: LaunchPath,
    pub overheads_included: BTreeSet<Overhead>,
    pub operands: Residency,
    pub trusted: bool,
    /// Recorded for diagnosis only; never a source of canonical numbers (e.g. TPU `pipelined_rot`, 08 §F).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub diagnostic: bool,
    /// For a derived mode: the modes it takes the minimum over.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_from: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceGrade {
    G1,
    G2,
    G3,
    G4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityStatus {
    Accepted,
    Flagged,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    pub status: QualityStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
    /// 06 §4.3 gates that could not be evaluated on this data.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unevaluated: Vec<String>,
}

impl Quality {
    pub fn accepted() -> Self {
        Self {
            status: QualityStatus::Accepted,
            reasons: vec![],
            unevaluated: vec![],
        }
    }

    pub fn flag(&mut self, reason: impl Into<String>) {
        self.status = self.status.max(QualityStatus::Flagged);
        self.reasons.push(reason.into());
    }

    pub fn reject(&mut self, reason: impl Into<String>) {
        self.status = QualityStatus::Rejected;
        self.reasons.push(reason.into());
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegacySource {
    pub file_name: String,
    pub file_sha256: String,
    pub format: String,
    pub importer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oplist_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Suite {
    LlmOps,
    Sweep,
    Peak,
    Hbm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasRecord {
    /// Unique within the session, e.g. `ops/gemm_2048_6144_4096`.
    pub id: String,
    pub suite: Suite,
    pub bench_key: String,
    pub op: BenchOp,
    #[serde(rename = "impl")]
    pub implementation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_name: Option<String>,
    pub modes: BTreeMap<String, ModeStats>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mode_errors: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<ClockWindow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<PowerWindow>,
    pub quality: Quality,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<PhaseOp>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeStats {
    pub median_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p90_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cv: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, Value>,
}

impl ModeStats {
    pub fn median(median_s: f64) -> Self {
        Self {
            median_s,
            min_s: None,
            p90_s: None,
            n: None,
            cv: None,
            details: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClockWindow {
    pub samples: u32,
    pub sm_hz_median: f64,
    pub sm_hz_min: f64,
    pub mem_hz_median: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_c: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerWindow {
    pub median_w: f64,
    pub max_w: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PhaseOp {
    pub phase: String,
    pub op: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpUse {
    pub suite: String,
    pub phase: String,
    pub op: String,
    pub bench_key: String,
    /// Record id that measures this occurrence.
    pub record: String,
    pub count: u64,
    pub tokens: u64,
}

impl MeasSession {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("MeasSession serializes")
    }

    pub fn canonical_json(&self) -> String {
        canonical_json(&self.to_value())
    }

    pub fn compute_hash(&self) -> String {
        let mut v = self.to_value();
        v.as_object_mut().expect("object").remove("hash");
        content_hash(SESSION_HASH_PREFIX, &v)
    }

    pub fn seal(mut self) -> Self {
        self.hash = Some(self.compute_hash());
        self
    }

    pub fn parse(json: &str) -> Result<Self, Diagnostic> {
        serde_json::from_str(json).map_err(|e| {
            Diagnostic::error(codes::SCHEMA, format!("not a {MEAS_SCHEMA} session: {e}"))
                .hint("legacy runner output is converted with `kiln bench import --legacy`")
        })
    }

    pub fn record(&self, id: &str) -> Option<&MeasRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    pub fn by_key<'a>(&'a self, bench_key: &'a str) -> impl Iterator<Item = &'a MeasRecord> {
        self.records
            .iter()
            .filter(move |r| r.bench_key == bench_key)
    }

    pub fn phases(&self) -> BTreeSet<&str> {
        self.uses.iter().map(|u| u.phase.as_str()).collect()
    }

    /// Count-weighted sum of per-op medians over a phase in one timing mode (06 §2.5 `sum` form).
    pub fn phase_sum_s(&self, phase: &str, mode: &str) -> Result<f64, Diagnostic> {
        let uses: Vec<_> = self.uses.iter().filter(|u| u.phase == phase).collect();
        if uses.is_empty() {
            return Err(Diagnostic::error(
                codes::LOOKUP,
                format!("no ops recorded for phase {phase:?}"),
            ));
        }
        uses.iter().try_fold(0.0, |acc, u| {
            let stats = self
                .record(&u.record)
                .and_then(|r| r.modes.get(mode))
                .ok_or_else(|| {
                    Diagnostic::error(
                        codes::LOOKUP,
                        format!("{} has no {mode:?} measurement", u.record),
                    )
                    .at(format!("uses[{phase}/{}]", u.op))
                })?;
            let sum = acc + u.count as f64 * stats.median_s;
            if !sum.is_finite() {
                return Err(Diagnostic::error(
                    codes::FIELD,
                    format!(
                        "{phase} {mode} time is not finite after {} x {}",
                        u.count, u.record
                    ),
                )
                .at(format!("uses[{phase}/{}]", u.op)));
            }
            Ok(sum)
        })
    }

    /// Structural checks for `kiln validate --kind measurement`.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if self.schema != MEAS_SCHEMA {
            out.push(Diagnostic::error(
                codes::SCHEMA,
                format!("schema {:?}, expected {MEAS_SCHEMA:?}", self.schema),
            ));
        }
        match &self.hash {
            None => out.push(
                Diagnostic::error(codes::HASH, "session has no hash")
                    .hint("sessions are sealed on import"),
            ),
            Some(h) if *h != self.compute_hash() => out.push(
                Diagnostic::error(
                    codes::HASH,
                    format!("stored hash {h} != computed {}", self.compute_hash()),
                )
                .hint("sessions are append-only; a correction is a new session"),
            ),
            _ => {}
        }
        match self.timing_modes.get(&self.canonical_mode) {
            None => out.push(Diagnostic::error(
                codes::SCHEMA,
                format!("canonical mode {:?} undefined", self.canonical_mode),
            )),
            Some(m) if m.diagnostic || !m.trusted => out.push(Diagnostic::error(
                codes::SCHEMA,
                format!(
                    "canonical mode {:?} is diagnostic or untrusted",
                    self.canonical_mode
                ),
            )),
            Some(_) => {}
        }
        for (name, m) in &self.timing_modes {
            for d in m
                .derived_from
                .iter()
                .filter(|d| !self.timing_modes.contains_key(*d))
            {
                out.push(Diagnostic::error(
                    codes::SCHEMA,
                    format!("mode {name} derives from undefined {d}"),
                ));
            }
        }
        let mut ids = BTreeSet::new();
        for r in &self.records {
            let p = format!("records[{}]", r.id);
            if !ids.insert(r.id.as_str()) {
                out.push(Diagnostic::error(codes::SCHEMA, "duplicate record id").at(&p));
            }
            if r.op.kind.is_contraction() && (r.op.flops().is_none() || r.op.min_bytes().is_none())
            {
                out.push(
                    Diagnostic::error(
                        codes::FIELD,
                        "descriptor has no representable FLOP or byte count",
                    )
                    .at(&p),
                );
            }
            if r.bench_key != r.op.key() {
                out.push(
                    Diagnostic::error(codes::HASH, "bench_key does not match op descriptor").at(&p),
                );
            }
            for (mode, s) in &r.modes {
                if !self.timing_modes.contains_key(mode) {
                    out.push(
                        Diagnostic::error(codes::SCHEMA, format!("undefined timing mode {mode:?}"))
                            .at(&p),
                    );
                }
                if !(s.median_s.is_finite() && s.median_s > 0.0) {
                    out.push(
                        Diagnostic::error(codes::SCHEMA, format!("{mode} median {} s", s.median_s))
                            .at(&p),
                    );
                }
            }
        }
        for u in self.uses.iter().filter(|u| {
            self.record(&u.record)
                .is_none_or(|r| r.bench_key != u.bench_key)
        }) {
            out.push(Diagnostic::error(
                codes::LOOKUP,
                format!("use {}/{} resolves to no record", u.phase, u.op),
            ));
        }
        out
    }
}

/// `E-CAL-*` codes owned by 06 for measurement handling.
pub mod codes {
    pub const FORMAT: &str = "E-CAL-0301";
    pub const FIELD: &str = "E-CAL-0302";
    pub const SUPERSEDED: &str = "E-CAL-0303";
    pub const HASH: &str = "E-CAL-0304";
    pub const APPEND_ONLY: &str = "E-CAL-0305";
    pub const OPLIST: &str = "E-CAL-0306";
    pub const LOOKUP: &str = "E-CAL-0307";
    pub const SCHEMA: &str = "E-CAL-0308";
    pub const IO: &str = "E-CAL-0309";
}
