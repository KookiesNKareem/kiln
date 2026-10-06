//! Calibration sets (06 §3.6, schema `kiln.calib/1`): versioned, hashed files of registered mechanism
//! parameters, each with a range (03 §9.1). A set is resolved against a design into the engine's
//! [`ParamSet`]: exact mechanism keys first, then the set's wildcard (`*`) fallbacks, then the assumed priors.
//! The schema has no field naming an op, shape, phase or workload, so a per-op scalar cannot be expressed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kiln_ir::common::{Diagnostic, content_hash};
use kiln_map::HwView;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::params::{Param, ParamSet, PessDir, Range};

pub const CALIB_SCHEMA: &str = "kiln.calib/1";
pub const CAL_CODE: &str = "E-CAL-0001";
/// Key names that would make a parameter per-op (06 §3.1 rule 1).
const FORBIDDEN_KEYS: &[&str] = &["op", "op_key", "shape", "phase", "workload", "scenario", "record", "name"];

/// One registered parameter (03 §9 table, 06 §3.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Registered {
    pub name: &'static str,
    pub key: &'static [&'static str],
    pub unit: &'static str,
    pub bounds: (f64, f64),
    /// Default prior (assumed): the fit's Gaussian prior centre and the fallback for unmeasured keys.
    pub prior: f64,
    pub pess_dir: PessDir,
    /// Plausible band for unmeasured keys (06 §3.2), `None` where the band is the measured span.
    pub band: Option<(f64, f64)>,
    pub term: &'static str,
}

const US: f64 = 1e-6;

// Bounds follow 03 §9 / 06 §3.2 except the lower bounds of `t_gap` and `t_min_kernel`: the initial 0.5 us each
// makes their sum >= 1.0 us, which the isolating A100 measurement (empty kernel in a CUDA-graph chain,
// t_min_kernel + t_gap = 0.90 us per kernel, 2026-10-05) rules out; 0.1 us keeps both strictly positive.

pub const REGISTRY: &[Registered] = &[
    Registered { name: "eta_res", key: &["dram_kind"], unit: "1", bounds: (0.5, 1.0), prior: 0.91, pess_dir: PessDir::Lower, band: Some((0.85, 0.97)), term: "DRAM residual efficiency (03 5.3)" },
    Registered { name: "t_dram_ramp", key: &["dram_kind"], unit: "s", bounds: (0.0, 20.0 * US), prior: 2.0 * US, pess_dir: PessDir::Upper, band: None, term: "DRAM pipeline fill+drain per streaming segment: eta_res(S) = eta_res * S / (S + B * eta_res * t_dram_ramp)" },
    Registered { name: "t_launch", key: &["exec_model"], unit: "s", bounds: (0.5 * US, 20.0 * US), prior: 5.0 * US, pess_dir: PessDir::Upper, band: None, term: "exposed launch latency per step (03 4.4)" },
    Registered { name: "t_min_kernel", key: &["exec_model"], unit: "s", bounds: (0.1 * US, 20.0 * US), prior: 2.5 * US, pess_dir: PessDir::Upper, band: None, term: "minimum kernel duration (03 4.4)" },
    Registered { name: "t_gap", key: &["exec_model"], unit: "s", bounds: (0.1 * US, 20.0 * US), prior: 1.0 * US, pess_dir: PessDir::Upper, band: None, term: "inter-kernel gap in a queue or graph (03 4.4)" },
    Registered { name: "t_dispatch", key: &["exec_model"], unit: "s", bounds: (0.1 * US, 20.0 * US), prior: 2.0 * US, pess_dir: PessDir::Upper, band: None, term: "per-group dispatch (03 4.4)" },
    Registered { name: "t_program", key: &["exec_model"], unit: "s", bounds: (0.1 * US, 200.0 * US), prior: 20.0 * US, pess_dir: PessDir::Upper, band: None, term: "per-program start (03 4.4)" },
    Registered { name: "t_sync", key: &["exec_model"], unit: "s", bounds: (0.1 * US, 50.0 * US), prior: 1.0 * US, pess_dir: PessDir::Upper, band: None, term: "barrier latency (03 4.4)" },
    Registered { name: "unit_eff", key: &["unit_template"], unit: "1", bounds: (0.7, 1.0), prior: 0.95, pess_dir: PessDir::Lower, band: Some((0.85, 1.0)), term: "per-unit pipeline efficiency (03 9)" },
    Registered { name: "eta_link", key: &["link_kind"], unit: "1", bounds: (0.6, 1.0), prior: 0.85, pess_dir: PessDir::Lower, band: Some((0.75, 0.95)), term: "link protocol efficiency (03 9)" },
    Registered { name: "f_cap_op", key: &["domain", "power_cap"], unit: "1", bounds: (0.9, 1.1), prior: 1.0, pess_dir: PessDir::Lower, band: None, term: "telemetry operating points: sustained clock vs MAC activity under the power cap (03 4.5 stand-in)" },
    Registered { name: "contention_scale", key: &["engine"], unit: "1", bounds: (0.0, 2.0), prior: 1.0, pess_dir: PessDir::Upper, band: None, term: "Tier A contention correction (03 4.3; Tier B only)" },
    Registered { name: "rho_max", key: &["engine"], unit: "1", bounds: (0.5, 0.99), prior: 0.95, pess_dir: PessDir::Upper, band: None, term: "Tier A contention cap (03 4.3; Tier B only)" },
];

pub fn registered(name: &str) -> Option<&'static Registered> {
    REGISTRY.iter().find(|r| r.name == name)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetKind {
    Platform,
    Generic,
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamStatus {
    Fit,
    Frozen,
    Assumed,
    /// Read from measured telemetry (clock operating points), not fitted on time.
    Telemetry,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeSpec {
    pub lower: f64,
    pub upper: f64,
    /// `device_spread`, `single_device`, `band`, `telemetry` (06 §3.2).
    pub basis: String,
}

/// Fit diagnostics of one parameter (06 §3.3).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitDiag {
    pub stage: String,
    pub n_records: usize,
    pub loss: f64,
    /// Median and p90 of |pred/meas - 1| over the stage's fit records after the fit.
    pub abs_err_median: f64,
    pub abs_err_p90: f64,
    pub at_bound: bool,
    pub iterations: usize,
    /// Per-device fits that pooled into the value (generic sets).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub per_device: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalParam {
    pub name: String,
    pub key: BTreeMap<String, String>,
    pub value: f64,
    pub unit: String,
    pub bounds: [f64; 2],
    pub prior: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci95: Option<[f64; 2]>,
    pub range: RangeSpec,
    pub pess_dir: PessDir,
    pub status: ParamStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Operating-point table (`f_cap_op`: `[activity, hz]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<Vec<[f64; 2]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<FitDiag>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitInfo {
    pub kiln_version: String,
    pub git_hash: String,
    pub method: String,
    pub loss: String,
    pub split_id: String,
    pub split_hash: String,
    pub fit_records: Vec<String>,
    pub test_records: Vec<String>,
    pub measurement_sessions: Vec<String>,
    #[serde(default)]
    pub residuals_by_class: BTreeMap<String, f64>,
    pub bootstrap_seed: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fit_devices: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// One measured observation behind a generic range: a device's span of a quantity and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeEvidence {
    pub device: String,
    pub quantity: String,
    pub lower: f64,
    pub upper: f64,
    pub n: usize,
    pub source: String,
}

/// The range a registered parameter takes on designs outside the fit devices: exactly the span of its evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRange {
    pub name: String,
    pub lower: f64,
    pub upper: f64,
    pub evidence: Vec<RangeEvidence>,
}

/// Generic range policy (06 §3.2, 08 §F cross-chip intervals): on a design whose `family` is not one of the
/// set's fit devices (every novel design, every held-out chip), each listed parameter's range is widened to
/// span the spread observed across measured devices; central values are unchanged. Fit devices keep their
/// fitted (bootstrap) ranges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangePolicy {
    pub applies_to: String,
    pub ranges: Vec<PolicyRange>,
}

impl RangePolicy {
    pub fn range(&self, name: &str) -> Option<&PolicyRange> {
        self.ranges.iter().find(|r| r.name == name)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibSet {
    pub schema: String,
    pub id: String,
    pub version: u32,
    pub kind: SetKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,
    pub parameters: Vec<CalParam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<FitInfo>,
    /// Generic sets only: ranges for designs outside `fit.fit_devices`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_policy: Option<RangePolicy>,
    /// Last `kiln calibrate report` snapshot; not hashed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance: Option<Value>,
    /// Appended by every report that reads test records; not hashed.
    #[serde(default)]
    pub test_access_log: Vec<Value>,
    pub created: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

/// Default directory of versioned sets: `$KILN_CALIB_DIR`, else `kiln/calibration/sets` in the source tree.
pub fn sets_dir() -> PathBuf {
    std::env::var_os("KILN_CALIB_DIR").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../calibration/sets"))
}

fn err(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(CAL_CODE, msg).at("calibration")
}

impl CalibSet {
    /// sha256 over the canonical set without `hash`, `acceptance` and `test_access_log` (the mutable report
    /// fields), so reporting never changes the hash results carry.
    pub fn compute_hash(&self) -> String {
        let mut v = serde_json::to_value(self).expect("set serializes");
        if let Some(o) = v.as_object_mut() {
            for k in ["hash", "acceptance", "test_access_log"] {
                o.remove(k);
            }
        }
        content_hash("cal1-", &v)
    }

    /// Rejects unregistered parameters, keys outside a parameter's mechanism key type, forbidden (per-op)
    /// keys, values outside bounds or ranges that do not bracket the value, and a stale `hash`.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = vec![];
        if self.schema != CALIB_SCHEMA {
            out.push(err(format!("schema {:?} is not {CALIB_SCHEMA}", self.schema)));
        }
        if self.kind == SetKind::Generic && self.platform.is_some() {
            out.push(err(format!("generic set {} names a platform", self.id)).hint("generic sets hold technology keys only (06 §3.1 rule 2)"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (i, p) in self.parameters.iter().enumerate() {
            let at = format!("parameters[{i}]");
            let Some(r) = registered(&p.name) else {
                out.push(err(format!("unregistered parameter {:?}", p.name)).at(at).hint("only parameters of 03 §9 / 06 §3.2 may be calibrated"));
                continue;
            };
            for k in p.key.keys() {
                if FORBIDDEN_KEYS.contains(&k.as_str()) {
                    out.push(err(format!("{}: key {k:?} would make the parameter per-op", p.name)).at(at.clone()));
                } else if !r.key.contains(&k.as_str()) {
                    out.push(err(format!("{}: key {k:?} is not its mechanism key {:?}", p.name, r.key)).at(at.clone()));
                }
            }
            let (lo, hi) = (r.bounds.0.min(p.bounds[0]), r.bounds.1.max(p.bounds[1]));
            if !(p.value >= lo && p.value <= hi) {
                out.push(err(format!("{} = {} outside registry bounds [{lo}, {hi}]", p.name, p.value)).at(at.clone()));
            }
            if !(p.range.lower <= p.value && p.value <= p.range.upper) {
                out.push(err(format!("{}: range [{}, {}] does not bracket {}", p.name, p.range.lower, p.range.upper, p.value)).at(at.clone()));
            }
            if !seen.insert((p.name.clone(), p.key.clone())) {
                out.push(err(format!("{} {:?} appears twice", p.name, p.key)).at(at));
            }
        }
        if let Some(pol) = &self.range_policy {
            if self.kind != SetKind::Generic {
                out.push(err(format!("{:?} set {} has a range policy", self.kind, self.id)).hint("only generic sets describe designs outside their fit devices"));
            }
            for (i, r) in pol.ranges.iter().enumerate() {
                let at = format!("range_policy.ranges[{i}]");
                if registered(&r.name).is_none() {
                    out.push(err(format!("range policy names unregistered parameter {:?}", r.name)).at(at.clone()));
                }
                let lo = r.evidence.iter().map(|e| e.lower).fold(f64::INFINITY, f64::min);
                let hi = r.evidence.iter().map(|e| e.upper).fold(f64::NEG_INFINITY, f64::max);
                if r.evidence.is_empty() || r.evidence.iter().any(|e| e.lower.partial_cmp(&e.upper).is_none_or(|o| o.is_gt())) || r.lower != lo || r.upper != hi {
                    out.push(err(format!("{}: policy range [{}, {}] is not the span of its evidence [{lo}, {hi}]", r.name, r.lower, r.upper)).at(at).hint("policy ranges are derived from measured evidence, never tuned"));
                }
            }
        }
        if let Some(h) = &self.hash
            && *h != self.compute_hash()
        {
            out.push(err(format!("set {} hash {h} does not match its content ({})", self.id, self.compute_hash())).hint("sets are immutable; bump the version"));
        }
        out
    }

    pub fn from_value(v: Value, origin: &str) -> Result<CalibSet, Diagnostic> {
        let set: CalibSet = serde_path_to_error::deserialize(v).map_err(|e| err(format!("{origin}: {e}")))?;
        if let Some(d) = set.validate().into_iter().next() {
            return Err(d);
        }
        Ok(set)
    }

    pub fn load(path: &Path) -> Result<CalibSet, Diagnostic> {
        let text = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| err(format!("{} is not JSON: {e}", path.display())))?;
        CalibSet::from_value(v, &path.display().to_string())
    }

    /// A set by id from [`sets_dir`] (`<id>.json`).
    pub fn by_id(id: &str) -> Result<CalibSet, Diagnostic> {
        let p = sets_dir().join(format!("{id}.json"));
        if !p.is_file() {
            return Err(err(format!("no calibration set {id:?} in {}", sets_dir().display())).hint("ids: null, assumed-v0, or a file in kiln/calibration/sets"));
        }
        CalibSet::load(&p)
    }

    /// Canonical pretty JSON with the hash filled in.
    pub fn to_json(&self) -> String {
        let mut s = self.clone();
        s.hash = Some(s.compute_hash());
        serde_json::to_string_pretty(&s).expect("set serializes") + "\n"
    }

    /// Whether the design is one of the set's fit devices (by `family`, the 06 platform key; a platform set's
    /// own chip). Designs without a family (everything evolution produces) never are.
    pub fn is_fit_device(&self, view: &HwView) -> bool {
        let Some(fam) = view.hw.family.as_deref() else { return false };
        self.platform.as_deref() == Some(fam) || self.fit.as_ref().is_some_and(|f| f.fit_devices.iter().any(|d| d == fam))
    }

    /// Engine parameters for one design: every assumed prior of the design's execution model and DRAM kind is
    /// replaced by the set's exact-key entry, else its `*` fallback; set-only parameters (ramp, unit
    /// efficiency, clock operating points) are added when their key matches. Unmatched keys keep the prior and
    /// are listed in `extrapolated`. On a design outside the fit devices, the range policy widens ranges.
    pub fn resolve(&self, view: &HwView, exec: kiln_ir::hw::types::ExecModel) -> ParamSet {
        let keys = design_keys(view, exec);
        let base = ParamSet::assumed(exec, keys.get("dram_kind").map_or("sram", String::as_str));
        let lookup = |name: &str, want: &BTreeMap<String, String>| -> Option<(&CalParam, bool)> {
            let named = || self.parameters.iter().filter(move |p| p.name == name);
            named().find(|p| p.key == *want).map(|p| (p, true)).or_else(|| named().find(|p| p.key.keys().eq(want.keys()) && p.key.values().all(|v| v == "*")).map(|p| (p, false)))
        };
        let mut params = vec![];
        let mut extrapolated = vec![];
        for p in base.params {
            match lookup(&p.name, &p.key) {
                Some((c, exact)) => {
                    if !exact {
                        extrapolated.push(p.name.clone());
                    }
                    params.push(engine_param(c, p.key.clone()));
                }
                None if self.kind == SetKind::Null => {}
                // Engine-keyed terms (Tier A contention) are fitted against Tier B, never on hardware (03 4.3):
                // a hardware set not carrying them is not extrapolating.
                None if p.key.contains_key("engine") => params.push(p),
                None => {
                    extrapolated.push(p.name.clone());
                    params.push(p);
                }
            }
        }
        if let Some(d) = keys.get("dram_kind") {
            let want = BTreeMap::from([("dram_kind".to_string(), d.clone())]);
            if let Some((c, exact)) = lookup("t_dram_ramp", &want) {
                if !exact {
                    extrapolated.push("t_dram_ramp".into());
                }
                params.push(engine_param(c, want));
            }
        }
        for (path, tech) in unit_templates(view) {
            let want = BTreeMap::from([("unit_template".to_string(), tech)]);
            if let Some((c, exact)) = lookup("unit_eff", &want) {
                if !exact {
                    extrapolated.push("unit_eff".into());
                }
                params.push(engine_param(c, BTreeMap::from([("unit_template".to_string(), path)])));
            }
        }
        for (ix, (domain, cap)) in capped_clocks(view) {
            let want = BTreeMap::from([("domain".to_string(), domain), ("power_cap".to_string(), cap)]);
            if let Some((c, true)) = lookup("f_cap_op", &want) {
                let mut k = want.clone();
                k.insert("clock_ix".into(), ix.to_string());
                params.push(engine_param(c, k));
            }
        }
        if let Some(pol) = self.range_policy.as_ref().filter(|_| !self.is_fit_device(view)) {
            for p in &mut params {
                if let Some(r) = pol.range(&p.name) {
                    let (lo, hi) = (p.range.lower.min(r.lower), p.range.upper.max(r.upper));
                    if (lo, hi) != (p.range.lower, p.range.upper) {
                        p.range.lower = lo;
                        p.range.upper = hi;
                        p.basis = "generic_policy".into();
                    }
                }
            }
        }
        extrapolated.sort();
        extrapolated.dedup();
        ParamSet { id: self.id.clone(), kind: format!("{:?}", self.kind).to_lowercase(), params, set_hash: Some(self.compute_hash()), extrapolated }
    }
}

fn engine_param(c: &CalParam, key: BTreeMap<String, String>) -> Param {
    Param {
        name: c.name.clone(),
        key,
        unit: c.unit.clone(),
        range: Range { lower: c.range.lower, central: c.value, upper: c.range.upper },
        pess_dir: c.pess_dir,
        status: serde_json::to_value(c.status).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default(),
        basis: c.range.basis.clone(),
        table: c.table.clone(),
    }
}

/// Mechanism keys a design exposes: `exec_model`, `dram_kind`.
pub fn design_keys(view: &HwView, exec: kiln_ir::hw::types::ExecModel) -> BTreeMap<String, String> {
    let mut k = BTreeMap::new();
    k.insert("exec_model".into(), crate::params::exec_key(exec).into());
    if let Some(d) = view.hw.memories.iter().find_map(|m| m.dram_kind()).and_then(|d| serde_json::to_value(d).ok()).and_then(|v| v.as_str().map(String::from)) {
        k.insert("dram_kind".into(), d);
    }
    k
}

/// MAC unit templates: `(entity path the engine keys on, mechanism key)`. The key names the platform family
/// (or design name), the unit and its geometry (`a100_40gb/tc:mma16x8x16`), so a novel design never inherits a
/// reference chip's unit efficiency (03 §9 principle 6).
pub fn unit_templates(view: &HwView) -> Vec<(String, String)> {
    let hw = &view.hw;
    let fam = hw.family.clone().unwrap_or_else(|| hw.name.clone());
    let mut v: Vec<(String, String)> = view
        .units
        .iter()
        .filter(|u| hw.units[u.unit].spec.kind.is_mac())
        .map(|u| {
            let geo = match &hw.units[u.unit].spec.kind {
                kiln_ir::hw::ComputeKind::Matrix(m) => match &m.geometry {
                    kiln_ir::hw::compute::Geometry::Systolic { rows, cols } => format!("systolic{rows}x{cols}"),
                    kiln_ir::hw::compute::Geometry::Mma { m, n, k } => format!("mma{m}x{n}x{k}"),
                    kiln_ir::hw::compute::Geometry::OuterProduct { rows, cols } => format!("outer{rows}x{cols}"),
                    kiln_ir::hw::compute::Geometry::Spatial { dims } => format!("spatial{}", dims.values().map(u32::to_string).collect::<Vec<_>>().join("x")),
                },
                _ => "mac".into(),
            };
            let leaf = u.template.rsplit('.').next().unwrap_or(&u.template).to_string();
            (u.template.clone(), format!("{fam}/{leaf}:{geo}"))
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Clock domains governed by a DVFS power cap: `(clock index, (domain id, cap in W as text))`.
pub fn capped_clocks(view: &HwView) -> Vec<(usize, (String, String))> {
    let hw = &view.hw;
    let mut out = vec![];
    for pd in hw.power_domains.iter().filter(|p| p.policy == kiln_ir::hw::phys::PowerPolicy::Dvfs) {
        for &c in &pd.clocks {
            if let Some(clk) = hw.clocks.get(c) {
                out.push((c, (clk.spec.id.to_string(), format!("{}W", pd.cap.0.round()))));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}
