//! 06 §6.2 `kiln.options/1` and §6.4 fitness configuration.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use kiln_ir::common::{Diagnostic, canonical_json, content_hash};
use kiln_ir::hw::types::ExecModel;
use kiln_trace::result::Aggregation;
use kiln_trace::{IntervalMethod, Tier, TraceLevel};
use kiln_wl::stack::{self, Stack};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const OPTIONS_CODE: &str = "E-OPT-0001";
/// The hardware-only stack fitness compares both sides under by default (08 §F software stack in scoring).
pub const STACK_IDEAL: &str = "kiln_ideal";
/// Each side under its own execution model's default stack: the realistic score.
pub const STACK_OWN: &str = "own";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TierChoice {
    #[serde(rename = "validate")]
    Validate,
    A,
    B,
    #[default]
    #[serde(rename = "cascade")]
    Cascade,
}

impl TierChoice {
    /// The engine tier that produces the score (M1 cascade stops at S3, tier A).
    pub fn engine_tier(self) -> Tier {
        match self {
            TierChoice::B => Tier::B,
            _ => Tier::A,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitnessKind {
    #[default]
    MatchedEnvelope,
    ExplicitEnvelope,
    PerfPerWatt,
    PerfPerArea,
    Pareto,
    BaselineRelative,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalBasis {
    #[default]
    Central,
    Low,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidScore {
    #[default]
    Zero,
    Graded,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    #[default]
    Across,
    Within,
}

/// Explicit envelope limits (06 §6.4); unset fields come from the baseline's `physical` result, except the
/// off-chip limits, which come from the baseline's design summary under every enveloped kind (off-chip memory is
/// bought, not designed). For those, an explicit `null` disables the limit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub die_mm2: Option<f64>,
    pub power_w: Option<f64>,
    pub node: Option<String>,
    pub package_mm2: Option<f64>,
    pub hbm_shoreline_mm: Option<f64>,
    pub reticle_mm2: Option<f64>,
    /// Off-chip (mem stack interface) bandwidth, bytes/s: absent = the baseline's, `null` = unlimited.
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub offchip_bw: Option<Option<f64>>,
    /// Off-chip (mem stack) capacity, bytes: absent = the baseline's, `null` = unlimited.
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub offchip_bytes: Option<Option<f64>>,
}

/// Present (`null` included) -> `Some`; absent stays `None` through `default`.
fn nullable<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Option<f64>>, D::Error> {
    Option::<f64>::deserialize(d).map(Some)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Fitness {
    pub kind: FitnessKind,
    pub baseline: String,
    pub envelope: Option<Envelope>,
    pub interval_basis: IntervalBasis,
    pub phase_weights: BTreeMap<String, f64>,
    pub aggregation: Aggregation,
    pub invalid_score: Option<InvalidScore>,
    pub prune_below: Option<f64>,
}

impl Default for Fitness {
    fn default() -> Self {
        Self {
            kind: FitnessKind::MatchedEnvelope,
            baseline: "a100_40gb".into(),
            envelope: None,
            interval_basis: IntervalBasis::Central,
            phase_weights: BTreeMap::new(),
            aggregation: Aggregation::Geomean,
            invalid_score: None,
            prune_below: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Timeouts {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self { a: 5.0, b: 300.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuditOptions {
    pub suspicion_ratio: f64,
    pub random_rate: f64,
    pub seeds: u32,
    pub heldout: bool,
}

impl Default for AuditOptions {
    fn default() -> Self {
        Self {
            suspicion_ratio: 1.15,
            random_rate: 0.02,
            seeds: 3,
            heldout: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Options {
    pub tier: TierChoice,
    pub calibration: Option<String>,
    pub fitness: Fitness,
    pub workloads: String,
    /// Workload for batch items given as bare designs (06 §6.1).
    pub workload: Option<Value>,
    pub interval: IntervalMethod,
    pub seeds: Vec<u64>,
    pub timeout_s: Timeouts,
    pub audit: AuditOptions,
    pub trace: TraceLevel,
    pub features: Option<Vec<String>>,
    pub invalid_score: InvalidScore,
    pub profile: String,
    pub parallelism: Parallelism,
    /// Software stack both sides of `score` run under: a built-in id or recipe file (`kiln_ideal` default,
    /// hardware-only), or `own` (each design under its execution model's default; the realistic score).
    pub stack: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tier: TierChoice::Cascade,
            calibration: None,
            fitness: Fitness::default(),
            workloads: "evolve".into(),
            workload: None,
            interval: IntervalMethod::Sensitivity,
            seeds: vec![0],
            timeout_s: Timeouts::default(),
            audit: AuditOptions::default(),
            trace: TraceLevel::Summary,
            features: None,
            invalid_score: InvalidScore::Zero,
            profile: "search".into(),
            parallelism: Parallelism::Across,
            stack: STACK_IDEAL.into(),
        }
    }
}

const PROFILES: [&str; 4] = ["full", "reference", "search", "stream_compat"];

impl Options {
    pub fn from_value(v: &Value) -> Result<Self, Diagnostic> {
        let o: Options = serde_path_to_error::deserialize(v).map_err(|e| {
            let path = e.path().to_string();
            Diagnostic::error(OPTIONS_CODE, format!("invalid options: {}", e.inner()))
                .at(format!("options.{path}"))
                .hint("see spec 06 §6.2 for the kiln.options/1 fields and their defaults")
        })?;
        o.check()?;
        Ok(o)
    }

    pub fn from_json(s: &str) -> Result<Self, Diagnostic> {
        let v: Value = serde_json::from_str(s).map_err(|e| {
            Diagnostic::error(OPTIONS_CODE, format!("options are not JSON: {e}")).at("options")
        })?;
        Self::from_value(&v)
    }

    fn check(&self) -> Result<(), Diagnostic> {
        let bad = |path: &str, msg: String, hint: &str| {
            Err(Diagnostic::error(OPTIONS_CODE, msg)
                .at(format!("options.{path}"))
                .hint(hint.to_string()))
        };
        if !PROFILES.contains(&self.profile.as_str()) {
            return bad(
                "profile",
                format!("unknown profile {:?}", self.profile),
                "one of full, reference, search, stream_compat",
            );
        }
        if self.seeds.is_empty() {
            return bad(
                "seeds",
                "seeds is empty".into(),
                "give at least one seed, e.g. [0]",
            );
        }
        for (k, t) in [("A", self.timeout_s.a), ("B", self.timeout_s.b)] {
            if !(t.is_finite() && t > 0.0) {
                return bad(
                    &format!("timeout_s.{k}"),
                    format!("timeout {t} s is not positive"),
                    "timeouts are wall-clock seconds > 0",
                );
            }
        }
        if let Some((p, w)) = self
            .fitness
            .phase_weights
            .iter()
            .find(|(_, w)| !(w.is_finite() && **w >= 0.0))
        {
            return bad(
                &format!("fitness.phase_weights.{p}"),
                format!("weight {w} is not a finite non-negative number"),
                "weights default to 1 per phase",
            );
        }
        if self.stack != STACK_OWN
            && let Err(d) = load_stack(&self.stack)
        {
            return bad(
                "stack",
                format!("unknown software stack {:?}: {}", self.stack, d.message),
                &format!(
                    "a built-in id ({}), a kiln.stack/1 file, or \"own\"",
                    stack::builtin_ids().join(", ")
                ),
            );
        }
        if self.fitness.kind == FitnessKind::ExplicitEnvelope && self.fitness.envelope.is_none() {
            return bad(
                "fitness.envelope",
                "explicit_envelope needs fitness.envelope".into(),
                "e.g. {\"die_mm2\": 826, \"power_w\": 400, \"node\": \"tsmc_n7\"}",
            );
        }
        Ok(())
    }

    pub fn timeout(&self) -> f64 {
        match self.tier.engine_tier() {
            Tier::A => self.timeout_s.a,
            Tier::B => self.timeout_s.b,
        }
    }

    pub fn invalid_score(&self) -> InvalidScore {
        self.fitness.invalid_score.unwrap_or(self.invalid_score)
    }

    /// The recipe a design with execution model `m` runs under: `stack`, or `m`'s default for `own`.
    pub fn stack_for(&self, m: ExecModel) -> Result<Arc<Stack>, Diagnostic> {
        if self.stack == STACK_OWN {
            load_stack(stack::default_for(m))
        } else {
            load_stack(&self.stack)
        }
    }

    /// These options with `stack` replaced.
    pub fn with_stack(&self, stack: &str) -> Options {
        Options {
            stack: stack.into(),
            ..self.clone()
        }
    }

    /// The recipe's `id@hash` (a recipe file is keyed by its content); for `own`, every built-in recipe's, since
    /// any of them can be a design's default.
    fn stack_key(&self) -> String {
        if self.stack == STACK_OWN {
            let labels: Vec<String> = stack::builtin_ids()
                .into_iter()
                .map(|i| load_stack(i).map_or_else(|_| i.to_string(), |s| s.label()))
                .collect();
            return format!("{STACK_OWN}[{}]", labels.join(","));
        }
        load_stack(&self.stack).map_or_else(|_| self.stack.clone(), |s| s.label())
    }

    /// Options that change metrics (06 §6.7); fitness is excluded because score is recomputed from cached metrics.
    pub fn metric_key(&self) -> Value {
        json!({"tier": self.tier, "seeds": self.seeds, "trace": self.trace, "interval": self.interval,
            "profile": self.profile, "stack": self.stack_key()})
    }

    /// Every option that can change a simulated metric (tier, interval method, seeds, trace, and any option
    /// added later): the full options minus fields that only shape scoring, reporting or scheduling. The
    /// calibration set enters baseline keys by its resolved hash. A baseline is reused only under an identical
    /// basis (08 §F scoring basis).
    pub fn basis_key(&self) -> Value {
        let mut v = serde_json::to_value(self).expect("options serialize");
        if let Some(o) = v.as_object_mut() {
            for k in [
                "calibration",
                "fitness",
                "workloads",
                "workload",
                "timeout_s",
                "audit",
                "features",
                "invalid_score",
                "profile",
                "parallelism",
            ] {
                o.remove(k);
            }
            o.insert("stack".into(), json!(self.stack_key()));
        }
        v
    }

    pub fn hash(&self) -> String {
        content_hash(
            "opt1-",
            &serde_json::to_value(self).expect("options serialize"),
        )
    }

    pub fn canonical_json(&self) -> String {
        canonical_json(&serde_json::to_value(self).expect("options serialize"))
    }
}

/// A built-in recipe (parsed once) or a recipe file.
fn load_stack(spec: &str) -> Result<Arc<Stack>, Diagnostic> {
    static BUILTIN: OnceLock<Vec<(&'static str, Arc<Stack>)>> = OnceLock::new();
    let all = BUILTIN.get_or_init(|| {
        stack::builtin_ids()
            .into_iter()
            .filter_map(|i| Stack::load(i).ok().map(|s| (i, Arc::new(s))))
            .collect()
    });
    match all.iter().find(|b| b.0 == spec) {
        Some(b) => Ok(b.1.clone()),
        None => Stack::load(spec).map(Arc::new),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_round_trip() {
        let o = Options::from_value(&json!({})).unwrap();
        assert_eq!(o, Options::default());
        assert_eq!(o.fitness.baseline, "a100_40gb");
        let back = Options::from_json(&o.canonical_json()).unwrap();
        assert_eq!(back, o);
    }

    #[test]
    fn rejects_unknown_fields_with_path() {
        let e =
            Options::from_value(&json!({"fitness": {"kind": "matched_envelope", "basline": "x"}}))
                .unwrap_err();
        assert_eq!(e.code, OPTIONS_CODE);
        assert!(e.message.contains("basline"), "{}", e.message);
        let e = Options::from_value(&json!({"tier": "C"})).unwrap_err();
        assert_eq!(e.path.as_deref(), Some("options.tier"));
        assert!(Options::from_value(&json!({"fitness": {"kind": "explicit_envelope"}})).is_err());
        assert!(Options::from_value(&json!({"timeout_s": {"A": 0}})).is_err());
    }

    #[test]
    fn invalid_score_override_and_timeout() {
        let o = Options::from_value(&json!({"tier": "B", "fitness": {"invalid_score": "graded"}}))
            .unwrap();
        assert_eq!(o.invalid_score(), InvalidScore::Graded);
        assert_eq!(o.timeout(), 300.0);
    }

    #[test]
    fn stack_defaults_to_ideal_and_keys_the_basis() {
        let o = Options::default();
        assert_eq!(o.stack, STACK_IDEAL);
        let ideal = o.stack_for(ExecModel::HostLaunched).unwrap();
        assert_eq!(ideal.id, STACK_IDEAL);
        assert_eq!(o.basis_key()["stack"], json!(ideal.label()));
        assert_eq!(o.metric_key()["stack"], json!(ideal.label()));
        let own = o.with_stack(STACK_OWN);
        assert_eq!(
            own.stack_for(ExecModel::HostLaunched).unwrap().id,
            "pytorch_cuda_graph_sdpa"
        );
        assert_eq!(
            own.stack_for(ExecModel::StaticDataflow).unwrap().id,
            "xla_tpu_fused"
        );
        assert_ne!(own.basis_key(), o.basis_key());
        let pt = Options::from_value(&json!({"stack": "pytorch_cuda_graph_sdpa"})).unwrap();
        assert_ne!(pt.basis_key(), o.basis_key());
        let e = Options::from_value(&json!({"stack": "no_such_stack"})).unwrap_err();
        assert_eq!(e.path.as_deref(), Some("options.stack"));
    }

    #[test]
    fn own_stack_keys_include_every_default_recipe() {
        let own = Options::default().with_stack(STACK_OWN);
        for key in [
            own.metric_key()["stack"].clone(),
            own.basis_key()["stack"].clone(),
        ] {
            let key = key.as_str().unwrap().to_string();
            for m in [
                ExecModel::HostLaunched,
                ExecModel::StaticDataflow,
                ExecModel::DeviceQueued,
            ] {
                let label = own.stack_for(m).unwrap().label();
                assert!(key.contains(&label), "{key} lacks {label}");
            }
        }
    }
}
