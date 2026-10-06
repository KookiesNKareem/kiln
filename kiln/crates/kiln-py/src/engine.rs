//! Boundary to the simulation engine (`kiln_sim::evaluate_prepared`, 03). The session owns S0, caching,
//! timeouts, fitness and features; the engine owns S1-S3 for one `(design, workload member)`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::{Design, HwModel};
use kiln_sim::{Prepared, SimOptions};
use kiln_trace::result::{EvalResult, Stage, Status};
use kiln_trace::{Provenance, RESULT_SCHEMA};
use kiln_wl::zoo::SuiteMember;

use crate::options::{Options, TierChoice};

pub const NOT_IMPLEMENTED: &str = "E-NOT-IMPLEMENTED";

#[derive(Clone, Debug)]
pub struct Calibration {
    pub id: String,
    pub hash: String,
    /// The loaded set (06 §3.6); `None` with no `params`: the assumed priors (`assumed-v0`).
    pub set: Option<Arc<kiln_sim::CalibSet>>,
    /// Explicit engine parameters (the `null` set).
    pub params: Option<kiln_sim::ParamSet>,
}

impl PartialEq for Calibration {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && self.hash == o.hash
    }
}

impl Eq for Calibration {}

pub struct EngineRequest {
    pub design: Arc<Design>,
    pub model: Arc<HwModel>,
    pub member: Arc<SuiteMember>,
    pub options: Arc<Options>,
    pub calibration: Arc<Calibration>,
    /// Cooperative-cancellation deadline (06 §6.8): the engine stops at its next check after it.
    pub deadline: Instant,
}

pub trait Engine: Send + Sync {
    fn evaluate(&self, req: &EngineRequest) -> EvalResult;
    /// Model version folded into the cache key (crate versions + git hash, 06 §6.7).
    fn version(&self) -> String;
}

/// Tier A (`kiln_sim`) over the session's validated model with the request's calibration set resolved
/// against the design (06 §3.6); provenance carries the set's id and hash.
pub struct SimEngine;

impl SimEngine {
    fn options(req: &EngineRequest) -> Result<SimOptions, Diagnostic> {
        let o = &req.options;
        let stack = o
            .stack_for(req.model.exec_model)
            .map_err(|d| d.at("options.stack"))?;
        Ok(SimOptions {
            interval: o.interval,
            trace: o.trace,
            threads: 1,
            git_hash: crate::GIT_HASH.into(),
            calibration: req.calibration.set.clone(),
            params: req.calibration.params.clone(),
            stack: Some(stack),
            deadline: Some(req.deadline),
            ..SimOptions::default()
        })
    }
}

impl Engine for SimEngine {
    fn evaluate(&self, req: &EngineRequest) -> EvalResult {
        if req.options.tier == TierChoice::B {
            let mut r = base_result(Status::InternalError, Stage::S0, provenance(req));
            r.errors.push(
                Diagnostic::error(
                    NOT_IMPLEMENTED,
                    "tier B (event-driven) is not in this build",
                )
                .at("options.tier")
                .hint("use tier \"A\" or \"cascade\" (M1 scores at tier A)")
                .into(),
            );
            return r;
        }
        let prepared = Prepared::new(req.design.clone(), req.model.clone())
            .and_then(|p| Self::options(req).map(|o| (p, o)));
        let (p, sim_opts) = match prepared {
            Ok(x) => x,
            Err(d) => {
                let mut r = base_result(Status::Invalid, Stage::S0, provenance(req));
                r.errors.push(d.into());
                return r;
            }
        };
        let runs = req
            .options
            .seeds
            .iter()
            .map(|&seed| {
                let o = SimOptions {
                    seed,
                    ..sim_opts.clone()
                };
                let r = kiln_sim::evaluate_prepared(&p, std::slice::from_ref(&*req.member), &o);
                (seed, r)
            })
            .collect();
        let (mut r, seed_scores) = median_seed(runs);
        r.provenance = provenance(req);
        if let Some(s) = r.sim.first() {
            let sp = &s.provenance;
            r.provenance.mapping_hash.clone_from(&sp.mapping_hash);
            r.provenance.window = sp.window;
            r.provenance.flags.clone_from(&sp.flags);
            r.provenance.flags.insert(
                "params".into(),
                sp.calibration_id.clone().unwrap_or_default(),
            );
        }
        if req.options.seeds.len() > 1 {
            r.provenance.flags.insert("seed_scores".into(), seed_scores);
        }
        r
    }

    fn version(&self) -> String {
        format!("kiln-sim-{}-{}", kiln_trace::KILN_VERSION, crate::GIT_HASH)
    }
}

/// 06 §6.2 multi-seed scoring: the run with the median score (the lower median for an even count), or the
/// first failed run when any seed fails, with every seed's score as JSON (for `provenance.flags.seed_scores`);
/// the relative spread goes to `audit.seed_spread`.
pub fn median_seed(mut runs: Vec<(u64, EvalResult)>) -> (EvalResult, String) {
    let scores: BTreeMap<String, f64> =
        runs.iter().map(|(s, r)| (s.to_string(), r.score)).collect();
    let pick = match runs.iter().position(|(_, r)| r.status != Status::Ok) {
        Some(i) => i,
        None => {
            let mut order: Vec<usize> = (0..runs.len()).collect();
            order.sort_by(|&a, &b| runs[a].1.score.total_cmp(&runs[b].1.score));
            order[(order.len() - 1) / 2]
        }
    };
    let mut r = runs.swap_remove(pick).1;
    if scores.len() > 1 {
        let (lo, hi) = scores
            .values()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        if r.status == Status::Ok && r.score > 0.0 {
            r.audit.seed_spread = Some((hi - lo) / r.score);
        }
    }
    (
        r,
        serde_json::to_string(&scores).expect("seed scores serialize"),
    )
}

pub fn default_engine() -> Arc<dyn Engine> {
    Arc::new(SimEngine)
}

pub fn provenance(req: &EngineRequest) -> Provenance {
    Provenance {
        kiln_version: kiln_trace::KILN_VERSION.into(),
        git_hash: crate::GIT_HASH.into(),
        design_hash: req.design.hash.clone(),
        workload_hash: crate::inputs::WorkloadSet::member_hash(&req.member),
        calibration_hash: req.calibration.hash.clone(),
        calibration_id: Some(req.calibration.id.clone()),
        mapping_hash: None,
        options_hash: None,
        tier: req.options.tier.engine_tier(),
        seeds: req.options.seeds.clone(),
        trust_level: Default::default(),
        window: None,
        chunk_bytes: None,
        flags: Default::default(),
    }
}

pub fn base_result(status: Status, stage: Stage, provenance: Provenance) -> EvalResult {
    EvalResult {
        schema: RESULT_SCHEMA.into(),
        status,
        score: 0.0,
        score_interval: None,
        score_components: None,
        score_realistic: None,
        interval: Default::default(),
        stage_reached: stage,
        tier: None,
        phases: vec![],
        ops: vec![],
        physical: None,
        features: Default::default(),
        violations: vec![],
        errors: vec![],
        warnings: vec![],
        audit: Default::default(),
        trace: None,
        provenance,
        timing: Default::default(),
        calibration: None,
        invariants: None,
        sim: vec![],
    }
}
