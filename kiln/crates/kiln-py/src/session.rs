//! 06 §6.1 `Session`: S0 validation, engine calls with timeout and panic isolation, cache, baseline, fitness,
//! features. Shared by `kiln-py` and `kiln eval` so both produce identical results.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kiln_ir::common::{Diagnostic, Id, Severity, content_hash};
use kiln_ir::hw::{Design, HwModel, Profile};
use kiln_trace::Provenance;
use kiln_trace::result::{EvalResult, ResultError, Stage, Status};
use serde_json::json;

use crate::cache::{self, Cache};
use crate::engine::{self, Calibration, Engine, EngineRequest};
use crate::inputs::{self, DesignInput, WorkloadInput, WorkloadSet};
use crate::options::{MAX_TIMEOUT_S, Options, STACK_OWN, TierChoice};
use crate::{features, fitness};

pub const TIMEOUT_CODE: &str = kiln_sim::TIMEOUT_CODE;
pub const INTERNAL_CODE: &str = "E-INTERNAL";
pub const CAL_CODE: &str = "E-CAL-0001";
pub const REALISTIC_CODE: &str = "W-FIT-REALISTIC";
pub const DEFAULT_CALIBRATION: &str = "generic-v1";
/// Slack after `timeout_s` before an evaluation is cut off (the engine's cooperative deadline).
const GRACE: Duration = Duration::from_millis(500);
const ENGINE_STACK: usize = 16 << 20;

/// `timeout_s` (+ [`GRACE`]) from now, clamped to [`MAX_TIMEOUT_S`] so it cannot overflow.
fn deadline(timeout_s: f64) -> Instant {
    let t = Duration::try_from_secs_f64(timeout_s.min(MAX_TIMEOUT_S)).unwrap_or(Duration::ZERO);
    let now = Instant::now();
    now.checked_add(t + GRACE).unwrap_or(now + GRACE)
}

#[derive(Clone, Debug, Default)]
pub struct SessionConfig {
    pub calibration: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub no_cache: bool,
    pub threads: Option<usize>,
    pub designs_dir: Option<PathBuf>,
}

type BaselineSlot = Arc<Mutex<Option<EvalResult>>>;

pub struct Session {
    engine: Arc<dyn Engine>,
    calibration: Arc<Calibration>,
    cache: Option<Cache>,
    threads: usize,
    designs_dir: PathBuf,
    baselines: Mutex<BTreeMap<String, BaselineSlot>>,
}

pub fn default_cache_dir() -> Option<PathBuf> {
    std::env::var_os("KILN_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(|d| PathBuf::from(d).join("kiln")))
        .or_else(|| std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache/kiln")))
}

/// A calibration set id (`generic-v1`, `platform:a100_40gb`, ... from `kiln/calibration/sets`; `assumed-v0` for
/// the uncalibrated priors; `null` for unit efficiencies and zero overheads), or a path to a set file.
pub fn resolve_calibration(spec: Option<&str>) -> Result<Calibration, Diagnostic> {
    let spec = spec.unwrap_or(DEFAULT_CALIBRATION);
    let p = std::path::Path::new(spec);
    let loaded = |set: kiln_sim::CalibSet| Calibration {
        id: set.id.clone(),
        hash: set.compute_hash(),
        set: Some(Arc::new(set)),
        params: None,
    };
    if p.is_file() {
        return kiln_sim::CalibSet::load(p).map(loaded);
    }
    let bad = || {
        Diagnostic::error(
            CAL_CODE,
            format!("calibration {spec:?} is neither a file nor a set id"),
        )
        .at("calibration")
        .hint(format!(
            "use {DEFAULT_CALIBRATION:?}, \"assumed-v0\", \"null\" or a path to a calibration set JSON"
        ))
    };
    let file_id = spec.replace(':', "-");
    Id::new(&file_id).map_err(|_| bad())?;
    match spec {
        "assumed-v0" => Ok(Calibration {
            id: spec.into(),
            hash: content_hash("cal1-", &json!({"id": spec})),
            set: None,
            params: None,
        }),
        "null" => {
            let ps = kiln_sim::ParamSet::null();
            Ok(Calibration {
                id: spec.into(),
                hash: ps.hash(),
                set: None,
                params: Some(ps),
            })
        }
        _ => kiln_sim::CalibSet::by_id(&file_id).map(loaded),
    }
}

fn profile(name: &str) -> Profile {
    match name {
        "full" => Profile::Full,
        "reference" => Profile::Reference,
        "stream_compat" => Profile::StreamCompat,
        _ => Profile::Search,
    }
}

fn split(diags: Vec<Diagnostic>) -> (Vec<ResultError>, Vec<ResultError>) {
    let (e, w): (Vec<_>, Vec<_>) = diags
        .into_iter()
        .partition(|d| d.severity == Severity::Error);
    (
        e.into_iter().map(Into::into).collect(),
        w.into_iter().map(Into::into).collect(),
    )
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

/// S0 failure: errors, warnings, and the design hash when the design parsed.
type LoadFailure = (Vec<ResultError>, Vec<ResultError>, Option<String>);

struct Loaded {
    design: Arc<Design>,
    model: Arc<HwModel>,
    warnings: Vec<ResultError>,
}

impl Session {
    pub fn new(cfg: SessionConfig) -> Result<Self, Diagnostic> {
        Self::with_engine(cfg, engine::default_engine())
    }

    pub fn with_engine(cfg: SessionConfig, engine: Arc<dyn Engine>) -> Result<Self, Diagnostic> {
        let calibration = Arc::new(resolve_calibration(cfg.calibration.as_deref())?);
        let cache = if cfg.no_cache {
            None
        } else {
            cfg.cache_dir.or_else(default_cache_dir).map(Cache::new)
        };
        let threads = cfg
            .threads
            .filter(|t| *t > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        Ok(Self {
            engine,
            calibration,
            cache,
            threads,
            designs_dir: cfg.designs_dir.unwrap_or_else(inputs::default_designs_dir),
            baselines: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn cache_dir(&self) -> Option<&std::path::Path> {
        self.cache.as_ref().map(Cache::dir)
    }

    fn calibration_for(&self, opts: &Options) -> Result<Arc<Calibration>, Diagnostic> {
        match &opts.calibration {
            None => Ok(self.calibration.clone()),
            Some(c) => resolve_calibration(Some(c)).map(Arc::new),
        }
    }

    fn provenance(
        &self,
        design_hash: &str,
        workload_hash: &str,
        opts: &Options,
        cal: &Calibration,
    ) -> Provenance {
        Provenance {
            kiln_version: kiln_trace::KILN_VERSION.into(),
            git_hash: crate::GIT_HASH.into(),
            design_hash: design_hash.into(),
            workload_hash: workload_hash.into(),
            calibration_hash: cal.hash.clone(),
            calibration_id: Some(cal.id.clone()),
            mapping_hash: None,
            options_hash: Some(opts.hash()),
            tier: opts.tier.engine_tier(),
            seeds: opts.seeds.clone(),
            trust_level: Default::default(),
            window: None,
            chunk_bytes: None,
            flags: Default::default(),
        }
    }

    /// `kiln.validate` / S0: IR + profile checks (01 §18), no simulation.
    pub fn validate(&self, design: &DesignInput, profile_name: &str) -> Vec<ResultError> {
        match self.load(design, profile_name) {
            Ok(l) => l.warnings,
            Err((errs, warns, _)) => errs.into_iter().chain(warns).collect(),
        }
    }

    fn load(&self, design: &DesignInput, profile_name: &str) -> Result<Loaded, LoadFailure> {
        let d = inputs::load_design(design, &self.designs_dir).map_err(|ds| {
            let (e, w) = split(ds);
            (e, w, None)
        })?;
        let hash = d.hash.clone();
        let report = kiln_sim::check_priced(d, profile(profile_name));
        let (errs, warns) = split(report.diagnostics);
        match (report.design, report.model) {
            (Some(design), Some(model)) if errs.is_empty() => Ok(Loaded {
                design: Arc::new(design),
                model: Arc::new(model),
                warnings: warns,
            }),
            _ => Err((errs, warns, Some(hash))),
        }
    }

    pub fn evaluate(
        &self,
        design: &DesignInput,
        workload: &WorkloadInput,
        opts: &Options,
    ) -> EvalResult {
        match inputs::resolve_workload(workload) {
            Ok(wl) => self.evaluate_set(design, &wl, opts),
            Err(d) => {
                let cal = self
                    .calibration_for(opts)
                    .unwrap_or_else(|_| self.calibration.clone());
                let mut r = engine::base_result(
                    Status::Invalid,
                    Stage::S0,
                    self.provenance("unknown", "unknown", opts, &cal),
                );
                r.errors.push(d.into());
                r.score = fitness::invalid_score(&r, opts.invalid_score());
                r
            }
        }
    }

    pub fn evaluate_set(
        &self,
        design: &DesignInput,
        wl: &WorkloadSet,
        opts: &Options,
    ) -> EvalResult {
        let (mut r, model) = self.metrics(design, wl, opts, opts.profile.as_str());
        if r.status == Status::Ok && opts.tier != TierChoice::Validate {
            let t = Instant::now();
            let id = &opts.fitness.baseline;
            let base = self.baseline(id, wl, opts);
            fitness::apply(&mut r, Some((id, &base)), opts);
            if r.status == Status::Ok {
                self.realistic(&mut r, model.as_deref(), design, wl, opts);
            }
            *r.timing.stages_s.entry(Stage::S3).or_default() += t.elapsed().as_secs_f64();
        } else {
            fitness::apply(&mut r, None, opts);
        }
        features::fill(&mut r, model.as_deref(), opts.features.as_deref());
        r
    }

    /// Sets `r.score_realistic`: the candidate under its own default stack over the baseline under its own (08
    /// §F). The candidate's metrics are reused when its default stack is the one `score` already used.
    fn realistic(
        &self,
        r: &mut EvalResult,
        model: Option<&HwModel>,
        design: &DesignInput,
        wl: &WorkloadSet,
        opts: &Options,
    ) {
        let own = opts.with_stack(STACK_OWN);
        let label = |o: &Options, m: &HwModel| o.stack_for(m.exec_model).ok().map(|s| s.label());
        let reuse =
            model.is_some_and(|m| label(opts, m).is_some() && label(opts, m) == label(&own, m));
        let cand = if reuse {
            r.clone()
        } else {
            self.metrics(design, wl, &own, opts.profile.as_str()).0
        };
        let id = &opts.fitness.baseline;
        let base = self.baseline(id, wl, &own);
        match fitness::realistic(&cand, (id, &base), opts) {
            Ok(rs) => r.score_realistic = Some(rs),
            Err(e) => r.warnings.push(
                Diagnostic::warning(
                    REALISTIC_CODE,
                    format!(
                        "no realistic-stack score: {} [{}]",
                        e.diag.message, e.diag.code
                    ),
                )
                .at("score_realistic")
                .hint("the candidate or the baseline does not evaluate under its own default stack; claims need both scores")
                .into(),
            ),
        }
    }

    /// The baseline simulated on `wl` (validated under the `full` profile) with the candidate's calibration set
    /// and simulation options, memoized per (baseline, design hash, workload, calibration hash,
    /// [`Options::basis_key`]) when its status is cacheable, and disk-cached. Never a measurement (08 §F scoring
    /// basis).
    pub fn baseline(&self, name: &str, wl: &WorkloadSet, opts: &Options) -> EvalResult {
        let input = DesignInput::Str(name.into());
        let compute = || {
            let (mut r, model) = self.metrics(&input, wl, opts, "full");
            features::fill(&mut r, model.as_deref(), None);
            r
        };
        let Ok(design) = inputs::load_design(&input, &self.designs_dir) else {
            return compute();
        };
        let cal = self
            .calibration_for(opts)
            .map_or_else(|_| self.calibration.hash.clone(), |c| c.hash.clone());
        let key = cache::key(&json!([
            name,
            design.hash,
            wl.hash(),
            opts.basis_key(),
            cal
        ]));
        let slot = self
            .baselines
            .lock()
            .expect("baseline map")
            .entry(key)
            .or_default()
            .clone();
        let mut memo = slot.lock().expect("baseline slot");
        if let Some(r) = memo.as_ref() {
            return r.clone();
        }
        let r = compute();
        if cache::cacheable(r.status) {
            *memo = Some(r.clone());
        }
        r
    }

    /// S0 + engine per workload member, merged; no fitness.
    fn metrics(
        &self,
        design: &DesignInput,
        wl: &WorkloadSet,
        opts: &Options,
        profile_name: &str,
    ) -> (EvalResult, Option<Arc<HwModel>>) {
        let t0 = Instant::now();
        let wl_hash = wl.hash();
        let cal = match self.calibration_for(opts) {
            Ok(c) => c,
            Err(d) => {
                let mut r = engine::base_result(
                    Status::Invalid,
                    Stage::S0,
                    self.provenance("unknown", &wl_hash, opts, &self.calibration),
                );
                r.errors.push(d.into());
                return (r, None);
            }
        };
        let loaded = match self.load(design, profile_name) {
            Ok(l) => l,
            Err((errors, warnings, hash)) => {
                let mut r = engine::base_result(
                    Status::Invalid,
                    Stage::S0,
                    self.provenance(hash.as_deref().unwrap_or("unknown"), &wl_hash, opts, &cal),
                );
                r.errors = errors;
                r.warnings = warnings;
                r.timing
                    .stages_s
                    .insert(Stage::S0, t0.elapsed().as_secs_f64());
                return (r, None);
            }
        };
        let s0 = t0.elapsed().as_secs_f64();
        let prov = self.provenance(&loaded.design.hash, &wl_hash, opts, &cal);
        if opts.tier == TierChoice::Validate {
            let mut r = engine::base_result(Status::Ok, Stage::S0, prov);
            r.warnings = loaded.warnings;
            r.timing.stages_s.insert(Stage::S0, s0);
            features::offchip(&mut r, &loaded.model);
            return (r, Some(loaded.model));
        }
        let options = Arc::new(opts.clone());
        let parts: Vec<EvalResult> = wl
            .members
            .iter()
            .map(|m| {
                let member = Arc::new(m.clone());
                let key = cache::key(&json!({
                    "engine": self.engine.version(),
                    "design": loaded.design.hash,
                    "workload": WorkloadSet::member_hash(m),
                    "calibration": cal.hash,
                    "options": opts.metric_key(),
                }));
                if let Some(mut hit) = self.cache.as_ref().and_then(|c| c.get(&key)) {
                    hit.timing.cache_hits += 1;
                    return hit;
                }
                let req = EngineRequest {
                    design: loaded.design.clone(),
                    model: loaded.model.clone(),
                    member,
                    options: options.clone(),
                    calibration: cal.clone(),
                    deadline: deadline(opts.timeout()),
                };
                let r = self.run_isolated(req, &m.name);
                if let Some(c) = &self.cache {
                    c.put(&key, &r);
                }
                r
            })
            .collect();
        let mut r = merge(
            parts,
            &wl.members
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
        );
        let single = wl.members.len() == 1;
        let flags = std::mem::take(&mut r.provenance.flags);
        r.provenance = Provenance {
            workload_hash: wl_hash,
            mapping_hash: r.provenance.mapping_hash.take().filter(|_| single),
            window: r.provenance.window,
            flags,
            ..prov
        };
        r.warnings.splice(0..0, loaded.warnings);
        *r.timing.stages_s.entry(Stage::S0).or_default() += s0;
        features::offchip(&mut r, &loaded.model);
        (r, Some(loaded.model))
    }

    fn run_isolated(&self, req: EngineRequest, member: &str) -> EvalResult {
        let wait = req.deadline.saturating_duration_since(Instant::now());
        let timeout = req.options.timeout();
        let prov = engine::provenance(&req);
        let (tx, rx) = mpsc::channel();
        let eng = self.engine.clone();
        let spawned = std::thread::Builder::new()
            .name("kiln-eval".into())
            .stack_size(ENGINE_STACK)
            .spawn(move || {
                let r = catch_unwind(AssertUnwindSafe(|| eng.evaluate(&req)));
                let _ = tx.send(r.map_err(|p| panic_message(&*p)));
            });
        let fail = |status, code: &str, msg: String, hint: &str| {
            let mut r = engine::base_result(status, Stage::S0, prov.clone());
            r.errors.push(
                Diagnostic::error(code, msg)
                    .at(format!("workload.{member}"))
                    .hint(hint)
                    .into(),
            );
            r
        };
        if let Err(e) = spawned {
            return fail(
                Status::InternalError,
                INTERNAL_CODE,
                format!("cannot start evaluation thread: {e}"),
                "retry with fewer workers",
            );
        }
        match rx.recv_timeout(wait) {
            Ok(Ok(r)) => r,
            Ok(Err(msg)) => fail(
                Status::InternalError,
                INTERNAL_CODE,
                format!(
                    "kiln panicked ({}): {msg}",
                    &content_hash("bt1-", &json!(msg))[4..16]
                ),
                "this is a kiln bug, not a design property; the batch continues",
            ),
            Err(e) => {
                if e == RecvTimeoutError::Timeout {
                    // The engine stops at its next deadline check; keep this worker until it has, so timed-out
                    // evaluations never pile up beyond `max_workers`.
                    let _ = rx.recv();
                }
                fail(
                    Status::Timeout,
                    TIMEOUT_CODE,
                    format!("evaluation of {member} exceeded {timeout:.1} s"),
                    "simplify the design (fewer chips, tiles or memory levels) or raise options.timeout_s",
                )
            }
        }
    }

    /// Results in input order; `max_workers` evaluations run concurrently (06 §6.1, `parallelism = across`).
    pub fn evaluate_batch(
        &self,
        items: &[(DesignInput, WorkloadInput)],
        opts: &Options,
        max_workers: Option<usize>,
    ) -> Vec<EvalResult> {
        let workers = max_workers
            .unwrap_or(self.threads)
            .clamp(1, items.len().max(1));
        let next = AtomicUsize::new(0);
        let out: Mutex<Vec<Option<EvalResult>>> = Mutex::new(vec![None; items.len()]);
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((d, w)) = items.get(i) else { break };
                        let r = self.evaluate(d, w, opts);
                        out.lock().expect("batch slots")[i] = Some(r);
                    }
                });
            }
        });
        out.into_inner()
            .expect("batch slots")
            .into_iter()
            .map(|r| r.expect("every item evaluated"))
            .collect()
    }
}

/// One result per workload member -> one result for the set; phase ids are prefixed on collision.
fn merge(parts: Vec<EvalResult>, names: &[&str]) -> EvalResult {
    let mut it = parts.into_iter().zip(names);
    let (mut r, _) = it.next().expect("at least one workload member");
    for (p, name) in it {
        let preset = name.split(':').next().unwrap_or(name);
        let mut map: BTreeMap<String, Id> = BTreeMap::new();
        for ph in &p.phases {
            let free = |c: &Id| !r.phases.iter().any(|q| q.phase == *c) && !map.values().any(|v| v == c);
            let id = std::iter::once(ph.phase.to_string())
                .chain((1..).map(|k| if k == 1 { format!("{preset}.{}", ph.phase) } else { format!("{preset}.{}.{k}", ph.phase) }))
                .filter_map(|c| Id::new(c).ok())
                .find(free)
                .expect("a free phase id");
            map.insert(ph.phase.to_string(), id);
        }
        let fix = |id: &mut Id| {
            if let Some(n) = map.get(id.as_str()) {
                *id = n.clone();
            }
        };
        if r.status == Status::Ok && p.status != Status::Ok {
            r.status = p.status;
        }
        r.stage_reached = r.stage_reached.min(p.stage_reached);
        r.phases.extend(p.phases.into_iter().map(|mut x| {
            fix(&mut x.phase);
            x
        }));
        r.ops.extend(p.ops.into_iter().map(|mut x| {
            fix(&mut x.phase);
            x
        }));
        r.sim.extend(p.sim.into_iter().map(|mut x| {
            fix(&mut x.phase);
            x
        }));
        r.errors.extend(p.errors);
        r.warnings.extend(p.warnings);
        r.violations.extend(p.violations);
        r.interval.corner_flips.extend(p.interval.corner_flips);
        r.physical = r.physical.or(p.physical);
        for (k, v) in p.features {
            r.features.entry(k).or_insert(v);
        }
        match (&mut r.invariants, p.invariants) {
            (Some(a), Some(b)) => a.checks.extend(b.checks),
            (a @ None, b) => *a = b,
            _ => {}
        }
        for (k, v) in p.timing.stages_s {
            *r.timing.stages_s.entry(k).or_default() += v;
        }
        r.timing.cache_hits += p.timing.cache_hits;
        r.audit.reasons.extend(p.audit.reasons);
        r.audit
            .extrapolated_components
            .extend(p.audit.extrapolated_components);
    }
    for v in [&mut r.errors, &mut r.warnings, &mut r.violations] {
        dedup(v);
    }
    r
}

/// One error per root cause (06 §6.5): identical diagnostics from several workload members are reported once.
fn dedup(v: &mut Vec<ResultError>) {
    let mut seen = std::collections::BTreeSet::new();
    v.retain(|e| {
        seen.insert((
            e.diag.code.clone(),
            e.diag.path.clone(),
            e.diag.message.clone(),
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::result_with;
    use kiln_trace::result::Feature;

    struct Fake;

    impl Engine for Fake {
        fn evaluate(&self, req: &EngineRequest) -> EvalResult {
            let phase = req.member.scenario.as_str().to_string();
            if req.design.doc.name.as_str().contains("panic") {
                panic!("boom");
            }
            if req.design.doc.name.as_str().contains("slow") {
                std::thread::sleep(Duration::from_secs(5));
            }
            let tdp = if req.design.doc.name.as_str().contains("tpu") {
                200.0
            } else {
                400.0
            };
            let tps = if req.design.doc.name.as_str().contains("tpu") {
                120.0
            } else {
                80.0
            };
            let mut r = result_with(&[(&phase, tps)], 800.0, tdp);
            r.provenance = engine::provenance(req);
            if let Ok(st) = req.options.stack_for(req.model.exec_model) {
                r.provenance.flags.insert("stack".into(), st.label());
            }
            r
        }

        fn version(&self) -> String {
            "fake".into()
        }
    }

    fn full() -> Options {
        Options::from_value(&json!({"profile": "full"})).unwrap()
    }

    fn session(cache: Option<PathBuf>) -> Session {
        let cfg = SessionConfig {
            no_cache: cache.is_none(),
            cache_dir: cache,
            ..Default::default()
        };
        Session::with_engine(cfg, Arc::new(Fake)).unwrap()
    }

    fn renamed(name: &str, path: &str) -> DesignInput {
        let p = inputs::default_designs_dir().join(path);
        let text = std::fs::read_to_string(p).unwrap();
        DesignInput::Str(text.replacen("name: \"", &format!("name: \"{name}-"), 1))
    }

    #[test]
    fn merged_phase_ids_stay_unique() {
        let names = ["llama3_8b:decode_b1", "llama3_8b:decode_b1+weights=fp8_e4m3", "llama3_8b:decode_b1+weights=int4_g128", "llama3_8b:decode_b1+kv=fp8_e4m3"];
        let parts = [100.0, 10.0, 1000.0, 1.0].iter().map(|&t| result_with(&[("decode_b1", t)], 800.0, 400.0)).collect();
        let r = merge(parts, &names);
        let ids: std::collections::BTreeSet<&str> = r.phases.iter().map(|p| p.phase.as_str()).collect();
        assert_eq!(ids.len(), 4, "{ids:?}");
        let tps: Vec<f64> = r.phases.iter().map(|p| p.tokens_per_s.central).collect();
        assert_eq!(tps, [100.0, 10.0, 1000.0, 1.0]);
    }

    #[test]
    fn sim_engine_scores_the_baseline_at_one() {
        let s = Session::new(SessionConfig {
            no_cache: true,
            ..Default::default()
        })
        .unwrap();
        let r = s.evaluate(
            &DesignInput::Str("a100".into()),
            &WorkloadInput::Str("llama3_8b:decode_b1".into()),
            &full(),
        );
        assert_eq!(r.status, Status::Ok, "{:?}", r.errors);
        assert_eq!(r.stage_reached, Stage::S3);
        assert!((r.score - 1.0).abs() < 1e-12, "{}", r.score);
        assert_eq!(r.phases.len(), 1);
        assert!(r.phases[0].time_s.central > 0.0);
        assert_eq!(r.provenance.git_hash, crate::GIT_HASH);
        assert!(r.provenance.mapping_hash.is_some());
        assert!(r.features.contains_key("peak_flops_bf16"));
        let b = Options::from_value(&json!({"profile": "full", "tier": "B"})).unwrap();
        let r = s.evaluate(
            &DesignInput::Str("a100".into()),
            &WorkloadInput::Str("llama3_8b:decode_b1".into()),
            &b,
        );
        assert_eq!(r.errors[0].diag.code, engine::NOT_IMPLEMENTED);
        assert!(engine::SimEngine.version().contains(crate::GIT_HASH));
    }

    #[test]
    fn scores_against_baseline_and_caches() {
        let dir = tempfile::tempdir().unwrap();
        let s = session(Some(dir.path().into()));
        let wl = WorkloadInput::Str("standard".into());
        let base = s.evaluate(&DesignInput::Str("a100".into()), &wl, &full());
        assert_eq!(base.status, Status::Ok, "{:?}", base.errors);
        assert_eq!(base.score, 1.0);
        assert_eq!(base.phases.len(), 4);
        assert_eq!(base.timing.cache_hits, 0);
        let again = s.evaluate(&DesignInput::Str("a100".into()), &wl, &full());
        assert_eq!(again.timing.cache_hits, 4);
        assert_eq!(
            again.deterministic_hash().len(),
            base.deterministic_hash().len()
        );
        assert_eq!(again.score, base.score);
        let tpu = s.evaluate(&DesignInput::Str("tpu_v5e".into()), &wl, &full());
        assert_eq!(tpu.status, Status::Ok, "{:?}", tpu.violations);
        assert!((tpu.score - 1.5).abs() < 1e-12);
    }

    /// The A100 reference with `edit` applied to its source text.
    fn a100_with(name: &str, from: &str, to: &str) -> DesignInput {
        let DesignInput::Str(t) = renamed(name, "reference/a100_sxm4_40gb.json5") else {
            unreachable!()
        };
        assert!(t.contains(from), "{from}");
        DesignInput::Str(t.replacen(from, to, 1))
    }

    fn codes(r: &EvalResult) -> Vec<&str> {
        r.violations.iter().map(|v| v.diag.code.as_str()).collect()
    }

    #[test]
    fn offchip_memory_is_pinned_to_the_baseline() {
        let s = session(None);
        let wl = WorkloadInput::Str("llama3_8b:decode_b1".into());
        let ev = |d: &DesignInput, o: serde_json::Value| {
            let mut o = o;
            o["profile"] = json!("full");
            s.evaluate(d, &wl, &Options::from_value(&o).unwrap())
        };
        let a100 = DesignInput::Str("a100_40gb".into());
        let same = ev(&a100, json!({}));
        assert_eq!(same.status, Status::Ok, "{:?}", same.violations);
        assert_eq!(same.score, 1.0);
        let feat = |r: &EvalResult, k: &str| match r.features.get(k) {
            Some(Feature::Scalar(x)) => *x,
            f => panic!("{k}: {f:?}"),
        };
        assert!((feat(&same, fitness::OFFCHIP_BW) - 1555.2e9).abs() < 1.0);
        assert_eq!(feat(&same, fitness::OFFCHIP_BYTES), 40.0 * 2f64.powi(30));

        let pins = a100_with("pins", "\"2.43Gbps\"", "\"4.86Gbps\"");
        let fast = ev(&pins, json!({}));
        assert_eq!(fast.status, Status::Envelope);
        assert_eq!(codes(&fast), ["E-ENV-0007"]);
        assert_eq!(fast.score, 0.0);
        let v = &fast.violations[0];
        assert_eq!(v.unit.as_deref(), Some("B/s"));
        assert!((v.value.unwrap() / v.limit.unwrap() - 2.0).abs() < 1e-12);
        assert!(
            v.diag.hint.as_ref().unwrap().contains("1555.2 GB/s"),
            "{v:?}"
        );
        let graded = ev(&pins, json!({"invalid_score": "graded"}));
        assert!((graded.score + 2.0).abs() < 1e-9, "{}", graded.score);

        let stack = a100_with("stack", "disabled: [\"hbm5\"], ", "");
        let more = ev(&stack, json!({}));
        assert_eq!(more.status, Status::Envelope);
        assert_eq!(codes(&more), ["E-ENV-0007", "E-ENV-0008"]);
        assert_eq!(more.score, 0.0);
        assert_eq!(more.violations[1].unit.as_deref(), Some("B"));

        // explicit_envelope sets its own off-chip limits; `null` disables one.
        let roomy = json!({"fitness": {"kind": "explicit_envelope", "envelope": {"die_mm2": 900.0,
            "power_w": 450.0, "offchip_bw": 4e12, "offchip_bytes": 64.0 * 2f64.powi(30)}}});
        for d in [&pins, &stack] {
            let r = ev(d, roomy.clone());
            assert_eq!(r.status, Status::Ok, "{:?}", r.violations);
        }
        let tight =
            json!({"fitness": {"kind": "explicit_envelope", "envelope": {"die_mm2": 900.0}}});
        assert_eq!(codes(&ev(&pins, tight)), ["E-ENV-0007"]);
        let unpinned =
            json!({"fitness": {"envelope": {"offchip_bw": null, "offchip_bytes": null}}});
        assert_eq!(ev(&stack, unpinned).status, Status::Ok);
        let bw_only = json!({"fitness": {"envelope": {"offchip_bytes": null}}});
        assert_eq!(codes(&ev(&stack, bw_only)), ["E-ENV-0007"]);
        assert_eq!(
            ev(&stack, json!({"fitness": {"kind": "baseline_relative"}})).status,
            Status::Ok
        );

        // A reference with more bandwidth than the baseline fails its matched envelope by design.
        let v6e = ev(&DesignInput::Str("tpu_v6e".into()), json!({}));
        assert_eq!(codes(&v6e), ["E-ENV-0007"], "{:?}", v6e.violations);
        let v = &v6e.violations[0];
        assert!((v.value.unwrap() - 1638.4e9).abs() < 1e6, "{v:?}");
        assert!(feat(&v6e, fitness::OFFCHIP_BYTES) < 40.0 * 2f64.powi(30));
    }

    #[test]
    fn s0_errors_timeout_and_panic_are_isolated() {
        let s = session(None);
        let wl = WorkloadInput::Str("llama3_8b:decode_b1".into());
        let mut opts = full();
        opts.timeout_s.a = 0.2;
        let items = vec![
            (
                renamed("panic", "reference/a100_sxm4_40gb.json5"),
                wl.clone(),
            ),
            (
                DesignInput::Str("{schema: \"kiln.hw/1.0\"}".into()),
                wl.clone(),
            ),
            (
                renamed("slow", "reference/a100_sxm4_40gb.json5"),
                wl.clone(),
            ),
            (
                DesignInput::Str("a100".into()),
                WorkloadInput::Str("nope:x".into()),
            ),
            (DesignInput::Str("a100".into()), wl.clone()),
        ];
        let rs = s.evaluate_batch(&items, &opts, Some(3));
        let st: Vec<_> = rs.iter().map(|r| r.status).collect();
        assert_eq!(
            st,
            [
                Status::InternalError,
                Status::Invalid,
                Status::Timeout,
                Status::Invalid,
                Status::Ok
            ]
        );
        assert!(rs[0].errors[0].diag.message.contains("boom"));
        assert_eq!(rs[2].errors[0].diag.code, TIMEOUT_CODE);
        assert!(rs.iter().all(|r| r.status == Status::Ok || r.score == 0.0));
        let one = s.evaluate(&items[4].0, &items[4].1, &opts);
        assert_eq!(one.deterministic_hash(), rs[4].deterministic_hash());
    }

    #[test]
    fn validate_tier_and_calibration() {
        let s = session(None);
        let wl = WorkloadInput::Str("llama3_8b:decode_b1".into());
        let o = Options::from_value(&json!({"tier": "validate", "profile": "full"})).unwrap();
        let r = s.evaluate(&DesignInput::Str("a100".into()), &wl, &o);
        assert_eq!((r.status, r.stage_reached), (Status::Ok, Stage::S0));
        assert!(
            s.validate(&DesignInput::Str("a100".into()), "full")
                .iter()
                .all(|e| e.diag.severity != Severity::Error)
        );
        let errs = s.validate(&DesignInput::Str("a100".into()), "search");
        assert_eq!(errs[0].diag.code, "E-IR-1102");
        assert!(resolve_calibration(Some("Bad Id!")).is_err());
        let generic = resolve_calibration(None).unwrap();
        assert!(generic.set.is_some() && generic.id == DEFAULT_CALIBRATION);
        assert_ne!(
            generic.hash,
            resolve_calibration(Some("assumed-v0")).unwrap().hash
        );
        assert!(
            resolve_calibration(Some("platform:a100_40gb"))
                .unwrap()
                .set
                .is_some()
        );
        assert!(resolve_calibration(Some("null")).unwrap().params.is_some());
        assert!(resolve_calibration(Some("no-such-set")).is_err());
    }

    fn real() -> Session {
        Session::new(SessionConfig {
            no_cache: true,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn simulated_baseline_identity_across_kinds_and_interval_modes() {
        let s = real();
        let a100 = DesignInput::Str("a100_40gb".into());
        let wl = WorkloadInput::Str("llama3_8b:decode_b1".into());
        let kinds = [
            "matched_envelope",
            "explicit_envelope",
            "perf_per_watt",
            "perf_per_area",
            "pareto",
            "baseline_relative",
        ];
        for interval in ["none", "sensitivity", "corners"] {
            for kind in kinds {
                for basis in ["central", "low"] {
                    let o = Options::from_value(&json!({"profile": "full", "interval": interval,
                        "fitness": {"kind": kind, "interval_basis": basis, "baseline": "a100_40gb",
                            // published 826 mm^2 at the area fit's tolerance (04 §12.3 +-5%): the model's A100 is 832
                            "envelope": {"die_mm2": 826.0 * 1.05}}}))
                    .unwrap();
                    let r = s.evaluate(&a100, &wl, &o);
                    assert_eq!(r.status, Status::Ok, "{kind} {interval}: {:?}", r.errors);
                    let si = r.score_interval.unwrap();
                    assert_eq!(si.central, 1.0, "{kind} {interval}");
                    let want = if basis == "central" { 1.0 } else { si.low };
                    assert_eq!(r.score, want, "{kind} {interval} {basis}");
                    if interval == "none" {
                        assert_eq!((si.low, si.high), (1.0, 1.0), "{kind}");
                    } else {
                        assert!(si.low <= 1.0 && si.high >= 1.0, "{kind} {interval}: {si:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn calibration_set_changes_both_sides_consistently() {
        let s = real();
        let wl = WorkloadInput::Str("llama3_8b:decode_b8".into());
        let wset = inputs::resolve_workload(&wl).unwrap();
        let mut scores = vec![];
        for cal in ["generic-v1", "platform:a100_40gb", "assumed-v0", "null"] {
            let o = Options::from_value(&json!({"profile": "full", "calibration": cal,
                "fitness": {"kind": "baseline_relative"}}))
            .unwrap();
            let hash = resolve_calibration(Some(cal)).unwrap().hash;
            let cand = s.evaluate(&DesignInput::Str("h100_sxm5_80gb".into()), &wl, &o);
            assert_eq!(cand.status, Status::Ok, "{cal}: {:?}", cand.errors);
            let base = s.baseline("a100_40gb", &wset, &o);
            assert_eq!(cand.provenance.calibration_hash, hash);
            assert_eq!(base.provenance.calibration_hash, hash);
            assert!(fitness::basis_mismatch(&cand, &base).is_none());
            let own = |d: &str| {
                let mut r = s.evaluate(&DesignInput::Str(d.into()), &wl, &o);
                assert_eq!(r.provenance.calibration_hash, hash);
                r.phases.remove(0).tokens_per_s.central
            };
            let (c, b) = (own("h100_sxm5_80gb"), own("a100_40gb"));
            assert_eq!(base.phases[0].tokens_per_s.central, b, "{cal}");
            assert_eq!(cand.score, c / b, "{cal}");
            scores.push((cal, b, cand.score));
        }
        // Sets move the baseline too (no stale baseline across sets); generic-v1 and the A100 platform set
        // agree on this memory-bound A100 step (same DRAM and launch terms).
        let mut bases: Vec<f64> = scores.iter().map(|x| x.1).collect();
        bases.dedup();
        assert_eq!(bases.len(), 3, "{scores:?}");
        let rel = |cal: &str| {
            Options::from_value(&json!({"profile": "full", "calibration": cal,
                "fitness": {"kind": "baseline_relative"}}))
            .unwrap()
        };
        let o = rel("generic-v1");
        let mut cand = s.evaluate(&DesignInput::Str("h100_sxm5_80gb".into()), &wl, &o);
        let other = rel("null");
        let base = s.baseline("a100_40gb", &wset, &other);
        fitness::apply(&mut cand, Some(("a100_40gb", &base)), &o);
        assert_eq!(cand.errors.last().unwrap().diag.code, fitness::ASYM_CODE);
    }

    #[test]
    fn ideal_and_realistic_scores_on_the_sim_engine() {
        let s = real();
        let wl = WorkloadInput::Str("llama3_8b:prefill_b1".into());
        let wset = inputs::resolve_workload(&wl).unwrap();
        let o = full();
        let stack_id = |l: &str| l.split('@').next().unwrap().to_string();
        let a100 = s.evaluate(&DesignInput::Str("a100_40gb".into()), &wl, &o);
        assert_eq!(a100.status, Status::Ok, "{:?}", a100.errors);
        assert_eq!(a100.score_interval.unwrap().central, 1.0);
        let rs = a100.score_realistic.clone().expect("realistic score");
        assert_eq!(rs.interval.central, 1.0);
        assert_eq!(stack_id(&rs.candidate_stack), "pytorch_cuda_graph_sdpa");
        let c = a100.score_components.as_ref().unwrap();
        assert_eq!(stack_id(c.candidate_stack.as_ref().unwrap()), "kiln_ideal");
        assert_eq!(c.candidate_stack, c.baseline_stack);

        // v6e has more off-chip bandwidth than the A100-40GB: outside the matched envelope, so the reference
        // comparison is baseline_relative (diagnostic).
        let matched = s.evaluate(&DesignInput::Str("tpu_v6e".into()), &wl, &o);
        assert_eq!(matched.status, Status::Envelope);
        assert_eq!(matched.violations[0].diag.code, "E-ENV-0007");
        let o = Options::from_value(
            &json!({"profile": "full", "fitness": {"kind": "baseline_relative"}}),
        )
        .unwrap();
        // The A100 baseline pays PyTorch's unfused kernels under the realistic stacks; a TPU pays XLA's fused
        // program, which charges fewer extra kernels: the realistic score is the higher one.
        let tpu = s.evaluate(&DesignInput::Str("tpu_v6e".into()), &wl, &o);
        assert_eq!(tpu.status, Status::Ok, "{:?}", tpu.errors);
        let rs = tpu.score_realistic.clone().expect("realistic score");
        assert_eq!(stack_id(&rs.candidate_stack), "xla_tpu_fused");
        assert_eq!(stack_id(&rs.baseline_stack), "pytorch_cuda_graph_sdpa");
        let ideal_base = s.baseline("a100_40gb", &wset, &o);
        let own_base = s.baseline("a100_40gb", &wset, &o.with_stack(STACK_OWN));
        let tps = |r: &EvalResult| r.phases[0].tokens_per_s.central;
        assert!(
            tps(&own_base) < tps(&ideal_base),
            "PyTorch costs the A100 time"
        );
        assert!(
            rs.score > tpu.score,
            "realistic {} vs ideal {}",
            rs.score,
            tpu.score
        );

        // Mismatched stacks: the candidate under PyTorch over the kiln_ideal baseline is refused.
        let pt = o.with_stack("pytorch_cuda_graph_sdpa");
        let mut cand = s.evaluate(&DesignInput::Str("h100_sxm5_80gb".into()), &wl, &pt);
        assert_eq!(cand.status, Status::Ok, "{:?}", cand.errors);
        fitness::apply(&mut cand, Some(("a100_40gb", &ideal_base)), &pt);
        assert_eq!(cand.errors.last().unwrap().diag.code, fitness::ASYM_CODE);
        assert!(fitness::stack_mismatch(&cand, &ideal_base, &o).is_some());
    }

    struct Flaky(AtomicUsize);

    impl Engine for Flaky {
        fn evaluate(&self, req: &EngineRequest) -> EvalResult {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("transient");
            }
            Fake.evaluate(req)
        }

        fn version(&self) -> String {
            "flaky".into()
        }
    }

    #[test]
    fn failed_baselines_are_not_memoized() {
        let s = Session::with_engine(
            SessionConfig {
                no_cache: true,
                ..Default::default()
            },
            Arc::new(Flaky(AtomicUsize::new(0))),
        )
        .unwrap();
        let wl =
            inputs::resolve_workload(&WorkloadInput::Str("llama3_8b:decode_b1".into())).unwrap();
        assert_eq!(
            s.baseline("a100", &wl, &full()).status,
            Status::InternalError
        );
        assert_eq!(s.baseline("a100", &wl, &full()).status, Status::Ok);
    }

    #[test]
    fn baseline_memo_follows_the_design_contents() {
        let s = session(None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.json5");
        let DesignInput::Str(t) = renamed("gpu", "reference/a100_sxm4_40gb.json5") else {
            unreachable!()
        };
        std::fs::write(&path, &t).unwrap();
        let wl =
            inputs::resolve_workload(&WorkloadInput::Str("llama3_8b:decode_b1".into())).unwrap();
        let name = path.to_str().unwrap();
        let tps = |r: &EvalResult| r.phases[0].tokens_per_s.central;
        assert_eq!(tps(&s.baseline(name, &wl, &full())), 80.0);
        std::fs::write(&path, t.replacen("name: \"gpu-", "name: \"tpu-", 1)).unwrap();
        assert_eq!(tps(&s.baseline(name, &wl, &full())), 120.0);
    }

    struct Busy {
        active: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Engine for Busy {
        fn evaluate(&self, req: &EngineRequest) -> EvalResult {
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(800));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Fake.evaluate(req)
        }

        fn version(&self) -> String {
            "busy".into()
        }
    }

    #[test]
    fn timed_out_evaluations_keep_their_worker() {
        let eng = Arc::new(Busy {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let cfg = SessionConfig {
            no_cache: true,
            ..Default::default()
        };
        let s = Session::with_engine(cfg, eng.clone()).unwrap();
        let wl = WorkloadInput::Str("llama3_8b:decode_b1".into());
        let items: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|n| (renamed(n, "reference/a100_sxm4_40gb.json5"), wl.clone()))
            .collect();
        let mut opts = full();
        opts.timeout_s.a = 0.05;
        let rs = s.evaluate_batch(&items, &opts, Some(1));
        assert!(rs.iter().all(|r| r.status == Status::Timeout));
        assert_eq!(eng.peak.load(Ordering::SeqCst), 1);
        assert_eq!(eng.active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn sim_engine_honours_the_deadline() {
        let s = real();
        let Ok(l) = s.load(&DesignInput::Str("a100".into()), "full") else {
            panic!("a100 loads")
        };
        let wl =
            inputs::resolve_workload(&WorkloadInput::Str("llama3_8b:decode_b1".into())).unwrap();
        let req = EngineRequest {
            design: l.design,
            model: l.model,
            member: Arc::new(wl.members[0].clone()),
            options: Arc::new(full()),
            calibration: s.calibration.clone(),
            deadline: Instant::now(),
        };
        let r = engine::SimEngine.evaluate(&req);
        assert_eq!(r.status, Status::Timeout, "{:?}", r.errors);
        assert_eq!(r.errors[0].diag.code, TIMEOUT_CODE);
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn sim_engine_scores_every_seed() {
        let s = real();
        let o = Options::from_value(
            &json!({"profile": "full", "seeds": [0, 1, 2], "timeout_s": {"A": 120.0}}),
        )
        .unwrap();
        let r = s.evaluate(
            &DesignInput::Str("a100".into()),
            &WorkloadInput::Str("llama3_8b:decode_b1".into()),
            &o,
        );
        assert_eq!(r.status, Status::Ok, "{:?}", r.errors);
        let seeds: BTreeMap<String, f64> =
            serde_json::from_str(&r.provenance.flags["seed_scores"]).unwrap();
        assert_eq!(seeds.keys().collect::<Vec<_>>(), ["0", "1", "2"]);
        assert_eq!(r.provenance.seeds, [0, 1, 2]);
    }

    #[test]
    fn median_seed_picks_the_median_or_the_failure() {
        let run = |score: f64, status: Status| {
            let mut r = result_with(&[("decode_b1", score)], 800.0, 400.0);
            r.score = score;
            r.status = status;
            r
        };
        let (r, scores) = engine::median_seed(vec![
            (0, run(3.0, Status::Ok)),
            (1, run(1.0, Status::Ok)),
            (2, run(2.0, Status::Ok)),
        ]);
        assert_eq!(r.score, 2.0);
        assert_eq!(r.audit.seed_spread, Some(1.0));
        assert_eq!(scores, r#"{"0":3.0,"1":1.0,"2":2.0}"#);
        let (r, _) =
            engine::median_seed(vec![(0, run(4.0, Status::Ok)), (1, run(2.0, Status::Ok))]);
        assert_eq!(r.score, 2.0);
        let (r, _) = engine::median_seed(vec![
            (0, run(4.0, Status::Ok)),
            (1, run(0.5, Status::Infeasible)),
        ]);
        assert_eq!(r.status, Status::Infeasible);
    }
}
