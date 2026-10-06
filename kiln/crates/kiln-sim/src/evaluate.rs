//! Library entry point (06 §6.6 cascade S0-S3; S4 audit is M4): design + workload -> `EvalResult`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use kiln_ir::bench::BenchOp;
use kiln_ir::common::{Diagnostic, Id, Severity};
use kiln_ir::hw::{Design, ExpandOptions, HwModel, Profile};
use kiln_ir::wl::{self, PhaseKind, ScenarioMode};
use kiln_map::program::Program;
use kiln_map::HwView;
use kiln_trace::result::{EvalResult, Feature, LayerTime, PhaseResult, PhysicalSummary, ResultError, Stage, Status, Timing};
use kiln_trace::sim::{BindingClass, Floor, FloorKind, Scope, SimResult};
use kiln_trace::{Corner, Interval, Provenance, RESULT_SCHEMA, Tier, TraceLevel};
use kiln_wl::zoo::SuiteMember;

use crate::run::{PhaseRun, SimOptions, TIMEOUT_CODE, simulate};

/// Per-design state reused across phases and workloads.
pub struct Prepared {
    pub design: Arc<Design>,
    pub view: Arc<HwView>,
    /// The design is a published chip of the shipped reference set (by structural hash, never by its own claims):
    /// its physical findings are model residuals (04 §17 rule 7, 08 §H).
    pub published_reference: bool,
}

impl Prepared {
    pub fn new(design: Arc<Design>, hw: Arc<HwModel>) -> Result<Prepared, Diagnostic> {
        let published_reference = published_references().contains(&design.hash);
        Ok(Prepared { design, view: Arc::new(HwView::new(hw)?), published_reference })
    }

    /// S0: expand and validate under `profile`.
    pub fn load(design: Design, profile: Profile) -> Result<Prepared, Vec<Diagnostic>> {
        let report = check_priced(design, profile);
        let errs: Vec<Diagnostic> = report.errors().cloned().collect();
        if !errs.is_empty() {
            return Err(errs);
        }
        let (Some(d), Some(m)) = (report.design, report.model) else {
            return Err(vec![Diagnostic::error("E-INTERNAL", "validation returned no model")]);
        };
        Prepared::new(Arc::new(d), Arc::new(m)).map_err(|e| vec![e])
    }

    pub fn from_file(path: &std::path::Path, profile: Profile) -> Result<Prepared, Vec<Diagnostic>> {
        Prepared::load(kiln_ir::hw::load_file(path)?, profile)
    }
}

/// Structural hashes of the shipped reference designs that are real chips: a published `die_area`, or a cap the
/// vendor does not publish (`assumed`, which the search profile rejects, E-IR-1106).
fn published_references() -> &'static BTreeSet<String> {
    static SET: OnceLock<BTreeSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference");
        let mut files: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json5")).collect();
        files.sort();
        files
            .into_iter()
            .filter_map(|f| kiln_ir::hw::load_file(f).ok())
            .filter_map(|d| {
                let die_area = d.doc.meta.claims.iter().any(|c| c.metric == "die_area");
                let hash = d.hash.clone();
                let m = kiln_ir::hw::check(d, Profile::Reference, &ExpandOptions::default()).model?;
                (die_area || m.power_domains.iter().any(|p| p.assumed.is_some())).then_some(hash)
            })
            .collect()
    })
}

/// 01 §18 validation where `search` prices "less is better" overrides against kiln-phys derived values
/// (00 decision 3 E-IR-UNPRICED): allowed at >= derived, rejected below.
pub fn check_priced(design: Design, profile: Profile) -> kiln_ir::hw::Report {
    kiln_ir::hw::check_priced(design, profile, &ExpandOptions::default(), Some(&kiln_phys::pricing::pricer))
}

pub fn base_provenance(design_hash: &str, workload_hash: &str, opts: &SimOptions) -> Provenance {
    Provenance {
        kiln_version: kiln_trace::KILN_VERSION.into(),
        git_hash: opts.git_hash.clone(),
        design_hash: design_hash.into(),
        workload_hash: workload_hash.into(),
        calibration_hash: String::new(),
        calibration_id: None,
        mapping_hash: None,
        options_hash: None,
        tier: Tier::A,
        seeds: vec![opts.seed],
        trust_level: Default::default(),
        window: None,
        chunk_bytes: None,
        flags: BTreeMap::new(),
    }
}

/// Tokens a step produces (decode) or consumes (prefill) for throughput.
pub fn step_tokens(s: &wl::Scenario) -> u64 {
    match &s.mode {
        ScenarioMode::Snapshot { kind: PhaseKind::Decode, seqs } => seqs.seqs(),
        ScenarioMode::Snapshot { seqs, .. } => seqs.tokens(),
        _ => 1,
    }
}

/// One phase of a suite member, whole-step (or layer when the resident set does not fit).
pub fn simulate_member(p: &Prepared, member: &SuiteMember, opts: &SimOptions) -> Result<(PhaseRun, Program), Vec<Diagnostic>> {
    let (model, sc) = (member.model(), member.scenario());
    let (_, mut lg, st) = kiln_wl::evaluate_snapshot(model, sc)?;
    // Low-precision operands no MAC mode of the design runs are converted explicitly (02 §5.7, 03 §2.7).
    kiln_wl::convert::insert_converts(&mut lg, &p.view.mac_modes(opts.stack(&p.view).dequantize)).map_err(|e| vec![e])?;
    let isolated = matches!(sc.eval_mode, wl::EvalMode::Isolated { .. });
    let prog = if isolated { Program::isolated(&lg) } else { Program::whole_step(model, &lg, opts.window) }.map_err(|e| vec![e])?;
    let r = st.resident;
    let prog = prog.with_resident(r.weights + r.kv_cache + r.constants);
    let whash = wl::workload_hash(model, sc, None);
    let prov = base_provenance(&p.design.hash, &whash, opts);
    let phase = Id::new(member.scenario.as_str()).map_err(|e| vec![e])?;
    let scope = if isolated { Scope::Op } else { Scope::Step };
    let mut run = simulate(&p.view, &prog, phase.clone(), scope, opts, prov.clone()).map_err(|e| vec![e])?;
    if let (Scope::Step, Some((g, need, cap))) = (scope, &run.report.capacity_overflow) {
        if !opts.layer_scope_fallback {
            return Err(vec![
                Diagnostic::error("E-MAP-CAP-001", format!("{}: resident model state {need} B exceeds {g} capacity {cap} B", member.name))
                    .at(g.clone())
                    .hint("the phase is infeasible on this design: add memory capacity or shrink the batch / context"),
            ]);
        }
        run = simulate(&p.view, &prog, phase, Scope::Layer, opts, prov).map_err(|e| vec![e])?;
    }
    Ok((run, prog))
}

/// A GEMM-family bench descriptor in isolation (`scope: op`, calibration only).
pub fn simulate_bench_op(p: &Prepared, op: &BenchOp, opts: &SimOptions) -> Result<PhaseRun, Diagnostic> {
    let prog = Program::bench_op(op)?;
    let prov = base_provenance(&p.design.hash, &op.key(), opts);
    let phase = Id::new(kiln_trace::meas::legacy::slug(&op.legacy_name().unwrap_or_else(|| "op".into()))).unwrap_or_else(|_| Id::new("op").expect("id"));
    simulate(&p.view, &prog, phase, Scope::Op, opts, prov)
}

fn phase_result(run: &PhaseRun, prog: &Program, tokens: u64) -> PhaseResult {
    let c = &run.central;
    let t = run.time;
    let clock = c.clocks.iter().map(|x| x.hz).fold(0.0, f64::max);
    let mut breakdown: BTreeMap<String, f64> = BTreeMap::new();
    for (k, v) in &c.bottleneck.time_by_binding {
        let key = match k {
            BindingClass::Compute => "compute".to_string(),
            BindingClass::Dram => "mem:offchip".into(),
            BindingClass::Port => "mem:onchip".into(),
            BindingClass::Link => "link".into(),
            BindingClass::Overhead => "overhead".into(),
            other => format!("{other:?}").to_lowercase(),
        };
        *breakdown.entry(key).or_default() += v / c.makespan_s;
    }
    let energy = run.energy;
    let floors = vec![
        Floor { kind: FloorKind::Roofline, path: None, seconds: c.t_a0_s },
        Floor { kind: FloorKind::Compute, path: Some("a2".into()), seconds: c.t_a2_s },
    ];
    let per_layer = match (prog.window, &run.central.scope) {
        (Some((w, _)), Scope::Step | Scope::Layer) => {
            let mid = w / 2;
            let tm: f64 = c.groups.iter().filter(|g| g.ops.first().is_some_and(|o| o.as_str().contains(&format!(".i{mid}.")))).map(|g| g.end_s - g.start_s).sum();
            if tm > 0.0 { vec![LayerTime { layer: mid, time_s: tm }] } else { vec![] }
        }
        _ => vec![],
    };
    // A layer-scope run (diagnostic fallback) times one layer: it never yields step throughput.
    let tk = if c.scope == Scope::Layer { 0.0 } else { tokens as f64 };
    PhaseResult {
        phase: c.phase.clone(),
        scope: c.scope,
        time_s: t,
        tokens_per_s: t.recip_scaled(tk),
        energy_j: energy,
        tokens_per_j: energy.recip_scaled(tk),
        avg_power_w: Interval::from_corners(
            c.power.avg_w,
            run.low.as_ref().map_or(c.power.avg_w, |l| l.power.avg_w),
            run.high.as_ref().map_or(c.power.avg_w, |h| h.power.avg_w),
        ),
        clock_hz: Interval::point(clock),
        floors,
        roofline_frac: if c.makespan_s > 0.0 { c.t_a0_s / c.makespan_s } else { 0.0 },
        bound_breakdown: breakdown,
        per_layer,
        trusted: false,
    }
}

fn fail(status: Status, stage: Stage, prov: Provenance, errs: Vec<Diagnostic>) -> EvalResult {
    EvalResult {
        schema: RESULT_SCHEMA.into(),
        status,
        score: 0.0,
        score_interval: None,
        score_components: None,
        score_realistic: None,
        interval: Default::default(),
        stage_reached: stage,
        tier: Some(Tier::A),
        phases: vec![],
        ops: vec![],
        physical: None,
        features: BTreeMap::new(),
        violations: vec![],
        errors: errs.into_iter().map(ResultError::from).collect(),
        warnings: vec![],
        audit: Default::default(),
        trace: None,
        provenance: prov,
        timing: Timing::default(),
        calibration: None,
        invariants: None,
        sim: vec![],
    }
}

/// S1-S3 for a prepared design over workload members; the score is the geomean of central tokens/s (no
/// baseline: fitness ratios are the caller's, 06 §6.4).
pub fn evaluate_prepared(p: &Prepared, members: &[SuiteMember], opts: &SimOptions) -> EvalResult {
    let t0 = Instant::now();
    let whash = match members {
        [m] => wl::workload_hash(m.model(), m.scenario(), None),
        ms => kiln_ir::common::content_hash(
            "wl1-",
            &serde_json::Value::from(ms.iter().map(|m| wl::workload_hash(m.model(), m.scenario(), None)).collect::<Vec<_>>()),
        ),
    };
    let prov = base_provenance(&p.design.hash, &whash, opts);
    let mut res = fail(Status::Ok, Stage::S3, prov.clone(), vec![]);
    let mut phase_power: Vec<PhasePower> = vec![];
    let mut cap_findings: Vec<Diagnostic> = vec![];
    let mut timing = BTreeMap::new();
    let mut logs = vec![];
    let mut all_ok = true;
    for m in members {
        let tokens = step_tokens(m.scenario());
        match simulate_member(p, m, opts) {
            Ok((run, prog)) => {
                for w in &run.report.warnings {
                    res.warnings.push(ResultError::from(w.clone()));
                }
                let fails: Vec<Diagnostic> = [Some(&run.central), run.low.as_ref(), run.high.as_ref()]
                    .into_iter()
                    .flatten()
                    .flat_map(|r| r.invariants.failures().map(|c| Diagnostic::error(c.id.code(), c.message.clone()).at(format!("{}:{:?}", r.phase, r.corner))))
                    .collect();
                if !fails.is_empty() {
                    all_ok = false;
                    res.status = Status::FloorViolation;
                    res.violations.extend(fails.into_iter().map(ResultError::from));
                }
                if run.central.scope == Scope::Layer {
                    all_ok = false;
                    res.status = Status::Infeasible;
                    res.errors.push(ResultError::from(
                        Diagnostic::error("E-MAP-CAP-001", format!("{}: the step does not fit; only a layer-scope diagnostic was run", m.name))
                            .hint("layer-scope results are never scored"),
                    ));
                }
                phase_power.push(cap_power(p, &run));
                // 04 §8: every enforced cap bounds its own members' power at the clocks the phase ran at, whether
                // or not the solve throttled.
                let central: Vec<f64> = run.caps_w.iter().map(|c| c.0).collect();
                for x in p.view.phys.cap_excess(&central) {
                    cap_findings.push(
                        Diagnostic::error("E-MAP-POWER-CAP", format!("{}: {} draws {:.1} W at the clocks the phase ran at, over its {:.0} W cap", m.name, x.path, x.power_w, x.cap_w))
                            .at(x.path.clone())
                            .hint("lower the base clock, cut leakage or raise the cap"),
                    );
                }
                let pr = phase_result(&run, &prog, tokens);
                logs.push(pr.tokens_per_s);
                res.invariants.get_or_insert_with(|| run.central.invariants.clone());
                res.interval = run.interval.clone();
                res.calibration.get_or_insert_with(Default::default).uncalibrated_time_s.insert(m.scenario.to_string(), run.central.calibration.uncalibrated_makespan_s.unwrap_or(0.0));
                res.calibration.as_mut().expect("set").contributions.insert(m.scenario.to_string(), run.central.calibration.params.clone());
                res.phases.push(pr);
                if opts.trace >= TraceLevel::Summary {
                    res.sim.push(run.central.clone());
                    res.sim.extend(run.low.clone());
                    res.sim.extend(run.high.clone());
                }
            }
            Err(errs) => {
                all_ok = false;
                let invalid = errs.iter().any(|e| e.code.starts_with("E-WL") || e.code.starts_with("E-IR"));
                let timeout = errs.iter().any(|e| e.code == TIMEOUT_CODE);
                res.status = if timeout { Status::Timeout } else if invalid { Status::Invalid } else { Status::Infeasible };
                res.errors.extend(errs.into_iter().filter(|e| e.severity == Severity::Error).map(ResultError::from));
                if timeout {
                    break;
                }
            }
        }
    }
    let (phys, mut findings, thermal) = physical(p, &phase_power, opts.interval != kiln_trace::IntervalMethod::None);
    findings.extend(cap_findings);
    res.physical = phys;
    res.warnings.extend(thermal.map(ResultError::from));
    for d in findings {
        if p.published_reference {
            // A published chip exists: its physical findings are model residuals, not design errors.
            res.warnings.push(ResultError::from(Diagnostic::warning("W-PHYS-RESIDUAL", format!("{}: {}", d.code, d.message)).at(d.path.clone().unwrap_or_default())));
        } else {
            all_ok = false;
            if res.status == Status::Ok {
                res.status = Status::Envelope;
            }
            res.errors.push(ResultError { section: Some("04".into()), ..ResultError::from(d) });
        }
    }
    timing.insert(Stage::S3, t0.elapsed().as_secs_f64());
    res.timing = Timing { stages_s: timing, cache_hits: 0 };
    if all_ok && !logs.is_empty() {
        let g = |f: fn(&Interval) -> f64| (logs.iter().map(|i| f(i).ln()).sum::<f64>() / logs.len() as f64).exp();
        let si = Interval::from_corners(g(|i| i.central), g(|i| i.low), g(|i| i.high));
        res.score = si.central;
        res.score_interval = Some(si);
        res.features.insert("score_rel_width".into(), Feature::Scalar(si.rel_width()));
        let bound_mem: f64 = res.phases.iter().filter(|p| p.phase.as_str().starts_with("decode")).map(|p| p.bound_breakdown.get("mem:offchip").copied().unwrap_or(0.0)).sum::<f64>();
        res.features.insert("bound_frac_mem".into(), Feature::Scalar(bound_mem / res.phases.iter().filter(|p| p.phase.as_str().starts_with("decode")).count().max(1) as f64));
        res.features.insert("peak_flops_bf16".into(), Feature::Scalar(p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None)));
        res.features.insert("onchip_bytes".into(), Feature::Scalar(p.view.hw.onchip_capacity(None, None).0 as f64));
    } else {
        res.score = 0.0;
    }
    res
}

/// Power of a phase at the reported cap level: (central, low corner, high corner, chip W central, T_j C, thermal
/// runaway, smallest cap margin central, smallest cap margin at the worse corner).
type PhasePower = (f64, f64, f64, f64, f64, bool, f64, f64);

fn cap_power(p: &Prepared, run: &PhaseRun) -> PhasePower {
    let ph = &p.view.phys;
    let level = ph.m3().map_or(kiln_ir::hw::phys::CapLevel::Board, |m| m.power.cap_level);
    let one = |r: &SimResult| {
        let plan = kiln_phys::ClockPlan { hz: r.clocks.iter().map(|c| c.hz).collect(), solved: true, throttled: false };
        let pe = crate::result::phase_energy(&p.view, &r.energy, &r.resources, r.makespan_s, &plan);
        ph.phase_power(&pe, &plan).unwrap_or_default()
    };
    let c = one(&run.central);
    let lo = run.low.as_ref().map_or(c, one);
    let hi = run.high.as_ref().map_or(c, one);
    let caps = ph.caps();
    let margin = |f: fn(&(f64, f64, f64)) -> f64| caps.iter().zip(&run.caps_w).map(|(c, w)| c.cap_w - f(w)).fold(f64::INFINITY, f64::min);
    (c.at(level), lo.at(level).min(hi.at(level)), lo.at(level).max(hi.at(level)), c.chip_w, c.t_j_c, c.runaway || lo.runaway || hi.runaway, margin(|w| w.0), margin(|w| w.1.max(w.2).max(w.0)))
}

/// 04 §10 envelope summary of the design (areas with their corner band, package, power at the phases, density,
/// shoreline, node, margins) and the E-PHYS findings (area overflow, reticle, shoreline, package, power density,
/// thermal limit, missing node tables).
fn physical(p: &Prepared, phases: &[PhasePower], bands: bool) -> (Option<PhysicalSummary>, Vec<Diagnostic>, Option<Diagnostic>) {
    let Some(rep) = p.view.phys.report() else { return (None, vec![], None) };
    let mut findings = rep.problems.clone();
    let die_total: f64 = rep.dies.iter().map(|d| d.area_mm2).sum();
    let peak = phases.iter().fold((0.0f64, 0.0f64, 0.0f64), |a, x| (a.0.max(x.0), a.1.max(x.1), a.2.max(x.2)));
    let chip = phases.iter().map(|x| x.3).fold(0.0, f64::max);
    let tj = phases.iter().map(|x| x.4).fold(f64::NEG_INFINITY, f64::max);
    let q = if die_total > 0.0 { chip / die_total } else { 0.0 };
    let q_hi = if die_total > 0.0 { chip * peak.2 / peak.0.max(1e-9) / rep.dies.iter().map(|d| d.area_low_mm2).sum::<f64>().max(1e-9) } else { 0.0 };
    if q > rep.q_avg_max_w_mm2 {
        findings.push(
            Diagnostic::error("E-PHYS-POWER-DENSITY", format!("average power density {q:.2} W/mm^2 exceeds {:.2} W/mm^2", rep.q_avg_max_w_mm2))
                .hint("spread the power over more area, lower the cap, or declare liquid cooling (power.thermal.cooling)"),
        );
    }

    if phases.iter().any(|x| x.5) {
        findings.push(
            Diagnostic::error("E-PHYS-THERMAL-RUNAWAY", format!("no stable junction temperature: the leakage-temperature loop gain reaches 1 (last estimate {tj:.0} C)"))
                .hint("cut leakage (smaller die, fewer or smaller memories), lower the clock or improve cooling"),
        );
    }
    let tdp = rep.tdp_w.unwrap_or(0.0);
    let reticle = 858.0;
    let max_die = rep.dies.iter().map(|d| d.area_mm2).fold(0.0, f64::max);
    let max_die_hi = rep.dies.iter().map(|d| d.area_high_mm2).fold(0.0, f64::max);
    let mut mc = BTreeMap::new();
    let mut mp = BTreeMap::new();
    mc.insert("reticle_mm2".to_string(), reticle - max_die);
    mp.insert("reticle_mm2".to_string(), reticle - max_die_hi);
    if tdp > 0.0 && !phases.is_empty() {
        mc.insert("power_w".into(), phases.iter().map(|x| x.6).fold(f64::INFINITY, f64::min));
        mp.insert("power_w".into(), phases.iter().map(|x| x.7).fold(f64::INFINITY, f64::min));
    }
    mc.insert("power_density_w_mm2".into(), rep.q_avg_max_w_mm2 - q);
    mp.insert("power_density_w_mm2".into(), rep.q_avg_max_w_mm2 - q_hi);
    if tj.is_finite() {
        mc.insert("tj_c".into(), rep.tj_max_c - tj);
    }
    // Tier A junction estimate above t_j max: Tier B's thermal controller would throttle (04 §8); reported.
    let thermal = (!phases.is_empty() && tj > rep.tj_max_c).then(|| {
        Diagnostic::warning("W-PHYS-THERMAL", format!("Tier A junction estimate {tj:.0} C exceeds t_j max {:.0} C", rep.tj_max_c)).hint("lower the cap or improve cooling")
    });
    // `interval: none` reports points, as for time (the parameter bands are the 04 §3 corners).
    let band = |c: f64, lo: f64, hi: f64| if bands { Interval { low: lo.min(c), central: c, high: hi.max(c) } } else { Interval::point(c) };
    let s = PhysicalSummary {
        die_mm2: rep.dies.iter().map(|d| (d.path.clone(), band(d.area_mm2, d.area_low_mm2, d.area_high_mm2))).collect(),
        package_mm2: band(rep.package_mm2, rep.package_low_mm2, rep.package_high_mm2),
        peak_power_w: band(peak.0, peak.1, peak.2),
        power_density_max_w_mm2: band(q, q, q_hi),
        hbm_shoreline_used_mm: rep.dies.iter().map(|d| d.hbm_shoreline_used_mm).sum(),
        hbm_shoreline_available_mm: rep.dies.iter().map(|d| d.hbm_shoreline_available_mm).sum(),
        tdp_w: tdp,
        node: rep.node.clone(),
        margins_central: mc,
        margins_pessimistic: mp,
    };
    (Some(s), findings, thermal)
}

/// `evaluate(design, workload, options)`: the design is a loaded document; `workload` a zoo member
/// (`llama3_8b:decode_b8`) or a suite (`standard`).
pub fn evaluate(design: Design, workload: &str, opts: &SimOptions) -> EvalResult {
    let prov = base_provenance(&design.hash, "", opts);
    let members = match if workload.contains(':') { kiln_wl::zoo::workload(workload).map(|m| vec![m]) } else { kiln_wl::zoo::suite(workload) } {
        Ok(m) => m,
        Err(e) => return fail(Status::Invalid, Stage::S0, prov, vec![e]),
    };
    let p = match Prepared::load(design, Profile::Reference) {
        Ok(p) => p,
        Err(errs) => return fail(Status::Invalid, Stage::S0, prov, errs),
    };
    evaluate_prepared(&p, &members, opts)
}

/// Central / low / high result of a phase by corner (convenience for reports).
pub fn corner(run: &PhaseRun, c: Corner) -> &SimResult {
    match c {
        Corner::Low => run.low.as_ref().unwrap_or(&run.central),
        Corner::High => run.high.as_ref().unwrap_or(&run.central),
        Corner::Central => &run.central,
    }
}
