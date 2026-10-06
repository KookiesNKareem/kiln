//! Trust suite (06 §12): T1 backtest designs and published-ratio corpus, T2 metamorphic relations.

pub mod predicate;
pub mod transform;

use std::path::PathBuf;

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::hw::{Design, HwModel, Profile, Report};
use kiln_ir::precision::Precision;
use kiln_ir::wl::{CacheState, ElemType, EvalMode, Model, PhaseKind, Scenario, SeqBatch, TensorClass, WorkloadDoc};
use kiln_trace::Corner;
use kiln_trace::result::{PhaseResult, Status};
use kiln_wl::StepStats;
use kiln_wl::zoo::SuiteMember;
use serde_json::Value;

pub use predicate::{Outcome, Verdict};

pub fn kiln_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn design_path(name: &str) -> PathBuf {
    kiln_root().join("designs/reference").join(format!("{name}.json5"))
}

pub fn corpus_path(file: &str) -> PathBuf {
    kiln_root().join("corpus/trust").join(file)
}

/// T1 chip-history designs (06 §12.1): built from published specs only, no platform key.
pub const BACKTEST: [&str; 4] = ["v100_sxm2_32gb", "h100_sxm5_80gb", "h100_pcie_80gb", "tpu_v4"];

pub fn load(name: &str) -> Result<Design, Vec<Diagnostic>> {
    kiln_ir::hw::load_file(design_path(name))
}

/// Reloads an edited canonical value and runs expansion + validation under `profile`.
pub fn check_value(v: Value, profile: Profile) -> Report {
    match Design::from_value(v) {
        Ok(d) => kiln_ir::hw::check(d, profile, &Default::default()),
        Err(diagnostics) => Report { design: None, model: None, diagnostics },
    }
}

pub fn read_json5(file: &str) -> Value {
    let text = std::fs::read_to_string(corpus_path(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
    json5::from_str(&text).unwrap_or_else(|e| panic!("{file}: {e}"))
}

/// M8's workload transform: the same preset and scenario with weights stored as `p`.
pub fn workload_with_weights(name: &str, p: Option<Precision>) -> Result<(Model, Scenario), Diagnostic> {
    let (preset, scenario) = name.split_once(':').ok_or_else(|| Diagnostic::error("E-WL-SCN-001", "want <preset>:<scenario>"))?;
    let mut cfg = kiln_wl::zoo::preset(preset)?;
    if let Some(p) = p {
        cfg.dtypes.weights = ElemType::from(p);
    }
    Ok((kiln_wl::zoo::build_model(&cfg)?, kiln_wl::zoo::scenario(scenario)?))
}

pub fn step_stats(model: &Model, scenario: &Scenario) -> Result<StepStats, Vec<Diagnostic>> {
    kiln_wl::evaluate_snapshot(model, scenario).map(|(_, _, s)| s)
}

/// Structural peak of every enabled unit of `kind` for element mode `mode` (vector: one op per lane-op; matrix: 2 per MAC).
pub fn unit_peak(m: &HwModel, kind: &str, mode: &str, class: kiln_ir::op_class::OpClass) -> f64 {
    let units: Vec<usize> = (0..m.units.len()).filter(|&u| m.units[u].spec.kind.name() == kind).collect();
    m.peak_ops(&units, mode, class)
}

/// Seam for the engine (`kiln_sim`, 03 §4.9 whole-step mode).
pub type EvalFn = fn(&Design, &str) -> Result<Outcome, String>;

pub fn engine() -> Option<EvalFn> {
    Some(evaluate)
}

pub const NO_ENGINE: &str = "needs kiln_sim::evaluate (Tier A whole-step engine, M1)";

/// A trust workload: `<preset>:<scenario>` or `gemm_<n>` (isolated `n x n x n` GEMM), with optional suffixes
/// `+dtype=<p>` (every bf16 tensor stored as `p`) and `+weights=<p>` (weights stored as `p`).
pub fn member(name: &str) -> Result<SuiteMember, Diagnostic> {
    let mut parts = name.split('+');
    let base = parts.next().unwrap_or_default();
    let (mut dtype, mut weights) = (None, None);
    for suffix in parts {
        let (k, v) = suffix.split_once('=').unwrap_or((suffix, ""));
        let p: Precision = serde_json::from_value(Value::from(v))
            .map_err(|e| Diagnostic::error("E-WL-SCN-001", format!("{name}: precision {v:?}: {e}")))?;
        match k {
            "dtype" => dtype = Some(p),
            "weights" => weights = Some(p),
            _ => return Err(Diagnostic::error("E-WL-SCN-001", format!("{name}: unknown suffix {k:?}"))),
        }
    }
    let (id, mut model, sid, sc) = if let Some(n) = base.strip_prefix("gemm_").and_then(|n| n.parse::<u64>().ok()) {
        let mut s = Scenario::snapshot(PhaseKind::Custom, SeqBatch::uniform(n, 1, 1));
        s.eval_mode = EvalMode::Isolated { cache: CacheState::Cold, launch: false };
        (base, kiln_wl::zoo::gemm_model(n, n, n), "step", s)
    } else {
        let (m, s) = workload_with_weights(base, None)?;
        let (preset, scenario) = base.split_once(':').expect("checked by workload_with_weights");
        (preset, m, scenario, s)
    };
    let mut retype = |from: ElemType, to: Precision, only: Option<TensorClass>| {
        let tensors = model.tensors.values_mut().chain(model.graphs.values_mut().flat_map(|g| g.tensors.values_mut()));
        for t in tensors.filter(|t| t.dtype == from && only.is_none_or(|c| t.class == c)) {
            t.dtype = ElemType::from(to);
        }
    };
    if let Some(p) = dtype {
        retype(ElemType::BF16, p, None);
    }
    if let Some(p) = weights {
        retype(dtype.map_or(ElemType::BF16, ElemType::from), p, Some(TensorClass::Weight));
    }
    let mut doc = WorkloadDoc::new(Id::new(id)?, model);
    let sid = Id::new(sid)?;
    doc.scenarios.insert(sid.clone(), sc);
    Ok(SuiteMember { name: name.into(), doc, scenario: sid })
}

/// Validates `design` under `full` and runs Tier A on one workload: the phase result, the central-corner
/// static energy and the total envelope die area (kiln-phys analytic roll-up, mm^2).
pub fn run_phase(design: &Design, workload: &str) -> Result<(PhaseResult, f64, f64), String> {
    let fmt = |ds: &[Diagnostic]| ds.iter().map(|d| format!("{}: {}", d.code, d.message)).collect::<Vec<_>>().join("; ");
    let m = member(workload).map_err(|d| fmt(&[d]))?;
    let p = kiln_sim::Prepared::load(design.clone(), Profile::Full).map_err(|ds| fmt(&ds))?;
    let mut r = kiln_sim::evaluate_prepared(&p, &[m], &kiln_sim::SimOptions::default());
    // Envelope findings (a transform can outgrow a fixed outline or the reticle) do not stop the timing relations.
    let envelope_only = r.status == Status::Envelope && !r.phases.is_empty();
    if r.status != Status::Ok && !envelope_only {
        let ds: Vec<Diagnostic> = r.errors.iter().chain(&r.violations).map(|e| e.diag.clone()).collect();
        return Err(format!("{:?}: {}", r.status, fmt(&ds)));
    }
    let static_j = r.sim.iter().filter(|s| s.corner == Corner::Central).map(|s| s.energy.static_j).sum();
    if r.phases.is_empty() {
        return Err("no phase result".into());
    }
    let area = r.physical.as_ref().map_or(f64::NAN, |p| p.die_mm2.values().map(|d| d.central).sum());
    Ok((r.phases.swap_remove(0), static_j, area))
}

/// [`run_phase`] as an [`Outcome`]: bound times are the phase's `bound_breakdown` fractions x central time.
/// Area is the kiln-phys envelope roll-up (NaN without a physical result).
pub fn evaluate(design: &Design, workload: &str) -> Result<Outcome, String> {
    let (ph, static_j, area) = run_phase(design, workload)?;
    let t = ph.time_s.central;
    let part = |k: &str| ph.bound_breakdown.get(k).copied().unwrap_or(0.0) * t;
    Ok(Outcome {
        time_s: t,
        t_offchip: part("mem:offchip"),
        t_compute: part("compute"),
        energy_j: ph.energy_j.central,
        area_mm2: area,
        static_power_w: static_j / t,
        bound_breakdown: ph.bound_breakdown,
    })
}
