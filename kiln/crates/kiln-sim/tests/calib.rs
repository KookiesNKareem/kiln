//! Calibration-set application (03 §9): one place per parameter, provenance carries the set hash, telemetry
//! clock operating points, and re-costing at a fixed mapping equals the full engine path.

use std::collections::BTreeMap;
use std::sync::Arc;

use kiln_ir::bench::BenchOp;
use kiln_ir::hw::Profile;
use kiln_sim::calib::{CALIB_SCHEMA, CalParam, CalibSet, ParamStatus, RangeSpec, SetKind};
use kiln_sim::engine::dram_ramp;
use kiln_sim::params::PessDir;
use kiln_sim::{Prepared, SimOptions, simulate_bench_op};
use kiln_trace::IntervalMethod;

fn a100() -> Prepared {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/a100_sxm4_40gb.json5");
    Prepared::from_file(&p, Profile::Reference).expect("design loads")
}

fn p(name: &str, key: &[(&str, &str)], value: f64, range: (f64, f64), pess: PessDir) -> CalParam {
    CalParam {
        name: name.into(),
        key: key.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<BTreeMap<_, _>>(),
        value,
        unit: "1".into(),
        bounds: [0.0, 1.0],
        prior: value,
        ci95: None,
        range: RangeSpec { lower: range.0, upper: range.1, basis: "single_device".into() },
        pess_dir: pess,
        status: ParamStatus::Fit,
        frozen_reason: None,
        source: None,
        table: None,
        diagnostics: None,
    }
}

fn platform() -> CalibSet {
    let us = 1e-6;
    let mut clock = p("f_cap_op", &[("domain", "gpc_clk"), ("power_cap", "400W")], 1.0, (0.99, 1.01), PessDir::Lower);
    clock.status = ParamStatus::Telemetry;
    clock.table = Some(vec![[0.0, 1.41e9], [0.5, 1.41e9], [0.9, 1.29e9]]);
    let mut s = CalibSet {
        schema: CALIB_SCHEMA.into(),
        id: "platform:test".into(),
        version: 1,
        kind: SetKind::Platform,
        platform: Some("a100_40gb".into()),
        parameters: vec![
            p("eta_res", &[("dram_kind", "hbm2")], 0.9, (0.88, 0.92), PessDir::Lower),
            CalParam { bounds: [0.0, 2e-5], ..p("t_dram_ramp", &[("dram_kind", "hbm2")], 5.0 * us, (4.0 * us, 6.0 * us), PessDir::Upper) },
            CalParam { bounds: [1e-7, 2e-5], ..p("t_gap", &[("exec_model", "host_launched")], 0.4 * us, (0.3 * us, 0.5 * us), PessDir::Upper) },
            CalParam { bounds: [1e-7, 2e-5], ..p("t_min_kernel", &[("exec_model", "host_launched")], 1.4 * us, (1.0 * us, 2.0 * us), PessDir::Upper) },
            CalParam { bounds: [0.7, 1.0], ..p("unit_eff", &[("unit_template", "a100_40gb/tc:mma16x8x16")], 0.93, (0.9, 0.96), PessDir::Lower) },
            clock,
        ],
        fit: None,
        range_policy: None,
        acceptance: None,
        test_access_log: vec![],
        created: "2026-10-05".into(),
        hash: None,
    };
    s.hash = Some(s.compute_hash());
    s
}

#[test]
fn ramp_is_continuous_and_monotone() {
    let t = 4e-6;
    assert!((dram_ramp(t, t) - 2.0 * t).abs() < 1e-18);
    let mut last = 0.0;
    for i in 1..400 {
        let b = i as f64 * 5e-8;
        let y = dram_ramp(b, t);
        assert!(y > last && y >= b);
        last = y;
    }
    assert_eq!(dram_ramp(1e-3, t), 1e-3 + t);
}

#[test]
fn set_is_applied_and_hashed_into_provenance() {
    let pr = a100();
    let set = Arc::new(platform());
    let opts = SimOptions { interval: IntervalMethod::Corners, calibration: Some(set.clone()), shadow_prices: false, ..SimOptions::default() };
    let gemv = BenchOp::gemm(1, 4096, 4096, true);
    let run = simulate_bench_op(&pr, &gemv, &opts).unwrap();
    assert_eq!(run.central.provenance.calibration_hash, set.compute_hash());
    assert_eq!(run.central.provenance.calibration_id.as_deref(), Some("platform:test"));
    assert_eq!(run.central.provenance.flags.get("clock").map(String::as_str), Some("telemetry_operating_points"));
    assert!(run.params.extrapolated.iter().all(|e| e == "t_launch"), "{:?}", run.params.extrapolated);
    // Low corner = pessimistic end of every parameter: slower.
    assert!(run.time.high > run.time.central && run.time.central > run.time.low);
    let unc = simulate_bench_op(&pr, &gemv, &SimOptions { interval: IntervalMethod::None, shadow_prices: false, ..SimOptions::default() }).unwrap();
    assert_ne!(unc.central.provenance.calibration_hash, set.compute_hash());
    // Dual reporting: uncalibrated (null) value present.
    assert!(run.central.calibration.uncalibrated_makespan_s.unwrap() < run.central.makespan_s);
    // Re-costing the same mapping at the central parameters reproduces the engine path.
    let params = opts.param_set(&pr.view).at(kiln_trace::Corner::Central);
    let prog = kiln_map::Program::bench_op(&gemv).unwrap();
    let base = pr.view.phys.clock_plan(&kiln_phys::ClockMode::Nominal);
    let t = kiln_sim::recost(&pr.view, &prog, &run.graph, kiln_trace::sim::Scope::Op, &params, &base);
    assert!((t / run.central.makespan_s - 1.0).abs() < 1e-12, "{t} vs {}", run.central.makespan_s);
}

#[test]
fn telemetry_operating_points_throttle_dense_gemm_only() {
    let pr = a100();
    let opts = SimOptions { interval: IntervalMethod::None, calibration: Some(Arc::new(platform())), shadow_prices: false, ..SimOptions::default() };
    let hz = |op: &BenchOp| simulate_bench_op(&pr, op, &opts).unwrap().central.clocks.iter().find(|c| c.domain.as_str().contains("gpc")).map(|c| c.hz).unwrap();
    assert!(hz(&BenchOp::gemm(8192, 8192, 8192, false)) < 1.35e9);
    assert_eq!(hz(&BenchOp::gemm(1, 8192, 8192, false)), 1.41e9);
}

fn generic(fit_devices: &[&str]) -> CalibSet {
    use kiln_sim::calib::{FitInfo, PolicyRange, RangeEvidence, RangePolicy};
    let ev = |device: &str, lower: f64, upper: f64| RangeEvidence { device: device.into(), quantity: "q".into(), lower, upper, n: 1, source: "test".into() };
    let mut s = platform();
    s.id = "generic-test".into();
    s.kind = SetKind::Generic;
    s.platform = None;
    s.parameters.retain(|p| p.name != "f_cap_op" && p.name != "unit_eff");
    s.fit = Some(FitInfo {
        kiln_version: "0".into(),
        git_hash: "0".into(),
        method: "test".into(),
        loss: "test".into(),
        split_id: "s".into(),
        split_hash: "h".into(),
        fit_records: vec![],
        test_records: vec![],
        measurement_sessions: vec![],
        residuals_by_class: BTreeMap::new(),
        bootstrap_seed: 0,
        fit_devices: fit_devices.iter().map(|d| d.to_string()).collect(),
        notes: vec![],
    });
    s.range_policy = Some(RangePolicy {
        applies_to: "test".into(),
        ranges: vec![
            PolicyRange { name: "eta_res".into(), lower: 0.67, upper: 0.95, evidence: vec![ev("x", 0.88, 0.95), ev("y", 0.67, 0.85)] },
            PolicyRange { name: "t_gap".into(), lower: 0.1e-6, upper: 0.9e-6, evidence: vec![ev("x", 0.1e-6, 0.9e-6)] },
        ],
    });
    s.hash = Some(s.compute_hash());
    s
}

#[test]
fn range_policy_widens_ranges_only_outside_the_fit_devices() {
    let pr = a100();
    let range = |set: CalibSet, name: &str| {
        let opts = SimOptions { calibration: Some(Arc::new(set)), ..SimOptions::default() };
        let ps = opts.param_set(&pr.view);
        let p = ps.params.iter().find(|p| p.name == name).unwrap().clone();
        (p.range, p.basis)
    };
    let fit = generic(&["a100_40gb"]);
    assert!(fit.validate().is_empty(), "{:?}", fit.validate());
    let (r, basis) = range(fit, "eta_res");
    assert_eq!((r.lower, r.central, r.upper), (0.88, 0.9, 0.92));
    assert_eq!(basis, "single_device");
    let (r, basis) = range(generic(&["tpu_v5e"]), "eta_res");
    assert_eq!((r.lower, r.central, r.upper), (0.67, 0.9, 0.95));
    assert_eq!(basis, "generic_policy");
    // Listed terms widen to the policy span; unlisted terms keep their fitted range.
    let (r, _) = range(generic(&["tpu_v5e"]), "t_gap");
    assert_eq!((r.lower, r.upper), (0.1e-6, 0.9e-6));
    let (r, _) = range(generic(&["tpu_v5e"]), "t_min_kernel");
    assert_eq!((r.lower, r.upper), (1.0e-6, 2.0e-6));
    // Policy ranges are the span of their evidence, and only generic sets carry one.
    let mut tuned = generic(&["tpu_v5e"]);
    tuned.range_policy.as_mut().unwrap().ranges[0].lower = 0.5;
    tuned.hash = None;
    assert!(tuned.validate().iter().any(|d| d.message.contains("span of its evidence")));
    let mut plat = generic(&["tpu_v5e"]);
    plat.kind = SetKind::Platform;
    plat.hash = None;
    assert!(plat.validate().iter().any(|d| d.message.contains("range policy")));
}
