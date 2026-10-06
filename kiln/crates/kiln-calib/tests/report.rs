//! Acceptance verdicts (06 §3.3.6, §3.5): every prediction failure, residual slope and class bias is gated.

use std::collections::BTreeMap;

use kiln_calib::report::{DeviceReport, OpRow, Role, StepRow, slope, stats, verdicts};

fn op(i: usize, ratio: f64) -> OpRow {
    let bytes = 1e6 * (i + 1) as f64;
    OpRow { name: format!("op{i}"), phase: "decode_b1".into(), class: "gemv_memory".into(), count: 1, meas_s: 1e-3, cal_s: 1e-3 * ratio, uncal_s: 1e-3, bytes, flops: 2.0 * bytes }
}

fn step(meas: f64) -> StepRow {
    StepRow { phase: "decode_b1".into(), layers: 32, attn: "sdpa".into(), meas_s: meas, central_s: meas, low_time_s: 0.9 * meas, high_time_s: 1.1 * meas, uncal_s: meas, covered: true }
}

fn device(ops: Vec<OpRow>, steps: Vec<StepRow>, expected_steps: usize, errors: Vec<String>) -> DeviceReport {
    let ratios: Vec<f64> = ops.iter().map(|o| o.cal_s / o.meas_s).collect();
    let mut class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for o in &ops {
        class.entry(o.class.clone()).or_default().push(o.cal_s / o.meas_s);
    }
    let class_cal: BTreeMap<_, _> = class.into_iter().map(|(k, v)| (k, stats(&v))).collect();
    let coverage = steps.iter().filter(|s| s.covered).count() as f64 / expected_steps.max(1) as f64;
    DeviceReport {
        device: "dev".into(),
        role: Role::Fit,
        op_cal: stats(&ratios),
        op_uncal: stats(&ratios),
        class_uncal: class_cal.clone(),
        class_cal,
        phase_sum: BTreeMap::new(),
        slopes: vec![],
        noise: vec![],
        coverage,
        expected_steps,
        chip_state: vec![],
        errors,
        ops,
        steps,
    }
}

#[test]
fn failed_whole_step_predictions_fail_acceptance() {
    let ops = (0..8).map(|i| op(i, 1.0)).collect();
    let d = device(ops, vec![], 4, vec!["step decode_b1 x32: E-MAP-CAP-001 does not fit".into()]);
    let v = verdicts(&d);
    assert!(v.iter().any(|v| !v.pass), "{v:?}");
    assert!(v.iter().any(|v| v.metric.contains("E-MAP-CAP-001") && !v.pass));
    assert!(v.iter().any(|v| v.metric.contains("coverage") && !v.pass));
}

#[test]
fn class_bias_fails_acceptance() {
    let ops = (0..8).map(|i| op(i, 1.06)).collect();
    let d = device(ops, vec![step(1.0)], 1, vec![]);
    let v = verdicts(&d);
    assert!(v.iter().any(|v| v.metric.contains("gemv_memory") && !v.pass), "{v:?}");
}

#[test]
fn significant_residual_slope_fails_acceptance() {
    let ops = (0..8).map(|i| op(i, 1.0)).collect();
    let mut d = device(ops, vec![step(1.0)], 1, vec![]);
    assert!(verdicts(&d).iter().all(|v| v.pass));
    d.slopes = vec![slope("log bytes", &[0.0, 1.0, 2.0, 3.0], &[-0.2, -0.05, 0.05, 0.2])];
    assert!(d.slopes[0].significant);
    assert!(verdicts(&d).iter().any(|v| v.metric.contains("log bytes") && !v.pass));
}

#[test]
fn exact_nonzero_slope_is_significant() {
    let s = slope("x", &[0.0, 1.0, 2.0, 3.0], &[0.0, 0.125, 0.25, 0.375]);
    assert_eq!(s.slope, 0.125);
    assert_eq!(s.t, f64::INFINITY);
    assert!(s.significant);
    let flat = slope("x", &[0.0, 1.0, 2.0, 3.0], &[0.1; 4]);
    assert_eq!(flat.t, 0.0);
    assert!(!flat.significant);
}
