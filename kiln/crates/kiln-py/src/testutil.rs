use kiln_trace::result::EvalResult;
use serde_json::json;

fn iv(x: f64) -> serde_json::Value {
    json!({"low": 0.9 * x, "central": x, "high": 1.1 * x})
}

/// A minimal ok result: phases `(id, tokens/s)`, one die of `die_mm2`, TDP `tdp_w`.
pub fn result_with(phases: &[(&str, f64)], die_mm2: f64, tdp_w: f64) -> EvalResult {
    let phases: Vec<_> = phases
        .iter()
        .map(|(id, tps)| {
            json!({"phase": id, "scope": "step", "time_s": {"low": 1.0 / (1.1 * tps), "central": 1.0 / tps, "high": 1.0 / (0.9 * tps)},
                "tokens_per_s": iv(*tps), "energy_j": iv(1.0), "tokens_per_j": iv(tps / 300.0), "avg_power_w": iv(300.0),
                "clock_hz": iv(1.4e9), "roofline_frac": 0.5,
                "bound_breakdown": {"compute": 0.2, "mem:chip0.hbm": 0.8}, "trusted": false})
        })
        .collect();
    serde_json::from_value(json!({
        "schema": "kiln.result/1", "status": "ok", "score": 0.0, "stage_reached": "S3", "tier": "A",
        "phases": phases,
        "physical": {"die_mm2": {"chip0.die": iv(die_mm2)}, "package_mm2": iv(2000.0), "peak_power_w": iv(tdp_w),
            "power_density_max_w_mm2": iv(0.5), "hbm_shoreline_used_mm": 40.0, "hbm_shoreline_available_mm": 60.0,
            "tdp_w": tdp_w, "node": "tsmc_n7", "margins_central": {"die_mm2": 26.0, "tdp_w": 0.0}},
        "provenance": {"kiln_version": "0.0.1", "git_hash": "test", "design_hash": "hw1-test", "workload_hash": "wl1-test",
            "calibration_hash": "cal1-test", "tier": "A", "seeds": [0]}
    }))
    .unwrap()
}
