//! Regressions from the r6 sim review: the Tier A contention term never lets more critical-path work run faster.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_map::lower::{Amount, TGroup, Task, TaskGraph, TaskKind};
use kiln_map::mapping::LaunchKind;
use kiln_map::{HwView, Pool};
use kiln_sim::SimParams;
use kiln_sim::engine::Engine;

/// One segment: `a` (`d` cycles on unit R) then `b` (`l` cycles on unit R2) on the critical path, and `c` (`o`
/// cycles on R) off it.
fn graph(r: u32, r2: u32, d: f64, l: f64, o: f64) -> TaskGraph {
    let task = |dem: u32, pred: (u32, u32)| Task { kind: TaskKind::Compute, op: 0, group: 0, lat_s: 0.0, lat_clk: (0, 0), dem: (dem, dem + 1), pred, bytes: 0.0 };
    TaskGraph {
        tasks: vec![task(0, (0, 0)), task(1, (0, 1)), task(2, (1, 1))],
        demands: vec![Amount::Res(r, d), Amount::Res(r2, l), Amount::Res(r, o)],
        preds: vec![0],
        groups: vec![TGroup { label: "g".into(), ops: vec![0], tasks: (0, 3), barrier_after: true, launch: LaunchKind::StaticProgram, iteration: None, fused: false }],
        op_node: vec![0],
        ..Default::default()
    }
}

#[test]
fn contention_is_monotone_in_critical_path_work() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../designs/reference/tpu_v5e.json5");
    let view = HwView::new(Arc::new(check_file(p, Profile::Reference).model.expect("v5e"))).expect("view");
    let mac = view.pool(Pool::Mac);
    let (r, r2) = (view.units[mac[0]].compute, view.units[mac[1]].compute);
    let params = SimParams { contention_scale: 1.0, ..SimParams::null() };
    let clocks = view.phys.clock_plan(&kiln_phys::ClockMode::Nominal);
    let time = |l: f64| {
        let g = graph(r, r2, 1e6, l, 8e6);
        Engine { view: &view, g: &g, params: &params, clocks: &clocks }.run(None).segs.iter().map(|s| s.time_est).sum::<f64>()
    };
    let mut prev = 0.0;
    for k in 0..40 {
        let t = time(8e6 + f64::from(k) * 0.25e6);
        assert!(t >= prev, "critical path {k}: {t} s < {prev} s");
        prev = t;
    }
}

/// 03 §9.1: corners re-cost the mapping on the physical model at each corner, so physical coefficient ranges widen
/// the energy interval even with no engine parameters.
#[test]
fn corners_carry_physical_uncertainty() {
    let p = kiln_sim::Prepared::from_file(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../designs/reference/tpu_v5e.json5")), Profile::Reference).unwrap();
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b1").unwrap();
    let opts = kiln_sim::SimOptions {
        interval: kiln_trace::IntervalMethod::Corners,
        params: Some(kiln_sim::ParamSet::null()),
        clock: kiln_phys::ClockMode::Nominal,
        ..kiln_sim::SimOptions::default()
    };
    let (run, _) = kiln_sim::simulate_member(&p, &m, &opts).unwrap();
    assert!(run.energy.low < run.energy.central && run.energy.central < run.energy.high, "{:?}", run.energy);
}
