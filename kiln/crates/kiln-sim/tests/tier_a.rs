//! Tier A unit and integration tests: closed forms, invariants on reference designs, whole-step
//! extrapolation, corners, and the `evaluate` entry point.

mod common;

use common::*;
use kiln_ir::bench::BenchOp;
use kiln_sim::{ParamSet, SimOptions, evaluate, explain_run, simulate_bench_op};
use kiln_trace::check::{check_result, check_sim};
use kiln_trace::result::Status;
use kiln_trace::sim::{BindingClass, CheckStatus, SimResult};
use kiln_trace::{Corner, IntervalMethod};

fn null() -> SimOptions {
    SimOptions { params: Some(ParamSet::null()), interval: IntervalMethod::None, shadow_prices: false, ..SimOptions::default() }
}

fn assert_clean(r: &SimResult) {
    let errs = check_sim(r);
    assert!(errs.is_empty(), "{}: {errs:?}", r.phase);
    let fails: Vec<_> = r.invariants.failures().collect();
    assert!(fails.is_empty(), "{}: {fails:?}", r.phase);
    assert!(r.invariants.checks.iter().filter(|c| c.status == CheckStatus::Pass).count() >= 9);
}

#[test]
fn gemv_on_v5e_is_bound_by_hbm_closed_form() {
    let p = reference("tpu_v5e.json5");
    let r = simulate_bench_op(&p, &BenchOp::gemm(1, 4096, 4096, true), &null()).unwrap();
    let from_hbm = (4096.0 * 4096.0 + 4096.0) * 2.0;
    let expect = from_hbm / 819.2e9;
    let t = r.central.makespan_s;
    assert!((t / expect - 1.0).abs() < 0.005, "{t} vs {expect}");
    assert_eq!(r.central.bottleneck.dominant().unwrap().0, BindingClass::Dram);
    assert_clean(&r.central);
}

#[test]
fn large_gemm_on_v5e_is_mxu_bound_closed_form() {
    let p = reference("tpu_v5e.json5");
    // At the nominal 1.5 GHz (the closed form's clock); under the assumed 200 W cap the GEMM throttles.
    let r = simulate_bench_op(&p, &BenchOp::gemm(4096, 4096, 4096, true), &SimOptions { clock: kiln_phys::ClockMode::Nominal, ..null() }).unwrap();
    let tiles_per_mxu = (4096.0 / 128.0) * (1024.0 / 128.0);
    let expect = tiles_per_mxu * 4096.0 / 1.5e9;
    let t = r.central.makespan_s;
    assert!(t >= expect * (1.0 - 1e-9) && t < expect * 1.02, "{t} vs {expect}");
    assert!(r.central.t_a0_s <= t);
    assert_clean(&r.central);
}

#[test]
fn overheads_enter_exactly_once_per_op() {
    let p = reference("tpu_v5e.json5");
    let op = BenchOp::gemm(1, 4096, 4096, true);
    let base = simulate_bench_op(&p, &op, &null()).unwrap().central.makespan_s;
    let mut set = ParamSet::assumed(kiln_ir::hw::types::ExecModel::StaticDataflow, "hbm2e");
    set.params.retain(|x| x.name == "t_sync");
    let o = SimOptions { params: Some(set), exec_model: Some(kiln_ir::hw::types::ExecModel::StaticDataflow), ..null() };
    let with = simulate_bench_op(&p, &op, &o).unwrap().central.makespan_s;
    assert!(((with - base) - 1e-6).abs() < 1e-12, "{with} - {base}");
}

#[test]
fn reference_designs_pass_invariants_at_every_corner() {
    for d in ["tpu_v5e.json5", "tpu_v6e.json5"] {
        let p = reference(d);
        for m in kiln_wl::zoo::suite("standard").unwrap() {
            let (run, _) = kiln_sim::simulate_member(&p, &m, &quick()).unwrap();
            for r in [Some(&run.central), run.low.as_ref(), run.high.as_ref()].into_iter().flatten() {
                assert_clean(r);
                assert!(r.t_a0_s <= r.t_a2_s * (1.0 + 1e-9) && r.t_a2_s <= r.makespan_s * (1.0 + 1e-9));
            }
            assert!(run.time.low <= run.time.central && run.time.central <= run.time.high);
            assert!(run.low.as_ref().unwrap().makespan_s >= run.central.makespan_s);
        }
    }
}

#[test]
fn a100_decode_passes_invariants() {
    let p = reference("a100_sxm4_40gb.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b1").unwrap();
    let (run, _) = kiln_sim::simulate_member(&p, &m, &SimOptions { interval: IntervalMethod::None, shadow_prices: false, ..SimOptions::default() }).unwrap();
    assert_clean(&run.central);
    assert!(run.central.makespan_s > 5e-3 && run.central.makespan_s < 50e-3);
}

#[test]
fn whole_step_extrapolation_matches_a_wider_window() {
    let p = reference("tpu_v5e.json5");
    let (model, sc) = tiny(true, 2);
    let o = SimOptions { interval: IntervalMethod::None, shadow_prices: false, ..SimOptions::default() };
    let t3 = run(&p, &program(&model, &sc, 3), &o).central.makespan_s;
    let t4 = run(&p, &program(&model, &sc, 4), &o).central.makespan_s;
    let t1 = run(&p, &program(&model, &sc, 1), &o).central.makespan_s;
    assert!((t3 / t4 - 1.0).abs() < 1e-9, "{t3} vs {t4}");
    assert!((t1 / t4 - 1.0).abs() < 1e-9, "{t1} vs {t4}");
    let prog = program(&model, &sc, 9);
    assert_eq!(prog.window, Some((4, 4)));
}

#[test]
fn evaluate_returns_a_consistent_result() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/tpu_v6e.json5");
    let d = kiln_ir::hw::load_file(&path).unwrap();
    let r = evaluate(d, "standard", &quick());
    assert_eq!(r.status, Status::Ok, "{:?} {:?}", r.errors, r.violations);
    assert!(check_result(&r).is_empty(), "{:?}", check_result(&r));
    assert_eq!(r.phases.len(), 4);
    assert!(r.score > 0.0);
    let si = r.score_interval.unwrap();
    assert!(si.low <= si.central && si.central <= si.high);
    assert_eq!(r.sim.len(), 12);
    let c = r.sim_for("decode_b8", Corner::Central).unwrap();
    let text = explain_run(c, None, 4);
    assert!(text.contains("decode_b8") && text.contains("off-chip memory bandwidth"), "{text}");
    let bad = evaluate(kiln_ir::hw::load_file(&path).unwrap(), "llama3_8b:nope", &quick());
    assert_eq!(bad.status, Status::Invalid);
}

#[test]
fn v5e_decode_b32_does_not_fit_and_is_infeasible_not_extrapolated() {
    let p = reference("tpu_v5e.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b32").unwrap();
    let strict = SimOptions { layer_scope_fallback: false, ..quick() };
    let errs = kiln_sim::simulate_member(&p, &m, &strict).unwrap_err();
    assert_eq!(errs[0].code, "E-MAP-CAP-001");
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/tpu_v5e.json5");
    let r = evaluate(kiln_ir::hw::load_file(&path).unwrap(), "llama3_8b:decode_b32", &strict);
    assert_eq!(r.status, Status::Infeasible);
    assert_eq!(r.score, 0.0);
    let diag = quick();
    let (run, _) = kiln_sim::simulate_member(&p, &m, &diag).unwrap();
    assert_eq!(run.central.scope, kiln_trace::sim::Scope::Layer);
    assert!(run.central.invariants.checks.iter().any(|c| c.id == kiln_trace::sim::InvariantId::I7 && c.status == CheckStatus::Skipped));
    let r = kiln_sim::evaluate_prepared(&p, std::slice::from_ref(&m), &diag);
    assert_eq!(r.status, Status::Infeasible, "a layer-scope diagnostic never scores");
    assert!(r.score == 0.0 && r.phases.iter().all(|ph| ph.tokens_per_s.central == 0.0));
    let a = reference("tpu_v6e.json5");
    let (r6, _) = kiln_sim::simulate_member(&a, &m, &quick()).unwrap();
    assert_eq!(r6.central.scope, kiln_trace::sim::Scope::Step);
}

/// 03 §4.2: a task's duration is the busiest resource's total over all its demands (direct ones and routed
/// profiles alike), so a successor on another resource cannot start before the task has occupied each one.
#[test]
fn a_task_occupies_each_resource_for_its_summed_demands() {
    use std::sync::Arc;

    use kiln_map::hwview::{Profile, ResClass};
    use kiln_map::lower::{Amount, TGroup, Task, TaskGraph, TaskKind};
    use kiln_map::mapping::LaunchKind;
    use kiln_sim::SimParams;
    use kiln_sim::engine::Engine;

    let p = reference("tpu_v5e.json5");
    let v = &p.view;
    let clocks = v.phys.clock_plan(&kiln_phys::ClockMode::Nominal);
    let r = v.resources.iter().position(|x| x.class == ResClass::Dram).unwrap();
    let s = v.resources.iter().position(|x| x.class == ResClass::Compute).unwrap();
    let ns_r = v.resources[r].capacity * 1e-9;
    let ns_s = clocks.hz(v.resources[s].clock).unwrap() * 1e-9;
    let makespan = |a: Vec<Amount>| {
        let task = |op: u32, kind, dem: (u32, u32), pred: (u32, u32)| Task { kind, op, group: 0, lat_s: 0.0, lat_clk: (0, 0), dem, pred, bytes: 0.0 };
        let na = a.len() as u32;
        let mut demands = a;
        demands.push(Amount::Res(s as u32, 10.0 * ns_s));
        let g = TaskGraph {
            tasks: vec![task(0, TaskKind::Transfer, (0, na), (0, 0)), task(1, TaskKind::Compute, (na, na + 1), (0, 1))],
            demands,
            preds: vec![0],
            groups: vec![TGroup { label: "g".into(), ops: vec![], tasks: (0, 2), barrier_after: true, launch: LaunchKind::StaticProgram, iteration: None, fused: false }],
            profiles: vec![Arc::new(Profile { src: 0, dst: 0, entries: vec![(r as u32, 1.0)], writes: vec![], latency_s: 0.0, lat_clk: vec![], hops: 1 })],
            op_node: vec![0, 1],
            ..TaskGraph::default()
        };
        Engine { view: v, g: &g, params: &SimParams::null(), clocks: &clocks }.run(None).segs.iter().map(|x| x.time_est).sum::<f64>()
    };
    for a in [vec![Amount::Res(r as u32, ns_r); 2], vec![Amount::Prof(0, ns_r), Amount::Res(r as u32, ns_r)], vec![Amount::Prof(0, ns_r); 2]] {
        let t = makespan(a.clone());
        assert!((t / 12e-9 - 1.0).abs() < 1e-9, "{a:?}: {t} s, want 12 ns");
    }
}

/// 03 §4.5: a throttled domain stretches the cycle-counted latencies lowering expressed at its nominal clock
/// (pipeline fill, on-die hops, on-chip access); clock-independent delays (DRAM, PHYs) keep their seconds.
#[test]
fn throttling_stretches_clocked_latency() {
    use kiln_map::hwview::ResClass;
    use kiln_map::lower::{Amount, TGroup, Task, TaskGraph, TaskKind};
    use kiln_map::mapping::LaunchKind;
    use kiln_sim::SimParams;
    use kiln_sim::engine::Engine;

    let p = reference("tpu_v5e.json5");
    let v = &p.view;
    let s = v.resources.iter().position(|x| x.class == ResClass::Compute).unwrap();
    let c = v.resources[s].clock.unwrap();
    let nom = v.phys.nominal_hz(c);
    let at = |k: f64, async_s: f64| {
        let g = TaskGraph {
            tasks: vec![Task { kind: TaskKind::Compute, op: 0, group: 0, lat_s: 100.0 / nom + async_s, lat_clk: (0, 1), dem: (0, 1), pred: (0, 0), bytes: 0.0 }],
            demands: vec![Amount::Res(s as u32, 1.0)],
            lat_clk: vec![(c, 100.0 / nom)],
            groups: vec![TGroup { label: "g".into(), ops: vec![], tasks: (0, 1), barrier_after: true, launch: LaunchKind::StaticProgram, iteration: None, fused: false }],
            op_node: vec![0],
            ..TaskGraph::default()
        };
        let mut hz = v.phys.clock_plan(&kiln_phys::ClockMode::Nominal).hz;
        hz[c] *= k;
        let plan = v.phys.clock_plan(&kiln_phys::ClockMode::Fixed(hz));
        Engine { view: v, g: &g, params: &SimParams::null(), clocks: &plan }.run(None).segs.iter().map(|x| x.time_est).sum::<f64>()
    };
    assert!((at(1.0, 0.0) * nom / 101.0 - 1.0).abs() < 1e-9);
    assert!((at(0.5, 0.0) * nom / 202.0 - 1.0).abs() < 1e-9, "{} cycles", at(0.5, 0.0) * nom);
    assert!((at(0.5, 1e-6) - 202.0 / nom - 1e-6).abs() < 1e-15);

    // Lowering keeps the clocked part of every latency: an operand folded in from HBM keeps its 400 ns stack
    // latency clock independent, while its on-chip hops and the unit's fill are cycles of the core clock.
    let d = mesh(Params { gr: 2, gc: 2, rows: 32, cols: 32, lanes: 32, sram_kib: 512, pin_gbps: 6 }, A, Extra::None);
    let (model, sc) = tiny(true, 4);
    let g = run(&d, &program(&model, &sc, 3), &quick()).graph;
    let clocked = |t: &Task| g.lat_clk_of(t).iter().map(|x| x.1).sum::<f64>();
    assert!(g.tasks.iter().all(|t| clocked(t) <= t.lat_s * (1.0 + 1e-12)));
    assert!(g.tasks.iter().any(|t| t.kind == TaskKind::Compute && clocked(t) > 0.0 && t.lat_s - clocked(t) >= 400e-9 * (1.0 - 1e-9)));
}

/// 03 §10: an op's reported envelope covers its own work: no op floor (compute, off-chip traffic) exceeds the op's
/// time, even when the op's tasks overlap other ops' traffic on the same resources.
#[test]
fn op_envelopes_cover_their_floors() {
    let p = reference("a100_sxm4_40gb.json5");
    let opts = kiln_sim::SimOptions { interval: kiln_trace::IntervalMethod::None, shadow_prices: false, layer_scope_fallback: true, trace: kiln_trace::TraceLevel::Ops, ..Default::default() };
    for ph in ["decode_b1", "decode_b32"] {
        let m = kiln_wl::zoo::workload(&format!("llama3_8b:{ph}")).unwrap();
        let (run, _) = kiln_sim::simulate_member(&p, &m, &opts).unwrap();
        let r = &run.central;
        assert!(!r.ops.is_empty());
        for o in &r.ops {
            for f in &o.floors {
                assert!(f.seconds <= o.time_s() * (1.0 + 1e-9), "{ph} {}: {:?} floor {} s above op time {} s", o.op, f.kind, f.seconds, o.time_s());
            }
        }
    }
}
