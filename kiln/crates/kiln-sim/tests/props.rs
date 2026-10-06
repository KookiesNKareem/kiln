//! L2 properties (06 §2.2) for Tier A: P1 floors, P2 energy, P3 monotonicity at fixed mapping, P4 renaming,
//! P6 dead hardware, P10 thread-count invariance, P11 round trip, P12 finite results.

mod common;

use common::*;
use kiln_ir::bench::BenchOp;
use kiln_map::Program;
use kiln_phys::ClockMode;
use kiln_sim::engine::Engine;
use kiln_sim::{SimOptions, simulate_bench_op};
use kiln_trace::check::check_sim;
use kiln_trace::sim::SimResult;
use kiln_trace::{Corner, IntervalMethod};
use proptest::prelude::*;

fn params() -> impl Strategy<Value = Params> {
    (1u32..4, 1u32..4, prop::sample::select(vec![8u32, 16, 32, 64]), prop::sample::select(vec![8u32, 16, 32, 64]), 4u32..65, 64u32..2048, 2u32..9)
        .prop_map(|(gr, gc, rows, cols, lanes, sram_kib, pin_gbps)| Params { gr, gc, rows, cols, lanes, sram_kib, pin_gbps })
}

fn workload() -> impl Strategy<Value = (bool, u64)> {
    (any::<bool>(), prop::sample::select(vec![1u64, 2, 4]))
}

fn canon(r: &SimResult) -> String {
    let mut v = serde_json::to_value(r).unwrap();
    v.as_object_mut().unwrap().remove("provenance");
    kiln_ir::common::canonical_json(&v)
}

fn numbers(r: &SimResult) -> (u64, u64, u64, u64) {
    (r.makespan_s.to_bits(), r.t_a2_s.to_bits(), r.t_a0_s.to_bits(), r.energy.total_j.to_bits())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]

    #[test]
    fn floors_energy_and_finiteness(p in params(), (decode, batch) in workload()) {
        let d = mesh(p, A, Extra::None);
        let (model, sc) = tiny(decode, batch);
        let r = run(&d, &program(&model, &sc, 3), &quick());
        for s in [&r.central, r.low.as_ref().unwrap(), r.high.as_ref().unwrap()] {
            prop_assert!(check_sim(s).is_empty(), "{:?}", check_sim(s));
            prop_assert!(s.invariants.passed(), "{:?}", s.invariants.failures().collect::<Vec<_>>());
            prop_assert!(s.t_a0_s <= s.t_a2_s * (1.0 + 1e-9) && s.t_a2_s <= s.makespan_s * (1.0 + 1e-9));
            let e = &s.energy;
            prop_assert!((e.total_j - e.component_sum()).abs() <= 1e-9 * e.total_j);
            prop_assert!(e.total_j > 0.0 && s.makespan_s.is_finite());
            prop_assert!(s.resources.iter().all(|x| x.busy_s.is_finite() && x.busy_s >= 0.0 && x.utilization <= 1.0 + 1e-9));
        }
        prop_assert!(r.low.as_ref().unwrap().makespan_s >= r.central.makespan_s);
        prop_assert!(r.central.makespan_s >= r.high.as_ref().unwrap().makespan_s);
    }

    #[test]
    fn renaming_ids_leaves_numbers_bit_identical(p in params(), (decode, batch) in workload()) {
        let (model, sc) = tiny(decode, batch);
        let prog = program(&model, &sc, 3);
        let a = run(&mesh(p, A, Extra::None), &prog, &quick());
        let b = run(&mesh(p, B, Extra::None), &prog, &quick());
        prop_assert_eq!(numbers(&a.central), numbers(&b.central));
        prop_assert_eq!(numbers(a.low.as_ref().unwrap()), numbers(b.low.as_ref().unwrap()));
    }

    #[test]
    fn unreachable_or_incapable_units_change_nothing(p in params(), (decode, batch) in workload()) {
        let (model, sc) = tiny(decode, batch);
        let prog = program(&model, &sc, 3);
        let base = run(&mesh(p, A, Extra::None), &prog, &quick());
        // P6 with the M3 floorplan (04 §6.1, §17): links are measured on the live geometry, so a block nothing reaches
        // leaves every time bit-identical; a reachable unit no op can use (int8 under bf16) grows its cluster, and
        // intra-cluster wires scale with the cluster's live area, so it may only slow things down.
        let dead = run(&mesh(p, A, Extra::DeadMemory), &prog, &quick());
        prop_assert_eq!(base.central.makespan_s.to_bits(), dead.central.makespan_s.to_bits());
        prop_assert_eq!(base.central.t_a2_s.to_bits(), dead.central.t_a2_s.to_bits());
        let i8u = run(&mesh(p, A, Extra::Int8Unit), &prog, &quick());
        for (x, y) in [(base.central.makespan_s, i8u.central.makespan_s), (base.central.t_a2_s, i8u.central.t_a2_s)] {
            prop_assert!(y >= x * (1.0 - 1e-9), "int8 unit made a bf16 step faster: {} -> {}", x, y);
        }
    }

    #[test]
    fn more_bandwidth_or_clock_never_slows_a_fixed_mapping(p in params(), k in 1.0f64..4.0, (decode, batch) in workload()) {
        let d = mesh(p, A, Extra::None);
        let (model, sc) = tiny(decode, batch);
        let prog = program(&model, &sc, 3);
        let r = run(&d, &prog, &quick());
        let params = quick().param_set(&d.view).at(Corner::Central);
        let clocks = d.view.phys.clock_plan(&ClockMode::Nominal);
        let mid = prog.window.map(|w| w.0 / 2);
        let t = |p: &kiln_sim::SimParams, c: &kiln_phys::ClockPlan| -> f64 {
            Engine { view: &d.view, g: &r.graph, params: p, clocks: c }.run(mid).segs.iter().map(|s| s.time_est).sum()
        };
        let base = t(&params, &clocks);
        let mut wide = params.clone();
        wide.cap_scale = (0..d.view.resources.len() as u32).map(|i| (i, k)).collect();
        prop_assert!(t(&wide, &clocks) <= base * (1.0 + 1e-12));
        let fast = d.view.phys.clock_plan(&ClockMode::Fixed(clocks.hz.iter().map(|h| h * k).collect()));
        prop_assert!(t(&params, &fast) <= base * (1.0 + 1e-12));
    }
}

#[test]
fn results_are_identical_across_thread_counts_and_runs() {
    let d = reference("tpu_v5e.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b8").unwrap();
    let one = SimOptions { threads: 1, ..quick() };
    let many = SimOptions { threads: 3, ..quick() };
    let (a, _) = kiln_sim::simulate_member(&d, &m, &one).unwrap();
    let (b, _) = kiln_sim::simulate_member(&d, &m, &many).unwrap();
    let (c, _) = kiln_sim::simulate_member(&d, &m, &one).unwrap();
    for (x, y) in [(&a, &b), (&a, &c)] {
        assert_eq!(canon(&x.central), canon(&y.central));
        assert_eq!(canon(x.low.as_ref().unwrap()), canon(y.low.as_ref().unwrap()));
        assert_eq!(x.mapping, y.mapping);
    }
}

#[test]
fn sim_result_round_trips_through_json() {
    let d = reference("tpu_v6e.json5");
    let r = simulate_bench_op(&d, &BenchOp::gemm(32, 4096, 4096, true), &SimOptions { interval: IntervalMethod::Corners, ..SimOptions::default() }).unwrap();
    let json = serde_json::to_string(&r.central).unwrap();
    let back: SimResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r.central);
    assert!(!r.central.ops.is_empty() && !r.central.bottleneck.summary.is_empty());
    let tr = &r.central.bottleneck.top_resources[0];
    assert!(tr.shadow_price > 0.5, "the binding resource has a high shadow price: {tr:?}");
}

#[test]
fn isolated_program_charges_every_operand_off_chip() {
    let d = reference("tpu_v6e.json5");
    let prog = Program::bench_op(&BenchOp::gemm(8, 4096, 4096, true)).unwrap();
    let r = run(&d, &prog, &SimOptions { interval: IntervalMethod::None, ..SimOptions::default() });
    let dram: f64 = r.central.resources.iter().filter(|x| x.kind == kiln_trace::sim::ResourceKind::DramChannel).map(|x| x.bytes).sum();
    let compulsory = (8.0 * 4096.0 + 4096.0 * 4096.0 + 8.0 * 4096.0) * 2.0;
    assert!(dram >= compulsory && dram < compulsory * 1.01, "{dram} vs {compulsory}");
}
