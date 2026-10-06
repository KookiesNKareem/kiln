//! Unit-level costs of non-MAC work: ganged vector units share a slice's work.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::geom::Slice;
use kiln_map::heuristic::placement;
use kiln_map::lower::Lowerer;
use kiln_map::{HwView, NestCost, NestQuery, Pool, Program, RooflineCost, UnitCostModel};

fn view(name: &str) -> HwView {
    let p = format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"));
    HwView::new(Arc::new(check_file(p, Profile::Reference).model.expect("expands"))).expect("view")
}

fn program(workload: &str) -> Program {
    let m = kiln_wl::zoo::workload(workload).expect("workload");
    let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
    Program::whole_step(m.model(), &lg, 1).expect("program")
}

fn whole(op: &kiln_map::program::POp) -> Slice {
    Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0)
}

fn cost(v: &HwView, prog: &Program, oi: usize, u: usize, gang: u32) -> NestCost {
    let op = &prog.ops[oi];
    let s = whole(op);
    let ui = &v.units[u];
    let q = NestQuery {
        prog,
        op,
        slice: &s,
        points: kiln_map::geom::slice_points(op, &s),
        hw: &v.hw,
        unit: ui.unit,
        level_caps: &[],
        level_mems: &[],
        level_bw: &[],
        residency: &[],
        gang,
        quick: true,
        partial: false,
    };
    RooflineCost.cost(&q).expect("cost")
}

#[test]
fn a_ganged_vector_slice_runs_on_every_member() {
    let v = view("a100_sxm4_40gb.json5");
    let u = v.pool(Pool::Vector)[0];
    let gang = v.units[u].members.len() as u32;
    assert_eq!(gang, 4, "the four SMSP ALUs of an SM are one gang");
    let prog = program("llama3_8b:prefill_b1");
    let oi = prog.ops.iter().position(|o| o.class() == KernelClass::Map && o.points() > 1 << 20).expect("a large map kernel");
    let (one, all) = (cost(&v, &prog, oi, u, 1), cost(&v, &prog, oi, u, gang));
    assert!((one.cycles / all.cycles - 4.0).abs() < 1e-3, "{} vs {}", one.cycles, all.cycles);
}

#[test]
fn gangs_need_a_private_level_above_the_feeds() {
    // An SM's four SMSPs share its L1: one gang. ember's tiles meet only at the interleaved level above them
    // and off chip: each is its own unit, so ops split over all 254 of them.
    let a100 = view("a100_sxm4_40gb.json5");
    assert!(a100.pool(Pool::Mac).iter().all(|&u| a100.units[u].members.len() == 4));
    let ember = view("ember.json5");
    for pool in [Pool::Mac, Pool::Vector] {
        let units = ember.pool(pool);
        assert_eq!(units.len(), 254, "{pool:?}");
        assert!(units.iter().all(|&u| ember.units[u].members.len() == 1));
    }
}

#[test]
fn the_level_above_a_scratchpad_is_the_largest_one_up() {
    // ember's tile SRAMs reach two level-2 memories over the NoC: the CIM macros' 512 KiB activation buffers and
    // the 64 MiB stacked L3s. The L3s back the tiles (and hold activations); the CIM buffers do not.
    let v = view("ember.json5");
    let u = &v.units[v.pool(Pool::Mac)[0]];
    let names: Vec<&str> = u.chain.iter().map(|&g| v.groups[g].name.as_str()).collect();
    assert_eq!(names, ["board.pkg.cc0_0.t0_0.sram", "board.pkg.sram.l3", "board.pkg.hbm"]);
    assert_eq!(v.shared_onchip().map(|g| v.groups[g].capacity), Some(4 * (64 << 20)));
}

#[test]
fn fewer_slices_than_units_spread_over_the_set() {
    let prog = program("llama3_8b:decode_b1");
    let op = prog.ops.iter().find(|o| o.class() == KernelClass::Contraction).expect("a contraction");
    let d = op.kernel.dims.iter().find(|d| d.extent >= 4).expect("a dim to split");
    let split = |n: u64| vec![kiln_map::mapping::SplitAxis { dim: d.name.clone(), parts: kiln_map::geom::even_parts(d.extent, n) }];
    assert_eq!(placement(op, 0, split(4), 16).slice_to_unit, [0, 4, 8, 12]);
    assert_eq!(placement(op, 0, split(4), 3).slice_to_unit, [0, 1, 2, 0]);
}
