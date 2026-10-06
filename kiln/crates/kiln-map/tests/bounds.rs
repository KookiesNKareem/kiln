//! Candidate bounding (03 §3.3 scoring) is exact: floor costs never exceed real costs, and skipping candidates
//! whose floor estimate cannot win leaves the mapping and its task graph unchanged.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::heuristic::{MapOptions, heuristic_lowered, placement};
use kiln_map::lower::Lowerer;
use kiln_map::{HwView, KilnCost, NestQuery, Pool, Program, UnitCostModel};

fn view(name: &str) -> HwView {
    let p = format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"));
    HwView::new(Arc::new(check_file(p, Profile::Reference).model.expect("expands"))).expect("view")
}

fn program(workload: &str) -> Program {
    let m = kiln_wl::zoo::workload(workload).expect("workload");
    let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
    Program::whole_step(m.model(), &lg, 1).expect("program")
}

#[test]
fn bounded_candidate_scoring_maps_identically() {
    for (design, workload) in [
        ("a100_sxm4_40gb.json5", "llama3_8b:decode_b8"),
        ("a100_sxm4_40gb.json5", "llama3_8b:prefill_b1"),
        ("tpu_v5e.json5", "llama3_8b:decode_b32"),
    ] {
        let v = view(design);
        let prog = program(workload);
        let run = |bound: bool| {
            let opts = MapOptions { bound_candidates: bound, ..MapOptions::default() };
            heuristic_lowered(&prog, &v, &KilnCost::new(), &opts).expect("maps")
        };
        let ((m0, r0, g0), (m1, r1, g1)) = (run(false), run(true));
        assert_eq!(m0, m1, "{design} {workload}");
        assert_eq!((g0.tasks, g0.demands, g0.preds), (g1.tasks, g1.demands, g1.preds), "{design} {workload}");
        assert!(r1.candidates_scored < r0.candidates_scored, "{design} {workload}: bounding skipped nothing");
    }
}

#[test]
fn floor_costs_never_exceed_costs() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = program("llama3_8b:prefill_b1");
    let cost = KilnCost::new();
    let units = v.pool(Pool::Mac);
    let mut checked = 0;
    for (oi, op) in prog.ops.iter().enumerate().filter(|(_, o)| o.class() == kiln_ir::wl::KernelClass::Contraction) {
        for split in kiln_map::heuristic::candidates(op, &v, &units, 3) {
            let p = placement(op, 0, split, units.len());
            for s in Lowerer::slices(op, &p).iter().filter(|s| !s.is_empty()).take(2) {
                let u = &v.units[units[0]];
                let caps: Vec<u64> = u.chain[..u.private].iter().map(|&g| v.groups[g].capacity / 4).collect();
                let mems: Vec<usize> = u.chain[..caps.len()].iter().map(|&g| v.groups[g].mems[0]).collect();
                let q = NestQuery { prog: &prog, op, slice: s, points: kiln_map::geom::slice_points(op, s), hw: &v.hw, unit: u.unit, level_caps: &caps, level_mems: &mems, level_bw: &[], gang: u.members.len() as u32, quick: true };
                let (f, c) = (cost.cost_floor(&q).expect("bound").expect("floor"), cost.cost(&q).expect("cost"));
                assert!(f.cycles <= c.cycles && f.fill_cycles <= c.fill_cycles, "{oi}: {} > {}", f.cycles, c.cycles);
                assert!(f.feed_bytes.iter().zip(&c.feed_bytes).all(|(a, b)| a <= b), "{oi}: {:?} > {:?}", f.feed_bytes, c.feed_bytes);
                assert!(f.reread.iter().flatten().zip(c.reread.iter().flatten()).all(|(a, b)| a <= b), "{oi}");
                checked += 1;
            }
        }
    }
    assert!(checked > 10);
}

/// The lowerer shares one cost among slices of a masked or param segment that agree on extents, live points and
/// operand footprint sizes: the cost models depend on a slice's position only through those.
#[test]
fn slice_costs_depend_on_position_only_through_points_and_footprints() {
    let v = view("a100_sxm4_40gb.json5");
    let (kc, mut shared) = (KilnCost::new(), 0);
    for w in ["llama3_8b:prefill_b1", "llama3_8b:decode_b8"] {
        let prog = program(w);
        for op in prog.ops.iter().filter(|o| o.segs.iter().any(|s| !s.cons.is_empty() || !s.params.is_empty())) {
            let units = v.pool(if op.class() == KernelClass::Contraction { Pool::Mac } else { Pool::Vector });
            let u = &v.units[units[0]];
            let caps: Vec<u64> = u.chain[..u.private].iter().map(|&g| v.groups[g].capacity / 4).collect();
            let mems: Vec<usize> = u.chain[..caps.len()].iter().map(|&g| v.groups[g].mems[0]).collect();
            type Seen = (Vec<u64>, u32, u128, Vec<u128>, kiln_map::NestCost, kiln_map::NestCost);
            let mut seen: Vec<Seen> = vec![];
            for split in kiln_map::heuristic::candidates(op, &v, &units, 4) {
                let p = placement(op, 0, split, units.len());
                for s in Lowerer::slices(op, &p).iter().filter(|s| !s.is_empty()) {
                    let points = kiln_map::geom::slice_points(op, s);
                    let fps: Vec<u128> = (0..op.operands.len()).map(|oi| kiln_map::geom::footprint(op, oi, s, &prog.tensors[op.operands[oi].tensor].shape).elems()).collect();
                    let ext: Vec<u64> = (0..s.lo.len()).map(|d| s.extent(d)).collect();
                    let q = NestQuery { prog: &prog, op, slice: s, points, hw: &v.hw, unit: u.unit, level_caps: &caps, level_mems: &mems, level_bw: &[], gang: u.members.len() as u32, quick: true };
                    let (a, b) = (kc.cost(&q).expect("cost"), kiln_map::RooflineCost.cost(&q).expect("roofline"));
                    match seen.iter().find(|x| x.0 == ext && x.1 == s.seg && x.2 == points && x.3 == fps) {
                        Some(x) => {
                            assert_eq!((&x.4, &x.5), (&a, &b), "{}: {s:?}", op.id);
                            shared += 1;
                        }
                        None => seen.push((ext, s.seg, points, fps, a, b)),
                    }
                }
            }
        }
    }
    assert!(shared > 50, "{shared} shared slices");
}
