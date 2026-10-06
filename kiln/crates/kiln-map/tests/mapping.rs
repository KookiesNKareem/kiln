//! Partition coverage, mapping validation, serialization, moves and seeded search (06 §2.1 kiln-map row).

use std::collections::BTreeSet;
use std::sync::Arc;

use kiln_ir::bench::BenchOp;
use kiln_ir::hw::{Profile, check_file};
use kiln_map::geom::slice_points;
use kiln_map::heuristic::{MapOptions, candidates, placement};
use kiln_map::lower::Lowerer;
use kiln_map::mapping::{Lifetime, SplitAxis};
use kiln_map::search::{BeamSearch, MappingSearch, Move, Score, apply};
use kiln_map::{HwView, Mapping, Pool, Program, RooflineCost, heuristic, lower};
use proptest::prelude::*;

fn view(name: &str) -> HwView {
    let p = format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"));
    HwView::new(Arc::new(check_file(p, Profile::Reference).model.expect("expands"))).expect("view")
}

/// Every iteration point of every slice, enumerated (small shapes only).
fn points(prog: &Program, op: usize, split: &[SplitAxis]) -> Vec<Vec<u64>> {
    let o = &prog.ops[op];
    let p = placement(o, 0, split.to_vec(), 1);
    let mut out = vec![];
    for s in Lowerer::slices(o, &p).into_iter().filter(|s| !s.is_empty()) {
        let mut idx = s.lo.clone();
        'odometer: loop {
            out.push(idx.clone());
            let mut d = idx.len();
            loop {
                if d == 0 {
                    break 'odometer;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < s.hi[d] {
                    break;
                }
                idx[d] = s.lo[d];
            }
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Slices partition the iteration space: each point exactly once, uneven splits included (I5, L3).
    #[test]
    fn candidate_splits_cover_every_point_once(m in 1u64..13, n in 1u64..13, k in 1u64..9, b in prop::option::of(1u64..4)) {
        let v = view("tpu_v5e.json5");
        let op = match b { Some(b) => BenchOp::bmm(b, m, n, k), None => BenchOp::gemm(m, n, k, true) };
        let prog = Program::bench_op(&op).unwrap();
        let units = v.pool(Pool::Mac);
        let total: u64 = prog.ops[0].segs[0].ext.iter().product();
        for split in candidates(&prog.ops[0], &v, &units, 32) {
            let pts = points(&prog, 0, &split);
            let set: BTreeSet<Vec<u64>> = pts.iter().cloned().collect();
            prop_assert_eq!(pts.len() as u64, total, "{:?}", split);
            prop_assert_eq!(set.len() as u64, total);
            let p = placement(&prog.ops[0], 0, split.clone(), units.len());
            let sum: u128 = Lowerer::slices(&prog.ops[0], &p).iter().map(|s| slice_points(&prog.ops[0], s)).sum();
            prop_assert_eq!(sum, prog.ops[0].points());
        }
    }
}

fn gemv_program() -> Program {
    Program::bench_op(&BenchOp::gemm(8, 4096, 4096, true)).unwrap()
}

#[test]
fn heuristic_mapping_validates_round_trips_and_lowers_identically() {
    let v = view("tpu_v5e.json5");
    let prog = gemv_program();
    let (m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    assert!(m.validate(&prog, &v).is_empty(), "{:?}", m.validate(&prog, &v));
    let json = serde_json::to_string(&m).unwrap();
    let back: Mapping = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m);
    assert_eq!(back.hash(), m.hash());
    let (_, _, g0) = kiln_map::heuristic::heuristic_lowered(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let g1 = lower(&prog, &v, &m, &RooflineCost).unwrap();
    assert_eq!(g0.tasks, g1.tasks);
    assert_eq!(g0.demands, g1.demands);
    assert!(m.tensors.values().all(|t| t.lifetime != Lifetime::Private));
}

#[test]
fn validation_names_the_broken_field() {
    let v = view("tpu_v5e.json5");
    let prog = gemv_program();
    let (mut m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let op = prog.ops[0].id.clone();
    m.ops.get_mut(&op).unwrap().split = vec![SplitAxis { dim: "n".into(), parts: vec![4000, 95] }];
    let codes: Vec<String> = m.validate(&prog, &v).into_iter().map(|d| d.code).collect();
    assert!(codes.contains(&"E-MAP-VAL-006".to_string()), "{codes:?}");
    assert!(codes.contains(&"E-MAP-VAL-007".to_string()), "{codes:?}");
    let (mut m2, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    m2.unit_sets[1].units.push("board.chip.die.nope".into());
    assert!(m2.validate(&prog, &v).iter().any(|d| d.code == "E-MAP-VAL-001"));
}

#[test]
fn moves_apply_and_keep_mappings_valid() {
    let v = view("tpu_v5e.json5");
    let prog = gemv_program();
    let (mut m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let op = prog.ops[0].id.clone();
    apply(&mut m, &prog, &Move::Resplit { op: op.clone(), split: vec![SplitAxis { dim: "n".into(), parts: vec![1000, 1000, 1000, 1096] }] }).unwrap();
    assert!(m.validate(&prog, &v).is_empty());
    assert_eq!(m.ops[&op].slice_to_unit, vec![0, 1, 2, 3]);
    apply(&mut m, &prog, &Move::Reassign { op: op.clone(), slice: 3, unit: 0 }).unwrap();
    assert!(m.validate(&prog, &v).is_empty());
    assert!(apply(&mut m, &prog, &Move::Reassign { op, slice: 9, unit: 0 }).is_err());
    let g = lower(&prog, &v, &m, &RooflineCost).unwrap();
    assert_eq!(g.ops[0].slices, 4);
    assert_eq!(g.ops[0].units, 3);
}

#[test]
fn beam_search_is_seeded_and_never_worse() {
    let v = view("tpu_v5e.json5");
    let prog = Program::bench_op(&BenchOp::gemm(512, 1024, 768, true)).unwrap();
    let (m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let eval = |m: &Mapping| -> Option<Score> {
        let g = lower(&prog, &v, m, &RooflineCost).ok()?;
        let busy = g.tasks.iter().filter(|t| t.kind == kiln_map::lower::TaskKind::Compute).count() as f64;
        let t: f64 = g.ops.iter().map(|o| o.issued_macs as f64).sum::<f64>() / busy.max(1.0);
        Some(Score { time_s: t, op_times: g.ops.iter().map(|o| (prog.ops[o.op].id.clone(), t)).collect() })
    };
    let s0 = eval(&m).unwrap().time_s;
    let beam = BeamSearch::default();
    let a = beam.search(&prog, &v, m.clone(), &eval, 7);
    let b = beam.search(&prog, &v, m.clone(), &eval, 7);
    assert_eq!(a.best, b.best);
    assert_eq!(a.score, b.score);
    assert!(a.score <= s0);
    assert!(a.evaluated > 1);
    assert!(a.best.validate(&prog, &v).is_empty());
}

#[test]
fn transfers_follow_footprints_not_all_to_all() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = Program::bench_op(&BenchOp::gemm(8, 6144, 4096, true)).unwrap();
    let (m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let g = lower(&prog, &v, &m, &RooflineCost).unwrap();
    let weight = 6144.0 * 4096.0 * 2.0;
    let into_l2: f64 = g.ops[0].level_bytes.iter().filter(|(gix, _)| v.groups[**gix].mems.len() == 80).map(|x| x.1).sum();
    assert!(into_l2 >= weight && into_l2 < weight * 1.05, "L2 receives {into_l2} B for a {weight} B weight");
}

#[test]
fn units_without_the_precision_are_never_used() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = gemv_program();
    let (m, _) = heuristic(&prog, &v, &RooflineCost, &MapOptions::default()).unwrap();
    let set = m.set_of(&prog.ops[0].id).unwrap();
    assert_eq!(set.name, "mac.bf16xbf16");
    assert_eq!(set.units.len(), 108, "one gang of 4 tensor cores per SM");
}
