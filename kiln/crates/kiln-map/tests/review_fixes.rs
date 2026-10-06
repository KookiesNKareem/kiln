//! Regressions from the cost/map review: what the physics prices must reach the task graph.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::heuristic::placement;
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
fn infeasible_tiles_are_errors_not_roofline_estimates() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = program("llama3_8b:decode_b8");
    let op = prog.ops.iter().find(|o| o.class() == KernelClass::Contraction).expect("a contraction");
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let u = &v.units[v.pool(Pool::Mac)[0]];
    let caps = vec![1u64; u.private.max(1)];
    let mems: Vec<usize> = u.chain[..caps.len()].iter().map(|&g| v.groups[g].mems[0]).collect();
    let q = NestQuery { prog: &prog, op, slice: &s, points: kiln_map::geom::slice_points(op, &s), hw: &v.hw, unit: u.unit, level_caps: &caps, level_mems: &mems, level_bw: &[], residency: &[], gang: u.members.len() as u32, quick: true };
    let kc = KilnCost::new();
    let e = kc.cost(&q).expect_err("one byte holds no tile");
    assert_eq!(e.code, "E-COST-INFEASIBLE", "{e:?}");
    assert_eq!(kc.fallbacks.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// The reference design with `from` replaced by `to` in its source, under its own design hash.
fn variant(name: &str, from: &str, to: &str) -> HwView {
    let src = std::fs::read_to_string(format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"))).expect("design");
    assert!(src.contains(from), "{from}");
    let dir = std::env::temp_dir().join(format!("kiln-map-review-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = dir.join(format!("{}-{name}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&p, src.replace(from, to)).expect("write");
    HwView::new(Arc::new(check_file(&p, Profile::Reference).model.expect("expands"))).expect("view")
}

fn whole_query<'a>(v: &'a HwView, prog: &'a Program, op: &'a kiln_map::program::POp, s: &'a kiln_map::geom::Slice) -> NestQuery<'a> {
    let u = &v.units[v.pool(Pool::Mac)[0]];
    NestQuery { prog, op, slice: s, points: kiln_map::geom::slice_points(op, s), hw: &v.hw, unit: u.unit, level_caps: &[], level_mems: &[], level_bw: &[], residency: &[], gang: u.members.len() as u32, quick: true }
}

#[test]
fn unit_templates_are_never_shared_across_designs() {
    let fast = view("a100_sxm4_40gb.json5");
    let slow = variant("a100_sxm4_40gb.json5", "\"bf16*bf16+fp32@0.125\"", "\"bf16*bf16+fp32@0.0625\"");
    assert_ne!(fast.hw.design_hash, slow.hw.design_hash);
    let prog = program("llama3_8b:prefill_b1");
    let op = prog.ops.iter().find(|o| o.class() == KernelClass::Contraction).expect("a contraction");
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let alone = KilnCost::new().cost(&whole_query(&slow, &prog, op, &s)).expect("cost").cycles;
    let kc = KilnCost::new();
    let first = kc.cost(&whole_query(&fast, &prog, op, &s)).expect("cost").cycles;
    let second = kc.cost(&whole_query(&slow, &prog, op, &s)).expect("cost").cycles;
    assert!(first < alone, "{first} vs {alone}");
    assert_eq!(second, alone, "the half-rate design answered from the full-rate template");
}

/// The reference design with every vector unit disabled.
fn without_vector_units(name: &str) -> HwView {
    let p = format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"));
    let mut hw = check_file(p, Profile::Reference).model.expect("expands");
    for u in 0..hw.units.len() {
        if matches!(hw.units[u].spec.kind, kiln_ir::hw::compute::ComputeKind::Vector(_)) {
            let n = hw.units[u].node;
            hw.nodes[n].enabled = false;
        }
    }
    HwView::new(Arc::new(hw)).expect("view")
}

fn single_group(op: &str) -> kiln_map::mapping::ExecGroup {
    use kiln_map::mapping::{ExecGroup, GroupKind, LaunchKind};
    ExecGroup { ops: vec![op.into()], kind: GroupKind::Single, launch: LaunchKind::HostLaunch, barrier_after: true }
}

#[test]
fn k_split_combines_need_a_unit_to_run_on() {
    use kiln_map::mapping::SplitAxis;
    let v = view("a100_sxm4_40gb.json5");
    let bare = without_vector_units("a100_sxm4_40gb.json5");
    assert!(bare.pool(Pool::Vector).is_empty());
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 4096, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[0];
    let units = bare.pool(Pool::Mac);
    let p = placement(op, 0, vec![SplitAxis { dim: "k".into(), parts: vec![2048, 2048] }], units.len());
    let mut l = Lowerer::new(&prog, &bare, &m, &kiln_map::RooflineCost).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    let e = l.lower_op(0, &p, &units, true).expect_err("no unit adds the partial sums");
    assert_eq!(e.code, "E-MAP-OP-004", "{e:?}");
}

#[test]
fn fused_converts_need_a_unit_to_run_on() {
    let v = view("a100_sxm4_40gb.json5");
    let bare = without_vector_units("a100_sxm4_40gb.json5");
    let w = kiln_wl::zoo::workload("llama3_8b:decode_b8+weights=fp8_e4m3").unwrap();
    let (_, mut lg, _) = kiln_wl::evaluate_snapshot(w.model(), w.scenario()).unwrap();
    kiln_wl::convert::insert_converts(&mut lg, &v.mac_modes(false)).unwrap();
    let prog = Program::whole_step(w.model(), &lg, 1).unwrap();
    let oi = (0..prog.ops.len()).find(|&i| (0..prog.ops[i].operands.len()).any(|o| prog.converted_from(i, o).is_some())).expect("a fused convert");
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[oi];
    let units = bare.pool(Pool::Mac);
    let p = placement(op, 0, vec![], units.len());
    let mut l = Lowerer::new(&prog, &bare, &m, &kiln_map::RooflineCost).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    let e = l.lower_op(oi, &p, &units, true).expect_err("no unit converts the operand");
    assert_eq!(e.code, "E-MAP-OP-004", "{e:?}");
}

#[test]
fn contraction_vector_work_occupies_a_vector_unit() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 64, true)).unwrap();
    let kc = KilnCost::new();
    let (_, _, g) = kiln_map::heuristic::heuristic_lowered(&prog, &v, &kc, &kiln_map::heuristic::MapOptions::default()).unwrap();
    assert!(g.ops[0].vec_ops >= 64 * 64, "the fp32 -> bf16 down-conversion of every output: {}", g.ops[0].vec_ops);
    let vector: Vec<u32> = v.pool(Pool::Vector).iter().flat_map(|&u| v.units[u].members.iter().map(|&m| v.units[m].compute)).collect();
    let on_vector: f64 = g
        .tasks
        .iter()
        .filter(|t| t.kind == kiln_map::lower::TaskKind::Compute)
        .flat_map(|t| g.demands_of(t))
        .map(|a| match *a {
            kiln_map::lower::Amount::Res(r, x) if vector.contains(&r) => x,
            _ => 0.0,
        })
        .sum();
    assert!(on_vector > 0.0, "vector work reported but never scheduled");
}

/// Bytes the lowered gemm writes into HBM on v5e with `vmem_cap`, and the output's bytes.
fn gemm_hbm_writes(vmem_cap: &str) -> (f64, f64) {
    let v = variant("tpu_v5e.json5", "vmem_cap: \"128MiB\"", &format!("vmem_cap: \"{vmem_cap}\""));
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(2048, 4096, 4096, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[0];
    let units = v.pool(Pool::Mac);
    let kc = KilnCost::new();
    let mut l = Lowerer::new(&prog, &v, &m, &kc).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    l.lower_op(0, &placement(op, 0, vec![], units.len()), &units, true).unwrap();
    l.end_group();
    let g = l.finish();
    let hbm = v.offchip.expect("hbm");
    (g.ops[0].level_bytes.get(&hbm).copied().unwrap_or(0.0), 2048.0 * 4096.0 * 2.0)
}

#[test]
fn partial_sum_spills_reach_the_task_graph() {
    let (into_hbm, out) = gemm_hbm_writes("4MiB");
    assert!(into_hbm > 4.0 * out, "partial sums spilled to HBM: {into_hbm} B written vs {out} B of output");
    // With v5e's 128 MiB of vmem the output-stationary nest keeps them on chip: only the result is written.
    let (into_hbm, out) = gemm_hbm_writes("128MiB");
    assert_eq!(into_hbm, out);
}

#[test]
fn staged_copies_never_outlive_a_write() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = program("llama3_8b:decode_b8");
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    // A contraction reading a tensor an earlier op writes (and no private temps, which only its own node produces).
    let private = |t: usize| m.tensors.get(&prog.tensors[prog.root(t)].id).is_some_and(|p| p.lifetime == kiln_map::mapping::Lifetime::Private);
    let (c, p) = (0..prog.ops.len())
        .filter(|&c| prog.ops[c].class() == KernelClass::Contraction && prog.placed(c) && !prog.ops[c].operands.iter().any(|o| private(o.tensor)))
        .find_map(|c| {
            let ins: Vec<usize> = prog.ops[c].operands.iter().filter(|o| o.access.reads()).map(|o| prog.root(o.tensor)).collect();
            (0..c)
                .rev()
                .find(|&p| prog.placed(p) && !prog.ops[p].operands.iter().any(|o| o.access.reads() && private(o.tensor)) && prog.ops[p].operands.iter().any(|o| o.access.writes() && ins.contains(&prog.root(o.tensor))))
                .map(|p| (c, p))
        })
        .expect("a producer and its consumer");
    let mut l = Lowerer::new(&prog, &v, &m, &kiln_map::RooflineCost).unwrap();
    l.begin_group(String::new(), &single_group(&prog.ops[c].id), None);
    let units = |i: usize| l.units_of(&m, &prog.ops[i].id).unwrap();
    let (uc, up) = (units(c), units(p));
    l.lower_op(c, &m.ops[&prog.ops[c].id], &uc, true).unwrap();
    l.lower_op(p, &m.ops[&prog.ops[p].id], &up, true).unwrap();
    let t0 = l.g.tasks.len();
    l.lower_op(c, &m.ops[&prog.ops[c].id], &uc, true).unwrap();
    let after_write = l.g.tasks[t0..].iter().any(|t| l.g.preds_of(t).iter().any(|&x| l.g.tasks[x as usize].op as usize == p));
    assert!(after_write, "the re-read of {} waits for {}'s write", prog.ops[c].id, prog.ops[p].id);
}

#[test]
fn tensor_homes_must_partition_every_used_tensor() {
    use kiln_map::mapping::{Lifetime, TensorRegion};
    let v = view("tpu_v5e.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(8, 512, 256, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    assert!(m.validate(&prog, &v).is_empty());
    let codes = |m: &kiln_map::Mapping| m.validate(&prog, &v).into_iter().map(|d| d.code).collect::<Vec<_>>();
    let (id, tp) = m.tensors.iter().find(|(_, t)| t.home.len() == 1 && t.home[0].region.hi[0] >= 2).expect("a homed tensor");
    let shape = tp.home[0].region.hi.clone();
    let half = |lo: u64, hi: u64| {
        let mut h = tp.home[0].clone();
        let (mut l, mut u) = (vec![0; shape.len()], shape.clone());
        l[0] = lo;
        u[0] = hi;
        h.region = TensorRegion { lo: l, hi: u };
        h
    };
    let mid = shape[0] / 2;
    let mut dup = m.clone();
    // Two copies of the lower half: the volumes add up, the upper half has no home.
    let lower = half(0, mid);
    let mut lower2 = half(0, mid);
    lower2.region.hi[0] = shape[0] - mid;
    dup.tensors.get_mut(id).unwrap().home = vec![lower, lower2];
    assert!(codes(&dup).contains(&"E-MAP-VAL-011".to_string()), "{:?}", codes(&dup));
    let mut out_of_bounds = m.clone();
    out_of_bounds.tensors.get_mut(id).unwrap().home = vec![half(1, shape[0] + 1)];
    assert!(codes(&out_of_bounds).contains(&"E-MAP-VAL-011".to_string()));
    let mut missing = m.clone();
    missing.tensors.shift_remove(id);
    assert!(!codes(&missing).is_empty(), "a used tensor without a home");
    let mut private = m.clone();
    private.tensors.get_mut(id).unwrap().lifetime = Lifetime::Private;
    assert!(!codes(&private).is_empty(), "a private input nothing produces");
}

#[test]
fn heuristic_whole_step_mappings_validate() {
    for (d, fuse) in [("tpu_v5e.json5", true), ("a100_sxm4_40gb.json5", false)] {
        let v = view(d);
        let prog = program("llama3_8b:decode_b8");
        let opts = kiln_map::heuristic::MapOptions { fuse_elementwise: fuse, ..Default::default() };
        let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &opts).unwrap();
        assert!(m.validate(&prog, &v).is_empty(), "{d}: {:?}", m.validate(&prog, &v));
    }
}

fn mx_gemm(m: u64, n: u64, k: u64) -> Program {
    let mut b = kiln_ir::bench::BenchOp::gemm(m, n, k, true);
    for x in ["a", "b"] {
        b.operands.get_mut(x).unwrap().dtype = "mxfp8_e4m3".into();
    }
    Program::bench_op(&b).unwrap()
}

#[test]
fn mx_scale_bytes_are_fed() {
    let v = view("ember.json5");
    let prog = mx_gemm(64, 256, 512);
    let op = &prog.ops[0];
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let q = whole_query(&v, &prog, op, &s);
    let mr = op.mac.as_ref().unwrap();
    // The roofline reads each input once: its elements and one E8M0 scale per 32.
    let roof = kiln_map::RooflineCost.cost(&q).unwrap();
    assert_eq!(roof.feed_bytes[mr.b], (256.0 * 512.0) * 33.0 / 32.0);
    // kiln-cost: the feed carries what the loop nest reads at the feed level, scale streams included.
    let t = kiln_cost::UnitTemplate::from_hw(&v.hw, q.unit, &kiln_cost::TemplateOptions { gang: q.gang, ..Default::default() }).unwrap();
    use kiln_ir::hw::compute::OperandRole as R;
    let roles: Vec<R> = (0..op.operands.len()).map(|i| if i == mr.a { R::A } else if i == mr.b { R::B } else { R::O }).collect();
    let dt: Vec<_> = op.operands.iter().map(|o| kiln_wl::convert::operand_spec(&prog.tensors[o.tensor].dtype)).collect();
    let nest = kiln_cost::OpNest::from_kernel(&op.kernel, &dt, Some(&roles)).unwrap();
    let opts = kiln_cost::CostOptions { budget: kiln_cost::SearchBudget { top_k_spatial: 2, max_evals_per_spatial: 300, stop_at_floor: true }, ..Default::default() };
    let e = kiln_cost::cost(&kiln_cost::CostQuery { unit: &t, nest: &nest, objective: kiln_cost::Objective::Latency, options: opts }).unwrap();
    let l0 = t.chains.iter().find(|c| c.role == R::A).unwrap().levels.iter().copied().find(|&l| t.levels[l].mem.is_some_and(|m| !v.hw.memories[m].is_local())).unwrap();
    let streams = nest.stream_operands();
    let want: u64 = e.accesses.iter().filter(|a| a.level == l0 && streams[a.operand] == mr.a).map(|a| a.read_bytes).sum();
    let elems: u64 = e.accesses.iter().filter(|a| a.level == l0 && a.operand == mr.a).map(|a| a.read_bytes).sum();
    assert!(want > elems);
    assert_eq!(KilnCost::new().cost(&q).unwrap().feed_bytes[mr.a], want as f64);
}

#[test]
fn sharded_output_homes_each_receive_their_part() {
    use kiln_map::mapping::TensorRegion;
    let v = view("tpu_v5e.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(8, 512, 256, true)).unwrap();
    let (mut m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[0];
    let out = &prog.tensors[op.operands[op.mac.as_ref().unwrap().out].tensor];
    let tp = m.tensors.get_mut(&out.id).unwrap();
    let h = tp.home[0].clone();
    let part = |lo: u64, hi: u64| kiln_map::mapping::Home { region: TensorRegion { lo: vec![lo, 0], hi: vec![hi, out.shape[1]] }, ..h.clone() };
    tp.home = vec![part(0, 4), part(4, 8)];
    assert!(m.validate(&prog, &v).is_empty(), "{:?}", m.validate(&prog, &v));
    let g = kiln_map::lower(&prog, &v, &m, &kiln_map::RooflineCost).unwrap();
    let into_hbm = g.ops[0].level_bytes.get(&v.offchip.unwrap()).copied().unwrap_or(0.0);
    assert_eq!(into_hbm, out.footprint() as f64);
}

#[test]
fn routing_choices_lowering_ignores_are_rejected() {
    use kiln_map::mapping::{Route, RoutingPolicy};
    let v = view("tpu_v5e.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(8, 512, 256, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let mut routed = m.clone();
    let op = prog.ops[0].id.clone();
    kiln_map::search::apply(&mut routed, &prog, &kiln_map::search::Move::Reroute { transfer: format!("{op}.o0.s0.h0"), route: Route::Path { links: vec!["x".into()] } }).unwrap();
    assert!(!routed.validate(&prog, &v).is_empty(), "an explicit route lowering does not apply");
    let mut dor = m.clone();
    dor.routing = RoutingPolicy::DimensionOrder;
    assert!(!dor.validate(&prog, &v).is_empty(), "a routing policy lowering does not apply");
}

#[test]
fn placements_run_on_their_own_unit_set_whatever_its_name() {
    use kiln_map::mapping::{Target, UnitSet};
    let v = view("a100_sxm4_40gb.json5");
    let prog = program("llama3_8b:decode_b8");
    let (mut m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let mut macs = (0..prog.ops.len()).filter(|&i| prog.placed(i) && prog.ops[i].class() == KernelClass::Contraction);
    let (first, second) = (macs.next().unwrap(), macs.next().unwrap());
    let set = m.set_of(&prog.ops[first].id).unwrap().clone();
    let last = set.units.last().unwrap().clone();
    m.unit_sets.push(UnitSet { name: set.name.clone(), units: vec![last.clone()] });
    let p = m.ops.get_mut(&prog.ops[second].id).unwrap();
    p.target = Target::Units { set: (m.unit_sets.len() - 1) as u32 };
    p.slice_to_unit.iter_mut().for_each(|u| *u = 0);
    assert!(m.validate(&prog, &v).is_empty(), "{:?}", m.validate(&prog, &v));
    let g = kiln_map::lower(&prog, &v, &m, &kiln_map::RooflineCost).unwrap();
    let u = v.unit_by_path(&last).unwrap();
    let mine: Vec<u32> = v.units[u].members.iter().map(|&x| v.units[x].compute).collect();
    let used: Vec<u32> = g
        .tasks
        .iter()
        .filter(|t| t.op as usize == second && t.kind == kiln_map::lower::TaskKind::Compute)
        .flat_map(|t| g.demands_of(t))
        .filter_map(|a| match *a {
            kiln_map::lower::Amount::Res(r, _) if v.resources[r as usize].kind == kiln_trace::sim::ResourceKind::ComputeUnit => Some(r),
            _ => None,
        })
        .collect();
    assert!(!used.is_empty() && used.iter().all(|r| mine.contains(r)), "ran on {used:?}, not {mine:?}");
}

#[test]
fn fused_converts_and_combines_cost_energy() {
    use kiln_map::mapping::SplitAxis;
    let v = view("a100_sxm4_40gb.json5");
    let w = kiln_wl::zoo::workload("llama3_8b:decode_b8+weights=fp8_e4m3").unwrap();
    let (_, mut lg, _) = kiln_wl::evaluate_snapshot(w.model(), w.scenario()).unwrap();
    kiln_wl::convert::insert_converts(&mut lg, &v.mac_modes(false)).unwrap();
    let prog = Program::whole_step(w.model(), &lg, 1).unwrap();
    let oi = (0..prog.ops.len()).find(|&i| (0..prog.ops[i].operands.len()).any(|o| prog.converted_from(i, o).is_some())).expect("a fused convert");
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[oi];
    let units = v.pool(Pool::Mac);
    let k = op.kernel.dims.iter().find(|d| d.kind == kiln_ir::wl::DimKind::Reduction).unwrap();
    let p = placement(op, 0, vec![SplitAxis { dim: k.name.clone(), parts: kiln_map::geom::even_parts(k.extent, 2) }], units.len());
    let mut l = Lowerer::new(&prog, &v, &m, &kiln_map::RooflineCost).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    l.lower_op(oi, &p, &units, true).unwrap();
    let st = l.g.ops[0].clone();
    let (e_mac, _, _, _) = v.phys.unit_energies(v.units[units[0]].unit, &st.mode, (16, 16));
    let vu = v.units[v.pool(Pool::Vector)[0]].unit;
    let (_, _, e_elem, _) = v.phys.unit_energies(vu, "fp32", (32, 32));
    assert!(st.vec_ops > 0 && e_elem > 0.0);
    let floor = st.useful_macs as f64 * e_mac + st.vec_ops as f64 * e_elem.min(v.phys.unit_energies(vu, "fp16", (16, 16)).2).min(v.phys.unit_energies(vu, "bf16", (16, 16)).2);
    assert!(st.e_compute_j >= floor * (1.0 - 1e-9), "{} J < {floor} J: converts and combines are free", st.e_compute_j);
}

const A100_ALU: &str = "precisions: [\"fp32@1\", \"int32@1\", \"fp16@4\", \"bf16@4\", \"fp64@0.5\"]";

/// The A100 with its vector ALUs declaring only `modes`.
fn alu_modes(modes: &str) -> HwView {
    variant("a100_sxm4_40gb.json5", A100_ALU, &format!("precisions: [{modes}]"))
}

fn map_cost(v: &HwView, prog: &Program, oi: usize) -> Result<kiln_map::NestCost, kiln_ir::common::Diagnostic> {
    let op = &prog.ops[oi];
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let u = &v.units[v.pool(Pool::Vector)[0]];
    let q = NestQuery { prog, op, slice: &s, points: kiln_map::geom::slice_points(op, &s), hw: &v.hw, unit: u.unit, level_caps: &[], level_mems: &[], level_bw: &[], residency: &[], gang: u.members.len() as u32, quick: true };
    kiln_map::RooflineCost.cost(&q)
}

#[test]
fn float_map_kernels_need_a_float_vector_mode() {
    let prog = program("llama3_8b:prefill_b1");
    let oi = prog.ops.iter().position(|o| o.class() == KernelClass::Map).expect("a map kernel");
    assert!(map_cost(&alu_modes("\"fp32@1\""), &prog, oi).is_ok());
    let e = map_cost(&alu_modes("\"int32@1\""), &prog, oi).expect_err("an int32 ALU does not run float math");
    assert_eq!(e.code, "E-MAP-PREC-002", "{e:?}");
}

#[test]
fn fractional_vector_rates_are_not_rounded_up() {
    let prog = program("llama3_8b:prefill_b1");
    let oi = prog.ops.iter().position(|o| o.class() == KernelClass::Map && o.points() > 1 << 20).expect("a large map kernel");
    let fast = map_cost(&alu_modes("\"fp32@1\""), &prog, oi).unwrap().cycles;
    let slow = map_cost(&alu_modes("\"fp32@0.0078125\""), &prog, oi).unwrap().cycles;
    assert!((slow / fast - 128.0).abs() < 1e-3 * 128.0, "{slow} vs {fast}: 64 lanes at 1/128 op per cycle");
}

/// Reduce-task vector cycles and the error, if any, of lowering a two-way k-split GEMM on `v`.
fn combine_cycles(v: &HwView) -> Result<f64, kiln_ir::common::Diagnostic> {
    use kiln_map::mapping::SplitAxis;
    let base = view("a100_sxm4_40gb.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 4096, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &base, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[0];
    let units = v.pool(Pool::Mac);
    let p = placement(op, 0, vec![SplitAxis { dim: "k".into(), parts: vec![2048, 2048] }], units.len());
    let mut l = Lowerer::new(&prog, v, &m, &kiln_map::RooflineCost).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    l.lower_op(0, &p, &units, true)?;
    Ok(l.g
        .tasks
        .iter()
        .filter(|t| t.kind == kiln_map::lower::TaskKind::Reduce)
        .flat_map(|t| l.g.demands_of(t))
        .map(|a| match *a {
            kiln_map::lower::Amount::Res(r, x) if v.resources[r as usize].kind == kiln_trace::sim::ResourceKind::ComputeUnit => x,
            _ => 0.0,
        })
        .sum())
}

#[test]
fn split_combines_run_at_the_accumulator_mode_rate() {
    let full = combine_cycles(&alu_modes("\"fp32@1\"")).unwrap();
    let eighth = combine_cycles(&alu_modes("\"fp32@0.125\"")).unwrap();
    assert!(full > 0.0 && (eighth / full - 8.0).abs() < 1e-9, "{eighth} vs {full}");
    let e = combine_cycles(&alu_modes("\"int32@1\"")).expect_err("int32 ALUs cannot add fp32 partial sums");
    assert_eq!(e.code, "E-MAP-PREC-002", "{e:?}");
}

#[test]
fn contraction_conversions_need_an_accumulator_mode() {
    let v = alu_modes("\"bf16@4\"");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 64, true)).unwrap();
    let kc = KilnCost::new();
    let e = kiln_map::heuristic::heuristic_lowered(&prog, &v, &kc, &kiln_map::heuristic::MapOptions::default()).expect_err("fp32 -> bf16 needs an fp32 mode");
    assert_eq!(e.code, "E-MAP-PREC-002", "{e:?}");
}

#[test]
fn vector_work_stays_beside_the_mac_feed() {
    let src = std::fs::read_to_string(format!("{}/../../designs/reference/a100_sxm4_40gb.json5", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let alu = format!("{A100_ALU}, // pub:cuda 64/256/256/32 per SM per clk\n          feeds: {{ any: \"rf\" }} }},");
    assert!(src.contains(&alu));
    let src = src
        .replace(&alu, &format!("{A100_ALU},\n          feeds: {{ any: \"vrf\" }} }},"))
        .replace("        { id: \"rf\", kind: \"register_file\"", "        { id: \"vrf\", kind: \"register_file\", capacity: \"64KiB\", ports: [ { dir: \"rw\", width_bits: 1024 } ] },\n        { id: \"rf\", kind: \"register_file\"")
        .replace("endpoints: [\"smsp*.rf\", \"l1\"]", "endpoints: [\"smsp*.rf\", \"smsp*.vrf\", \"l1\"]");
    let p = std::env::temp_dir().join(format!("kiln-map-review-{}-vrf.json5", std::process::id()));
    std::fs::write(&p, src).unwrap();
    let r = check_file(&p, Profile::Reference);
    let v = HwView::new(Arc::new(r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics)))).expect("view");
    let (mac, vec) = (v.pool(Pool::Mac)[0], v.pool(Pool::Vector)[0]);
    assert_ne!(v.units[mac].chain[0], v.units[vec].chain[0]);
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 64, true)).unwrap();
    let kc = KilnCost::new();
    let e = kiln_map::heuristic::heuristic_lowered(&prog, &v, &kc, &kiln_map::heuristic::MapOptions::default()).expect_err("the conversion's partials never reach the vector unit");
    assert_eq!(e.code, "E-MAP-OP-004", "{e:?}");
}

#[test]
fn vector_modes_hold_every_operand_precision() {
    use kiln_ir::precision::Precision;
    let prog = program("llama3_8b:prefill_b1");
    let float = |p: &Program, oi: usize| p.ops[oi].operands.iter().map(|o| p.tensors[o.tensor].dtype.scalar.compute()).filter(|x| x.is_float()).collect::<Vec<_>>();
    let oi = (0..prog.ops.len())
        .find(|&i| prog.ops[i].class() == KernelClass::Map && float(&prog, i).first() == Some(&Precision::Bf16) && float(&prog, i).contains(&Precision::Fp32))
        .expect("a map kernel over bf16 and fp32 operands");
    let e = map_cost(&alu_modes("\"bf16@4\""), &prog, oi).expect_err("a bf16-only ALU cannot hold the fp32 operand");
    assert_eq!(e.code, "E-MAP-PREC-002", "{e:?}");
    assert!(map_cost(&alu_modes("\"fp32@1\""), &prog, oi).unwrap().mode.starts_with("fp32"));
    let mut ints = prog.clone();
    let ts: Vec<usize> = ints.ops[oi].operands.iter().map(|o| o.tensor).collect();
    for t in ts {
        ints.tensors[t].dtype = kiln_ir::wl::ElemType::plain(Precision::Int32);
    }
    let c = map_cost(&alu_modes("\"int32@1\""), &ints, oi).expect("an int32 kernel runs on an int32 ALU");
    assert!(c.mode.starts_with("int32"), "{}", c.mode);
}

#[test]
fn homes_name_exactly_one_memory_group() {
    let v = view("a100_sxm4_40gb.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 4096, true)).unwrap();
    let (mut m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let hbm = v.offchip.unwrap();
    let stacks = v.group_paths(hbm);
    assert!(stacks.len() > 1);
    let tp = m.tensors.values_mut().find(|t| t.home.first().is_some_and(|h| h.mems == stacks)).expect("an HBM-homed tensor");
    tp.home[0].mems = vec![stacks[0].clone()];
    let codes: Vec<String> = m.validate(&prog, &v).into_iter().map(|d| d.code).collect();
    assert!(codes.contains(&"E-MAP-VAL-011".to_string()), "one stack of an interleaved group: {codes:?}");
    assert!(kiln_map::lower(&prog, &v, &m, &kiln_map::RooflineCost).is_err(), "lowered over all {} stacks", stacks.len());
}

#[test]
fn private_reads_need_every_element_produced_in_their_span() {
    use kiln_map::mapping::Lifetime;
    let v = view("a100_sxm4_40gb.json5");
    let prog = program("llama3_8b:decode_b8");
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    // The first KV-cache append and the first op reading the cache after it.
    let (p, c, t) = (0..prog.ops.len())
        .filter(|&p| prog.placed(p) && prog.ops[p].class() == KernelClass::Scatter)
        .find_map(|p| {
            let t = prog.ops[p].operands.iter().find(|o| o.access.writes())?.tensor;
            let t = prog.root(t);
            let c = (p + 1..prog.ops.len()).find(|&c| prog.placed(c) && prog.ops[c].operands.iter().any(|o| o.access.reads() && prog.root(o.tensor) == t))?;
            Some((p, c, t))
        })
        .expect("an append and its reader");
    let mut bad = m.clone();
    bad.tensors.get_mut(&prog.tensors[t].id).unwrap().lifetime = Lifetime::Private;
    let gp = bad.groups.iter().position(|g| g.ops.contains(&prog.ops[p].id)).unwrap();
    let gc = bad.groups.iter().position(|g| g.ops.contains(&prog.ops[c].id)).unwrap();
    for g in &mut bad.groups[gp..gc] {
        g.barrier_after = false;
    }
    let codes: Vec<String> = bad.validate(&prog, &v).into_iter().map(|d| d.code).collect();
    assert!(codes.contains(&"E-MAP-VAL-017".to_string()), "the append writes one row of the cache, the reader reads all: {codes:?}");
    let e = kiln_map::lower(&prog, &v, &bad, &kiln_map::RooflineCost).expect_err("unproduced rows have no source");
    assert_eq!(e.code, "E-MAP-VAL-017", "{e:?}");
}

/// Most bytes of non-resident tensors homed on group `g` live at once (a tensor lives from the first op touching it to
/// the last, model state for the whole program).
fn live_peak(prog: &Program, m: &kiln_map::Mapping, v: &HwView, g: usize) -> u128 {
    use kiln_map::mapping::Lifetime;
    let n = prog.ops.len();
    let mut live = vec![0u128; n];
    for (id, tp) in &m.tensors {
        if tp.lifetime == Lifetime::Resident {
            continue;
        }
        let t = prog.tensor(id).unwrap();
        let bytes: u128 = tp.home.iter().filter(|h| v.group_by_paths(&h.mems) == Some(g)).map(|h| prog.tensors[t].bytes(h.region.lo.iter().zip(&h.region.hi).map(|(l, h)| u128::from(h - l)).product())).sum();
        let uses: Vec<usize> = (0..n).filter(|&i| prog.ops[i].operands.iter().any(|o| prog.root(o.tensor) == t)).collect();
        let (a, b) = if prog.tensors[t].model_state() { (0, n - 1) } else { (*uses.first().unwrap_or(&0), *uses.last().unwrap_or(&0)) };
        live[a..=b].iter_mut().for_each(|x| *x += bytes);
    }
    live.into_iter().max().unwrap_or(0)
}

#[test]
fn live_activations_fit_their_on_chip_home() {
    use kiln_map::mapping::{Home, Lifetime, TensorRegion};
    // Decode b8's 64 KiB activations each fit 128 KiB of vmem; three of them are live at once.
    let v = variant("tpu_v5e.json5", "vmem_cap: \"128MiB\"", "vmem_cap: \"128KiB\"");
    let prog = program("llama3_8b:decode_b8");
    let s = v.shared_onchip().expect("vmem");
    let cap = u128::from(v.groups[s].capacity);
    let opts = kiln_map::heuristic::MapOptions { onchip_activation_fraction: 1.0, ..Default::default() };
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &opts).unwrap();
    assert!(m.validate(&prog, &v).is_empty(), "{:?}", m.validate(&prog, &v));
    let peak = live_peak(&prog, &m, &v, s);
    assert!(peak > 0 && peak <= cap, "{peak} B of activations live at once in {cap} B of vmem");
    // Every activation that fits vmem on its own, homed there.
    let mut all = m.clone();
    for (id, tp) in all.tensors.iter_mut() {
        let t = &prog.tensors[prog.tensor(id).unwrap()];
        if matches!(tp.lifetime, Lifetime::Spilled { .. } | Lifetime::Streamed) && t.class == kiln_ir::wl::TensorClass::Activation && t.footprint() <= cap {
            *tp = kiln_map::mapping::TensorPlacement { home: vec![Home { region: TensorRegion { lo: vec![0; t.shape.len()], hi: t.shape.clone() }, mems: v.group_paths(s) }], interleave_granule_b: None, lifetime: Lifetime::Streamed };
        }
    }
    assert!(live_peak(&prog, &all, &v, s) > cap);
    let codes: Vec<String> = all.validate(&prog, &v).into_iter().map(|d| d.code).collect();
    assert!(codes.contains(&"E-MAP-CAP-003".to_string()), "{codes:?}");
}

#[test]
fn slices_are_costed_at_their_position() {
    use kiln_map::mapping::SplitAxis;
    let v = view("ember.json5");
    let prog = mx_gemm(64, 256, 64);
    let op = &prog.ops[0];
    let a = op.mac.as_ref().unwrap().a;
    // k slices [0, 2) and [31, 33): equal extents, one and two blocks of 32 scales.
    let split = vec![SplitAxis { dim: "k".into(), parts: vec![2, 29, 2, 31] }];
    let slices = Lowerer::slices(op, &placement(op, 0, split.clone(), 1));
    let kc = KilnCost::new();
    let fed = |s: &kiln_map::geom::Slice| kc.cost(&whole_query(&v, &prog, op, s)).unwrap().feed_bytes[a];
    assert_eq!(fed(&slices[2]) - fed(&slices[0]), 64.0, "64 rows read one more E8M0 scale each");
    // Lowered, the second slice must not reuse the first's cost.
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let u = v.pool(Pool::Mac)[0];
    let feed = v.units[u].feed_in.iter().find(|x| x.0 == kiln_ir::hw::compute::OperandRole::A).unwrap().1;
    let mut l = Lowerer::new(&prog, &v, &m, &kc).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    l.lower_op(0, &placement(op, 0, split, 1), &[u], true).unwrap();
    let per_task: Vec<f64> = l
        .g
        .tasks
        .iter()
        .filter(|t| t.kind == kiln_map::lower::TaskKind::Compute)
        .map(|t| l.g.demands_of(t).iter().map(|d| if let kiln_map::lower::Amount::Res(r, x) = *d { if r == feed { x } else { 0.0 } } else { 0.0 }).sum())
        .filter(|&x: &f64| x > 0.0)
        .collect();
    assert_eq!(per_task.len(), 4, "{per_task:?}");
    assert_eq!(per_task[2] - per_task[0], 64.0, "{per_task:?}");
}

/// MAC compute cycles of a lowered 1024^3 gemm on `v` with every operand homed on group `g`.
fn mac_cycles_homed(v: &HwView, g: usize) -> f64 {
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(1024, 1024, 1024, true)).unwrap();
    let (mut m, _) = kiln_map::heuristic(&prog, v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    for tp in m.tensors.values_mut() {
        tp.home.iter_mut().for_each(|h| h.mems = v.group_paths(g));
        tp.lifetime = kiln_map::mapping::Lifetime::Streamed;
    }
    assert!(m.validate(&prog, v).is_empty(), "{:?}", m.validate(&prog, v));
    let g = kiln_map::lower(&prog, v, &m, &KilnCost::new()).unwrap();
    let mac: Vec<u32> = v.pool(Pool::Mac).iter().flat_map(|&u| v.units[u].members.iter().map(|&x| v.units[x].compute)).collect();
    g.tasks.iter().flat_map(|t| g.demands_of(t)).map(|a| if let kiln_map::lower::Amount::Res(r, x) = *a { if mac.contains(&r) { x } else { 0.0 } } else { 0.0 }).sum()
}

#[test]
fn operand_residency_reaches_the_unit_cost_model() {
    // HBM so slow that refilling from it would dominate; operands homed in vmem never touch it.
    let v = variant("tpu_v5e.json5", "hbm_pin: \"3.2Gbps\"", "hbm_pin: \"0.01Gbps\"");
    let (vmem, hbm) = (v.shared_onchip().unwrap(), v.offchip.unwrap());
    let (on_chip, off_chip) = (mac_cycles_homed(&v, vmem), mac_cycles_homed(&v, hbm));
    assert!(on_chip < 0.5 * off_chip, "{on_chip} cycles from vmem vs {off_chip} from HBM");
    let fast = view("tpu_v5e.json5");
    assert_eq!(on_chip, mac_cycles_homed(&fast, fast.shared_onchip().unwrap()), "HBM speed changed the cost of vmem-resident operands");
}

#[test]
fn vector_work_moves_its_bytes_through_the_vector_feed() {
    use kiln_map::mapping::SplitAxis;
    let v = view("a100_sxm4_40gb.json5");
    let prog = Program::bench_op(&kiln_ir::bench::BenchOp::gemm(64, 64, 4096, true)).unwrap();
    let (m, _) = kiln_map::heuristic(&prog, &v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).unwrap();
    let op = &prog.ops[0];
    let units = v.pool(Pool::Mac);
    let p = placement(op, 0, vec![SplitAxis { dim: "k".into(), parts: vec![2048, 2048] }], units.len());
    let kc = KilnCost::new();
    let mut l = Lowerer::new(&prog, &v, &m, &kc).unwrap();
    l.begin_group(String::new(), &single_group(&op.id), None);
    l.lower_op(0, &p, &units, true).unwrap();
    let vec_units: Vec<usize> = v.pool(Pool::Vector).iter().flat_map(|&x| v.units[x].members.clone()).collect();
    let feeds: Vec<u32> = vec_units.iter().flat_map(|&x| v.units[x].feed_in.iter().chain(&v.units[x].feed_out).map(|f| f.1)).collect();
    let bytes = |kind: kiln_map::lower::TaskKind| -> f64 {
        l.g.tasks.iter().filter(|t| t.kind == kind).flat_map(|t| l.g.demands_of(t)).map(|a| if let kiln_map::lower::Amount::Res(r, x) = *a { if feeds.contains(&r) { x } else { 0.0 } } else { 0.0 }).sum()
    };
    // The combine reads two fp32 partials and writes one per output element.
    assert!(bytes(kiln_map::lower::TaskKind::Reduce) >= 64.0 * 64.0 * 4.0 * 3.0, "{}", bytes(kiln_map::lower::TaskKind::Reduce));
    // The down-conversion and any scale passes read accumulators and write results on the vector unit.
    assert!(bytes(kiln_map::lower::TaskKind::Compute) >= 64.0 * 64.0 * (4.0 + 2.0), "{}", bytes(kiln_map::lower::TaskKind::Compute));
}

#[test]
fn vector_units_sharing_a_feed_still_gang_per_sm() {
    // Two ALUs per SMSP register file: an SM's eight ALUs are one mappable unit, as its tensor cores are.
    let v = variant("a100_sxm4_40gb.json5", "{ id: \"alu\", kind: \"vector\", lanes: 16,", "{ id: \"alu\", kind: \"vector\", count: 2, lanes: 16,");
    let pool = v.pool(Pool::Vector);
    assert_eq!(pool.len(), 108);
    assert_eq!(v.units[pool[0]].members.len(), 8);
}
