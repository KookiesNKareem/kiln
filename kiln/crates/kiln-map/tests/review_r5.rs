//! Regressions from the r5 cost/map review: work and storage reach the units and memories that hold them.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::heuristic::placement;
use kiln_map::lower::{Amount, Lowerer};
use kiln_map::{HwView, KilnCost, NestQuery, Pool, Program, UnitCostModel};

fn design(name: &str) -> String {
    std::fs::read_to_string(format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"))).expect("design")
}

/// The reference design with each `(from, to)` replaced in its source.
fn variant(name: &str, edits: &[(&str, &str)]) -> HwView {
    let mut src = design(name);
    for (from, to) in edits {
        assert!(src.contains(from), "{from}");
        src = src.replace(from, to);
    }
    let dir = std::env::temp_dir().join(format!("kiln-map-r5-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = dir.join(format!("{}-{name}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&p, src).expect("write");
    let r = check_file(&p, Profile::Reference);
    HwView::new(Arc::new(r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics)))).expect("view")
}

fn program(workload: &str) -> Program {
    let m = kiln_wl::zoo::workload(workload).expect("workload");
    let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
    Program::whole_step(m.model(), &lg, 1).expect("program")
}

/// Cost of op `oi` as one slice on the first vector unit.
fn vector_cost(v: &HwView, prog: &Program, oi: usize) -> kiln_map::NestCost {
    let op = &prog.ops[oi];
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let u = &v.units[v.pool(Pool::Vector)[0]];
    let q = NestQuery { prog, op, slice: &s, points: kiln_map::geom::slice_points(op, &s), hw: &v.hw, unit: u.unit, level_caps: &[], level_mems: &[], level_bw: &[], residency: &[], gang: u.members.len() as u32, quick: true, partial: false };
    KilnCost::new().cost(&q).expect("cost")
}

const V5E_VPU: &str = "{ id: \"vpu\", kind: \"vector\", lanes: 128, sublanes: 8,";
const V5E_EUP: &str = "{ id: \"eup\", kind: \"special\", lanes: 128,";
const V5E_CLOCKS: &str = "clocks: [ { id: \"core\", freq: \"=clk\" } ],";

#[test]
fn vector_slices_pay_pipeline_drain_and_issue_overhead() {
    let base = variant("tpu_v5e.json5", &[]);
    let piped = variant("tpu_v5e.json5", &[(V5E_VPU, &format!("{V5E_VPU} pipeline: {{ drain: 100, issue_overhead: 2 }},"))]);
    let prog = program("llama3_8b:decode_b1");
    let oi = prog.ops.iter().position(|o| o.class() == KernelClass::Map).expect("a map kernel");
    let (a, b) = (vector_cost(&base, &prog, oi), vector_cost(&piped, &prog, oi));
    assert_eq!(b.fill_cycles, a.fill_cycles + 100.0, "the drain is charged once per slice");
    // At least one operation per point, 1024 lanes (128 x 8 sublanes) per instruction.
    let insts = (prog.ops[oi].points() as f64 / 1024.0).ceil();
    assert!(b.cycles >= a.cycles + 2.0 * insts, "issue overhead per instruction: {} vs {} ({insts} instructions)", b.cycles, a.cycles);
}

/// Busy demand per compute resource named `suffix` over every compute task of the decode step on `v`.
fn special_demand(v: &HwView, prog: &Program, suffix: &str) -> f64 {
    let (_, _, g) = kiln_map::heuristic::heuristic_lowered(prog, v, &KilnCost::new(), &kiln_map::heuristic::MapOptions::default()).expect("lowers");
    let res: Vec<u32> = (0..v.resources.len() as u32).filter(|&r| v.resources[r as usize].path.ends_with(suffix)).collect();
    assert!(!res.is_empty(), "{suffix}");
    g.tasks
        .iter()
        .flat_map(|t| g.demands_of(t))
        .map(|a| match *a {
            Amount::Res(r, x) if res.contains(&r) => x,
            _ => 0.0,
        })
        .sum()
}

#[test]
fn special_units_are_charged_only_the_functions_they_implement() {
    let prog = program("llama3_8b:decode_b1");
    let oi = prog.ops.iter().position(|o| o.class() != KernelClass::Contraction && o.body().exp > 0 && o.body().log == 0).expect("an exp");
    let base = variant("tpu_v5e.json5", &[]);
    // A log-only unit on a 1 MHz clock next to the EUP: softmax has no log for it.
    let lg = format!("{{ id: \"lg\", kind: \"special\", lanes: 128, functions: [\"log\"], precisions: [\"fp32@1\"], feeds: {{ any: \"vreg\" }}, clock: \"slow\" }},\n        {V5E_EUP}");
    let slow_log = variant("tpu_v5e.json5", &[(V5E_EUP, &lg), (V5E_CLOCKS, "clocks: [ { id: \"core\", freq: \"=clk\" }, { id: \"slow\", freq: \"1MHz\" } ],")]);
    let (a, b) = (vector_cost(&base, &prog, oi), vector_cost(&slow_log, &prog, oi));
    assert_eq!(a.special_cycles.len(), 1);
    assert_eq!(b.special_cycles.len(), 2);
    let eup = kiln_map::cost::special_units(&slow_log.hw, slow_log.units[slow_log.pool(Pool::Vector)[0]].unit)
        .iter()
        .position(|&u| slow_log.hw.nodes[slow_log.hw.units[u].node].path.ends_with("eup"))
        .expect("eup");
    assert_eq!(b.special_cycles[1 - eup], 0.0, "the log unit runs no exp");
    assert_eq!((b.cycles, b.special_cycles[eup]), (a.cycles, a.special_cycles[0]));
    assert_eq!(special_demand(&slow_log, &prog, "lg"), 0.0);
    assert_eq!(special_demand(&slow_log, &prog, "eup"), special_demand(&base, &prog, "eup"));
}

/// One tile: a 16-lane vector unit and an 8x8 systolic array on a `sram_kib` scratchpad, over one HBM stack.
fn tile_design(sram_kib: u32) -> HwView {
    let doc = serde_json::json!({
        "schema": "kiln.hw/1.0", "name": "r5", "tech": "tsmc_n5",
        "clocks": [ { "id": "clk", "freq": 1e9 } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "clk",
                "clusters": [ { "id": "tile",
                    "units": [
                        { "id": "mxu", "kind": "matrix", "geometry": { "systolic": { "rows": 8, "cols": 8 } },
                          "precisions": ["bf16*bf16+fp32"], "feeds": { "a": "sram", "b": "sram", "o": "sram" } },
                        { "id": "vpu", "kind": "vector", "lanes": 16, "precisions": ["fp32@1"], "feeds": { "any": "sram" } } ],
                    "memories": [ { "id": "sram", "kind": "scratchpad", "capacity": sram_kib * 1024, "banks": 4,
                                    "ports": [ { "dir": "rw", "width_bits": 256 } ] } ] } ],
                "networks": [ { "id": "noc", "topology": "crossbar", "endpoints": ["tile.sram"], "link": "256b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "2Gbps", "attach": "die.noc" } ] } }
    });
    let d = kiln_ir::hw::Design::from_source(&kiln_ir::hw::MemLoader::default(), None, &doc.to_string()).expect("design parses");
    let r = kiln_ir::hw::check(d, Profile::Full, &kiln_ir::hw::ExpandOptions::default());
    HwView::new(Arc::new(r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics)))).expect("view")
}

/// `a = relu(x)`, `b = silu(x)` (128 KiB each), then `consumer` reads both, all in one barrier-free span with `a`
/// and `b` private; returns the bytes written into HBM.
fn private_span_hbm_bytes(v: &HwView, consumer: serde_json::Value) -> f64 {
    use kiln_map::mapping::{ExecGroup, GroupKind, LaunchKind, Lifetime};
    let out = consumer["outputs"][0].as_str().unwrap().to_string();
    let shape = if consumer["op"] == "einsum" { serde_json::json!([1, 1]) } else { serde_json::json!([1, 65536]) };
    let model: kiln_ir::wl::Model = serde_json::from_value(serde_json::json!({
        "symbols": {}, "entry": {"forward": "main"}, "tensors": {},
        "graphs": { "main": { "params": ["x"], "results": [out], "tensors": {
            "x": {"shape": [1, 65536], "dtype": "bf16", "class": "input"},
            "a": {"shape": [1, 65536], "dtype": "bf16", "class": "activation"},
            "b": {"shape": [1, 65536], "dtype": "bf16", "class": "activation"},
            out.clone(): {"shape": shape, "dtype": "bf16", "class": "output"},
        }, "nodes": [
            {"id": "pa", "op": "map", "fn": "relu", "inputs": ["x"], "outputs": ["a"]},
            {"id": "pb", "op": "map", "fn": "silu", "inputs": ["x"], "outputs": ["b"]},
            consumer,
        ]}},
    }))
    .expect("model");
    let sc = kiln_wl::zoo::whole_step(kiln_ir::wl::PhaseKind::Decode, kiln_ir::wl::SeqBatch::uniform(1, 1, 1));
    let (_, lg, _) = kiln_wl::evaluate_snapshot(&model, &sc).expect("lowers");
    let prog = Program::whole_step(&model, &lg, 1).expect("program");
    let (mut m, _) = kiln_map::heuristic(&prog, v, &kiln_map::RooflineCost, &kiln_map::heuristic::MapOptions::default()).expect("maps");
    for t in ["a", "b"] {
        let tp = m.tensors.get_mut(t).expect("placed");
        tp.home.clear();
        tp.lifetime = Lifetime::Private;
    }
    let ops: Vec<String> = m.groups.iter().flat_map(|g| g.ops.clone()).collect();
    m.groups = vec![ExecGroup { ops, kind: GroupKind::Fused { on_chip: vec!["a".into(), "b".into()] }, launch: LaunchKind::HostLaunch, barrier_after: true }];
    assert!(m.validate(&prog, v).is_empty(), "{:?}", m.validate(&prog, v));
    let g = kiln_map::lower(&prog, v, &m, &kiln_map::RooflineCost).expect("lowers");
    let hbm = v.offchip.expect("hbm");
    g.ops.iter().filter_map(|o| o.level_bytes.get(&hbm)).sum()
}

#[test]
fn private_intermediates_that_do_not_fit_spill_at_a_price() {
    let dot = || serde_json::json!({"id": "c", "op": "einsum", "eq": "mi,ni->mn", "inputs": ["a", "b"], "outputs": ["y"]});
    let add = || serde_json::json!({"id": "c", "op": "map", "fn": "add", "inputs": ["a", "b"], "outputs": ["y"]});
    let (small, large) = (tile_design(64), tile_design(1024));
    let fits = private_span_hbm_bytes(&large, dot());
    let spills = private_span_hbm_bytes(&small, dot());
    // The dot product reduces over every element of a and b: nothing streams, 256 KiB cannot stay in 64 KiB.
    assert!(spills >= fits + 2.0 * 131072.0, "a and b are written to HBM: {spills} vs {fits}");
    // An elementwise consumer streams a and b through tiles as they are produced: nothing spills.
    assert_eq!(private_span_hbm_bytes(&small, add()), private_span_hbm_bytes(&large, add()));
}
