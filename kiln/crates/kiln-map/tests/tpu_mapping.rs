//! Mapping on TPU reference designs: stack-level elementwise fusion (03 §3.6), balanced unit sets over mixed
//! pools, coalesced contracting dims on wide systolic arrays.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::heuristic::{MapOptions, heuristic_lowered};
use kiln_map::{HwView, KilnCost, Program};

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
fn fused_elementwise_nodes_join_one_contraction_group_each() {
    let v = view("tpu_v5e.json5");
    let prog = program("llama3_8b:prefill_b1");
    let map = |fuse: bool| heuristic_lowered(&prog, &v, &KilnCost::new(), &MapOptions { fuse_elementwise: fuse, ..MapOptions::default() }).expect("maps").0;
    let (plain, fused) = (map(false), map(true));
    assert_eq!(plain.groups.len(), prog.nodes.len());
    let layer: Vec<_> = fused.groups.iter().filter(|g| prog.iteration_of_op(prog.op(&g.ops[0]).unwrap()) == Some(0)).collect();
    // qkv (with the norm before it and RoPE / KV append after it), attention, o_proj, gate_up, down.
    assert_eq!(layer.len(), 5, "{:?}", layer.iter().map(|g| &g.ops).collect::<Vec<_>>());
    for g in &layer {
        assert!(g.barrier_after);
        assert!(g.ops.iter().any(|o| prog.ops[prog.op(o).unwrap()].class() == KernelClass::Contraction), "{:?}", g.ops);
    }
    let ops = |m: &kiln_map::mapping::Mapping| m.groups.iter().flat_map(|g| g.ops.clone()).collect::<Vec<_>>();
    assert_eq!(ops(&plain), ops(&fused));
}

#[test]
fn mixed_vector_pools_keep_the_units_that_balance() {
    // v6e: the TensorCore VPU next to 32 narrow SparseCore tiles; equal round-robin slices would run every
    // vector op at a tile's pace, so the vector set is the VPU alone.
    let v = view("tpu_v6e.json5");
    let prog = program("llama3_8b:decode_b8");
    let (m, _, _) = heuristic_lowered(&prog, &v, &KilnCost::new(), &MapOptions::default()).expect("maps");
    for s in m.unit_sets.iter().filter(|s| s.name.starts_with("vector")) {
        assert_eq!(s.units.len(), 1, "{}: {:?}", s.name, s.units);
        assert!(s.units[0].ends_with("tc.vpu"), "{:?}", s.units);
    }
    // A homogeneous pool stays whole.
    let a = view("a100_sxm4_40gb.json5");
    let (m, _, _) = heuristic_lowered(&prog, &a, &KilnCost::new(), &MapOptions::default()).expect("maps");
    assert_eq!(m.unit_sets.iter().find(|s| s.name == "vector").unwrap().units.len(), a.pool(kiln_map::Pool::Vector).len());
}

#[test]
fn contracting_head_dims_fill_a_256_row_array() {
    // o_proj contracts over (heads, head_dim = 128); on v6e's 256x256 MXUs both dims share the rows.
    let v = view("tpu_v6e.json5");
    let prog = program("llama3_8b:prefill_b1");
    let (_, _, g) = heuristic_lowered(&prog, &v, &KilnCost::new(), &MapOptions::default()).expect("maps");
    let o = g.ops.iter().find(|o| prog.ops[o.op].id.contains("o_proj")).expect("o_proj");
    assert_eq!(o.issued_macs, o.useful_macs);
}
