//! Software-stack recipes (08 §F): modelled kernels per layer match the A100 profiler counts of the measured
//! PyTorch CUDA-graph SDPA step; recipes without rules add no kernels, XLA's fuses elementwise nodes.

mod common;

use std::sync::Arc;

use common::*;
use kiln_sim::SimOptions;
use kiln_trace::IntervalMethod;
use kiln_wl::stack::Stack;

fn opts(stack: Option<&str>) -> SimOptions {
    SimOptions {
        interval: IntervalMethod::None,
        shadow_prices: false,
        layer_scope_fallback: true,
        stack: stack.map(|s| Arc::new(Stack::load(s).unwrap())),
        ..SimOptions::default()
    }
}

#[test]
fn pytorch_kernel_counts_match_the_a100_profiler() {
    let p = reference("a100_sxm4_40gb.json5");
    // calibration/measurements/a100_2026-10-05_seq_r2.json, step_sdpa: (n_kernels - 11) / 32 per layer, 11 outside.
    for (phase, per_layer, step) in [("decode_b1", 45, 1451), ("decode_b8", 45, 1451), ("decode_b32", 43, 1387), ("prefill_b1", 43, 1387)] {
        let m = kiln_wl::zoo::workload(&format!("llama3_8b:{phase}")).unwrap();
        let (run, prog) = kiln_sim::simulate_member(&p, &m, &opts(None)).unwrap();
        assert!(run.central.provenance.flags["stack"].starts_with("pytorch_cuda_graph_sdpa@"));
        let (w, l) = prog.window.unwrap();
        let mid = kiln_sim::stack::kernel_counts(&run.graph, Some(w / 2));
        assert_eq!(mid.0 + mid.1, per_layer, "{phase}: {mid:?} kernels per layer");
        let pe = kiln_sim::stack::kernel_counts(&run.graph, None);
        let total = (pe.0 + pe.1) as u64 + l * per_layer as u64;
        // Outside the layers PyTorch runs embedding, 8 norm kernels, lm_head and argmax; kiln's logits_select
        // node (last token per sequence) is a view there, so kiln launches one kernel more per step.
        assert_eq!(total, step + 1, "{phase}: prologue/epilogue {pe:?}");
    }
}

#[test]
fn xla_fusion_beats_node_granularity_on_tpus_and_ideal_gpus_are_faster() {
    // Neither recipe has rules and both fuse elementwise nodes into the adjacent contraction (08 §F: the ideal
    // stack is at least as good as the best real one); without fusion a TPU step is slower.
    for d in ["tpu_v5e.json5", "tpu_v6e.json5"] {
        let p = reference(d);
        for ph in ["decode_b8", "prefill_b1"] {
            let m = kiln_wl::zoo::workload(&format!("llama3_8b:{ph}")).unwrap();
            let (xla, _) = kiln_sim::simulate_member(&p, &m, &opts(None)).unwrap();
            let (ideal, _) = kiln_sim::simulate_member(&p, &m, &opts(Some("kiln_ideal"))).unwrap();
            let mut unfused = Stack::load("kiln_ideal").unwrap();
            unfused.fuse_elementwise = false;
            let mut o = opts(None);
            o.stack = Some(Arc::new(unfused));
            let (node, _) = kiln_sim::simulate_member(&p, &m, &o).unwrap();
            assert!(xla.graph.stack.is_empty() && ideal.graph.stack.is_empty());
            assert_eq!(xla.graph.groups.len(), ideal.graph.groups.len(), "{d} {ph}");
            assert!(xla.graph.groups.len() < node.graph.groups.len(), "{d} {ph}");
            assert!(ideal.central.makespan_s <= xla.central.makespan_s * (1.0 + 1e-9), "{d} {ph}");
            assert!(xla.central.makespan_s < node.central.makespan_s, "{d} {ph}");
        }
    }
    let p = reference("a100_sxm4_40gb.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b1").unwrap();
    let (torch, _) = kiln_sim::simulate_member(&p, &m, &opts(None)).unwrap();
    let (ideal, _) = kiln_sim::simulate_member(&p, &m, &opts(Some("kiln_ideal"))).unwrap();
    assert!(torch.central.makespan_s > ideal.central.makespan_s * 1.05);
    assert!(ideal.graph.stack.is_empty() && !torch.graph.stack.is_empty());
    for r in [&torch.central, &ideal.central] {
        assert!(r.invariants.checks.iter().all(|c| c.status != kiln_trace::sim::CheckStatus::Fail), "{:?}", r.invariants);
    }
}

#[test]
fn unused_cache_slots_do_not_change_the_recipe() {
    let p = reference("a100_sxm4_40gb.json5");
    let (model, sc) = tiny(false, 1);
    let kernels = |slots: u64| {
        let mut inst = kiln_wl::expand::expand(&sc, &model).unwrap().remove(0);
        inst.bindings.insert("slots".into(), slots);
        let (_, lg, _) = kiln_wl::evaluate_instance(&model, &sc, &inst).unwrap();
        let prog = kiln_map::Program::whole_step(&model, &lg, 3).unwrap();
        let r = run(&p, &prog, &opts(Some("pytorch_cuda_graph_sdpa")));
        let mut names: Vec<String> = r.graph.stack.iter().map(|k| k.name.clone()).collect();
        names.sort();
        names
    };
    let one = kernels(1);
    assert!(one.iter().any(|k| k == "memset"), "{one:?}");
    assert_eq!(kernels(64), one);
}

#[test]
fn library_tiles_pad_issued_macs_under_the_realistic_stack_only() {
    let p = reference("a100_sxm4_40gb.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b32").unwrap();
    let issue = |stack: &str| {
        let o = SimOptions { trace: kiln_trace::TraceLevel::Ops, ..opts(Some(stack)) };
        let (r, _) = kiln_sim::simulate_member(&p, &m, &o).unwrap();
        let ratio = |part: &str| {
            let (u, i) = r.central.ops.iter().filter(|o| o.op.as_str().contains(part) && o.macs_useful > 0).fold((0u64, 0u64), |a, o| (a.0 + o.macs_useful, a.1 + o.macs_issued));
            i as f64 / u as f64
        };
        (ratio(".attn."), ratio(".qkv."), r.central.energy.padding_j)
    };
    let (torch, ideal) = (issue("pytorch_cuda_graph_sdpa"), issue("kiln_ideal"));
    // flash_fwd's 128-row query tiles hold the 4 query heads of a KV head; cuBLAS tiles 32 tokens 64 wide.
    assert!((torch.0 - 32.0).abs() < 1e-6 && (torch.1 - 2.0).abs() < 1e-6, "{torch:?}");
    assert!(ideal.0 < 32.0 && ideal.1 <= 2.0, "kiln_ideal issues on the array's own granule only: {ideal:?}");
    assert!(torch.2 > ideal.2, "padded MACs cost energy: {torch:?} vs {ideal:?}");
}
