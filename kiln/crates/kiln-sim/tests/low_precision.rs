//! Low-precision weights and KV cache end to end: kiln-wl's explicit converts, fused by the mapper into the
//! contraction that reads them; native mixed and MX modes used as declared; no silent upcast.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use kiln_ir::hw::{Design, Profile};
use kiln_sim::{PhaseRun, Prepared, SimOptions};
use kiln_trace::IntervalMethod;
use kiln_trace::sim::ResourceKind;
use serde_json::Value;

fn opts(stack: kiln_wl::stack::Stack) -> SimOptions {
    SimOptions { interval: IntervalMethod::None, shadow_prices: false, stack: Some(Arc::new(stack)), ..SimOptions::default() }
}

fn ideal() -> kiln_wl::stack::Stack {
    kiln_wl::stack::Stack::load("kiln_ideal").unwrap()
}

fn sim(p: &Prepared, w: &str, o: &SimOptions) -> (PhaseRun, kiln_map::Program) {
    kiln_sim::simulate_member(p, &kiln_wl::zoo::workload(w).unwrap(), o).unwrap_or_else(|e| panic!("{w}: {e:?}"))
}

fn dram_bytes(r: &PhaseRun) -> f64 {
    r.central.resources.iter().filter(|x| x.kind == ResourceKind::DramChannel).map(|x| x.bytes).sum()
}

fn converts(prog: &kiln_map::Program) -> Vec<usize> {
    (0..prog.ops.len()).filter(|&i| prog.ops[i].id.contains("cvt")).collect()
}

#[test]
fn fp8_weights_halve_decode_weight_traffic_through_fused_converts() {
    let p = reference("a100_sxm4_40gb.json5");
    let o = opts(ideal());
    let (b, bp) = sim(&p, "llama3_8b:decode_b1", &o);
    let (f, fp) = sim(&p, "llama3_8b:decode_b1+weights=fp8_e4m3", &o);
    assert!(converts(&bp).is_empty());
    let cv = converts(&fp);
    assert_eq!(cv.len(), 3 * 4 + 1, "qkv, o_proj, gate_up, down per window layer, and lm_head");
    for &c in &cv {
        assert!(fp.fused_into(c).is_some() && !fp.placed(c), "{}", fp.ops[c].id);
        assert!(!f.mapping.ops.contains_key(&fp.ops[c].id));
    }
    assert_eq!(f.graph.ops.len(), b.graph.ops.len(), "fused converts lower no tasks of their own");
    let (db, df) = (dram_bytes(&b), dram_bytes(&f));
    assert!(df < 0.55 * db && df > 0.45 * db, "DRAM bytes {df:.3e} vs {db:.3e}");
    assert!(f.central.makespan_s < 0.6 * b.central.makespan_s);
    // The converts are executed vector work: every weight element once at batch 1.
    let vec = |r: &PhaseRun| r.graph.ops.iter().map(|o| o.vec_ops).sum::<u128>();
    assert!(vec(&f) > vec(&b) + 3 * 218_103_808, "{} vs {}", vec(&f), vec(&b));
}

#[test]
fn fp8_kv_cache_is_converted_for_attention_only() {
    let p = reference("tpu_v5e.json5");
    let o = opts(ideal());
    let (_, prog) = sim(&p, "llama3_8b:decode_b8+kv=fp8_e4m3", &o);
    let attn: Vec<&str> = converts(&prog).iter().map(|&c| prog.ops[c].id.as_str()).collect();
    assert_eq!(attn.len(), 3 * 2, "K and V per window layer: {attn:?}");
    assert!(attn.iter().all(|id| id.contains(".attn.")));
}

/// H100 with one more tensor-core mode, `bf16 x fp8_e4m3` at the fp8 rate.
fn h100_mixed() -> Prepared {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/h100_sxm5_80gb.json5");
    let mut v = kiln_ir::hw::load_file(&path).unwrap().canonical;
    fn add(v: &mut Value) -> bool {
        match v {
            Value::Object(m) if m.get("id") == Some(&Value::from("tc")) => {
                m["precisions"].as_array_mut().unwrap().push(Value::from("bf16*fp8_e4m3+fp32@0.5"));
                true
            }
            Value::Object(m) => m.values_mut().any(add),
            Value::Array(xs) => xs.iter_mut().any(add),
            _ => false,
        }
    }
    assert!(add(&mut v));
    Prepared::load(Design::from_value(v).unwrap(), Profile::Full).unwrap()
}

#[test]
fn declared_mixed_and_mx_modes_run_natively() {
    let w = "llama3_8b:decode_b8+weights=fp8_e4m3";
    let p = h100_mixed();
    let (native, prog) = sim(&p, w, &opts(ideal()));
    assert!(converts(&prog).is_empty(), "bf16 x fp8 is declared");
    assert!(native.graph.ops.iter().any(|o| o.mode.starts_with("bf16*fp8_e4m3")), "{:?}", native.graph.ops.iter().map(|o| &o.mode).collect::<Vec<_>>());
    // A stack without mixed-input kernels dequantizes anyway.
    let mut deq = ideal();
    deq.dequantize = true;
    let (_, prog) = sim(&p, w, &opts(deq));
    assert_eq!(converts(&prog).len(), 3 * 4 + 1);
    // ember declares bf16 x mxfp4: MX weights stream with their E8M0 scales and run in that mode.
    let e = reference("ember.json5");
    let (mx, prog) = sim(&e, "llama3_8b:decode_b8+weights=mxfp4", &opts(ideal()));
    assert!(converts(&prog).is_empty());
    assert!(mx.graph.ops.iter().any(|o| o.mode.contains("mxfp4")));
}

#[test]
fn no_mode_and_no_lossless_widening_is_a_mapping_error() {
    // V100 has no bf16 MACs and fp16 cannot hold bf16: bf16 activations stay unmappable (no silent narrowing).
    let p = reference("v100_sxm2_32gb.json5");
    let e = kiln_sim::simulate_member(&p, &kiln_wl::zoo::workload("llama3_8b:decode_b1").unwrap(), &opts(ideal())).unwrap_err();
    assert_eq!(e[0].code, "E-MAP-OP-004");
}

#[test]
fn fp8_activations_and_weights_run_in_the_fp8_tensor_core_mode() {
    // W8A8 (`+acts=` quantizes every activation, so every GEMM input) on H100: fp8 x fp8 at twice the bf16 rate.
    let p = reference("h100_sxm5_80gb.json5");
    let o = opts(ideal());
    let (b, _) = sim(&p, "llama3_8b:prefill_b1", &o);
    let (f, prog) = sim(&p, "llama3_8b:prefill_b1+weights=fp8_e4m3+acts=fp8_e4m3", &o);
    let gemms: Vec<&str> = f.graph.ops.iter().filter(|s| s.useful_macs > 0 && !prog.ops[s.op].id.contains(".attn.")).map(|s| s.mode.as_str()).collect();
    assert!(!gemms.is_empty() && gemms.iter().all(|m| m.starts_with("fp8_e4m3*fp8_e4m3")), "{gemms:?}");
    // The bf16 KV cache keeps attention in bf16: fp8 queries and probabilities are converted for it.
    assert!(converts(&prog).iter().all(|&c| prog.ops[c].id.contains(".attn.")));
    assert!(f.central.makespan_s * 1.3 < b.central.makespan_s, "{} vs {}", f.central.makespan_s, b.central.makespan_s);
}
