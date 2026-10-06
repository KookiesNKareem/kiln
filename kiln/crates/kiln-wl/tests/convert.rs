//! Explicit converts before contractions (02 §5.7, 03 §2.7) and the `+weights=` / `+kv=` workload suffixes.

use kiln_ir::precision::Precision::{self, *};
use kiln_ir::wl::{ElemType, KernelClass, TensorClass};
use kiln_wl::convert::{MacMode, MacModes, choose, choose_with, insert_converts, widens};

fn t(p: Precision) -> ElemType {
    ElemType::from(p)
}

/// Modes with the usual accumulators: fp32 for float pairs, int32 for integer ones.
fn modes(m: &[(Precision, Precision, f64)], dequantize: bool) -> MacModes {
    let acc = |a: Precision| if a.is_integer() { Int32 } else { Fp32 };
    MacModes { modes: m.iter().map(|&(a, b, rate)| MacMode { a, b, acc: acc(a), rate }).collect(), dequantize }
}

#[test]
fn widenings_are_lossless() {
    for p in [Fp8E4m3, Fp8E5m2, Fp6E2m3, Fp4E2m1, Int8, Uint8, Int4, Mxfp4, Mxfp8E4m3, Mxint8, Nvfp4, Fp8E4m3Pt, Int8Pc, Int4G128] {
        assert!(widens(Bf16, &t(p)) && widens(Fp32, &t(p)), "{p:?} into bf16/fp32");
    }
    assert!(widens(Fp16, &t(Fp8E4m3)) && widens(Fp16, &t(Fp8E5m2)) && widens(Fp16, &t(Int8)));
    // MX and scaled values exceed fp16's range; bf16 does not fit fp16's or fp8's significand.
    for (to, from) in [(Fp16, Mxfp4), (Fp16, Fp8E4m3Pt), (Fp16, Bf16), (Fp8E4m3, Bf16), (Fp8E5m2, Fp8E4m3), (Int8, Fp8E4m3), (Fp8E4m3, Fp8E4m3Pt)] {
        assert!(!widens(to, &t(from)), "{from:?} into {to:?}");
    }
    assert!(widens(Mxfp8E4m3, &t(Mxfp4)) && widens(Int8, &t(Int4)) && widens(Fp8E4m3, &t(Fp4E2m1)));
    assert!(!widens(Mxfp8E4m3, &t(Fp8E4m3)), "a plain fp8 tensor has no block scales to reuse");
}

#[test]
fn modes_are_native_or_the_fastest_widening() {
    let a100 = modes(&[(Bf16, Bf16, 1.0), (Fp16, Fp16, 1.0), (Int8, Int8, 2.0), (Int4, Int4, 4.0)], false);
    assert_eq!(choose(&a100, &t(Bf16), &t(Bf16)), Some((None, None)));
    assert_eq!(choose(&a100, &t(Bf16), &t(Fp8E4m3)), Some((None, Some(Bf16))));
    assert_eq!(choose(&a100, &t(Fp8E4m3), &t(Fp8E4m3)), Some((Some(Bf16), Some(Bf16))));
    assert_eq!(choose(&a100, &t(Int8), &t(Int4)), Some((None, Some(Int8))), "int8 x int8 beats a bf16 widening");
    assert_eq!(choose(&a100, &t(Bf16), &t(Fp32)), None, "never a narrowing");
    // A declared mixed mode runs the pair as is, unless the software stack has no mixed-input kernels.
    let mixed = modes(&[(Bf16, Bf16, 1.0), (Fp8E4m3, Fp8E4m3, 2.0), (Bf16, Fp8E4m3, 1.0)], false);
    assert_eq!(choose(&mixed, &t(Bf16), &t(Fp8E4m3)), Some((None, None)));
    let deq = MacModes { dequantize: true, ..mixed };
    assert_eq!(choose(&deq, &t(Bf16), &t(Fp8E4m3)), Some((None, Some(Bf16))));
    assert_eq!(choose(&deq, &t(Fp8E4m3), &t(Fp8E4m3)), Some((None, None)));
    // MX operands run on their element type's mode (scales cost a vector pass in kiln-cost), or a mixed MX mode.
    let ember = modes(&[(Bf16, Bf16, 0.5), (Mxfp4, Mxfp4, 4.0), (Bf16, Mxfp4, 0.5)], false);
    assert_eq!(choose(&ember, &t(Bf16), &t(Mxfp4)), Some((None, None)));
    assert_eq!(choose(&modes(&[(Fp4E2m1, Fp4E2m1, 4.0)], false), &t(Mxfp4), &t(Mxfp4)), Some((None, None)));
}

fn lowered(name: &str) -> (kiln_wl::LoweredGraph, kiln_wl::StepStats) {
    let m = kiln_wl::zoo::workload(name).unwrap();
    let (_, lg, st) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).unwrap();
    (lg, st)
}

#[test]
fn weight_and_kv_suffixes_set_storage_dtypes() {
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b8+weights=fp8_e4m3+kv=fp8_e4m3_pt").unwrap();
    let model = m.model();
    let w8a8 = kiln_wl::zoo::workload("llama3_8b:prefill_b1+acts=fp8_e4m3").unwrap();
    let acts = |m: &kiln_ir::wl::Model| m.graphs.values().flat_map(|g| g.tensors.values()).filter(|x| x.class == TensorClass::Activation).map(|x| x.dtype).collect::<Vec<_>>();
    assert!(acts(w8a8.model()).contains(&t(Fp8E4m3)) && !acts(w8a8.model()).contains(&t(Bf16)), "{:?}", acts(w8a8.model()));
    let dtypes = |c: TensorClass| model.tensors.values().filter(|x| x.class == c).map(|x| x.dtype).collect::<Vec<_>>();
    assert!(dtypes(TensorClass::Weight).iter().all(|d| *d == t(Fp8E4m3)));
    assert!(dtypes(TensorClass::KvCache).iter().all(|d| *d == t(Fp8E4m3Pt)));
    assert_eq!(m.name, "llama3_8b:decode_b8+weights=fp8_e4m3+kv=fp8_e4m3_pt");
    for bad in ["llama3_8b:decode_b8+weights=fp9", "llama3_8b:decode_b8+act=fp8_e4m3", "llama3_8b:decode_b8+weights"] {
        assert_eq!(kiln_wl::zoo::workload(bad).unwrap_err().code, "E-WL-SCN-001", "{bad}");
    }
    let ((_, b), (_, f)) = (lowered("llama3_8b:decode_b1"), lowered("llama3_8b:decode_b1+weights=fp8_e4m3"));
    assert_eq!(f.weight_read * 2, b.weight_read);
}

#[test]
fn converts_feed_exactly_the_contractions_that_need_them() {
    let a100 = modes(&[(Bf16, Bf16, 1.0), (Fp16, Fp16, 1.0), (Int8, Int8, 2.0)], false);
    let (mut base, _) = lowered("llama3_8b:decode_b8");
    assert_eq!(insert_converts(&mut base, &a100).unwrap(), 0, "bf16 needs none");
    let (mut lg, _) = lowered("llama3_8b:decode_b8+weights=fp8_e4m3+kv=fp8_e4m3");
    let n = insert_converts(&mut lg, &a100).unwrap();
    // Per layer qkv, o_proj, gate_up, down, attention K and V; then lm_head.
    assert_eq!(n, 7);
    for node in &lg.nodes {
        let ks = &node.lowered.kernels;
        for (i, k) in ks.iter().enumerate().filter(|(_, k)| k.id.contains("cvt")) {
            assert_eq!(k.class, KernelClass::Map);
            assert_eq!(k.body.cvt, 1);
            let out = &k.operands[1].tensor;
            let temp = node.lowered.temps.iter().find(|(id, _)| id == out).expect("a node temp");
            assert_eq!(temp.1.dtype, t(Bf16));
            let readers: Vec<&kiln_ir::wl::Kernel> = ks.iter().filter(|x| x.operands.iter().any(|o| &o.tensor == out && o.access == kiln_ir::wl::Access::Read)).collect();
            assert_eq!(readers.len(), 1, "{}", k.id);
            assert_eq!(readers[0].class, KernelClass::Contraction);
            assert!(ks[i + 1..].iter().any(|x| std::ptr::eq(x, readers[0])), "the convert precedes its contraction");
            assert!(!node.node.outputs.contains(out) && !node.node.inputs.contains(out));
        }
    }
}

#[test]
fn upcast_ok_false_forbids_widening_converts() {
    let a100 = modes(&[(Bf16, Bf16, 1.0), (Fp16, Fp16, 1.0), (Int8, Int8, 2.0)], false);
    let w = kiln_wl::zoo::workload("llama3_8b:decode_b8+weights=fp8_e4m3").unwrap();
    let mut model = w.model().clone();
    let lower = |m: &kiln_ir::wl::Model| kiln_wl::evaluate_snapshot(m, w.scenario()).unwrap().1;
    let mut lg = lower(&model);
    assert_eq!(insert_converts(&mut lg, &a100).unwrap(), 5, "qkv, o_proj, gate_up, down, lm_head");
    model.tensors.values_mut().filter(|x| x.class == TensorClass::Weight).for_each(|x| x.upcast_ok = false);
    let mut lg = lower(&model);
    assert_eq!(insert_converts(&mut lg, &a100).unwrap(), 0, "fp8 weights may not be widened; the mapper rejects the native op");
}

#[test]
fn scaled_widenings_hold_the_element_and_in_reduction_scales() {
    use kiln_ir::wl::Scaling;
    let pt = |scalar, scale| ElemType { scalar, scaling: Scaling::PerTensor { scale } };
    assert!(!widens(Bf16, &pt(Fp32, Fp32)), "1 + 2^-16 rounds in bf16");
    assert!(!widens(Tf32, &pt(Fp32, Fp32)) && widens(Fp32, &pt(Fp32, Fp32)));
    assert!(!widens(Bf16, &pt(Fp16, Fp32)) && widens(Bf16, &pt(Fp8E4m3, Fp32)), "per-tensor scales apply to the accumulator");
    let blk = |scalar, scale| ElemType { scalar, scaling: Scaling::Block { axis: -1, block: 128, scale, zero_point: None, tensor_scale: None } };
    assert!(widens(Bf16, &blk(Int4, Bf16)) && widens(Bf16, &blk(Int4, E8m0)));
    assert!(!widens(Bf16, &blk(Int4, Fp32)) && widens(Fp32, &blk(Int4, Fp32)), "a block scale is applied inside the reduction");
    let none = modes(&[(Bf16, Bf16, 1.0)], false);
    assert_eq!(choose(&none, &t(Bf16), &pt(Fp32, Fp32)), None);
}

#[test]
fn zoo_accum_config_reaches_layer_contractions() {
    let mut cfg = kiln_wl::zoo::preset("llama3_8b").unwrap();
    cfg.dtypes.accum = Fp16;
    let m = kiln_wl::zoo::build_model(&cfg).unwrap();
    let accums: Vec<_> = m
        .graphs
        .values()
        .flat_map(|g| &g.nodes)
        .filter_map(|n| match &n.op {
            kiln_ir::wl::Op::Einsum(a) => Some((n.id.to_string(), a.accum)),
            _ => None,
        })
        .collect();
    assert!(accums.len() > 3 && accums.iter().all(|(_, a)| *a == Some(Fp16)), "{accums:?}");
}

#[test]
fn integer_widening_respects_signedness() {
    // Codex r2: uint8 255 is not an int8, int8 -1 is not a uint8.
    assert!(!widens(Int8, &t(Uint8)) && !widens(Uint8, &t(Int8)) && !widens(Uint8, &t(Int4)));
    assert!(widens(Int16, &t(Uint8)) && widens(Int8, &t(Uint4)) && widens(Uint8, &t(Uint4)) && widens(Int8, &t(Int4)));
    let int8_only = modes(&[(Int8, Int8, 1.0)], false);
    assert_eq!(choose(&int8_only, &t(Uint8), &t(Uint8)), None);
    assert_eq!(choose(&int8_only, &t(Uint4), &t(Int4)), Some((Some(Int8), Some(Int8))));
}

#[test]
fn unregistered_scaling_is_converted_not_dropped() {
    use kiln_ir::wl::Scaling;
    // fp8 with bf16 scales per 32-element block: no registry name, so no mode may read it directly.
    let raw = ElemType { scalar: Fp8E4m3, scaling: Scaling::Block { axis: -1, block: 32, scale: Bf16, zero_point: None, tensor_scale: None } };
    assert!(raw.shorthand().is_none());
    assert!(!widens(Fp8E4m3, &raw), "dropping the scales is not a widening");
    let fp8_only = modes(&[(Fp8E4m3, Fp8E4m3, 4.0)], false);
    assert_eq!(choose(&fp8_only, &t(Fp8E4m3), &raw), None);
    let with_bf16 = modes(&[(Fp8E4m3, Fp8E4m3, 4.0), (Bf16, Bf16, 1.0)], false);
    assert_eq!(choose(&with_bf16, &t(Bf16), &raw), Some((None, Some(Bf16))));
    // Even at the scalar's own width the convert applies the scales.
    let raw_bf16 = ElemType { scalar: Bf16, ..raw };
    assert_eq!(choose(&with_bf16, &t(Bf16), &raw_bf16), Some((None, Some(Bf16))));
}

#[test]
fn widening_respects_the_required_accumulator() {
    let acc = |a, b, acc, rate| MacMode { a, b, acc, rate };
    // A fast fp8 mode accumulating in fp16 cannot run an fp32-accumulated contraction; bf16 (fp32 acc) can.
    let m = MacModes { modes: vec![acc(Fp8E4m3, Fp8E4m3, Fp16, 4.0), acc(Bf16, Bf16, Fp32, 1.0)], dequantize: false };
    assert_eq!(choose_with(&m, &t(Fp8E4m3), &t(Fp8E4m3), None, [true, true]), Some((None, None)));
    assert_eq!(choose_with(&m, &t(Fp8E4m3), &t(Fp8E4m3), Some(Fp32), [true, true]), Some((Some(Bf16), Some(Bf16))));
    assert_eq!(choose_with(&m, &t(Fp8E4m3), &t(Fp8E4m3), Some(Fp16), [true, true]), Some((None, None)));
    let int = MacModes { modes: vec![acc(Int8, Int8, Int16, 4.0)], dequantize: false };
    assert_eq!(choose_with(&int, &t(Int4), &t(Int4), Some(Int32), [true, true]), None);
    assert!(kiln_wl::convert::accumulates(Fp32, Bf16) && !kiln_wl::convert::accumulates(Fp16, Bf16));
}
