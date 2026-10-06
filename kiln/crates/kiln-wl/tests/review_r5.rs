//! Regressions from the round-5 review of kiln-wl.

use kiln_ir::precision::Precision::*;
use kiln_ir::wl::*;
use kiln_wl::convert::{MacMode, MacModes, insert_converts};
use kiln_wl::{NodeCtx, Lowered, cost_hint, evaluate_snapshot, lower, zoo};
use serde_json::json;

const NONE: SeqBatch = SeqBatch { segments: vec![] };

fn lowered(op: Op, inputs: Vec<TypeInfo>, outputs: Vec<TypeInfo>) -> Result<Lowered, kiln_ir::common::Diagnostic> {
    let ins: Vec<String> = (0..inputs.len()).map(|i| format!("in{i}")).collect();
    let outs: Vec<String> = (0..outputs.len()).map(|i| format!("out{i}")).collect();
    let node = Node::new("n", op, &ins.iter().map(String::as_str).collect::<Vec<_>>(), &outs.iter().map(String::as_str).collect::<Vec<_>>());
    let ctx = NodeCtx { node: &node, inputs, outputs, seqs: &NONE, moe_rows: None, group_size: None };
    let l = lower(&ctx)?;
    cost_hint(&ctx)?;
    Ok(l)
}

/// Per-element ops summed over kernels: `(add, mul, cvt, max)`.
fn ops(l: &Lowered) -> (u64, u64, u64, u64) {
    l.kernels.iter().fold((0, 0, 0, 0), |acc, k| {
        let n: u64 = k.dims.iter().map(|d| d.extent).product();
        (acc.0 + n * u64::from(k.body.add), acc.1 + n * u64::from(k.body.mul), acc.2 + n * u64::from(k.body.cvt), acc.3 + n * u64::from(k.body.max))
    })
}

fn ti(shape: &[u64], p: impl Into<ElemType>, class: TensorClass) -> TypeInfo {
    TypeInfo::new(shape.to_vec(), p.into(), class)
}

#[test]
fn inserted_asymmetric_dequantization_subtracts_the_zero_point() {
    let m: Model = serde_json::from_value(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {"main": {"params": ["x", "w"], "results": ["y"], "tensors": {
            "x": {"shape": [1, 128], "dtype": "fp32", "class": "input"},
            "w": {"shape": [1, 128], "dtype": "int4_g128", "class": "input"},
            "y": {"shape": [1, 1], "dtype": "fp32", "class": "output"},
        }, "nodes": [{"id": "mm", "op": "einsum", "eq": "td,nd->tn", "inputs": ["x", "w"], "outputs": ["y"]}]}},
    }))
    .unwrap();
    let (_, mut lg, _) = evaluate_snapshot(&m, &zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(1, 1, 1))).unwrap();
    let fp32 = MacModes { modes: vec![MacMode { a: Fp32, b: Fp32, acc: Fp32, rate: 1.0 }], dequantize: false };
    assert_eq!(insert_converts(&mut lg, &fp32).unwrap(), 1);
    let k = lg.nodes[0].lowered.kernels.iter().find(|k| k.class == KernelClass::Map).unwrap();
    assert_eq!((k.body.add, k.body.mul, k.body.cvt), (1, 1, 1), "(q - z) * scale: {:?}", k.body);
}

#[test]
fn explicit_dequantization_prices_the_zero_point_only_when_declared() {
    let x = [2, 128];
    let int4 = lowered(Op::Dequantize(DequantizeAttrs { target: None }), vec![ti(&x, Int4G128, TensorClass::Weight)], vec![ti(&x, Bf16, TensorClass::Activation)]).unwrap();
    assert_eq!(ops(&int4), (256, 256, 256, 0));
    let fp8 = lowered(Op::Dequantize(DequantizeAttrs { target: None }), vec![ti(&x, Fp8E4m3Pt, TensorClass::Weight)], vec![ti(&x, Bf16, TensorClass::Activation)]).unwrap();
    assert_eq!(ops(&fp8), (0, 256, 256, 0));
}

#[test]
fn calibrated_quantization_still_applies_its_scale() {
    let x = [1, 128];
    let q = |target: kiln_ir::precision::Precision, amax_from| {
        let op = Op::Quantize(QuantizeAttrs { target: Some(target.into()), amax_from });
        ops(&lowered(op, vec![ti(&x, Bf16, TensorClass::Activation)], vec![ti(&x, target, TensorClass::Activation)]).unwrap())
    };
    assert_eq!(q(Fp8E4m3Pt, AmaxFrom::Calibrated), (0, 128, 128, 0), "x / s, no amax reduction");
    assert_eq!(q(Fp8E4m3Pt, AmaxFrom::Dynamic), (0, 128, 129, 128));
    assert_eq!(q(Int4G128, AmaxFrom::Calibrated), (128, 128, 128, 0), "x / s + z");
    assert_eq!(q(Fp8E4m3, AmaxFrom::Calibrated), (0, 0, 128, 0), "unscaled: a cast");
}

#[test]
fn flattened_row_counts_beyond_u64_are_rejected() {
    let s = [1 << 32, 1 << 32, 1];
    let op = Op::GatedAct(GatedActAttrs { func: MapFn::Silu, layout: GateLayout::TwoInputs });
    let a = TensorClass::Activation;
    let e = lowered(op, vec![ti(&s, Bf16, a), ti(&s, Bf16, a)], vec![ti(&s, Bf16, a)]).unwrap_err();
    assert_eq!(e.code, "E-WL-DIM-001", "{e:?}");
}
