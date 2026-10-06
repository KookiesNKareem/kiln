//! Regressions from the round-6 review of kiln-wl.

use indexmap::IndexMap;
use kiln_ir::wl::*;
use kiln_wl::{bind, zoo};

#[test]
fn mesh_products_that_wrap_to_one_rank_are_rejected() {
    let m = zoo::workload("llama3_8b:decode_b32").unwrap();
    let b = bind(m.model(), &SeqBatch::uniform(1, 1, 16), &IndexMap::new()).unwrap();
    let plan = ParallelPlan {
        mesh: IndexMap::from([("a".to_string(), 11), ("b".to_string(), 998_724_481), ("c".to_string(), 3_358_236_963)]),
        template: None,
        shardings: vec![],
        pipeline: None,
        allow_uneven: false,
    };
    assert_eq!(kiln_wl::partition::partition(m.model(), &b, &plan).unwrap_err().code, "E-WL-DIM-001");
}

fn einsum_model(eq: &str, w_shape: [u64; 2], w_dtype: &str) -> Model {
    serde_json::from_value(serde_json::json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {"main": {"params": ["x", "w"], "results": ["y"], "tensors": {
            "x": {"shape": [1, 128], "dtype": "bf16", "class": "input"},
            "w": {"shape": w_shape, "dtype": w_dtype, "class": "input"},
            "y": {"shape": [1, 4], "dtype": "fp32", "class": "output"},
        }, "nodes": [{"id": "mm", "op": "einsum", "eq": eq, "inputs": ["x", "w"], "outputs": ["y"]}]}},
    }))
    .unwrap()
}

#[test]
fn per_axis_scales_along_a_reduced_dim_are_applied_before_the_reduction() {
    use kiln_ir::precision::Precision::*;
    use kiln_wl::convert::{MacMode, MacModes, insert_converts};
    let modes = MacModes { modes: vec![MacMode { a: Bf16, b: Bf16, acc: Fp32, rate: 1.0 }, MacMode { a: Fp32, b: Fp32, acc: Fp32, rate: 0.25 }], dequantize: false };
    let lowered = |m: &Model| kiln_wl::evaluate_snapshot(m, &zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(1, 1, 1))).unwrap().1;
    let mut out_axis = lowered(&einsum_model("td,nd->tn", [4, 128], "int8_pc"));
    assert_eq!(insert_converts(&mut out_axis, &modes).unwrap(), 1, "a per-output-channel scale waits for the accumulator");
    let cvt = out_axis.nodes[0].lowered.temps.last().unwrap();
    assert_eq!(cvt.1.dtype, ElemType::plain(Bf16));
    let mut red_axis = lowered(&einsum_model("td,dn->tn", [128, 4], "int8_pc"));
    assert_eq!(insert_converts(&mut red_axis, &modes).unwrap_err().code, "E-WL-DT-005", "int8 x fp32 scale products need 32 significant bits");
}

#[test]
fn secondary_block_scales_are_charged() {
    use kiln_ir::precision::Precision::*;
    let x = vec![1u64, 32];
    let node = Node::new("n", Op::Dequantize(DequantizeAttrs { target: None }), &["in0"], &["out0"]);
    let none = SeqBatch { segments: vec![] };
    let ctx = kiln_wl::NodeCtx {
        node: &node,
        inputs: vec![TypeInfo::new(x.clone(), Nvfp4.into(), TensorClass::Weight)],
        outputs: vec![TypeInfo::new(x, ElemType::plain(Fp32), TensorClass::Activation)],
        seqs: &none,
        moe_rows: None,
        group_size: None,
    };
    let body = kiln_wl::lower(&ctx).unwrap().kernels[0].body;
    assert_eq!((body.mul, body.cvt), (2, 1), "block scale, then the fp32 tensor scale");
}

#[test]
fn stacked_resident_bytes_that_overflow_are_dimension_errors() {
    let big = 1u64 << 60;
    let m: Model = serde_json::from_value(serde_json::json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {"w": {"shape": [big, big], "dtype": "bf16", "class": "weight", "stack": 128}},
        "graphs": {"main": {"params": ["x"], "results": ["x"], "tensors": {
            "x": {"shape": [1], "dtype": "bf16", "class": "input"},
        }, "nodes": []}},
    }))
    .unwrap();
    let e = kiln_wl::evaluate_snapshot(&m, &zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(1, 1, 1))).unwrap_err();
    assert_eq!(e[0].code, "E-WL-DIM-001", "{e:?}");
    let b = bind(&m, &SeqBatch::uniform(1, 1, 1), &IndexMap::new()).unwrap();
    let plan = ParallelPlan { mesh: IndexMap::from([("tp".to_string(), 1)]), template: None, shardings: vec![], pipeline: None, allow_uneven: false };
    assert_eq!(kiln_wl::partition::partition(&m, &b, &plan).unwrap_err().code, "E-WL-DIM-001");
}

#[test]
fn moe_dispatch_outputs_must_hold_the_routed_rows() {
    use kiln_ir::precision::Precision::*;
    let none = SeqBatch { segments: vec![] };
    let run = |layout: DispatchLayout, out: &[u64]| {
        let a = MoeDispatchAttrs { n_experts: 2, top_k: 1, capacity_factor: None, drop_policy: DropPolicy::NoDrop, layout };
        let node = Node::new("n", Op::MoeDispatch(a), &["x", "idx"], &["xe"]);
        let ctx = kiln_wl::NodeCtx {
            node: &node,
            inputs: vec![TypeInfo::new(vec![8, 64], ElemType::plain(Bf16), TensorClass::Activation), TypeInfo::new(vec![8, 1], ElemType::plain(Int32), TensorClass::Activation)],
            outputs: vec![TypeInfo::new(out.to_vec(), ElemType::plain(Bf16), TensorClass::Activation)],
            seqs: &none,
            moe_rows: None,
            group_size: None,
        };
        (kiln_wl::lower(&ctx).map(|_| ()).map_err(|e| e.code), kiln_wl::cost_hint(&ctx).map(|_| ()).map_err(|e| e.code))
    };
    let bad = (Err("E-WL-SHAPE-001".to_string()), Err("E-WL-SHAPE-001".to_string()));
    assert_eq!(run(DispatchLayout::CapacityPadded, &[1]), bad);
    assert_eq!(run(DispatchLayout::CapacityPadded, &[2, 3, 64]), bad, "2 x 3 slots for 8 dropless rows");
    assert_eq!(run(DispatchLayout::CapacityPadded, &[2, 4, 64]), (Ok(()), Ok(())));
    assert_eq!(run(DispatchLayout::Ragged, &[7, 64]), bad);
    assert_eq!(run(DispatchLayout::Ragged, &[8, 64]), (Ok(()), Ok(())));
}
