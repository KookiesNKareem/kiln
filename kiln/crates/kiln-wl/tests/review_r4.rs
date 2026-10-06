//! Regressions from the round-4 review of kiln-wl.

use kiln_ir::common::Id;
use kiln_ir::precision::Precision::{self, *};
use kiln_ir::wl::*;
use kiln_wl::convert::{MacMode, MacModes, insert_converts, widens};
use kiln_wl::{evaluate_snapshot, zoo};
use serde_json::{Value, json};

fn model(v: Value) -> Model {
    serde_json::from_value(v).unwrap()
}

fn decode() -> Scenario {
    zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(1, 1, 1))
}

fn block(scalar: Precision, scale: Precision, zero_point: Option<Precision>) -> ElemType {
    ElemType { scalar, scaling: Scaling::Block { axis: -1, block: 32, scale, zero_point, tensor_scale: None } }
}

#[test]
fn block_scaled_widening_holds_the_scaled_value() {
    // int4 7 x bf16 scale 1.0078125 = 7.0546875 needs 11 significant bits; bf16 rounds it to 7.0625.
    assert!(!widens(Bf16, &block(Int4, Bf16, None)));
    assert!(widens(Fp32, &block(Int4, Bf16, None)));
    assert!(!widens(Bf16, &ElemType::from(Int4G128)) && widens(Fp32, &ElemType::from(Int4G128)));
    // A power-of-two scale only shifts the element; nvfp4's e2m1 x e4m3 products fit bf16.
    assert!(widens(Bf16, &block(Int4, E8m0, None)) && widens(Bf16, &ElemType::from(Nvfp4)));
    assert!(widens(Fp32, &block(Fp8E4m3, Bf16, None)) && !widens(Tf32, &block(Fp8E4m3, Bf16, None)));
}

#[test]
fn converted_operands_get_distinct_temps() {
    let m = model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {"main": {"params": ["a.b", "a_b"], "results": ["y"], "tensors": {
            "a.b": {"shape": [2, 4], "dtype": "fp8_e4m3", "class": "input"},
            "a_b": {"shape": [3, 4], "dtype": "fp8_e4m3", "class": "input"},
            "y": {"shape": [2, 3], "dtype": "bf16", "class": "output"},
        }, "nodes": [{"id": "mm", "op": "einsum", "eq": "td,nd->tn", "inputs": ["a.b", "a_b"], "outputs": ["y"]}]}},
    }));
    let (_, mut lg, _) = evaluate_snapshot(&m, &decode()).unwrap();
    let bf16 = MacModes { modes: vec![MacMode { a: Bf16, b: Bf16, acc: Fp32, rate: 1.0 }], dequantize: false };
    assert_eq!(insert_converts(&mut lg, &bf16).unwrap(), 2);
    let n = &lg.nodes[0];
    let k = n.lowered.kernels.iter().find(|k| k.class == KernelClass::Contraction).unwrap();
    let reads: Vec<&Id> = k.operands.iter().filter(|o| o.access == Access::Read).map(|o| &o.tensor).collect();
    assert_ne!(reads[0], reads[1], "{reads:?}");
    let sources: Vec<&Id> = reads
        .iter()
        .map(|t| &n.lowered.kernels.iter().find(|c| c.operands.iter().any(|o| &o.tensor == *t && o.access == Access::Write)).unwrap().operands[0].tensor)
        .collect();
    assert_eq!(sources.iter().map(|t| t.as_str()).collect::<Vec<_>>(), ["a.b", "a_b"]);
}

fn called(param_shape: u64) -> Model {
    model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {
            "main": {"params": ["x"], "results": ["y"], "tensors": {
                "x": {"shape": [8], "dtype": "bf16", "class": "input"},
                "y": {"shape": [param_shape], "dtype": "bf16", "class": "output"},
            }, "nodes": [{"id": "c", "op": "call", "graph": "sub", "inputs": ["x"], "outputs": ["y"]}]},
            "sub": {"params": ["a"], "results": ["r"], "tensors": {
                "a": {"shape": [param_shape], "dtype": "bf16", "class": "activation"},
                "r": {"shape": [param_shape], "dtype": "bf16", "class": "activation"},
            }, "nodes": [{"id": "s", "op": "act", "fn": "relu", "inputs": ["a"], "outputs": ["r"]}]},
        },
    }))
}

#[test]
fn param_declarations_must_match_and_keep_the_callers_binding() {
    let e = evaluate_snapshot(&called(4), &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-SHAPE-001", "{e:?}");
    let (_, _, st) = evaluate_snapshot(&called(8), &decode()).unwrap();
    assert_eq!((st.io_read, st.io_written), (16, 16), "a declared param keeps the caller's input origin");
}

fn repeated(y_shape: u64) -> Model {
    model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {
            "main": {"params": ["x"], "results": ["y"], "tensors": {
                "x": {"shape": [4], "dtype": "bf16", "class": "input"},
                "y": {"shape": [y_shape], "dtype": "bf16", "class": "output"},
            }, "nodes": [{"id": "rep", "op": "repeat", "body": "b", "count": 2, "stacked": [],
                          "carry": [{"init": "x", "param": "a", "yield": "r", "out": "y"}], "inputs": ["x"], "outputs": ["y"]}]},
            "b": {"params": ["a"], "results": ["r"], "tensors": {"r": {"shape": [4], "dtype": "bf16", "class": "activation"}}, "nodes": [
                {"id": "s", "op": "act", "fn": "relu", "inputs": ["a"], "outputs": ["r"]},
            ]},
        },
    }))
}

#[test]
fn repeat_yields_bind_to_the_carry_out() {
    let (_, lg, st) = evaluate_snapshot(&repeated(4), &decode()).unwrap();
    assert_eq!(lg.nodes[0].multiplicity, 2);
    assert_eq!((st.io_read, st.io_written), (8, 8), "the final carry is written to the host once");
    let e = evaluate_snapshot(&repeated(8), &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-SHAPE-001", "{e:?}");
}

#[test]
fn nested_repeat_carries_reach_the_outer_output() {
    let rep = |body: &str, x: &str, p: &str, r: &str, y: &str| {
        json!({"id": format!("rep_{body}"), "op": "repeat", "body": body, "count": 3, "stacked": [],
               "carry": [{"init": x, "param": p, "yield": r, "out": y}], "inputs": [x], "outputs": [y]})
    };
    let act = json!({"shape": [1], "dtype": "bf16", "class": "activation"});
    let m = model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {
            "main": {"params": ["x"], "results": ["y"], "tensors": {
                "x": {"shape": [1], "dtype": "bf16", "class": "input"},
                "y": {"shape": [1], "dtype": "bf16", "class": "output"},
            }, "nodes": [rep("b1", "x", "a", "r", "y")]},
            "b1": {"params": ["a"], "results": ["r"], "tensors": {"r": act}, "nodes": [rep("b2", "a", "p", "q", "r")]},
            "b2": {"params": ["p"], "results": ["q"], "tensors": {"q": act}, "nodes": [
                {"id": "s", "op": "act", "fn": "relu", "inputs": ["p"], "outputs": ["q"]},
            ]},
        },
    }));
    let (_, _, st) = evaluate_snapshot(&m, &decode()).unwrap();
    assert_eq!((st.io_read, st.io_written), (2, 2));
}

#[test]
fn param_count_overflow_is_diagnosed() {
    let weights = |shape: Value, stack: u64| {
        model(json!({
            "symbols": {},
            "entry": {"forward": "main"},
            "tensors": {"w": {"shape": shape, "dtype": "bf16", "class": "weight", "stack": stack}},
            "graphs": {"main": {"params": [], "results": [], "tensors": {}, "nodes": []}},
        }))
    };
    let b = |m: &Model| kiln_wl::bind(m, &SeqBatch::uniform(1, 1, 1), &Default::default()).unwrap();
    for (shape, stack) in [(json!([4294967296u64, 4294967296u64]), 1), (json!([4294967296u64]), 4294967296u64)] {
        let m = weights(shape, stack);
        assert_eq!(kiln_wl::param_count(&m, &b(&m)).unwrap_err().code, "E-WL-DIM-001");
    }
    let m = weights(json!([0, 4294967296u64, 4294967296u64]), 1);
    assert_eq!(kiln_wl::param_count(&m, &b(&m)).unwrap(), 0);
    let m = weights(json!([4096, 4096]), 32);
    assert_eq!(kiln_wl::param_count(&m, &b(&m)).unwrap(), 4096 * 4096 * 32);
}
