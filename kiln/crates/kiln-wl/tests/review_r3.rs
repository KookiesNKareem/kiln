//! Regressions from the round-3 review of kiln-wl.

use kiln_ir::wl::*;
use kiln_wl::evaluate_snapshot;
use kiln_wl::zoo;
use serde_json::{Value, json};

fn model(v: Value) -> Model {
    serde_json::from_value(v).unwrap()
}

fn decode() -> Scenario {
    zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(1, 1, 1))
}

fn called(y_shape: u64, call_outputs: Value) -> Model {
    let mut m = json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {
            "main": {"params": ["x"], "results": ["y"], "tensors": {
                "x": {"shape": [4], "dtype": "bf16", "class": "input"},
                "y": {"shape": [y_shape], "dtype": "bf16", "class": "output"},
            }, "nodes": [
                {"id": "c", "op": "call", "graph": "sub", "inputs": ["x"], "outputs": call_outputs},
            ]},
            "sub": {"params": ["a"], "results": ["r"], "tensors": {"r": {"shape": [4], "dtype": "bf16", "class": "activation"}}, "nodes": [
                {"id": "s", "op": "act", "fn": "relu", "inputs": ["a"], "outputs": ["r"]},
            ]},
        },
    });
    if call_outputs.as_array().unwrap().len() > 1 {
        m["graphs"]["main"]["tensors"]["z"] = json!({"shape": [4], "dtype": "bf16", "class": "activation"});
    }
    model(m)
}

#[test]
fn call_results_take_the_callers_output_class() {
    let (_, _, st) = evaluate_snapshot(&called(4, json!(["y"])), &decode()).unwrap();
    assert_eq!((st.io_read, st.io_written), (8, 8));
    assert_eq!(st.compulsory_bytes(), 16);
}

#[test]
fn call_results_must_match_the_callers_outputs() {
    let e = evaluate_snapshot(&called(8, json!(["y"])), &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-SHAPE-001", "{e:?}");
    let e = evaluate_snapshot(&called(4, json!(["y", "z"])), &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-REF-001", "{e:?}");
}

fn nested_repeats(count: u64) -> Model {
    let rep = |body: &str, x: &str, p: &str, r: &str, y: &str| {
        json!({"id": format!("rep_{body}"), "op": "repeat", "body": body, "count": count, "stacked": [],
               "carry": [{"init": x, "param": p, "yield": r, "out": y}], "inputs": [x], "outputs": [y]})
    };
    let act = json!({"shape": [1], "dtype": "bf16", "class": "activation"});
    model(json!({
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
    }))
}

#[test]
fn nested_repeat_multiplicity_overflow_is_diagnosed() {
    let (_, lg, _) = evaluate_snapshot(&nested_repeats(1 << 16), &decode()).unwrap();
    assert_eq!(lg.nodes[0].multiplicity, 1 << 32);
    let e = evaluate_snapshot(&nested_repeats(1 << 32), &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-DIM-001", "{e:?}");
}

#[test]
fn empty_dynamic_quantization_is_zero_work() {
    let m = model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {},
        "graphs": {"main": {"params": ["x"], "results": ["y"], "tensors": {
            "x": {"shape": [0], "dtype": "bf16", "class": "input"},
            "y": {"shape": [0], "dtype": "fp8_e4m3_pt", "class": "output"},
        }, "nodes": [{"id": "q", "op": "quantize", "amax_from": "dynamic", "inputs": ["x"], "outputs": ["y"]}]}},
    }));
    let (_, lg, st) = evaluate_snapshot(&m, &decode()).unwrap();
    assert_eq!(lg.nodes[0].cost, CostHint::default());
    assert_eq!(st.compulsory_bytes(), 0);
}

#[test]
fn element_count_overflow_is_diagnosed() {
    let m = model(json!({
        "symbols": {},
        "entry": {"forward": "main"},
        "tensors": {"w": {"shape": [8796093022208u64, 8796093022208u64, 4398046511104u64], "dtype": "bf16", "class": "weight"}},
        "graphs": {
            "main": {"params": ["x"], "results": ["y"], "tensors": {
                "x": {"shape": [4], "dtype": "bf16", "class": "input"},
                "y": {"shape": [4], "dtype": "bf16", "class": "output"},
            }, "nodes": [
                {"id": "s", "op": "act", "fn": "relu", "inputs": ["x"], "outputs": ["y"]},
            ]},
        },
    }));
    let e = evaluate_snapshot(&m, &decode()).unwrap_err();
    assert_eq!(e[0].code, "E-WL-DIM-001", "{e:?}");
}
