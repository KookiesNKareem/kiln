//! Worked examples of 02 §14 and the legacy golden file.

use kiln_ir::wl::*;
use kiln_wl::zoo::{self, preset};
use kiln_wl::{LoweredGraph, evaluate_snapshot};

fn node_flops(lg: &LoweredGraph, id: &str) -> u128 {
    lg.nodes
        .iter()
        .filter(|n| n.node.id.as_str() == id)
        .map(|n| n.cost.flops_mm)
        .sum()
}

#[test]
fn llama3_8b_decode_b8_kv2048() {
    let m = zoo::workload("llama3_8b:decode_b8_kv2048").unwrap();
    let (b, lg, st) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
    assert_eq!(
        (
            b.values["T"],
            b.values["N"],
            b.values["slots"],
            b.values["kv_cap"]
        ),
        (8, 8, 8, 2048)
    );
    assert_eq!(node_flops(&lg, "qkv"), 402_653_184);
    assert_eq!(node_flops(&lg, "attn"), 268_435_456);
    assert_eq!(node_flops(&lg, "o_proj"), 268_435_456);
    assert_eq!(node_flops(&lg, "gate_up"), 1_879_048_192);
    assert_eq!(node_flops(&lg, "down"), 939_524_096);
    assert_eq!(node_flops(&lg, "lm_head"), 8_405_385_216);
    assert_eq!(st.useful.flops_mm, 128_664_469_504);
    let layer_w: u128 = lg
        .nodes
        .iter()
        .filter(|n| n.multiplicity == 32)
        .map(|n| n.cost.weight_bytes)
        .sum();
    assert_eq!(layer_w, 436_207_616 + 16_384);
    let embed_rows = 8 * 4096 * 2;
    assert_eq!(st.weight_read, 15_009_849_344 + embed_rows);
    assert_eq!(st.kv_read, 2_147_483_648);
    assert_eq!(st.kv_written, 1_048_576);
    assert_eq!(st.resident.weights, 16_060_522_496);
    assert_eq!(st.resident.kv_cache, 2_147_483_648);
    assert_eq!(st.resident.weights + st.resident.kv_cache, 18_208_006_144);
    let attn = lg
        .nodes
        .iter()
        .find(|n| n.node.id.as_str() == "attn")
        .unwrap();
    assert_eq!(
        attn.cost.transc,
        524_288 + 8 * 32,
        "softmax points + per-row reciprocals"
    );
    let c = st.compulsory_bytes();
    assert!(
        (c as f64 / 17.16e9 - 1.0).abs() < 5e-4,
        "compulsory bytes {c}"
    );
    assert_eq!(
        st.weight_read - embed_rows + st.kv_read + st.kv_written,
        17_158_381_568
    );
}

#[test]
fn llama3_8b_prefill_causal_attention_is_mask_aware() {
    let m = zoo::workload("llama3_8b:prefill_b1").unwrap();
    let (_, lg, _) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
    assert_eq!(node_flops(&lg, "attn"), 34_376_515_584);
    let proj = ["qkv", "o_proj", "gate_up", "down"]
        .iter()
        .map(|n| node_flops(&lg, n))
        .sum::<u128>();
    assert_eq!(proj, 893_353_197_568);
    let compat = zoo::suite("legacy").unwrap().remove(0);
    let (_, lg, _) = evaluate_snapshot(compat.model(), compat.scenario()).unwrap();
    assert_eq!(
        node_flops(&lg, "attn"),
        68_719_476_736,
        "harness-compat attention is dense"
    );
}

#[test]
fn param_counts_match_presets() {
    assert_eq!(
        zoo::param_count(&preset("llama3_8b").unwrap()).unwrap(),
        8_030_261_248
    );
    let p70 = zoo::param_count(&preset("llama3_70b").unwrap()).unwrap();
    assert_eq!(p70, 70_553_706_496);
    assert!((p70 as f64 / 70.55e9 - 1.0).abs() < 1e-3);
    assert_eq!(
        zoo::param_count(&preset("llama3_1_8b").unwrap()).unwrap(),
        8_030_261_248
    );
    assert_eq!(
        zoo::param_count(&preset("llama2_70b").unwrap()).unwrap(),
        68_976_648_192,
        "HF meta-llama/Llama-2-70b-hf"
    );
    let gptj = zoo::param_count(&preset("gptj_6b").unwrap()).unwrap();
    assert_eq!(
        gptj, 6_050_258_944,
        "weights without the 0.6 M bias entries"
    );
    assert!(
        (gptj as f64 / 6_053_381_344.0 - 1.0).abs() < 1e-3,
        "EleutherAI model card"
    );
}

#[test]
fn gptj_block_is_parallel_layernorm_ungated() {
    let m = zoo::workload("gptj_6b:decode_b1").unwrap();
    let (_, lg, st) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
    assert_eq!(
        st.weight_read,
        2 * (6_050_258_944 - 50_400 * 4096 + 4096),
        "one embedding row per token"
    );
    let block = &m.model().graphs[&kiln_ir::common::Id::new("block").unwrap()];
    let ops: Vec<&str> = block.nodes.iter().map(|n| n.op.name()).collect();
    assert_eq!(ops.iter().filter(|o| **o == "layer_norm").count(), 1);
    assert!(!ops.contains(&"gated_act") && ops.contains(&"act"));
    let up = block.nodes.iter().find(|n| n.id.as_str() == "up").unwrap();
    assert_eq!(
        up.inputs[0].as_str(),
        "xn",
        "MLP reads the attention pre-norm"
    );
    assert!(node_flops(&lg, "attn") > 0);
}

#[test]
fn spec_example_document_matches_zoo() {
    let src = include_str!("fixtures/llama3_8b_decode_b8.json");
    let doc: WorkloadDoc = serde_json::from_str(src).unwrap();
    assert!(validate_doc(&doc).is_empty(), "{:?}", validate_doc(&doc));
    let m = doc.expanded().unwrap();
    let zm = zoo::build_model(&preset("llama3_8b").unwrap()).unwrap();
    assert_eq!(canonical_model(m), canonical_model(&zm));
    assert_eq!(model_hash(m), model_hash(&zm));
    assert!(model_hash(m).starts_with("wl1-"));
    let s = &doc.scenarios[&kiln_ir::common::Id::new("decode_b8_kv2048").unwrap()];
    assert_eq!(
        scenario_hash(s),
        scenario_hash(&zoo::scenario("decode_b8_kv2048").unwrap())
    );
    let (_, _, st) = evaluate_snapshot(m, s).unwrap();
    assert_eq!(st.useful.flops_mm, 128_664_469_504);
}

#[test]
fn zoo_sugar_expands_before_hashing() {
    let sugar: WorkloadDoc = serde_json::from_value(serde_json::json!({
        "kiln_workload": "0.1", "id": "l2",
        "model": {"zoo": {"preset": "llama3_8b", "overrides": {"n_layers": 2}}}
    }))
    .unwrap();
    let doc = zoo::expand_doc(&sugar).unwrap();
    let mut cfg = preset("llama3_8b").unwrap();
    cfg.n_layers = 2;
    let direct = zoo::build_model(&cfg).unwrap();
    assert_eq!(model_hash(doc.expanded().unwrap()), model_hash(&direct));
    assert_ne!(
        model_hash(&direct),
        model_hash(&zoo::build_model(&preset("llama3_8b").unwrap()).unwrap())
    );
    let bad = WorkloadDoc {
        model: ModelSrc::Zoo {
            zoo: ZooRef {
                preset: "llama3_8b".into(),
                overrides: serde_json::from_str(r#"{"layers": 2}"#).unwrap(),
            },
        },
        ..sugar
    };
    assert_eq!(zoo::expand_doc(&bad).unwrap_err().code, "E-WL-ZOO-001");
}

#[test]
fn canonical_json_round_trips() {
    let m = zoo::workload("llama3_8b:decode_b1").unwrap();
    let json = serde_json::to_string(&m.doc).unwrap();
    let back: WorkloadDoc = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m.doc);
    let h = workload_hash(m.model(), m.scenario(), None);
    assert_eq!(
        h,
        workload_hash(back.expanded().unwrap(), m.scenario(), None)
    );
    let mut shuffled = m.model().clone();
    shuffled.graphs.values_mut().for_each(|g| g.nodes.reverse());
    assert_eq!(
        model_hash(&shuffled),
        model_hash(m.model()),
        "node order in the file does not change the hash"
    );
}

#[test]
fn legacy_oplist_golden() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/oplist.json")).unwrap();
    let rows = kiln_wl::legacy::oplist().unwrap();
    let ours = serde_json::to_value(&rows).unwrap();
    let (g, o) = (golden.as_array().unwrap(), ours.as_array().unwrap());
    assert_eq!(o.len(), 33);
    for (i, (a, b)) in g.iter().zip(o).enumerate() {
        assert_eq!(a, b, "row {i}");
    }
    assert_eq!(g.len(), o.len());
}

#[test]
fn legacy_rows_conserve_ir_flops() {
    for m in zoo::suite("legacy").unwrap() {
        let (_, lg, _) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
        let rows: Vec<_> = kiln_wl::legacy::oplist()
            .unwrap()
            .into_iter()
            .filter(|r| r.phase == m.scenario.as_str())
            .collect();
        let harness: u128 = rows.iter().map(|r| r.flops * u128::from(r.count)).sum();
        let ir: u128 = lg
            .nodes
            .iter()
            .map(|n| n.cost.flops_mm * u128::from(n.multiplicity))
            .sum();
        assert_eq!(harness, ir, "{}", m.name);
    }
}

#[test]
fn standard_suite_scores_whole_steps() {
    let s = zoo::suite("standard").unwrap();
    let names: Vec<&str> = s.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "llama3_8b:prefill_b1",
            "llama3_8b:decode_b1",
            "llama3_8b:decode_b8",
            "llama3_8b:decode_b32"
        ]
    );
    for m in &s {
        let sc = m.scenario();
        assert_eq!(
            (sc.scope, sc.eval_mode),
            (ScoringScope::Step, EvalMode::WholeGraph)
        );
        let (_, lg, st) = evaluate_snapshot(m.model(), sc).unwrap();
        assert!(lg.nodes.iter().any(|n| n.node.op.name() == "sample"));
        assert!(st.useful.vec_ops > 0 && st.useful.transc > 0);
    }
    assert!(zoo::suite("evolve").is_err());
}
