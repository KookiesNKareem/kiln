//! Validation, binding, expansion, bench export and partition behaviour.

use indexmap::IndexMap;
use kiln_ir::common::Id;
use kiln_ir::wl::*;
use kiln_wl::bench::{BenchKind, BenchOp, Residency, export};
use kiln_wl::expand::{decode_points, expand};
use kiln_wl::zoo::{self, preset};
use kiln_wl::{bind, evaluate_snapshot};

fn llama() -> Model {
    zoo::build_model(&preset("llama3_8b").unwrap()).unwrap()
}

fn id(s: &str) -> Id {
    Id::new(s).unwrap()
}

fn codes(m: &Model) -> Vec<String> {
    validate_model(m).into_iter().map(|d| d.code).collect()
}

#[test]
fn zoo_models_validate_clean() {
    assert!(validate_model(&llama()).is_empty());
    let mut cfg = preset("llama3_70b").unwrap();
    cfg.fuse_qkv = false;
    cfg.harness_compat = true;
    let m = zoo::build_model(&cfg).unwrap();
    assert!(validate_model(&m).is_empty(), "{:?}", validate_model(&m));
    let s = zoo::scenario("decode_b2").unwrap();
    let (_, lg, st) = evaluate_snapshot(&m, &s).unwrap();
    assert!(lg.nodes.iter().any(|n| n.node.id.as_str() == "q_proj"));
    assert_eq!(st.resident.weights, 2 * 70_553_706_496);
}

#[test]
fn structural_errors() {
    let mut m = llama();
    let block = m.graphs.get_mut(&id("block")).unwrap();
    block.nodes.push(Node::new(
        "dup",
        Op::Map(MapAttrs {
            func: FnSpec::One(MapFn::Add),
        }),
        &["x", "a"],
        &["h"],
    ));
    assert!(
        codes(&m).contains(&"E-WL-CLS-001".to_string()),
        "two producers of h"
    );

    let mut m = llama();
    m.graphs.get_mut(&id("fwd")).unwrap().nodes.push(Node::new(
        "w",
        Op::Act(ActAttrs { func: MapFn::Relu }),
        &["hf"],
        &["w_final_norm"],
    ));
    assert!(
        validate_model(&m)
            .iter()
            .any(|d| d.code == "E-WL-CLS-001" && d.message.contains("writes"))
    );
    let mut m = llama();
    m.tensors.get_mut(&id("w_o")).unwrap().class = TensorClass::Activation;
    assert!(codes(&m).contains(&"E-WL-CLS-001".to_string()));

    let mut m = llama();
    let block = m.graphs.get_mut(&id("block")).unwrap();
    let n = block
        .nodes
        .iter_mut()
        .find(|n| n.id.as_str() == "attn_norm")
        .unwrap();
    n.inputs[0] = id("y");
    assert!(
        codes(&m).contains(&"E-WL-DAG-001".to_string()),
        "{:?}",
        codes(&m)
    );

    let mut m = llama();
    let block = m.graphs.get_mut(&id("block")).unwrap();
    let t = block.tensors[&id("ck1")].clone();
    block.tensors.insert(id("ck2"), t);
    assert!(codes(&m).contains(&"E-WL-ALIAS-001".to_string()));

    let mut m = llama();
    m.tensors.get_mut(&id("w_o")).unwrap().shape[0] = DimExpr::sym("D");
    assert!(codes(&m).contains(&"E-WL-SYM-001".to_string()));

    let mut m = llama();
    m.graphs.get_mut(&id("fwd")).unwrap().nodes[0].inputs[1] = id("w_missing");
    assert!(codes(&m).contains(&"E-WL-REF-001".to_string()));

    let mut m = llama();
    m.tensors.get_mut(&id("w_o")).unwrap().dtype =
        ElemType::plain(kiln_ir::precision::Precision::Mxfp4);
    assert!(codes(&m).contains(&"E-WL-DT-002".to_string()));
}

#[test]
fn binding_errors() {
    let m = llama();
    let bad = SeqBatch {
        segments: vec![Segment::new(1, 4, 2)],
    };
    assert_eq!(
        bind(&m, &bad, &IndexMap::new()).unwrap_err()[0].code,
        "E-WL-SCN-002"
    );

    let mut m2 = m.clone();
    m2.symbols.get_mut("L").unwrap().divisible_by = Some(5);
    assert_eq!(
        bind(&m2, &SeqBatch::uniform(1, 1, 8), &IndexMap::new()).unwrap_err()[0].code,
        "E-WL-SYM-002"
    );

    let mut m3 = m.clone();
    m3.symbols.insert("D".into(), SymbolDecl::size(None));
    m3.tensors.get_mut(&id("w_o")).unwrap().shape[0] = DimExpr::sym("D");
    assert_eq!(
        bind(&m3, &SeqBatch::uniform(1, 1, 8), &IndexMap::new()).unwrap_err()[0].code,
        "E-WL-SYM-001"
    );

    let mut m4 = m.clone();
    m4.tensors.get_mut(&id("w_o")).unwrap().shape[0] = DimExpr::parse("T/3").unwrap();
    let s = zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(2, 1, 8));
    assert_eq!(
        evaluate_snapshot(&m4, &s).unwrap_err()[0].code,
        "E-WL-DIM-001"
    );

    let mut tight = zoo::whole_step(PhaseKind::Decode, SeqBatch::uniform(2, 1, 64));
    tight.bindings.insert("kv_cap".into(), 32);
    let e = evaluate_snapshot(&m, &tight).unwrap_err();
    assert_eq!(e[0].code, "E-WL-SCN-003");
    assert!(e[0].path.as_deref().unwrap().contains("kv_append"));
}

#[test]
fn mixed_continuous_step_lowers() {
    let m = llama();
    let seqs = SeqBatch {
        segments: vec![
            Segment::new(1, 512, 1024),
            Segment::new(6, 1, 300),
            Segment::new(1, 1, 2000),
        ],
    };
    let s = zoo::whole_step(PhaseKind::Mixed, seqs.clone());
    let (b, lg, st) = evaluate_snapshot(&m, &s).unwrap();
    assert_eq!((b.values["T"], b.values["N"]), (519, 8));
    let attn = lg
        .nodes
        .iter()
        .find(|n| n.node.id.as_str() == "attn")
        .unwrap();
    let pts: u128 = 512 * 512 + 512 * 513 / 2 + 6 * 300 + 2000;
    assert_eq!(attn.cost.flops_mm, 2 * 32 * pts * 256);
    assert!(st.kv_read > 0);
}

#[test]
fn static_expansion_and_points() {
    assert_eq!(decode_points(255, 3), vec![1, 128, 255]);
    assert_eq!(decode_points(2, 3), vec![1, 2]);
    let s = zoo::scenario("static_b8_p1024_g256").unwrap();
    let inst = expand(&s, &llama()).unwrap();
    assert_eq!(inst.len(), 4);
    assert_eq!(inst[0].seqs, SeqBatch::uniform(8, 1024, 1024));
    assert_eq!(inst[1].seqs, SeqBatch::uniform(8, 1, 1025));
    assert_eq!(inst[3].seqs, SeqBatch::uniform(8, 1, 1024 + 255));
    assert!(inst.iter().all(|i| i.bindings["kv_cap"] == 1024 + 255));
    let mut exact = s.clone();
    if let ScenarioMode::Static {
        decode_sampling, ..
    } = &mut exact.mode
    {
        *decode_sampling = DecodeSampling::Exact;
    }
    assert_eq!(expand(&exact, &llama()).unwrap().len(), 256);
}

#[test]
fn seq_batch_canonical_merges() {
    let a = SeqBatch {
        segments: vec![
            Segment::new(2, 1, 9),
            Segment::new(1, 4, 9),
            Segment::new(3, 1, 9),
        ],
    };
    let c = a.canonical();
    assert_eq!(
        c.segments,
        vec![Segment::new(5, 1, 9), Segment::new(1, 4, 9)]
    );
    assert_eq!(
        binding_hash(&a, &IndexMap::new()),
        binding_hash(&c, &IndexMap::new())
    );
}

#[test]
fn bench_export_keys() {
    let m = zoo::workload("llama3_8b:decode_b8").unwrap();
    let (_, lg, _) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
    let ops = export(&lg, Residency::Resident);
    let role = |r: &'static str| ops.iter().filter(move |o| o.at.role.as_deref() == Some(r));
    let qkv = role("attn.qkv").next().unwrap();
    assert_eq!(qkv.op.kind, BenchKind::Linear);
    assert_eq!(qkv.op.legacy_name().unwrap(), "gemm_8_6144_4096_linear");
    assert_eq!(qkv.op.legacy_key().unwrap(), "gemm_8_6144_4096");
    assert_eq!(qkv.op, BenchOp::gemm(8, 6144, 4096, true));
    assert!(qkv.key().starts_with("bop1-"));
    let score: Vec<_> = role("attn.core").collect();
    assert_eq!(
        score
            .iter()
            .map(|o| o.op.legacy_name().unwrap())
            .collect::<Vec<_>>(),
        ["bmm_64_4_2048_128", "bmm_64_4_128_2048"]
    );
    assert_eq!(score[0].op, BenchOp::bmm(64, 4, 2048, 128));
    let o2 = role("attn.o").next().unwrap();
    let mut moved = o2.clone();
    moved.at.count = 99;
    moved.at.source = "elsewhere".into();
    moved.at.residency = Residency::ColdDram;
    assert_eq!(
        moved.key(),
        o2.key(),
        "occurrence data and residency are not part of the key"
    );
    let norm = role("attn.norm").next().unwrap();
    assert_eq!(norm.op.kind, BenchKind::Rmsnorm);
    assert_eq!(norm.op.operands["in0"].shape, [8, 4096]);
    assert!(ops.iter().all(|o| o.at.ir_op != "layout"));

    let manifest = kiln_wl::bench::manifest("llama3_8b:decode_b8").unwrap();
    let residual = manifest
        .ops
        .iter()
        .find(|o| o.op.op.as_deref() == Some("map"))
        .unwrap();
    assert_eq!(
        residual.uses.len(),
        2,
        "both residual adds share one descriptor"
    );
    let head = manifest
        .get(&role("lm_head").next().unwrap().key())
        .unwrap();
    assert_eq!(head.op.operands["out"].dtype, "fp32");
    let keys: std::collections::BTreeSet<_> = manifest.ops.iter().map(|o| &o.key).collect();
    assert_eq!(keys.len(), manifest.ops.len());
}

#[test]
fn trivial_partition() {
    let m = llama();
    let b = bind(&m, &SeqBatch::uniform(1, 1, 16), &IndexMap::new()).unwrap();
    let plan = ParallelPlan {
        mesh: IndexMap::from([("tp".to_string(), 1)]),
        template: None,
        shardings: vec![],
        pipeline: None,
        allow_uneven: false,
    };
    let p = kiln_wl::partition::partition(&m, &b, &plan).unwrap();
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.stages[0].layers, 0..32);
    let tp4 = ParallelPlan {
        mesh: IndexMap::from([("tp".to_string(), 4)]),
        ..plan
    };
    assert_eq!(
        kiln_wl::partition::partition(&m, &b, &tp4)
            .unwrap_err()
            .code,
        "E-WL-OP-001"
    );
}

#[test]
fn hints_equal_lowering_on_whole_models() {
    for name in [
        "llama3_8b:prefill_b1",
        "llama3_8b:decode_b32",
        "llama3_70b:decode_b1_kv32768",
    ] {
        let m = zoo::workload(name).unwrap();
        let (_, lg, _) = evaluate_snapshot(m.model(), m.scenario()).unwrap();
        for n in &lg.nodes {
            assert_eq!(n.hint, n.cost, "{name} {}", n.path);
        }
    }
}

#[test]
fn self_dependent_node_is_a_cycle() {
    let mut m = llama();
    let block = m.graphs.get_mut(&id("block")).unwrap();
    let n = block.nodes.iter_mut().find(|n| n.id.as_str() == "attn_norm").unwrap();
    n.inputs[0] = n.outputs[0].clone();
    assert!(codes(&m).contains(&"E-WL-DAG-001".to_string()), "{:?}", codes(&m));
}
