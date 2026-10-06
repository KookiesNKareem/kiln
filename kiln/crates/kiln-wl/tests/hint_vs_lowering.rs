//! 02 §5.1 invariant: every op's closed-form `cost_hint` equals the counts derived from its lowered kernels.

use kiln_ir::common::Id;
use kiln_ir::precision::Precision;
use kiln_ir::wl::*;
use kiln_wl::count::{class_demands, node_cost};
use kiln_wl::lower::MoeRows;
use kiln_wl::{NodeCtx, cost_hint, lower};
use proptest::prelude::*;
use proptest::strategy::Strategy;

type Sampling = kiln_ir::wl::Strategy;

const BF: ElemType = ElemType::BF16;
const F32: ElemType = ElemType::FP32;
const I32: ElemType = ElemType::INT32;
use TensorClass::{Activation as A, KvCache as K, Weight as W};

fn t(shape: &[u64], dtype: ElemType, class: TensorClass) -> TypeInfo {
    TypeInfo::new(shape.to_vec(), dtype, class)
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

fn check(
    op: Op,
    inputs: Vec<TypeInfo>,
    outputs: Vec<TypeInfo>,
    seqs: &SeqBatch,
    moe_rows: Option<MoeRows>,
) -> Result<(), TestCaseError> {
    check_ids(op, inputs, outputs, seqs, moe_rows, None)
}

fn check_ids(
    op: Op,
    inputs: Vec<TypeInfo>,
    outputs: Vec<TypeInfo>,
    seqs: &SeqBatch,
    moe_rows: Option<MoeRows>,
    ins: Option<Vec<&str>>,
) -> Result<(), TestCaseError> {
    let in_names = names("in", inputs.len());
    let ins: Vec<&str> = ins.unwrap_or_else(|| in_names.iter().map(String::as_str).collect());
    let outs = names("out", outputs.len());
    let outs: Vec<&str> = outs.iter().map(String::as_str).collect();
    let node = Node::new("n", op, &ins, &outs);
    let ctx = NodeCtx {
        node: &node,
        inputs,
        outputs,
        seqs,
        moe_rows,
        group_size: Some(4),
    };
    let hint = cost_hint(&ctx).map_err(|e| TestCaseError::fail(e.to_string()))?;
    let lowered = lower(&ctx).map_err(|e| TestCaseError::fail(e.to_string()))?;
    let derived = node_cost(&ctx, &lowered)
        .map_err(|e| TestCaseError::fail(e.to_string()))?
        .cost;
    prop_assert_eq!(hint, derived, "{:?}", node.op);
    for k in &lowered.kernels {
        prop_assert!(k.id.starts_with("n.k"));
        class_demands(k).map_err(|e| TestCaseError::fail(e.to_string()))?;
    }
    Ok(())
}

fn seqs() -> impl Strategy<Value = SeqBatch> {
    prop::collection::vec((1u64..4, 1u64..7, 0u64..9), 1..4).prop_map(|v| SeqBatch {
        segments: v
            .into_iter()
            .map(|(c, q, extra)| Segment::new(c, q, q + extra))
            .collect(),
    })
}

fn fnspec() -> impl Strategy<Value = MapFn> {
    prop::sample::select(vec![
        MapFn::Add,
        MapFn::Sub,
        MapFn::Mul,
        MapFn::Div,
        MapFn::Silu,
        MapFn::GeluTanh,
        MapFn::GeluErf,
        MapFn::Relu,
        MapFn::Sigmoid,
        MapFn::Exp,
        MapFn::Cast,
        MapFn::Scale,
    ])
}

fn dtype() -> impl Strategy<Value = ElemType> {
    prop::sample::select(vec![
        BF,
        F32,
        ElemType::from(Precision::Fp8E4m3Pt),
        ElemType::from(Precision::Mxfp8E4m3),
        ElemType::from(Precision::Int4G128),
    ])
}

const ONE: SeqBatch = SeqBatch { segments: vec![] };

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn einsum_linear(m in 1u64..40, n in 1u64..40, k in 1u64..40, wdt in dtype()) {
        let k = k * 128;
        let op = Op::Einsum(EinsumAttrs { eq: "mk,nk->mn".into(), accum: None });
        check(op, vec![t(&[m, k], BF, A), t(&[n, k], wdt, W)], vec![t(&[m, n], BF, A)], &ONE, None)?;
    }

    #[test]
    fn einsum_bmm_and_view(b in 1u64..5, m in 1u64..9, n in 1u64..9, k in 1u64..9, h in 1u64..5) {
        let bmm = Op::Einsum(EinsumAttrs { eq: "bmk,bkn->bmn".into(), accum: None });
        check(bmm, vec![t(&[b, m, k], BF, A), t(&[b, k, n], BF, A)], vec![t(&[b, m, n], F32, A)], &ONE, None)?;
        let view = Op::Einsum(EinsumAttrs { eq: "thd,nhd->tn".into(), accum: None });
        check(view, vec![t(&[m, h, k], BF, A), t(&[n, h * k], BF, W)], vec![t(&[m, n], BF, A)], &ONE, None)?;
    }

    #[test]
    fn map_act_and_broadcast(r in 1u64..9, c in 1u64..9, f in fnspec(), g in fnspec(), bc in any::<bool>()) {
        let rhs = if bc { vec![c] } else { vec![r, c] };
        let op = Op::Map(MapAttrs { func: FnSpec::Steps(vec![f, g]) });
        check(op, vec![t(&[r, c], BF, A), t(&rhs, BF, W)], vec![t(&[r, c], BF, A)], &ONE, None)?;
        check(Op::Act(ActAttrs { func: f }), vec![t(&[r, c], BF, A)], vec![t(&[r, c], F32, A)], &ONE, None)?;
    }

    #[test]
    fn reduce_all_combiners(a in 1u64..6, b in 1u64..6, c in 1u64..6, axis in -3i32..3, comb in prop::sample::select(vec![ReduceKind::Sum, ReduceKind::Max, ReduceKind::Min, ReduceKind::Mean, ReduceKind::Sumsq])) {
        let s = [a, b, c];
        let ax = if axis < 0 { axis + 3 } else { axis } as usize;
        let out: Vec<u64> = s.iter().enumerate().filter(|(i, _)| *i != ax).map(|(_, &d)| d).collect();
        check(Op::Reduce(ReduceAttrs { axes: vec![axis], combiner: comb }), vec![t(&s, BF, A)], vec![t(&out, F32, A)], &ONE, None)?;
    }

    #[test]
    fn gather_scatter(v in 1u64..20, d in 1u64..9, i in 1u64..9, sum in any::<bool>()) {
        check(Op::Gather(GatherAttrs { axis: 0 }), vec![t(&[v, d], BF, W), t(&[i], I32, A)], vec![t(&[i, d], BF, A)], &ONE, None)?;
        let combine = sum.then_some(ScatterCombine::Sum);
        check(Op::Scatter(ScatterAttrs { axis: 0, combine }), vec![t(&[v, d], BF, A), t(&[i], I32, A), t(&[i, d], BF, A)], vec![t(&[v, d], BF, A)], &ONE, None)?;
    }

    #[test]
    fn norms_softmax(r in 1u64..9, d in 1u64..33, fused in any::<bool>(), scale in prop::option::of(0.1f64..2.0)) {
        let x = t(&[r, d], BF, A);
        let mut ins = vec![x.clone(), t(&[d], BF, W)];
        let mut outs = vec![x.clone()];
        if fused {
            ins.push(x.clone());
            outs.push(x.clone());
        }
        check(Op::RmsNorm(RmsNormAttrs { eps: 1e-5, fused_residual: fused }), ins, outs, &ONE, None)?;
        check(Op::LayerNorm(LayerNormAttrs { eps: 1e-5 }), vec![x.clone(), t(&[d], BF, W), t(&[d], BF, W)], vec![x.clone()], &ONE, None)?;
        check(Op::Softmax(SoftmaxAttrs { axis: -1, scale }), vec![x.clone()], vec![x], &ONE, None)?;
    }

    #[test]
    fn rope_partial_and_styles(tk in 1u64..9, h in 1u64..5, hk in 1u64..3, half in 1u64..5, extra in 0u64..3, inter in any::<bool>()) {
        let (rd, dh) = (2 * half, 2 * half + extra);
        let style = if inter { RopeStyle::Interleaved } else { RopeStyle::Half };
        let op = Op::Rope(RopeAttrs { theta: 1e4, rotary_dim: rd as u32, style, scaling: None });
        let (q, k) = (t(&[tk, h, dh], BF, A), t(&[tk, hk, dh], BF, A));
        check(op, vec![q.clone(), k.clone(), t(&[tk], I32, TensorClass::Input), t(&[64, half, 2], F32, TensorClass::Constant)], vec![q, k], &ONE, None)?;
    }

    #[test]
    fn gated_act_layouts(r in 1u64..9, f in 1u64..9, func in fnspec(), lay in prop::sample::select(vec![GateLayout::ConcatHalves, GateLayout::Interleaved, GateLayout::TwoInputs])) {
        let ins = if lay == GateLayout::TwoInputs { vec![t(&[r, f], BF, A), t(&[r, f], BF, A)] } else { vec![t(&[r, 2 * f], BF, A)] };
        check(Op::GatedAct(GatedActAttrs { func, layout: lay }), ins, vec![t(&[r, f], BF, A)], &ONE, None)?;
    }

    #[test]
    fn embedding_select_kv_append(s in seqs(), d in 1u64..9, hk in 1u64..3, e in 1u64..5, all in any::<bool>(), cvt in any::<bool>()) {
        let (tk, n) = (s.tokens(), s.seqs());
        check(Op::Embedding(Empty {}), vec![t(&[tk], I32, TensorClass::Input), t(&[50, d], BF, W)], vec![t(&[tk, d], BF, A)], &s, None)?;
        let which = if all { Which::All } else { Which::Last };
        let rows = if all { tk } else { n };
        check(Op::LogitsSelect(LogitsSelectAttrs { which, seqs: "seqs".into() }), vec![t(&[tk, d], BF, A)], vec![t(&[rows, d], BF, A)], &s, None)?;
        let cdt = if cvt { ElemType::from(Precision::Fp8E4m3Pt) } else { BF };
        let cache = t(&[n + 1, s.max_kv() + 2, hk, e], cdt, K);
        check(
            Op::KvAppend(KvAppendAttrs { seqs: "seqs".into() }),
            vec![cache.clone(), cache.clone(), t(&[tk, hk, e], BF, A), t(&[tk, hk, e], BF, A)],
            vec![cache.clone(), cache],
            &s,
            None,
        )?;
    }

    #[test]
    fn attention_masks_gqa_segments(s in seqs(), hk in 1u32..3, g in 1u32..4, dh in 1u32..5, dv in 1u32..5, mask in 0u8..4, w in 1u32..6, cap in any::<bool>()) {
        let mask = match mask { 0 => Mask::None, 1 => Mask::Causal, _ => Mask::SlidingWindow { window: w } };
        let h = hk * g;
        let a = AttnAttrs { n_heads: h, n_kv_heads: hk, head_dim: dh, v_head_dim: Some(dv), scale: None, mask, seqs: "seqs".into(), softcap: cap.then_some(30.0), sinks: false, impl_hint: AttnImpl::Auto };
        let (tk, n, kv) = (s.tokens(), s.seqs(), s.max_kv());
        let (h, hk, dh, dv) = (u64::from(h), u64::from(hk), u64::from(dh), u64::from(dv));
        check(
            Op::Attention(a),
            vec![t(&[tk, h, dh], BF, A), t(&[n, kv, hk, dh], BF, K), t(&[n, kv, hk, dv], BF, K)],
            vec![t(&[tk, h, dv], BF, A)],
            &s,
            None,
        )?;
    }

    #[test]
    fn moe_ops(tk in 1u64..12, e in 1u32..6, k in 1u32..3, d in 1u64..6, f in 1u64..6, sm in any::<bool>(), sig in any::<bool>(), norm in any::<bool>(), bias in any::<bool>(), scale in any::<bool>(), loads in prop::collection::vec(0u64..9, 5)) {
        let k = k.min(e);
        let (ku, eu) = (u64::from(k), u64::from(e));
        let route = MoeRouteAttrs {
            n_experts: e, top_k: k, scoring: if sig { Scoring::Sigmoid } else { Scoring::Softmax }, norm_topk: norm,
            softmax_after_topk: sm, group_limited: None, bias_correction: bias, routed_scaling: scale.then_some(2.5),
        };
        check(Op::MoeRoute(route), vec![t(&[tk, eu], F32, A)], vec![t(&[tk, ku], I32, A), t(&[tk, ku], F32, A)], &ONE, None)?;
        let c = 6u64;
        let disp = MoeDispatchAttrs { n_experts: e, top_k: k, capacity_factor: None, drop_policy: DropPolicy::NoDrop, layout: DispatchLayout::CapacityPadded };
        check(Op::MoeDispatch(disp), vec![t(&[tk, d], BF, A), t(&[tk, ku], I32, A)], vec![t(&[eu, c, d], BF, A)], &ONE, None)?;
        let rows = Some(MoeRows { rows: loads[..e as usize].to_vec(), drop_overflow: true });
        let ge = Op::GroupedEinsum(EinsumAttrs { eq: "ecd,end->ecn".into(), accum: None });
        check(ge, vec![t(&[eu, c, d], BF, A), t(&[eu, f, d], BF, W)], vec![t(&[eu, c, f], BF, A)], &ONE, rows)?;
        check(Op::MoeCombine(Empty {}), vec![t(&[eu, c, d], BF, A), t(&[tk, ku], I32, A), t(&[tk, ku], F32, A)], vec![t(&[tk, d], BF, A)], &ONE, None)?;
    }

    #[test]
    fn topk_and_sampling(r in 1u64..6, v in 2u64..40, k in 1u32..6, temp in prop::option::of(0.5f64..2.0), strat in 0u8..4, p in 0.1f64..0.9) {
        let k = k.min(v as u32);
        let ku = u64::from(k);
        check(Op::TopK(TopKAttrs { k, sorted: true }), vec![t(&[r, v], F32, A)], vec![t(&[r, ku], F32, A), t(&[r, ku], I32, A)], &ONE, None)?;
        let strategy = match strat { 0 => Sampling::Greedy, 1 => Sampling::TopK { k }, 2 => Sampling::TopP { p }, _ => Sampling::MinP { p } };
        check(Op::Sample(SampleAttrs { strategy, temperature: temp }), vec![t(&[r, v], F32, A)], vec![t(&[r], I32, TensorClass::Output)], &ONE, None)?;
    }

    #[test]
    fn quantize_dequantize(r in 1u64..5, blocks in 1u64..4, dynamic in any::<bool>(), target in prop::sample::select(vec![Precision::Mxfp8E4m3, Precision::Mxfp4, Precision::Nvfp4, Precision::Int8Pc, Precision::Fp8E4m3Pt, Precision::Fp8E4m3, Precision::Int4G128])) {
        let x = [r, blocks * 128];
        let q = ElemType::from(target);
        let amax_from = if dynamic { AmaxFrom::Dynamic } else { AmaxFrom::Calibrated };
        check(Op::Quantize(QuantizeAttrs { target: Some(q), amax_from }), vec![t(&x, BF, A)], vec![t(&x, q, A)], &ONE, None)?;
        check(Op::Dequantize(DequantizeAttrs { target: None }), vec![t(&x, q, W)], vec![t(&x, BF, A)], &ONE, None)?;
    }

    #[test]
    fn collectives(a in 1u64..5, b in 1u64..9, kind in prop::sample::select(vec![CollKind::AllReduce, CollKind::AllGather, CollKind::ReduceScatter, CollKind::AllToAll, CollKind::Broadcast, CollKind::Reduce]), mx in any::<bool>()) {
        let n = 4;
        let (i, o) = match kind {
            CollKind::AllGather => (vec![a, b], vec![a * n, b]),
            CollKind::ReduceScatter => (vec![a * n, b], vec![a, b]),
            _ => (vec![a, b], vec![a, b]),
        };
        let attrs = CollectiveAttrs { kind, group: Group::MeshAxes(vec!["tp".into()]), reduce: Some(if mx { ReduceOp::Max } else { ReduceOp::Sum }), axis: Some(0), root: None, algo_hint: None, ragged: None };
        check(Op::Collective(attrs), vec![t(&i, BF, A)], vec![t(&o, BF, A)], &ONE, None)?;
        check(Op::SendRecv(SendRecvAttrs { peer: Peer::Shift { axis: "pp".into(), delta: 1 }, tag: 0 }), vec![t(&i, BF, A)], vec![t(&i, BF, A)], &ONE, None)?;
    }
}

#[test]
fn opaque_and_self_einsum() {
    let cost = CostHint {
        flops_mm: 10,
        vec_ops: 3,
        transc: 1,
        convert: 0,
        bytes_in: 7,
        bytes_out: 5,
        weight_bytes: 2,
    };
    check(
        Op::Opaque(OpaqueAttrs { cost }),
        vec![t(&[4], BF, A)],
        vec![t(&[4], BF, A)],
        &ONE,
        None,
    )
    .unwrap();
    let op = Op::Einsum(EinsumAttrs {
        eq: "mk,nk->mn".into(),
        accum: None,
    });
    check_ids(
        op,
        vec![t(&[3, 5], BF, A), t(&[3, 5], BF, A)],
        vec![t(&[3, 3], BF, A)],
        &ONE,
        None,
        Some(vec!["x", "x"]),
    )
    .unwrap();
}

#[test]
fn shape_errors_are_structured() {
    let op = Op::Einsum(EinsumAttrs {
        eq: "mk,nk->mn".into(),
        accum: None,
    });
    let node = Node::new("bad", op, &["a", "b"], &["c"]);
    let ctx = NodeCtx {
        node: &node,
        inputs: vec![t(&[2, 3], BF, A), t(&[4, 5], BF, W)],
        outputs: vec![t(&[2, 4], BF, A)],
        seqs: &ONE,
        moe_rows: None,
        group_size: None,
    };
    let e = lower(&ctx).unwrap_err();
    assert_eq!(e.code, "E-WL-SHAPE-001");
    assert_eq!(e.path.as_deref(), Some("nodes.bad"));
    let node = Node::new(
        "bad",
        Op::Einsum(EinsumAttrs {
            eq: "mm,nk->mn".into(),
            accum: None,
        }),
        &["a", "b"],
        &["c"],
    );
    let ctx = NodeCtx { node: &node, ..ctx };
    assert_eq!(lower(&ctx).unwrap_err().code, "E-WL-EIN-001");
    let _ = Id::new("x").unwrap();
}

#[test]
fn repeated_operand_reads_count_once() {
    let node = Node::new("n", Op::Map(MapAttrs { func: FnSpec::One(MapFn::Add) }), &["x", "x"], &["y"]);
    let seqs = SeqBatch::uniform(1, 1, 1);
    let x = t(&[1024], BF, A);
    let ctx = NodeCtx { node: &node, inputs: vec![x.clone(), x.clone()], outputs: vec![x], seqs: &seqs, moe_rows: None, group_size: None };
    let hint = cost_hint(&ctx).unwrap();
    let cost = node_cost(&ctx, &lower(&ctx).unwrap()).unwrap();
    assert_eq!((cost.cost.bytes_in, cost.cost.bytes_out), (2048, 2048));
    assert_eq!(hint, cost.cost);
    assert_eq!(cost.traffic.iter().map(|t| t.read).sum::<u128>(), 2048);
}

#[test]
fn wide_attention_heads_keep_hint_macs() {
    let d = 32768u32;
    let a = AttnAttrs { n_heads: 1, n_kv_heads: 1, head_dim: d, v_head_dim: Some(d), scale: None, mask: Mask::None, seqs: "seqs".into(), softcap: None, sinks: false, impl_hint: AttnImpl::Auto };
    let d = u64::from(d);
    let s = SeqBatch::uniform(1, 1, 1);
    check(Op::Attention(a), vec![t(&[1, 1, d], BF, A), t(&[1, 1, 1, d], BF, K), t(&[1, 1, 1, d], BF, K)], vec![t(&[1, 1, d], BF, A)], &s, None).unwrap();
}
