//! Model zoo (02 §11.3), built-in workload sets (02 §11.4) and whole-step scenarios (02 §12.5).

use indexmap::IndexMap;
use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::precision::Precision;
use kiln_ir::wl::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Llama,
    Mixtral,
    DeepseekV3,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttnConfig {
    Gqa {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
    },
    Mla {
        heads: u32,
        q_lora_rank: Option<u32>,
        kv_lora_rank: u32,
        qk_nope_head_dim: u32,
        qk_rope_head_dim: u32,
        v_head_dim: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpConfig {
    /// Gated (`act(gate) * up`, Llama).
    Dense { d_ff: u32, act: MapFn },
    /// `down(act(up(x)))` (GPT-J, GPT-2).
    Ungated { d_ff: u32, act: MapFn },
    Moe {
        n_experts: u32,
        top_k: u32,
        d_ff_expert: u32,
        n_shared: u32,
        d_ff_shared: u32,
        scoring: Scoring,
        norm_topk: bool,
        softmax_after_topk: bool,
        group_limited: Option<GroupLimit>,
        capacity_factor: Option<DimExpr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    Rms,
    Layer,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeConfig {
    pub theta: f64,
    #[serde(default)]
    pub rotary_dim: Option<u32>,
    #[serde(default)]
    pub scaling: Option<RopeScaling>,
    #[serde(default)]
    pub style: RopeStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtypeConfig {
    pub weights: ElemType,
    pub activations: ElemType,
    pub kv_cache: ElemType,
    pub accum: Precision,
    pub logits: ElemType,
}

impl Default for DtypeConfig {
    fn default() -> Self {
        Self {
            weights: ElemType::BF16,
            activations: ElemType::BF16,
            kv_cache: ElemType::BF16,
            accum: Precision::Fp32,
            logits: ElemType::FP32,
        }
    }
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub family: Family,
    pub d_model: u32,
    pub n_layers: u32,
    pub vocab: u32,
    pub attn: AttnConfig,
    pub mlp: MlpConfig,
    #[serde(default)]
    pub first_dense_layers: u32,
    #[serde(default)]
    pub dense_d_ff: Option<u32>,
    pub norm: NormKind,
    pub norm_eps: f64,
    pub rope: RopeConfig,
    pub tie_embeddings: bool,
    pub mask: Mask,
    #[serde(default)]
    pub dtypes: DtypeConfig,
    #[serde(default = "yes")]
    pub fuse_qkv: bool,
    #[serde(default = "yes")]
    pub fuse_gate_up: bool,
    #[serde(default)]
    pub mtp_layers: u32,
    /// GPT-J block: attention and MLP both read the one pre-norm output, `y = x + attn + mlp`.
    #[serde(default)]
    pub parallel_block: bool,
    /// 02 §13.1: separate gate/up GEMMs, dense (unmasked) attention, unfused attention, and logits in the
    /// activation dtype (the harness priced every tensor as bf16).
    #[serde(default)]
    pub harness_compat: bool,
}

const LLAMA3_ROPE: RopeScaling = RopeScaling::Llama3 {
    factor: 8.0,
    low_freq_factor: 1.0,
    high_freq_factor: 4.0,
    original_max_pos: 8192,
};

fn llama3(d: u32, layers: u32, heads: u32, d_ff: u32) -> TransformerConfig {
    TransformerConfig {
        family: Family::Llama,
        d_model: d,
        n_layers: layers,
        vocab: 128_256,
        attn: AttnConfig::Gqa {
            heads,
            kv_heads: 8,
            head_dim: 128,
        },
        mlp: MlpConfig::Dense {
            d_ff,
            act: MapFn::Silu,
        },
        first_dense_layers: 0,
        dense_d_ff: None,
        norm: NormKind::Rms,
        norm_eps: 1e-5,
        rope: RopeConfig {
            theta: 500_000.0,
            rotary_dim: None,
            scaling: Some(LLAMA3_ROPE),
            style: RopeStyle::Half,
        },
        tie_embeddings: false,
        mask: Mask::Causal,
        dtypes: DtypeConfig::default(),
        fuse_qkv: true,
        fuse_gate_up: true,
        mtp_layers: 0,
        parallel_block: false,
        harness_compat: false,
    }
}

/// Llama 2 70B, `meta-llama/Llama-2-70b-hf` config.json: hidden 8192, 80 layers, 64 heads / 8 KV heads,
/// intermediate 28672, vocab 32000, rope_theta 10000 (no scaling), rms_norm_eps 1e-5, untied embeddings.
fn llama2_70b() -> TransformerConfig {
    TransformerConfig {
        vocab: 32_000,
        rope: RopeConfig {
            theta: 10_000.0,
            rotary_dim: None,
            scaling: None,
            style: RopeStyle::Half,
        },
        ..llama3(8192, 80, 64, 28_672)
    }
}

/// GPT-J 6B, `EleutherAI/gpt-j-6b` config.json: n_embd 4096, n_layer 28, n_head 16 (head_dim 256, MHA),
/// n_inner null (= 4 x 4096), activation gelu_new, layer_norm_epsilon 1e-5, rotary_dim 64 (rotate_every_two),
/// vocab_size 50400, untied lm_head, parallel attention + MLP from one LayerNorm. Bias vectors (fc_in, fc_out,
/// lm_head, 0.6 M parameters) are not modelled.
fn gptj_6b() -> TransformerConfig {
    TransformerConfig {
        vocab: 50_400,
        attn: AttnConfig::Gqa {
            heads: 16,
            kv_heads: 16,
            head_dim: 256,
        },
        mlp: MlpConfig::Ungated {
            d_ff: 16_384,
            act: MapFn::GeluTanh,
        },
        norm: NormKind::Layer,
        rope: RopeConfig {
            theta: 10_000.0,
            rotary_dim: Some(64),
            scaling: None,
            style: RopeStyle::Interleaved,
        },
        fuse_gate_up: false,
        parallel_block: true,
        ..llama3(4096, 28, 16, 16_384)
    }
}

pub const PRESETS: &[&str] = &[
    "llama3_8b",
    "llama3_1_8b",
    "llama3_70b",
    "llama2_70b",
    "gptj_6b",
];

pub fn preset(name: &str) -> Result<TransformerConfig, Diagnostic> {
    match name {
        // `llama3_8b` already carries the Llama 3.1 rope scaling; 3.1 8B has the same shapes (config.json).
        "llama3_8b" | "llama3_1_8b" => Ok(llama3(4096, 32, 32, 14_336)),
        "llama3_70b" => Ok(llama3(8192, 80, 64, 28_672)),
        "llama2_70b" => Ok(llama2_70b()),
        "gptj_6b" => Ok(gptj_6b()),
        _ => Err(
            Diagnostic::error("E-WL-ZOO-001", format!("unknown zoo preset {name:?}")).hint(
                format!(
                    "M0 presets: {}; mixtral_8x7b, deepseek_v3 and vit_l16 are deferred",
                    PRESETS.join(", ")
                ),
            ),
        ),
    }
}

/// Expands `{"zoo": {"preset", "overrides"}}` into a config (overrides are a JSON merge onto the preset).
pub fn config_from_ref(r: &ZooRef) -> Result<TransformerConfig, Diagnostic> {
    let base = preset(&r.preset)?;
    if r.overrides.is_empty() {
        return Ok(base);
    }
    let mut v = serde_json::to_value(&base).expect("config serializes");
    let obj = v.as_object_mut().expect("config is an object");
    for (k, val) in &r.overrides {
        if !obj.contains_key(k) {
            return Err(
                Diagnostic::error("E-WL-ZOO-001", format!("unknown override field {k:?}"))
                    .at("model.zoo.overrides"),
            );
        }
        obj.insert(k.clone(), val.clone());
    }
    serde_json::from_value(v).map_err(|e| {
        Diagnostic::error("E-WL-ZOO-001", format!("invalid overrides: {e}"))
            .at("model.zoo.overrides")
    })
}

/// Replaces zoo sugar by the expanded model (what gets hashed, 02 §11.3).
pub fn expand_doc(doc: &WorkloadDoc) -> Result<WorkloadDoc, Diagnostic> {
    let ModelSrc::Zoo { zoo } = &doc.model else {
        return Ok(doc.clone());
    };
    let model = build_model(&config_from_ref(zoo)?)?;
    Ok(WorkloadDoc {
        model: ModelSrc::Full(Box::new(model)),
        ..doc.clone()
    })
}

fn id(s: &str) -> Id {
    Id::new(s).expect("zoo ids are valid")
}

fn d(v: u32) -> DimExpr {
    DimExpr::int(v.into())
}

fn sym(s: &str) -> DimExpr {
    DimExpr::sym(s)
}

pub fn param_count(cfg: &TransformerConfig) -> Result<u64, Diagnostic> {
    let m = build_model(cfg)?;
    let b = crate::graph::bind(&m, &SeqBatch::uniform(1, 1, 1), &IndexMap::new())
        .map_err(|mut e| e.remove(0))?;
    crate::param_count(&m, &b)
}

/// Dense GQA decoder (Llama family), exactly the graph of 02 §14.1 (fused qkv and gate/up by default).
pub fn build_model(cfg: &TransformerConfig) -> Result<Model, Diagnostic> {
    let AttnConfig::Gqa {
        heads,
        kv_heads,
        head_dim,
    } = cfg.attn
    else {
        return Err(Diagnostic::error(
            "E-WL-ZOO-001",
            "MLA attention generation is deferred past M0",
        ));
    };
    let (d_ff, act, gated) = match cfg.mlp {
        MlpConfig::Dense { d_ff, act } => (d_ff, act, true),
        MlpConfig::Ungated { d_ff, act } => (d_ff, act, false),
        MlpConfig::Moe { .. } => {
            return Err(Diagnostic::error(
                "E-WL-ZOO-001",
                "MoE generation is deferred past M0",
            ));
        }
    };
    if cfg.mtp_layers > 0 || cfg.first_dense_layers > 0 {
        return Err(Diagnostic::error(
            "E-WL-ZOO-001",
            "only dense decoders without MTP are generated in M0",
        ));
    }
    let layer_norm = cfg.norm == NormKind::Layer;
    let parallel = cfg.parallel_block;
    let compat = cfg.harness_compat;
    let fuse_gu = gated && cfg.fuse_gate_up && !compat;
    let mask = if compat { Mask::None } else { cfg.mask.clone() };
    let impl_hint = if compat {
        AttnImpl::Unfused
    } else {
        AttnImpl::Auto
    };
    let mut dt = cfg.dtypes;
    if compat {
        dt.logits = dt.activations;
    }
    let (dm, h, hk, dh, f, v) = (cfg.d_model, heads, kv_heads, head_dim, d_ff, cfg.vocab);
    let rd = cfg.rope.rotary_dim.unwrap_or(dh);

    let mut tensors: IndexMap<Id, TensorDecl> = IndexMap::new();
    let mut weight = |name: &str, shape: Vec<DimExpr>, stacked: bool| {
        let t = TensorDecl::new(shape, dt.weights, TensorClass::Weight);
        tensors.insert(id(name), if stacked { t.stacked(sym("L")) } else { t });
    };
    weight("w_embed", vec![d(v), d(dm)], false);
    weight("w_attn_norm", vec![d(dm)], true);
    if layer_norm {
        weight("b_attn_norm", vec![d(dm)], true);
    }
    if cfg.fuse_qkv {
        weight("w_qkv", vec![d((h + 2 * hk) * dh), d(dm)], true);
    } else {
        weight("w_q", vec![d(h * dh), d(dm)], true);
        weight("w_k", vec![d(hk * dh), d(dm)], true);
        weight("w_v", vec![d(hk * dh), d(dm)], true);
    }
    weight("w_o", vec![d(dm), d(h * dh)], true);
    if !parallel {
        weight("w_mlp_norm", vec![d(dm)], true);
        if layer_norm {
            weight("b_mlp_norm", vec![d(dm)], true);
        }
    }
    if fuse_gu {
        weight("w_gate_up", vec![d(2 * f), d(dm)], true);
    } else {
        if gated {
            weight("w_gate", vec![d(f), d(dm)], true);
        }
        weight("w_up", vec![d(f), d(dm)], true);
    }
    weight("w_down", vec![d(dm), d(f)], true);
    weight("w_final_norm", vec![d(dm)], false);
    if layer_norm {
        weight("b_final_norm", vec![d(dm)], false);
    }
    if !cfg.tie_embeddings {
        weight("w_lm", vec![d(v), d(dm)], false);
    }
    tensors.insert(
        id("rope_tab"),
        TensorDecl::new(
            [sym("kv_cap"), d(rd / 2), d(2)],
            ElemType::FP32,
            TensorClass::Constant,
        ),
    );
    let cache = [sym("slots"), sym("kv_cap"), d(hk), d(dh)];
    for c in ["kv_k", "kv_v"] {
        tensors.insert(
            id(c),
            TensorDecl::new(cache.clone(), dt.kv_cache, TensorClass::KvCache).stacked(sym("L")),
        );
    }

    let act_t =
        |shape: Vec<DimExpr>| TensorDecl::new(shape, dt.activations, TensorClass::Activation);
    let td = |n: u32| act_t(vec![sym("T"), d(n)]);
    let thd = |n: u32| act_t(vec![sym("T"), d(n), d(dh)]);

    let mut bt: IndexMap<Id, TensorDecl> = IndexMap::new();
    let mut body = Vec::new();
    let mut stacked: Vec<(&str, &str)> = vec![("w_attn_norm", "wn1")];
    if layer_norm {
        stacked.push(("b_attn_norm", "bn1"));
    }
    let norm = |eps: f64| {
        if layer_norm {
            Op::LayerNorm(LayerNormAttrs { eps })
        } else {
            Op::RmsNorm(RmsNormAttrs {
                eps,
                fused_residual: false,
            })
        }
    };
    let norm_in = |x: &'static str, w: &'static str, b: &'static str| -> Vec<&'static str> {
        if layer_norm {
            vec![x, w, b]
        } else {
            vec![x, w]
        }
    };
    // `None` is the lowering default (fp32 for float activations, int32 for integer), kept so default models hash unchanged.
    let default_accum = if dt.activations.is_float() { Precision::Fp32 } else { Precision::Int32 };
    let layer_accum = (dt.accum != default_accum).then_some(dt.accum);
    let einsum = |eq: &str| {
        Op::Einsum(EinsumAttrs {
            eq: eq.into(),
            accum: layer_accum,
        })
    };
    bt.insert(id("xn"), td(dm));
    body.push(
        Node::new(
            "attn_norm",
            norm(cfg.norm_eps),
            &norm_in("x", "wn1", "bn1"),
            &["xn"],
        )
        .role("attn.norm"),
    );
    for (n, s) in [("q", h), ("k", hk), ("v", hk)] {
        bt.insert(id(n), thd(s));
    }
    if cfg.fuse_qkv {
        stacked.push(("w_qkv", "wqkv"));
        bt.insert(id("qkv"), td((h + 2 * hk) * dh));
        body.push(
            Node::new("qkv", einsum("td,nd->tn"), &["xn", "wqkv"], &["qkv"]).role("attn.qkv"),
        );
        let split = LayoutAttrs {
            kind: LayoutKind::Split,
            perm: None,
            axis: Some(1),
            start: None,
            len: None,
            stride: None,
            sizes: Some(vec![d(h * dh), d(hk * dh), d(hk * dh)]),
            reshape: Some(vec![
                vec![sym("T"), d(h), d(dh)],
                vec![sym("T"), d(hk), d(dh)],
                vec![sym("T"), d(hk), d(dh)],
            ]),
        };
        body.push(Node::new(
            "split",
            Op::Layout(split),
            &["qkv"],
            &["q", "k", "v"],
        ));
    } else {
        for (n, outer, w, r) in [
            ("q", "w_q", "wq", "attn.q"),
            ("k", "w_k", "wk", "attn.k"),
            ("v", "w_v", "wv", "attn.v"),
        ] {
            stacked.push((outer, w));
            body.push(
                Node::new(
                    &format!("{n}_proj"),
                    einsum("td,hed->the"),
                    &["xn", w],
                    &[n],
                )
                .role(r),
            );
        }
    }
    bt.insert(id("qr"), thd(h));
    bt.insert(id("kr"), thd(hk));
    let rope = RopeAttrs {
        theta: cfg.rope.theta,
        rotary_dim: rd,
        style: cfg.rope.style,
        scaling: cfg.rope.scaling.clone(),
    };
    body.push(
        Node::new(
            "rope",
            Op::Rope(rope),
            &["q", "k", "pos", "rope_tab"],
            &["qr", "kr"],
        )
        .role("attn.rope"),
    );
    for (c, p) in [("ck1", "ck"), ("cv1", "cv")] {
        bt.insert(
            id(c),
            TensorDecl::new(cache.clone(), dt.kv_cache, TensorClass::KvCache).alias(id(p)),
        );
    }
    body.push(
        Node::new(
            "kv_append",
            Op::KvAppend(KvAppendAttrs {
                seqs: "seqs".into(),
            }),
            &["ck", "cv", "kr", "v"],
            &["ck1", "cv1"],
        )
        .role("attn.kv_append"),
    );
    let attn = AttnAttrs {
        n_heads: h,
        n_kv_heads: hk,
        head_dim: dh,
        v_head_dim: None,
        scale: None,
        mask,
        seqs: "seqs".into(),
        softcap: None,
        sinks: false,
        impl_hint,
    };
    bt.insert(id("o"), thd(h));
    body.push(
        Node::new("attn", Op::Attention(attn), &["qr", "ck1", "cv1"], &["o"]).role("attn.core"),
    );
    stacked.push(("w_o", "wo"));
    for n in ["a", "h", "hn", "dn", "y"] {
        if !(parallel && n == "hn") {
            bt.insert(id(n), td(dm));
        }
    }
    body.push(Node::new("o_proj", einsum("thd,nhd->tn"), &["o", "wo"], &["a"]).role("attn.o"));
    let add = || {
        Op::Map(MapAttrs {
            func: FnSpec::One(MapFn::Add),
        })
    };
    body.push(Node::new("res1", add(), &["x", "a"], &["h"]).role("residual"));
    let mlp_in = if parallel {
        "xn"
    } else {
        stacked.push(("w_mlp_norm", "wn2"));
        if layer_norm {
            stacked.push(("b_mlp_norm", "bn2"));
        }
        body.push(
            Node::new(
                "mlp_norm",
                norm(cfg.norm_eps),
                &norm_in("h", "wn2", "bn2"),
                &["hn"],
            )
            .role("mlp.norm"),
        );
        "hn"
    };
    bt.insert(id("m"), td(f));
    if fuse_gu {
        stacked.push(("w_gate_up", "wgu"));
        bt.insert(id("gu"), td(2 * f));
        body.push(
            Node::new("gate_up", einsum("td,nd->tn"), &[mlp_in, "wgu"], &["gu"])
                .role("mlp.gate_up"),
        );
        body.push(
            Node::new(
                "act",
                Op::GatedAct(GatedActAttrs {
                    func: act,
                    layout: GateLayout::ConcatHalves,
                }),
                &["gu"],
                &["m"],
            )
            .role("mlp.act"),
        );
    } else if !gated {
        stacked.push(("w_up", "wu"));
        bt.insert(id("u"), td(f));
        body.push(Node::new("up", einsum("td,nd->tn"), &[mlp_in, "wu"], &["u"]).role("mlp.up"));
        body.push(
            Node::new("act", Op::Act(ActAttrs { func: act }), &["u"], &["m"]).role("mlp.act"),
        );
    } else {
        stacked.extend([("w_gate", "wg"), ("w_up", "wu")]);
        bt.insert(id("g"), td(f));
        bt.insert(id("u"), td(f));
        body.push(Node::new("gate", einsum("td,nd->tn"), &[mlp_in, "wg"], &["g"]).role("mlp.gate"));
        body.push(Node::new("up", einsum("td,nd->tn"), &[mlp_in, "wu"], &["u"]).role("mlp.up"));
        body.push(
            Node::new(
                "act",
                Op::GatedAct(GatedActAttrs {
                    func: act,
                    layout: GateLayout::TwoInputs,
                }),
                &["g", "u"],
                &["m"],
            )
            .role("mlp.act"),
        );
    }
    stacked.push(("w_down", "wd"));
    body.push(Node::new("down", einsum("tf,nf->tn"), &["m", "wd"], &["dn"]).role("mlp.down"));
    body.push(Node::new("res2", add(), &["h", "dn"], &["y"]).role("residual"));
    stacked.extend([("kv_k", "ck"), ("kv_v", "cv")]);

    let mut params = vec![id("x"), id("pos"), id("rope_tab")];
    params.extend(stacked.iter().map(|(_, p)| id(p)));
    let block = Graph {
        params,
        results: vec![id("y"), id("ck1"), id("cv1")],
        tensors: bt,
        nodes: body,
        regions: IndexMap::new(),
    };

    let mut ft: IndexMap<Id, TensorDecl> = IndexMap::new();
    ft.insert(
        id("ids"),
        TensorDecl::new([sym("T")], ElemType::INT32, TensorClass::Input),
    );
    ft.insert(
        id("pos"),
        TensorDecl::new([sym("T")], ElemType::INT32, TensorClass::Input),
    );
    for n in ["h0", "hl", "hf"] {
        ft.insert(id(n), td(dm));
    }
    ft.insert(id("hs"), act_t(vec![sym("N"), d(dm)]));
    ft.insert(
        id("logits"),
        TensorDecl::new([sym("N"), d(v)], dt.logits, TensorClass::Activation),
    );
    ft.insert(
        id("next"),
        TensorDecl::new([sym("N")], ElemType::INT32, TensorClass::Output),
    );
    let repeat = RepeatAttrs {
        body: id("block"),
        count: sym("L"),
        carry: vec![Carry {
            init: id("h0"),
            param: id("x"),
            yield_: id("y"),
            out: id("hl"),
        }],
        stacked: stacked
            .iter()
            .map(|(o, p)| Stacked {
                outer: id(o),
                param: id(p),
            })
            .collect(),
        broadcast: vec![(id("pos"), id("pos")), (id("rope_tab"), id("rope_tab"))],
    };
    let lm_w = if cfg.tie_embeddings {
        "w_embed"
    } else {
        "w_lm"
    };
    let lm = Op::Einsum(EinsumAttrs {
        eq: "nd,vd->nv".into(),
        accum: Some(dt.accum),
    });
    let fwd = Graph {
        params: vec![id("ids"), id("pos")],
        results: vec![id("next")],
        tensors: ft,
        nodes: vec![
            Node::new(
                "embed",
                Op::Embedding(Empty {}),
                &["ids", "w_embed"],
                &["h0"],
            )
            .role("embed"),
            Node::new("layers", Op::Repeat(repeat), &["h0", "pos"], &["hl"]),
            Node::new(
                "final_norm",
                norm(cfg.norm_eps),
                &norm_in("hl", "w_final_norm", "b_final_norm"),
                &["hf"],
            )
            .role("final_norm"),
            Node::new(
                "select",
                Op::LogitsSelect(LogitsSelectAttrs {
                    which: Which::Last,
                    seqs: "seqs".into(),
                }),
                &["hf"],
                &["hs"],
            )
            .role("logits_select"),
            Node::new("lm_head", lm, &["hs", lm_w], &["logits"]).role("lm_head"),
            Node::new(
                "sample",
                Op::Sample(SampleAttrs {
                    strategy: Strategy::Greedy,
                    temperature: None,
                }),
                &["logits"],
                &["next"],
            )
            .role("sample"),
        ],
        regions: IndexMap::new(),
    };

    let mut symbols = IndexMap::new();
    symbols.insert("seqs".to_string(), SymbolDecl::segments());
    for s in ["T", "N", "kv_cap", "slots"] {
        symbols.insert(s.to_string(), SymbolDecl::size(None));
    }
    symbols.insert("L".to_string(), SymbolDecl::size(Some(cfg.n_layers.into())));
    let mut graphs = IndexMap::new();
    graphs.insert(id("fwd"), fwd);
    graphs.insert(id("block"), block);
    Ok(Model {
        symbols,
        graphs,
        entry: EntryPoints { forward: id("fwd") },
        tensors,
        routing: None,
    })
}

/// A single-einsum workload `y[m, n] = x[m, k] · w[n, k]` (smoke suite, GEMM sweeps).
pub fn gemm_model(m: u64, n: u64, k: u64) -> Model {
    let mut tensors = IndexMap::new();
    tensors.insert(
        id("w"),
        TensorDecl::new(
            [DimExpr::int(n), DimExpr::int(k)],
            ElemType::BF16,
            TensorClass::Weight,
        ),
    );
    let mut gt = IndexMap::new();
    gt.insert(
        id("x"),
        TensorDecl::new(
            [DimExpr::int(m), DimExpr::int(k)],
            ElemType::BF16,
            TensorClass::Input,
        ),
    );
    gt.insert(
        id("y"),
        TensorDecl::new(
            [DimExpr::int(m), DimExpr::int(n)],
            ElemType::BF16,
            TensorClass::Output,
        ),
    );
    let node = Node::new(
        "gemm",
        Op::Einsum(EinsumAttrs {
            eq: "mk,nk->mn".into(),
            accum: None,
        }),
        &["x", "w"],
        &["y"],
    );
    let mut graphs = IndexMap::new();
    graphs.insert(
        id("main"),
        Graph {
            params: vec![id("x")],
            results: vec![id("y")],
            tensors: gt,
            nodes: vec![node],
            regions: IndexMap::new(),
        },
    );
    let mut symbols = IndexMap::new();
    symbols.insert("seqs".to_string(), SymbolDecl::segments());
    Model {
        symbols,
        graphs,
        entry: EntryPoints {
            forward: id("main"),
        },
        tensors,
        routing: None,
    }
}

/// The scored unit (02 D9, §12.5): one whole step, end to end, in whole-graph mode.
pub fn whole_step(kind: PhaseKind, seqs: SeqBatch) -> Scenario {
    let mut s = Scenario::snapshot(kind, seqs);
    s.bindings.insert("kv_cap".into(), s_max_kv(&s));
    s
}

fn s_max_kv(s: &Scenario) -> u64 {
    match &s.mode {
        ScenarioMode::Snapshot { seqs, .. } => seqs.max_kv(),
        _ => 0,
    }
}

/// Scenario names: `prefill_b{B}[_s{S}]` (S = 2048), `decode_b{B}[_kv{K}]` (K = 2048),
/// `static_b{B}_p{P}_g{G}`.
pub fn scenario(name: &str) -> Result<Scenario, Diagnostic> {
    let bad = || {
        Diagnostic::error("E-WL-SCN-001", format!("unknown scenario name {name:?}"))
            .hint("e.g. prefill_b1, decode_b8, decode_b1_kv32768, static_b8_p1024_g256")
    };
    let parts: Vec<&str> = name.split('_').collect();
    let num = |p: &str, pre: &str| {
        p.strip_prefix(pre)
            .and_then(|x| x.parse::<u64>().ok())
            .filter(|&x| x > 0)
    };
    match parts.as_slice() {
        ["prefill", b, rest @ ..] => {
            let b = num(b, "b").ok_or_else(bad)?;
            let s = match rest {
                [] => 2048,
                [s] => num(s, "s").ok_or_else(bad)?,
                _ => return Err(bad()),
            };
            Ok(whole_step(PhaseKind::Prefill, SeqBatch::uniform(b, s, s)))
        }
        ["decode", b, rest @ ..] => {
            let b = num(b, "b").ok_or_else(bad)?;
            let kv = match rest {
                [] => 2048,
                [k] => num(k, "kv").ok_or_else(bad)?,
                _ => return Err(bad()),
            };
            Ok(whole_step(PhaseKind::Decode, SeqBatch::uniform(b, 1, kv)))
        }
        ["static", b, p, g] => {
            let (batch, prompt_len, gen_len) = (
                num(b, "b").ok_or_else(bad)?,
                num(p, "p").ok_or_else(bad)?,
                num(g, "g").ok_or_else(bad)?,
            );
            let mut s = Scenario::snapshot(PhaseKind::Prefill, SeqBatch::uniform(1, 1, 1));
            s.mode = ScenarioMode::Static {
                batch,
                prompt_len,
                gen_len,
                prefix_cached: 0,
                decode_sampling: DecodeSampling::default(),
            };
            Ok(s)
        }
        _ => Err(bad()),
    }
}

#[derive(Clone, Debug)]
pub struct SuiteMember {
    /// `"<preset>:<scenario>"`.
    pub name: String,
    pub doc: WorkloadDoc,
    pub scenario: Id,
}

impl SuiteMember {
    pub fn model(&self) -> &Model {
        self.doc
            .expanded()
            .expect("suite members carry expanded models")
    }

    pub fn scenario(&self) -> &Scenario {
        &self.doc.scenarios[&self.scenario]
    }
}

/// One workload `"<preset>:<scenario>"` in standard (whole-step) form, with optional storage dtypes
/// `+weights=<dtype>` (every weight tensor), `+kv=<dtype>` (the KV cache) and `+acts=<dtype>` (every
/// activation, so GEMM inputs too: W8A8 with the quantize fused into each producer), e.g.
/// `llama3_8b:decode_b8+weights=fp8_e4m3+kv=fp8_e4m3_pt`. Dtypes are registry names (01 §6.1), including MX
/// and scaled variants (`mxfp4`, `int4_g128`, `int8_pc`).
pub fn workload(name: &str) -> Result<SuiteMember, Diagnostic> {
    member(name, false)
}

/// Applies `+weights=` / `+kv=` / `+acts=` suffixes of a workload name to `cfg`; returns the name without them.
pub fn apply_dtype_suffixes<'a>(name: &'a str, cfg: &mut TransformerConfig) -> Result<&'a str, Diagnostic> {
    let mut parts = name.split('+');
    let base = parts.next().unwrap_or_default();
    for suffix in parts {
        let bad = |msg: String| {
            Diagnostic::error("E-WL-SCN-001", format!("{name}: {msg}"))
                .hint("suffixes are +weights=<dtype>, +kv=<dtype> and +acts=<dtype>, e.g. +weights=fp8_e4m3")
        };
        let (k, v) = suffix
            .split_once('=')
            .ok_or_else(|| bad(format!("suffix {suffix:?} is not key=dtype")))?;
        let spec: kiln_ir::precision::PrecisionSpec =
            v.parse().map_err(|e: Diagnostic| bad(format!("dtype {v:?}: {}", e.message)))?;
        let t = ElemType::from_spec(spec);
        t.check().map_err(|e| bad(e.message))?;
        match k {
            "weights" => cfg.dtypes.weights = t,
            "kv" => cfg.dtypes.kv_cache = t,
            "acts" => cfg.dtypes.activations = t,
            _ => return Err(bad(format!("unknown suffix {k:?}"))),
        }
    }
    Ok(base)
}

fn member(name: &str, compat: bool) -> Result<SuiteMember, Diagnostic> {
    let mut cfg = preset(name.split_once(':').map_or(name, |x| x.0))?;
    let base = apply_dtype_suffixes(name, &mut cfg)?;
    let (p, s) = base.split_once(':').ok_or_else(|| {
        Diagnostic::error(
            "E-WL-SCN-001",
            format!("workload name {name:?} is not <preset>:<scenario>"),
        )
    })?;
    cfg.harness_compat = compat;
    let mut sc = scenario(s)?;
    if compat {
        sc.eval_mode = EvalMode::Isolated {
            cache: CacheState::Cold,
            launch: false,
        };
    }
    let sid = Id::new(s).map_err(|e| e.at("scenario"))?;
    let mut doc = WorkloadDoc::new(
        Id::new(p).expect("preset names are ids"),
        build_model(&cfg)?,
    );
    doc.scenarios.insert(sid.clone(), sc);
    Ok(SuiteMember {
        name: name.into(),
        doc,
        scenario: sid,
    })
}

pub const LLAMA_SNAPSHOTS: [&str; 4] = ["prefill_b1", "decode_b1", "decode_b8", "decode_b32"];

/// Built-in sets of 02 §11.4. `legacy` and `standard` are complete (and take the `+weights=` / `+kv=` suffixes
/// of [`workload`], applied to every member: `standard+weights=fp8_e4m3`); `smoke` is the two single-GEMM
/// graphs; `evolve` and `heldout` need Mixtral/ViT/TP members deferred past M0.
pub fn suite(name: &str) -> Result<Vec<SuiteMember>, Diagnostic> {
    let (base, suffix) = name.find('+').map_or((name, ""), |i| name.split_at(i));
    match base {
        "legacy" | "standard" => LLAMA_SNAPSHOTS
            .iter()
            .map(|s| member(&format!("llama3_8b:{s}{suffix}"), base == "legacy"))
            .collect(),
        _ if !suffix.is_empty() => Err(Diagnostic::error(
            "E-WL-SCN-001",
            format!("suite {base:?} takes no dtype suffixes"),
        )),
        "smoke" => Ok([
            ("gemm_1024", 1024, 1024, 1024),
            ("gemm_16_8192", 16, 8192, 8192),
        ]
        .into_iter()
        .map(|(n, m, nn, k)| {
            let mut doc = WorkloadDoc::new(id(n), gemm_model(m, nn, k));
            let sid = id("step");
            let mut s = Scenario::snapshot(PhaseKind::Custom, SeqBatch::uniform(m, 1, 1));
            s.eval_mode = EvalMode::Isolated {
                cache: CacheState::Cold,
                launch: false,
            };
            doc.scenarios.insert(sid.clone(), s);
            SuiteMember {
                name: n.into(),
                doc,
                scenario: sid,
            }
        })
        .collect()),
        "evolve" | "heldout" => Err(Diagnostic::error(
            "E-WL-SCN-001",
            format!("suite {name:?} is not available in M0"),
        )
        .hint("it needs Mixtral, ViT and TP members (deferred)")),
        _ => Err(
            Diagnostic::error("E-WL-SCN-001", format!("unknown suite {name:?}"))
                .hint("legacy, standard, smoke, evolve, heldout"),
        ),
    }
}
