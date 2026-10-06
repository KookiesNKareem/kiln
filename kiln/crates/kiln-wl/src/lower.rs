//! Per-node lowering to affine kernels (02 §4-§6) and the matching closed-form cost hints (02 §5.1).
//!
//! Invariant (tested over shape corpora): `cost_hint(ctx) == count::node_cost(ctx, &lower(ctx)?)`.

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::precision::Precision;
use kiln_ir::wl::*;

/// Everything a node needs to lower: bound operand types and step-level context.
#[derive(Clone, Debug)]
pub struct NodeCtx<'a> {
    pub node: &'a Node,
    pub inputs: Vec<TypeInfo>,
    pub outputs: Vec<TypeInfo>,
    pub seqs: &'a SeqBatch,
    /// Kept rows per expert for `grouped_einsum` (from the routing model); `None` = every row.
    pub moe_rows: Option<MoeRows>,
    /// Group size for collectives with `MeshAxes` groups (from the plan's mesh).
    pub group_size: Option<u64>,
}

/// Rows routed to each expert, and whether the dispatch may drop rows beyond the declared capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeRows {
    pub rows: Vec<u64>,
    pub drop_overflow: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lowered {
    pub kernels: Vec<Kernel>,
    /// Node-internal tensors (scores, row statistics); never part of compulsory bytes.
    pub temps: Vec<(Id, TypeInfo)>,
}

pub(crate) const Z: ScalarBody = ScalarBody {
    mac: 0,
    add: 0,
    mul: 0,
    fma: 0,
    max: 0,
    cmp: 0,
    select: 0,
    exp: 0,
    log: 0,
    rcp: 0,
    rsqrt: 0,
    tanh: 0,
    erf: 0,
    sin_cos: 0,
    cvt: 0,
};

pub(crate) fn err(code: &str, ctx: &NodeCtx, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(code, msg).at(format!("nodes.{}", ctx.node.id))
}

fn shape_err(ctx: &NodeCtx, what: &str, expected: &[u64], found: &[u64]) -> Diagnostic {
    err(
        "E-WL-SHAPE-001",
        ctx,
        format!("{what}: expected shape {expected:?}, found {found:?}"),
    )
}

fn unsupported(ctx: &NodeCtx, what: &str) -> Diagnostic {
    err(
        "E-WL-OP-001",
        ctx,
        format!(
            "{} {what} is not supported by the M0 lowering",
            ctx.node.op.name()
        ),
    )
}

fn numel(s: &[u64]) -> u128 {
    s.iter().map(|&d| u128::from(d)).product()
}

fn arity(ctx: &NodeCtx, nin: usize, nout: usize) -> Result<(), Diagnostic> {
    if ctx.inputs.len() != nin || ctx.outputs.len() != nout {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!(
                "expected {nin} inputs and {nout} outputs, found {} and {}",
                ctx.inputs.len(),
                ctx.outputs.len()
            ),
        ));
    }
    Ok(())
}

fn expect_out(ctx: &NodeCtx, i: usize, shape: &[u64]) -> Result<(), Diagnostic> {
    let found = &ctx.outputs[i].shape;
    if found.as_slice() != shape {
        return Err(shape_err(
            ctx,
            &format!("output {}", ctx.node.outputs[i]),
            shape,
            found,
        ));
    }
    Ok(())
}

fn rows_cols(s: &[u64]) -> (u64, u64) {
    let cols = s.last().copied().unwrap_or(1);
    (s.iter().rev().skip(1).product(), cols)
}

fn ceil_log2(k: u32) -> u16 {
    (32 - k.saturating_sub(1).leading_zeros()) as u16
}

pub(crate) fn fn_body(f: MapFn) -> ScalarBody {
    match f {
        MapFn::Add | MapFn::Sub => ScalarBody { add: 1, ..Z },
        MapFn::Mul | MapFn::Scale => ScalarBody { mul: 1, ..Z },
        MapFn::Div => ScalarBody {
            mul: 1,
            rcp: 1,
            ..Z
        },
        MapFn::Silu => ScalarBody {
            exp: 1,
            add: 1,
            rcp: 1,
            mul: 1,
            ..Z
        },
        MapFn::Sigmoid => ScalarBody {
            exp: 1,
            add: 1,
            rcp: 1,
            ..Z
        },
        MapFn::GeluTanh => ScalarBody {
            tanh: 1,
            mul: 4,
            add: 2,
            ..Z
        },
        MapFn::GeluErf => ScalarBody {
            erf: 1,
            mul: 2,
            add: 1,
            ..Z
        },
        MapFn::Relu => ScalarBody { max: 1, ..Z },
        MapFn::Exp => ScalarBody { exp: 1, ..Z },
        MapFn::Cast => ScalarBody { cvt: 1, ..Z },
    }
}

// ---------------------------------------------------------------- kernel construction helpers

pub(crate) fn ix(dims: &[&str]) -> Vec<IndexExpr> {
    dims.iter().map(|d| IndexExpr::dim(d)).collect()
}

fn konst(v: i64) -> IndexExpr {
    IndexExpr::Affine {
        terms: vec![],
        offset: v,
    }
}

fn term(coeff: i64, dim: Option<&str>, param: Option<&str>) -> Term {
    Term {
        coeff,
        dim: dim.map(Into::into),
        param: param.map(Into::into),
    }
}

/// Packed token row of local query `q` of sequence `s` in its segment.
fn tok(s: &str, q: Option<&str>) -> IndexExpr {
    let mut t = vec![
        term(1, None, Some("tok_base")),
        term(1, Some(s), Some("q_len")),
    ];
    if let Some(q) = q {
        t.push(term(1, Some(q), None));
    }
    IndexExpr::terms(t, 0)
}

fn slot(s: &str) -> IndexExpr {
    IndexExpr::terms(
        vec![term(1, None, Some("slot_base")), term(1, Some(s), None)],
        0,
    )
}

fn kernel(class: KernelClass, dims: &[(&str, u64)], red: &[&str]) -> Kernel {
    Kernel {
        id: String::new(),
        dims: dims
            .iter()
            .map(|&(n, e)| LoopDim {
                name: n.into(),
                extent: e,
                kind: if red.contains(&n) {
                    DimKind::Reduction
                } else {
                    DimKind::Parallel
                },
            })
            .collect(),
        domain: Domain::Box,
        operands: vec![],
        body: Z,
        combine: None,
        accum: None,
        class,
        opaque_cost: None,
    }
}

trait KernelExt: Sized {
    fn op(self, t: &Id, a: Access, idx: Vec<IndexExpr>) -> Self;
    fn rd(self, t: &Id, idx: Vec<IndexExpr>) -> Self {
        self.op(t, Access::Read, idx)
    }
    fn wr(self, t: &Id, idx: Vec<IndexExpr>) -> Self {
        self.op(t, Access::Write, idx)
    }
    fn rw(self, t: &Id, idx: Vec<IndexExpr>) -> Self {
        self.op(t, Access::ReadWrite, idx)
    }
    fn body(self, b: ScalarBody) -> Self;
    fn comb(self, c: Combiner) -> Self;
    fn dom(self, d: Domain) -> Self;
}

impl KernelExt for Kernel {
    fn op(mut self, t: &Id, a: Access, idx: Vec<IndexExpr>) -> Self {
        self.operands.push(Operand::new(t, a, idx));
        self
    }
    fn body(mut self, b: ScalarBody) -> Self {
        self.body = b;
        self
    }
    fn comb(mut self, c: Combiner) -> Self {
        self.combine = Some(c);
        self
    }
    fn dom(mut self, d: Domain) -> Self {
        self.domain = d;
        self
    }
}

struct Kb<'c> {
    node: &'c str,
    out: Lowered,
}

impl<'c> Kb<'c> {
    fn new(ctx: &'c NodeCtx) -> Self {
        Self {
            node: ctx.node.id.as_str(),
            out: Lowered::default(),
        }
    }

    fn temp(&mut self, name: &str, shape: Vec<u64>, dtype: ElemType) -> Id {
        let id = Id::new(format!("{}.{name}", self.node))
            .expect("node ids and temp names are valid ids");
        self.out.temps.push((
            id.clone(),
            TypeInfo::new(shape, dtype, TensorClass::Activation),
        ));
        id
    }

    fn push(&mut self, mut k: Kernel) {
        k.id = format!("{}.k{}", self.node, self.out.kernels.len());
        self.out.kernels.push(k);
    }

    fn done(self) -> Lowered {
        self.out
    }
}

/// Per-segment domains with the standard params `tok_base`, `q_len`, `slot_base`, `past`.
fn segmented(
    seqs: &SeqBatch,
    seg_dim: &str,
    mut f: impl FnMut(&Segment) -> (Vec<(&'static str, u64)>, Vec<DiffConstraint>),
) -> Domain {
    let (mut tok_base, mut slot_base) = (0i64, 0i64);
    let segments = seqs
        .segments
        .iter()
        .map(|s| {
            let (ext, constraints) = f(s);
            let mut extents = vec![(seg_dim.to_string(), s.count)];
            extents.extend(ext.into_iter().map(|(n, e)| (n.to_string(), e)));
            let params = vec![
                ("tok_base".into(), tok_base),
                ("q_len".into(), s.q_len as i64),
                ("slot_base".into(), slot_base),
                ("past".into(), s.past() as i64),
            ];
            tok_base += (s.count * s.q_len) as i64;
            slot_base += s.count as i64;
            SegmentDomain {
                extents,
                params,
                constraints,
            }
        })
        .collect();
    Domain::Segmented {
        seg_dim: seg_dim.into(),
        segments,
    }
}

fn bytes(t: &TypeInfo, elems: u128) -> u128 {
    t.bytes(elems)
}

/// Hint builder: inputs/outputs by position with element counts.
struct H<'c> {
    ctx: &'c NodeCtx<'c>,
    h: CostHint,
}

impl<'c> H<'c> {
    fn new(ctx: &'c NodeCtx<'c>) -> Self {
        Self {
            ctx,
            h: CostHint::default(),
        }
    }
    fn input(mut self, i: usize, elems: u128) -> Self {
        let names = &self.ctx.node.inputs;
        if names[..i].contains(&names[i]) {
            return self;
        }
        let t = &self.ctx.inputs[i];
        let b = bytes(t, elems);
        self.h.bytes_in += b;
        if t.class == TensorClass::Weight {
            self.h.weight_bytes += b;
        }
        self
    }
    fn output(mut self, i: usize, elems: u128) -> Self {
        let names = &self.ctx.node.outputs;
        if names[..i].contains(&names[i]) {
            return self;
        }
        self.h.bytes_out += bytes(&self.ctx.outputs[i], elems);
        self
    }
    fn work(mut self, points: u128, b: ScalarBody) -> Self {
        self.h.vec_ops += points * u128::from(b.vector());
        self.h.transc += points * u128::from(b.transcendental());
        self.h.convert += points * u128::from(b.cvt);
        self.h.flops_mm += 2 * points * u128::from(b.mac);
        self
    }
    fn all_inputs(mut self) -> Self {
        for i in 0..self.ctx.inputs.len() {
            let n = self.ctx.inputs[i].numel();
            self = self.input(i, n);
        }
        self
    }
    fn all_outputs(mut self) -> Self {
        for i in 0..self.ctx.outputs.len() {
            let n = self.ctx.outputs[i].numel();
            self = self.output(i, n);
        }
        self
    }
}

// ---------------------------------------------------------------- dispatch

pub fn lower(ctx: &NodeCtx) -> Result<Lowered, Diagnostic> {
    match &ctx.node.op {
        Op::Einsum(a) => einsum(ctx, a).map(|(l, _)| l),
        Op::Map(a) => map(ctx, a.func.steps()).map(|(l, _)| l),
        Op::Act(a) => map(ctx, &[a.func]).map(|(l, _)| l),
        Op::Reduce(a) => reduce(ctx, a).map(|(l, _)| l),
        Op::Gather(a) => gather(ctx, a).map(|(l, _)| l),
        Op::Scatter(a) => scatter(ctx, a).map(|(l, _)| l),
        Op::Layout(a) => layout(ctx, a).map(|(l, _)| l),
        Op::Attention(a) => attention(ctx, a).map(|(l, _)| l),
        Op::Softmax(a) => softmax_op(ctx, a).map(|(l, _)| l),
        Op::RmsNorm(a) => rms_norm(ctx, a).map(|(l, _)| l),
        Op::LayerNorm(_) => layer_norm(ctx).map(|(l, _)| l),
        Op::Rope(a) => rope(ctx, a).map(|(l, _)| l),
        Op::GatedAct(a) => gated_act(ctx, a).map(|(l, _)| l),
        Op::Embedding(_) => embedding(ctx).map(|(l, _)| l),
        Op::LogitsSelect(a) => logits_select(ctx, a).map(|(l, _)| l),
        Op::KvAppend(_) => kv_append(ctx).map(|(l, _)| l),
        Op::MoeRoute(a) => moe_route(ctx, a).map(|(l, _)| l),
        Op::MoeDispatch(a) => moe_dispatch(ctx, a).map(|(l, _)| l),
        Op::GroupedEinsum(a) => grouped_einsum(ctx, a).map(|(l, _)| l),
        Op::MoeCombine(_) => moe_combine(ctx).map(|(l, _)| l),
        Op::TopK(a) => top_k(ctx, a).map(|(l, _)| l),
        Op::Sample(a) => sample(ctx, a).map(|(l, _)| l),
        Op::Quantize(a) => quantize(ctx, a).map(|(l, _)| l),
        Op::Dequantize(_) => dequantize(ctx).map(|(l, _)| l),
        Op::Collective(a) => collective(ctx, a).map(|(l, _)| l),
        Op::SendRecv(_) => send_recv(ctx).map(|(l, _)| l),
        Op::Opaque(a) => opaque(ctx, a).map(|(l, _)| l),
        Op::Mla(_) => Err(unsupported(ctx, "(MLA lowering is deferred past M0)")),
        Op::Call(_) | Op::Repeat(_) => {
            Err(unsupported(ctx, "(graph-level op, lowered by lower_graph)"))
        }
    }
}

pub fn cost_hint(ctx: &NodeCtx) -> Result<CostHint, Diagnostic> {
    match &ctx.node.op {
        Op::Einsum(a) => einsum(ctx, a).map(|(_, h)| h),
        Op::Map(a) => map(ctx, a.func.steps()).map(|(_, h)| h),
        Op::Act(a) => map(ctx, &[a.func]).map(|(_, h)| h),
        Op::Reduce(a) => reduce(ctx, a).map(|(_, h)| h),
        Op::Gather(a) => gather(ctx, a).map(|(_, h)| h),
        Op::Scatter(a) => scatter(ctx, a).map(|(_, h)| h),
        Op::Layout(a) => layout(ctx, a).map(|(_, h)| h),
        Op::Attention(a) => attention(ctx, a).map(|(_, h)| h),
        Op::Softmax(a) => softmax_op(ctx, a).map(|(_, h)| h),
        Op::RmsNorm(a) => rms_norm(ctx, a).map(|(_, h)| h),
        Op::LayerNorm(_) => layer_norm(ctx).map(|(_, h)| h),
        Op::Rope(a) => rope(ctx, a).map(|(_, h)| h),
        Op::GatedAct(a) => gated_act(ctx, a).map(|(_, h)| h),
        Op::Embedding(_) => embedding(ctx).map(|(_, h)| h),
        Op::LogitsSelect(a) => logits_select(ctx, a).map(|(_, h)| h),
        Op::KvAppend(_) => kv_append(ctx).map(|(_, h)| h),
        Op::MoeRoute(a) => moe_route(ctx, a).map(|(_, h)| h),
        Op::MoeDispatch(a) => moe_dispatch(ctx, a).map(|(_, h)| h),
        Op::GroupedEinsum(a) => grouped_einsum(ctx, a).map(|(_, h)| h),
        Op::MoeCombine(_) => moe_combine(ctx).map(|(_, h)| h),
        Op::TopK(a) => top_k(ctx, a).map(|(_, h)| h),
        Op::Sample(a) => sample(ctx, a).map(|(_, h)| h),
        Op::Quantize(a) => quantize(ctx, a).map(|(_, h)| h),
        Op::Dequantize(_) => dequantize(ctx).map(|(_, h)| h),
        Op::Collective(a) => collective(ctx, a).map(|(_, h)| h),
        Op::SendRecv(_) => send_recv(ctx).map(|(_, h)| h),
        Op::Opaque(a) => opaque(ctx, a).map(|(_, h)| h),
        Op::Mla(_) => Err(unsupported(ctx, "(MLA lowering is deferred past M0)")),
        Op::Call(_) | Op::Repeat(_) => Err(unsupported(ctx, "(graph-level op)")),
    }
}

type Out = Result<(Lowered, CostHint), Diagnostic>;

// ---------------------------------------------------------------- einsum

pub(crate) struct EinsumEq {
    pub ins: Vec<Vec<char>>,
    pub out: Vec<char>,
}

pub(crate) fn parse_eq(eq: &str) -> Result<EinsumEq, String> {
    let (lhs, out) = eq.split_once("->").ok_or("missing '->'")?;
    let ins: Vec<Vec<char>> = lhs.split(',').map(|s| s.trim().chars().collect()).collect();
    let out: Vec<char> = out.trim().chars().collect();
    for op in ins.iter().chain(std::iter::once(&out)) {
        if op.iter().any(|c| !c.is_ascii_lowercase()) {
            return Err(format!("letters must be a-z in {eq:?}"));
        }
        if op.iter().enumerate().any(|(i, c)| op[i + 1..].contains(c)) {
            return Err(format!(
                "repeated letter within one operand in {eq:?} (diagonals are not supported)"
            ));
        }
    }
    if let Some(c) = out.iter().find(|c| !ins.iter().any(|i| i.contains(c))) {
        return Err(format!("output letter {c:?} does not appear in any input"));
    }
    Ok(EinsumEq { ins, out })
}

/// Letter sizes from operand shapes; operands may be row-major views that merge trailing letters.
pub(crate) fn letter_sizes(
    ctx: &NodeCtx,
    eq: &EinsumEq,
    shapes: &[&[u64]],
) -> Result<Vec<(char, u64)>, Diagnostic> {
    let ops: Vec<&Vec<char>> = eq.ins.iter().chain(std::iter::once(&eq.out)).collect();
    let mut sizes: Vec<(char, u64)> = Vec::new();
    for (letters, shape) in ops.iter().zip(shapes) {
        if shape.len() > letters.len() {
            return Err(err(
                "E-WL-EIN-001",
                ctx,
                format!("operand of rank {} has letters {letters:?}", shape.len()),
            ));
        }
        if shape.len() == letters.len() {
            for (&c, &d) in letters.iter().zip(shape.iter()) {
                match sizes.iter().find(|(l, _)| *l == c) {
                    Some(&(_, s)) if s != d => {
                        return Err(err(
                            "E-WL-SHAPE-001",
                            ctx,
                            format!("letter {c:?} is {s} in one operand and {d} in another"),
                        ));
                    }
                    Some(_) => {}
                    None => sizes.push((c, d)),
                }
            }
        }
    }
    for (letters, shape) in ops.iter().zip(shapes) {
        if shape.len() == letters.len() {
            continue;
        }
        let mut li = letters.iter();
        for (di, &d) in shape.iter().enumerate() {
            let last = di + 1 == shape.len();
            let mut prod = 1u64;
            let mut took = 0;
            while took == 0 || prod < d || last {
                let Some(c) = li.next() else { break };
                took += 1;
                let s = sizes
                    .iter()
                    .find(|(l, _)| l == c)
                    .map(|x| x.1)
                    .ok_or_else(|| {
                        err(
                            "E-WL-EIN-002",
                            ctx,
                            format!("letter {c:?} only appears in a viewed operand"),
                        )
                        .hint("declare the operand with one dim per letter")
                    })?;
                prod *= s;
            }
            if prod != d {
                return Err(err(
                    "E-WL-EIN-002",
                    ctx,
                    format!(
                        "declared shape {shape:?} is not a row-major merge of letters {letters:?}"
                    ),
                )
                .hint("only splits of trailing dims are free views"));
            }
        }
        if li.next().is_some() {
            return Err(err(
                "E-WL-EIN-002",
                ctx,
                format!("declared shape {shape:?} is not a row-major merge of letters {letters:?}"),
            ));
        }
    }
    Ok(sizes)
}

fn default_accum(t: &TypeInfo) -> Precision {
    if t.dtype.is_float() {
        Precision::Fp32
    } else {
        Precision::Int32
    }
}

fn einsum(ctx: &NodeCtx, a: &EinsumAttrs) -> Out {
    arity(ctx, 2, 1)?;
    let eq = parse_eq(&a.eq).map_err(|m| err("E-WL-EIN-001", ctx, m))?;
    if eq.ins.len() != 2 {
        return Err(err(
            "E-WL-EIN-001",
            ctx,
            "einsum takes exactly 2 inputs in v0",
        ));
    }
    let shapes = [
        ctx.inputs[0].shape.as_slice(),
        ctx.inputs[1].shape.as_slice(),
        ctx.outputs[0].shape.as_slice(),
    ];
    let sizes = letter_sizes(ctx, &eq, &shapes)?;
    let size = |c: char| sizes.iter().find(|(l, _)| *l == c).map_or(1, |x| x.1);
    let mut order: Vec<char> = eq.out.clone();
    for c in eq.ins.iter().flatten() {
        if !order.contains(c) {
            order.push(*c);
        }
    }
    let names: Vec<String> = order.iter().map(char::to_string).collect();
    let dims: Vec<(&str, u64)> = names
        .iter()
        .zip(&order)
        .map(|(n, &c)| (n.as_str(), size(c)))
        .collect();
    let red: Vec<&str> = names
        .iter()
        .zip(&order)
        .filter(|(_, c)| !eq.out.contains(c))
        .map(|(n, _)| n.as_str())
        .collect();
    let idx = |letters: &[char]| {
        letters
            .iter()
            .map(|c| IndexExpr::dim(&c.to_string()))
            .collect()
    };
    let mut k = kernel(KernelClass::Contraction, &dims, &red)
        .rd(&ctx.node.inputs[0], idx(&eq.ins[0]))
        .rd(&ctx.node.inputs[1], idx(&eq.ins[1]))
        .wr(&ctx.node.outputs[0], idx(&eq.out))
        .body(ScalarBody { mac: 1, ..Z });
    if !red.is_empty() {
        k = k.comb(Combiner::Sum);
    }
    k.accum = Some(a.accum.unwrap_or_else(|| default_accum(&ctx.inputs[0])));
    let points: u128 = dims.iter().map(|d| u128::from(d.1)).product();
    let mut kb = Kb::new(ctx);
    kb.push(k);
    Ok((
        kb.done(),
        H::new(ctx)
            .all_inputs()
            .all_outputs()
            .work(points, ScalarBody { mac: 1, ..Z })
            .h,
    ))
}

// ---------------------------------------------------------------- elementwise

fn broadcast(ctx: &NodeCtx) -> Result<Vec<u64>, Diagnostic> {
    let r = ctx.inputs.iter().map(|t| t.shape.len()).max().unwrap_or(0);
    let mut out = vec![1u64; r];
    for t in &ctx.inputs {
        let off = r - t.shape.len();
        for (i, &d) in t.shape.iter().enumerate() {
            let o = &mut out[off + i];
            if *o == 1 {
                *o = d;
            } else if d != 1 && d != *o {
                return Err(err(
                    "E-WL-SHAPE-001",
                    ctx,
                    format!(
                        "shapes do not broadcast: {:?}",
                        ctx.inputs.iter().map(|t| &t.shape).collect::<Vec<_>>()
                    ),
                ));
            }
        }
    }
    Ok(out)
}

fn map(ctx: &NodeCtx, fns: &[MapFn]) -> Out {
    if ctx.inputs.is_empty() || ctx.outputs.len() != 1 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            "map takes ≥ 1 input and 1 output",
        ));
    }
    let shape = broadcast(ctx)?;
    expect_out(ctx, 0, &shape)?;
    let body = fns.iter().map(|&f| fn_body(f)).fold(Z, |a, b| a + b);
    let names: Vec<String> = (0..shape.len()).map(|i| format!("d{i}")).collect();
    let dims: Vec<(&str, u64)> = names
        .iter()
        .map(String::as_str)
        .zip(shape.iter().copied())
        .collect();
    let mut k = kernel(KernelClass::Map, &dims, &[]).body(body);
    for (t, info) in ctx.node.inputs.iter().zip(&ctx.inputs) {
        let off = shape.len() - info.shape.len();
        let idx = info
            .shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d == 1 && shape[off + i] != 1 {
                    konst(0)
                } else {
                    IndexExpr::dim(&names[off + i])
                }
            })
            .collect();
        k = k.rd(t, idx);
    }
    let names_ref: Vec<&str> = names.iter().map(String::as_str).collect();
    k = k.wr(&ctx.node.outputs[0], ix(&names_ref));
    let mut kb = Kb::new(ctx);
    kb.push(k);
    Ok((
        kb.done(),
        H::new(ctx)
            .all_inputs()
            .all_outputs()
            .work(numel(&shape), body)
            .h,
    ))
}

fn norm_axes(ctx: &NodeCtx, axes: &[i32], rank: usize) -> Result<Vec<usize>, Diagnostic> {
    let mut v = axes
        .iter()
        .map(|&a| {
            let a = if a < 0 { a + rank as i32 } else { a };
            usize::try_from(a)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| {
                    err(
                        "E-WL-SHAPE-001",
                        ctx,
                        format!("axis {a} out of range for rank {rank}"),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn reduce(ctx: &NodeCtx, a: &ReduceAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let s = &ctx.inputs[0].shape;
    let axes = norm_axes(ctx, &a.axes, s.len())?;
    let kept: Vec<u64> = s
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &d)| d)
        .collect();
    if numel(&ctx.outputs[0].shape) != numel(&kept) {
        return Err(shape_err(
            ctx,
            "reduce output",
            &kept,
            &ctx.outputs[0].shape,
        ));
    }
    let names: Vec<String> = (0..s.len()).map(|i| format!("d{i}")).collect();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();
    let dims: Vec<(&str, u64)> = all.iter().copied().zip(s.iter().copied()).collect();
    let red: Vec<&str> = axes.iter().map(|&i| all[i]).collect();
    let keep: Vec<&str> = all.iter().copied().filter(|n| !red.contains(n)).collect();
    let (body, comb) = match a.combiner {
        ReduceKind::Sum | ReduceKind::Mean => (ScalarBody { add: 1, ..Z }, Combiner::Sum),
        ReduceKind::Max => (ScalarBody { max: 1, ..Z }, Combiner::Max),
        ReduceKind::Min => (ScalarBody { max: 1, ..Z }, Combiner::Min),
        ReduceKind::Sumsq => (
            ScalarBody {
                mul: 1,
                add: 1,
                ..Z
            },
            Combiner::Sum,
        ),
    };
    let (x, y) = (&ctx.node.inputs[0], &ctx.node.outputs[0]);
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Reduce, &dims, &red)
            .rd(x, ix(&all))
            .wr(y, ix(&keep))
            .body(body)
            .comb(comb),
    );
    let mut h = H::new(ctx).all_inputs().all_outputs().work(numel(s), body);
    if a.combiner == ReduceKind::Mean {
        let kd: Vec<(&str, u64)> = dims
            .iter()
            .copied()
            .filter(|(n, _)| keep.contains(n))
            .collect();
        kb.push(
            kernel(KernelClass::Map, &kd, &[])
                .rw(y, ix(&keep))
                .body(ScalarBody { mul: 1, ..Z }),
        );
        h = h.work(numel(&kept), ScalarBody { mul: 1, ..Z });
    }
    Ok((kb.done(), h.h))
}

fn axis_of(ctx: &NodeCtx, axis: i32, rank: usize) -> Result<usize, Diagnostic> {
    Ok(norm_axes(ctx, &[axis], rank)?[0])
}

fn gather(ctx: &NodeCtx, a: &GatherAttrs) -> Out {
    arity(ctx, 2, 1)?;
    let (src, idx) = (&ctx.inputs[0].shape, &ctx.inputs[1].shape);
    let ax = axis_of(ctx, a.axis, src.len())?;
    let shape: Vec<u64> = src[..ax]
        .iter()
        .chain(idx.iter())
        .chain(src[ax + 1..].iter())
        .copied()
        .collect();
    expect_out(ctx, 0, &shape)?;
    let names: Vec<String> = (0..shape.len()).map(|i| format!("d{i}")).collect();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();
    let dims: Vec<(&str, u64)> = all.iter().copied().zip(shape.iter().copied()).collect();
    let idx_dims = &all[ax..ax + idx.len()];
    let mut src_ix = ix(&all[..ax]);
    src_ix.push(IndexExpr::Indirect {
        via: ctx.node.inputs[1].clone(),
        index: ix(idx_dims),
    });
    src_ix.extend(ix(&all[ax + idx.len()..]));
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Gather, &dims, &[])
            .rd(&ctx.node.inputs[1], ix(idx_dims))
            .rd(&ctx.node.inputs[0], src_ix)
            .wr(&ctx.node.outputs[0], ix(&all)),
    );
    let n = numel(&shape);
    Ok((
        kb.done(),
        H::new(ctx).input(0, n).input(1, numel(idx)).output(0, n).h,
    ))
}

/// Inputs `dst, idx, updates`; output is the updated `dst` (an alias).
fn scatter(ctx: &NodeCtx, a: &ScatterAttrs) -> Out {
    arity(ctx, 3, 1)?;
    let (dst, idx, upd) = (
        &ctx.inputs[0].shape,
        &ctx.inputs[1].shape,
        &ctx.inputs[2].shape,
    );
    let ax = axis_of(ctx, a.axis, dst.len())?;
    let expect: Vec<u64> = dst[..ax]
        .iter()
        .chain(idx.iter())
        .chain(dst[ax + 1..].iter())
        .copied()
        .collect();
    if *upd != expect {
        return Err(shape_err(ctx, "scatter updates", &expect, upd));
    }
    expect_out(ctx, 0, dst)?;
    let names: Vec<String> = (0..upd.len()).map(|i| format!("d{i}")).collect();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();
    let dims: Vec<(&str, u64)> = all.iter().copied().zip(upd.iter().copied()).collect();
    let idx_dims = &all[ax..ax + idx.len()];
    let mut dst_ix = ix(&all[..ax]);
    dst_ix.push(IndexExpr::Indirect {
        via: ctx.node.inputs[1].clone(),
        index: ix(idx_dims),
    });
    dst_ix.extend(ix(&all[ax + idx.len()..]));
    let mut k = kernel(KernelClass::Scatter, &dims, &[])
        .rd(&ctx.node.inputs[1], ix(idx_dims))
        .rd(&ctx.node.inputs[2], ix(&all));
    let n = numel(upd);
    let mut h = H::new(ctx).input(1, numel(idx)).input(2, n).output(0, n);
    if a.combine.is_some() {
        k = k
            .rd(&ctx.node.inputs[0], dst_ix.clone())
            .body(ScalarBody { add: 1, ..Z })
            .comb(Combiner::Sum);
        h = h.input(0, n).work(n, ScalarBody { add: 1, ..Z });
    }
    k = k.wr(&ctx.node.outputs[0], dst_ix);
    let mut kb = Kb::new(ctx);
    kb.push(k);
    Ok((kb.done(), h.h))
}

/// Views and copies: no FLOPs, and no compulsory bytes unless the mapper materializes them.
fn layout(ctx: &NodeCtx, a: &LayoutAttrs) -> Out {
    if ctx.inputs.is_empty() || ctx.outputs.is_empty() {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            "layout op needs inputs and outputs",
        ));
    }
    let n_in: u128 = ctx.inputs.iter().map(TypeInfo::numel).sum();
    let n_out: u128 = ctx.outputs.iter().map(TypeInfo::numel).sum();
    let conserving = matches!(
        a.kind,
        LayoutKind::Reshape | LayoutKind::Transpose | LayoutKind::Concat | LayoutKind::Split
    );
    if conserving && n_in != n_out {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("{:?} changes element count {n_in} -> {n_out}", a.kind),
        ));
    }
    let mut kb = Kb::new(ctx);
    for (t, info) in ctx.node.outputs.iter().zip(&ctx.outputs) {
        let names: Vec<String> = (0..info.shape.len()).map(|i| format!("d{i}")).collect();
        let all: Vec<&str> = names.iter().map(String::as_str).collect();
        let dims: Vec<(&str, u64)> = all
            .iter()
            .copied()
            .zip(info.shape.iter().copied())
            .collect();
        let mut k = kernel(KernelClass::Layout, &dims, &[]);
        for i in &ctx.node.inputs {
            k = k.rd(i, ix(&all));
        }
        kb.push(k.wr(t, ix(&all)));
    }
    Ok((kb.done(), CostHint::default()))
}

// ---------------------------------------------------------------- norms, activations, softmax

fn rms_norm(ctx: &NodeCtx, a: &RmsNormAttrs) -> Out {
    let (nin, nout) = if a.fused_residual { (3, 2) } else { (2, 1) };
    arity(ctx, nin, nout)?;
    let x = &ctx.inputs[0].shape;
    let (rows, d) = rows_cols(x);
    if ctx.inputs[1].shape != [d] {
        return Err(shape_err(ctx, "norm weight", &[d], &ctx.inputs[1].shape));
    }
    for o in 0..nout {
        expect_out(ctx, o, x)?;
    }
    if a.fused_residual && ctx.inputs[2].shape != *x {
        return Err(shape_err(ctx, "residual", x, &ctx.inputs[2].shape));
    }
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let dims = [("r", rows), ("c", d)];
    let src = if a.fused_residual {
        kb.push(
            kernel(KernelClass::Map, &dims, &[])
                .rd(&n.inputs[0], ix(&["r", "c"]))
                .rd(&n.inputs[2], ix(&["r", "c"]))
                .wr(&n.outputs[1], ix(&["r", "c"]))
                .body(ScalarBody { add: 1, ..Z }),
        );
        n.outputs[1].clone()
    } else {
        n.inputs[0].clone()
    };
    let ms = kb.temp("ms", vec![rows], ElemType::FP32);
    let rs = kb.temp("rs", vec![rows], ElemType::FP32);
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["c"])
            .rd(&src, ix(&["r", "c"]))
            .wr(&ms, ix(&["r"]))
            .body(ScalarBody {
                mul: 1,
                add: 1,
                ..Z
            })
            .comb(Combiner::Sum),
    );
    kb.push(
        kernel(KernelClass::Map, &[("r", rows)], &[])
            .rd(&ms, ix(&["r"]))
            .wr(&rs, ix(&["r"]))
            .body(ScalarBody {
                add: 1,
                rsqrt: 1,
                ..Z
            }),
    );
    kb.push(
        kernel(KernelClass::Map, &dims, &[])
            .rd(&src, ix(&["r", "c"]))
            .rd(&rs, ix(&["r"]))
            .rd(&n.inputs[1], ix(&["c"]))
            .wr(&n.outputs[0], ix(&["r", "c"]))
            .body(ScalarBody { mul: 2, ..Z }),
    );
    let e = u128::from(rows) * u128::from(d);
    let mut h = H::new(ctx)
        .all_inputs()
        .all_outputs()
        .work(
            e,
            ScalarBody {
                mul: 3,
                add: 1,
                ..Z
            },
        )
        .work(
            rows.into(),
            ScalarBody {
                add: 1,
                rsqrt: 1,
                ..Z
            },
        );
    if a.fused_residual {
        h = h.work(e, ScalarBody { add: 1, ..Z });
    }
    Ok((kb.done(), h.h))
}

/// Per element 4 add + 3 mul (two reductions, centre, scale, shift); per row 2 add + 3 mul + 1 rsqrt.
fn layer_norm(ctx: &NodeCtx) -> Out {
    arity(ctx, 3, 1)?;
    let x = &ctx.inputs[0].shape;
    let (rows, d) = rows_cols(x);
    for i in [1, 2] {
        if ctx.inputs[i].shape != [d] {
            return Err(shape_err(
                ctx,
                "norm weight/bias",
                &[d],
                &ctx.inputs[i].shape,
            ));
        }
    }
    expect_out(ctx, 0, x)?;
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let dims = [("r", rows), ("c", d)];
    let s1 = kb.temp("s1", vec![rows], ElemType::FP32);
    let s2 = kb.temp("s2", vec![rows], ElemType::FP32);
    let st = kb.temp("st", vec![rows, 2], ElemType::FP32);
    let rc = ix(&["r", "c"]);
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["c"])
            .rd(&n.inputs[0], rc.clone())
            .wr(&s1, ix(&["r"]))
            .body(ScalarBody { add: 1, ..Z })
            .comb(Combiner::Sum),
    );
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["c"])
            .rd(&n.inputs[0], rc.clone())
            .wr(&s2, ix(&["r"]))
            .body(ScalarBody {
                mul: 1,
                add: 1,
                ..Z
            })
            .comb(Combiner::Sum),
    );
    kb.push(
        kernel(KernelClass::Map, &[("r", rows)], &[])
            .rd(&s1, ix(&["r"]))
            .rd(&s2, ix(&["r"]))
            .wr(&st, vec![IndexExpr::dim("r"), konst(0)])
            .wr(&st, vec![IndexExpr::dim("r"), konst(1)])
            .body(ScalarBody {
                mul: 3,
                add: 2,
                rsqrt: 1,
                ..Z
            }),
    );
    kb.push(
        kernel(KernelClass::Map, &dims, &[])
            .rd(&n.inputs[0], rc.clone())
            .rd(&st, vec![IndexExpr::dim("r"), konst(0)])
            .rd(&st, vec![IndexExpr::dim("r"), konst(1)])
            .rd(&n.inputs[1], ix(&["c"]))
            .rd(&n.inputs[2], ix(&["c"]))
            .wr(&n.outputs[0], rc)
            .body(ScalarBody {
                add: 2,
                mul: 2,
                ..Z
            }),
    );
    let e = u128::from(rows) * u128::from(d);
    let h = H::new(ctx)
        .all_inputs()
        .all_outputs()
        .work(
            e,
            ScalarBody {
                add: 4,
                mul: 3,
                ..Z
            },
        )
        .work(
            rows.into(),
            ScalarBody {
                mul: 3,
                add: 2,
                rsqrt: 1,
                ..Z
            },
        );
    Ok((kb.done(), h.h))
}

/// Softmax over the last axis of `src` viewed as `[rows, cols]`, writing `dst`. Per element 1 max, 2 add,
/// 1 exp, 1 mul (+1 mul when scaled); per row 1 rcp.
fn softmax_kernels(kb: &mut Kb, tag: &str, src: &Id, dst: &Id, rows: u64, cols: u64, scaled: bool) {
    let m = kb.temp(&format!("{tag}max"), vec![rows], ElemType::FP32);
    let e = kb.temp(&format!("{tag}exp"), vec![rows, cols], ElemType::FP32);
    let s = kb.temp(&format!("{tag}sum"), vec![rows], ElemType::FP32);
    let r = kb.temp(&format!("{tag}rcp"), vec![rows], ElemType::FP32);
    let dims = [("r", rows), ("c", cols)];
    let rc = ix(&["r", "c"]);
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["c"])
            .rd(src, rc.clone())
            .wr(&m, ix(&["r"]))
            .body(ScalarBody { max: 1, ..Z })
            .comb(Combiner::Max),
    );
    kb.push(
        kernel(KernelClass::Map, &dims, &[])
            .rd(src, rc.clone())
            .rd(&m, ix(&["r"]))
            .wr(&e, rc.clone())
            .body(ScalarBody {
                add: 1,
                exp: 1,
                mul: u16::from(scaled),
                ..Z
            }),
    );
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["c"])
            .rd(&e, rc.clone())
            .wr(&s, ix(&["r"]))
            .body(ScalarBody { add: 1, ..Z })
            .comb(Combiner::Sum),
    );
    kb.push(
        kernel(KernelClass::Map, &[("r", rows)], &[])
            .rd(&s, ix(&["r"]))
            .wr(&r, ix(&["r"]))
            .body(ScalarBody { rcp: 1, ..Z }),
    );
    kb.push(
        kernel(KernelClass::Map, &dims, &[])
            .rd(&e, rc.clone())
            .rd(&r, ix(&["r"]))
            .wr(dst, rc)
            .body(ScalarBody { mul: 1, ..Z }),
    );
}

fn softmax_work(h: H<'_>, rows: u64, cols: u64, scaled: bool) -> H<'_> {
    let e = u128::from(rows) * u128::from(cols);
    h.work(
        e,
        ScalarBody {
            max: 1,
            add: 2,
            exp: 1,
            mul: 1 + u16::from(scaled),
            ..Z
        },
    )
    .work(rows.into(), ScalarBody { rcp: 1, ..Z })
}

fn softmax_op(ctx: &NodeCtx, a: &SoftmaxAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let x = &ctx.inputs[0].shape;
    if axis_of(ctx, a.axis, x.len())? != x.len() - 1 {
        return Err(unsupported(ctx, "over a non-last axis"));
    }
    expect_out(ctx, 0, x)?;
    let (rows, cols) = rows_cols(x);
    let scaled = a.scale.is_some_and(|s| s != 1.0);
    let mut kb = Kb::new(ctx);
    softmax_kernels(
        &mut kb,
        "",
        &ctx.node.inputs[0],
        &ctx.node.outputs[0],
        rows,
        cols,
        scaled,
    );
    let h = softmax_work(H::new(ctx).all_inputs().all_outputs(), rows, cols, scaled);
    Ok((kb.done(), h.h))
}

fn gated_act(ctx: &NodeCtx, a: &GatedActAttrs) -> Out {
    let two = a.layout == GateLayout::TwoInputs;
    arity(ctx, if two { 2 } else { 1 }, 1)?;
    let x = &ctx.inputs[0].shape;
    let (rows, c) = rows_cols(x);
    let f = if two { c } else { c / 2 };
    if !two && c % 2 != 0 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("gated input last dim {c} is odd"),
        ));
    }
    if two && ctx.inputs[1].shape != *x {
        return Err(shape_err(ctx, "up input", x, &ctx.inputs[1].shape));
    }
    let mut out = x.clone();
    *out.last_mut().expect("rank ≥ 1") = f;
    expect_out(ctx, 0, &out)?;
    let body = fn_body(a.func) + ScalarBody { mul: 1, ..Z };
    let n = ctx.node;
    let (g, u) = match a.layout {
        GateLayout::TwoInputs => (
            (&n.inputs[0], IndexExpr::dim("f")),
            (&n.inputs[1], IndexExpr::dim("f")),
        ),
        GateLayout::ConcatHalves => (
            (&n.inputs[0], IndexExpr::dim("f")),
            (&n.inputs[0], IndexExpr::dim_offset("f", f as i64)),
        ),
        GateLayout::Interleaved => (
            (
                &n.inputs[0],
                IndexExpr::terms(vec![term(2, Some("f"), None)], 0),
            ),
            (
                &n.inputs[0],
                IndexExpr::terms(vec![term(2, Some("f"), None)], 1),
            ),
        ),
    };
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Map, &[("r", rows), ("f", f)], &[])
            .rd(g.0, vec![IndexExpr::dim("r"), g.1])
            .rd(u.0, vec![IndexExpr::dim("r"), u.1])
            .wr(&n.outputs[0], ix(&["r", "f"]))
            .body(body),
    );
    let h = H::new(ctx)
        .all_inputs()
        .all_outputs()
        .work(u128::from(rows) * u128::from(f), body);
    Ok((kb.done(), h.h))
}

/// Inputs `x_1..x_m, pos, table[P, rotary_dim/2, 2]`; outputs `y_1..y_m`, each `[T, h_i, Dh]`.
fn rope(ctx: &NodeCtx, a: &RopeAttrs) -> Out {
    let m = ctx.outputs.len();
    if m == 0 || ctx.inputs.len() != m + 2 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            "rope takes x_1..x_m, pos, table and emits y_1..y_m",
        ));
    }
    let pos = &ctx.inputs[m].shape;
    let t = pos.first().copied().unwrap_or(0);
    let rd = u64::from(a.rotary_dim);
    let half = rd / 2;
    if pos.len() != 1 || rd % 2 != 0 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            "pos must be [T] and rotary_dim even",
        ));
    }
    let tab = &ctx.inputs[m + 1].shape;
    if tab.len() != 3 || tab[1] != half || tab[2] != 2 {
        return Err(shape_err(
            ctx,
            "rope table",
            &[tab.first().copied().unwrap_or(0), half, 2],
            tab,
        ));
    }
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let cs = kb.temp("cs", vec![t, half, 2], ElemType::FP32);
    kb.push(
        kernel(KernelClass::Gather, &[("t", t), ("f", half), ("c", 2)], &[])
            .rd(&n.inputs[m], ix(&["t"]))
            .rd(
                &n.inputs[m + 1],
                vec![
                    IndexExpr::Indirect {
                        via: n.inputs[m].clone(),
                        index: ix(&["t"]),
                    },
                    IndexExpr::dim("f"),
                    IndexExpr::dim("c"),
                ],
            )
            .wr(&cs, ix(&["t", "f", "c"])),
    );
    let body = ScalarBody {
        mul: 2,
        add: 1,
        ..Z
    };
    let mut h = H::new(ctx)
        .input(m, t.into())
        .input(m + 1, u128::from(t) * u128::from(rd));
    for i in 0..m {
        let x = &ctx.inputs[i].shape;
        if x.len() != 3 || x[0] != t || x[2] < rd {
            return Err(err(
                "E-WL-SHAPE-001",
                ctx,
                format!(
                    "rope input {} must be [T={t}, heads, Dh ≥ {rd}], found {x:?}",
                    n.inputs[i]
                ),
            ));
        }
        expect_out(ctx, i, x)?;
        let (heads, dh) = (x[1], x[2]);
        let e = match a.style {
            RopeStyle::Half => IndexExpr::terms(
                vec![term(half as i64, Some("p"), None), term(1, Some("f"), None)],
                0,
            ),
            RopeStyle::Interleaved => {
                IndexExpr::terms(vec![term(2, Some("f"), None), term(1, Some("p"), None)], 0)
            }
        };
        let xi = vec![IndexExpr::dim("t"), IndexExpr::dim("h"), e];
        let mut k = kernel(
            KernelClass::Map,
            &[("t", t), ("h", heads), ("p", 2), ("f", half)],
            &[],
        )
        .rd(&n.inputs[i], xi.clone())
        .rd(
            &cs,
            vec![IndexExpr::dim("t"), IndexExpr::dim("f"), konst(0)],
        )
        .rd(
            &cs,
            vec![IndexExpr::dim("t"), IndexExpr::dim("f"), konst(1)],
        )
        .body(body);
        if rd < dh {
            kb.push(
                kernel(KernelClass::Map, &[("t", t), ("h", heads), ("e", dh)], &[])
                    .rd(&n.inputs[i], ix(&["t", "h", "e"]))
                    .wr(&n.outputs[i], ix(&["t", "h", "e"])),
            );
            k = k.rw(&n.outputs[i], xi);
        } else {
            k = k.wr(&n.outputs[i], xi);
        }
        kb.push(k);
        let all = numel(x);
        h = h
            .input(i, all)
            .output(i, all)
            .work(u128::from(t) * u128::from(heads) * u128::from(rd), body);
    }
    Ok((kb.done(), h.h))
}

// ---------------------------------------------------------------- embedding, selection, KV cache

fn embedding(ctx: &NodeCtx) -> Out {
    arity(ctx, 2, 1)?;
    let (ids, table) = (&ctx.inputs[0].shape, &ctx.inputs[1].shape);
    if ids.len() != 1 || table.len() != 2 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("embedding needs ids [T] and table [V, d], found {ids:?}, {table:?}"),
        ));
    }
    let (t, d) = (ids[0], table[1]);
    expect_out(ctx, 0, &[t, d])?;
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Gather, &[("t", t), ("c", d)], &[])
            .rd(&n.inputs[0], ix(&["t"]))
            .rd(
                &n.inputs[1],
                vec![
                    IndexExpr::Indirect {
                        via: n.inputs[0].clone(),
                        index: ix(&["t"]),
                    },
                    IndexExpr::dim("c"),
                ],
            )
            .wr(&n.outputs[0], ix(&["t", "c"])),
    );
    let td = u128::from(t) * u128::from(d);
    Ok((
        kb.done(),
        H::new(ctx).input(0, t.into()).input(1, td).output(0, td).h,
    ))
}

fn logits_select(ctx: &NodeCtx, a: &LogitsSelectAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let h = &ctx.inputs[0].shape;
    let (t, d) = rows_cols(h);
    if t != ctx.seqs.tokens() {
        return Err(shape_err(
            ctx,
            "hidden rows (T)",
            &[ctx.seqs.tokens(), d],
            h,
        ));
    }
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let rows = match a.which {
        Which::Last => {
            let nseq = ctx.seqs.seqs();
            expect_out(ctx, 0, &[nseq, d])?;
            let last = IndexExpr::terms(
                vec![
                    term(1, None, Some("tok_base")),
                    term(1, Some("s"), Some("q_len")),
                    term(1, None, Some("q_len")),
                ],
                -1,
            );
            kb.push(
                kernel(KernelClass::Gather, &[("s", 0), ("c", d)], &[])
                    .dom(segmented(ctx.seqs, "s", |_| (vec![], vec![])))
                    .rd(&n.inputs[0], vec![last, IndexExpr::dim("c")])
                    .wr(&n.outputs[0], vec![slot("s"), IndexExpr::dim("c")]),
            );
            nseq
        }
        Which::All => {
            expect_out(ctx, 0, &[t, d])?;
            kb.push(
                kernel(KernelClass::Gather, &[("t", t), ("c", d)], &[])
                    .rd(&n.inputs[0], ix(&["t", "c"]))
                    .wr(&n.outputs[0], ix(&["t", "c"])),
            );
            t
        }
    };
    let e = u128::from(rows) * u128::from(d);
    Ok((kb.done(), H::new(ctx).input(0, e).output(0, e).h))
}

fn cache_dims(ctx: &NodeCtx, i: usize) -> Result<(u64, u64, u64, u64), Diagnostic> {
    match ctx.inputs[i].shape.as_slice() {
        &[slots, cap, h, e] => Ok((slots, cap, h, e)),
        s => Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("KV cache must be [slots, kv_cap, Hkv, D], found {s:?}"),
        )),
    }
}

fn check_capacity(ctx: &NodeCtx, slots: u64, cap: u64) -> Result<(), Diagnostic> {
    if ctx.seqs.seqs() > slots || ctx.seqs.max_kv() > cap {
        return Err(err(
            "E-WL-SCN-003",
            ctx,
            format!("{} sequences up to kv_len {} do not fit a cache of {slots} slots × {cap} positions", ctx.seqs.seqs(), ctx.seqs.max_kv()),
        )
        .hint("bind kv_cap ≥ max kv_len and slots ≥ N"));
    }
    Ok(())
}

/// Inputs `cache_k, cache_v, k[T,Hkv,Dh], v[T,Hkv,Dv]`; outputs the in-place cache versions.
fn kv_append(ctx: &NodeCtx) -> Out {
    arity(ctx, 4, 2)?;
    let mut kb = Kb::new(ctx);
    let mut h = H::new(ctx);
    let t = ctx.seqs.tokens();
    for (c, x) in [(0usize, 2usize), (1, 3)] {
        let (slots, cap, hk, e) = cache_dims(ctx, c)?;
        check_capacity(ctx, slots, cap)?;
        if ctx.inputs[x].shape != [t, hk, e] {
            return Err(shape_err(ctx, "new K/V", &[t, hk, e], &ctx.inputs[x].shape));
        }
        expect_out(ctx, c, &ctx.inputs[c].shape)?;
        let cvt = u16::from(ctx.inputs[x].dtype != ctx.outputs[c].dtype);
        let n = ctx.node;
        kb.push(
            kernel(
                KernelClass::Scatter,
                &[("s", 0), ("q", 0), ("h", hk), ("e", e)],
                &[],
            )
            .dom(segmented(ctx.seqs, "s", |s| (vec![("q", s.q_len)], vec![])))
            .rd(
                &n.inputs[x],
                vec![
                    tok("s", Some("q")),
                    IndexExpr::dim("h"),
                    IndexExpr::dim("e"),
                ],
            )
            .wr(
                &n.outputs[c],
                vec![
                    slot("s"),
                    IndexExpr::terms(
                        vec![term(1, None, Some("past")), term(1, Some("q"), None)],
                        0,
                    ),
                    IndexExpr::dim("h"),
                    IndexExpr::dim("e"),
                ],
            )
            .body(ScalarBody { cvt, ..Z }),
        );
        let el = u128::from(t) * u128::from(hk) * u128::from(e);
        h = h
            .input(x, el)
            .output(c, el)
            .work(el, ScalarBody { cvt, ..Z });
    }
    Ok((kb.done(), h.h))
}

// ---------------------------------------------------------------- attention

/// Masked (q, j) points of one sequence and the number of distinct j it reads.
fn mask_counts(ctx: &NodeCtx, mask: &Mask, s: &Segment) -> Result<(u128, u128), Diagnostic> {
    let (q, kv, p) = (
        u128::from(s.q_len),
        u128::from(s.kv_len),
        u128::from(s.past()),
    );
    Ok(match mask {
        Mask::None => (q * kv, kv),
        Mask::Causal => (q * p + q * (q + 1) / 2, kv),
        Mask::SlidingWindow { window } => {
            let w = u128::from(*window);
            let a = w.saturating_sub(p).min(q);
            (
                a * (p + 1) + a * a.saturating_sub(1) / 2 + (q - a) * w,
                kv - (p + 1).saturating_sub(w),
            )
        }
        Mask::Chunked { .. } | Mask::Explicit { .. } => {
            return Err(unsupported(ctx, "with chunked/explicit masks"));
        }
    })
}

fn mask_constraints(mask: &Mask, s: &Segment) -> Vec<DiffConstraint> {
    let p = s.past() as i64;
    match mask {
        Mask::Causal => vec![DiffConstraint::diff("j", "q", p)],
        Mask::SlidingWindow { window } => vec![
            DiffConstraint::diff("j", "q", p),
            DiffConstraint::diff("q", "j", i64::from(*window) - 1 - p),
        ],
        _ => vec![],
    }
}

/// Unfused lowering (02 §5.4): k0 QK^T, softmax kernels over j, PV, then per-row 1/l and output scaling.
fn attention(ctx: &NodeCtx, a: &AttnAttrs) -> Out {
    arity(ctx, 3, 1)?;
    if a.sinks {
        return Err(unsupported(ctx, "with sinks"));
    }
    let (h, hk, dh, dv) = (
        u64::from(a.n_heads),
        u64::from(a.n_kv_heads),
        u64::from(a.head_dim),
        u64::from(a.dv()),
    );
    if hk == 0 || h % hk != 0 {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("n_heads {h} must be a multiple of n_kv_heads {hk}"),
        ));
    }
    let g = h / hk;
    let t = ctx.seqs.tokens();
    if ctx.inputs[0].shape != [t, h, dh] {
        return Err(shape_err(ctx, "q", &[t, h, dh], &ctx.inputs[0].shape));
    }
    for (i, d) in [(1, dh), (2, dv)] {
        let (slots, cap, chk, cd) = cache_dims(ctx, i)?;
        if (chk, cd) != (hk, d) {
            return Err(shape_err(
                ctx,
                "kv cache",
                &[slots, cap, hk, d],
                &ctx.inputs[i].shape,
            ));
        }
        check_capacity(ctx, slots, cap)?;
    }
    expect_out(ctx, 0, &[t, h, dv])?;
    let mut p_attn = 0u128;
    let (mut k_rows, mut rows) = (0u128, 0u128);
    for s in &ctx.seqs.segments {
        let (pts, reach) = mask_counts(ctx, &a.mask, s)?;
        p_attn += u128::from(s.count) * u128::from(h) * pts;
        k_rows += u128::from(s.count) * u128::from(hk) * reach;
        rows += u128::from(s.count) * u128::from(h) * u128::from(s.q_len);
    }
    let n = ctx.node;
    let qdt = ctx.inputs[0].dtype;
    let mut kb = Kb::new(ctx);
    let sc = kb.temp("s", vec![p_attn as u64], qdt);
    let mx = kb.temp("m", vec![rows as u64], ElemType::FP32);
    let pr = kb.temp("p", vec![p_attn as u64], qdt);
    let l = kb.temp("l", vec![rows as u64], ElemType::FP32);
    let rl = kb.temp("rl", vec![rows as u64], ElemType::FP32);
    let dom = |with_j: bool| {
        segmented(ctx.seqs, "s", |s| {
            if with_j {
                (
                    vec![("q", s.q_len), ("j", s.kv_len)],
                    mask_constraints(&a.mask, s),
                )
            } else {
                (vec![("q", s.q_len)], vec![])
            }
        })
    };
    let head = IndexExpr::terms(
        vec![term(g as i64, Some("hk"), None), term(1, Some("g"), None)],
        0,
    );
    let qi = |e: &str| vec![tok("s", Some("q")), head.clone(), IndexExpr::dim(e)];
    let kvi = |e: &str| {
        vec![
            slot("s"),
            IndexExpr::dim("j"),
            IndexExpr::dim("hk"),
            IndexExpr::dim(e),
        ]
    };
    let sij = ix(&["s", "hk", "g", "q", "j"]);
    let row = ix(&["s", "hk", "g", "q"]);
    let base = [("s", 0), ("hk", hk), ("g", g), ("q", 0), ("j", 0)];
    let mut k0 = kernel(
        KernelClass::Contraction,
        &[base.as_slice(), &[("e", dh)]].concat(),
        &["e"],
    )
    .dom(dom(true))
    .rd(&n.inputs[0], qi("e"))
    .rd(&n.inputs[1], kvi("e"))
    .wr(&sc, sij.clone())
    .body(ScalarBody { mac: 1, ..Z })
    .comb(Combiner::Sum);
    k0.accum = Some(Precision::Fp32);
    kb.push(k0);
    let softcap = ScalarBody {
        tanh: 1,
        mul: 2,
        ..Z
    };
    if a.softcap.is_some() {
        kb.push(
            kernel(KernelClass::Map, &base, &[])
                .dom(dom(true))
                .rw(&sc, sij.clone())
                .body(softcap),
        );
    }
    kb.push(
        kernel(KernelClass::Reduce, &base, &["j"])
            .dom(dom(true))
            .rd(&sc, sij.clone())
            .wr(&mx, row.clone())
            .body(ScalarBody { max: 1, ..Z })
            .comb(Combiner::Max),
    );
    kb.push(
        kernel(KernelClass::Reduce, &base, &["j"])
            .dom(dom(true))
            .rd(&sc, sij.clone())
            .rd(&mx, row.clone())
            .wr(&pr, sij.clone())
            .wr(&l, row.clone())
            .body(ScalarBody {
                add: 2,
                exp: 1,
                ..Z
            })
            .comb(Combiner::Sum),
    );
    let mut k2 = kernel(
        KernelClass::Contraction,
        &[base.as_slice(), &[("v", dv)]].concat(),
        &["j"],
    )
    .dom(dom(true))
    .rd(&pr, sij)
    .rd(&n.inputs[2], kvi("v"))
    .wr(&n.outputs[0], qi("v"))
    .body(ScalarBody { mac: 1, ..Z })
    .comb(Combiner::Sum);
    k2.accum = Some(Precision::Fp32);
    kb.push(k2);
    let rbase = [("s", 0), ("hk", hk), ("g", g), ("q", 0)];
    kb.push(
        kernel(KernelClass::Map, &rbase, &[])
            .dom(dom(false))
            .rd(&l, row.clone())
            .wr(&rl, row.clone())
            .body(ScalarBody { rcp: 1, ..Z }),
    );
    kb.push(
        kernel(
            KernelClass::Map,
            &[rbase.as_slice(), &[("v", dv)]].concat(),
            &[],
        )
        .dom(dom(false))
        .rd(&rl, row)
        .rw(&n.outputs[0], qi("v"))
        .body(ScalarBody { mul: 1, ..Z }),
    );
    let mut hh = H::new(ctx)
        .input(0, u128::from(t) * u128::from(h) * u128::from(dh))
        .input(1, k_rows * u128::from(dh))
        .input(2, k_rows * u128::from(dv))
        .output(0, u128::from(t) * u128::from(h) * u128::from(dv))
        .work(
            p_attn,
            ScalarBody {
                max: 1,
                add: 2,
                exp: 1,
                ..Z
            },
        )
        .work(rows, ScalarBody { rcp: 1, ..Z })
        .work(rows * u128::from(dv), ScalarBody { mul: 1, ..Z });
    hh.h.flops_mm += 2 * p_attn * (u128::from(dh) + u128::from(dv));
    if a.softcap.is_some() {
        hh = hh.work(p_attn, softcap);
    }
    Ok((kb.done(), hh.h))
}

// ---------------------------------------------------------------- MoE, top-k, sampling

/// Top-k over the last axis of `src [rows, cols]`: a selection reduce (⌈log2 k⌉+1 cmp per element, the
/// heap model of 02 §5.3) and a gather-class materialization of the k winners into `outs`.
fn topk_kernels(kb: &mut Kb, src: &Id, outs: &[&Id], rows: u64, cols: u64, k: u32) {
    let sel = kb.temp("sel", vec![rows, u64::from(k)], ElemType::INT32);
    kb.push(
        kernel(KernelClass::Reduce, &[("r", rows), ("c", cols)], &["c"])
            .rd(src, ix(&["r", "c"]))
            .wr(&sel, vec![IndexExpr::dim("r"), konst(0)])
            .body(ScalarBody {
                cmp: ceil_log2(k) + 1,
                ..Z
            })
            .comb(Combiner::TopK(k)),
    );
    let mut g =
        kernel(KernelClass::Gather, &[("r", rows), ("i", k.into())], &[]).rd(&sel, ix(&["r", "i"]));
    for o in outs {
        g = g.wr(o, ix(&["r", "i"]));
    }
    kb.push(g);
}

fn topk_cmp(rows: u64, cols: u64, k: u32) -> (u128, ScalarBody) {
    (
        u128::from(rows) * u128::from(cols),
        ScalarBody {
            cmp: ceil_log2(k) + 1,
            ..Z
        },
    )
}

fn top_k(ctx: &NodeCtx, a: &TopKAttrs) -> Out {
    arity(ctx, 1, 2)?;
    let (rows, cols) = rows_cols(&ctx.inputs[0].shape);
    let k = u64::from(a.k);
    if k == 0 || k > cols {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("k = {k} must be in 1..={cols}"),
        ));
    }
    let mut shape = ctx.inputs[0].shape.clone();
    *shape.last_mut().expect("rank ≥ 1") = k;
    expect_out(ctx, 0, &shape)?;
    expect_out(ctx, 1, &shape)?;
    let mut kb = Kb::new(ctx);
    topk_kernels(
        &mut kb,
        &ctx.node.inputs[0],
        &[&ctx.node.outputs[0], &ctx.node.outputs[1]],
        rows,
        cols,
        a.k,
    );
    let (p, b) = topk_cmp(rows, cols, a.k);
    Ok((
        kb.done(),
        H::new(ctx).all_inputs().all_outputs().work(p, b).h,
    ))
}

/// Logits `[T, E]` → `idx [T, k]`, `wts [T, k]`.
fn moe_route(ctx: &NodeCtx, a: &MoeRouteAttrs) -> Out {
    arity(ctx, 1, 2)?;
    if a.group_limited.is_some() {
        return Err(unsupported(ctx, "with group_limited routing"));
    }
    let (t, e) = rows_cols(&ctx.inputs[0].shape);
    if e != u64::from(a.n_experts) || a.top_k == 0 || a.top_k > a.n_experts {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!(
                "router logits have {e} experts, attrs say {} (top_k {})",
                a.n_experts, a.top_k
            ),
        ));
    }
    let k = u64::from(a.top_k);
    expect_out(ctx, 0, &[t, k])?;
    expect_out(ctx, 1, &[t, k])?;
    let n = ctx.node;
    let (idx, wts) = (&n.outputs[0], &n.outputs[1]);
    let mut kb = Kb::new(ctx);
    let mut h = H::new(ctx).all_inputs().all_outputs();
    let te = u128::from(t) * u128::from(e);
    let tk = u128::from(t) * u128::from(k);
    let sig = ScalarBody {
        exp: 1,
        add: 1,
        rcp: 1,
        ..Z
    };
    let bias = ScalarBody { add: 1, ..Z };
    let mul = ScalarBody { mul: 1, ..Z };
    if a.softmax_after_topk {
        let vals = kb.temp("vals", vec![t, k], ElemType::FP32);
        topk_kernels(&mut kb, &n.inputs[0], &[idx, &vals], t, e, a.top_k);
        let (p, b) = topk_cmp(t, e, a.top_k);
        h = h.work(p, b);
        match a.scoring {
            Scoring::Softmax => {
                softmax_kernels(&mut kb, "w", &vals, wts, t, k, false);
                h = softmax_work(h, t, k, false);
            }
            Scoring::Sigmoid => {
                kb.push(
                    kernel(KernelClass::Map, &[("r", t), ("c", k)], &[])
                        .rd(&vals, ix(&["r", "c"]))
                        .wr(wts, ix(&["r", "c"]))
                        .body(sig),
                );
                h = h.work(tk, sig);
            }
        }
    } else {
        let probs = kb.temp("probs", vec![t, e], ElemType::FP32);
        match a.scoring {
            Scoring::Softmax => {
                softmax_kernels(&mut kb, "", &n.inputs[0], &probs, t, e, false);
                h = softmax_work(h, t, e, false);
            }
            Scoring::Sigmoid => {
                kb.push(
                    kernel(KernelClass::Map, &[("r", t), ("c", e)], &[])
                        .rd(&n.inputs[0], ix(&["r", "c"]))
                        .wr(&probs, ix(&["r", "c"]))
                        .body(sig),
                );
                h = h.work(te, sig);
            }
        }
        let src = if a.bias_correction {
            let biased = kb.temp("biased", vec![t, e], ElemType::FP32);
            kb.push(
                kernel(KernelClass::Map, &[("r", t), ("c", e)], &[])
                    .rd(&probs, ix(&["r", "c"]))
                    .wr(&biased, ix(&["r", "c"]))
                    .body(bias),
            );
            h = h.work(te, bias);
            biased
        } else {
            probs
        };
        topk_kernels(&mut kb, &src, &[idx, wts], t, e, a.top_k);
        let (p, b) = topk_cmp(t, e, a.top_k);
        h = h.work(p, b);
        if a.norm_topk {
            let s = kb.temp("wsum", vec![t], ElemType::FP32);
            kb.push(
                kernel(KernelClass::Reduce, &[("r", t), ("c", k)], &["c"])
                    .rd(wts, ix(&["r", "c"]))
                    .wr(&s, ix(&["r"]))
                    .body(bias)
                    .comb(Combiner::Sum),
            );
            kb.push(
                kernel(KernelClass::Map, &[("r", t)], &[])
                    .rw(&s, ix(&["r"]))
                    .body(ScalarBody { rcp: 1, ..Z }),
            );
            kb.push(
                kernel(KernelClass::Map, &[("r", t), ("c", k)], &[])
                    .rd(&s, ix(&["r"]))
                    .rw(wts, ix(&["r", "c"]))
                    .body(mul),
            );
            h = h
                .work(tk, bias)
                .work(t.into(), ScalarBody { rcp: 1, ..Z })
                .work(tk, mul);
        }
    }
    if a.routed_scaling.is_some() {
        kb.push(
            kernel(KernelClass::Map, &[("r", t), ("c", k)], &[])
                .rw(wts, ix(&["r", "c"]))
                .body(mul),
        );
        h = h.work(tk, mul);
    }
    Ok((kb.done(), h.h))
}

/// `x [T, d], idx [T, k]` → `xe` (`[E, C, d]` padded or `[T·k, d]` ragged). Every assignment row is moved
/// (overflow drops are not subtracted in M0).
fn moe_dispatch(ctx: &NodeCtx, a: &MoeDispatchAttrs) -> Out {
    arity(ctx, 2, 1)?;
    let (t, d) = rows_cols(&ctx.inputs[0].shape);
    let k = u64::from(a.top_k);
    if ctx.inputs[1].shape != [t, k] {
        return Err(shape_err(ctx, "routing idx", &[t, k], &ctx.inputs[1].shape));
    }
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Scatter, &[("t", t), ("i", k), ("c", d)], &[])
            .rd(&n.inputs[0], ix(&["t", "c"]))
            .rd(&n.inputs[1], ix(&["t", "i"]))
            .wr(
                &n.outputs[0],
                vec![
                    IndexExpr::Indirect {
                        via: n.inputs[1].clone(),
                        index: ix(&["t", "i"]),
                    },
                    IndexExpr::dim("c"),
                ],
            ),
    );
    let h = H::new(ctx)
        .all_inputs()
        .output(0, u128::from(t) * u128::from(k) * u128::from(d));
    Ok((kb.done(), h.h))
}

/// `xe [E, C, ...] × w [E, ...] → [E, C, ...]` over kept rows only: expert letter first everywhere, the ragged
/// row letter in `xe` and the output but not in the weight.
fn grouped_einsum(ctx: &NodeCtx, a: &EinsumAttrs) -> Out {
    arity(ctx, 2, 1)?;
    let eq = parse_eq(&a.eq).map_err(|m| err("E-WL-EIN-001", ctx, m))?;
    let bad_eq = || {
        err(
            "E-WL-EIN-001",
            ctx,
            format!(
                "grouped einsum {:?} must have two operands, lead with the expert letter and have a ragged row letter",
                a.eq
            ),
        )
    };
    let (Some(&ex), [x, w]) = (eq.out.first(), eq.ins.as_slice()) else {
        return Err(bad_eq());
    };
    if x.first() != Some(&ex) || w.first() != Some(&ex) {
        return Err(bad_eq());
    }
    let rag = x
        .iter()
        .copied()
        .find(|c| eq.out.contains(c) && !w.contains(c) && *c != ex)
        .ok_or_else(bad_eq)?;
    let shapes = [
        ctx.inputs[0].shape.as_slice(),
        ctx.inputs[1].shape.as_slice(),
        ctx.outputs[0].shape.as_slice(),
    ];
    let sizes = letter_sizes(ctx, &eq, &shapes)?;
    let size = |c: char| sizes.iter().find(|(l, _)| *l == c).map_or(1, |x| x.1);
    let (n_exp, cap) = (size(ex), size(rag));
    let rows: Vec<u64> = match &ctx.moe_rows {
        Some(r) if r.rows.len() as u64 == n_exp => {
            if let Some((e, &l)) = r.rows.iter().enumerate().find(|(_, l)| **l > cap)
                && !r.drop_overflow
            {
                return Err(err(
                    "E-WL-SHAPE-001",
                    ctx,
                    format!(
                        "expert {e} receives {l} rows, above the declared capacity {cap}, and the dispatch does not drop overflow"
                    ),
                )
                .hint("raise the dispatch capacity or set drop_policy: drop_overflow"));
            }
            r.rows.iter().map(|&x| x.min(cap)).collect()
        }
        Some(r) => {
            return Err(err(
                "E-WL-SHAPE-001",
                ctx,
                format!("{} expert loads for {n_exp} experts", r.rows.len()),
            ));
        }
        None => vec![cap; n_exp as usize],
    };
    let mut order: Vec<char> = eq.out.clone();
    for c in eq.ins.iter().flatten() {
        if !order.contains(c) {
            order.push(*c);
        }
    }
    let names: Vec<String> = order.iter().map(char::to_string).collect();
    let dims: Vec<(&str, u64)> = names
        .iter()
        .zip(&order)
        .map(|(nm, &c)| (nm.as_str(), size(c)))
        .collect();
    let red: Vec<&str> = names
        .iter()
        .zip(&order)
        .filter(|(_, c)| !eq.out.contains(c))
        .map(|(nm, _)| nm.as_str())
        .collect();
    let (es, rs) = (ex.to_string(), rag.to_string());
    let mut segments = Vec::new();
    let mut base = 0i64;
    for run in rows.chunk_by(|a, b| a == b) {
        segments.push(SegmentDomain {
            extents: vec![(es.clone(), run.len() as u64), (rs.clone(), run[0])],
            params: vec![("e_base".into(), base)],
            constraints: vec![],
        });
        base += run.len() as i64;
    }
    let idx = |letters: &[char]| -> Vec<IndexExpr> {
        letters
            .iter()
            .map(|c| {
                if *c == ex {
                    IndexExpr::terms(
                        vec![term(1, None, Some("e_base")), term(1, Some(&es), None)],
                        0,
                    )
                } else {
                    IndexExpr::dim(&c.to_string())
                }
            })
            .collect()
    };
    let n = ctx.node;
    let mut k = kernel(KernelClass::Contraction, &dims, &red)
        .dom(Domain::Segmented {
            seg_dim: es.clone(),
            segments,
        })
        .rd(&n.inputs[0], idx(&eq.ins[0]))
        .rd(&n.inputs[1], idx(&eq.ins[1]))
        .wr(&n.outputs[0], idx(&eq.out))
        .body(ScalarBody { mac: 1, ..Z })
        .comb(Combiner::Sum);
    k.accum = Some(a.accum.unwrap_or_else(|| default_accum(&ctx.inputs[0])));
    let mut kb = Kb::new(ctx);
    kb.push(k);
    let kept: u128 = rows.iter().map(|&r| u128::from(r)).sum();
    let active = rows.iter().filter(|&&r| r > 0).count() as u128;
    let rest = |letters: &[char]| -> u128 {
        letters
            .iter()
            .filter(|c| **c != ex && **c != rag)
            .map(|&c| u128::from(size(c)))
            .product()
    };
    let other: u128 = order
        .iter()
        .filter(|c| **c != ex && **c != rag)
        .map(|&c| u128::from(size(c)))
        .product();
    let h = H::new(ctx)
        .input(0, kept * rest(&eq.ins[0]))
        .input(1, active * rest(&eq.ins[1]))
        .output(0, kept * rest(&eq.out))
        .work(kept * other, ScalarBody { mac: 1, ..Z });
    Ok((kb.done(), h.h))
}

/// `ye, idx [T, k], wts [T, k]` → `[T, d]`: per (token, k) d mul + d add.
fn moe_combine(ctx: &NodeCtx) -> Out {
    arity(ctx, 3, 1)?;
    let (t, k) = rows_cols(&ctx.inputs[1].shape);
    let d = ctx.inputs[0].shape.last().copied().unwrap_or(0);
    if ctx.inputs[2].shape != [t, k] {
        return Err(shape_err(
            ctx,
            "routing weights",
            &[t, k],
            &ctx.inputs[2].shape,
        ));
    }
    expect_out(ctx, 0, &[t, d])?;
    let n = ctx.node;
    let body = ScalarBody {
        mul: 1,
        add: 1,
        ..Z
    };
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(
            KernelClass::Scatter,
            &[("t", t), ("i", k), ("c", d)],
            &["i"],
        )
        .rd(
            &n.inputs[0],
            vec![
                IndexExpr::Indirect {
                    via: n.inputs[1].clone(),
                    index: ix(&["t", "i"]),
                },
                IndexExpr::dim("c"),
            ],
        )
        .rd(&n.inputs[1], ix(&["t", "i"]))
        .rd(&n.inputs[2], ix(&["t", "i"]))
        .wr(&n.outputs[0], ix(&["t", "c"]))
        .body(body)
        .comb(Combiner::Sum),
    );
    let tk = u128::from(t) * u128::from(k);
    let h = H::new(ctx)
        .input(0, tk * u128::from(d))
        .input(1, tk)
        .input(2, tk)
        .all_outputs()
        .work(tk * u128::from(d), body);
    Ok((kb.done(), h.h))
}

/// logits `[N, V]` → ids `[N]`. Greedy: V cmp per row. top_k: top-k + softmax over k + a draw scan over k
/// (1 add + 1 cmp per element). top_p: softmax over V + 3 scan passes (1 add + 1 cmp each). min_p: softmax +
/// threshold (1 cmp) + draw scan. A temperature ≠ 1 adds 1 mul per logit for non-greedy strategies.
fn sample(ctx: &NodeCtx, a: &SampleAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let (rows, v) = rows_cols(&ctx.inputs[0].shape);
    expect_out(ctx, 0, &[rows])?;
    let n = ctx.node;
    let (logits, ids) = (&n.inputs[0], &n.outputs[0]);
    let mut kb = Kb::new(ctx);
    let mut h = H::new(ctx).all_inputs().all_outputs();
    let rv = u128::from(rows) * u128::from(v);
    let rc = ix(&["r", "c"]);
    let draw = |kb: &mut Kb, src: &Id, cols: u64, body: ScalarBody| {
        kb.push(
            kernel(KernelClass::Reduce, &[("r", rows), ("c", cols)], &["c"])
                .rd(src, ix(&["r", "c"]))
                .wr(ids, ix(&["r"]))
                .body(body)
                .comb(Combiner::Scan),
        );
    };
    if matches!(a.strategy, Strategy::Greedy) {
        kb.push(
            kernel(KernelClass::Reduce, &[("r", rows), ("c", v)], &["c"])
                .rd(logits, rc)
                .wr(ids, ix(&["r"]))
                .body(ScalarBody { cmp: 1, ..Z })
                .comb(Combiner::ArgMax),
        );
        return Ok((kb.done(), h.work(rv, ScalarBody { cmp: 1, ..Z }).h));
    }
    let src = if a.temperature.is_some_and(|t| t != 1.0) {
        let scaled = kb.temp("scaled", vec![rows, v], ElemType::FP32);
        kb.push(
            kernel(KernelClass::Map, &[("r", rows), ("c", v)], &[])
                .rd(logits, rc.clone())
                .wr(&scaled, rc)
                .body(ScalarBody { mul: 1, ..Z }),
        );
        h = h.work(rv, ScalarBody { mul: 1, ..Z });
        scaled
    } else {
        logits.clone()
    };
    let scan = ScalarBody {
        add: 1,
        cmp: 1,
        ..Z
    };
    match a.strategy {
        Strategy::TopK { k } => {
            let k64 = u64::from(k);
            if k == 0 || k64 > v {
                return Err(err(
                    "E-WL-SHAPE-001",
                    ctx,
                    format!("top_k {k} must be in 1..={v}"),
                ));
            }
            let vals = kb.temp("vals", vec![rows, k64], ElemType::FP32);
            let probs = kb.temp("probs", vec![rows, k64], ElemType::FP32);
            topk_kernels(&mut kb, &src, &[&vals], rows, v, k);
            softmax_kernels(&mut kb, "", &vals, &probs, rows, k64, false);
            draw(&mut kb, &probs, k64, scan);
            let (p, b) = topk_cmp(rows, v, k);
            h = softmax_work(h.work(p, b), rows, k64, false)
                .work(u128::from(rows) * u128::from(k64), scan);
        }
        Strategy::TopP { .. } | Strategy::MinP { .. } => {
            let probs = kb.temp("probs", vec![rows, v], ElemType::FP32);
            softmax_kernels(&mut kb, "", &src, &probs, rows, v, false);
            let body = if matches!(a.strategy, Strategy::TopP { .. }) {
                ScalarBody {
                    add: 3,
                    cmp: 3,
                    ..Z
                }
            } else {
                ScalarBody {
                    add: 1,
                    cmp: 2,
                    ..Z
                }
            };
            draw(&mut kb, &probs, v, body);
            h = softmax_work(h, rows, v, false).work(rv, body);
        }
        Strategy::Greedy => unreachable!("handled above"),
    }
    Ok((kb.done(), h.h))
}

// ---------------------------------------------------------------- quantization

/// Elements per scale group of an output type: block size, per-row, or whole tensor.
fn scale_group(ctx: &NodeCtx, et: &ElemType, shape: &[u64]) -> Result<Option<u64>, Diagnostic> {
    let n = numel(shape) as u64;
    Ok(match et.scaling {
        Scaling::None => None,
        Scaling::PerTensor { .. } => Some(n),
        Scaling::PerAxis { axis, .. } if axis_of(ctx, axis, shape.len())? == 0 => {
            Some(n / shape[0].max(1))
        }
        Scaling::Block { axis, block, .. }
            if axis_of(ctx, axis, shape.len())? == shape.len() - 1 =>
        {
            let b = u64::from(block);
            if shape.last().is_none_or(|&l| l % b != 0) {
                return Err(unsupported(
                    ctx,
                    &format!("with a blocked dim not divisible by block {b} (padding)"),
                ));
            }
            Some(b)
        }
        _ => {
            return Err(unsupported(
                ctx,
                "with a scale axis other than per-row or the last (blocked) axis",
            ));
        }
    })
}

fn quantize(ctx: &NodeCtx, a: &QuantizeAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let x = &ctx.inputs[0].shape;
    expect_out(ctx, 0, x)?;
    let out_t = ctx.outputs[0].dtype;
    if a.target.is_some_and(|t| t != out_t) {
        return Err(err(
            "E-WL-DT-002",
            ctx,
            "quantize target differs from the output tensor dtype",
        ));
    }
    let total = numel(x);
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let h = H::new(ctx).all_inputs().all_outputs();
    let group = scale_group(ctx, &out_t, x)?.filter(|_| a.amax_from == AmaxFrom::Dynamic);
    let Some(g) = group else {
        let cast = ScalarBody { cvt: 1, ..Z };
        kb.push(
            kernel(KernelClass::Map, &[("i", total as u64)], &[])
                .rd(&n.inputs[0], ix(&["i"]))
                .wr(&n.outputs[0], ix(&["i"]))
                .body(cast),
        );
        return Ok((kb.done(), h.work(total, cast).h));
    };
    let groups = (total / u128::from(g)) as u64;
    let amax = kb.temp("amax", vec![groups], ElemType::FP32);
    let scale = kb.temp("scale", vec![groups], ElemType::FP32);
    let dims = [("b", groups), ("i", g)];
    kb.push(
        kernel(KernelClass::Reduce, &dims, &["i"])
            .rd(&n.inputs[0], ix(&["b", "i"]))
            .wr(&amax, ix(&["b"]))
            .body(ScalarBody { max: 1, ..Z })
            .comb(Combiner::Max),
    );
    kb.push(
        kernel(KernelClass::Map, &[("b", groups)], &[])
            .rd(&amax, ix(&["b"]))
            .wr(&scale, ix(&["b"]))
            .body(ScalarBody { cvt: 1, ..Z }),
    );
    kb.push(
        kernel(KernelClass::Map, &dims, &[])
            .rd(&n.inputs[0], ix(&["b", "i"]))
            .rd(&scale, ix(&["b"]))
            .wr(&n.outputs[0], ix(&["b", "i"]))
            .body(ScalarBody {
                mul: 1,
                cvt: 1,
                ..Z
            }),
    );
    let h = h
        .work(
            total,
            ScalarBody {
                max: 1,
                mul: 1,
                cvt: 1,
                ..Z
            },
        )
        .work(groups.into(), ScalarBody { cvt: 1, ..Z });
    Ok((kb.done(), h.h))
}

fn dequantize(ctx: &NodeCtx) -> Out {
    arity(ctx, 1, 1)?;
    let x = &ctx.inputs[0].shape;
    expect_out(ctx, 0, x)?;
    let total = numel(x);
    let body = ScalarBody {
        mul: 1,
        cvt: 1,
        ..Z
    };
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Map, &[("i", total as u64)], &[])
            .rd(&ctx.node.inputs[0], ix(&["i"]))
            .wr(&ctx.node.outputs[0], ix(&["i"]))
            .body(body),
    );
    Ok((
        kb.done(),
        H::new(ctx).all_inputs().all_outputs().work(total, body).h,
    ))
}

// ---------------------------------------------------------------- collectives

pub fn group_size(ctx: &NodeCtx, a: &CollectiveAttrs) -> Result<u64, Diagnostic> {
    match &a.group {
        Group::Explicit(groups) => groups
            .first()
            .map(|g| g.len() as u64)
            .filter(|&n| n > 0)
            .ok_or_else(|| err("E-WL-SHAPE-001", ctx, "empty explicit group")),
        Group::MeshAxes(_) => ctx.group_size.ok_or_else(|| {
            err(
                "E-WL-SYM-001",
                ctx,
                "collective over mesh axes needs the plan's group size",
            )
        }),
    }
}

/// Ideal per-rank send bytes (02 §6 table); `s` is the full logical tensor bytes.
pub fn ideal_send_bytes(kind: CollKind, n: u64, s: u128) -> u128 {
    let (n, m) = (u128::from(n), u128::from(n.saturating_sub(1)));
    match kind {
        CollKind::AllReduce => (2 * m * s).div_ceil(n),
        CollKind::AllGather | CollKind::ReduceScatter | CollKind::AllToAll => (m * s).div_ceil(n),
        CollKind::Broadcast | CollKind::Reduce | CollKind::AllToAllV => s,
    }
}

fn collective(ctx: &NodeCtx, a: &CollectiveAttrs) -> Out {
    arity(ctx, 1, 1)?;
    let nr = group_size(ctx, a)?;
    let (i, o) = (&ctx.inputs[0].shape, &ctx.outputs[0].shape);
    let ax = |rank: usize| a.axis.map_or(Ok(0), |x| axis_of(ctx, x, rank));
    let ok = match a.kind {
        CollKind::AllReduce | CollKind::Broadcast | CollKind::Reduce => i == o,
        CollKind::AllToAll | CollKind::AllToAllV => {
            a.kind == CollKind::AllToAllV || numel(i) == numel(o)
        }
        CollKind::AllGather => {
            let x = ax(i.len())?;
            i.len() == o.len()
                && (0..i.len()).all(|d| {
                    if d == x {
                        o[d] == i[d] * nr
                    } else {
                        o[d] == i[d]
                    }
                })
        }
        CollKind::ReduceScatter => {
            let x = ax(i.len())?;
            i.len() == o.len()
                && (0..i.len()).all(|d| {
                    if d == x {
                        i[d] == o[d] * nr
                    } else {
                        o[d] == i[d]
                    }
                })
        }
    };
    if !ok {
        return Err(err(
            "E-WL-SHAPE-001",
            ctx,
            format!("{:?} over {nr} ranks cannot map {i:?} to {o:?}", a.kind),
        ));
    }
    let n = ctx.node;
    let mut kb = Kb::new(ctx);
    let dims_of = |s: &[u64]| -> Vec<(String, u64)> {
        s.iter()
            .enumerate()
            .map(|(k, &d)| (format!("d{k}"), d))
            .collect()
    };
    for (t, s, acc) in [
        (&n.inputs[0], i, Access::Read),
        (&n.outputs[0], o, Access::Write),
    ] {
        let d = dims_of(s);
        let dr: Vec<(&str, u64)> = d.iter().map(|(a, b)| (a.as_str(), *b)).collect();
        let names: Vec<&str> = dr.iter().map(|x| x.0).collect();
        kb.push(kernel(KernelClass::Collective, &dr, &[]).op(t, acc, ix(&names)));
    }
    let mut h = H::new(ctx).all_inputs().all_outputs();
    let reducing = matches!(
        a.kind,
        CollKind::AllReduce | CollKind::ReduceScatter | CollKind::Reduce
    );
    if reducing {
        let ops = (u128::from(nr - 1) * numel(i)).div_ceil(u128::from(nr));
        let body = match a.reduce {
            Some(ReduceOp::Max | ReduceOp::Min) => ScalarBody { max: 1, ..Z },
            _ => ScalarBody { add: 1, ..Z },
        };
        kb.push(
            kernel(KernelClass::Collective, &[("i", ops as u64)], &[])
                .body(body)
                .comb(Combiner::Sum),
        );
        h = h.work(ops, body);
    }
    Ok((kb.done(), h.h))
}

fn send_recv(ctx: &NodeCtx) -> Out {
    arity(ctx, 1, 1)?;
    let s = &ctx.inputs[0].shape;
    expect_out(ctx, 0, s)?;
    let total = numel(s) as u64;
    let mut kb = Kb::new(ctx);
    kb.push(
        kernel(KernelClass::Collective, &[("i", total)], &[])
            .rd(&ctx.node.inputs[0], ix(&["i"]))
            .wr(&ctx.node.outputs[0], ix(&["i"])),
    );
    Ok((kb.done(), H::new(ctx).all_inputs().all_outputs().h))
}

fn opaque(ctx: &NodeCtx, a: &OpaqueAttrs) -> Out {
    let mut k = kernel(KernelClass::Opaque, &[], &[]);
    k.opaque_cost = Some(a.cost);
    let mut kb = Kb::new(ctx);
    kb.push(k);
    Ok((kb.done(), a.cost))
}
