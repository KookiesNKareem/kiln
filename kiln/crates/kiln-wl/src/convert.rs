//! Explicit precision conversion before contractions (02 §5.7, 03 §2.7: never a silent upcast). A
//! contraction whose operand pair no MAC mode of the design runs (bf16 activations x fp8 weights on an A100, an
//! fp8 KV cache under bf16 queries on a TPU) gets a `convert` map kernel per operand that needs widening, into a
//! node-local temp of the mode's precision that the contraction then reads. The convert is executed work
//! (convert class on the vector units) and its temp never leaves the chip, so the off-chip bytes stay those of
//! the stored dtype. A native mode (an exact pair, or the element type of an MX/scaled operand) needs nothing.

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::precision::{Precision, PrecisionKind, PrecisionSpec};
use kiln_ir::wl::{
    Access, DiffConstraint, Domain, ElemType, Kernel, KernelClass, LoopDim, Operand, ScalarBody, Scaling, SegmentDomain,
    TensorClass, TypeInfo,
};

use crate::graph::{LoweredGraph, LoweredNode};

/// A MAC operand pair a design runs, its accumulator, and its aggregate throughput (MAC/s), the tie-break among
/// widenings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacMode {
    pub a: Precision,
    pub b: Precision,
    pub acc: Precision,
    pub rate: f64,
}

/// MAC modes a design runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MacModes {
    pub modes: Vec<MacMode>,
    /// The software stack has no mixed-input kernels (02 §7.4.1 `dequantize`): operands that differ from the
    /// other side are converted even when a mixed mode exists.
    pub dequantize: bool,
}

/// The registry precision a MAC mode must name for operands of this element type: the shorthand it expands
/// from (`mxfp4`, `fp8_e4m3_pt`, `int4_g128`), else its scalar. A scaled type with no registry name maps to its
/// scalar here, so a contraction must never read one directly ([`unregistered_scaling`]): [`choose`] converts it.
pub fn operand_spec(t: &ElemType) -> PrecisionSpec {
    t.shorthand().unwrap_or(PrecisionSpec::new(t.scalar))
}

/// A scaled element type that is no registry name: no MAC mode names it and the cost model's precision for it
/// ([`operand_spec`]) drops its scales, so it reaches a contraction only through a convert that applies them.
pub fn unregistered_scaling(t: &ElemType) -> bool {
    !matches!(t.scaling, Scaling::None) && t.shorthand().is_none()
}

/// `acc` holds every value an accumulator of precision `required` holds: the same precision, a float at least
/// as wide in exponent and significand, or a wider integer. kiln-cost's mode choice uses the same rule.
pub fn accumulates(acc: Precision, required: Precision) -> bool {
    if acc == required {
        return true;
    }
    match (acc.exponent_bits(), required.exponent_bits()) {
        (Some(ea), Some(er)) => ea >= er && acc.element_bits() - ea >= required.element_bits() - er,
        (None, None) => acc.is_integer() && required.is_integer() && acc.element_bits() >= required.element_bits(),
        _ => false,
    }
}

/// Element type of an MX or block-scaled name, compute precision of a storage variant.
pub fn base(p: Precision) -> Precision {
    use Precision::*;
    match p {
        Mxfp8E4m3 => Fp8E4m3,
        Mxfp8E5m2 => Fp8E5m2,
        Mxfp6E3m2 => Fp6E3m2,
        Mxfp6E2m3 => Fp6E2m3,
        Mxfp4 | Nvfp4 => Fp4E2m1,
        Mxint8 => Int8,
        p => p.compute(),
    }
}

/// A mode operand precision `mode` runs operands of `op` without conversion: the same name, or (as kiln-cost's
/// mode choice) the element type of an MX/block/scaled operand, whose scales then cost a vector pass.
pub fn accepts(mode: Precision, op: Precision) -> bool {
    mode == op || (mode == base(op) && mode.kind() != PrecisionKind::Mx)
}

/// (exponent bits, significand bits incl. the implicit one) of a float element type; integers as (0, bits).
/// Integer signedness is checked separately ([`int_holds`]).
fn format(p: Precision) -> Option<(u32, u32)> {
    use Precision::*;
    Some(match p {
        Fp32 => (8, 24),
        Tf32 => (8, 11),
        Bf16 => (8, 8),
        Fp16 => (5, 11),
        Fp8E4m3 => (4, 4),
        Fp8E5m2 => (5, 3),
        Fp6E3m2 => (3, 3),
        Fp6E2m3 => (2, 4),
        Fp4E2m1 => (2, 2),
        Int4 | Uint4 => (0, 4),
        Int8 | Uint8 => (0, 8),
        Int16 => (0, 16),
        Int32 => (0, 32),
        _ => return None,
    })
}

/// Integer `to` holds every value of integer `from`: an unsigned target needs an unsigned source; a signed target
/// needs one more bit than an unsigned source.
fn int_holds(to: Precision, from: Precision) -> bool {
    let unsigned = |p: Precision| matches!(p, Precision::Uint4 | Precision::Uint8);
    let (bt, bf) = (to.element_bits(), from.element_bits());
    match (unsigned(to), unsigned(from)) {
        (true, true) | (false, false) => bt >= bf,
        (false, true) => bt > bf,
        (true, false) => false,
    }
}

/// `to` represents every value of `from` exactly (a lossless widening, so converting is not a silent change of
/// numerics). Block-scaled sources widen into 8-bit-exponent floats that hold their elements and their
/// E8M0/bf16/fp8 scales (applied inside the reduction), or into an MX type of the same block whose element type
/// holds theirs; per-tensor and per-axis scaled sources into those floats when they hold the elements (the scale
/// applies to the accumulator).
pub fn widens(to: Precision, from: &ElemType) -> bool {
    let src = operand_spec(from).precision;
    if to == src && !unregistered_scaling(from) {
        return true;
    }
    let wide = matches!(to, Precision::Fp32 | Precision::Tf32 | Precision::Bf16);
    let holds = |t: Precision, f: Precision| match (format(t), format(f)) {
        (Some((0, _)), Some((0, _))) => int_holds(t, f),
        (Some((_, mt)), Some((0, bits))) => mt >= bits,
        (Some((et, mt)), Some((ef, mf))) => et >= ef && mt >= mf && ef > 0 && et > 0,
        _ => false,
    };
    match from.scaling {
        Scaling::None => holds(to, from.scalar) && (wide || to.kind() != PrecisionKind::Mx),
        Scaling::Block { scale: Precision::E8m0, block, zero_point: None, tensor_scale: None, .. } if src.kind() == PrecisionKind::Mx => {
            wide || (block == 32 && to.kind() == PrecisionKind::Mx && holds(base(to), from.scalar))
        }
        Scaling::PerTensor { .. } | Scaling::PerAxis { .. } => wide && holds(to, from.scalar),
        Scaling::Block { scale, zero_point, .. } => {
            wide && holds(to, from.scalar) && (scale == Precision::E8m0 || holds(to, scale)) && zero_point.is_none_or(|z| holds(to, z))
        }
    }
}

/// Target precisions of the `a` and `b` operands (`None`: unconverted).
pub type Converts = (Option<Precision>, Option<Precision>);

/// The (a, b) mode a contraction runs in: a native one if any accepts the operands, else the fastest lossless
/// widening (fewer converted operands, then the narrower pair, on ties). `None`: no mode can run it.
pub fn choose(modes: &MacModes, a: &ElemType, b: &ElemType) -> Option<Converts> {
    choose_with(modes, a, b, None, [true, true])
}

/// [`choose`] for a contraction that requires accumulator `accum` (only modes whose accumulator holds it run it,
/// as in kiln-cost's mode choice), where `upcast[i]` false (the tensor's `upcast_ok: false`, 02 §3) forbids
/// widening that operand. An operand of an unregistered scaled type is always converted (its scales applied).
pub fn choose_with(modes: &MacModes, a: &ElemType, b: &ElemType, accum: Option<Precision>, upcast: [bool; 2]) -> Option<Converts> {
    let (pa, pb) = (operand_spec(a).precision, operand_spec(b).precision);
    let (raw_a, raw_b) = (unregistered_scaling(a), unregistered_scaling(b));
    let usable: Vec<MacMode> = modes.modes.iter().copied().filter(|m| accum.is_none_or(|r| accumulates(m.acc, r))).collect();
    let same = base(pa) == base(pb) || pa.compute() == pb.compute();
    let native = !raw_a && !raw_b && usable.iter().any(|m| accepts(m.a, pa) && accepts(m.b, pb));
    if native && (!modes.dequantize || same) {
        return Some((None, None));
    }
    let mut best: Option<((f64, i32, i64), Converts)> = None;
    for &MacMode { a: ma, b: mb, rate, .. } in &usable {
        if modes.dequantize && base(ma) != base(mb) {
            continue;
        }
        let ca = if accepts(ma, pa) && !modes.dequantize && !raw_a { None } else if widens(ma, a) { Some(ma) } else { continue };
        let cb = if accepts(mb, pb) && !modes.dequantize && !raw_b { None } else if widens(mb, b) { Some(mb) } else { continue };
        // Under `dequantize` an operand already at the mode's precision needs no convert either.
        let (ca, cb) = (ca.filter(|&m| m != pa || raw_a), cb.filter(|&m| m != pb || raw_b));
        if (ca.is_some() && !upcast[0]) || (cb.is_some() && !upcast[1]) {
            continue;
        }
        let key = (rate, -(i32::from(ca.is_some()) + i32::from(cb.is_some())), -i64::from(ma.element_bits() + mb.element_bits()));
        let better = best.as_ref().is_none_or(|(k, _)| key.0.total_cmp(&k.0).then(key.1.cmp(&k.1)).then(key.2.cmp(&k.2)).is_gt());
        if better {
            best = Some((key, (ca, cb)));
        }
    }
    best.map(|b| b.1)
}

/// Type and class of a node-local tensor name (inputs, outputs, temps), with the model-level class of inputs.
fn local(n: &LoweredNode, name: &Id) -> Option<TypeInfo> {
    let io = n.node.inputs.iter().zip(&n.inputs).chain(n.node.outputs.iter().zip(&n.outputs));
    let from_io = io.into_iter().find(|(l, _)| *l == name).map(|(l, ti)| {
        let mut t = ti.clone();
        if let Some((_, o)) = n.origins.iter().find(|(x, _)| x == l) {
            t.class = o.class;
        }
        t
    });
    from_io.or_else(|| n.lowered.temps.iter().find(|(l, _)| l == name).map(|(_, t)| t.clone()))
}

/// Read operands `(a, b)` of a contraction as the mapper assigns them: `b` is the model-state side (weights, KV
/// cache) when exactly one is.
fn roles(k: &Kernel, n: &LoweredNode) -> Option<(usize, usize)> {
    let reads: Vec<usize> = (0..k.operands.len()).filter(|&i| k.operands[i].access == Access::Read).collect();
    let (&a, &b) = (reads.first()?, reads.get(1)?);
    let state = |i: usize| local(n, &k.operands[i].tensor).is_some_and(|t| t.class.is_model_level());
    Some(if state(a) && !state(b) { (b, a) } else { (a, b) })
}

/// Each constraint projected onto the kept dims: an eliminated dim's term is replaced by its least value over
/// `[0, extent)`, so the result holds wherever some value of the eliminated dims satisfies the constraint (a
/// sliding window keeps its KV band; a causal mask's KV reads become its reach). Constraints left with no
/// term are dropped when they hold.
fn project(cs: &[DiffConstraint], keep: &dyn Fn(&str) -> bool, extent: &dyn Fn(&str) -> u64) -> Vec<DiffConstraint> {
    cs.iter()
        .filter_map(|c| {
            let (kept, gone): (Vec<_>, Vec<_>) = c.terms.iter().cloned().partition(|(_, d)| keep(d));
            let least: i64 = gone.iter().map(|(a, d)| (i64::from(*a) * extent(d).saturating_sub(1) as i64).min(0)).sum();
            let rhs = c.rhs - least;
            (!kept.is_empty() || rhs < 0).then_some(DiffConstraint { terms: kept, rhs })
        })
        .collect()
}

/// The map kernel converting what contraction `k` reads of operand `oi` into `to`: the contraction's dims that
/// index the operand, over its domain projected onto them, the operand's own index map on both sides.
fn convert_kernel(k: &Kernel, oi: usize, src: &ElemType, to: Id, id: String) -> Kernel {
    let op = &k.operands[oi];
    let used = op.dims();
    let keep = |d: &str| used.iter().any(|u| u == d);
    let dims: Vec<LoopDim> = k.dims.iter().filter(|d| keep(&d.name)).map(|d| LoopDim { kind: kiln_ir::wl::DimKind::Parallel, ..d.clone() }).collect();
    let base = |d: &str| k.extent(d).unwrap_or(1);
    let domain = match &k.domain {
        Domain::Box => Domain::Box,
        Domain::Constrained(cs) => {
            let c = project(cs, &keep, &base);
            if c.is_empty() { Domain::Box } else { Domain::Constrained(c) }
        }
        Domain::Segmented { seg_dim, segments } if keep(seg_dim) => Domain::Segmented {
            seg_dim: seg_dim.clone(),
            segments: segments
                .iter()
                .map(|s| SegmentDomain {
                    extents: s.extents.iter().filter(|(d, _)| keep(d)).cloned().collect(),
                    params: s.params.clone(),
                    constraints: project(&s.constraints, &keep, &|d| s.extents.iter().find(|(n, _)| n == d).map_or_else(|| base(d), |x| x.1)),
                })
                .collect(),
        },
        Domain::Segmented { .. } => Domain::Box,
    };
    let dims = match &k.domain {
        // The operand does not vary with the segment: one box at each dim's largest extent.
        Domain::Segmented { seg_dim, segments } if !keep(seg_dim) => dims
            .into_iter()
            .map(|mut d| {
                d.extent = segments.iter().filter_map(|s| s.extents.iter().find(|(n, _)| *n == d.name).map(|x| x.1)).fold(d.extent, u64::max);
                d
            })
            .collect(),
        _ => dims,
    };
    let scaled = !matches!(src.scaling, Scaling::None);
    Kernel {
        id,
        dims,
        domain,
        operands: vec![Operand::new(&op.tensor, Access::Read, op.index.clone()), Operand::new(&to, Access::Write, op.index.clone())],
        body: ScalarBody { cvt: 1, mul: u16::from(scaled), ..ScalarBody::default() },
        combine: None,
        accum: None,
        class: KernelClass::Map,
        opaque_cost: None,
    }
}

/// Inserts the converts every contraction of `lg` needs under `modes`; returns how many it inserted. A
/// contraction no mode can run, even after widening, is left as is (the mapper reports it, E-MAP-OP-004), unless
/// it reads an unregistered scaled type (E-WL-DT-005: the mapper would cost it without its scales).
pub fn insert_converts(lg: &mut LoweredGraph, modes: &MacModes) -> Result<usize, Diagnostic> {
    let mut inserted = 0;
    for n in &mut lg.nodes {
        let mut ki = 0;
        while ki < n.lowered.kernels.len() {
            let k = &n.lowered.kernels[ki];
            if k.class != KernelClass::Contraction {
                ki += 1;
                continue;
            }
            let Some((a, b)) = roles(k, n) else {
                ki += 1;
                continue;
            };
            let ty = |i: usize| local(n, &k.operands[i].tensor);
            let (Some(ta), Some(tb)) = (ty(a), ty(b)) else {
                ki += 1;
                continue;
            };
            let upcast = |i: usize| n.origins.iter().find(|(x, _)| *x == k.operands[i].tensor).is_none_or(|(_, o)| o.upcast_ok);
            let Some((ca, cb)) = choose_with(modes, &ta.dtype, &tb.dtype, k.accum, [upcast(a), upcast(b)]) else {
                if let Some(t) = [&ta, &tb].into_iter().find(|t| unregistered_scaling(&t.dtype)) {
                    return Err(Diagnostic::error(
                        "E-WL-DT-005",
                        format!(
                            "contraction {} reads a {} operand with scaling {:?} that no MAC mode runs and none \
                             widens into losslessly; its scales would be dropped",
                            k.id, t.dtype.scalar, t.dtype.scaling
                        ),
                    )
                    .hint("use a registry scaled type (mxfp4, fp8_e4m3_pt, int4_g128, ...) or a design with a \
                                bf16/fp32 MAC mode the operand widens into"));
                }
                ki += 1;
                continue;
            };
            let mut add = vec![];
            for (oi, t, to) in [(a, &ta, ca), (b, &tb, cb)] {
                let Some(to) = to else { continue };
                let name = Id::new(format!("cvt{}_{}", ki, k.operands[oi].tensor.as_str().replace('.', "_")))?;
                let kernel = convert_kernel(k, oi, &t.dtype, name.clone(), format!("{}cvt{oi}", k.id));
                add.push((oi, name, TypeInfo::new(t.shape.clone(), ElemType::from(to), TensorClass::Activation), kernel));
            }
            let n_add = add.len();
            for (oi, name, ti, _) in &add {
                n.lowered.kernels[ki].operands[*oi].tensor = name.clone();
                n.lowered.temps.push((name.clone(), ti.clone()));
            }
            for (_, _, _, kernel) in add.into_iter().rev() {
                n.lowered.kernels.insert(ki, kernel);
            }
            inserted += n_add;
            ki += n_add + 1;
        }
    }
    Ok(inserted)
}
