//! Counting rules (02 §4.1) applied to lowered kernels, and the op-class split (02 §4.2).

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::op_class::OpClass;
use kiln_ir::wl::{Combiner, CostHint, IndexExpr, Kernel, KernelClass};

use crate::lower::{Lowered, NodeCtx};

/// Compulsory bytes one node moves for one external tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Traffic {
    pub tensor: Id,
    pub read: u128,
    pub written: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeCost {
    pub cost: CostHint,
    pub traffic: Vec<Traffic>,
}

/// Distinct elements of `tensor` (of `numel` elements) touched with the given access: summed over a kernel's
/// operands with distinct index maps (identical maps address the same region; affine regions overlap at most
/// up to the whole tensor), maxed across kernels (later kernels re-touch an earlier kernel's region). Indirect
/// accesses count every gathered element.
fn touched(kernels: &[Kernel], tensor: &Id, numel: u128, reads: bool) -> Result<u128, Diagnostic> {
    let mut best = 0u128;
    for k in kernels
        .iter()
        .filter(|k| k.class != KernelClass::Layout && k.opaque_cost.is_none())
    {
        let mut sum = 0u128;
        let mut seen = Vec::new();
        let mut indirect = false;
        for op in k.operands.iter().filter(|o| {
            &o.tensor == tensor
                && if reads {
                    o.access.reads()
                } else {
                    o.access.writes()
                }
        }) {
            indirect |= op.index.iter().any(|e| matches!(e, IndexExpr::Indirect { .. }));
            if !seen.contains(&&op.index) {
                seen.push(&op.index);
                sum += k.operand_footprint(op)?;
            }
        }
        best = best.max(if indirect { sum } else { sum.min(numel) });
    }
    Ok(best)
}

/// Counts of a lowered node under 02 §4.1, including compulsory bytes per external tensor.
pub fn node_cost(ctx: &NodeCtx, l: &Lowered) -> Result<NodeCost, Diagnostic> {
    let mut cost = CostHint::default();
    for k in &l.kernels {
        let w = k.work()?;
        cost.flops_mm += w.flops_mm;
        cost.vec_ops += w.vec_ops;
        cost.transc += w.transc;
        cost.convert += w.convert;
        if let Some(c) = k.opaque_cost {
            cost.bytes_in += c.bytes_in;
            cost.bytes_out += c.bytes_out;
            cost.weight_bytes += c.weight_bytes;
        }
    }
    let mut traffic: Vec<Traffic> = Vec::new();
    let n = ctx.node;
    for (i, t) in n.inputs.iter().enumerate() {
        if n.inputs[..i].contains(t) {
            continue;
        }
        let info = &ctx.inputs[i];
        let b = info.bytes(touched(&l.kernels, t, info.numel(), true)?);
        cost.bytes_in += b;
        if info.class == kiln_ir::wl::TensorClass::Weight {
            cost.weight_bytes += b;
        }
        traffic.push(Traffic {
            tensor: t.clone(),
            read: b,
            written: 0,
        });
    }
    for (i, t) in n.outputs.iter().enumerate() {
        if n.outputs[..i].contains(t) {
            continue;
        }
        let b = ctx.outputs[i].bytes(touched(&l.kernels, t, ctx.outputs[i].numel(), false)?);
        cost.bytes_out += b;
        match traffic.iter_mut().find(|x| &x.tensor == t) {
            Some(x) => x.written = b,
            None => traffic.push(Traffic {
                tensor: t.clone(),
                read: 0,
                written: b,
            }),
        }
    }
    Ok(NodeCost { cost, traffic })
}

/// Work per 01 op class for one kernel (02 §4.2). Combiner arithmetic is in the body, so `Reduction` counts
/// reduced elements on top of the body's elementwise ops, as the table states.
pub fn class_demands(k: &Kernel) -> Result<Vec<(OpClass, u128)>, Diagnostic> {
    let w = k.work()?;
    let mut out = Vec::new();
    let mut add = |c: OpClass, v: u128| {
        if v > 0 {
            out.push((c, v));
        }
    };
    let collective = k.class == KernelClass::Collective;
    add(OpClass::Matmul, w.flops_mm / 2);
    add(
        if collective {
            OpClass::CollectiveReduce
        } else {
            OpClass::Elementwise
        },
        w.vec_ops,
    );
    add(OpClass::Transcendental, w.transc);
    add(OpClass::Convert, w.convert);
    match (k.class, k.combine) {
        (_, Some(Combiner::TopK(_))) => add(OpClass::SortTopk, w.points),
        (_, Some(Combiner::Scan)) => add(OpClass::Scan, w.points),
        (KernelClass::Reduce, _) => add(OpClass::Reduction, w.points),
        (KernelClass::Layout, _) => add(OpClass::Permute, w.points),
        (KernelClass::Gather | KernelClass::Scatter, _) => add(OpClass::GatherScatter, w.points),
        _ => {}
    }
    Ok(out)
}
