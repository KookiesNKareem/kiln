//! `OpNest` construction from lowered kernels and the operand streams the model costs (elements + MX scales).

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{PrecisionSpec, ScaleLayout};
use kiln_ir::wl::{Access, DimKind, Domain, IndexExpr, Kernel, KernelClass};

use crate::types::*;

pub(crate) fn err(code: &str, msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(code, msg)
}

fn axis_of(e: &IndexExpr, kernel: &Kernel) -> Result<AxisExpr, Diagnostic> {
    let ix = |name: &str| {
        kernel
            .dims
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| err("E-COST-NEST", format!("kernel {}: index uses unknown dim {name:?}", kernel.id)))
    };
    match e {
        IndexExpr::Affine { terms, offset } => {
            let mut out = AxisExpr { terms: vec![], div: 1, offset: *offset };
            for t in terms {
                match &t.dim {
                    Some(d) => out.terms.push((ix(d)?, t.coeff)),
                    None if t.param.is_none() => out.offset += t.coeff,
                    None => {}
                }
            }
            Ok(out)
        }
        IndexExpr::FloorDiv { inner, by } => {
            let mut a = axis_of(inner, kernel)?;
            a.div = a.div.saturating_mul((*by).max(1));
            Ok(a)
        }
        // Indirect rows: the footprint over the index dims is an upper bound on the gathered rows.
        IndexExpr::Indirect { index, .. } => {
            let mut out = AxisExpr { terms: vec![], div: 1, offset: 0 };
            for i in index {
                for (d, c) in axis_of(i, kernel)?.terms {
                    if !out.terms.iter().any(|&(x, _)| x == d) {
                        out.terms.push((d, c));
                    }
                }
            }
            Ok(out)
        }
    }
}

impl OpNest {
    pub(crate) fn from_kernel_impl(
        kernel: &Kernel,
        dtypes: &[PrecisionSpec],
        roles: Option<&[OperandRole]>,
    ) -> Result<OpNest, Diagnostic> {
        if dtypes.len() != kernel.operands.len() {
            return Err(err(
                "E-COST-NEST",
                format!("kernel {}: {} dtypes for {} operands", kernel.id, dtypes.len(), kernel.operands.len()),
            ));
        }
        if let Some(r) = roles
            && r.len() != kernel.operands.len()
        {
            return Err(err("E-COST-NEST", format!("kernel {}: {} roles for {} operands", kernel.id, r.len(), kernel.operands.len())));
        }
        let contraction = kernel.class == KernelClass::Contraction;
        let mut reads = 0;
        let mut operands = Vec::with_capacity(kernel.operands.len());
        for (i, op) in kernel.operands.iter().enumerate() {
            let is_output = op.access != Access::Read;
            let role = match roles {
                Some(r) => r[i],
                None if is_output => {
                    if contraction {
                        OperandRole::O
                    } else {
                        OperandRole::Out
                    }
                }
                None => {
                    reads += 1;
                    match (contraction, reads) {
                        (true, 1) => OperandRole::A,
                        (true, 2) => OperandRole::B,
                        _ => OperandRole::In,
                    }
                }
            };
            let axes = op.index.iter().map(|e| axis_of(e, kernel)).collect::<Result<Vec<_>, _>>()?;
            operands.push(NestOperand {
                tensor: op.tensor.to_string(),
                role,
                axes,
                dtype: dtypes[i],
                is_output,
                source: None,
                sink: None,
                block_axis: None,
            });
        }
        let kind = match kernel.class {
            KernelClass::Contraction => NestKind::Contraction,
            KernelClass::Map => NestKind::Map,
            KernelClass::Reduce => NestKind::Reduce,
            _ => NestKind::Other,
        };
        let b = &kernel.body;
        let points = match kernel.domain {
            Domain::Box => None,
            _ => Some(u64::try_from(kernel.points()?).unwrap_or(u64::MAX)),
        };
        // Segmented domains: the sequence dim spans all segments, other dims their largest per-segment extent.
        let extent = |name: &str, base: u64| match &kernel.domain {
            Domain::Segmented { seg_dim, segments } => {
                let per = segments.iter().map(|s| s.extents.iter().find(|(n, _)| n == name).map_or(base, |&(_, v)| v));
                if name == seg_dim { per.sum() } else { per.max().unwrap_or(base) }
            }
            _ => base,
        };
        Ok(OpNest {
            kind,
            dims: kernel
                .dims
                .iter()
                .map(|d| NestDim {
                    name: d.name.clone(),
                    size: extent(&d.name, d.extent),
                    kind: match d.kind {
                        DimKind::Parallel => LoopKind::Parallel,
                        DimKind::Reduction => LoopKind::Reduction,
                    },
                })
                .collect(),
            operands,
            macs_per_point: if contraction { u32::from(b.mac) } else { 0 },
            vector_ops_per_point: u32::from(b.vector()) + u32::from(b.transcendental()) + u32::from(b.cvt),
            points,
            accum: if contraction { kernel.accum } else { None },
        })
    }

    pub(crate) fn tile_impl(&self, sizes: &[u64], points: Option<u64>) -> OpNest {
        let mut n = self.clone();
        for (d, &s) in n.dims.iter_mut().zip(sizes) {
            d.size = s;
        }
        n.points = points;
        n
    }

    pub(crate) fn positioned_impl(&self, lo: &[u64]) -> OpNest {
        let mut n = self.clone();
        for (o, src) in n.operands.iter_mut().zip(&self.operands) {
            let block = (!o.is_output).then(|| o.dtype.block_size()).flatten().map_or(1, u64::from);
            let bx = if block > 1 { src.block_axis.or_else(|| block_axis(self, src)) } else { None };
            for (i, a) in o.axes.iter_mut().enumerate() {
                let shift: i64 = a.terms.iter().map(|&(d, c)| c * lo.get(d).map_or(0, |&x| x as i64)).sum();
                let div = a.div.saturating_mul(if bx == Some(i) { block } else { 1 });
                a.offset = if div > 1 { (a.offset + shift).rem_euclid(div as i64) } else { 0 };
            }
        }
        n
    }

    pub fn box_points(&self) -> u64 {
        self.dims.iter().map(|d| d.size).product()
    }

    pub fn useful_points(&self) -> u64 {
        self.points.unwrap_or_else(|| self.box_points())
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        if self.dims.len() > 64 {
            return Err(err("E-COST-NEST", "more than 64 loop dims"));
        }
        if self.operands.iter().filter(|o| o.is_output).count() != 1 {
            return Err(err("E-COST-NEST", "a nest needs exactly one output operand").hint("lower fused groups to several nests"));
        }
        for o in &self.operands {
            for a in &o.axes {
                if a.terms.iter().any(|&(d, _)| d >= self.dims.len()) {
                    return Err(err("E-COST-NEST", format!("operand {} indexes a dim out of range", o.tensor)));
                }
            }
        }
        Ok(())
    }
}

/// One costed data stream: a nest operand's elements, or the block-scale stream of an MX/block operand.
#[derive(Clone, Debug)]
pub(crate) struct Stream {
    pub operand: usize,
    pub role: OperandRole,
    pub axes: Vec<AxisExpr>,
    /// Bitmask of dims the stream's index depends on.
    pub rel: u64,
    pub is_output: bool,
    /// MX/block scale stream of `operand`.
    pub scale: bool,
    /// Every axis is one dim with unit coefficient and no division, each dim used once: the footprint is the
    /// product of the relevant extents.
    pub simple: bool,
    /// Storage bits per element at rest (final dtype for outputs).
    pub bits: u32,
    /// Accumulator bits for partial sums (outputs only).
    pub acc_bits: u32,
    /// Most temporal iterations the stream stays in the array between level-0 reads (`UnitTemplate::operand_run`).
    pub run: u64,
}

impl Stream {
    pub fn relevant(&self, d: usize) -> bool {
        self.rel & (1 << d) != 0
    }

    /// Distinct elements touched by a tile with per-dim extents `ext` at the origin (exact for one term per axis,
    /// an upper bound for sums): per axis, at most the coordinates the terms enumerate and at most their span,
    /// floor-divided where the span (shifted by the offset) crosses `div` boundaries.
    pub fn footprint(&self, ext: &[u64]) -> u64 {
        self.axes
            .iter()
            .map(|a| {
                let span: u64 = a.terms.iter().map(|&(d, c)| c.unsigned_abs() * (ext[d] - 1)).sum::<u64>() + 1;
                let points: u64 = a.terms.iter().filter(|t| t.1 != 0).map(|&(d, _)| ext[d]).product();
                let distinct = points.min(span);
                if a.div <= 1 {
                    return distinct;
                }
                let lo = a.offset + a.terms.iter().filter(|t| t.1 < 0).map(|&(d, c)| c * (ext[d] as i64 - 1)).sum::<i64>();
                let div = a.div as i64;
                let blocks = (lo + span as i64 - 1).div_euclid(div) - lo.div_euclid(div) + 1;
                distinct.min(blocks as u64)
            })
            .product()
    }
}

fn is_simple(axes: &[AxisExpr]) -> bool {
    let mut seen = 0u64;
    axes.iter().all(|a| {
        let ok = a.div == 1 && a.terms.len() == 1 && a.terms[0].1.abs() == 1 && seen & (1 << a.terms[0].0) == 0;
        if ok {
            seen |= 1 << a.terms[0].0;
        }
        ok
    })
}

fn rel_mask(axes: &[AxisExpr]) -> u64 {
    axes.iter().flat_map(|a| a.terms.iter()).filter(|&&(_, c)| c != 0).fold(0, |m, &(d, _)| m | (1 << d))
}

pub(crate) fn streams(nest: &OpNest, acc_bits: u32) -> Vec<Stream> {
    let mut v: Vec<Stream> = nest
        .operands
        .iter()
        .enumerate()
        .map(|(i, o)| Stream {
            operand: i,
            role: o.role,
            rel: rel_mask(&o.axes),
            axes: o.axes.clone(),
            is_output: o.is_output,
            scale: false,
            simple: is_simple(&o.axes),
            bits: o.dtype.precision.element_bits(),
            acc_bits: if o.is_output { acc_bits.max(o.dtype.precision.element_bits()) } else { 0 },
            run: u64::MAX,
        })
        .collect();
    for (i, o) in nest.operands.iter().enumerate().filter(|(_, o)| !o.is_output) {
        let Some(block) = o.dtype.block_size() else { continue };
        let ScaleLayout::Block { scale_bits, zero_point_bits, .. } = o.dtype.precision.scale_layout(block) else {
            continue;
        };
        let mut axes = o.axes.clone();
        let Some(bx) = o.block_axis.or_else(|| block_axis(nest, o)) else { continue };
        axes[bx].div = axes[bx].div.saturating_mul(u64::from(block));
        v.push(Stream {
            operand: i,
            role: o.role,
            rel: rel_mask(&axes),
            simple: is_simple(&axes),
            axes,
            is_output: false,
            scale: true,
            bits: scale_bits + zero_point_bits,
            acc_bits: 0,
            run: u64::MAX,
        });
    }
    v
}

fn block_axis(nest: &OpNest, o: &NestOperand) -> Option<usize> {
    let red = |a: &AxisExpr| a.terms.iter().any(|&(d, _)| nest.dims[d].kind == LoopKind::Reduction);
    o.axes.iter().rposition(red).or_else(|| o.axes.len().checked_sub(1))
}

/// Canonical contraction class of a dim, used when a template axis names `m`/`n`/`k`/`b` but the kernel uses
/// other names: `k` reduction, `n` parallel dim of the second input, `m` of the first, `b` of both.
pub(crate) fn dim_class(nest: &OpNest, d: usize) -> Option<&'static str> {
    if nest.dims[d].kind == LoopKind::Reduction {
        return Some("k");
    }
    let uses = |r: OperandRole| nest.operands.iter().any(|o| o.role == r && rel_mask(&o.axes) & (1 << d) != 0);
    match (uses(OperandRole::A), uses(OperandRole::B)) {
        (true, true) => Some("b"),
        (true, false) => Some("m"),
        (false, true) => Some("n"),
        _ => None,
    }
}
