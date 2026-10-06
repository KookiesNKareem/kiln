//! Affine kernels (02 §4): iteration space, mask-constrained domain, operand index maps, scalar body.
//!
//! Kernels are produced by lowering a bound node, so extents and constraint bounds are integers here.

use serde::{Deserialize, Serialize};

use crate::common::{Diagnostic, Id};
use crate::precision::Precision;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimKind {
    Parallel,
    Reduction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopDim {
    pub name: String,
    pub extent: u64,
    pub kind: DimKind,
}

/// `Σ coeff_i · dim_i ≤ rhs` with at most two terms and coefficients ±1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffConstraint {
    pub terms: Vec<(i8, String)>,
    pub rhs: i64,
}

impl DiffConstraint {
    /// `hi - lo ≤ rhs`.
    pub fn diff(hi: &str, lo: &str, rhs: i64) -> Self {
        Self {
            terms: vec![(1, hi.into()), (-1, lo.into())],
            rhs,
        }
    }
}

/// One segment of a `Segmented` domain: overrides of dim extents, named per-segment constants usable in index
/// maps (e.g. `past`, `tok_base`), and that segment's mask constraints.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentDomain {
    pub extents: Vec<(String, u64)>,
    #[serde(default)]
    pub params: Vec<(String, i64)>,
    #[serde(default)]
    pub constraints: Vec<DiffConstraint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Box,
    Constrained(Vec<DiffConstraint>),
    /// Union over segments of disjoint sub-domains; `seg_dim` enumerates the sequences of a segment.
    Segmented {
        seg_dim: String,
        segments: Vec<SegmentDomain>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// `coeff · dim · param` (either factor may be absent).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub coeff: i64,
    #[serde(default)]
    pub dim: Option<String>,
    #[serde(default)]
    pub param: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexExpr {
    Affine { terms: Vec<Term>, offset: i64 },
    FloorDiv { inner: Box<IndexExpr>, by: u64 },
    Indirect { via: Id, index: Vec<IndexExpr> },
}

impl IndexExpr {
    pub fn dim(d: &str) -> Self {
        Self::Affine {
            terms: vec![Term {
                coeff: 1,
                dim: Some(d.into()),
                param: None,
            }],
            offset: 0,
        }
    }

    pub fn dim_offset(d: &str, offset: i64) -> Self {
        Self::Affine {
            terms: vec![Term {
                coeff: 1,
                dim: Some(d.into()),
                param: None,
            }],
            offset,
        }
    }

    pub fn terms(terms: Vec<Term>, offset: i64) -> Self {
        Self::Affine { terms, offset }
    }

    pub fn dims(&self, out: &mut Vec<String>) {
        match self {
            Self::Affine { terms, .. } => {
                for d in terms.iter().filter_map(|t| t.dim.as_ref()) {
                    if !out.contains(d) {
                        out.push(d.clone());
                    }
                }
            }
            Self::FloorDiv { inner, .. } => inner.dims(out),
            Self::Indirect { index, .. } => index.iter().for_each(|i| i.dims(out)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operand {
    pub tensor: Id,
    pub access: Access,
    pub index: Vec<IndexExpr>,
}

impl Operand {
    pub fn new(tensor: &Id, access: Access, index: Vec<IndexExpr>) -> Self {
        Self {
            tensor: tensor.clone(),
            access,
            index,
        }
    }

    /// Identity index over the named dims.
    pub fn ident(tensor: &Id, access: Access, dims: &[&str]) -> Self {
        Self::new(
            tensor,
            access,
            dims.iter().map(|d| IndexExpr::dim(d)).collect(),
        )
    }

    pub fn dims(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.index.iter().for_each(|i| i.dims(&mut v));
        v
    }
}

/// Operation counts per iteration point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScalarBody {
    pub mac: u16,
    pub add: u16,
    pub mul: u16,
    pub fma: u16,
    pub max: u16,
    pub cmp: u16,
    pub select: u16,
    pub exp: u16,
    pub log: u16,
    pub rcp: u16,
    pub rsqrt: u16,
    pub tanh: u16,
    pub erf: u16,
    pub sin_cos: u16,
    pub cvt: u16,
}

impl ScalarBody {
    pub const fn vector(&self) -> u16 {
        self.add + self.mul + self.fma + self.max + self.cmp + self.select
    }

    pub const fn transcendental(&self) -> u16 {
        self.exp + self.log + self.rcp + self.rsqrt + self.tanh + self.erf + self.sin_cos
    }
}

impl std::ops::Add for ScalarBody {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            mac: self.mac + o.mac,
            add: self.add + o.add,
            mul: self.mul + o.mul,
            fma: self.fma + o.fma,
            max: self.max + o.max,
            cmp: self.cmp + o.cmp,
            select: self.select + o.select,
            exp: self.exp + o.exp,
            log: self.log + o.log,
            rcp: self.rcp + o.rcp,
            rsqrt: self.rsqrt + o.rsqrt,
            tanh: self.tanh + o.tanh,
            erf: self.erf + o.erf,
            sin_cos: self.sin_cos + o.sin_cos,
            cvt: self.cvt + o.cvt,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Combiner {
    Sum,
    Max,
    Min,
    Prod,
    OnlineSoftmax,
    ArgMax,
    TopK(u32),
    /// Cumulative pass (top-p threshold, SSM scans).
    Scan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelClass {
    Contraction,
    Map,
    Reduce,
    Gather,
    Scatter,
    Layout,
    Collective,
    Opaque,
}

/// Closed-form per-node cost (02 §5.1). `convert` is an addition to the spec struct so `cvt` work (quantize,
/// dequantize, casts) is covered by the hint/lowering equality test too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CostHint {
    pub flops_mm: u128,
    pub vec_ops: u128,
    pub transc: u128,
    pub convert: u128,
    pub bytes_in: u128,
    pub bytes_out: u128,
    pub weight_bytes: u128,
}

impl std::ops::Add for CostHint {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            flops_mm: self.flops_mm + o.flops_mm,
            vec_ops: self.vec_ops + o.vec_ops,
            transc: self.transc + o.transc,
            convert: self.convert + o.convert,
            bytes_in: self.bytes_in + o.bytes_in,
            bytes_out: self.bytes_out + o.bytes_out,
            weight_bytes: self.weight_bytes + o.weight_bytes,
        }
    }
}

impl std::ops::Mul<u128> for CostHint {
    type Output = Self;
    fn mul(self, k: u128) -> Self {
        Self {
            flops_mm: self.flops_mm * k,
            vec_ops: self.vec_ops * k,
            transc: self.transc * k,
            convert: self.convert * k,
            bytes_in: self.bytes_in * k,
            bytes_out: self.bytes_out * k,
            weight_bytes: self.weight_bytes * k,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kernel {
    pub id: String,
    pub dims: Vec<LoopDim>,
    pub domain: Domain,
    pub operands: Vec<Operand>,
    pub body: ScalarBody,
    #[serde(default)]
    pub combine: Option<Combiner>,
    #[serde(default)]
    pub accum: Option<Precision>,
    pub class: KernelClass,
    /// Declared cost of an `opaque` kernel (counts are not derivable from a body).
    #[serde(default)]
    pub opaque_cost: Option<CostHint>,
}

/// Arithmetic work of one kernel under the counting rules of 02 §4.1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub points: u128,
    pub flops_mm: u128,
    pub vec_ops: u128,
    pub transc: u128,
    pub convert: u128,
}

impl Kernel {
    pub fn extent(&self, dim: &str) -> Option<u64> {
        self.dims.iter().find(|d| d.name == dim).map(|d| d.extent)
    }

    /// Exact number of integer points in the domain (02 §4.3).
    pub fn points(&self) -> Result<u128, Diagnostic> {
        let all: Vec<&str> = self.dims.iter().map(|d| d.name.as_str()).collect();
        self.projection(&all)
    }

    /// Number of distinct points of the domain projected onto `keep` (exact). The segments of a `Segmented` domain
    /// are distinct iterations, so segment identity is always retained.
    pub fn projection(&self, keep: &[&str]) -> Result<u128, Diagnostic> {
        match &self.domain {
            Domain::Box => project(&self.base_extents(), &[], keep),
            Domain::Constrained(cs) => project(&self.base_extents(), cs, keep),
            Domain::Segmented { segments, .. } => segments
                .iter()
                .try_fold(0u128, |acc, s| Ok(acc + project(&self.segment_extents(s), &s.constraints, keep)?)),
        }
    }

    fn base_extents(&self) -> Vec<(&str, u64)> {
        self.dims.iter().map(|d| (d.name.as_str(), d.extent)).collect()
    }

    fn segment_extents(&self, s: &SegmentDomain) -> Vec<(&str, u64)> {
        self.dims
            .iter()
            .map(|d| (d.name.as_str(), s.extents.iter().find(|(m, _)| *m == d.name).map_or(d.extent, |&(_, v)| v)))
            .collect()
    }

    /// Distinct elements an operand touches, assuming its index map is injective over the dims it uses
    /// (true for every lowering kiln emits). Segments count separately only where the operand's index ranges
    /// (per-segment params such as `tok_base`) keep them apart; otherwise their projections are unioned.
    pub fn operand_footprint(&self, op: &Operand) -> Result<u128, Diagnostic> {
        let dims = op.dims();
        let keep: Vec<&str> = dims.iter().map(String::as_str).collect();
        let Domain::Segmented { segments, .. } = &self.domain else { return self.projection(&keep) };
        let mut live = Vec::new();
        for s in segments {
            let ext = self.segment_extents(s);
            let n = project(&ext, &s.constraints, &keep)?;
            if n > 0 {
                let ranges: Option<Vec<(i128, i128)>> = op.index.iter().map(|e| index_range(e, &ext, &s.params)).collect();
                live.push((s, ext, n, ranges));
            }
        }
        let mut group: Vec<usize> = (0..live.len()).collect();
        for i in 0..live.len() {
            for j in 0..i {
                if overlap(&live[i].3, &live[j].3) {
                    let (gi, gj) = (group[i], group[j]);
                    group.iter_mut().filter(|g| **g == gi).for_each(|g| *g = gj);
                }
            }
        }
        let mut total = 0u128;
        for g in group.iter().enumerate().filter(|(i, g)| *i == **g).map(|(_, &g)| g) {
            let members: Vec<_> = live.iter().zip(&group).filter(|(_, x)| **x == g).map(|(m, _)| m).collect();
            let (s0, ext0) = (members[0].0, &members[0].1);
            let cdims: Vec<&str> = s0.constraints.iter().flat_map(|c| c.terms.iter().map(|(_, d)| d.as_str())).collect();
            let params = op.index.iter().flat_map(index_params).collect::<Vec<_>>();
            let param = |s: &SegmentDomain, p: &str| s.params.iter().find(|(n, _)| n == p).map(|x| x.1);
            let uniform = members.iter().all(|(s, ext, ..)| {
                s.constraints == s0.constraints
                    && cdims.iter().all(|d| ext.iter().find(|x| x.0 == *d) == ext0.iter().find(|x| x.0 == *d))
                    && params.iter().all(|p| param(s, p) == param(s0, p))
            });
            let same_params = members.iter().all(|(s, ..)| params.iter().all(|p| param(s, p) == param(s0, p)));
            total += if members.len() == 1 {
                members[0].2
            } else if !same_params {
                let sets: Vec<(&SegmentDomain, &[(&str, u64)])> = members.iter().map(|(s, ext, ..)| (*s, ext.as_slice())).collect();
                index_union(&sets, &op.index, &keep)?
            } else if !uniform {
                let sets: Vec<SegSet> = members.iter().map(|(s, ext, ..)| (s.constraints.as_slice(), ext.as_slice())).collect();
                union_footprint(&sets, &keep)?
            } else {
                let kc: Vec<&str> = keep.iter().copied().filter(|d| cdims.contains(d)).collect();
                let free: Vec<&str> = keep.iter().copied().filter(|d| !cdims.contains(d)).collect();
                let boxes: Vec<Vec<u64>> =
                    members.iter().map(|(_, ext, ..)| free.iter().map(|d| ext.iter().find(|x| x.0 == *d).map_or(1, |x| x.1)).collect()).collect();
                project(ext0, &s0.constraints, &kc)? * box_union(&boxes.iter().map(Vec::as_slice).collect::<Vec<_>>())
            };
        }
        Ok(total)
    }

    pub fn work(&self) -> Result<Work, Diagnostic> {
        if let Some(c) = self.opaque_cost {
            return Ok(Work {
                points: 0,
                flops_mm: c.flops_mm,
                vec_ops: c.vec_ops,
                transc: c.transc,
                convert: c.convert,
            });
        }
        let points = self.points()?;
        let b = &self.body;
        let mm = if self.class == KernelClass::Contraction {
            2 * u128::from(b.mac) * points
        } else {
            0
        };
        Ok(Work {
            points,
            flops_mm: mm,
            vec_ops: points * u128::from(b.vector()),
            transc: points * u128::from(b.transcendental()),
            convert: points * u128::from(b.cvt),
        })
    }
}

/// Value range of an index expression over a segment, `None` when it cannot be bounded.
fn index_range(e: &IndexExpr, ext: &[(&str, u64)], params: &[(String, i64)]) -> Option<(i128, i128)> {
    match e {
        IndexExpr::Affine { terms, offset } => terms.iter().try_fold((i128::from(*offset), i128::from(*offset)), |(lo, hi), t| {
            let p = match &t.param {
                Some(p) => i128::from(params.iter().find(|(n, _)| n == p)?.1),
                None => 1,
            };
            let k = i128::from(t.coeff) * p;
            let top = match &t.dim {
                Some(d) => k * (i128::from(ext.iter().find(|x| x.0 == d)?.1) - 1),
                None => k,
            };
            let base = if t.dim.is_some() { 0 } else { top };
            Some((lo + base.min(top), hi + base.max(top)))
        }),
        IndexExpr::FloorDiv { inner, by } if *by > 0 => {
            let (lo, hi) = index_range(inner, ext, params)?;
            let by = i128::from(*by);
            Some((lo.div_euclid(by), hi.div_euclid(by)))
        }
        _ => None,
    }
}

fn index_params(e: &IndexExpr) -> Vec<&str> {
    match e {
        IndexExpr::Affine { terms, .. } => terms.iter().filter_map(|t| t.param.as_deref()).collect(),
        IndexExpr::FloorDiv { inner, .. } => index_params(inner),
        IndexExpr::Indirect { index, .. } => index.iter().flat_map(index_params).collect(),
    }
}

/// Whether two segments' index ranges may share an element (every position's ranges intersect).
fn overlap(a: &Option<Vec<(i128, i128)>>, b: &Option<Vec<(i128, i128)>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.iter().zip(b).all(|(x, y)| x.0 <= y.1 && y.0 <= x.1),
        _ => true,
    }
}

fn eval_index(e: &IndexExpr, at: &[(&str, i128)], params: &[(String, i64)]) -> Option<i128> {
    match e {
        IndexExpr::Affine { terms, offset } => terms.iter().try_fold(i128::from(*offset), |acc, t| {
            let d = match &t.dim {
                Some(d) => at.iter().find(|x| x.0 == d)?.1,
                None => 1,
            };
            let p = match &t.param {
                Some(p) => i128::from(params.iter().find(|(n, _)| n == p)?.1),
                None => 1,
            };
            Some(acc + i128::from(t.coeff) * d * p)
        }),
        IndexExpr::FloorDiv { inner, by } if *by > 0 => Some(eval_index(inner, at, params)?.div_euclid(i128::from(*by))),
        _ => None,
    }
}

/// Distinct index tuples an operand touches over overlapping segments whose index params differ, by enumerating
/// each segment's points on the operand's dims.
fn index_union(sets: &[(&SegmentDomain, &[(&str, u64)])], index: &[IndexExpr], keep: &[&str]) -> Result<u128, Diagnostic> {
    const BUDGET: u128 = 1 << 22;
    let ext = |e: &[(&str, u64)], d: &str| e.iter().find(|x| x.0 == d).map_or(1, |x| x.1);
    let points: u128 = sets
        .iter()
        .map(|(_, e)| keep.iter().fold(1u128, |a, d| a.saturating_mul(u128::from(ext(e, d)))))
        .fold(0, u128::saturating_add);
    let too_big = || {
        Diagnostic::error(
            "E-WL-DOM-001",
            format!("operand footprint over {} overlapping segments with differing index params is too large to count exactly", sets.len()),
        )
        .hint("keep the segments' index ranges disjoint, or give them identical index params")
    };
    if points > BUDGET {
        return Err(too_big());
    }
    let mut seen = std::collections::BTreeSet::new();
    for (s, e) in sets {
        let span: Vec<u64> = keep.iter().map(|d| ext(e, d)).collect();
        let n: u64 = span.iter().product();
        let mut at = vec![0u64; keep.len()];
        for _ in 0..n {
            let fixed: Vec<(&str, i128)> = keep.iter().copied().zip(at.iter().map(|&x| i128::from(x))).collect();
            if admits(&s.constraints, e, &fixed) {
                let tuple: Option<Vec<i128>> = index.iter().map(|ix| eval_index(ix, &fixed, &s.params)).collect();
                seen.insert(tuple.ok_or_else(|| {
                    Diagnostic::error("E-WL-DOM-001", "indirect operand over overlapping segments with differing index params has no exact footprint")
                        .hint("keep the segments' index ranges disjoint, or give them identical index params")
                })?);
            }
            for (x, &m) in at.iter_mut().zip(&span).rev() {
                *x += 1;
                if *x < m {
                    break;
                }
                *x = 0;
            }
        }
    }
    Ok(seen.len() as u128)
}

/// One segment's constraints and extents.
type SegSet<'a> = (&'a [DiffConstraint], &'a [(&'a str, u64)]);

/// Distinct points of the union over segments of each segment's domain projected onto `keep`, by enumerating
/// the kept constrained dims and unioning the free dims' boxes at each point.
fn union_footprint(sets: &[SegSet], keep: &[&str]) -> Result<u128, Diagnostic> {
    const BUDGET: u128 = 1 << 24;
    let mut cdims: Vec<&str> = vec![];
    for (cs, _) in sets {
        for (_, d) in cs.iter().flat_map(|c| &c.terms) {
            if keep.contains(&d.as_str()) && !cdims.contains(&d.as_str()) {
                cdims.push(d);
            }
        }
    }
    let free: Vec<&str> = keep.iter().copied().filter(|d| !cdims.contains(d)).collect();
    let ext = |e: &[(&str, u64)], d: &str| e.iter().find(|x| x.0 == d).map_or(1, |x| x.1);
    let span: Vec<u64> = cdims.iter().map(|d| sets.iter().map(|(_, e)| ext(e, d)).max().unwrap_or(0)).collect();
    let points = span.iter().fold(1u128, |a, &x| a.saturating_mul(u128::from(x)));
    if points.saturating_mul(sets.len() as u128) > BUDGET {
        return Err(Diagnostic::error(
            "E-WL-DOM-001",
            format!("operand footprint over {} overlapping segments with differing constraints on {cdims:?} is too large to count exactly", sets.len()),
        )
        .hint("give the segments identical constraints on the operand's dims, or keep their index ranges disjoint"));
    }
    let boxes: Vec<Vec<u64>> = sets.iter().map(|(_, e)| free.iter().map(|d| ext(e, d)).collect()).collect();
    let mut total = 0u128;
    let mut at = vec![0u64; cdims.len()];
    for _ in 0..points {
        let fixed: Vec<(&str, i128)> = cdims.iter().copied().zip(at.iter().map(|&x| i128::from(x))).collect();
        let hit: Vec<&[u64]> = sets.iter().zip(&boxes).filter(|((cs, e), _)| admits(cs, e, &fixed)).map(|(_, b)| b.as_slice()).collect();
        total += box_union(&hit);
        for (x, &n) in at.iter_mut().zip(&span).rev() {
            *x += 1;
            if *x < n {
                break;
            }
            *x = 0;
        }
    }
    Ok(total)
}

/// Whether the segment's domain has a point with the `fixed` coordinates (dims it lacks have extent 1).
fn admits(cs: &[DiffConstraint], ext: &[(&str, u64)], fixed: &[(&str, i128)]) -> bool {
    let extent = |d: &str| i128::from(ext.iter().find(|x| x.0 == d).map_or(1, |x| x.1));
    if fixed.iter().any(|&(d, v)| v >= extent(d)) {
        return false;
    }
    let val = |d: &str| fixed.iter().find(|x| x.0 == d).map(|x| x.1);
    let open: Vec<&str> = cs.iter().flat_map(|c| &c.terms).map(|(_, d)| d.as_str()).filter(|d| val(d).is_none()).collect();
    let Some(&b) = open.first() else {
        return cs.iter().all(|c| c.terms.iter().map(|(k, d)| i128::from(*k) * val(d).unwrap_or(0)).sum::<i128>() <= i128::from(c.rhs));
    };
    if open.iter().any(|d| *d != b) {
        return project(ext, cs, &[]).is_ok_and(|n| n > 0);
    }
    let (mut lo, mut hi) = (0i128, extent(b) - 1);
    for c in cs {
        let rest: i128 = c.terms.iter().filter(|(_, d)| d != b).map(|(k, d)| i128::from(*k) * val(d).unwrap_or(0)).sum();
        let r = i128::from(c.rhs) - rest;
        match c.terms.iter().find(|(_, d)| d == b).map(|(k, _)| *k) {
            Some(k) if k > 0 => hi = hi.min(r),
            Some(_) => lo = lo.max(-r),
            None if r < 0 => return false,
            None => {}
        }
    }
    lo <= hi
}

/// Points in the union of origin-anchored boxes `[0, e_0) x [0, e_1) x ...`.
fn box_union(boxes: &[&[u64]]) -> u128 {
    match boxes.first() {
        None => 0,
        Some([]) => 1,
        Some(_) => {
            let mut cuts: Vec<u64> = boxes.iter().map(|b| b[0]).collect();
            cuts.sort_unstable();
            cuts.dedup();
            let mut prev = 0;
            let mut total = 0;
            for t in cuts {
                let rest: Vec<&[u64]> = boxes.iter().filter(|b| b[0] >= t).map(|b| &b[1..]).collect();
                total += u128::from(t - prev) * box_union(&rest);
                prev = t;
            }
            total
        }
    }
}

fn dom_err(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error("E-WL-DOM-001", msg)
        .hint("constraints may touch at most two dims, with coefficients ±1")
}

/// Bound on the inner dim as a function of the outer: `s·a + c`.
#[derive(Clone, Copy, Debug)]
struct Lin {
    s: i128,
    c: i128,
}

impl Lin {
    fn at(self, a: i128) -> i128 {
        self.s * a + self.c
    }
}

fn project(ext: &[(&str, u64)], cs: &[DiffConstraint], keep: &[&str]) -> Result<u128, Diagnostic> {
    if ext.iter().any(|&(_, e)| e == 0) {
        return Ok(0);
    }
    let mut cdims: Vec<&str> = Vec::new();
    for c in cs {
        if c.terms.is_empty() || c.terms.len() > 2 || c.terms.iter().any(|(k, _)| k.abs() != 1) {
            return Err(dom_err(format!("invalid constraint {:?}", c.terms)));
        }
        for (_, d) in &c.terms {
            if !ext.iter().any(|(n, _)| n == d) {
                return Err(dom_err(format!("constraint names unknown dim {d:?}")));
            }
            if !cdims.contains(&d.as_str()) {
                cdims.push(d);
            }
        }
    }
    if cdims.len() > 2 {
        return Err(dom_err(format!(
            "constraints span {} dims {cdims:?}",
            cdims.len()
        )));
    }
    let extent = |d: &str| i128::from(ext.iter().find(|(n, _)| *n == d).expect("dim").1);
    let free: u128 = ext
        .iter()
        .filter(|(n, _)| !cdims.contains(n) && keep.contains(n))
        .map(|&(_, e)| u128::from(e))
        .product();
    let kept = |d: &str| keep.contains(&d);
    let counted = match cdims.as_slice() {
        [] => 1,
        [a] => {
            let (lo, hi) = single_bounds(cs, a, extent(a));
            let n = (hi - lo + 1).max(0) as u128;
            if kept(a) { n } else { u128::from(n > 0) }
        }
        [a, b] => match (kept(a), kept(b)) {
            (true, true) => pair_count(cs, a, b, extent(a), extent(b), false),
            (true, false) => pair_count(cs, a, b, extent(a), extent(b), true),
            (false, true) => pair_count(cs, b, a, extent(b), extent(a), true),
            (false, false) => u128::from(pair_count(cs, a, b, extent(a), extent(b), true) > 0),
        },
        _ => unreachable!("checked above"),
    };
    Ok(free * counted)
}

fn single_bounds(cs: &[DiffConstraint], a: &str, ext: i128) -> (i128, i128) {
    let (mut lo, mut hi) = (0i128, ext - 1);
    for c in cs
        .iter()
        .filter(|c| c.terms.len() == 1 && c.terms[0].1 == a)
    {
        let r = i128::from(c.rhs);
        if c.terms[0].0 > 0 {
            hi = hi.min(r)
        } else {
            lo = lo.max(-r)
        }
    }
    (lo, hi)
}

/// Σ over outer `a` of the inner (`b`) range length, or with `nonempty` the number of `a` whose range is
/// non-empty. Closed form: the summand is piecewise linear with breakpoints where bound pieces cross, so the
/// sum is a handful of arithmetic series.
fn pair_count(
    cs: &[DiffConstraint],
    a: &str,
    b: &str,
    ext_a: i128,
    ext_b: i128,
    nonempty: bool,
) -> u128 {
    let (mut a_lo, mut a_hi) = (0i128, ext_a - 1);
    let mut upper = vec![Lin { s: 0, c: ext_b - 1 }];
    let mut lower = vec![Lin { s: 0, c: 0 }];
    for c in cs {
        let r = i128::from(c.rhs);
        let coef = |d: &str| {
            c.terms
                .iter()
                .find(|(_, n)| n == d)
                .map_or(0, |(k, _)| i128::from(*k))
        };
        let (ka, kb) = (coef(a), coef(b));
        match kb {
            1 => upper.push(Lin { s: -ka, c: r }),
            -1 => lower.push(Lin { s: ka, c: -r }),
            _ if ka > 0 => a_hi = a_hi.min(r),
            _ => a_lo = a_lo.max(-r),
        }
    }
    if a_lo > a_hi {
        return 0;
    }
    let h = |x: i128| {
        upper.iter().map(|u| u.at(x)).min().expect("nonempty")
            - lower.iter().map(|l| l.at(x)).max().expect("nonempty")
            + 1
    };
    let mut bps = vec![a_lo, a_hi + 1];
    let mut cross = |p: Lin, q: Lin| {
        if p.s != q.s {
            let (n, d) = (q.c - p.c, p.s - q.s);
            let x = if d > 0 {
                n.div_euclid(d)
            } else {
                (-n).div_euclid(-d)
            };
            bps.extend([x, x + 1]);
        }
    };
    for (i, &p) in upper.iter().enumerate() {
        upper[i + 1..].iter().for_each(|&q| cross(p, q));
        lower
            .iter()
            .for_each(|&q| cross(Lin { s: p.s, c: p.c + 1 }, q));
    }
    for (i, &p) in lower.iter().enumerate() {
        lower[i + 1..].iter().for_each(|&q| cross(p, q));
    }
    bps.retain(|&x| (a_lo..=a_hi + 1).contains(&x));
    bps.sort_unstable();
    bps.dedup();
    let mut total = 0i128;
    for w in bps.windows(2) {
        let (x0, x1) = (w[0], w[1] - 1);
        let (h0, h1) = (h(x0), h(x1));
        let n = x1 - x0 + 1;
        total += match (h0 > 0 && h1 > 0, h0 <= 0 && h1 <= 0) {
            (true, _) if nonempty => n,
            (true, _) => n * (h0 + h1) / 2,
            (_, true) => 0,
            _ => (x0..=x1)
                .map(|x| {
                    if nonempty {
                        i128::from(h(x) > 0)
                    } else {
                        h(x).max(0)
                    }
                })
                .sum(),
        };
    }
    total as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(dims: &[(&str, u64)], domain: Domain) -> Kernel {
        Kernel {
            id: "t.k0".into(),
            dims: dims
                .iter()
                .map(|&(n, e)| LoopDim {
                    name: n.into(),
                    extent: e,
                    kind: DimKind::Parallel,
                })
                .collect(),
            domain,
            operands: vec![],
            body: ScalarBody::default(),
            combine: None,
            accum: None,
            class: KernelClass::Map,
            opaque_cost: None,
        }
    }

    fn brute(dims: &[(&str, u64)], cs: &[DiffConstraint], keep: &[&str]) -> u128 {
        let mut seen = std::collections::BTreeSet::new();
        let total: u64 = dims.iter().map(|d| d.1).product();
        for mut idx in 0..total {
            let mut pt = Vec::new();
            for d in dims.iter().rev() {
                pt.push((d.0, (idx % d.1) as i64));
                idx /= d.1;
            }
            let val = |n: &str| pt.iter().find(|p| p.0 == n).unwrap().1;
            if cs.iter().all(|c| {
                c.terms
                    .iter()
                    .map(|(k, d)| i64::from(*k) * val(d))
                    .sum::<i64>()
                    <= c.rhs
            }) {
                seen.insert(keep.iter().map(|k| val(k)).collect::<Vec<_>>());
            }
        }
        seen.len() as u128
    }

    #[test]
    fn causal_and_window_closed_forms() {
        let (q, p) = (5u64, 7i64);
        let k = q as i64 + p;
        let causal = kernel(
            &[("q", q), ("j", k as u64)],
            Domain::Constrained(vec![DiffConstraint::diff("j", "q", p)]),
        );
        assert_eq!(
            causal.points().unwrap(),
            u128::from(q) * p as u128 + u128::from(q * (q + 1) / 2)
        );
        let w = 4;
        let window = kernel(
            &[("q", q), ("j", k as u64)],
            Domain::Constrained(vec![
                DiffConstraint::diff("j", "q", p),
                DiffConstraint::diff("q", "j", w - 1 - p),
            ]),
        );
        let expect: i64 = (0..q as i64).map(|qq| w.min(p + qq + 1)).sum();
        assert_eq!(window.points().unwrap(), expect as u128);
        assert_eq!(
            window.projection(&["j"]).unwrap(),
            (w + q as i64 - 1) as u128
        );
    }

    #[test]
    fn llama_prefill_causal_points() {
        let s = 2048u64;
        let k = kernel(
            &[("h", 32), ("q", s), ("j", s)],
            Domain::Constrained(vec![DiffConstraint::diff("j", "q", 0)]),
        );
        let p = k.points().unwrap();
        assert_eq!(p, 32 * u128::from(s * (s + 1) / 2));
        assert_eq!(2 * p * 256, 34_376_515_584);
    }

    #[test]
    fn rejects_three_dim_constraints() {
        let k = kernel(
            &[("a", 2), ("b", 2), ("c", 2)],
            Domain::Constrained(vec![
                DiffConstraint::diff("a", "b", 0),
                DiffConstraint::diff("b", "c", 0),
            ]),
        );
        assert_eq!(k.points().unwrap_err().code, "E-WL-DOM-001");
    }

    #[test]
    fn exhaustive_small_domains_match_brute_force() {
        let names = ["a", "b"];
        let mut cases = 0;
        for ea in 1..=5u64 {
            for eb in 1..=5u64 {
                for r1 in -3..=4i64 {
                    for r2 in -3..=4i64 {
                        for (s1, s2) in [
                            ((1, -1), (-1, 1)),
                            ((1, 1), (-1, -1)),
                            ((1, -1), (1, 1)),
                            ((-1, 1), (-1, -1)),
                        ] {
                            let cs = vec![
                                DiffConstraint {
                                    terms: vec![(s1.0, "a".into()), (s1.1, "b".into())],
                                    rhs: r1,
                                },
                                DiffConstraint {
                                    terms: vec![(s2.0, "a".into()), (s2.1, "b".into())],
                                    rhs: r2,
                                },
                                DiffConstraint {
                                    terms: vec![(1, "a".into())],
                                    rhs: r2 + 2,
                                },
                            ];
                            let dims = [("a", ea), ("b", eb), ("c", 2)];
                            let k = kernel(&dims, Domain::Constrained(cs.clone()));
                            for keep in [
                                &names[..],
                                &["a"][..],
                                &["b"][..],
                                &["c"][..],
                                &["a", "b", "c"][..],
                            ] {
                                assert_eq!(
                                    k.projection(keep).unwrap(),
                                    brute(&dims, &cs, keep),
                                    "{cs:?} {keep:?}"
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 1000);
    }
    #[test]
    fn segmented_footprints_union_shared_coordinates() {
        let seg = |k: u64, base: i64| SegmentDomain {
            extents: vec![("s".into(), 1), ("k".into(), k)],
            params: vec![("tok_base".into(), base)],
            constraints: vec![],
        };
        let k = |segments| kernel(&[("s", 1), ("k", 4)], Domain::Segmented { seg_dim: "s".into(), segments });
        let x = Id::new("x").unwrap();
        let shared = Operand::ident(&x, Access::Read, &["k"]);
        let based = Operand::new(
            &x,
            Access::Read,
            vec![
                IndexExpr::terms(vec![Term { coeff: 1, dim: None, param: Some("tok_base".into()) }, Term { coeff: 1, dim: Some("s".into()), param: None }], 0),
                IndexExpr::dim("k"),
            ],
        );
        let same = k(vec![seg(4, 0), seg(4, 1)]);
        assert_eq!(same.points().unwrap(), 8);
        assert_eq!(same.operand_footprint(&shared).unwrap(), 4);
        assert_eq!(same.operand_footprint(&based).unwrap(), 8);
        let ragged = k(vec![seg(4, 0), seg(6, 1), seg(0, 2)]);
        assert_eq!(ragged.operand_footprint(&shared).unwrap(), 6);
        assert_eq!(ragged.operand_footprint(&based).unwrap(), 10);
        let none = Operand::new(&x, Access::Read, vec![]);
        assert_eq!(same.operand_footprint(&none).unwrap(), 1);
    }

    #[test]
    fn overlapping_constrained_segments_union_their_footprints() {
        let seg = |k: u64, hi: i64| SegmentDomain {
            extents: vec![("s".into(), 1), ("k".into(), k)],
            params: vec![],
            constraints: vec![DiffConstraint { terms: vec![(1, "k".into())], rhs: hi }],
        };
        let k = |segments| kernel(&[("s", 1), ("k", 4)], Domain::Segmented { seg_dim: "s".into(), segments });
        let x = Id::new("x").unwrap();
        let op = Operand::ident(&x, Access::Read, &["k"]);
        assert_eq!(k(vec![seg(4, 1), seg(4, 2)]).operand_footprint(&op).unwrap(), 3);
        assert_eq!(k(vec![seg(4, 1), seg(2, 3)]).operand_footprint(&op).unwrap(), 2);

        let pair = |q: u64, kv: u64, p: i64| SegmentDomain {
            extents: vec![("s".into(), 1), ("q".into(), q), ("j".into(), kv), ("h".into(), q)],
            params: vec![],
            constraints: vec![DiffConstraint::diff("j", "q", p)],
        };
        let dims = [("s", 1), ("q", 8), ("j", 8), ("h", 8)];
        let segs = vec![pair(3, 5, 1), pair(5, 4, 0), pair(2, 8, 5)];
        let kern = kernel(&dims, Domain::Segmented { seg_dim: "s".into(), segments: segs.clone() });
        for keep in [&["j"][..], &["q"][..], &["q", "j"][..], &["j", "h"][..], &["q", "j", "h"][..]] {
            let mut seen = std::collections::BTreeSet::new();
            for s in &segs {
                let ext: Vec<(&str, u64)> = s.extents.iter().map(|(n, e)| (n.as_str(), *e)).collect();
                let e = |n: &str| ext.iter().find(|x| x.0 == n).unwrap().1 as i64;
                for q in 0..e("q") {
                    for j in 0..e("j") {
                        for h in 0..e("h") {
                            if j - q <= s.constraints[0].rhs {
                                let v = |n: &str| match n { "q" => q, "j" => j, _ => h };
                                seen.insert(keep.iter().map(|n| v(n)).collect::<Vec<_>>());
                            }
                        }
                    }
                }
            }
            assert_eq!(kern.operand_footprint(&Operand::ident(&x, Access::Read, keep)).unwrap(), seen.len() as u128, "{keep:?}");
        }
    }

    #[test]
    fn overlapping_segments_with_differing_params_union_their_index_sets() {
        let seg = |base: i64| SegmentDomain { extents: vec![("s".into(), 1), ("i".into(), 4)], params: vec![("base".into(), base)], constraints: vec![] };
        let k = |bases: &[i64]| kernel(&[("s", 1), ("i", 4)], Domain::Segmented { seg_dim: "s".into(), segments: bases.iter().map(|&b| seg(b)).collect() });
        let x = Id::new("x").unwrap();
        let shifted = Operand::new(
            &x,
            Access::Read,
            vec![IndexExpr::terms(vec![Term { coeff: 1, dim: Some("i".into()), param: None }, Term { coeff: 1, dim: None, param: Some("base".into()) }], 0)],
        );
        assert_eq!(k(&[0, 2]).operand_footprint(&shifted).unwrap(), 6);
        assert_eq!(k(&[0, 4]).operand_footprint(&shifted).unwrap(), 8);
        assert_eq!(k(&[0, 1, 2]).operand_footprint(&shifted).unwrap(), 6);
        let halved = Operand::new(&x, Access::Read, vec![IndexExpr::FloorDiv { inner: Box::new(shifted.index[0].clone()), by: 2 }]);
        assert_eq!(k(&[0, 2]).operand_footprint(&halved).unwrap(), 3);
        let gathered = Operand::new(&x, Access::Read, vec![IndexExpr::Indirect { via: Id::new("idx").unwrap(), index: vec![shifted.index[0].clone()] }]);
        assert_eq!(k(&[0, 2]).operand_footprint(&gathered).unwrap_err().code, "E-WL-DOM-001");
    }
}
