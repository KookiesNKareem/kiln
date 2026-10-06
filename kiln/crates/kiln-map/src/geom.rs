//! Slices of an iteration space and their affine footprints in tensor coordinates (03 §3.5).

use kiln_ir::wl::{DiffConstraint, Domain, IndexExpr, Kernel};
use serde::{Deserialize, Serialize};

use crate::program::{POp, Seg};

/// A box of one segment of an op's domain: `[lo[d], hi[d])` per kernel dim.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Slice {
    pub seg: u32,
    pub lo: Vec<u64>,
    pub hi: Vec<u64>,
}

impl Slice {
    pub fn whole(seg: u32, s: &Seg) -> Slice {
        Slice { seg, lo: vec![0; s.ext.len()], hi: s.ext.clone() }
    }

    pub fn extent(&self, d: usize) -> u64 {
        self.hi[d] - self.lo[d]
    }

    pub fn box_points(&self) -> u128 {
        (0..self.lo.len()).map(|d| u128::from(self.extent(d))).product()
    }

    pub fn is_empty(&self) -> bool {
        (0..self.lo.len()).any(|d| self.hi[d] <= self.lo[d])
    }
}

/// Exact number of domain points in a slice (difference constraints shifted to the slice origin).
pub fn slice_points(op: &POp, s: &Slice) -> u128 {
    let seg = &op.segs[s.seg as usize];
    if s.is_empty() {
        return 0;
    }
    if seg.cons.is_empty() {
        return s.box_points();
    }
    let k = &op.kernel;
    let shifted: Vec<DiffConstraint> = seg
        .cons
        .iter()
        .map(|c| {
            let off: i64 = c
                .terms
                .iter()
                .map(|(co, d)| i64::from(*co) * op.dim_ix(d).map_or(0, |i| s.lo[i] as i64))
                .sum();
            DiffConstraint { terms: c.terms.clone(), rhs: c.rhs - off }
        })
        .collect();
    // Only dims and domain enter the count; the rest of the kernel is not copied.
    let sub = Kernel {
        id: String::new(),
        dims: k
            .dims
            .iter()
            .enumerate()
            .map(|(i, d)| kiln_ir::wl::LoopDim { extent: s.extent(i), ..d.clone() })
            .collect(),
        domain: Domain::Constrained(shifted),
        operands: vec![],
        body: k.body,
        combine: None,
        accum: None,
        class: k.class,
        opaque_cost: None,
    };
    sub.points().unwrap_or_else(|_| s.box_points())
}

pub const MAX_RANK: usize = 6;

/// A footprint box in tensor coordinates: per tensor dim the covered range and the number of distinct
/// coordinates in it (strided or indirect accesses touch fewer than `hi - lo`). Fixed-size, `Copy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TBox {
    pub n: u8,
    pub lo: [i64; MAX_RANK],
    pub hi: [i64; MAX_RANK],
    pub count: [u64; MAX_RANK],
}

impl TBox {
    pub const EMPTY: TBox = TBox { n: 0, lo: [0; MAX_RANK], hi: [0; MAX_RANK], count: [0; MAX_RANK] };

    pub fn new(lo: &[i64], hi: &[i64], count: &[u64]) -> TBox {
        let mut b = TBox::EMPTY;
        for d in 0..lo.len().min(MAX_RANK) {
            b.push(lo[d], hi[d], count[d]);
        }
        b
    }

    pub fn push(&mut self, lo: i64, hi: i64, count: u64) {
        let d = self.n as usize;
        if d < MAX_RANK {
            self.lo[d] = lo;
            self.hi[d] = hi;
            self.count[d] = count;
            self.n += 1;
        }
    }

    pub fn rank(&self) -> usize {
        self.n as usize
    }

    pub fn elems(&self) -> u128 {
        self.count[..self.rank()].iter().map(|&c| u128::from(c)).product()
    }

    pub fn intersect(&self, o: &TBox) -> Option<TBox> {
        if self.n != o.n {
            return None;
        }
        let mut r = TBox::EMPTY;
        for d in 0..self.rank() {
            let (lo, hi) = (self.lo[d].max(o.lo[d]), self.hi[d].min(o.hi[d]));
            if hi <= lo {
                return None;
            }
            let frac = |b: &TBox| b.count[d] as f64 / (b.hi[d] - b.lo[d]).max(1) as f64;
            let c = ((hi - lo) as f64 * frac(self).min(frac(o))).round().max(1.0) as u64;
            r.push(lo, hi, c.min(self.count[d]).min(o.count[d]));
        }
        Some(r)
    }

    pub fn covers(&self, o: &TBox) -> bool {
        self.n == o.n && (0..self.rank()).all(|d| self.lo[d] <= o.lo[d] && o.hi[d] <= self.hi[d])
    }
}

fn dim_range(op: &POp, s: &Slice, d: &str) -> Option<(i64, i64)> {
    op.dim_ix(d).map(|i| (s.lo[i] as i64, s.hi[i] as i64))
}

/// Value range `[lo, hi)` and distinct-count of one index expression over a slice.
fn expr_range(op: &POp, s: &Slice, e: &IndexExpr) -> (i64, i64, u64) {
    let seg = &op.segs[s.seg as usize];
    let param = |p: &str| seg.params.iter().find(|(n, _)| n == p).map_or(0, |x| x.1);
    match e {
        IndexExpr::Affine { terms, offset } => {
            let (mut lo, mut hi, mut prod, mut dims) = (*offset, *offset, 1u64, 0usize);
            for t in terms {
                let k = t.coeff * t.param.as_deref().map_or(1, param);
                match t.dim.as_deref().and_then(|d| dim_range(op, s, d)) {
                    Some((a, b)) if b > a => {
                        let (x, y) = (k * a, k * (b - 1));
                        lo += x.min(y);
                        hi += x.max(y);
                        if k != 0 {
                            prod = prod.saturating_mul((b - a) as u64);
                            dims += 1;
                        }
                    }
                    Some(_) => return (0, 0, 0),
                    None => {
                        lo += k;
                        hi += k;
                    }
                }
            }
            let span = (hi - lo + 1) as u64;
            let count = if dims == 0 { 1 } else { prod.min(span) };
            (lo, hi + 1, count)
        }
        IndexExpr::FloorDiv { inner, by } => {
            let (lo, hi, c) = expr_range(op, s, inner);
            let by = (*by).max(1) as i64;
            let (a, b) = (lo.div_euclid(by), (hi - 1).div_euclid(by) + 1);
            (a, b, c.min((b - a) as u64))
        }
        IndexExpr::Indirect { index, .. } => {
            let mut lo = 0i64;
            let mut count = 1u64;
            for (j, ie) in index.iter().enumerate() {
                let (l, _, c) = expr_range(op, s, ie);
                if j == 0 {
                    lo = l;
                }
                count = count.saturating_mul(c);
            }
            (lo, lo + count as i64, count)
        }
    }
}

/// Footprint of operand `oi` of `op` over slice `s`, clipped to the tensor shape.
pub fn footprint(op: &POp, oi: usize, s: &Slice, shape: &[u64]) -> TBox {
    let o = &op.operands[oi];
    let mut b = TBox::EMPTY;
    for (d, e) in o.index.iter().enumerate() {
        let (mut lo, mut hi, mut c) = expr_range(op, s, e);
        if let Some(&ext) = shape.get(d) {
            let ext = ext as i64;
            lo = lo.clamp(0, ext);
            hi = hi.clamp(lo, ext);
            c = c.min((hi - lo) as u64);
        }
        b.push(lo, hi, c);
    }
    b
}

/// Total elements covered by a set of boxes: identical boxes count once, nested boxes are absorbed, and
/// remaining partial overlaps are counted by coordinate-compressed union (exact for boxes).
pub fn union_elems(boxes: &[TBox]) -> u128 {
    let mut v: Vec<&TBox> = boxes.iter().collect();
    v.sort();
    v.dedup();
    let keep: Vec<&TBox> =
        v.iter().enumerate().filter(|(i, b)| !v.iter().enumerate().any(|(j, o)| j != *i && o.covers(b) && *o != **b)).map(|(_, b)| *b).collect();
    let disjoint = keep.iter().enumerate().all(|(i, a)| keep[i + 1..].iter().all(|b| a.intersect(b).is_none()));
    if disjoint {
        return keep.iter().map(|b| b.elems()).sum();
    }
    let mut total = 0u128;
    for (i, b) in keep.iter().enumerate() {
        let mut parts = vec![**b];
        for o in &keep[..i] {
            parts = parts.into_iter().flat_map(|p| subtract(&p, o)).collect();
        }
        total += parts.iter().map(TBox::elems).sum::<u128>();
    }
    total
}

/// `a \ b` as disjoint boxes (density per dim preserved).
fn subtract(a: &TBox, b: &TBox) -> Vec<TBox> {
    if a.intersect(b).is_none() {
        return vec![*a];
    }
    let mut out = vec![];
    let mut rest = *a;
    for d in 0..a.rank() {
        let dens = rest.count[d] as f64 / (rest.hi[d] - rest.lo[d]).max(1) as f64;
        let piece = |lo: i64, hi: i64, r: &TBox| {
            let mut p = *r;
            p.lo[d] = lo;
            p.hi[d] = hi;
            p.count[d] = ((hi - lo) as f64 * dens).round().max(1.0) as u64;
            p
        };
        if rest.lo[d] < b.lo[d] {
            out.push(piece(rest.lo[d], b.lo[d], &rest));
        }
        if b.hi[d] < rest.hi[d] {
            out.push(piece(b.hi[d], rest.hi[d], &rest));
        }
        let (lo, hi) = (rest.lo[d].max(b.lo[d]), rest.hi[d].min(b.hi[d]));
        rest = piece(lo, hi, &rest);
    }
    out
}

/// Boxes with payloads, searchable by overlap. Sorted on the dim with the most distinct lower bounds, so a
/// grid of `n` boxes answers an overlap query in `O(log n + k)`.
#[derive(Clone, Debug, Default)]
pub struct BoxIndex<V> {
    dim: usize,
    wmax: i64,
    items: Vec<(TBox, V)>,
}

impl<V: Clone> BoxIndex<V> {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn extend(&mut self, new: impl IntoIterator<Item = (TBox, V)>) {
        self.items.extend(new);
        let nd = self.items.first().map_or(0, |b| b.0.rank());
        self.dim = (0..nd)
            .max_by_key(|&d| {
                let mut v: Vec<i64> = self.items.iter().map(|b| b.0.lo[d]).collect();
                v.sort_unstable();
                v.dedup();
                (v.len(), std::cmp::Reverse(d))
            })
            .unwrap_or(0);
        let d = self.dim;
        self.items.sort_by(|a, b| a.0.lo[d].cmp(&b.0.lo[d]).then_with(|| a.0.cmp(&b.0)));
        self.wmax = self.items.iter().map(|b| b.0.hi[d] - b.0.lo[d]).max().unwrap_or(0);
    }

    /// Entries overlapping `x` (all entries when dimensionality differs, e.g. through a layout view).
    pub fn overlapping<'s>(&'s self, x: &'s TBox) -> impl Iterator<Item = &'s (TBox, V)> + 's {
        let d = self.dim;
        let same = self.items.first().is_none_or(|b| b.0.n == x.n);
        let (a, b) = if same && x.n > 0 {
            let lo = x.lo[d] - self.wmax;
            (self.items.partition_point(|e| e.0.lo[d] <= lo), self.items.partition_point(|e| e.0.lo[d] < x.hi[d]))
        } else {
            (0, self.items.len())
        };
        self.items[a..b.max(a)].iter().filter(move |e| !same || e.0.intersect(x).is_some())
    }
}

/// Splits `n` into `p` parts, `floor`/`ceil` sizes, larger parts first (L3: no divisibility requirement).
pub fn even_parts(n: u64, p: u64) -> Vec<u64> {
    let p = p.clamp(1, n.max(1));
    let (q, r) = (n / p, n % p);
    (0..p).map(|i| q + u64::from(i < r)).collect()
}

/// Splits `n` into `p` parts that are multiples of `granule` where possible (the tail absorbs the rest).
pub fn aligned_parts(n: u64, p: u64, granule: u64) -> Vec<u64> {
    let g = granule.max(1);
    if g == 1 || n < g * p {
        return even_parts(n, p);
    }
    let units = n.div_ceil(g);
    let mut parts: Vec<u64> = even_parts(units, p).into_iter().map(|u| u * g).collect();
    let excess = parts.iter().sum::<u64>() - n;
    if let Some(last) = parts.iter_mut().rev().find(|x| **x > excess) {
        *last -= excess;
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn b(lo: &[i64], hi: &[i64]) -> TBox {
        TBox::new(lo, hi, &lo.iter().zip(hi).map(|(l, h)| (h - l) as u64).collect::<Vec<_>>())
    }

    #[test]
    fn union_dedups_and_overlaps() {
        assert_eq!(union_elems(&[b(&[0, 0], &[4, 4]), b(&[0, 0], &[4, 4])]), 16);
        assert_eq!(union_elems(&[b(&[0, 0], &[4, 4]), b(&[4, 0], &[8, 4])]), 32);
        assert_eq!(union_elems(&[b(&[0, 0], &[4, 4]), b(&[2, 2], &[6, 6])]), 28);
        assert_eq!(union_elems(&[b(&[0, 0], &[8, 8]), b(&[2, 2], &[6, 6])]), 64);
    }

    proptest! {
        #[test]
        fn box_index_matches_brute_force(boxes in prop::collection::vec((0i64..20, 1i64..6, 0i64..20, 1i64..6), 0..40), q in (0i64..24, 1i64..8, 0i64..24, 1i64..8)) {
            let mk = |(a, w, c, h): (i64, i64, i64, i64)| b(&[a, c], &[a + w, c + h]);
            let mut ix = BoxIndex::default();
            ix.extend(boxes.iter().enumerate().map(|(i, x)| (mk(*x), i)));
            let qb = mk(q);
            let mut got: Vec<usize> = ix.overlapping(&qb).map(|e| e.1).collect();
            got.sort_unstable();
            let want: Vec<usize> = boxes.iter().enumerate().filter(|(_, x)| mk(**x).intersect(&qb).is_some()).map(|(i, _)| i).collect();
            prop_assert_eq!(got, want);
        }

        #[test]
        fn parts_partition(n in 1u64..100_000, p in 1u64..600, g in 1u64..256) {
            for parts in [even_parts(n, p), aligned_parts(n, p, g)] {
                prop_assert_eq!(parts.iter().sum::<u64>(), n);
                prop_assert!(parts.iter().all(|&x| x > 0));
                prop_assert!(parts.len() as u64 <= p.min(n));
            }
            let e = even_parts(n, p);
            prop_assert!(e.iter().max().unwrap() - e.iter().min().unwrap() <= 1);
        }
    }
}
