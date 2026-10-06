//! Macro placers over an abstract problem (04 §6.4-§6.5). Tier A: recursive bisection into a slicing tree, each
//! cut ordered by a 1-D quadratic embedding with terminal propagation (Fiedler vector when no terminal pulls),
//! refined by Fiduccia-Mattheyses passes; region sizes are area-proportional. Tier B: simulated annealing over
//! the normalized Polish expression of that tree (Wong-Liu moves), seeded from the design hash, fixed schedule.
//! Both are deterministic: index-ordered loops, fixed iteration counts, ties by index.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Rect {
        Rect { x0, y0, x1, y1 }
    }
    pub fn w(&self) -> f64 {
        self.x1 - self.x0
    }
    pub fn h(&self) -> f64 {
        self.y1 - self.y0
    }
    pub fn area(&self) -> f64 {
        self.w().max(0.0) * self.h().max(0.0)
    }
    pub fn cx(&self) -> f64 {
        0.5 * (self.x0 + self.x1)
    }
    pub fn cy(&self) -> f64 {
        0.5 * (self.y0 + self.y1)
    }
    pub fn shift(&self, dx: f64, dy: f64) -> Rect {
        Rect::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }
    /// Splits along x (vertical cut) at fraction `f` of the width.
    pub fn split_v(&self, f: f64) -> (Rect, Rect) {
        let x = self.x0 + f * self.w();
        (Rect::new(self.x0, self.y0, x, self.y1), Rect::new(x, self.y0, self.x1, self.y1))
    }
    pub fn split_h(&self, f: f64) -> (Rect, Rect) {
        let y = self.y0 + f * self.h();
        (Rect::new(self.x0, self.y0, self.x1, y), Rect::new(self.x0, y, self.x1, self.y1))
    }
}

/// A placement problem: macros with areas, power hints, weighted nets (pairs) and fixed terminal pulls.
#[derive(Clone, Debug, Default)]
pub struct Problem {
    pub area: Vec<f64>,
    pub power: Vec<f64>,
    /// `(i, j, weight)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// `(i, x, y, weight)`: pull of macro `i` toward a fixed point (pinned PHYs).
    pub terms: Vec<(usize, f64, f64, f64)>,
    pub region: Rect,
    /// Soft-block aspect range (h / w).
    pub aspect: (f64, f64),
    /// Weight of the power-density spreading term (04 §6.4 lambda_th), relative to wirelength.
    pub lambda_th: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Tok {
    Leaf(usize),
    /// Vertical cut: left | right.
    V,
    /// Horizontal cut: bottom / top.
    H,
}

/// Rectangles of every leaf of a Polish expression in `region`, areas proportional to leaf areas.
pub fn realize(expr: &[Tok], area: &[f64], region: Rect) -> Vec<Rect> {
    // Post-order sizes, then top-down assignment from the root (last token).
    let mut stack: Vec<(usize, f64)> = vec![];
    let mut kids: Vec<Option<(usize, usize)>> = vec![None; expr.len()];
    let mut sz = vec![0.0; expr.len()];
    for (i, t) in expr.iter().enumerate() {
        match t {
            Tok::Leaf(m) => {
                sz[i] = area[*m].max(1e-9);
                stack.push((i, sz[i]));
            }
            Tok::V | Tok::H => {
                let (r, ra) = stack.pop().expect("valid expression");
                let (l, la) = stack.pop().expect("valid expression");
                kids[i] = Some((l, r));
                sz[i] = la + ra;
                stack.push((i, sz[i]));
            }
        }
    }
    let mut out = vec![Rect::default(); area.len()];
    let mut todo = vec![(expr.len() - 1, region)];
    while let Some((i, r)) = todo.pop() {
        match expr[i] {
            Tok::Leaf(m) => out[m] = r,
            op => {
                let (l, rr) = kids[i].expect("operator has children");
                let f = sz[l] / (sz[l] + sz[rr]);
                let (a, b) = if op == Tok::V { r.split_v(f) } else { r.split_h(f) };
                todo.push((l, a));
                todo.push((rr, b));
            }
        }
    }
    out
}

/// Placement objective (04 §6.4): weighted HPWL of nets (center to center) and terminal pulls, plus soft-aspect
/// violation and power-density spreading over an 8 x 8 bin grid.
pub fn phi(p: &Problem, rects: &[Rect]) -> f64 {
    let mut wl = 0.0;
    for &(i, j, w) in &p.edges {
        wl += w * ((rects[i].cx() - rects[j].cx()).abs() + (rects[i].cy() - rects[j].cy()).abs());
    }
    for &(i, x, y, w) in &p.terms {
        wl += w * ((rects[i].cx() - x).abs() + (rects[i].cy() - y).abs());
    }
    let span = p.region.w() + p.region.h();
    let wsum: f64 = p.edges.iter().map(|e| e.2).sum::<f64>() + p.terms.iter().map(|t| t.3).sum::<f64>();
    let mut asp = 0.0;
    for (i, r) in rects.iter().enumerate() {
        if r.w() <= 0.0 || r.h() <= 0.0 {
            continue;
        }
        let a = r.h() / r.w();
        let v = if a < p.aspect.0 { (p.aspect.0 / a).ln() } else if a > p.aspect.1 { (a / p.aspect.1).ln() } else { 0.0 };
        asp += v * p.area[i];
    }
    let total_area: f64 = p.area.iter().sum::<f64>().max(1e-9);
    let mut th = 0.0;
    let ptot: f64 = p.power.iter().sum();
    if p.lambda_th > 0.0 && ptot > 0.0 {
        let mut bins = [[0.0f64; 8]; 8];
        let (bw, bh) = (p.region.w() / 8.0, p.region.h() / 8.0);
        for (i, r) in rects.iter().enumerate() {
            let q = p.power[i] / r.area().max(1e-9);
            for (bi, row) in bins.iter_mut().enumerate() {
                for (bj, cell) in row.iter_mut().enumerate() {
                    let b = Rect::new(p.region.x0 + bj as f64 * bw, p.region.y0 + bi as f64 * bh, p.region.x0 + (bj + 1) as f64 * bw, p.region.y0 + (bi + 1) as f64 * bh);
                    let ov = (r.x1.min(b.x1) - r.x0.max(b.x0)).max(0.0) * (r.y1.min(b.y1) - r.y0.max(b.y0)).max(0.0);
                    *cell += q * ov;
                }
            }
        }
        let mean = ptot / 64.0;
        for row in &bins {
            for &c in row {
                th += ((c - mean) / mean).powi(2);
            }
        }
        th /= 64.0;
    }
    wl / wsum.max(1e-30) / span.max(1e-9) + asp / total_area + p.lambda_th * th
}

/// Tier A: recursive bisection. Returns the Polish expression and the rectangles.
pub fn tier_a(p: &Problem) -> (Vec<Tok>, Vec<Rect>) {
    let n = p.area.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for &(i, j, w) in &p.edges {
        if i != j {
            adj[i].push((j, w));
            adj[j].push((i, w));
        }
    }
    let mut center = vec![(p.region.cx(), p.region.cy()); n];
    let mut expr = vec![];
    let all: Vec<usize> = (0..n).collect();
    bisect(p, &adj, &all, p.region, &mut center, &mut expr);
    let rects = realize(&expr, &p.area, p.region);
    (expr, rects)
}

fn bisect(p: &Problem, adj: &[Vec<(usize, f64)>], set: &[usize], r: Rect, center: &mut [(f64, f64)], expr: &mut Vec<Tok>) {
    if set.len() == 1 {
        expr.push(Tok::Leaf(set[0]));
        center[set[0]] = (r.cx(), r.cy());
        return;
    }
    let vertical = r.w() >= r.h();
    let axis = |c: (f64, f64)| if vertical { c.0 } else { c.1 };
    let (lo, hi) = if vertical { (r.x0, r.x1) } else { (r.y0, r.y1) };
    let mid = 0.5 * (lo + hi);
    let in_set: Vec<bool> = {
        let mut v = vec![false; p.area.len()];
        set.iter().for_each(|&i| v[i] = true);
        v
    };
    // External pull per macro: terminals and macros outside the set at their current region centers.
    let mut ext_w = vec![0.0; p.area.len()];
    let mut ext_x = vec![0.0; p.area.len()];
    for &(i, x, y, w) in &p.terms {
        if in_set[i] {
            ext_w[i] += w;
            ext_x[i] += w * axis((x, y));
        }
    }
    for &i in set {
        for &(j, w) in &adj[i] {
            if !in_set[j] {
                ext_w[i] += w;
                ext_x[i] += w * axis(center[j]);
            }
        }
    }
    let has_pull = set.iter().any(|&i| ext_w[i] > 0.0);
    // 1-D embedding: quadratic placement with external pulls (Gauss-Seidel, 40 sweeps) or, without pulls, the
    // Fiedler vector by power iteration on (c I - L), deflated against the constant vector.
    let k = set.len();
    let pos_in: Vec<usize> = {
        let mut v = vec![usize::MAX; p.area.len()];
        set.iter().enumerate().for_each(|(a, &i)| v[i] = a);
        v
    };
    let mut x: Vec<f64> = (0..k).map(|a| a as f64 / (k - 1).max(1) as f64 - 0.5).collect();
    if has_pull {
        let mut xs: Vec<f64> = set.iter().map(|_| mid).collect();
        for _ in 0..40 {
            for (a, &i) in set.iter().enumerate() {
                let mut num = ext_x[i];
                let mut den = ext_w[i];
                for &(j, w) in &adj[i] {
                    if in_set[j] {
                        num += w * xs[pos_in[j]];
                        den += w;
                    }
                }
                xs[a] = if den > 0.0 { num / den } else { mid };
            }
        }
        x = xs;
    } else {
        let deg: Vec<f64> = set.iter().map(|&i| adj[i].iter().filter(|e| in_set[e.0]).map(|e| e.1).sum()).collect();
        let c = 2.0 * deg.iter().copied().fold(0.0, f64::max) + 1e-12;
        for _ in 0..40 {
            let mut y = vec![0.0; k];
            for (a, &i) in set.iter().enumerate() {
                let mut lx = deg[a] * x[a];
                for &(j, w) in &adj[i] {
                    if in_set[j] {
                        lx -= w * x[pos_in[j]];
                    }
                }
                y[a] = c * x[a] - lx;
            }
            let m = y.iter().sum::<f64>() / k as f64;
            y.iter_mut().for_each(|v| *v -= m);
            let nrm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            if nrm <= 1e-300 {
                break;
            }
            x = y.into_iter().map(|v| v / nrm).collect();
        }
        if x[0] > 0.0 {
            x.iter_mut().for_each(|v| *v = -*v);
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(set[a].cmp(&set[b])));
    let total: f64 = set.iter().map(|&i| p.area[i]).sum();
    // Split index: least cut weight among splits within 10% of the half-area balance (else the most balanced);
    // the cut is updated incrementally as macros move across in embedding order.
    let mut best: Option<(bool, f64, f64, usize)> = None;
    let mut acc = 0.0;
    let mut cut = 0.0;
    let mut on_left = vec![false; k];
    for s in 1..k {
        let a = order[s - 1];
        acc += p.area[set[a]];
        for &(j, w) in &adj[set[a]] {
            if in_set[j] && pos_in[j] != a {
                if on_left[pos_in[j]] { cut -= w } else { cut += w }
            }
        }
        on_left[a] = true;
        let imb = (acc / total - 0.5).abs();
        let cand = (imb <= 0.1, cut, imb, s);
        best = Some(match best {
            None => cand,
            Some(b) => {
                let better = match (cand.0, b.0) {
                    (true, false) => true,
                    (false, true) => false,
                    (true, true) => cand.1 < b.1 - 1e-12 * b.1.abs() || ((cand.1 - b.1).abs() <= 1e-12 * b.1.abs() && cand.2 < b.2),
                    (false, false) => cand.2 < b.2,
                };
                if better { cand } else { b }
            }
        });
    }
    let split = best.map_or(k / 2, |b| b.3);
    let mut left = vec![false; k];
    order[..split].iter().for_each(|&a| left[a] = true);
    // Fiduccia-Mattheyses: 4 passes of single-macro moves with positive gain that keep the balance.
    if k > 2 {
        let mut la: f64 = (0..k).filter(|&b| left[b]).map(|b| p.area[set[b]]).sum();
        let mut cnt = (0..k).filter(|&b| left[b]).count();
        for _ in 0..4 {
            let mut moved = false;
            for a in 0..k {
                let i = set[a];
                let (mut same, mut other) = (0.0, 0.0);
                for &(j, w) in &adj[i] {
                    if in_set[j] {
                        if left[pos_in[j]] == left[a] { same += w } else { other += w }
                    }
                }
                let term_gain = match ext_pull_side(ext_w[i], ext_x[i], mid) {
                    Some(l) if l != left[a] => 0.5 * ext_w[i],
                    Some(_) => -0.5 * ext_w[i],
                    None => 0.0,
                };
                if other - same + term_gain <= 1e-12 {
                    continue;
                }
                let (new_la, new_cnt) = if left[a] { (la - p.area[i], cnt - 1) } else { (la + p.area[i], cnt + 1) };
                if (new_la / total - 0.5).abs() <= 0.1 && new_cnt >= 1 && new_cnt < k {
                    left[a] = !left[a];
                    la = new_la;
                    cnt = new_cnt;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }
    let a_set: Vec<usize> = (0..k).filter(|&a| left[a]).map(|a| set[a]).collect();
    let b_set: Vec<usize> = (0..k).filter(|&a| !left[a]).map(|a| set[a]).collect();
    let fa = a_set.iter().map(|&i| p.area[i]).sum::<f64>() / total;
    let (ra, rb) = if vertical { r.split_v(fa) } else { r.split_h(fa) };
    for &i in &a_set {
        center[i] = (ra.cx(), ra.cy());
    }
    for &i in &b_set {
        center[i] = (rb.cx(), rb.cy());
    }
    bisect(p, adj, &a_set, ra, center, expr);
    bisect(p, adj, &b_set, rb, center, expr);
    expr.push(if vertical { Tok::V } else { Tok::H });
}

fn ext_pull_side(w: f64, wx: f64, mid: f64) -> Option<bool> {
    if w <= 0.0 { None } else { Some(wx / w < mid) }
}

/// SplitMix64: the seeded RNG of Tier B (seed from the design hash).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn valid(expr: &[Tok]) -> bool {
    // Balloting property (every prefix has more operands than operators); equal adjacent operators are allowed
    // (the Tier A tree is not normalized, and both readings realize the same rectangles).
    let mut operands = 0i64;
    for t in expr {
        match t {
            Tok::Leaf(_) => operands += 1,
            _ => {
                operands -= 1;
                if operands < 1 {
                    return false;
                }
            }
        }
    }
    operands == 1
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AnnealStats {
    pub moves: usize,
    pub accepted: usize,
    pub phi_start: f64,
    pub phi_end: f64,
}

/// Tier B: Wong-Liu annealing from `init` with `moves` proposals (geometric cooling from the mean uphill step
/// to 1e-3 of it). Returns the best expression seen.
pub fn tier_b(p: &Problem, init: &[Tok], seed: u64, moves: usize) -> (Vec<Tok>, Vec<Rect>, AnnealStats) {
    let mut cur = init.to_vec();
    let mut cur_rects = realize(&cur, &p.area, p.region);
    let mut cur_phi = phi(p, &cur_rects);
    let mut best = (cur.clone(), cur_rects.clone(), cur_phi);
    let mut st = AnnealStats { moves, phi_start: cur_phi, ..Default::default() };
    if p.area.len() < 3 {
        st.phi_end = cur_phi;
        return (best.0, best.1, st);
    }
    let mut rng = Rng::new(seed);
    let propose = |e: &[Tok], rng: &mut Rng| -> Option<Vec<Tok>> {
        let mut v = e.to_vec();
        match rng.below(3) {
            0 => {
                // M1: swap two adjacent operands (in operand order).
                let leaves: Vec<usize> = v.iter().enumerate().filter(|(_, t)| matches!(t, Tok::Leaf(_))).map(|(i, _)| i).collect();
                let a = rng.below(leaves.len() - 1);
                v.swap(leaves[a], leaves[a + 1]);
            }
            1 => {
                // M2: complement a chain of operators.
                let ops: Vec<usize> = v.iter().enumerate().filter(|(_, t)| !matches!(t, Tok::Leaf(_))).map(|(i, _)| i).collect();
                let mut i = ops[rng.below(ops.len())];
                while i < v.len() && !matches!(v[i], Tok::Leaf(_)) {
                    v[i] = if v[i] == Tok::V { Tok::H } else { Tok::V };
                    i += 1;
                }
            }
            _ => {
                // M3: swap an adjacent operand and operator.
                let i = rng.below(v.len() - 1);
                if matches!(v[i], Tok::Leaf(_)) == matches!(v[i + 1], Tok::Leaf(_)) {
                    return None;
                }
                v.swap(i, i + 1);
            }
        }
        valid(&v).then_some(v)
    };
    let mut ups = 0.0;
    let mut nup = 0;
    for _ in 0..50 {
        if let Some(v) = propose(&cur, &mut rng) {
            let d = phi(p, &realize(&v, &p.area, p.region)) - cur_phi;
            if d > 0.0 {
                ups += d;
                nup += 1;
            }
        }
    }
    let t0 = if nup > 0 { ups / f64::from(nup) } else { 1e-3 * cur_phi.max(1e-9) };
    let decay = (1e-3f64).powf(1.0 / moves.max(1) as f64);
    let mut temp = t0;
    for _ in 0..moves {
        temp *= decay;
        let Some(v) = propose(&cur, &mut rng) else { continue };
        let rects = realize(&v, &p.area, p.region);
        let f = phi(p, &rects);
        let d = f - cur_phi;
        if d <= 0.0 || rng.unit() < (-d / temp.max(1e-300)).exp() {
            cur = v;
            cur_rects = rects;
            cur_phi = f;
            st.accepted += 1;
            if cur_phi < best.2 {
                best = (cur.clone(), cur_rects.clone(), cur_phi);
            }
        }
    }
    st.phi_end = best.2;
    (best.0, best.1, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(n: usize) -> Problem {
        Problem {
            area: vec![1.0; n],
            power: vec![1.0; n],
            edges: (1..n).map(|i| (i - 1, i, 1.0)).collect(),
            terms: vec![],
            region: Rect::new(0.0, 0.0, n as f64, 1.0),
            aspect: (0.5, 2.0),
            lambda_th: 0.0,
        }
    }

    #[test]
    fn tier_a_covers_region_without_overlap() {
        let p = chain(9);
        let (e, r) = tier_a(&p);
        assert!(valid(&e));
        let total: f64 = r.iter().map(Rect::area).sum();
        assert!((total - p.region.area()).abs() < 1e-9);
        for i in 0..r.len() {
            for j in i + 1..r.len() {
                let ov = (r[i].x1.min(r[j].x1) - r[i].x0.max(r[j].x0)).max(0.0) * (r[i].y1.min(r[j].y1) - r[i].y0.max(r[j].y0)).max(0.0);
                assert!(ov < 1e-9, "{i} {j}");
            }
        }
        // A chain on a 9 x 1 strip is placed in order (either direction): neighbours are adjacent.
        let mut xs: Vec<(f64, usize)> = r.iter().enumerate().map(|(i, r)| (r.cx(), i)).collect();
        xs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let seq: Vec<usize> = xs.iter().map(|x| x.1).collect();
        let fwd: Vec<usize> = (0..9).collect();
        let rev: Vec<usize> = (0..9).rev().collect();
        assert!(seq == fwd || seq == rev, "{seq:?}");
    }

    #[test]
    fn terminals_pull_macros() {
        let mut p = chain(4);
        p.edges.clear();
        p.region = Rect::new(0.0, 0.0, 4.0, 4.0);
        p.terms = vec![(3, 0.0, 2.0, 10.0), (0, 4.0, 2.0, 10.0)];
        let (_, r) = tier_a(&p);
        assert!(r[3].cx() < r[0].cx(), "{r:?}");
    }

    #[test]
    fn tier_b_never_worse_and_deterministic() {
        let mut p = chain(12);
        p.region = Rect::new(0.0, 0.0, 4.0, 3.0);
        p.edges.push((0, 11, 5.0));
        let (e, r) = tier_a(&p);
        let a = phi(&p, &r);
        let (e1, r1, s1) = tier_b(&p, &e, 7, 3000);
        let (e2, _, _) = tier_b(&p, &e, 7, 3000);
        assert_eq!(e1, e2);
        assert!(phi(&p, &r1) <= a + 1e-12 && s1.phi_end <= s1.phi_start);
        assert!(valid(&e1));
    }
}
