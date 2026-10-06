//! Deterministic bounded least squares on log-time residuals (06 §3.3): Levenberg-Marquardt in bound-
//! normalized coordinates, Huber (delta 0.05) by iteratively reweighted least squares, a Gaussian prior per
//! parameter (sigma = 1/4 of its bound width), forward-difference Jacobian, fixed iteration order.

pub const HUBER_DELTA: f64 = 0.05;
/// A parameter within this normalized distance of a bound has converged onto it.
pub const BOUND_TOL: f64 = 1e-3;

#[derive(Clone, Debug, PartialEq)]
pub struct Bounded {
    pub lo: f64,
    pub hi: f64,
    pub prior: f64,
}

impl Bounded {
    pub fn to_u(&self, x: f64) -> f64 {
        (x - self.lo) / (self.hi - self.lo)
    }
    pub fn from_u(&self, u: f64) -> f64 {
        self.lo + u.clamp(0.0, 1.0) * (self.hi - self.lo)
    }
}

pub fn huber(r: f64) -> f64 {
    let a = r.abs();
    if a <= HUBER_DELTA { 0.5 * r * r } else { HUBER_DELTA * (a - 0.5 * HUBER_DELTA) }
}

fn weight(r: f64) -> f64 {
    let a = r.abs();
    if a <= HUBER_DELTA { 1.0 } else { HUBER_DELTA / a }
}

/// Prior precision in normalized coordinates: sigma = 1/4 of the bound width.
const PRIOR_PREC: f64 = 16.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    pub x: Vec<f64>,
    pub loss: f64,
    pub iterations: usize,
    pub at_bound: Vec<bool>,
}

/// Total objective at normalized point `u`: Huber over residuals in units of its scale `delta` (so the data
/// term is chi-square-like, `(r / delta)^2 / 2` near zero) plus the Gaussian prior term.
fn objective(r: &[f64], u: &[f64], up: &[f64]) -> f64 {
    r.iter().map(|&x| huber(x)).sum::<f64>() / (HUBER_DELTA * HUBER_DELTA)
        + u.iter().zip(up).map(|(a, b)| 0.5 * PRIOR_PREC * (a - b).powi(2)).sum::<f64>()
}

/// Solves `A x = b` for a small dense system (Gaussian elimination with partial pivoting).
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let (top, rest) = a.split_at_mut(c + 1);
        let pivot = &top[c];
        for (r, row) in rest.iter_mut().enumerate() {
            let f = row[c] / pivot[c];
            for (x, p) in row[c..].iter_mut().zip(&pivot[c..]) {
                *x -= f * p;
            }
            b[c + 1 + r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Minimizes `sum huber(res(x)_i) + prior` over the box. `res` maps parameter values (not normalized) to
/// log residuals; `start` (values) seeds the search (the prior when `None`).
pub fn minimize(params: &[Bounded], start: Option<&[f64]>, max_iter: usize, res: &dyn Fn(&[f64]) -> Vec<f64>) -> Solution {
    let n = params.len();
    let up: Vec<f64> = params.iter().map(|p| p.to_u(p.prior).clamp(0.0, 1.0)).collect();
    let mut u: Vec<f64> = match start {
        Some(s) => params.iter().zip(s).map(|(p, &x)| p.to_u(x).clamp(0.0, 1.0)).collect(),
        None => up.clone(),
    };
    let xs = |u: &[f64]| -> Vec<f64> { params.iter().zip(u).map(|(p, &v)| p.from_u(v)).collect() };
    let mut r = res(&xs(&u));
    let mut loss = objective(&r, &u, &up);
    let mut lambda = 1e-3;
    let mut it = 0;
    let mut converged = false;
    while it < max_iter && !converged {
        it += 1;
        let w: Vec<f64> = r.iter().map(|&x| weight(x) / (HUBER_DELTA * HUBER_DELTA)).collect();
        let h = 1e-5;
        let mut jac = vec![vec![0.0; n]; r.len()];
        for j in 0..n {
            let mut v = u.clone();
            let step = if v[j] + h <= 1.0 { h } else { -h };
            v[j] += step;
            let rj = res(&xs(&v));
            for (i, row) in jac.iter_mut().enumerate() {
                row[j] = (rj[i] - r[i]) / step;
            }
        }
        let mut jtj = vec![vec![0.0; n]; n];
        let mut g = vec![0.0; n];
        for (i, row) in jac.iter().enumerate() {
            for a in 0..n {
                g[a] += w[i] * row[a] * r[i];
                for b in 0..n {
                    jtj[a][b] += w[i] * row[a] * row[b];
                }
            }
        }
        for a in 0..n {
            jtj[a][a] += PRIOR_PREC;
            g[a] += PRIOR_PREC * (u[a] - up[a]);
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut m = jtj.clone();
            for (a, row) in m.iter_mut().enumerate() {
                row[a] += lambda * (1.0 + jtj[a][a]);
            }
            let Some(d) = solve_dense(m, g.iter().map(|x| -x).collect()) else { break };
            let un: Vec<f64> = u.iter().zip(&d).map(|(a, b)| (a + b).clamp(0.0, 1.0)).collect();
            let rn = res(&xs(&un));
            let ln = objective(&rn, &un, &up);
            if ln < loss {
                let step: f64 = un.iter().zip(&u).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let dl = loss - ln;
                (u, r, loss) = (un, rn, ln);
                lambda = (lambda / 10.0).max(1e-12);
                improved = true;
                converged = step < 1e-9 || dl < 1e-14 * (1.0 + loss);
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    Solution {
        x: xs(&u),
        loss,
        iterations: it,
        at_bound: u.iter().map(|&v| !(BOUND_TOL..=1.0 - BOUND_TOL).contains(&v)).collect(),
    }
}

/// splitmix64: seeded, platform-independent resampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Percentile (linear interpolation) of a sample.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q * (s.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    if i + 1 < s.len() { s[i] * (1.0 - f) + s[i + 1] * f } else { s[i] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_two_parameter_model_in_log_space() {
        // t = a * x + b on x in [1, 100], log residuals; a in [0.5, 2], b in [0, 5].
        let xs: Vec<f64> = (1..=60).map(|i| i as f64 * 1.7).collect();
        let truth = [1.3, 2.2];
        let meas: Vec<f64> = xs.iter().map(|x| truth[0] * x + truth[1]).collect();
        let p = [Bounded { lo: 0.5, hi: 2.0, prior: 1.0 }, Bounded { lo: 0.0, hi: 5.0, prior: 1.0 }];
        let res = |t: &[f64]| xs.iter().zip(&meas).map(|(x, m)| ((t[0] * x + t[1]) / m).ln()).collect();
        let s = minimize(&p, None, 200, &res);
        assert!((s.x[0] - truth[0]).abs() < 2e-3 && (s.x[1] - truth[1]).abs() < 5e-2, "{:?}", s.x);
        assert_eq!(s.at_bound, vec![false, false]);
    }

    #[test]
    fn flags_a_parameter_pushed_onto_its_bound() {
        let p = [Bounded { lo: 0.85, hi: 1.0, prior: 0.9 }];
        let res = |t: &[f64]| (0..40).map(|_| (t[0] / 0.6f64).ln()).collect();
        let s = minimize(&p, None, 200, &res);
        assert_eq!(s.at_bound, vec![true]);
    }
}
