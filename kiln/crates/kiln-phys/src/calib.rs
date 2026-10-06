//! Calibration (04 §12): MAP fit in log-space with log-normal priors, Levenberg-Marquardt with a fixed iteration
//! count and projected bounds, Laplace posterior sigmas, bound and residual reporting. The area group runs here
//! (designs -> roll-up); the power group's residuals come from the engine (kiln-sim drives it with telemetry).

use std::collections::BTreeMap;

use kiln_ir::hw::HwModel;
use serde::Serialize;

use crate::characterize;
use crate::floorplan::{self, FloorplanInput, PlaceTier};
use crate::params::Params;
use crate::report::die_parts;
use crate::tables::{CalibEntry, CalibSet, Tables};

/// A free parameter: registry name, scope key, prior, log-sigma and bounds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Free {
    pub name: String,
    pub key: Option<String>,
    pub prior: f64,
    pub sigma: f64,
    pub bounds: [f64; 2],
}

impl Free {
    /// From the registry / node table / DRAM table prior with its confidence sigma and bounds.
    pub fn registered(name: &str, key: Option<&str>) -> Free {
        let t = Tables::get();
        let p = Params::priors();
        let (c, _, _) = p.band(name, key);
        let (sigma, bounds) = match name {
            "util_std_cell" | "p_ll" | "p_ls" | "v_th" | "alpha" => {
                let path = match name {
                    "util_std_cell" => "logic.util_std_cell",
                    "p_ll" => "leakage.p_ll_w_mm2",
                    "p_ls" => "sram.p_ls_mw_per_mib",
                    "v_th" => "dvfs.v_th",
                    _ => "dvfs.alpha",
                };
                let s = key.and_then(|k| t.node(k)).and_then(|n| n.sourced.get(path).cloned()).expect("node prior");
                (s.c.sigma(), s.bounds.unwrap_or([c * 0.1, c * 10.0]))
            }
            "e_dram_core" => (0.3, [2.0, 6.0]),
            _ => {
                let r = t.params.get(name).unwrap_or_else(|| panic!("registered {name}"));
                (r.prior.c.sigma(), r.prior.bounds.unwrap_or([c * 0.1, c * 10.0]))
            }
        };
        Free { name: name.to_owned(), key: key.map(String::from), prior: c, sigma, bounds }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Residual {
    pub name: String,
    pub predicted: f64,
    pub target: f64,
    /// Normalized residual (in sigmas).
    pub z: f64,
    pub held_out: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FitResult {
    pub values: Vec<f64>,
    pub sigma_post: Vec<f64>,
    pub at_bound: Vec<bool>,
    pub cost: f64,
    pub iterations: usize,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let p = (i..n).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs())).unwrap_or(i);
        a.swap(i, p);
        b.swap(i, p);
        let d = if a[i][i].abs() < 1e-300 { 1e-300 } else { a[i][i] };
        let pivot = a[i].clone();
        for r in i + 1..n {
            let f = a[r][i] / d;
            for (x, p) in a[r][i..].iter_mut().zip(&pivot[i..]) {
                *x -= f * p;
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|c| a[i][c] * x[c]).sum();
        x[i] = (b[i] - s) / if a[i][i].abs() < 1e-300 { 1e-300 } else { a[i][i] };
    }
    x
}

fn invert(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| solve(a.to_vec(), (0..n).map(|i| f64::from(u8::from(i == j))).collect())).collect();
    (0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect()
}

/// MAP fit: minimizes `sum r_i(theta)^2 + sum ((theta_k - log prior_k) / sigma_k)^2` over `theta = log p` within
/// the log bounds. `resid(values)` returns data residuals already divided by their sigmas.
pub fn fit(free: &[Free], resid: &dyn Fn(&[f64]) -> Vec<f64>, iterations: usize) -> FitResult {
    let n = free.len();
    let lo: Vec<f64> = free.iter().map(|f| f.bounds[0].ln()).collect();
    let hi: Vec<f64> = free.iter().map(|f| f.bounds[1].ln()).collect();
    let mut th: Vec<f64> = free.iter().enumerate().map(|(k, f)| f.prior.ln().clamp(lo[k], hi[k])).collect();
    let all = |th: &[f64]| -> Vec<f64> {
        let v: Vec<f64> = th.iter().map(|x| x.exp()).collect();
        let mut r = resid(&v);
        for (k, f) in free.iter().enumerate() {
            r.push((th[k] - f.prior.ln()) / f.sigma);
        }
        r
    };
    let cost = |r: &[f64]| r.iter().map(|x| x * x).sum::<f64>();
    let mut r = all(&th);
    let mut c = cost(&r);
    let mut lambda = 1e-2;
    let mut jac = vec![vec![0.0; n]; r.len()];
    let mut it = 0;
    for _ in 0..iterations {
        it += 1;
        for k in 0..n {
            let h = 1e-4;
            let mut t2 = th.clone();
            t2[k] += h;
            let r2 = all(&t2);
            for i in 0..r.len() {
                jac[i][k] = (r2[i] - r[i]) / h;
            }
        }
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for i in 0..r.len() {
            for a in 0..n {
                jtr[a] += jac[i][a] * r[i];
                for b in 0..n {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut m = jtj.clone();
            for (a, row) in m.iter_mut().enumerate() {
                row[a] *= 1.0 + lambda;
                row[a] += 1e-12;
            }
            let step = solve(m, jtr.iter().map(|x| -x).collect());
            let t2: Vec<f64> = (0..n).map(|k| (th[k] + step[k]).clamp(lo[k], hi[k])).collect();
            let r2 = all(&t2);
            let c2 = cost(&r2);
            if c2 < c {
                th = t2;
                r = r2;
                c = c2;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved && lambda > 1e8 {
            break;
        }
    }
    // Laplace: posterior covariance (J^T J)^-1 at the optimum.
    for k in 0..n {
        let h = 1e-4;
        let mut t2 = th.clone();
        t2[k] += h;
        let r2 = all(&t2);
        for i in 0..r.len() {
            jac[i][k] = (r2[i] - r[i]) / h;
        }
    }
    let mut jtj = vec![vec![0.0; n]; n];
    for row in &jac {
        for a in 0..n {
            for b in 0..n {
                jtj[a][b] += row[a] * row[b];
            }
        }
    }
    let cov = invert(&jtj);
    FitResult {
        values: th.iter().map(|x| x.exp()).collect(),
        sigma_post: (0..n).map(|k| cov[k][k].max(0.0).sqrt()).collect(),
        at_bound: (0..n).map(|k| (th[k] - lo[k]).abs() < 1e-3 || (th[k] - hi[k]).abs() < 1e-3).collect(),
        cost: c,
        iterations: it,
    }
}

/// Area model output of one design at `params`: envelope mm^2 of its first compute die, part fractions,
/// transistor estimate (billions).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AreaPoint {
    pub die_mm2: f64,
    pub transistors_b: f64,
    pub fractions: BTreeMap<String, f64>,
    /// Die-area fraction of every instance subtree by entity id (`cmem`, `mxu`, ...), summed over instances: block area
    /// over die area, whitespace excluded (published floorplan fractions are of drawn blocks; TPUv4i's fill is absorbed
    /// by its interconnect blocks, Jouppi et al. 2021 Fig. 6).
    pub entities: BTreeMap<String, f64>,
}

pub fn area_of(hw: &HwModel, params: &Params) -> AreaPoint {
    let ch = characterize::characterize(hw, params);
    let fp = floorplan::build(&FloorplanInput { hw, ch: &ch, params, tier: PlaceTier::A, seed: 0 });
    let d = &fp.dies[0];
    let parts = die_parts(hw, &ch, d.node);
    let mut ge = 0.0;
    let mut bits = 0.0;
    let mut st = vec![d.node];
    while let Some(x) = st.pop() {
        ge += ch.nodes[x].ge;
        bits += ch.nodes[x].sram_bits;
        st.extend(ch.kids[x].iter().copied());
    }
    let env = d.area_env_um2;
    let fractions = crate::characterize::PARTS.iter().map(|p| (format!("{p:?}").to_lowercase(), parts[*p as usize] / env)).collect();
    let sub = floorplan::subtree_areas(hw, &ch);
    let mut entities: BTreeMap<String, f64> = BTreeMap::new();
    let mut st = vec![d.node];
    while let Some(x) = st.pop() {
        if x != d.node {
            *entities.entry(hw.nodes[x].entity_id.clone()).or_default() += sub[x] / env;
        }
        st.extend(ch.kids[x].iter().copied());
    }
    AreaPoint { die_mm2: env / 1e6, transistors_b: (4.0 * ge + 6.0 * bits) / 1e9, fractions, entities }
}


/// Parameters with `values` substituted for `free`.
pub fn with_values(base: &Params, free: &[Free], values: &[f64]) -> Params {
    let mut p = base.clone();
    for (f, v) in free.iter().zip(values) {
        p = p.with(&f.name, f.key.as_deref(), *v);
    }
    p
}

/// Writes fitted values as calibration-set entries.
pub fn entries(free: &[Free], r: &FitResult) -> Vec<CalibEntry> {
    free.iter()
        .enumerate()
        .map(|(k, f)| CalibEntry {
            name: f.name.clone(),
            key: f.key.clone(),
            value: r.values[k],
            sigma: r.sigma_post[k].min(f.sigma),
            prior: f.prior,
            bounds: f.bounds,
            at_bound: r.at_bound[k],
        })
        .collect()
}

pub fn calib_set(id: &str, scope: &str, platform: Option<&str>, params: Vec<CalibEntry>, report: serde_json::Value) -> CalibSet {
    CalibSet {
        schema: "kiln.phys.calib/1".into(),
        id: id.into(),
        scope: scope.into(),
        platform: platform.map(String::from),
        params,
        note: None,
        targets_hash: Some(crate::data::inputs_hash().into()),
        report: Some(report),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_scale_from_data() {
        let free = vec![Free { name: "a".into(), key: None, prior: 1.0, sigma: 0.7, bounds: [0.1, 10.0] }];
        let r = fit(&free, &|v| vec![(v[0] * 2.0 / 6.0).ln() / 0.01], 50);
        assert!((r.values[0] - 3.0).abs() < 0.01, "{:?}", r);
        assert!(r.sigma_post[0] < 0.02 && !r.at_bound[0]);
        let b = fit(&free, &|v| vec![(v[0] / 50.0).ln() / 0.01], 50);
        assert!(b.at_bound[0] && (b.values[0] - 10.0).abs() < 1e-6);
    }
}
