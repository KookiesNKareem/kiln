//! Thermal model (04 §9). Tier A: average and block power density, junction estimate (in `power`). Tier B:
//! steady-state grid (HotSpot-style RC network): one active layer per die with lateral silicon conduction and a
//! vertical path to the sink, solved by Jacobi-preconditioned CG (fixed tolerance, iteration cap, fixed order).

use serde::Serialize;

use crate::place::Rect;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ThermalMap {
    pub g: usize,
    pub outline: Rect,
    /// Temperature per bin, row-major from the lower-left, degC.
    pub t_c: Vec<f64>,
    pub t_max_c: f64,
    pub iterations: usize,
}

/// Steady state of `power` (W per rectangle) on a `g x g` grid over `outline`: lateral conductance
/// `k_si * t_si` between bins, vertical `A_bin / r_ja` to the inlet temperature.
pub fn solve(outline: Rect, power: &[(Rect, f64)], g: usize, r_ja_k_mm2_w: f64, t_inlet_c: f64) -> ThermalMap {
    let n = g * g;
    let (bw, bh) = (outline.w() / g as f64, outline.h() / g as f64);
    let mut q = vec![0.0; n];
    for (r, p) in power {
        let a = r.area().max(1e-9);
        for i in 0..g {
            for j in 0..g {
                let b = Rect::new(outline.x0 + j as f64 * bw, outline.y0 + i as f64 * bh, outline.x0 + (j + 1) as f64 * bw, outline.y0 + (i + 1) as f64 * bh);
                let ov = (r.x1.min(b.x1) - r.x0.max(b.x0)).max(0.0) * (r.y1.min(b.y1) - r.y0.max(b.y0)).max(0.0);
                q[i * g + j] += p * ov / a;
            }
        }
    }
    // Silicon k = 120 W/mK at 85 C, 0.75 mm thick bulk; lateral conductance between square-ish bins ~ k t.
    let k_t = 120.0 * 0.75e-3;
    let gx = k_t * bh / bw.max(1e-9);
    let gy = k_t * bw / bh.max(1e-9);
    let gv = (bw * bh * 1e-6) / r_ja_k_mm2_w;
    let apply = |x: &[f64], y: &mut [f64]| {
        for i in 0..g {
            for j in 0..g {
                let k = i * g + j;
                let mut v = gv * x[k];
                if j > 0 {
                    v += gx * (x[k] - x[k - 1]);
                }
                if j + 1 < g {
                    v += gx * (x[k] - x[k + 1]);
                }
                if i > 0 {
                    v += gy * (x[k] - x[k - g]);
                }
                if i + 1 < g {
                    v += gy * (x[k] - x[k + g]);
                }
                y[k] = v;
            }
        }
    };
    let diag: Vec<f64> = (0..n)
        .map(|k| {
            let (i, j) = (k / g, k % g);
            gv + gx * (f64::from(u8::from(j > 0)) + f64::from(u8::from(j + 1 < g))) + gy * (f64::from(u8::from(i > 0)) + f64::from(u8::from(i + 1 < g)))
        })
        .collect();
    let mut x = vec![0.0; n];
    let mut r = q.clone();
    let mut z: Vec<f64> = r.iter().zip(&diag).map(|(a, d)| a / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let norm0 = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let mut ap = vec![0.0; n];
    let mut it = 0;
    while it < 500 {
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        it += 1;
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() / norm0 < 1e-8 {
            break;
        }
        for k in 0..n {
            z[k] = r[k] / diag[k];
        }
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
    }
    let t: Vec<f64> = x.iter().map(|v| v + t_inlet_c).collect();
    let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ThermalMap { g, outline, t_c: t, t_max_c: t_max, iterations: it }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_power_matches_the_lumped_rise() {
        let o = Rect::new(0.0, 0.0, 20_000.0, 20_000.0);
        let m = solve(o, &[(o, 400.0)], 32, 90.0, 30.0);
        let lumped = 30.0 + 400.0 * 90.0 / 400.0;
        assert!((m.t_max_c - lumped).abs() < 0.5, "{} vs {lumped}", m.t_max_c);
    }

    #[test]
    fn hotspot_is_hotter_than_the_average() {
        let o = Rect::new(0.0, 0.0, 20_000.0, 20_000.0);
        let hot = Rect::new(0.0, 0.0, 4_000.0, 4_000.0);
        let m = solve(o, &[(o, 200.0), (hot, 200.0)], 32, 90.0, 30.0);
        let avg = 30.0 + 400.0 * 90.0 / 400.0;
        assert!(m.t_max_c > avg + 5.0 && m.iterations < 500);
    }
}
