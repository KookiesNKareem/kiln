//! Calibration (05 §6.8): predicted vs measured (log-log, y = x with +-10% / +-25% bands, fit filled and test
//! hollow, noise floor as error bars), error histogram of log2(pred/meas) with median |err|, p90 and MAPE, and
//! the worst outliers.

use std::collections::BTreeMap;

use kiln_trace::calib_report::CalibRow;

use crate::chart::{self, Axis, fmt_time};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::cat;
use crate::views::{Selection, ViewSpec};

#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub median_abs: f64,
    pub p90_abs: f64,
    pub mape: f64,
    pub geomean_ratio: f64,
}

fn pctl(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    v[k]
}

/// Error statistics of `pred/meas - 1`.
pub fn stats(rows: &[&CalibRow]) -> Stats {
    let mut abs: Vec<f64> = rows
        .iter()
        .map(|r| (r.predicted_s_tier_a / r.measured_s - 1.0).abs())
        .collect();
    let mape = abs.iter().sum::<f64>() / abs.len().max(1) as f64;
    let gm = (rows
        .iter()
        .map(|r| (r.predicted_s_tier_a / r.measured_s).ln())
        .sum::<f64>()
        / rows.len().max(1) as f64)
        .exp();
    Stats {
        n: rows.len(),
        median_abs: pctl(&mut abs.clone(), 0.5),
        p90_abs: pctl(&mut abs, 0.9),
        mape,
        geomean_ratio: gm,
    }
}

pub fn scene(rows: &[CalibRow], spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let shown: Vec<(usize, &CalibRow)> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            spec.device.as_deref().is_none_or(|d| d == r.device)
                && r.measured_s > 0.0
                && r.predicted_s_tier_a > 0.0
        })
        .collect();
    let devices: Vec<String> = {
        let mut d: Vec<String> = rows.iter().map(|r| r.device.clone()).collect();
        d.sort();
        d.dedup();
        d
    };
    let set = rows
        .first()
        .map_or(String::new(), |r| r.calibration_set_hash.clone());
    chart::title(
        &mut s,
        &th,
        "Calibration  predicted vs measured (Tier A)",
        &format!(
            "device: {} | set {} | n = {}",
            spec.device.clone().unwrap_or_else(|| devices.join(", ")),
            set,
            shown.len()
        ),
    );
    let plot = Rect::new(
        90.0,
        70.0,
        (spec.width * 0.5).min(spec.height - 150.0),
        (spec.width * 0.5).min(spec.height - 150.0),
    );
    if shown.is_empty() {
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "no calibration rows",
            "kiln viz --calibration <calibration_report.arrow | kiln calibrate report --format json output>",
        );
        return s;
    }
    let all: Vec<f64> = shown
        .iter()
        .flat_map(|(_, r)| [r.measured_s, r.predicted_s_tier_a])
        .collect();
    let (lo, hi) = (
        all.iter().copied().fold(f64::INFINITY, f64::min),
        all.iter().copied().fold(0.0, f64::max),
    );
    let x = Axis::log_fit(lo, hi, plot.x, plot.right());
    let y = Axis::log_fit(lo, hi, plot.bottom(), plot.y);
    let (x, y) = (
        Axis {
            min: x.min.min(y.min),
            max: x.max.max(y.max),
            ..x
        },
        Axis {
            min: x.min.min(y.min),
            max: x.max.max(y.max),
            ..y
        },
    );
    chart::axes(
        &mut s,
        &th,
        plot,
        &x,
        &y,
        "measured time",
        "predicted time",
        &fmt_time,
        &fmt_time,
    );
    s.clip(plot);
    for (k, c) in [(1.25, th.grid), (1.1, th.grid)] {
        let pts = vec![
            [x.map(x.min), y.map(x.min * k)],
            [x.map(x.max), y.map(x.max * k)],
            [x.map(x.max), y.map(x.max / k)],
            [x.map(x.min), y.map(x.min / k)],
        ];
        s.polygon(pts, Some(c.with_alpha(110)), None, Hit::None);
    }
    s.seg(
        [x.map(x.min), y.map(x.min)],
        [x.map(x.max), y.map(x.max)],
        Stroke::solid(1.25, th.fg),
    );
    let mut kinds: Vec<&str> = shown.iter().map(|(_, r)| r.op_kind.as_str()).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let color = |k: &str| cat(kinds.iter().position(|x| *x == k).unwrap_or(0));
    for (i, r) in &shown {
        let c = [x.map(r.measured_s), y.map(r.predicted_s_tier_a)];
        if let Some(cv) = r.measured_cv {
            s.seg(
                [x.map(r.measured_s * (1.0 - cv)), c[1]],
                [x.map(r.measured_s * (1.0 + cv)), c[1]],
                Stroke::solid(1.0, th.muted),
            );
        }
        let col = color(&r.op_kind);
        let hit = Hit::Calib(*i as u32);
        let st = Stroke::solid(
            if sel.has(hit) { 2.5 } else { 1.0 },
            if sel.has(hit) { th.accent } else { col },
        );
        if r.split == "fit" {
            s.circle(c, 4.0, Some(col.with_alpha(200)), Some(st), hit);
        } else {
            s.polygon(
                vec![
                    [c[0], c[1] - 5.0],
                    [c[0] + 4.5, c[1] + 3.5],
                    [c[0] - 4.5, c[1] + 3.5],
                ],
                None,
                Some(Stroke { width: 1.5, ..st }),
                hit,
            );
        }
    }
    s.unclip();
    s.text(
        [plot.x + 6.0, plot.y + 6.0],
        "bands: +-10%, +-25%",
        10.0,
        th.muted,
        HAlign::Left,
        VAlign::Top,
    );
    // Legend.
    let lx = plot.right() + 20.0;
    let items: Vec<(Color, String)> = kinds.iter().map(|k| (color(k), k.to_string())).collect();
    let mut ly = chart::swatches(&mut s, &th, lx, plot.y, &items);
    s.circle([lx + 6.0, ly + 6.0], 4.0, Some(th.muted), None, Hit::None);
    s.text(
        [lx + 18.0, ly + 6.0],
        "fit split",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Middle,
    );
    ly += 18.0;
    s.polygon(
        vec![
            [lx + 6.0, ly + 1.0],
            [lx + 10.5, ly + 9.5],
            [lx + 1.5, ly + 9.5],
        ],
        None,
        Some(Stroke::solid(1.5, th.muted)),
        Hit::None,
    );
    s.text(
        [lx + 18.0, ly + 6.0],
        "test split / held out",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Middle,
    );

    // Histogram of log2(pred/meas).
    let hx = lx + 190.0;
    let hw = spec.width - hx - 24.0;
    let hp = Rect::new(hx + 40.0, plot.y, hw - 40.0, plot.h * 0.45);
    let errs: Vec<f64> = shown.iter().map(|(_, r)| r.log2_err()).collect();
    let lim = errs
        .iter()
        .map(|e| e.abs())
        .fold(0.25f64, f64::max)
        .min(4.0);
    let bins = 24usize;
    let mut h = vec![0usize; bins];
    for e in &errs {
        let k = (((e + lim) / (2.0 * lim)) * bins as f64)
            .floor()
            .clamp(0.0, bins as f64 - 1.0) as usize;
        h[k] += 1;
    }
    let hmax = h.iter().copied().max().unwrap_or(1).max(1);
    let xa = Axis::new(-lim, lim, false, hp.x, hp.right());
    let ya = Axis::new(0.0, hmax as f64 * 1.1, false, hp.bottom(), hp.y);
    chart::axes(
        &mut s,
        &th,
        hp,
        &xa,
        &ya,
        "log2(predicted / measured)",
        "ops",
        &|v| format!("{v:.2}"),
        &|v| format!("{v:.0}"),
    );
    for (k, n) in h.iter().enumerate() {
        let x0 = xa.map(-lim + 2.0 * lim * k as f64 / bins as f64);
        let x1 = xa.map(-lim + 2.0 * lim * (k + 1) as f64 / bins as f64);
        s.fill(
            Rect::new(
                x0,
                ya.map(*n as f64),
                (x1 - x0 - 1.0).max(1.0),
                hp.bottom() - ya.map(*n as f64),
            ),
            th.accent.with_alpha(200),
        );
    }
    s.seg(
        [xa.map(0.0), hp.y],
        [xa.map(0.0), hp.bottom()],
        Stroke::solid(1.0, th.fg),
    );
    // Stats per device and split.
    let mut groups: BTreeMap<(String, String), Vec<&CalibRow>> = BTreeMap::new();
    for (_, r) in &shown {
        groups
            .entry((r.device.clone(), r.split.clone()))
            .or_default()
            .push(r);
        groups
            .entry((r.device.clone(), "all".into()))
            .or_default()
            .push(r);
    }
    let mut sy = hp.bottom() + 50.0;
    s.text(
        [hx, sy],
        "device / split      n   median|err|   p90|err|   MAPE   geomean pred/meas",
        11.0,
        th.muted,
        HAlign::Left,
        VAlign::Top,
    );
    sy += 18.0;
    for ((d, sp), rs) in &groups {
        let st = stats(rs);
        s.text_full(
            [hx, sy],
            format!(
                "{:<19} {:>4}   {:>9.1}%   {:>7.1}%   {:>4.1}%   {:.3}",
                format!("{d}/{sp}"),
                st.n,
                100.0 * st.median_abs,
                100.0 * st.p90_abs,
                100.0 * st.mape,
                st.geomean_ratio
            ),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
            true,
            false,
        );
        sy += 16.0;
    }
    // Outliers.
    let mut worst: Vec<&(usize, &CalibRow)> = shown.iter().collect();
    worst.sort_by(|a, b| {
        b.1.log2_err()
            .abs()
            .total_cmp(&a.1.log2_err().abs())
            .then(a.0.cmp(&b.0))
    });
    let oy = (plot.bottom() + 50.0).max(sy + 20.0);
    s.text(
        [16.0, oy],
        "Worst outliers",
        13.0,
        th.fg,
        HAlign::Left,
        VAlign::Top,
    );
    let mut yy = oy + 20.0;
    for (i, r) in worst.into_iter().take(spec.top) {
        if yy > spec.height - 14.0 {
            break;
        }
        let line = format!(
            "{:<10} {:<46} pred {:>9}  meas {:>9}  {:+.0}%  [{}]",
            r.device,
            text::fit(&r.name, 11.0, true, 46.0 * 6.7).unwrap_or_default(),
            fmt_time(r.predicted_s_tier_a),
            fmt_time(r.measured_s),
            100.0 * (r.predicted_s_tier_a / r.measured_s - 1.0),
            r.split
        );
        s.text_full(
            [16.0, yy],
            line,
            11.0,
            if sel.has(Hit::Calib(*i as u32)) {
                th.accent
            } else {
                th.fg
            },
            HAlign::Left,
            VAlign::Top,
            true,
            false,
        );
        yy += 16.0;
    }
    s
}
