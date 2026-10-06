//! Roofline (05 §6.4): per-op (or per-family) points against the chip's compute and memory-level ceilings,
//! colored by binding class, sized by time, with corner error bars; per-phase aggregate diamonds; optional
//! measured calibration points drawn hollow with a line to their prediction.

use std::collections::BTreeMap;

use kiln_trace::analysis::{RoofPoint, phase_code, reference_level, roofline_points};
use kiln_trace::calib_report::CalibRow;
use kiln_trace::trace::Trace;

use crate::chart::{self, Axis, fmt_bw, fmt_flops, fmt_num};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{binding_color, binding_label, cat};
use crate::views::{Selection, ViewSpec};

/// Compute ceilings to draw: dense peaks per first operand type, bf16 first.
fn compute_ceilings(t: &Trace, chip: u32) -> Vec<(String, f64)> {
    let mut best: BTreeMap<String, f64> = BTreeMap::new();
    for c in t.ceilings.iter().filter(|c| c.kind == 0 && c.chip == chip) {
        let dtype = c.name.split('*').next().unwrap_or(&c.name).to_string();
        let e = best.entry(dtype).or_insert(0.0);
        *e = e.max(c.value);
    }
    // bf16 first (the reference precision of 05 §6.4), then the fastest other modes.
    let mut v: Vec<(String, f64)> = best.into_iter().collect();
    v.sort_by(|a, b| {
        (b.0 == "bf16")
            .cmp(&(a.0 == "bf16"))
            .then(b.1.total_cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });
    v.truncate(3);
    v
}

fn marker(
    s: &mut Scene,
    shape: u8,
    c: [f32; 2],
    r: f32,
    fill: Option<Color>,
    stroke: Stroke,
    hit: Hit,
) {
    match shape % 4 {
        0 => s.circle(c, r, fill, Some(stroke), hit),
        1 => s.rect(
            Rect::new(c[0] - r, c[1] - r, 2.0 * r, 2.0 * r),
            fill,
            Some(stroke),
            hit,
        ),
        2 => s.polygon(
            vec![
                [c[0], c[1] - r * 1.2],
                [c[0] + r * 1.1, c[1] + r * 0.8],
                [c[0] - r * 1.1, c[1] + r * 0.8],
            ],
            fill,
            Some(stroke),
            hit,
        ),
        _ => s.polygon(
            vec![
                [c[0], c[1] - r * 1.3],
                [c[0] + r * 1.3, c[1]],
                [c[0], c[1] + r * 1.3],
                [c[0] - r * 1.3, c[1]],
            ],
            fill,
            Some(stroke),
            hit,
        ),
    }
}

pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection, calib: Option<&[CalibRow]>) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let phase = phase_code(t, spec.phase.as_deref());
    let level = reference_level(t);
    let level_name = t
        .ceilings
        .iter()
        .find(|c| c.kind == 1 && c.level == level)
        .map_or("all levels".to_string(), |c| c.name.clone());
    let pts = roofline_points(t, phase, level, spec.aggregate);
    let design = t.manifest.design_name.clone().unwrap_or_default();
    chart::title(
        &mut s,
        &th,
        &format!("Roofline  {design}"),
        &format!(
            "reference level: {level_name} | {} | {} | color = binding term, size = time, bars = slow/fast corners",
            spec.phase.as_deref().unwrap_or("all phases"),
            if spec.aggregate {
                "per op family across layers"
            } else {
                "per execution group (labelled by its largest op)"
            }
        ),
    );
    let plot = Rect::new(
        90.0,
        64.0,
        spec.width - 90.0 - 250.0,
        spec.height - 64.0 - 60.0,
    );
    if pts.is_empty() {
        chart::banner(
            &mut s,
            &th,
            plot,
            "no ops with FLOPs and bytes at the reference level",
            "re-run with kiln eval --trace ops and a design with memory levels",
        );
        return s;
    }
    let chip = t
        .resources
        .get(
            t.ops[pts[0].op as usize]
                .chips
                .first()
                .copied()
                .unwrap_or(0) as usize,
        )
        .map_or(0, |r| r.chip);
    let comp = compute_ceilings(t, chip);
    let mems: Vec<(String, f64)> = t
        .ceilings
        .iter()
        .filter(|c| c.kind == 1 && c.chip == chip)
        .map(|c| (c.name.clone(), c.value))
        .collect();
    let peak = comp.first().map_or(0.0, |c| c.1);
    let mut xs: Vec<f64> = pts.iter().map(RoofPoint::intensity).collect();
    let mut ys: Vec<f64> = pts.iter().map(RoofPoint::attained).collect();
    ys.extend(pts.iter().filter_map(|p| p.attained_low));
    let meas: Vec<(usize, &CalibRow)> = calib
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .filter(|(_, r)| r.flops.is_some_and(|f| f > 0.0) && r.bytes.is_some_and(|b| b > 0.0))
        .collect();
    for (_, r) in &meas {
        let (f, b) = (r.flops.unwrap_or(0.0), r.bytes.unwrap_or(1.0));
        xs.push(f / b);
        ys.push(f / r.measured_s);
        ys.push(f / r.predicted_s_tier_a);
    }
    if peak > 0.0 {
        ys.push(peak);
    }
    let fmin = |v: &[f64]| {
        v.iter()
            .copied()
            .filter(|x| *x > 0.0 && x.is_finite())
            .fold(f64::INFINITY, f64::min)
    };
    let fmax = |v: &[f64]| {
        v.iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(0.0, f64::max)
    };
    let x = Axis::log_fit(fmin(&xs) / 2.0, fmax(&xs) * 2.0, plot.x, plot.right());
    let y = Axis::log_fit(fmin(&ys) / 2.0, fmax(&ys) * 1.5, plot.bottom(), plot.y);
    chart::axes(
        &mut s,
        &th,
        plot,
        &x,
        &y,
        &format!("arithmetic intensity (FLOP/B w.r.t. {level_name})"),
        "attained FLOP/s",
        &fmt_num,
        &fmt_flops,
    );
    s.clip(plot);
    // Memory diagonals (dashed except the reference level) and compute ceilings.
    for (k, (name, bw)) in mems.iter().enumerate() {
        let reference = t
            .ceilings
            .iter()
            .any(|c| c.kind == 1 && c.level == level && c.name == *name);
        let ridge_top = if peak > 0.0 { peak } else { y.max };
        let (xa, xb) = (x.min, (ridge_top / bw).min(x.max));
        let st = if reference {
            Stroke::solid(2.0, th.fg)
        } else {
            Stroke::dashed(1.0, th.muted, 4.0)
        };
        s.seg([x.map(xa), y.map(bw * xa)], [x.map(xb), y.map(bw * xb)], st);
        let lx = xa * 10f64.powf(0.3 + 0.25 * k as f64);
        if lx < xb {
            s.text(
                [x.map(lx) + 4.0, y.map(bw * lx) - 4.0],
                format!("{name} {}", fmt_bw(*bw)),
                10.0,
                if reference { th.fg } else { th.muted },
                HAlign::Left,
                VAlign::Bottom,
            );
        }
    }
    let ref_bw = t
        .ceilings
        .iter()
        .find(|c| c.kind == 1 && c.level == level)
        .map(|c| c.value);
    for (k, (name, f)) in comp.iter().enumerate() {
        let x0 = ref_bw.map_or(x.min, |b| f / b);
        let st = if k == 0 {
            Stroke::solid(2.0, th.fg)
        } else {
            Stroke::dashed(1.0, th.muted, 4.0)
        };
        s.seg(
            [x.map(x0.max(x.min)), y.map(*f)],
            [plot.right(), y.map(*f)],
            st,
        );
        s.text(
            [plot.right() - 4.0, y.map(*f) - 3.0],
            format!("peak {name} {}", fmt_flops(*f)),
            10.0,
            if k == 0 { th.fg } else { th.muted },
            HAlign::Right,
            VAlign::Bottom,
        );
    }
    // Points.
    let total: f64 = pts.iter().map(|p| p.time_s).sum();
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|&a, &b| pts[b].time_s.total_cmp(&pts[a].time_s).then(a.cmp(&b)));
    for &i in &order {
        let p = &pts[i];
        let c = [x.map(p.intensity()), y.map(p.attained())];
        if let (Some(lo), Some(hi)) = (p.attained_low, p.attained_high)
            && hi > lo
        {
            s.seg(
                [c[0], y.map(lo)],
                [c[0], y.map(hi)],
                Stroke::solid(1.0, th.muted),
            );
            s.seg(
                [c[0] - 3.0, y.map(lo)],
                [c[0] + 3.0, y.map(lo)],
                Stroke::solid(1.0, th.muted),
            );
            s.seg(
                [c[0] - 3.0, y.map(hi)],
                [c[0] + 3.0, y.map(hi)],
                Stroke::solid(1.0, th.muted),
            );
        }
        let r = 3.0 + 9.0 * ((p.time_s / total.max(1e-300)).sqrt() as f32);
        let col = p
            .binding
            .map_or(th.muted, |b| binding_color(t.binding_name(b)))
            .with_alpha(210);
        let hit = Hit::Op(p.op);
        marker(
            &mut s,
            p.phase,
            c,
            r,
            Some(col),
            Stroke::solid(
                if sel.has(hit) { 2.5 } else { 0.75 },
                if sel.has(hit) { th.accent } else { th.bg },
            ),
            hit,
        );
    }
    // Per-phase aggregates.
    let mut per_phase: BTreeMap<u8, (f64, f64, f64)> = BTreeMap::new();
    for p in &pts {
        let e = per_phase.entry(p.phase).or_default();
        e.0 += p.flops;
        e.1 += p.bytes;
        e.2 += p.time_s;
    }
    for (ph, (f, b, tt)) in &per_phase {
        let c = [x.map(f / b), y.map(f / tt)];
        marker(
            &mut s,
            3,
            c,
            9.0,
            None,
            Stroke::solid(2.0, th.fg),
            Hit::Phase(*ph),
        );
        s.text(
            [c[0] + 12.0, c[1]],
            t.phase_name(*ph).to_string(),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
    }
    // Measured calibration points.
    for (k, r) in &meas {
        let (f, b) = (r.flops.unwrap_or(0.0), r.bytes.unwrap_or(1.0));
        let (cm, cp) = (
            [x.map(f / b), y.map(f / r.measured_s)],
            [x.map(f / b), y.map(f / r.predicted_s_tier_a)],
        );
        s.seg(cm, cp, Stroke::solid(0.75, th.muted));
        s.circle(
            cm,
            4.0,
            None,
            Some(Stroke::solid(1.25, th.bad)),
            Hit::Calib(*k as u32),
        );
        s.circle(cp, 1.5, Some(th.bad), None, Hit::None);
    }
    // Labels on the longest ops.
    for &i in order.iter().take(spec.top.min(8)) {
        let p = &pts[i];
        let c = [x.map(p.intensity()), y.map(p.attained())];
        if let Some(l) = text::fit(&p.label, 10.0, false, 180.0) {
            s.text(
                [c[0] + 8.0, c[1] - 8.0],
                l,
                10.0,
                th.fg,
                HAlign::Left,
                VAlign::Bottom,
            );
        }
    }
    s.unclip();
    // Legend.
    let lx = plot.right() + 24.0;
    s.text(
        [lx, 70.0],
        "binding term",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Top,
    );
    let mut bs: Vec<u8> = pts.iter().filter_map(|p| p.binding).collect();
    bs.sort_unstable();
    bs.dedup();
    let items: Vec<(Color, String)> = bs
        .iter()
        .map(|&b| {
            (
                binding_color(t.binding_name(b)),
                binding_label(t.binding_name(b)).to_string(),
            )
        })
        .collect();
    let mut yy = chart::swatches(&mut s, &th, lx, 90.0, &items) + 12.0;
    s.text(
        [lx, yy],
        "phase (marker)",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Top,
    );
    yy += 20.0;
    for p in &t.phases {
        if phase.is_some_and(|c| c != p.phase) {
            continue;
        }
        marker(
            &mut s,
            p.phase,
            [lx + 6.0, yy + 6.0],
            5.0,
            Some(cat(11)),
            Stroke::solid(0.5, th.bg),
            Hit::None,
        );
        s.text(
            [lx + 18.0, yy + 6.0],
            p.id.clone(),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
        yy += 18.0;
    }
    marker(
        &mut s,
        3,
        [lx + 6.0, yy + 8.0],
        6.0,
        None,
        Stroke::solid(2.0, th.fg),
        Hit::None,
    );
    s.text(
        [lx + 18.0, yy + 8.0],
        "phase aggregate",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Middle,
    );
    yy += 22.0;
    if !meas.is_empty() {
        s.circle(
            [lx + 6.0, yy + 6.0],
            4.0,
            None,
            Some(Stroke::solid(1.25, th.bad)),
            Hit::None,
        );
        let dev = meas[0].1.device.clone();
        s.text(
            [lx + 18.0, yy + 6.0],
            format!("measured ({dev}), line to prediction"),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
    }
    s
}
