//! Bottleneck attribution (05 §6.5): run time by limiter, per-family stacked bars split by binding term,
//! top limiters with recoverable time (upper bound), energy split, and 03's `explain_run` text verbatim.

use kiln_trace::analysis::{family_breakdown, phase_code, recoverable, run_time, time_by_binding};
use kiln_trace::trace::Trace;

use crate::chart::{self, fmt_energy, fmt_time};
use crate::scene::{HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{binding_color, binding_label, cat};
use crate::views::{Selection, ViewSpec};

pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let phase = phase_code(t, spec.phase.as_deref());
    let design = t.manifest.design_name.clone().unwrap_or_default();
    let total = run_time(t, phase);
    chart::title(
        &mut s,
        &th,
        &format!("Bottleneck  {design}"),
        &format!(
            "{} | makespan {} | tier {:?} (Tier A: per-group attribution, no contention)",
            spec.phase.as_deref().unwrap_or("all phases"),
            fmt_time(total),
            t.manifest.tier
        ),
    );
    if t.phases.is_empty() {
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "no simulated phases in this trace",
            "",
        );
        return s;
    }
    let left = 16.0;
    let w_left = (spec.width * 0.58).max(400.0);
    // 1. Run time by limiter.
    let tb = time_by_binding(t, phase);
    let sum: f64 = tb.values().sum();
    let mut y = 70.0;
    s.text(
        [left, y],
        "Time by limiter (sums to the makespan)",
        13.0,
        th.fg,
        HAlign::Left,
        VAlign::Top,
    );
    y += 22.0;
    let bar = Rect::new(left, y, w_left - 32.0, 26.0);
    let mut x = bar.x;
    for (b, v) in &tb {
        let w = if sum > 0.0 {
            (v / sum) as f32 * bar.w
        } else {
            0.0
        };
        let c = binding_color(t.binding_name(*b));
        s.rect(
            Rect::new(x, bar.y, w, bar.h),
            Some(c),
            Some(Stroke::solid(0.5, th.bg)),
            Hit::Binding(*b),
        );
        let lab = format!(
            "{} {:.0}%",
            binding_label(t.binding_name(*b)),
            100.0 * v / sum.max(1e-300)
        );
        if let Some(l) = text::fit(&lab, 11.0, false, w - 6.0) {
            s.text(
                [x + 4.0, bar.y + bar.h / 2.0],
                l,
                11.0,
                th.on(c),
                HAlign::Left,
                VAlign::Middle,
            );
        }
        x += w;
    }
    y += 40.0;
    // 2. Per-family stacked bars.
    let fams = family_breakdown(t, phase);
    let sched: f64 = fams.iter().map(|f| f.1.values().sum::<f64>()).sum();
    s.text([left, y], "Scheduled time by op family (repeat window; group time attributed by roofline bound), split by binding term", 13.0, th.fg, HAlign::Left, VAlign::Top);
    y += 22.0;
    let label_w = 170.0;
    let bar_w = w_left - 32.0 - label_w - 60.0;
    let rows = ((spec.height - y - 200.0) / 20.0).max(4.0) as usize;
    let fmax = fams.first().map_or(1.0, |f| f.1.values().sum::<f64>());
    for (k, (name, parts)) in fams.iter().take(rows.min(spec.top.max(12))).enumerate() {
        let ry = y + k as f32 * 20.0;
        if let Some(l) = text::fit(name, 11.0, false, label_w - 8.0) {
            s.text(
                [left + label_w - 8.0, ry + 8.0],
                l,
                11.0,
                th.fg,
                HAlign::Right,
                VAlign::Middle,
            );
        }
        let mut bx = left + label_w;
        for (b, v) in parts {
            let w = (v / fmax) as f32 * bar_w;
            let name_b = if *b == u8::MAX {
                "?"
            } else {
                t.binding_name(*b)
            };
            s.rect(
                Rect::new(bx, ry, w, 16.0),
                Some(binding_color(name_b)),
                None,
                Hit::Family(k as u32),
            );
            bx += w;
        }
        let ft: f64 = parts.values().sum();
        s.text(
            [bx + 6.0, ry + 8.0],
            format!("{} ({:.0}%)", fmt_time(ft), 100.0 * ft / sched.max(1e-300)),
            10.0,
            th.muted,
            HAlign::Left,
            VAlign::Middle,
        );
        if sel.has(Hit::Family(k as u32)) {
            s.rect(
                Rect::new(left + label_w, ry, bx - left - label_w, 16.0),
                None,
                Some(Stroke::solid(2.0, th.accent)),
                Hit::None,
            );
        }
    }
    let y_after = y + rows.min(fams.len()).min(spec.top.max(12)) as f32 * 20.0 + 16.0;
    // 3. Energy split (central corner, from run scalars).
    let mut ey = y_after;
    let phases: Vec<&kiln_trace::trace::PhaseRow> = t
        .phases
        .iter()
        .filter(|p| phase.is_none_or(|c| c == p.phase))
        .collect();
    let mut parts: Vec<(String, f64)> = vec![];
    for p in &phases {
        if let Some(e) = t.scalar(&format!("energy.{}", p.id)) {
            let mut add = |k: String, v: f64| {
                if v > 0.0 {
                    match parts.iter_mut().find(|x| x.0 == k) {
                        Some(x) => x.1 += v,
                        None => parts.push((k, v)),
                    }
                }
            };
            add("compute".into(), e["compute_j"].as_f64().unwrap_or(0.0));
            for (k, v) in e["memory_j"].as_object().into_iter().flatten() {
                add(format!("mem {k}"), v.as_f64().unwrap_or(0.0));
            }
            for (k, v) in e["link_j"].as_object().into_iter().flatten() {
                add(format!("link {k}"), v.as_f64().unwrap_or(0.0));
            }
            add("near-memory".into(), e["nmp_j"].as_f64().unwrap_or(0.0));
            add("static".into(), e["static_j"].as_f64().unwrap_or(0.0));
            add(
                "conversion".into(),
                e["conversion_j"].as_f64().unwrap_or(0.0),
            );
            add("padding".into(), e["padding_j"].as_f64().unwrap_or(0.0));
        }
    }
    let etot: f64 = parts.iter().map(|p| p.1).sum();
    if etot > 0.0 && ey + 80.0 < spec.height {
        s.text(
            [left, ey],
            format!("Energy ({})", fmt_energy(etot)),
            13.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
        ey += 22.0;
        let bar = Rect::new(left, ey, w_left - 32.0, 22.0);
        let mut x = bar.x;
        for (k, (name, v)) in parts.iter().enumerate() {
            let w = (v / etot) as f32 * bar.w;
            let c = cat(k);
            s.rect(
                Rect::new(x, bar.y, w, bar.h),
                Some(c),
                Some(Stroke::solid(0.5, th.bg)),
                Hit::None,
            );
            if let Some(l) = text::fit(
                &format!("{name} {:.0}%", 100.0 * v / etot),
                10.0,
                false,
                w - 4.0,
            ) {
                s.text(
                    [x + 3.0, bar.y + bar.h / 2.0],
                    l,
                    10.0,
                    th.on(c),
                    HAlign::Left,
                    VAlign::Middle,
                );
            }
            x += w;
        }
        ey += 34.0;
    }
    // 4. Top limiters (right column).
    let rx = left + w_left;
    let rw = spec.width - rx - 16.0;
    let mut ry = 70.0;
    s.text(
        [rx, ry],
        "Top limiters: time recoverable if lifted (upper bound)",
        13.0,
        th.fg,
        HAlign::Left,
        VAlign::Top,
    );
    ry += 24.0;
    for (k, (b, res, gain, n)) in recoverable(t, phase).iter().take(spec.top).enumerate() {
        let name = t.binding_name(*b);
        let what = match res {
            Some(r) => format!(
                "{} on {}",
                binding_label(name),
                t.resources[*r as usize].path
            ),
            None => binding_label(name).to_string(),
        };
        s.rect(
            Rect::new(rx, ry + 2.0, 10.0, 10.0),
            Some(binding_color(name)),
            None,
            Hit::Binding(*b),
        );
        let line = format!("{}. {what}", k + 1);
        if let Some(l) = text::fit(&line, 11.0, false, rw - 150.0) {
            s.text(
                [rx + 16.0, ry + 7.0],
                l,
                11.0,
                th.fg,
                HAlign::Left,
                VAlign::Middle,
            );
        }
        s.text(
            [rx + rw, ry + 7.0],
            format!(
                "-{} ({:.0}% of window), {n} ops",
                fmt_time(*gain),
                100.0 * gain / sched.max(1e-300)
            ),
            11.0,
            th.muted,
            HAlign::Right,
            VAlign::Middle,
        );
        ry += 20.0;
    }
    ry += 10.0;
    // Shadow prices (03: makespan sensitivity to +10% capacity).
    let tops: Vec<&kiln_trace::trace::BottleneckRow> = t
        .bottleneck
        .iter()
        .filter(|b| b.section == 1 && phase.is_none_or(|p| p == b.phase))
        .collect();
    if !tops.is_empty() {
        s.text(
            [rx, ry],
            "Busiest resources (utilization, shadow price)",
            13.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
        ry += 22.0;
        for b in tops.iter().take(6) {
            let path = b
                .resource
                .map_or("?".to_string(), |r| t.resources[r as usize].path.clone());
            if let Some(l) = text::fit(&path, 11.0, false, rw - 120.0) {
                s.text([rx, ry + 7.0], l, 11.0, th.fg, HAlign::Left, VAlign::Middle);
            }
            s.text(
                [rx + rw, ry + 7.0],
                format!(
                    "{:.0}%  sp {:.2}",
                    100.0 * b.utilization.unwrap_or(0.0),
                    b.shadow_price.unwrap_or(0.0)
                ),
                11.0,
                th.muted,
                HAlign::Right,
                VAlign::Middle,
            );
            ry += 18.0;
        }
    }
    // 5. explain_run text, verbatim.
    let ty = ey.max(ry + 16.0);
    if ty + 40.0 < spec.height {
        s.text(
            [left, ty],
            "Explanation (kiln explain_run, as given to the evolution LLM)",
            13.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
        let mut yy = ty + 22.0;
        for p in &phases {
            for line in text::wrap(&p.summary, 11.0, false, spec.width - 48.0) {
                if yy + 14.0 > spec.height - 8.0 {
                    break;
                }
                s.text([left, yy], line, 11.0, th.fg, HAlign::Left, VAlign::Top);
                yy += 15.0;
            }
            yy += 6.0;
        }
    }
    s
}
