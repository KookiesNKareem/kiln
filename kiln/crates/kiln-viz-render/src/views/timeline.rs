//! Timeline (05 §6.3), coarse Tier A form: phase and execution-group tracks, one row per op family from the
//! op envelopes, then span lanes per resource when the trace has spans (`ops` level). Tier A slices are
//! estimates and drawn with dashed outlines under an "analytical schedule" banner.

use std::collections::BTreeMap;

use kiln_trace::analysis::phase_code;
use kiln_trace::trace::{NONE_U32, Trace, span_flags};

use crate::chart::{self, fmt_time};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{binding_color, cat};
use crate::views::{Selection, ViewSpec};

/// `(start, end, color, label, hit, estimated)` in ticks.
pub type Slice = (i64, i64, Color, String, Hit, bool);

pub struct Track {
    pub label: String,
    pub slices: Vec<Slice>,
}

/// Visible tick window `[t0, t1]`.
pub fn window(t: &Trace, spec: &ViewSpec) -> (i64, i64) {
    let phase = phase_code(t, spec.phase.as_deref());
    let ph: Vec<_> = t
        .phases
        .iter()
        .filter(|p| phase.is_none_or(|c| c == p.phase))
        .collect();
    let (a, b) = (
        ph.iter().map(|p| p.t_offset).min().unwrap_or(0),
        ph.iter().map(|p| p.t_end).max().unwrap_or(1),
    );
    match spec.window {
        Some([w0, w1]) => (
            (w0 / t.tick_s()).round() as i64,
            ((w1 / t.tick_s()).round() as i64).max((w0 / t.tick_s()).round() as i64 + 1),
        ),
        None => (a, b.max(a + 1)),
    }
}

pub fn tracks(t: &Trace, spec: &ViewSpec) -> Vec<Track> {
    let phase = phase_code(t, spec.phase.as_deref());
    let kind_color = |k: u16| cat(usize::from(k));
    let mut out = vec![];
    out.push(Track {
        label: "phases".into(),
        slices: t
            .phases
            .iter()
            .filter(|p| phase.is_none_or(|c| c == p.phase))
            .map(|p| {
                (
                    p.t_offset,
                    p.t_end,
                    cat(9 + usize::from(p.phase)),
                    p.id.clone(),
                    Hit::Phase(p.phase),
                    false,
                )
            })
            .collect(),
    });
    out.push(Track {
        label: "execution groups".into(),
        slices: t
            .groups
            .iter()
            .filter(|g| phase.is_none_or(|c| c == g.phase))
            .map(|g| {
                (
                    g.t_start,
                    g.t_end,
                    binding_color(t.binding_name(g.binding)),
                    format!("g{}", g.group),
                    Hit::None,
                    true,
                )
            })
            .collect(),
    });
    let mut fam: BTreeMap<(u8, &str), Vec<Slice>> = BTreeMap::new();
    let mut first: BTreeMap<(u8, &str), i64> = BTreeMap::new();
    for (i, o) in t.ops.iter().enumerate() {
        if phase.is_some_and(|c| c != o.phase) {
            continue;
        }
        let k = (o.phase, o.family.as_str());
        fam.entry(k).or_default().push((
            o.t_start,
            o.t_end,
            kind_color(o.kind),
            o.path.clone(),
            Hit::Op(i as u32),
            true,
        ));
        let f = first.entry(k).or_insert(o.t_start);
        *f = (*f).min(o.t_start);
    }
    let mut keys: Vec<(u8, &str)> = fam.keys().copied().collect();
    keys.sort_by_key(|k| (k.0, first[k], k.1));
    for k in keys {
        out.push(Track {
            label: format!("op {}", k.1),
            slices: fam.remove(&k).unwrap_or_default(),
        });
    }
    let mut lanes: BTreeMap<(u32, u16), Vec<Slice>> = BTreeMap::new();
    for s in &t.spans {
        let op = (s.op != NONE_U32).then(|| &t.ops[s.op as usize]);
        if op.is_some_and(|o| phase.is_some_and(|c| c != o.phase)) {
            continue;
        }
        lanes.entry((s.resource, s.lane)).or_default().push((
            s.t_start,
            s.t_start + s.dur,
            op.map_or(Color::rgb(150, 150, 150), |o| kind_color(o.kind)),
            op.map_or_else(String::new, |o| o.path.clone()),
            if s.op != NONE_U32 {
                Hit::Op(s.op)
            } else {
                Hit::Resource(s.resource)
            },
            s.flags & span_flags::ESTIMATED != 0,
        ));
    }
    for ((r, l), sl) in lanes {
        let p = &t.resources[r as usize].path;
        out.push(Track {
            label: if l == 0 {
                p.clone()
            } else {
                format!("{p} #{l}")
            },
            slices: sl,
        });
    }
    out
}

pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let design = t.manifest.design_name.clone().unwrap_or_default();
    let (t0, t1) = window(t, spec);
    let tick = t.tick_s();
    chart::title(
        &mut s,
        &th,
        &format!("Timeline  {design}"),
        &format!(
            "window {} .. {} | {} | {}",
            fmt_time(t0 as f64 * tick),
            fmt_time(t1 as f64 * tick),
            if t.spans.is_empty() {
                "op envelopes per family (summary level)"
            } else {
                "op envelopes per family, then span lanes per binding resource (ops level)"
            },
            if t.manifest.tier == kiln_trace::Tier::A {
                "Tier A analytical schedule of the repeat window, no contention (dashed = estimated)"
            } else {
                "Tier B"
            }
        ),
    );
    let label_w = 230.0;
    let plot = Rect::new(
        16.0 + label_w,
        84.0,
        spec.width - 32.0 - label_w,
        spec.height - 100.0,
    );
    if t.phases.is_empty() {
        chart::banner(
            &mut s,
            &th,
            plot,
            "timeline requires ops",
            "re-run with kiln eval --trace ops -o run.kiln",
        );
        return s;
    }
    let tr = tracks(t, spec);
    let row_h = 16.0f32;
    let gap = 3.0;
    let max_rows = ((plot.h - 4.0) / (row_h + gap)).floor().max(1.0) as usize;
    let x = |ticks: i64| plot.x + ((ticks - t0) as f64 / (t1 - t0) as f64) as f32 * plot.w;
    // Time axis.
    let ax = chart::Axis::new(
        t0 as f64 * tick,
        t1 as f64 * tick,
        false,
        plot.x,
        plot.right(),
    );
    for v in ax.ticks() {
        let px = ax.map(v);
        s.seg(
            [px, plot.y - 4.0],
            [px, plot.bottom()],
            Stroke::solid(1.0, th.grid),
        );
        s.text(
            [px, plot.y - 6.0],
            fmt_time(v),
            10.0,
            th.muted,
            HAlign::Center,
            VAlign::Bottom,
        );
    }
    s.clip(Rect::new(plot.x, plot.y - 2.0, plot.w, plot.h + 2.0));
    for (k, track) in tr.iter().take(max_rows).enumerate() {
        let y = plot.y + k as f32 * (row_h + gap);
        for (a, b, c, label, hit, est) in &track.slices {
            if *b < t0 || *a > t1 {
                continue;
            }
            let (xa, xb) = (x(*a).max(plot.x - 1.0), x(*b).min(plot.right() + 1.0));
            let w = (xb - xa).max(1.0);
            let r = Rect::new(xa, y, w, row_h);
            let stroke = if sel.has(*hit) {
                Some(Stroke::solid(2.0, th.accent))
            } else if *est && w > 3.0 {
                Some(Stroke::dashed(0.75, th.fg.with_alpha(140), 2.0))
            } else {
                None
            };
            s.rect(r, Some(*c), stroke, *hit);
            if w > 40.0
                && let Some(l) = text::fit(label, 10.0, false, w - 4.0)
            {
                s.text(
                    [xa + 2.0, y + row_h / 2.0],
                    l,
                    10.0,
                    th.on(*c),
                    HAlign::Left,
                    VAlign::Middle,
                );
            }
        }
    }
    s.unclip();
    for (k, track) in tr.iter().take(max_rows).enumerate() {
        let y = plot.y + k as f32 * (row_h + gap);
        if let Some(l) = text::fit(&track.label, 10.0, false, label_w - 8.0) {
            s.text(
                [plot.x - 8.0, y + row_h / 2.0],
                l,
                10.0,
                if k < 2 { th.fg } else { th.muted },
                HAlign::Right,
                VAlign::Middle,
            );
        }
    }
    if tr.len() > max_rows {
        s.text(
            [plot.x - 8.0, plot.bottom() + 4.0],
            format!("+{} more tracks", tr.len() - max_rows),
            10.0,
            th.muted,
            HAlign::Right,
            VAlign::Top,
        );
    }
    // Op kind legend.
    let kinds = t
        .manifest
        .enums
        .get("ops.kind")
        .cloned()
        .unwrap_or_default();
    let mut lx = plot.x;
    for (i, k) in kinds.iter().enumerate() {
        s.rect(
            Rect::new(lx, 58.0, 10.0, 10.0),
            Some(cat(i)),
            None,
            Hit::None,
        );
        s.text(
            [lx + 14.0, 63.0],
            k.clone(),
            10.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
        lx += 24.0 + text::measure(k, 10.0, false);
    }
    s
}
