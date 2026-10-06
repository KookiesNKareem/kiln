//! Design diff / compare (05 §6.6): small-multiple floorplans on one color scale, a headline table with
//! intervals and deltas against the first run, and a per-family waterfall with limiter transitions.

use kiln_trace::analysis::align;
use kiln_trace::trace::Trace;

use crate::chart::{self, fmt_energy, fmt_power, fmt_time};
use crate::scene::{HAlign, Hit, Prim, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::binding_label;
use crate::views::{Selection, ViewKind, ViewSpec, floorplan};

/// Copies `src` into `dst` translated by `(dx, dy)` and clipped to its own bounds.
pub fn embed(dst: &mut Scene, src: &Scene, dx: f32, dy: f32) {
    dst.clip(Rect::new(dx, dy, src.width, src.height));
    let mv = |p: [f32; 2]| [p[0] + dx, p[1] + dy];
    for p in &src.prims {
        dst.prims.push(match p.clone() {
            Prim::Rect {
                r,
                fill,
                stroke,
                radius,
                hit,
            } => Prim::Rect {
                r: Rect::new(r.x + dx, r.y + dy, r.w, r.h),
                fill,
                stroke,
                radius,
                hit,
            },
            Prim::Line { pts, stroke, hit } => Prim::Line {
                pts: pts.into_iter().map(mv).collect(),
                stroke,
                hit,
            },
            Prim::Polygon {
                pts,
                fill,
                stroke,
                hit,
            } => Prim::Polygon {
                pts: pts.into_iter().map(mv).collect(),
                fill,
                stroke,
                hit,
            },
            Prim::Circle {
                c,
                r,
                fill,
                stroke,
                hit,
            } => Prim::Circle {
                c: mv(c),
                r,
                fill,
                stroke,
                hit,
            },
            Prim::Text {
                pos,
                text,
                size,
                color,
                h,
                v,
                mono,
                vertical,
            } => Prim::Text {
                pos: mv(pos),
                text,
                size,
                color,
                h,
                v,
                mono,
                vertical,
            },
            Prim::ClipPush(r) => Prim::ClipPush(Rect::new(r.x + dx, r.y + dy, r.w, r.h)),
            Prim::ClipPop => Prim::ClipPop,
        });
    }
    dst.unclip();
}

/// One headline value: central and optional `(low, high)`.
type Cell = Option<(f64, Option<(f64, f64)>)>;

pub fn run_name(t: &Trace, i: usize) -> String {
    t.manifest
        .design_name
        .clone()
        .unwrap_or_else(|| format!("run {}", (b'A' + i as u8) as char))
}

fn pct(a: f64, b: f64) -> String {
    if a == 0.0 {
        "-".into()
    } else {
        format!("{:+.1}%", 100.0 * (b - a) / a)
    }
}

pub fn scene(runs: &[&Trace], spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let names: Vec<String> = runs
        .iter()
        .enumerate()
        .map(|(i, t)| run_name(t, i))
        .collect();
    chart::title(
        &mut s,
        &th,
        &format!("Compare  {}", names.join("  vs  ")),
        "floorplans share one utilization scale | deltas are B - A relative to A; intervals [low, high] from the parameter corners",
    );
    if runs.len() < 2 {
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "compare needs two or more runs",
            "kiln viz --compare a.kiln b.kiln",
        );
        return s;
    }
    // Floorplans.
    let table_w = 470.0;
    let fp_h = (spec.height * 0.52).max(240.0);
    let n = runs.len() as f32;
    let fp_w = ((spec.width - 32.0 - table_w) / n).max(160.0);
    for (i, t) in runs.iter().enumerate() {
        let sub = ViewSpec {
            view: ViewKind::Floorplan,
            width: fp_w,
            height: fp_h,
            labels: false,
            drill_px: 18.0,
            ..spec.clone()
        };
        let fs = floorplan::scene(t, &sub, sel);
        embed(&mut s, &fs, 16.0 + i as f32 * fp_w, 56.0);
    }
    // Headline table.
    let tx = spec.width - table_w - 8.0;
    let mut ty = 64.0;
    let col =
        |k: usize| tx + 150.0 + k as f32 * ((table_w - 160.0) / (runs.len() as f32 + 1.0).max(2.0));
    s.text([tx, ty], "Headline", 13.0, th.fg, HAlign::Left, VAlign::Top);
    for (k, nm) in names.iter().enumerate() {
        if let Some(l) = text::fit(nm, 11.0, false, 90.0) {
            s.text(
                [col(k), ty + 2.0],
                l,
                11.0,
                th.muted,
                HAlign::Left,
                VAlign::Top,
            );
        }
    }
    s.text(
        [col(runs.len()), ty + 2.0],
        "delta B/A",
        11.0,
        th.muted,
        HAlign::Left,
        VAlign::Top,
    );
    ty += 22.0;
    let phase_ids: Vec<String> = runs[0].phases.iter().map(|p| p.id.clone()).collect();
    let row = |s: &mut Scene,
               ty: &mut f32,
               label: String,
               vals: Vec<Cell>,
               fmt: &dyn Fn(f64) -> String| {
        s.text([tx, *ty], label, 11.0, th.fg, HAlign::Left, VAlign::Top);
        for (k, v) in vals.iter().enumerate() {
            if let Some((c, iv)) = v {
                s.text(
                    [col(k), *ty],
                    fmt(*c),
                    11.0,
                    th.fg,
                    HAlign::Left,
                    VAlign::Top,
                );
                if let Some((lo, hi)) = iv
                    && hi > lo
                {
                    s.text(
                        [col(k), *ty + 13.0],
                        format!("[{}, {}]", fmt(*lo), fmt(*hi)),
                        9.0,
                        th.muted,
                        HAlign::Left,
                        VAlign::Top,
                    );
                }
            }
        }
        if let (Some(Some((a, _))), Some(Some((b, _)))) = (vals.first(), vals.get(1)) {
            let d = pct(*a, *b);
            s.text(
                [col(runs.len()), *ty],
                d,
                11.0,
                th.fg,
                HAlign::Left,
                VAlign::Top,
            );
        }
        *ty += 28.0;
    };
    for ph in &phase_ids {
        let get = |t: &Trace| t.phases.iter().find(|p| &p.id == ph).cloned();
        let vals: Vec<Cell> = runs
            .iter()
            .map(|t| get(t).map(|p| (p.makespan_s, p.makespan_low_s.zip(p.makespan_high_s))))
            .collect();
        row(&mut s, &mut ty, format!("{ph} time"), vals, &fmt_time);
        let e: Vec<_> = runs
            .iter()
            .map(|t| get(t).map(|p| (p.energy_j, None)))
            .collect();
        row(&mut s, &mut ty, format!("{ph} energy"), e, &fmt_energy);
        let p: Vec<_> = runs
            .iter()
            .map(|t| get(t).map(|p| (p.avg_power_w, None)))
            .collect();
        row(&mut s, &mut ty, format!("{ph} avg power"), p, &fmt_power);
        if ty > 56.0 + fp_h - 20.0 {
            break;
        }
    }
    // Physical headline (04) when the runs carry it.
    let area: Vec<Cell> = runs
        .iter()
        .map(|t| {
            t.manifest
                .headline
                .area_mm2
                .map(|i| (i.central, Some((i.low, i.high))))
        })
        .collect();
    if area.iter().any(Option::is_some) {
        row(&mut s, &mut ty, "package area (mm^2)".into(), area, &|v| {
            format!("{v:.0}")
        });
    }
    let power: Vec<Cell> = runs
        .iter()
        .map(|t| {
            t.manifest
                .headline
                .power_w
                .map(|i| (i.central, Some((i.low, i.high))))
        })
        .collect();
    if power.iter().any(Option::is_some) {
        row(&mut s, &mut ty, "power".into(), power, &fmt_power);
    }
    if let Some(sc) = runs[1].manifest.headline.score {
        s.text(
            [tx, ty],
            format!(
                "score {} (vs its baseline) [{:.3}, {:.3}]",
                chart::fmt_num(sc.central),
                sc.low,
                sc.high
            ),
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
    }
    // Waterfall A vs B.
    let (rows, same_wl) = align(runs[0], runs[1]);
    let wy = 56.0 + fp_h + 16.0;
    s.text(
        [16.0, wy],
        format!(
            "Per op family delta, {} -> {} (sorted by |delta time|){}",
            names[0],
            names[1],
            if same_wl {
                ""
            } else {
                "  WARNING: workloads differ, matched by path"
            }
        ),
        13.0,
        if same_wl { th.fg } else { th.bad },
        HAlign::Left,
        VAlign::Top,
    );
    let max = rows
        .iter()
        .map(|r| r.delta().abs())
        .fold(0.0, f64::max)
        .max(1e-300);
    let label_w = 220.0;
    let zero = 16.0 + label_w + (spec.width - 32.0 - label_w - 300.0) / 2.0;
    let half = (spec.width - 32.0 - label_w - 300.0) / 2.0;
    let avail = ((spec.height - wy - 30.0) / 18.0).max(1.0) as usize;
    s.seg(
        [zero, wy + 22.0],
        [zero, wy + 22.0 + avail.min(rows.len()) as f32 * 18.0],
        Stroke::solid(1.0, th.outline),
    );
    for (k, r) in rows.iter().take(avail.min(spec.top.max(10))).enumerate() {
        let y = wy + 24.0 + k as f32 * 18.0;
        if let Some(l) = text::fit(&r.key, 11.0, false, label_w - 8.0) {
            s.text(
                [16.0 + label_w - 8.0, y + 7.0],
                l,
                11.0,
                th.fg,
                HAlign::Right,
                VAlign::Middle,
            );
        }
        let d = r.delta();
        let w = (d.abs() / max) as f32 * half;
        let (x0, c) = if d < 0.0 {
            (zero - w, th.good)
        } else {
            (zero, th.bad)
        };
        s.rect(
            Rect::new(x0, y, w.max(1.0), 14.0),
            Some(c),
            None,
            Hit::RunOp(1, k as u32),
        );
        let lim = |b: Option<u8>, t: &Trace| {
            b.map_or("-".to_string(), |b| {
                binding_label(t.binding_name(b)).to_string()
            })
        };
        let trans = if r.a_binding == r.b_binding {
            lim(r.a_binding, runs[0])
        } else {
            format!(
                "{} -> {}",
                lim(r.a_binding, runs[0]),
                lim(r.b_binding, runs[1])
            )
        };
        s.text(
            [zero + half + 8.0, y + 7.0],
            format!(
                "{}{}  {trans}",
                if d >= 0.0 { "+" } else { "-" },
                fmt_time(d.abs())
            ),
            11.0,
            th.muted,
            HAlign::Left,
            VAlign::Middle,
        );
    }
    s
}
