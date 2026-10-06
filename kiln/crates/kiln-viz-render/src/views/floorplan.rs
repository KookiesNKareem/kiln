//! Floorplan (05 §6.1): blocks of the selected subtree, colored by utilization / idle / energy / bytes /
//! unit kind; containers drill into their children above `drill_px`, otherwise show the mean of their
//! leaves. Idle blocks are hatched, blocks without data get a dotted outline.

use kiln_trace::analysis::{ResStat, phase_code, resource_stats};
use kiln_trace::trace::Trace;

use crate::chart::{self, fmt_bytes, fmt_energy, fmt_pct};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{Theme, cat};
use crate::views::{FloorColor, Selection, ViewSpec};

const CONTAINERS: &[&str] = &[
    "system", "host", "board", "package", "die", "cluster", "switch",
];

pub struct Model {
    pub stats: Vec<ResStat>,
    /// Mean leaf value per row (containers) or own value (leaves), in the color mode's units.
    pub value: Vec<Option<f64>>,
    pub fp_row: Vec<Option<usize>>,
    pub children: Vec<Vec<u32>>,
    pub lo: f64,
    pub hi: f64,
    pub kinds: Vec<String>,
}

fn is_container(t: &Trace, i: usize) -> bool {
    CONTAINERS.contains(&t.resource_kind(&t.resources[i]))
}

pub fn model(t: &Trace, spec: &ViewSpec) -> Model {
    let phase = phase_code(t, spec.phase.as_deref());
    let stats = resource_stats(t, phase);
    let n = t.resources.len();
    let mut fp_row = vec![None; n];
    for (k, f) in t.floorplan.iter().enumerate() {
        if (f.resource as usize) < n {
            fp_row[f.resource as usize] = Some(k);
        }
    }
    let all_children = t.children();
    let children: Vec<Vec<u32>> = all_children
        .iter()
        .map(|c| {
            c.iter()
                .copied()
                .filter(|&x| fp_row[x as usize].is_some())
                .collect()
        })
        .collect();
    let mut kinds: Vec<String> = vec![];
    let own = |i: usize| -> Option<f64> {
        let s = stats[i];
        match spec.color {
            FloorColor::Utilization => s.util,
            FloorColor::Idle => s.util.map(|u| 1.0 - u),
            FloorColor::Energy => (s.energy_j > 0.0).then_some(s.energy_j),
            FloorColor::Bytes => (s.bytes > 0.0).then_some(s.bytes),
            FloorColor::Kind => None,
        }
    };
    let mut value = vec![None; n];
    let mut sum = vec![(0.0f64, 0u32); n];
    // Post-order: children have longer paths than parents, and rows are sorted by path, so a reverse scan
    // visits children before parents only within one subtree; do an explicit DFS instead.
    let roots: Vec<usize> = (0..n)
        .filter(|&i| t.resources[i].parent.is_none())
        .collect();
    let mut stack: Vec<(usize, bool)> = roots.iter().map(|&r| (r, false)).collect();
    while let Some((i, done)) = stack.pop() {
        if !done {
            stack.push((i, true));
            stack.extend(all_children[i].iter().map(|&c| (c as usize, false)));
            continue;
        }
        if is_container(t, i) {
            let (s, c) = all_children[i].iter().fold((0.0, 0), |acc, &ch| {
                (acc.0 + sum[ch as usize].0, acc.1 + sum[ch as usize].1)
            });
            sum[i] = (s, c);
            value[i] = (c > 0).then(|| match spec.color {
                FloorColor::Energy | FloorColor::Bytes => s,
                _ => s / f64::from(c),
            });
        } else if let Some(v) = own(i) {
            value[i] = Some(v);
            sum[i] = (v, 1);
        }
    }
    if spec.color == FloorColor::Kind {
        for r in &t.resources {
            let k = t.resource_kind(r).to_string();
            if !CONTAINERS.contains(&k.as_str()) && !kinds.contains(&k) {
                kinds.push(k);
            }
        }
        kinds.sort();
    }
    let leaf_vals: Vec<f64> = (0..n)
        .filter(|&i| !is_container(t, i) && fp_row[i].is_some())
        .filter_map(|i| value[i])
        .collect();
    let (lo, hi) = match spec.color {
        FloorColor::Utilization | FloorColor::Idle => (0.0, 1.0),
        _ => (
            leaf_vals
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .min(f64::INFINITY),
            leaf_vals.iter().copied().fold(0.0, f64::max),
        ),
    };
    Model {
        stats,
        value,
        fp_row,
        children,
        lo: if lo.is_finite() { lo } else { 0.0 },
        hi,
        kinds,
    }
}

fn norm(m: &Model, spec: &ViewSpec, v: f64) -> f64 {
    match spec.color {
        FloorColor::Utilization | FloorColor::Idle => v,
        _ if log_scale(m, spec) => (v.max(m.lo).ln() - m.lo.ln()) / (m.hi.ln() - m.lo.ln()),
        _ if m.hi > m.lo => (v - m.lo) / (m.hi - m.lo),
        _ => 1.0,
    }
}

fn log_scale(m: &Model, spec: &ViewSpec) -> bool {
    !matches!(spec.color, FloorColor::Utilization | FloorColor::Idle)
        && m.hi > 0.0
        && m.lo > 0.0
        && m.hi / m.lo > 100.0
}

fn denorm(m: &Model, spec: &ViewSpec, t: f64) -> f64 {
    match spec.color {
        FloorColor::Utilization | FloorColor::Idle => t,
        _ if log_scale(m, spec) => (m.lo.ln() + t * (m.hi.ln() - m.lo.ln())).exp(),
        _ => m.lo + t * (m.hi - m.lo),
    }
}

fn fmt_value(spec: &ViewSpec, v: f64) -> String {
    match spec.color {
        FloorColor::Utilization | FloorColor::Idle => fmt_pct(v),
        FloorColor::Energy => fmt_energy(v),
        FloorColor::Bytes => fmt_bytes(v),
        FloorColor::Kind => String::new(),
    }
}

/// Floorplan scene; `t.floorplan` must be non-empty (otherwise a banner).
pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let design = t
        .manifest
        .design_name
        .clone()
        .unwrap_or_else(|| short_hash(&t.manifest.provenance.design_hash));
    let unplaced = t.manifest.floorplan_source.as_deref() == Some(kiln_trace::layout::UNPLACED);
    let phase = spec.phase.clone().unwrap_or_else(|| "all phases".into());
    let sub = format!(
        "{}color: {} | {} | tier {:?}",
        if unplaced {
            "UNPLACED (hierarchy layout, nominal sizes, not physical) | "
        } else {
            ""
        },
        spec.color.name(),
        phase,
        t.manifest.tier
    );
    chart::title(&mut s, &th, &format!("Floorplan  {design}"), &sub);
    let legend_w = 150.0;
    let area = Rect::new(16.0, 60.0, spec.width - 32.0 - legend_w, spec.height - 76.0);
    if t.floorplan.is_empty() {
        chart::banner(
            &mut s,
            &th,
            area,
            "no floorplan in this trace",
            "re-run kiln eval with a design (-o run.kiln) to embed one",
        );
        return s;
    }
    let m = model(t, spec);
    let root = spec
        .root
        .as_deref()
        .and_then(|p| t.resource_by_path(p))
        .filter(|&r| m.fp_row[r as usize].is_some());
    let roots: Vec<u32> = match root {
        Some(r) => vec![r],
        None => (0..t.resources.len() as u32)
            .filter(|&i| t.resources[i as usize].parent.is_none() && m.fp_row[i as usize].is_some())
            .collect(),
    };
    // World bounds of the roots.
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &r in &roots {
        let f = &t.floorplan[m.fp_row[r as usize].expect("placed")];
        x0 = x0.min(f.x_um);
        y0 = y0.min(f.y_um);
        x1 = x1.max(f.x_um + f.w_um);
        y1 = y1.max(f.y_um + f.h_um);
    }
    let k = ((f64::from(area.w) / (x1 - x0)).min(f64::from(area.h) / (y1 - y0))) as f32;
    let ox = area.x + (area.w - (x1 - x0) as f32 * k) / 2.0;
    let oy = area.y + (area.h - (y1 - y0) as f32 * k) / 2.0;
    let to_px = |r: usize| {
        let f = &t.floorplan[m.fp_row[r].expect("placed")];
        Rect::new(
            ox + ((f.x_um - x0) as f32) * k,
            oy + ((f.y_um - y0) as f32) * k,
            f.w_um as f32 * k,
            f.h_um as f32 * k,
        )
    };
    let color_of = |i: usize| -> Option<Color> {
        if spec.color == FloorColor::Kind {
            let kind = t.resource_kind(&t.resources[i]);
            return m.kinds.iter().position(|x| x == kind).map(cat);
        }
        m.value[i].map(|v| spec.cmap.at(norm(&m, spec, v)))
    };
    let mut stack: Vec<u32> = roots.iter().rev().copied().collect();
    let mut drawn = 0usize;
    let mut labels: Vec<(Rect, String, Color, bool)> = vec![];
    while let Some(r) = stack.pop() {
        let i = r as usize;
        let px = to_px(i);
        if px.w < 0.5 || px.h < 0.5 {
            continue;
        }
        drawn += 1;
        let container = is_container(t, i);
        let path = &t.resources[i].path;
        let name = path.rsplit('.').next().unwrap_or(path).to_string();
        let selected = sel.has(Hit::Resource(r));
        let open = container && px.w.min(px.h) > spec.drill_px && !m.children[i].is_empty();
        if container && open {
            s.rect(
                px,
                Some(th.panel),
                Some(Stroke::solid(1.0, th.outline)),
                Hit::Resource(r),
            );
            // Header strip of the hierarchy layout (6% of the height, capped by the width).
            if spec.labels && px.w > 40.0 && (0.06 * px.h).min(0.25 * px.w) >= 9.0 {
                labels.push((
                    Rect::new(px.x + 3.0, px.y + 1.0, px.w - 6.0, 10.0),
                    name,
                    th.muted,
                    false,
                ));
            }
            stack.extend(m.children[i].iter().rev().copied());
        } else {
            let fill = color_of(i);
            match fill {
                Some(c) => {
                    s.rect(
                        px,
                        Some(c),
                        Some(Stroke::solid(0.5, th.bg)),
                        Hit::Resource(r),
                    );
                    let idle = matches!(spec.color, FloorColor::Utilization)
                        && m.value[i].is_some_and(|v| v <= 1e-6);
                    if idle && px.w > 4.0 && px.h > 4.0 {
                        s.hatch(px, 5.0, Stroke::solid(0.75, th.hatch));
                    }
                    if spec.labels && px.w > 40.0 && px.h > 14.0 {
                        labels.push((px, name, th.on(c), true));
                    }
                }
                None => {
                    s.rect(
                        px,
                        Some(th.idle),
                        Some(Stroke::dashed(0.75, th.outline, 2.0)),
                        Hit::Resource(r),
                    );
                    if spec.labels && px.w > 40.0 && px.h > 14.0 {
                        labels.push((px, name, th.muted, true));
                    }
                }
            }
        }
        if selected {
            s.rect(px, None, Some(Stroke::solid(2.5, th.accent)), Hit::None);
        }
    }
    for (r, name, c, centered) in labels {
        let size = if centered {
            (r.h * 0.4).clamp(8.0, 12.0)
        } else {
            10.0
        };
        if let Some(txt) = text::fit(&name, size, false, r.w - 4.0) {
            if centered {
                s.text(
                    [r.x + r.w / 2.0, r.y + r.h / 2.0],
                    txt,
                    size,
                    c,
                    HAlign::Center,
                    VAlign::Middle,
                );
            } else {
                s.text([r.x, r.y], txt, size, c, HAlign::Left, VAlign::Top);
            }
        }
    }
    // Legend.
    let lx = spec.width - legend_w;
    if spec.color == FloorColor::Kind {
        let items: Vec<(Color, String)> = m
            .kinds
            .iter()
            .enumerate()
            .map(|(i, k)| (cat(i), k.clone()))
            .collect();
        s.text(
            [lx, 70.0],
            "unit kind",
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
        chart::swatches(&mut s, &th, lx, 90.0, &items);
    } else {
        let cm = spec.cmap;
        let label = match spec.color {
            FloorColor::Utilization => "busy fraction",
            FloorColor::Idle => "idle fraction",
            FloorColor::Energy => "energy",
            FloorColor::Bytes => "bytes moved",
            FloorColor::Kind => "",
        };
        let mm = &m;
        let fmt_t = |t: f64| fmt_value(spec, denorm(mm, spec, t));
        chart::gradient_legend(
            &mut s,
            &th,
            Rect::new(lx, 90.0, 16.0, 220.0),
            &|t| cm.at(t),
            label,
            &fmt_t,
        );
        legend_marks(
            &mut s,
            &th,
            lx,
            330.0,
            spec.color == FloorColor::Utilization,
        );
    }
    s.text(
        [spec.width - 16.0, spec.height - 8.0],
        format!("{drawn} blocks drawn of {} placed", t.floorplan.len()),
        10.0,
        th.muted,
        HAlign::Right,
        VAlign::Bottom,
    );
    s
}

fn legend_marks(s: &mut Scene, th: &Theme, x: f32, y: f32, idle: bool) {
    let mut yy = y;
    if idle {
        let r = Rect::new(x, yy, 14.0, 14.0);
        s.rect(
            r,
            Some(crate::theme::ColorMap::Cividis.at(0.0)),
            None,
            Hit::None,
        );
        s.hatch(r, 4.0, Stroke::solid(0.75, th.hatch));
        s.text(
            [x + 20.0, yy + 7.0],
            "idle (0%)",
            11.0,
            th.fg,
            HAlign::Left,
            VAlign::Middle,
        );
        yy += 20.0;
    }
    s.rect(
        Rect::new(x, yy, 14.0, 14.0),
        Some(th.idle),
        Some(Stroke::dashed(0.75, th.outline, 2.0)),
        Hit::None,
    );
    s.text(
        [x + 20.0, yy + 7.0],
        "no data",
        11.0,
        th.fg,
        HAlign::Left,
        VAlign::Middle,
    );
}

pub fn short_hash(h: &str) -> String {
    let tail = h.split('-').next_back().unwrap_or(h);
    tail.chars().take(10).collect()
}
