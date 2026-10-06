//! Floorplan on kiln-phys's placement (05 §6.1): package and die outlines to scale (mm), placed blocks at
//! their positions and areas, PHYs and their shoreline, memory stacks, harvested parts, router sites, and the
//! links between drawn blocks (each channel routed as a Manhattan L, aggregated onto the blocks drawn at the
//! current depth). Stacked-die layers are drawn side by side, or one layer with the others as outlines.
//! Containers open into their children above `drill_px`, as in the hierarchy view.

use std::collections::{BTreeMap, BTreeSet};

use kiln_trace::trace::{FloorplanRow, Trace};

use crate::chart::{self, fmt_bw, fmt_bytes, fmt_energy, fmt_pct, fmt_time};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{ColorMap, Theme, cat};
use crate::views::floorplan::{self, color_label, denorm, fmt_area, fmt_value, norm};
use crate::views::{FloorColor, Selection, ViewSpec, WireColor};

const SITE: u8 = 4;
/// Containers that frame others without being a layer's silicon (drawn in every layer panel).
const FRAMES: &[&str] = &["system", "host", "board", "package", "switch"];
const CONTAINERS: &[&str] = &[
    "system", "host", "board", "package", "die", "cluster", "switch",
];

pub fn is_placed(t: &Trace) -> bool {
    t.manifest
        .floorplan_source
        .as_deref()
        .is_some_and(|s| s != kiln_trace::layout::UNPLACED)
        && t.floorplan.iter().any(|f| f.source != 0)
}

/// Fixed color per block category (`floorplan.block`), shared by the floorplan legend and the design sheet.
pub fn block_color(name: &str) -> Color {
    match name {
        "compute" => cat(0),
        "sram" => cat(2),
        "noc" => cat(1),
        "phy" => cat(3),
        "hbm" => cat(7),
        "control" => cat(5),
        _ => cat(11),
    }
}

pub const BLOCKS: [&str; 7] = ["compute", "sram", "noc", "phy", "hbm", "control", "misc"];

/// Effective wire coloring: `Auto` is utilization when the run has channel aggregates, else bandwidth.
pub fn wire_mode(t: &Trace, spec: &ViewSpec) -> WireColor {
    match spec.wire_color {
        WireColor::Auto if !t.aggregates_resource.is_empty() => WireColor::Utilization,
        WireColor::Auto => WireColor::Bandwidth,
        w => w,
    }
}

fn wire_label(w: WireColor) -> &'static str {
    match w {
        WireColor::Auto | WireColor::Utilization => "wires: link utilization",
        WireColor::Traffic => "wires: bytes moved",
        WireColor::Bandwidth => "wires: bandwidth",
        WireColor::Length => "wires: length",
        WireColor::Energy => "wires: energy / bit",
        WireColor::Latency => "wires: latency",
    }
}

fn fmt_wire(w: WireColor, v: f64) -> String {
    match w {
        WireColor::Auto | WireColor::Utilization => fmt_pct(v),
        WireColor::Traffic => fmt_bytes(v),
        WireColor::Bandwidth => fmt_bw(v),
        WireColor::Length => format!("{} mm", chart::fmt_num(v / 1000.0)),
        WireColor::Energy => fmt_energy(v),
        WireColor::Latency => fmt_time(v),
    }
}

struct Panel {
    layer: Option<u8>,
    area: Rect,
    k: f32,
    ox: f32,
    oy: f32,
    x0: f64,
    y0: f64,
}

impl Panel {
    fn px(&self, f: &FloorplanRow) -> Rect {
        Rect::new(
            self.ox + ((f.x_um - self.x0) as f32) * self.k,
            self.oy + ((f.y_um - self.y0) as f32) * self.k,
            f.w_um as f32 * self.k,
            f.h_um as f32 * self.k,
        )
    }

    fn pt(&self, x: f64, y: f64) -> [f32; 2] {
        [
            self.ox + ((x - self.x0) as f32) * self.k,
            self.oy + ((y - self.y0) as f32) * self.k,
        ]
    }
}

/// How a resource ended up on screen in one panel.
#[derive(Clone, Copy, PartialEq)]
enum Drawn {
    Open,
    Closed([f32; 2]),
}

struct Agg {
    a: [f32; 2],
    b: [f32; 2],
    value: f64,
    bw: f64,
    members: usize,
    hit: u32,
    links: BTreeSet<u32>,
}

pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let m = floorplan::model(t, spec);
    let n = t.resources.len();
    let kind_of = |i: usize| t.resource_kind(&t.resources[i]);
    let row = |i: usize| m.fp_row[i].map(|k| &t.floorplan[k]);
    let wmode = wire_mode(t, spec);

    // Layers of stacked dies.
    let layers: BTreeSet<u8> = (0..n)
        .filter(|&i| kind_of(i) == "die")
        .filter_map(|i| row(i).map(|f| f.layer))
        .collect();
    let stacked = layers.len() > 1;
    let design = t
        .manifest
        .design_name
        .clone()
        .unwrap_or_else(|| floorplan::short_hash(&t.manifest.provenance.design_hash));
    let layer_txt = match (stacked, spec.layer) {
        (false, _) => String::new(),
        (true, None) => " | layers side by side".into(),
        (true, Some(l)) => format!(" | layer {l} (others outlined)"),
    };
    let run = if t.phases.is_empty() {
        "design only (no run)".to_string()
    } else {
        spec.phase.clone().unwrap_or_else(|| "all phases".into())
    };
    let sub = format!(
        "{} placement, to scale | color: {} | {} | {}{}",
        t.manifest.floorplan_source.as_deref().unwrap_or("kiln-phys"),
        spec.color.name(),
        if spec.wires {
            wire_label(wmode).to_string()
        } else {
            "wires off".into()
        },
        run,
        layer_txt
    );
    chart::title(&mut s, &th, &format!("Floorplan  {design}"), &sub);
    let legend_w = 170.0;
    let canvas = Rect::new(16.0, 60.0, spec.width - 40.0 - legend_w, spec.height - 96.0);

    // Roots and world bounds (site markers excluded).
    let root = spec
        .root
        .as_deref()
        .and_then(|p| t.resource_by_path(p))
        .filter(|&r| m.fp_row[r as usize].is_some());
    let roots: Vec<u32> = match root {
        Some(r) => vec![r],
        None => (0..n as u32)
            .filter(|&i| t.resources[i as usize].parent.is_none() && m.fp_row[i as usize].is_some())
            .collect(),
    };
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for &r in &roots {
        let f = row(r as usize).expect("placed");
        x0 = x0.min(f.x_um);
        y0 = y0.min(f.y_um);
        x1 = x1.max(f.x_um + f.w_um);
        y1 = y1.max(f.y_um + f.h_um);
    }
    let (ww, wh) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let panel_layers: Vec<Option<u8>> = match (stacked, spec.layer) {
        (true, None) => layers.iter().map(|&l| Some(l)).collect(),
        (_, l) => vec![l],
    };
    let np = panel_layers.len() as f32;
    // Side by side along the axis that keeps the larger scale.
    let horiz = {
        let kh = (canvas.w / np / ww as f32).min(canvas.h / wh as f32);
        let kv = (canvas.w / ww as f32).min(canvas.h / np / wh as f32);
        kh >= kv
    };
    let gap = if np > 1.0 { 24.0 } else { 0.0 };
    let panels: Vec<Panel> = panel_layers
        .iter()
        .enumerate()
        .map(|(pi, &layer)| {
            let area = if horiz {
                let w = (canvas.w - gap * (np - 1.0)) / np;
                Rect::new(canvas.x + pi as f32 * (w + gap), canvas.y + 16.0, w, canvas.h - 16.0)
            } else {
                let h = (canvas.h - gap * (np - 1.0)) / np;
                Rect::new(canvas.x, canvas.y + 16.0 + pi as f32 * (h + gap), canvas.w, h - 16.0)
            };
            let k = (area.w / ww as f32).min(area.h / wh as f32);
            Panel {
                layer,
                area,
                k,
                ox: area.x + (area.w - ww as f32 * k) / 2.0,
                oy: area.y + (area.h - wh as f32 * k) / 2.0,
                x0,
                y0,
            }
        })
        .collect();

    let color_of = |i: usize| -> Option<Color> {
        if spec.color == FloorColor::Kind {
            return Some(block_color(
                t.enum_name("floorplan.block", u32::from(row(i)?.block)),
            ));
        }
        m.value[i].map(|v| spec.cmap.at(norm(&m, spec, v)))
    };
    let includes = |p: &Panel, i: usize| -> bool {
        match p.layer {
            None => true,
            Some(l) => FRAMES.contains(&kind_of(i)) || row(i).is_some_and(|f| f.layer == l),
        }
    };

    let mut drawn: Vec<Vec<Option<Drawn>>> = vec![vec![None; n]; panels.len()];
    let mut blocks = 0usize;
    let mut labels: Vec<(Rect, String, Color, bool)> = vec![];
    let mut sites: Vec<(usize, u32)> = vec![];
    for (pi, p) in panels.iter().enumerate() {
        if stacked && let Some(l) = p.layer {
            s.text(
                [p.area.x, p.area.y - 4.0],
                format!("layer {l}"),
                12.0,
                th.fg,
                HAlign::Left,
                VAlign::Bottom,
            );
        }
        let mut stack: Vec<u32> = roots.iter().rev().copied().collect();
        while let Some(r) = stack.pop() {
            let i = r as usize;
            let f = row(i).expect("placed");
            if !includes(p, i) {
                continue;
            }
            if f.source == SITE {
                continue;
            }
            let px = p.px(f);
            if px.w < 0.5 || px.h < 0.5 {
                continue;
            }
            let kind = kind_of(i);
            let container = CONTAINERS.contains(&kind);
            let path = &t.resources[i].path;
            let name = path.rsplit('.').next().unwrap_or(path).to_string();
            let kids: Vec<u32> = m.children[i]
                .iter()
                .copied()
                .filter(|&c| includes(p, c as usize))
                .collect();
            let open = container && px.w.min(px.h) > spec.drill_px && !kids.is_empty();
            if open {
                let (fill, stroke) = match kind {
                    "die" => (th.panel, Stroke::solid(1.5, th.fg.with_alpha(160))),
                    "package" => (th.bg, Stroke::solid(1.25, th.outline)),
                    _ => (th.panel, Stroke::solid(0.75, th.outline)),
                };
                s.rect(px, Some(fill), Some(stroke), Hit::Resource(r));
                drawn[pi][i] = Some(Drawn::Open);
                if spec.labels && px.w > 48.0 && px.h > 24.0 && kind != "cluster" {
                    // Dies and packages: name and outline area above the outline (PHYs line the edges).
                    if matches!(kind, "die" | "package") {
                        let tech = t
                            .package_geometry
                            .iter()
                            .find(|g| g.kind == 0 && g.resource == Some(r))
                            .map(|g| format!(" ({})", g.label))
                            .unwrap_or_default();
                        let lbl = format!("{kind} {name}  {}{tech}", fmt_area(f.w_um * f.h_um));
                        let y = if kind == "die" { px.y - 13.0 } else { px.bottom() - 14.0 };
                        labels.push((Rect::new(px.x + 4.0, y, px.w - 8.0, 11.0), lbl, th.fg, false));
                    } else {
                        labels.push((Rect::new(px.x + 4.0, px.y + 3.0, px.w - 8.0, 11.0), name, th.muted, false));
                    }
                }
                // Children: the stack pops the largest first, so small blocks stay on top.
                let mut kids = kids;
                kids.sort_by(|a, b| {
                    let (fa, fb) = (row(*a as usize).expect("kid"), row(*b as usize).expect("kid"));
                    (fa.w_um * fa.h_um).total_cmp(&(fb.w_um * fb.h_um)).then(b.cmp(a))
                });
                stack.extend(kids);
            } else {
                blocks += 1;
                drawn[pi][i] = Some(Drawn::Closed([px.x + px.w / 2.0, px.y + px.h / 2.0]));
                match color_of(i) {
                    Some(c) => {
                        let edge = if kind == "die" {
                            Stroke::solid(1.5, th.fg.with_alpha(160))
                        } else {
                            Stroke::solid(0.5, th.bg)
                        };
                        s.rect(px, Some(c), Some(edge), Hit::Resource(r));
                        let idle = spec.color == FloorColor::Utilization
                            && m.value[i].is_some_and(|v| v <= 1e-6);
                        if idle && px.w > 4.0 && px.h > 4.0 {
                            s.hatch(px, 5.0, Stroke::solid(0.75, th.hatch));
                        }
                        if spec.labels && px.w > 36.0 && px.h > 13.0 {
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
                        if spec.labels && px.w > 36.0 && px.h > 13.0 {
                            labels.push((px, name, th.muted, true));
                        }
                    }
                }
            }
            if sel.has(Hit::Resource(r)) {
                s.rect(px, None, Some(Stroke::solid(2.5, th.accent)), Hit::None);
            }
        }
        // Other layers' dies as outlines.
        if let Some(l) = p.layer {
            for i in 0..n {
                if kind_of(i) == "die"
                    && let Some(f) = row(i)
                    && f.layer != l
                    && roots.iter().any(|&r| is_under(t, i, r as usize))
                {
                    s.rect(
                        p.px(f),
                        None,
                        Some(Stroke::dashed(1.0, th.muted, 4.0)),
                        Hit::None,
                    );
                }
            }
        }
        // Shoreline marks and harvested parts.
        for g in &t.package_geometry {
            let kind = t.enum_name("package_geometry.kind", u32::from(g.kind));
            // Harvested rows of older traces carry only their path: the nearest ancestor resource owns them.
            let owner = g.resource.or_else(|| {
                std::iter::successors(Some(g.label.as_str()), |p| p.rsplit_once('.').map(|x| x.0))
                    .find_map(|p| t.resource_by_path(p))
            });
            let in_view = match owner {
                Some(r) => drawn[pi].get(r as usize).is_some_and(Option::is_some),
                None => root.is_none(),
            };
            if !in_view || p.layer.is_some_and(|l| l != g.layer) {
                continue;
            }
            let gr = p.px(&FloorplanRow {
                resource: 0,
                die: 0,
                layer: 0,
                x_um: g.x_um,
                y_um: g.y_um,
                w_um: g.w_um,
                h_um: g.h_um,
                poly: None,
                rotation: 0,
                source: 0,
                block: 0,
                area_um2: None,
                leak_w: None,
            });
            match kind {
                "shoreline" => {
                    let hbm = g.value2.unwrap_or(0.0) > 0.0;
                    let c = if hbm { block_color("hbm") } else { block_color("phy") };
                    let t2 = gr.w.min(gr.h).max(2.0);
                    let r = if gr.w >= gr.h {
                        Rect::new(gr.x, gr.y + (gr.h - t2) / 2.0, gr.w, t2)
                    } else {
                        Rect::new(gr.x + (gr.w - t2) / 2.0, gr.y, t2, gr.h)
                    };
                    s.rect(r, Some(c.with_alpha(150)), None, Hit::None);
                }
                "harvested" => {
                    if gr.w > 2.0 && gr.h > 2.0 {
                        s.rect(gr, Some(th.idle), Some(Stroke::dashed(1.0, th.outline, 3.0)), Hit::None);
                        s.hatch(gr, 6.0, Stroke::solid(0.75, th.hatch));
                        if gr.w > 50.0 && gr.h > 14.0 {
                            labels.push((gr, "harvested".into(), th.muted, true));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    // Router sites: markers where the nearest placed ancestor is open (routers hang off row-less networks).
    for (pi, p) in panels.iter().enumerate() {
        for f in t.floorplan.iter().filter(|f| f.source == SITE) {
            if includes(p, f.resource as usize) {
                sites.push((pi, f.resource));
            }
        }
    }
    for &(pi, r) in &sites {
        let i = r as usize;
        let parent_open = std::iter::successors(t.resources[i].parent, |&q| t.resources[q as usize].parent)
            .find(|&q| row(q as usize).is_some())
            .is_some_and(|q| drawn[pi][q as usize] == Some(Drawn::Open));
        if !parent_open {
            continue;
        }
        let f = row(i).expect("site");
        drawn[pi][i] = Some(Drawn::Closed(
            panels[pi].pt(f.x_um + f.w_um / 2.0, f.y_um + f.h_um / 2.0),
        ));
    }

    // Wires: each channel's endpoints resolve to the block drawn for them; links between the same two drawn
    // blocks merge (bandwidth and traffic sum, utilization, length, energy and latency take the maximum).
    let mut wire_count = (0usize, 0usize);
    let mut wire_range = (f64::INFINITY, 0.0f64);
    let mut via_marks: Vec<[f32; 2]> = vec![];
    if spec.wires && !t.wires.is_empty() {
        let endpoint = |pi: usize, r: u32| -> Option<(usize, [f32; 2])> {
            let mut x = r as usize;
            let own = std::iter::successors(Some(x), |&q| t.resources[q].parent.map(|p| p as usize))
                .find(|&q| row(q).is_some())?;
            if !includes(&panels[pi], own) {
                return None;
            }
            let mut prev: Option<usize> = None;
            loop {
                match drawn[pi][x] {
                    Some(Drawn::Closed(c)) => return Some((x, c)),
                    Some(Drawn::Open) => {
                        let y = prev?;
                        let f = row(y)?;
                        return Some((y, panels[pi].pt(f.x_um + f.w_um / 2.0, f.y_um + f.h_um / 2.0)));
                    }
                    None => {}
                }
                if row(x).is_some() {
                    prev = Some(x);
                }
                x = t.resources[x].parent? as usize;
            }
        };
        let mut agg: BTreeMap<(usize, usize, usize), Agg> = BTreeMap::new();
        let mut vias: BTreeSet<(usize, usize)> = BTreeSet::new();
        for w in &t.wires {
            let st = (w.link != u32::MAX)
                .then(|| m.stats.get(w.link as usize).copied())
                .flatten();
            let value = match wmode {
                WireColor::Auto | WireColor::Utilization => st.and_then(|x| x.util).unwrap_or(0.0),
                WireColor::Traffic => st.map_or(0.0, |x| x.bytes),
                WireColor::Bandwidth => w.bw_bps,
                WireColor::Length => w.length_um,
                WireColor::Energy => w.e_j_per_bit,
                WireColor::Latency => w.latency_s,
            };
            let ends: Vec<Option<(usize, usize, [f32; 2])>> = [w.src, w.dst]
                .iter()
                .map(|&e| {
                    (0..panels.len()).find_map(|pi| endpoint(pi, e).map(|(x, c)| (pi, x, c)))
                })
                .collect();
            let (Some((pa, a, ca)), Some((pb, b, cb))) = (ends[0], ends[1]) else {
                continue;
            };
            wire_count.1 += 1;
            if pa != pb {
                vias.insert((pa, a));
                vias.insert((pb, b));
                continue;
            }
            if a == b {
                continue;
            }
            let key = (pa, a.min(b), a.max(b));
            let (ca, cb) = if a <= b { (ca, cb) } else { (cb, ca) };
            let e = agg.entry(key).or_insert(Agg {
                a: ca,
                b: cb,
                value: 0.0,
                bw: 0.0,
                members: 0,
                hit: w.link,
                links: BTreeSet::new(),
            });
            e.members += 1;
            e.bw += w.bw_bps;
            let sums = matches!(wmode, WireColor::Traffic | WireColor::Bandwidth);
            // Traffic belongs to the link resource, which each of its channel rows reports in full.
            let repeat = wmode == WireColor::Traffic && !e.links.insert(w.link);
            if sums {
                if !repeat {
                    e.value += value;
                }
            } else if value > e.value {
                e.value = value;
                e.hit = w.link;
            }
        }
        for a in agg.values() {
            if a.value > 0.0 {
                wire_range.0 = wire_range.0.min(a.value);
                wire_range.1 = wire_range.1.max(a.value);
            }
        }
        let (lo, hi) = (
            if wire_range.0.is_finite() { wire_range.0 } else { 0.0 },
            wire_range.1,
        );
        let log = lo > 0.0 && hi / lo > 100.0;
        let unit_wmode = matches!(wmode, WireColor::Auto | WireColor::Utilization);
        let wnorm = |v: f64| -> f64 {
            if unit_wmode {
                v
            } else if log {
                (v.max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln())
            } else if hi > lo {
                (v - lo) / (hi - lo)
            } else {
                1.0
            }
        };
        let bw_max = agg.values().map(|a| a.bw).fold(0.0, f64::max);
        let wcmap = wire_cmap(spec);
        let mut order: Vec<&Agg> = agg.values().collect();
        order.sort_by(|x, y| x.value.total_cmp(&y.value));
        for a in order {
            let pts = if (a.a[0] - a.b[0]).abs() < 0.5 || (a.a[1] - a.b[1]).abs() < 0.5 {
                vec![a.a, a.b]
            } else {
                vec![a.a, [a.b[0], a.a[1]], a.b]
            };
            let width = if bw_max > 0.0 && a.bw > 0.0 {
                0.75 + 2.25 * ((a.bw / bw_max).ln() / 1e3f64.ln() + 1.0).clamp(0.0, 1.0) as f32
            } else {
                0.75
            };
            let idle = a.value <= 0.0;
            let stroke = if idle {
                Stroke::dashed(0.75, th.muted.with_alpha(140), 3.0)
            } else {
                Stroke::solid(width, wcmap.at(wnorm(a.value)).with_alpha(225))
            };
            let hit = if a.hit == u32::MAX { Hit::None } else { Hit::Resource(a.hit) };
            if hit != Hit::None && sel.has(hit) {
                s.polyline(pts.clone(), Stroke::solid(width + 3.0, th.accent), Hit::None);
            }
            s.polyline(pts, stroke, hit);
            wire_count.0 += 1;
        }
        for (pi, x) in vias {
            if let Some(Drawn::Closed(c)) = drawn[pi][x] {
                via_marks.push(c);
            }
        }
    }
    // Router markers over the wires.
    for &(pi, r) in &sites {
        if kind_of(r as usize) != "router" {
            continue;
        }
        if let Some(Drawn::Closed(c)) = drawn[pi][r as usize] {
            let d = 3.5;
            let col = color_of(r as usize).unwrap_or(block_color("noc"));
            s.polygon(
                vec![[c[0], c[1] - d], [c[0] + d, c[1]], [c[0], c[1] + d], [c[0] - d, c[1]]],
                Some(col),
                Some(Stroke::solid(0.75, th.fg)),
                Hit::Resource(r),
            );
        }
    }
    for (r, name, c, centered) in labels {
        let size = if centered { (r.h * 0.4).clamp(8.0, 11.0) } else { 10.0 };
        if let Some(txt) = text::fit(&name, size, false, r.w - 4.0) {
            if centered {
                s.text([r.x + r.w / 2.0, r.y + r.h / 2.0], txt, size, c, HAlign::Center, VAlign::Middle);
            } else {
                s.text([r.x, r.y], txt, size, c, HAlign::Left, VAlign::Top);
            }
        }
    }
    // Vertical (3D) links: a ring at each end, in its layer's panel.
    for c in &via_marks {
        s.circle(*c, 5.0, None, Some(Stroke::solid(2.0, th.accent)), Hit::None);
        s.circle(*c, 1.5, Some(th.accent), None, Hit::None);
    }
    for p in &panels {
        scale_bar(&mut s, &th, p, ww);
    }

    // Legend.
    let lx = spec.width - legend_w - 8.0;
    let mut ly = 76.0;
    if spec.color == FloorColor::Kind {
        s.text([lx, ly], "block kind", 11.0, th.fg, HAlign::Left, VAlign::Top);
        let present: BTreeSet<u8> = t.floorplan.iter().map(|f| f.block).collect();
        let items: Vec<(Color, String)> = BLOCKS
            .iter()
            .filter(|b| {
                present
                    .iter()
                    .any(|&c| t.enum_name("floorplan.block", u32::from(c)) == **b)
            })
            .map(|b| (block_color(b), (*b).to_string()))
            .collect();
        ly = chart::swatches(&mut s, &th, lx, ly + 18.0, &items) + 8.0;
    } else {
        let cm = spec.cmap;
        let fmt_t = |x: f64| fmt_value(spec, denorm(&m, spec, x));
        chart::gradient_legend(
            &mut s,
            &th,
            Rect::new(lx, ly + 16.0, 14.0, 150.0),
            &|x| cm.at(x),
            color_label(spec.color),
            &fmt_t,
        );
        ly += 180.0;
    }
    if spec.wires && wire_count.0 > 0 {
        let wcmap = wire_cmap(spec);
        let (lo, hi) = (
            if wire_range.0.is_finite() { wire_range.0 } else { 0.0 },
            wire_range.1,
        );
        let unit = matches!(wmode, WireColor::Auto | WireColor::Utilization);
        let log = !unit && lo > 0.0 && hi / lo > 100.0;
        let fmt_t = |x: f64| {
            let v = if unit {
                x
            } else if log {
                (lo.ln() + x * (hi.ln() - lo.ln())).exp()
            } else {
                lo + x * (hi - lo)
            };
            fmt_wire(wmode, v)
        };
        chart::gradient_legend(
            &mut s,
            &th,
            Rect::new(lx, ly + 18.0, 14.0, 110.0),
            &|x| wcmap.at(x),
            wire_label(wmode),
            &fmt_t,
        );
        ly += 140.0;
        s.text([lx, ly], "width ~ log bandwidth", 10.0, th.muted, HAlign::Left, VAlign::Top);
        ly += 14.0;
        if unit {
            s.seg([lx, ly + 6.0], [lx + 14.0, ly + 6.0], Stroke::dashed(0.75, th.muted, 3.0));
            s.text([lx + 20.0, ly + 6.0], "idle link", 10.0, th.muted, HAlign::Left, VAlign::Middle);
            ly += 16.0;
        }
    }
    ly += 6.0;
    if !via_marks.is_empty() {
        s.circle([lx + 7.0, ly + 6.0], 5.0, None, Some(Stroke::solid(2.0, th.accent)), Hit::None);
        s.text([lx + 20.0, ly + 6.0], "vertical link end (3D)", 10.0, th.fg, HAlign::Left, VAlign::Middle);
        ly += 16.0;
    }
    let marks: [(&str, Color); 3] = [
        ("HBM shoreline", block_color("hbm").with_alpha(150)),
        ("other PHY shoreline", block_color("phy").with_alpha(150)),
        ("router (site)", block_color("noc")),
    ];
    for (lbl, c) in marks {
        s.rect(Rect::new(lx, ly + 4.0, 14.0, 5.0), Some(c), None, Hit::None);
        s.text([lx + 20.0, ly + 6.0], lbl, 10.0, th.fg, HAlign::Left, VAlign::Middle);
        ly += 16.0;
    }
    if spec.color == FloorColor::Utilization {
        let r = Rect::new(lx, ly, 14.0, 12.0);
        s.rect(r, Some(ColorMap::Cividis.at(0.0)), None, Hit::None);
        s.hatch(r, 4.0, Stroke::solid(0.75, th.hatch));
        s.text([lx + 20.0, ly + 6.0], "idle (0%)", 10.0, th.fg, HAlign::Left, VAlign::Middle);
        ly += 16.0;
    }
    if spec.color != FloorColor::Kind {
        s.rect(Rect::new(lx, ly, 14.0, 12.0), Some(th.idle), Some(Stroke::dashed(0.75, th.outline, 2.0)), Hit::None);
        s.text([lx + 20.0, ly + 6.0], "no data", 10.0, th.fg, HAlign::Left, VAlign::Middle);
    }

    // Footer: counts and what kiln-phys left unplaced.
    let unplaced: Vec<String> = t
        .scalar("design_summary")
        .and_then(|v| v.get("unplaced").cloned())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .take(3)
        .filter_map(|u| {
            Some(format!(
                "{} {:.1} mm^2",
                u.get("path")?.as_str()?.rsplit('.').next()?,
                u.get("area_mm2")?.as_f64()?
            ))
        })
        .collect();
    let mut foot = format!(
        "{blocks} blocks drawn of {} placed | {} wires drawn ({} links)",
        t.floorplan.len(),
        wire_count.0,
        wire_count.1
    );
    if !unplaced.is_empty() {
        foot.push_str(&format!(" | not placed (no channels): {}", unplaced.join(", ")));
    }
    if let Some(txt) = text::fit(&foot, 10.0, false, spec.width - 32.0) {
        s.text([16.0, spec.height - 8.0], txt, 10.0, th.muted, HAlign::Left, VAlign::Bottom);
    }
    s
}

fn wire_cmap(spec: &ViewSpec) -> ColorMap {
    match spec.cmap {
        ColorMap::Cividis => ColorMap::Viridis,
        ColorMap::Viridis => ColorMap::Cividis,
    }
}

fn is_under(t: &Trace, mut i: usize, root: usize) -> bool {
    loop {
        if i == root {
            return true;
        }
        match t.resources[i].parent {
            Some(p) => i = p as usize,
            None => return false,
        }
    }
}

/// Scale bar with a round length near a fifth of the panel's world width.
fn scale_bar(s: &mut Scene, th: &Theme, p: &Panel, world_w: f64) {
    let target = world_w / 5.0;
    let mag = 10f64.powf(target.log10().floor());
    let len = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .rfind(|l| *l <= target)
        .unwrap_or(mag);
    let px = (len as f32) * p.k;
    let (x, y) = (p.area.x, p.area.bottom() + 12.0);
    let st = Stroke::solid(1.25, th.fg);
    s.seg([x, y], [x + px, y], st);
    s.seg([x, y - 4.0], [x, y + 4.0], st);
    s.seg([x + px, y - 4.0], [x + px, y + 4.0], st);
    let label = if len >= 1000.0 {
        format!("{} mm", chart::fmt_num(len / 1000.0))
    } else {
        format!("{} um", chart::fmt_num(len))
    };
    s.text([x + px + 6.0, y], label, 10.0, th.fg, HAlign::Left, VAlign::Middle);
}
