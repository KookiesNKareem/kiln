//! Headless view rendering (05 §2.3, §7): view models -> [`Scene`] -> PNG (tiny-skia) or SVG. No winit or
//! wgpu, so `kiln-py` and evolution workers render images without a display or GPU.

pub mod chart;
pub mod raster;
pub mod scene;
pub mod svg;
pub mod text;
pub mod theme;
pub mod views;

use kiln_trace::analysis::{phase_code, resource_stats};
use kiln_trace::archive::Archive;
use kiln_trace::calib_report::CalibRow;
use kiln_trace::trace::Trace;

pub use scene::{Hit, Scene};
pub use views::{FloorColor, Selection, ViewKind, ViewSpec};

use crate::chart::fmt_pct;
use crate::scene::{HAlign, Rect, Stroke, VAlign};

/// Data a view can draw from; views whose data is missing render a banner (05 §3.3).
#[derive(Default)]
pub struct Inputs<'a> {
    pub runs: Vec<&'a Trace>,
    pub archive: Option<&'a Archive>,
    pub calib: Option<&'a [CalibRow]>,
}

pub fn render(inp: &Inputs, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let missing = |what: &str, hint: &str| {
        let mut s = Scene::new(spec.width, spec.height, th.bg);
        chart::title(&mut s, &th, spec.view.title(), "");
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            what,
            hint,
        );
        s
    };
    let run = inp.runs.first().copied();
    match spec.view {
        ViewKind::Floorplan => run.map_or_else(
            || {
                missing(
                    "floorplan needs a run",
                    "kiln viz render run.kiln --view floorplan",
                )
            },
            |t| views::floorplan::scene(t, spec, sel),
        ),
        ViewKind::Roofline => run.map_or_else(
            || {
                missing(
                    "roofline needs a run",
                    "kiln viz render run.kiln --view roofline",
                )
            },
            |t| {
                let cal: Option<Vec<CalibRow>> = inp.calib.map(|c| {
                    c.iter()
                        .filter(|r| spec.device.as_deref().is_none_or(|d| d == r.device))
                        .cloned()
                        .collect()
                });
                views::roofline::scene(t, spec, sel, cal.as_deref())
            },
        ),
        ViewKind::Bottleneck => run.map_or_else(
            || missing("bottleneck needs a run", ""),
            |t| views::bottleneck::scene(t, spec, sel),
        ),
        ViewKind::Timeline => run.map_or_else(
            || missing("timeline needs a run", ""),
            |t| views::timeline::scene(t, spec, sel),
        ),
        ViewKind::Noc => run.map_or_else(
            || missing("NoC view needs a run", ""),
            |t| noc(t, spec, sel),
        ),
        ViewKind::Compare => views::compare::scene(&inp.runs, spec, sel),
        ViewKind::Archive => inp.archive.map_or_else(
            || missing("no archive loaded", "kiln viz --archive <dir>"),
            |a| views::archive::scene(a, spec, sel),
        ),
        ViewKind::Calibration => views::calibration::scene(inp.calib.unwrap_or(&[]), spec, sel),
    }
}

/// NoC view, Tier A form: links ranked by whole-phase utilization. The time-resolved heatmap needs the
/// link-time matrix of a Tier B trace (05 §6.2).
pub fn noc(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let design = t.manifest.design_name.clone().unwrap_or_default();
    chart::title(
        &mut s,
        &th,
        &format!("NoC / links  {design}"),
        "Tier A: utilization per link over the phase (no time axis); the link x time heatmap needs a Tier B trace (`full` level, M4)",
    );
    let stats = resource_stats(t, phase_code(t, spec.phase.as_deref()));
    let channel =
        kiln_trace::trace::code(&t.manifest.enums, "resources.kind", "channel").map(|c| c as u16);
    let mut links: Vec<(u32, f64)> = t
        .resources
        .iter()
        .enumerate()
        .filter(|(_, r)| Some(r.kind) == channel)
        .filter_map(|(i, _)| Some((i as u32, stats[i].util?)))
        .collect();
    links.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if links.is_empty() {
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "no link utilization in this trace",
            "",
        );
        return s;
    }
    let label_w = (spec.width * 0.45).min(620.0);
    let bar_w = spec.width - label_w - 120.0;
    let rows = ((spec.height - 90.0) / 16.0) as usize;
    for (k, (r, u)) in links.iter().take(rows).enumerate() {
        let y = 70.0 + k as f32 * 16.0;
        let c = spec.cmap.at(*u);
        if let Some(l) = text::fit(&t.resources[*r as usize].path, 10.0, true, label_w - 24.0) {
            s.text_full(
                [16.0 + label_w - 8.0, y + 6.0],
                l,
                10.0,
                th.fg,
                HAlign::Right,
                VAlign::Middle,
                true,
                false,
            );
        }
        s.rect(
            Rect::new(16.0 + label_w, y, (*u as f32 * bar_w).max(1.0), 12.0),
            Some(c),
            if sel.has(Hit::Resource(*r)) {
                Some(Stroke::solid(2.0, th.accent))
            } else {
                None
            },
            Hit::Resource(*r),
        );
        s.text(
            [16.0 + label_w + (*u as f32 * bar_w) + 6.0, y + 6.0],
            fmt_pct(*u),
            10.0,
            th.muted,
            HAlign::Left,
            VAlign::Middle,
        );
    }
    if links.len() > rows {
        s.text(
            [16.0, spec.height - 8.0],
            format!("+{} more links", links.len() - rows),
            10.0,
            th.muted,
            HAlign::Left,
            VAlign::Bottom,
        );
    }
    s
}

pub fn to_png(s: &Scene, scale: f32) -> Vec<u8> {
    raster::to_png(s, scale)
}

pub fn to_svg(s: &Scene) -> String {
    svg::to_svg(s)
}
