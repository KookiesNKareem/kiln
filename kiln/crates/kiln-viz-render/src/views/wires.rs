//! Wires / links (04 §7.2 as 05 §3.4 `wires`): every channel's kiln-phys link cost. A length vs energy-per-bit
//! scatter (one point per distinct link, colored by channel kind) and a table of link classes: links of one
//! kind, source, width and bandwidth whose endpoints differ only in instance indices, with their length,
//! latency and energy ranges (and peak utilization in a run). Per-link rows: `kiln trace export --csv wires`.

use std::collections::{BTreeMap, BTreeSet};

use kiln_trace::analysis::{phase_code, resource_stats};
use kiln_trace::trace::{Trace, WireRow};

use crate::chart::{self, Axis, fmt_bw, fmt_num, fmt_pct};
use crate::scene::{HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::cat;
use crate::views::{Selection, ViewSpec};

/// `a.b3.c0_1` -> `a.b#.c#_#`: instance indices dropped, so links of one array share a template.
fn template(p: &str) -> String {
    let mut out = String::with_capacity(p.len());
    let mut digits = false;
    for ch in p.chars() {
        if ch.is_ascii_digit() {
            if !digits {
                out.push('#');
            }
            digits = true;
        } else {
            out.push(ch);
            digits = false;
        }
    }
    out
}

fn tail(p: &str, n: usize) -> &str {
    let mut idx = p.len();
    for _ in 0..n {
        match p[..idx].rfind('.') {
            Some(i) => idx = i,
            None => return p,
        }
    }
    &p[idx + 1..]
}

fn fmt_len(um: f64) -> String {
    if um >= 1000.0 {
        format!("{} mm", fmt_num(um / 1000.0))
    } else {
        format!("{} um", fmt_num(um))
    }
}

fn range(lo: f64, hi: f64, f: &dyn Fn(f64) -> String) -> String {
    if (hi - lo).abs() <= 1e-9 * hi.abs().max(1e-30) {
        f(hi)
    } else {
        format!("{}-{}", f(lo), f(hi))
    }
}

struct Class<'a> {
    first: &'a WireRow,
    n: usize,
    len: (f64, f64),
    lat: (f64, f64),
    e: (f64, f64),
    stages: u32,
    util: Option<f64>,
}

pub fn scene(t: &Trace, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let design = t.manifest.design_name.clone().unwrap_or_default();
    if t.wires.is_empty() {
        chart::title(&mut s, &th, &format!("Wires  {design}"), "");
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "no wires in this trace",
            "wires come from kiln-phys placement: kiln viz render design.json5 --view wires (or a new kiln eval -o run.kiln)",
        );
        return s;
    }
    let stats = resource_stats(t, phase_code(t, spec.phase.as_deref()));
    let has_run = !t.aggregates_resource.is_empty();
    let path = |r: u32| t.resources.get(r as usize).map_or("?", |x| x.path.as_str());
    let kind_name = |w: &WireRow| t.enum_name("wires.kind", u32::from(w.kind)).to_string();
    let kinds: Vec<String> = t
        .wires
        .iter()
        .map(kind_name)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let kcolor = |w: &WireRow| cat(kinds.iter().position(|k| *k == kind_name(w)).unwrap_or(0));
    let total_len: f64 = t.wires.iter().map(|w| w.length_um).sum();
    chart::title(
        &mut s,
        &th,
        &format!("Wires  {design}"),
        &format!(
            "{} links (kiln-phys link costs, 04 §7.2) | total routed length {} | per-link rows: kiln trace export <run.kiln|design> --csv wires",
            t.wires.len(),
            fmt_len(total_len)
        ),
    );

    // Scatter: length vs energy per bit, one point per distinct (kind, length, energy).
    let plot_w = (spec.width * 0.36).clamp(260.0, 560.0);
    let plot = Rect::new(84.0, 76.0, plot_w - 84.0, (spec.height * 0.5).clamp(200.0, 420.0));
    let pts: Vec<&WireRow> = t.wires.iter().filter(|w| w.e_j_per_bit > 0.0).collect();
    let (lmin, lmax) = pts.iter().fold((f64::INFINITY, 0.0f64), |a, w| {
        (a.0.min(w.length_um.max(1.0)), a.1.max(w.length_um.max(1.0)))
    });
    let (emin, emax) = pts
        .iter()
        .fold((f64::INFINITY, 0.0f64), |a, w| (a.0.min(w.e_j_per_bit), a.1.max(w.e_j_per_bit)));
    if !pts.is_empty() {
        let x = Axis::log_fit(lmin, lmax, plot.x, plot.right());
        let y = Axis::log_fit(emin * 1e12, emax * 1e12, plot.bottom(), plot.y);
        chart::axes(
            &mut s,
            &th,
            plot,
            &x,
            &y,
            "length (um, 0 drawn at 1 um)",
            "energy (pJ/bit)",
            &|v| fmt_num(v),
            &|v| fmt_num(v),
        );
        let mut seen: BTreeSet<(u8, u64, u64)> = BTreeSet::new();
        for w in &pts {
            let key = (w.kind, w.length_um.max(1.0).to_bits(), w.e_j_per_bit.to_bits());
            if !seen.insert(key) {
                continue;
            }
            let c = [x.map(w.length_um.max(1.0)), y.map(w.e_j_per_bit * 1e12)];
            let hit = if w.link == u32::MAX { Hit::None } else { Hit::Resource(w.link) };
            let stroke = if hit != Hit::None && sel.has(hit) {
                Some(Stroke::solid(2.0, th.accent))
            } else {
                None
            };
            s.circle(c, 3.0, Some(kcolor(w).with_alpha(200)), stroke, hit);
        }
        let items: Vec<(crate::scene::Color, String)> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let n = t.wires.iter().filter(|w| kind_name(w) == *k).count();
                (cat(i), format!("{} ({n})", k.replace('_', " ")))
            })
            .collect();
        s.text([plot.x, plot.bottom() + 46.0], "channel kind", 11.0, th.fg, HAlign::Left, VAlign::Top);
        chart::swatches(&mut s, &th, plot.x, plot.bottom() + 64.0, &items);
    }

    // Link classes.
    let mut classes: BTreeMap<(u8, u8, u64, u64, String), Class> = BTreeMap::new();
    for w in &t.wires {
        let key = (
            w.kind,
            w.source,
            w.width_bits.unwrap_or(0.0).to_bits(),
            w.bw_bps.to_bits(),
            format!("{}>{}", template(path(w.src)), template(path(w.dst))),
        );
        let util = (w.link != u32::MAX)
            .then(|| stats.get(w.link as usize).and_then(|x| x.util))
            .flatten();
        let c = classes.entry(key).or_insert(Class {
            first: w,
            n: 0,
            len: (f64::INFINITY, 0.0),
            lat: (f64::INFINITY, 0.0),
            e: (f64::INFINITY, 0.0),
            stages: 0,
            util: None,
        });
        c.n += 1;
        c.len = (c.len.0.min(w.length_um), c.len.1.max(w.length_um));
        c.lat = (c.lat.0.min(w.latency_s), c.lat.1.max(w.latency_s));
        c.e = (c.e.0.min(w.e_j_per_bit), c.e.1.max(w.e_j_per_bit));
        c.stages = c.stages.max(w.pipeline_stages);
        if let Some(u) = util {
            c.util = Some(c.util.map_or(u, |x: f64| x.max(u)));
        }
    }
    let mut rows: Vec<&Class> = classes.values().collect();
    rows.sort_by(|a, b| {
        b.len
            .1
            .total_cmp(&a.len.1)
            .then(b.n.cmp(&a.n))
            .then(path(a.first.src).cmp(path(b.first.src)))
    });
    let tx = plot_w + 28.0;
    let tw = spec.width - tx - 16.0;
    let mut cols: Vec<(&str, f32)> = vec![
        ("link class (first link of n)", 0.31),
        ("kind / source", 0.12),
        ("n", 0.05),
        ("length", 0.11),
        ("width", 0.06),
        ("bandwidth", 0.09),
        ("latency", 0.09),
        ("pJ/bit", 0.1),
        ("stages", 0.05),
    ];
    if has_run {
        cols.iter_mut().for_each(|c| c.1 *= 0.93);
        cols.push(("peak util", 0.07));
    }
    let xs: Vec<f32> = cols
        .iter()
        .scan(0.0, |a, c| {
            *a += c.1;
            Some(tx + *a * tw)
        })
        .collect();
    let left = |ci: usize| if ci == 0 { tx } else { xs[ci - 1] };
    let cell = |s: &mut Scene, ci: usize, y: f32, v: &str, c: crate::scene::Color| {
        if let Some(txt) = text::fit(v, 10.0, false, xs[ci] - left(ci) - 6.0) {
            if ci == 0 {
                s.text([left(ci), y], txt, 10.0, c, HAlign::Left, VAlign::Top);
            } else {
                s.text([xs[ci], y], txt, 10.0, c, HAlign::Right, VAlign::Top);
            }
        }
    };
    let mut y = 70.0;
    for (ci, (h, _)) in cols.iter().enumerate() {
        cell(&mut s, ci, y, h, th.muted);
    }
    y += 15.0;
    s.seg([tx, y - 2.0], [tx + tw, y - 2.0], Stroke::solid(1.0, th.grid));
    let max_rows = ((spec.height - y - 28.0) / 14.0).max(0.0) as usize;
    for c in rows.iter().take(max_rows) {
        let w = c.first;
        let name = format!("{} -> {}", tail(path(w.src), 2), tail(path(w.dst), 2));
        let hit = if w.link == u32::MAX { Hit::None } else { Hit::Resource(w.link) };
        if hit != Hit::None && sel.has(hit) {
            s.rect(Rect::new(tx - 2.0, y - 1.0, tw + 4.0, 14.0), Some(th.accent.with_alpha(40)), None, Hit::None);
        }
        s.rect(Rect::new(tx - 2.0, y - 1.0, tw + 4.0, 14.0), None, None, hit);
        s.rect(Rect::new(tx - 10.0, y + 2.0, 6.0, 8.0), Some(kcolor(w)), None, Hit::None);
        let vals = [
            name,
            format!(
                "{} / {}",
                kind_name(w).replace('_', " "),
                t.enum_name("wires.source", u32::from(w.source)).replace('_', " ")
            ),
            c.n.to_string(),
            range(c.len.0, c.len.1, &fmt_len),
            w.width_bits.map_or_else(|| "-".into(), |b| format!("{b:.0} b")),
            fmt_bw(w.bw_bps),
            range(c.lat.0, c.lat.1, &|v| format!("{} ns", fmt_num(v * 1e9))),
            range(c.e.0 * 1e12, c.e.1 * 1e12, &fmt_num),
            c.stages.to_string(),
        ];
        for (ci, v) in vals.iter().enumerate() {
            cell(&mut s, ci, y, v, th.fg);
        }
        if has_run {
            cell(&mut s, cols.len() - 1, y, &c.util.map_or_else(|| "-".into(), fmt_pct), th.fg);
        }
        y += 14.0;
    }
    if rows.len() > max_rows {
        s.text(
            [tx, y + 4.0],
            format!("+{} more link classes ({} classes, {} links)", rows.len() - max_rows, rows.len(), t.wires.len()),
            10.0,
            th.muted,
            HAlign::Left,
            VAlign::Top,
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_and_tails() {
        assert_eq!(template("a.gpc0_1.sm12.l1"), "a.gpc#_#.sm#.l#");
        assert_eq!(tail("a.b.c.d", 2), "c.d");
        assert_eq!(tail("d", 2), "d");
    }
}
