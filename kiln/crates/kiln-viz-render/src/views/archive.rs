//! Evolution view (05 §6.7): MAP-Elites grid over two descriptor axes (best fitness per cell, empty cells
//! hatched, invalid-only cells crossed), score over generations, and the lineage of one elite.

use std::collections::{BTreeMap, BTreeSet};

use kiln_trace::archive::{Archive, DesignRecord, DesignStatus, Direction};

use crate::chart::{self, Axis, fmt_num};
use crate::scene::{HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::cat;
use crate::views::{Selection, ViewSpec};

pub struct Grid {
    pub nx: u32,
    pub ny: u32,
    /// (x bin, y bin) -> index into `latest` of the best elite.
    pub best: BTreeMap<(u32, u32), usize>,
    pub invalid_only: BTreeSet<(u32, u32)>,
    pub lo: f64,
    pub hi: f64,
}

pub fn grid(a: &Archive, latest: &[&DesignRecord], xa: usize, ya: usize) -> Grid {
    let ax = |i: usize| a.meta.axes.get(i);
    let nx = ax(xa).map_or(1, |d| d.n_bins());
    let ny = ax(ya).map_or(1, |d| d.n_bins());
    let better = |f: f64, g: f64| {
        if a.meta.fitness.direction == Direction::Minimize {
            f < g
        } else {
            f > g
        }
    };
    let mut best: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut tried: BTreeSet<(u32, u32)> = BTreeSet::new();
    let cell_of = |d: &DesignRecord| -> (u32, u32) {
        let pick = |i: usize| {
            d.cell
                .get(i)
                .copied()
                .or_else(|| Some(ax(i)?.bin_of(*d.descriptor_values.get(i)?)))
                .unwrap_or(0)
        };
        (pick(xa), pick(ya))
    };
    for (k, d) in latest.iter().enumerate() {
        let c = cell_of(d);
        tried.insert(c);
        if matches!(d.status, DesignStatus::Elite | DesignStatus::Displaced)
            && d.fitness.is_finite()
        {
            match best.get(&c) {
                Some(&j) if !better(d.fitness, latest[j].fitness) => {}
                _ => {
                    best.insert(c, k);
                }
            }
        }
    }
    let invalid_only = tried
        .into_iter()
        .filter(|c| !best.contains_key(c))
        .collect();
    let fs: Vec<f64> = best.values().map(|&k| latest[k].fitness).collect();
    let lo = fs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = fs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Grid {
        nx,
        ny,
        best,
        invalid_only,
        lo: if lo.is_finite() { lo } else { 0.0 },
        hi: if hi.is_finite() { hi } else { 1.0 },
    }
}

/// Ancestors of `id` (inclusive) by generation.
pub fn lineage<'a>(a: &'a Archive, id: &str) -> Vec<&'a DesignRecord> {
    let mut out: Vec<&DesignRecord> = vec![];
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut stack = vec![id.to_string()];
    while let Some(i) = stack.pop() {
        if !seen.insert(i.clone()) {
            continue;
        }
        if let Some(d) = a.design(&i) {
            stack.extend(d.parent_ids.iter().cloned());
            out.push(d);
        }
    }
    out.sort_by(|x, y| {
        x.generation
            .cmp(&y.generation)
            .then(x.design_id.cmp(&y.design_id))
    });
    out
}

pub fn scene(a: &Archive, spec: &ViewSpec, sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let latest = a.latest();
    let xa = spec.x_axis.min(a.meta.axes.len().saturating_sub(1));
    let ya = spec.y_axis.min(a.meta.axes.len().saturating_sub(1));
    let name = |i: usize| {
        a.meta
            .axes
            .get(i)
            .map_or("?".to_string(), |d| format!("{} ({})", d.name, d.unit))
    };
    let elites = latest
        .iter()
        .filter(|d| d.status == DesignStatus::Elite)
        .count();
    chart::title(
        &mut s,
        &th,
        &format!(
            "Evolution  {}",
            a.dir
                .file_name()
                .map_or(String::new(), |f| f.to_string_lossy().into_owned())
        ),
        &format!(
            "{} designs, {} elites, {} generations | fitness {} ({:?}){}",
            latest.len(),
            elites,
            a.generations.len(),
            a.meta.fitness.name,
            a.meta.fitness.direction,
            if a.partial {
                " | last append partial"
            } else {
                ""
            }
        ),
    );
    let gw = (spec.width * 0.55).min(spec.height - 140.0);
    let plot = Rect::new(90.0, 70.0, gw, gw.min(spec.height - 150.0));
    let g = grid(a, &latest, xa, ya);
    let (cw, ch) = (plot.w / g.nx as f32, plot.h / g.ny as f32);
    for ix in 0..g.nx {
        for iy in 0..g.ny {
            let r = Rect::new(
                plot.x + ix as f32 * cw,
                plot.bottom() - (iy + 1) as f32 * ch,
                cw,
                ch,
            );
            match g.best.get(&(ix, iy)) {
                Some(&k) => {
                    let d = latest[k];
                    let t = if g.hi > g.lo {
                        (d.fitness - g.lo) / (g.hi - g.lo)
                    } else {
                        1.0
                    };
                    let c = spec.cmap.at(t);
                    s.rect(
                        r,
                        Some(c),
                        Some(Stroke::solid(0.5, th.bg)),
                        Hit::Cell(ix, iy),
                    );
                    if sel.has(Hit::Cell(ix, iy))
                        || spec.lineage_of.as_deref() == Some(d.design_id.as_str())
                    {
                        s.rect(r, None, Some(Stroke::solid(2.5, th.accent)), Hit::None);
                    }
                    if cw > 34.0 && ch > 14.0 {
                        s.text(
                            [r.x + r.w / 2.0, r.y + r.h / 2.0],
                            format!("{:.2}", d.fitness),
                            9.0,
                            th.on(c),
                            HAlign::Center,
                            VAlign::Middle,
                        );
                    }
                }
                None => {
                    s.rect(
                        r,
                        Some(th.idle),
                        Some(Stroke::solid(0.5, th.bg)),
                        Hit::Cell(ix, iy),
                    );
                    s.hatch(r, 6.0, Stroke::solid(0.5, th.hatch));
                    if g.invalid_only.contains(&(ix, iy)) {
                        s.seg(
                            [r.x + 3.0, r.y + 3.0],
                            [r.right() - 3.0, r.bottom() - 3.0],
                            Stroke::solid(1.25, th.bad),
                        );
                        s.seg(
                            [r.right() - 3.0, r.y + 3.0],
                            [r.x + 3.0, r.bottom() - 3.0],
                            Stroke::solid(1.25, th.bad),
                        );
                    }
                }
            }
        }
    }
    s.rect(plot, None, Some(Stroke::solid(1.0, th.outline)), Hit::None);
    // Bin edge labels.
    let fmt_edge = |v: f64| fmt_num(v);
    if let Some(d) = a.meta.axes.get(xa) {
        let e = d.bin_edges();
        let step = (e.len() / 8).max(1);
        for (i, v) in e.iter().enumerate().step_by(step) {
            s.text(
                [plot.x + i as f32 * cw, plot.bottom() + 6.0],
                fmt_edge(*v),
                10.0,
                th.muted,
                HAlign::Center,
                VAlign::Top,
            );
        }
    }
    if let Some(d) = a.meta.axes.get(ya) {
        let e = d.bin_edges();
        let step = (e.len() / 8).max(1);
        for (i, v) in e.iter().enumerate().step_by(step) {
            s.text(
                [plot.x - 6.0, plot.bottom() - i as f32 * ch],
                fmt_edge(*v),
                10.0,
                th.muted,
                HAlign::Right,
                VAlign::Middle,
            );
        }
    }
    s.text(
        [plot.x + plot.w / 2.0, plot.bottom() + 24.0],
        name(xa),
        12.0,
        th.fg,
        HAlign::Center,
        VAlign::Top,
    );
    s.text_full(
        [plot.x - 62.0, plot.y + plot.h / 2.0],
        name(ya),
        12.0,
        th.fg,
        HAlign::Center,
        VAlign::Bottom,
        false,
        true,
    );
    let cm = spec.cmap;
    let (lo, hi) = (g.lo, g.hi);
    chart::gradient_legend(
        &mut s,
        &th,
        Rect::new(plot.right() + 14.0, plot.y + 20.0, 14.0, 160.0),
        &|t| cm.at(t),
        "fitness",
        &|t| format!("{:.3}", lo + t * (hi - lo)),
    );
    s.text(
        [plot.right() + 14.0, plot.y + 200.0],
        "hatched = empty",
        10.0,
        th.muted,
        HAlign::Left,
        VAlign::Top,
    );
    s.text(
        [plot.right() + 14.0, plot.y + 214.0],
        "X = invalid only",
        10.0,
        th.bad,
        HAlign::Left,
        VAlign::Top,
    );

    // Score over generations.
    let rx = plot.right() + 110.0;
    let rw = spec.width - rx - 24.0;
    let sp = Rect::new(rx + 50.0, 80.0, rw - 50.0, (spec.height - 140.0) * 0.45);
    if a.generations.is_empty() || rw < 200.0 {
        chart::banner(
            &mut s,
            &th,
            Rect::new(rx, 80.0, rw.max(100.0), 120.0),
            "no generations yet",
            "",
        );
    } else {
        let gens = &a.generations;
        let gmax = gens.iter().map(|g| g.generation).max().unwrap_or(1).max(1);
        let ys: Vec<f64> = gens
            .iter()
            .flat_map(|g| {
                [
                    g.best,
                    g.median,
                    g.best_low.unwrap_or(g.best),
                    g.best_high.unwrap_or(g.best),
                ]
            })
            .filter(|v| v.is_finite())
            .collect();
        let (ylo, yhi) = (
            ys.iter().copied().fold(f64::INFINITY, f64::min),
            ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        let x = Axis::new(0.0, f64::from(gmax), false, sp.x, sp.right());
        let y = Axis::new(
            ylo.min(0.0),
            (yhi * 1.1).max(ylo + 1e-9),
            false,
            sp.bottom(),
            sp.y,
        );
        chart::axes(
            &mut s,
            &th,
            sp,
            &x,
            &y,
            "generation",
            "fitness",
            &|v| format!("{v:.0}"),
            &|v| format!("{v:.2}"),
        );
        let band: Vec<[f32; 2]> = gens
            .iter()
            .filter_map(|g| {
                g.best_high
                    .map(|h| [x.map(f64::from(g.generation)), y.map(h)])
            })
            .chain(gens.iter().rev().filter_map(|g| {
                g.best_low
                    .map(|l| [x.map(f64::from(g.generation)), y.map(l)])
            }))
            .collect();
        if band.len() >= 4 {
            s.polygon(band, Some(cat(0).with_alpha(50)), None, Hit::None);
        }
        let series = |f: &dyn Fn(&kiln_trace::archive::GenerationRecord) -> f64| -> Vec<[f32; 2]> {
            gens.iter()
                .filter(|g| f(g).is_finite())
                .map(|g| [x.map(f64::from(g.generation)), y.map(f(g))])
                .collect()
        };
        s.line(series(&|g| g.best), Stroke::solid(2.0, cat(0)));
        s.line(series(&|g| g.median), Stroke::solid(1.5, cat(1)));
        let cmax = gens
            .iter()
            .map(|g| g.coverage)
            .filter(|c| c.is_finite())
            .fold(0.0, f64::max)
            .max(1e-9);
        s.line(
            gens.iter()
                .filter(|g| g.coverage.is_finite())
                .map(|g| {
                    [
                        x.map(f64::from(g.generation)),
                        y.map(y.min + (y.max - y.min) * g.coverage / cmax),
                    ]
                })
                .collect(),
            Stroke::dashed(1.25, cat(2), 4.0),
        );
        let inv: Vec<[f32; 2]> = gens
            .iter()
            .map(|g| {
                [
                    x.map(f64::from(g.generation)),
                    y.map(
                        y.min
                            + (y.max - y.min)
                                * (g.invalid_count as f64 / (g.evaluations.max(1)) as f64),
                    ),
                ]
            })
            .collect();
        s.line(inv, Stroke::dashed(1.0, th.bad, 2.0));
        chart::swatches(
            &mut s,
            &th,
            sp.x + 8.0,
            sp.y + 8.0,
            &[
                (cat(0), "best (band: low/high)".into()),
                (cat(1), "median".into()),
                (cat(2), format!("coverage (max {:.1}%)", 100.0 * cmax)),
                (th.bad, "invalid rate".into()),
            ],
        );
    }

    // Lineage.
    let target = spec.lineage_of.clone().or_else(|| {
        let better = |f: f64, g: f64| {
            if a.meta.fitness.direction == Direction::Minimize {
                f < g
            } else {
                f > g
            }
        };
        latest
            .iter()
            .filter(|d| d.status == DesignStatus::Elite)
            .fold(None::<&&DesignRecord>, |b, d| match b {
                Some(b) if !better(d.fitness, b.fitness) => Some(b),
                _ => Some(d),
            })
            .map(|d| d.design_id.clone())
    });
    let ly = sp.bottom() + 50.0;
    if let Some(id) = target {
        let lin = lineage(a, &id);
        s.text(
            [rx, ly],
            format!("Lineage of {id}"),
            13.0,
            th.fg,
            HAlign::Left,
            VAlign::Top,
        );
        let mut y = ly + 22.0;
        let mut pos: BTreeMap<&str, [f32; 2]> = BTreeMap::new();
        let mut by_gen: BTreeMap<u32, Vec<&DesignRecord>> = BTreeMap::new();
        for d in &lin {
            by_gen.entry(d.generation).or_default().push(d);
        }
        let row_h = ((spec.height - y - 16.0) / by_gen.len().max(1) as f32).clamp(18.0, 34.0);
        for ds in by_gen.values() {
            for (k, d) in ds.iter().enumerate() {
                let p = [rx + 10.0 + k as f32 * 200.0, y + 8.0];
                pos.insert(d.design_id.as_str(), p);
                for par in &d.parent_ids {
                    if let Some(q) = pos.get(par.as_str()) {
                        s.seg(*q, p, Stroke::solid(1.0, th.outline));
                    }
                }
                let fc = if d.status == DesignStatus::Elite {
                    cat(0)
                } else {
                    th.outline
                };
                s.circle(p, 6.0, Some(fc), Some(Stroke::solid(1.0, th.bg)), Hit::None);
                let label = format!(
                    "{} g{} {:.3}  {}",
                    d.design_id, d.generation, d.fitness, d.mutation_summary
                );
                if let Some(l) = text::fit(&label, 10.0, false, spec.width - p[0] - 30.0) {
                    s.text(
                        [p[0] + 10.0, p[1]],
                        l,
                        10.0,
                        th.fg,
                        HAlign::Left,
                        VAlign::Middle,
                    );
                }
            }
            y += row_h;
            if y > spec.height - 12.0 {
                break;
            }
        }
    }
    s
}
