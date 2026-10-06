//! Design sheet (05 §8 `kiln viz <design>`): what the design is without running it, from the trace's
//! `design_summary` (kiln-ir structure + kiln-phys roll-ups): peak ops per precision, memory hierarchy, off-chip
//! memory, interconnect, area per die with its breakdown, a power estimate against the TDP, clocks and V/f,
//! validation under the reference and search profiles, and physical findings.

use serde_json::Value;

use kiln_trace::trace::Trace;

use crate::chart::{self, fmt_bw, fmt_bytes, fmt_num, fmt_power};
use crate::scene::{Color, HAlign, Hit, Rect, Scene, Stroke, VAlign};
use crate::text;
use crate::theme::{Theme, cat};
use crate::views::placed::block_color;
use crate::views::{Selection, ViewSpec};

fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

fn st<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn arr<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn fmt_ops(x: f64, key: &str) -> String {
    let int = key.starts_with("int") || key.starts_with("uint") || key.contains("*int");
    let units: &[(f64, &str)] = if int {
        &[(1e15, "POP/s"), (1e12, "TOP/s"), (1e9, "GOP/s")]
    } else {
        &[(1e15, "PFLOP/s"), (1e12, "TFLOP/s"), (1e9, "GFLOP/s")]
    };
    let (k, u) = units
        .iter()
        .find(|(k, _)| x >= *k * 0.9995)
        .copied()
        .unwrap_or(units[units.len() - 1]);
    format!("{} {u}", fmt_num(x / k))
}

/// Path with instance indices dropped (`a.sm3.lsu` -> `a.sm#.lsu`).
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

fn fmt_hz(x: f64) -> String {
    if x >= 1e9 {
        format!("{} GHz", fmt_num(x / 1e9))
    } else {
        format!("{} MHz", fmt_num(x / 1e6))
    }
}

/// A text column that grows downward; rows clip to its width.
struct Col<'a> {
    s: &'a mut Scene,
    th: &'a Theme,
    x: f32,
    y: f32,
    w: f32,
    bottom: f32,
}

impl Col<'_> {
    fn room(&self, h: f32) -> bool {
        self.y + h <= self.bottom
    }

    fn section(&mut self, title: &str) {
        if !self.room(30.0) {
            return;
        }
        self.y += 8.0;
        self.s
            .text([self.x, self.y], title, 13.0, self.th.fg, HAlign::Left, VAlign::Top);
        self.y += 17.0;
        self.s.seg(
            [self.x, self.y],
            [self.x + self.w, self.y],
            Stroke::solid(1.0, self.th.grid),
        );
        self.y += 4.0;
    }

    fn kv(&mut self, k: &str, v: &str) {
        self.kv_c(k, v, self.th.fg);
    }

    fn kv_c(&mut self, k: &str, v: &str, c: Color) {
        if !self.room(15.0) {
            return;
        }
        let kw = (self.w * 0.42).min(170.0);
        if let Some(t) = text::fit(k, 11.0, false, kw - 6.0) {
            self.s
                .text([self.x, self.y], t, 11.0, self.th.muted, HAlign::Left, VAlign::Top);
        }
        if let Some(t) = text::fit(v, 11.0, false, self.w - kw) {
            self.s
                .text([self.x + kw, self.y], t, 11.0, c, HAlign::Left, VAlign::Top);
        }
        self.y += 15.0;
    }

    /// Table with columns at the given fractions of the width (right-aligned after the first).
    fn table(&mut self, cols: &[(&str, f32)], rows: &[Vec<String>], max_rows: usize) {
        if !self.room(30.0) {
            return;
        }
        let xs: Vec<f32> = cols
            .iter()
            .scan(0.0, |acc, c| {
                *acc += c.1;
                Some(*acc)
            })
            .collect();
        let cell = |s: &mut Scene, th: &Theme, x0: f32, ci: usize, y: f32, v: &str, head: bool| {
            let right = x0 + xs[ci] * self.w;
            let left = x0 + if ci == 0 { 0.0 } else { xs[ci - 1] * self.w };
            let c = if head { th.muted } else { th.fg };
            if let Some(t) = text::fit(v, 10.5, false, right - left - 4.0) {
                if ci == 0 {
                    s.text([left, y], t, 10.5, c, HAlign::Left, VAlign::Top);
                } else {
                    s.text([right, y], t, 10.5, c, HAlign::Right, VAlign::Top);
                }
            }
        };
        for (ci, (h, _)) in cols.iter().enumerate() {
            cell(self.s, self.th, self.x, ci, self.y, h, true);
        }
        self.y += 14.0;
        let shown = rows.len().min(max_rows);
        for r in &rows[..shown] {
            if !self.room(14.0) {
                return;
            }
            for (ci, v) in r.iter().enumerate().take(cols.len()) {
                cell(self.s, self.th, self.x, ci, self.y, v, false);
            }
            self.y += 14.0;
        }
        if rows.len() > shown && self.room(14.0) {
            self.s.text(
                [self.x, self.y],
                format!("+{} more", rows.len() - shown),
                10.0,
                self.th.muted,
                HAlign::Left,
                VAlign::Top,
            );
            self.y += 14.0;
        }
    }

    /// Horizontal stacked bar of `(label, value, color)` scaled to `full`, with an optional marker.
    fn bar(&mut self, parts: &[(String, f64, Color)], full: f64, marker: Option<(f64, &str)>) {
        if !self.room(26.0) || full <= 0.0 {
            return;
        }
        let r = Rect::new(self.x, self.y + 2.0, self.w - 4.0, 14.0);
        let mut x = r.x;
        for (_, v, c) in parts {
            let w = (*v / full) as f32 * r.w;
            if w > 0.0 {
                self.s.rect(Rect::new(x, r.y, w, r.h), Some(*c), None, Hit::None);
                x += w;
            }
        }
        self.s
            .rect(r, None, Some(Stroke::solid(0.75, self.th.outline)), Hit::None);
        if let Some((v, lbl)) = marker {
            let mx = r.x + (v / full) as f32 * r.w;
            self.s
                .seg([mx, r.y - 3.0], [mx, r.bottom() + 3.0], Stroke::solid(2.0, self.th.bad));
            self.s.text(
                [mx, r.bottom() + 2.0],
                lbl,
                9.5,
                self.th.bad,
                HAlign::Center,
                VAlign::Top,
            );
            self.y += 10.0;
        }
        self.y += 22.0;
    }

    /// Legend rows of a bar: swatch, label, value.
    fn bar_legend(&mut self, parts: &[(String, f64, Color)], fmt: &dyn Fn(f64) -> String) {
        let half = (self.w - 4.0) / 2.0;
        for (lbl, v, c) in parts {
            if *v <= 0.0 {
                continue;
            }
            if !self.room(13.0) {
                return;
            }
            self.s
                .rect(Rect::new(self.x, self.y + 2.0, 9.0, 9.0), Some(*c), None, Hit::None);
            if let Some(t) = text::fit(lbl, 10.0, false, half - 16.0) {
                self.s.text(
                    [self.x + 14.0, self.y],
                    t,
                    10.0,
                    self.th.muted,
                    HAlign::Left,
                    VAlign::Top,
                );
            }
            self.s.text(
                [self.x + half + 60.0, self.y],
                fmt(*v),
                10.0,
                self.th.fg,
                HAlign::Right,
                VAlign::Top,
            );
            self.y += 13.0;
        }
    }
}

pub fn scene(t: &Trace, spec: &ViewSpec, _sel: &Selection) -> Scene {
    let th = spec.theme();
    let mut s = Scene::new(spec.width, spec.height, th.bg);
    let Some(d) = t.scalar("design_summary") else {
        chart::title(&mut s, &th, "Design", "");
        chart::banner(
            &mut s,
            &th,
            Rect::new(16.0, 64.0, spec.width - 32.0, spec.height - 80.0),
            "no design sheet in this trace",
            "built by kiln-phys with the design: kiln viz render design.json5 --view design (or kiln eval <design> -o run.kiln)",
        );
        return s;
    };
    let name = t
        .manifest
        .design_name
        .clone()
        .unwrap_or_else(|| st(&d, "name").to_string());
    let hash = st(&d, "design_hash");
    let sub = format!(
        "{} | {} | {} | exec {} | {} chip(s) | {} | calibration {}",
        if st(&d, "family").is_empty() { "no family" } else { st(&d, "family") },
        st(&d, "node"),
        st(&d, "model"),
        st(&d, "exec_model"),
        f(&d, "chips").unwrap_or(0.0),
        hash.get(..16).unwrap_or(hash),
        arr(&d, "calibration")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    );
    chart::title(&mut s, &th, &format!("Design  {name}"), &sub);
    let top = 60.0;
    let bottom = spec.height - 12.0;
    let cw = (spec.width - 32.0 - 2.0 * 24.0) / 3.0;
    let xs = [16.0, 16.0 + cw + 24.0, 16.0 + 2.0 * (cw + 24.0)];

    // Column 1: compute, clocks, validation.
    {
        let mut c = Col { s: &mut s, th: &th, x: xs[0], y: top, w: cw, bottom };
        c.section("Peak compute (dense, all chips)");
        let peaks: Vec<(String, f64)> = d
            .get("peak_ops")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                    .collect()
            })
            .unwrap_or_default();
        let mut dense: Vec<&(String, f64)> = peaks.iter().filter(|p| !p.0.ends_with(":sparse")).collect();
        dense.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (k, v) in dense.iter().take(8) {
            c.kv(k, &fmt_ops(*v, k));
        }
        let sparse: Vec<&(String, f64)> = peaks.iter().filter(|p| p.0.ends_with(":sparse")).collect();
        if let Some(best) = sparse.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
            c.kv(&format!("{} (sparse)", best.0.trim_end_matches(":sparse")), &fmt_ops(best.1, &best.0));
        }
        let elem: Vec<(String, f64)> = d
            .get("elem_ops")
            .and_then(Value::as_object)
            .map(|o| o.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))).collect())
            .unwrap_or_default();
        if !elem.is_empty() {
            c.section("Vector / elementwise");
            let mut e = elem;
            e.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (k, v) in e.iter().take(4) {
                c.kv(k, &fmt_ops(*v, k).replace("FLOP", "op"));
            }
        }
        c.section("Clocks and V/f");
        let rows: Vec<Vec<String>> = arr(&d, "clocks")
            .iter()
            .map(|k| {
                let path = st(k, "path");
                vec![
                    path.rsplit('.').next().unwrap_or(path).to_string(),
                    f(k, "nominal_hz").map(fmt_hz).unwrap_or_default(),
                    f(k, "base_hz").map(fmt_hz).unwrap_or_else(|| "-".into()),
                    match (f(k, "v_min"), f(k, "v_max")) {
                        (Some(a), Some(b)) => format!("{a:.2}-{b:.2} V"),
                        _ => "-".into(),
                    },
                    f(k, "leak_w").map(fmt_power).unwrap_or_default(),
                ]
            })
            .collect();
        c.table(
            &[("domain", 0.32), ("nominal", 0.18), ("base", 0.17), ("V/f range", 0.18), ("leak", 0.15)],
            &rows,
            6,
        );
        c.section("Validation");
        for v in arr(&d, "validation") {
            let (e, w) = (f(v, "errors").unwrap_or(0.0), f(v, "warnings").unwrap_or(0.0));
            let txt = if e > 0.0 {
                format!("{e:.0} error(s), {w:.0} warning(s)")
            } else {
                format!("valid, {w:.0} warning(s)")
            };
            c.kv_c(&format!("{} profile", st(v, "profile")), &txt, if e > 0.0 { th.bad } else { th.good });
            for x in arr(v, "first").iter().take(2) {
                c.kv_c(&format!("  {}", st(x, "code")), st(x, "message"), th.muted);
            }
        }
        if arr(&d, "validation").is_empty() {
            c.kv("profiles", "not checked (trace built without the design document)");
        }
        let probs = arr(&d, "problems");
        c.section("Physical findings (kiln-phys)");
        if probs.is_empty() {
            c.kv_c("E-PHYS", "none", th.good);
        }
        for p in probs.iter().take(4) {
            c.kv_c(st(p, "code"), st(p, "message"), th.bad);
        }
    }

    // Column 2: memory, off-chip, interconnect.
    {
        let mut c = Col { s: &mut s, th: &th, x: xs[1], y: top, w: cw, bottom };
        c.section("Memory hierarchy (per chip)");
        let rows: Vec<Vec<String>> = arr(&d, "memory")
            .iter()
            .map(|l| {
                vec![
                    format!("L{} {}", f(l, "level").unwrap_or(0.0), st(l, "name")),
                    st(l, "kind").replace('_', " "),
                    format!("{:.0}", f(l, "instances").unwrap_or(0.0)),
                    fmt_bytes(f(l, "capacity_b").unwrap_or(0.0)),
                    fmt_bw(f(l, "bandwidth_bps").unwrap_or(0.0)),
                ]
            })
            .collect();
        c.table(
            &[("level", 0.3), ("kind", 0.2), ("count", 0.12), ("capacity", 0.18), ("bandwidth", 0.2)],
            &rows,
            8,
        );
        c.kv("on-chip total", &fmt_bytes(f(&d, "onchip_capacity_b").unwrap_or(0.0)));
        if let Some(o) = d.get("offchip") {
            c.section("Off-chip memory");
            let n = f(o, "stacks").unwrap_or(0.0);
            let harvested = f(o, "harvested").unwrap_or(0.0);
            c.kv(
                "stacks",
                &format!(
                    "{n:.0} x {}{}",
                    st(o, "kind"),
                    if harvested > 0.0 { format!(" (+{harvested:.0} harvested)") } else { String::new() }
                ),
            );
            c.kv("capacity", &fmt_bytes(f(o, "capacity_b").unwrap_or(0.0)));
            c.kv("bandwidth", &fmt_bw(f(o, "bandwidth_bps").unwrap_or(0.0)));
            if let Some(p) = f(o, "per_stack_bps") {
                c.kv("per stack", &fmt_bw(p));
            }
        }
        c.section("Interconnect");
        // Replicated networks (one per SM, tile, ...) summarize as one row with their count.
        let mut nets: Vec<(String, usize, &Value)> = vec![];
        for n in arr(&d, "networks") {
            let key = format!(
                "{}|{}|{}|{}|{}",
                template(st(n, "path")),
                st(n, "topology"),
                f(n, "endpoints").unwrap_or(0.0),
                f(n, "links").unwrap_or(0.0),
                f(n, "link_bw_bps").unwrap_or(0.0)
            );
            match nets.iter_mut().find(|x| x.0 == key) {
                Some(x) => x.1 += 1,
                None => nets.push((key, 1, n)),
            }
        }
        let rows: Vec<Vec<String>> = nets
            .iter()
            .map(|(_, count, n)| {
                let path = st(n, "path");
                let name = path.rsplit('.').next().unwrap_or(path);
                vec![
                    if *count > 1 { format!("{name} x{count}") } else { name.to_string() },
                    st(n, "topology").to_string(),
                    format!("{:.0}", f(n, "endpoints").unwrap_or(0.0)),
                    format!("{:.0}", f(n, "routers").unwrap_or(0.0)),
                    format!("{:.0}", f(n, "links").unwrap_or(0.0)),
                    f(n, "link_bw_bps").map(fmt_bw).unwrap_or_default(),
                ]
            })
            .collect();
        c.table(
            &[("network", 0.24), ("topology", 0.18), ("ends", 0.11), ("routers", 0.14), ("links", 0.12), ("link bw", 0.21)],
            &rows,
            8,
        );
        for n in arr(&d, "inter_chip") {
            c.kv(
                &format!("inter-chip {}", st(n, "path").rsplit('.').next().unwrap_or("")),
                &format!(
                    "{} x{:.0}, {}",
                    st(n, "topology"),
                    f(n, "endpoints").unwrap_or(0.0),
                    f(n, "link_bw_bps").map(fmt_bw).unwrap_or_else(|| "-".into())
                ),
            );
        }
        if let Some(o) = d.get("channels").and_then(Value::as_object) {
            let txt: Vec<String> = o.iter().map(|(k, v)| format!("{k} {}", v.as_u64().unwrap_or(0))).collect();
            c.kv("channels", &txt.join(", "));
        }
    }

    // Column 3: area and power.
    {
        let mut c = Col { s: &mut s, th: &th, x: xs[2], y: top, w: cw, bottom };
        let area = d.get("area").cloned().unwrap_or(Value::Null);
        c.section("Area");
        let (pkg_lo, pkg_hi) = (
            f(&area, "package_low_mm2").unwrap_or(0.0),
            f(&area, "package_high_mm2").unwrap_or(0.0),
        );
        c.kv(
            "package",
            &format!(
                "{} mm^2 [{}, {}] ({})",
                fmt_num(f(&area, "package_mm2").unwrap_or(0.0)),
                fmt_num(pkg_lo.min(pkg_hi)),
                fmt_num(pkg_lo.max(pkg_hi)),
                st(&area, "package_table")
            ),
        );
        let dies = arr(&area, "dies");
        let full = dies.iter().filter_map(|x| f(x, "area_mm2")).fold(0.0, f64::max);
        // Identical dies (chiplets) summarize once.
        let mut shown: Vec<(String, usize, &Value)> = vec![];
        for die in dies {
            let key = format!(
                "{:.3}|{}|{}",
                f(die, "area_mm2").unwrap_or(0.0),
                st(die, "node"),
                f(die, "layer").unwrap_or(0.0)
            );
            match shown.iter_mut().find(|x| x.0 == key) {
                Some(x) => x.1 += 1,
                None => shown.push((key, 1, die)),
            }
        }
        let part_color = |p: &str| match p {
            "datapath" => block_color("compute"),
            "sram" => block_color("sram"),
            "rf" => block_color("sram").lerp(th.bg, 0.4),
            "control" => block_color("control"),
            "noc" => block_color("noc"),
            "phy" => block_color("phy"),
            _ => block_color("misc"),
        };
        for (_, count, die) in shown.iter().take(3) {
            let path = st(die, "path");
            c.kv(
                &format!(
                    "{}{} ({}, layer {:.0})",
                    path.rsplit('.').next().unwrap_or(path),
                    if *count > 1 { format!(" x{count}") } else { String::new() },
                    st(die, "node"),
                    f(die, "layer").unwrap_or(0.0)
                ),
                &format!(
                    "{} mm^2 [{}, {}], outline {}",
                    fmt_num(f(die, "area_mm2").unwrap_or(0.0)),
                    fmt_num(f(die, "area_low_mm2").unwrap_or(0.0)),
                    fmt_num(f(die, "area_high_mm2").unwrap_or(0.0)),
                    fmt_num(f(die, "outline_mm2").unwrap_or(0.0))
                ),
            );
            let mut parts: Vec<(String, f64, Color)> = die
                .get("parts_mm2")
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?, part_color(k))))
                        .collect()
                })
                .unwrap_or_default();
            parts.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            parts.push(("whitespace".into(), f(die, "whitespace_mm2").unwrap_or(0.0).max(0.0), th.grid));
            c.bar(&parts, full, None);
            c.bar_legend(&parts, &|v| format!("{} mm^2", fmt_num(v)));
            c.kv(
                "transistors / SRAM",
                &format!(
                    "{} B / {} MiB",
                    fmt_num(f(die, "transistors_b").unwrap_or(0.0)),
                    fmt_num(f(die, "sram_mib").unwrap_or(0.0))
                ),
            );
            let used = f(die, "hbm_shoreline_used_mm").unwrap_or(0.0);
            if used > 0.0 {
                c.kv(
                    "HBM shoreline",
                    &format!(
                        "{} of {} mm{}",
                        fmt_num(used),
                        fmt_num(f(die, "hbm_shoreline_available_mm").unwrap_or(0.0)),
                        if die.get("shoreline_limited").and_then(Value::as_bool) == Some(true) {
                            " (shoreline-limited)"
                        } else {
                            ""
                        }
                    ),
                );
            }
        }
        if shown.len() > 3 {
            c.kv("", &format!("+{} more die types", shown.len() - 3));
        }
        let power = d.get("power").cloned().unwrap_or(Value::Null);
        c.section("Power estimate (nominal clocks)");
        let tdp = f(&power, "tdp_w");
        let assumed = power.get("tdp_assumed_w").and_then(Value::as_array).and_then(|a| a.first()?.as_f64());
        c.kv(
            "TDP",
            &match (tdp, assumed) {
                (Some(w), _) => format!("{} (cap at {} level)", fmt_power(w), st(&power, "cap_level")),
                (None, Some(w)) => format!("{} (assumed, not enforced)", fmt_power(w)),
                _ => "not declared".into(),
            },
        );
        c.kv("static at 85 C", &fmt_power(f(&power, "static_85c_w").unwrap_or(0.0)));
        if let Some(p) = power.get("peak").filter(|p| !p.is_null()) {
            let terms = [
                ("static", "static_w", 11),
                ("clock tree", "clock_w", 9),
                ("control logic", "control_w", 5),
                ("MAC datapath", "mac_w", 0),
                ("DRAM/IO PHY", "phy_w", 3),
                ("DRAM", "dram_w", 7),
                ("board fixed", "board_fixed_w", 8),
                ("VR loss", "vr_loss_w", 4),
            ];
            let parts: Vec<(String, f64, Color)> = terms
                .iter()
                .map(|(l, k, ci)| ((*l).to_string(), f(p, k).unwrap_or(0.0), cat(*ci)))
                .collect();
            let total = f(p, "board_w").unwrap_or(0.0);
            c.kv(
                "all MACs + DRAM at peak",
                &format!(
                    "{} board, {} chip, Tj {:.0} C{}",
                    fmt_power(total),
                    fmt_power(f(p, "chip_w").unwrap_or(0.0)),
                    f(p, "t_j_c").unwrap_or(0.0),
                    if p.get("runaway").and_then(Value::as_bool) == Some(true) { " (thermal runaway)" } else { "" }
                ),
            );
            let cap = tdp.or(assumed);
            let full = total.max(cap.unwrap_or(0.0)) * 1.02;
            c.bar(&parts, full, cap.map(|w| (w, "TDP")));
            c.bar_legend(&parts, &|v| fmt_power(v));
            if let Some(i) = power.get("idle").filter(|p| !p.is_null()) {
                c.kv("idle (no activity)", &fmt_power(f(i, "board_w").unwrap_or(0.0)));
            }
            c.kv_c(
                "note",
                "estimate excludes on-chip SRAM and wire traffic energy; run a workload for phase power",
                th.muted,
            );
        }
        if let Some(tj) = f(&power, "tj_max_c") {
            c.kv("Tj max / q max", &format!("{tj:.0} C / {} W/mm^2", fmt_num(f(&power, "q_avg_max_w_mm2").unwrap_or(0.0))));
        }
    }
    s
}
