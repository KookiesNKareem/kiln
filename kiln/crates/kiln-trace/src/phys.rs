//! Physical geometry and the design sheet from kiln-phys (04): the `floorplan`, `wires` and `package_geometry`
//! tables (05 §3.4) and the `design_summary` run scalar, from the expanded model alone (no simulation).
//!
//! Dies, die-level macros, PHYs on the shoreline, memory stacks and packages are kiln-phys's placement; blocks
//! inside a macro are its hierarchical layout. kiln-phys arranges dies without compute-unit area (04 §6.1, 06
//! P6), so units (and their local buffers) have no rectangle there: they are filled in here, by area, inside
//! their placed parent. Coordinates are um, y down, one frame per package (packages side by side).

use std::collections::BTreeMap;

use kiln_ir::common::{Diagnostic, Severity};
use kiln_ir::hw::HwModel;
use kiln_ir::hw::model::{ChannelKind, ContainerKind, MemSpec, NodeIx};
use kiln_ir::hw::types::BlockKind;
use kiln_phys::characterize::Part;
use kiln_phys::floorplan::{structural_live, subtree_areas, unit_subtrees};
use kiln_phys::place::Rect;
use kiln_phys::power::PhaseEnergy;
use kiln_phys::tables::WireClassId;
use kiln_phys::wire::LinkSource;
use kiln_phys::{ClockMode, Phys};
use serde::Serialize;
use serde_json::{Value, json};

use crate::trace::{FloorplanRow, NONE_U32, PackageRow, WireRow};

/// `floorplan.source` codes (default enum order).
pub mod source {
    pub const UNPLACED: u8 = 0;
    pub const PLACED: u8 = 1;
    pub const LAYOUT: u8 = 2;
    pub const FILLED: u8 = 3;
    pub const SITE: u8 = 4;
}

/// `floorplan.block` codes (default enum order).
pub mod block {
    pub const MISC: u8 = 0;
    pub const COMPUTE: u8 = 1;
    pub const SRAM: u8 = 2;
    pub const NOC: u8 = 3;
    pub const PHY: u8 = 4;
    pub const HBM: u8 = 5;
    pub const CONTROL: u8 = 6;
}

/// Structural validation of the design under one profile (01 §18), shown on the design sheet.
#[derive(Clone, Debug, Default)]
pub struct ProfileCheck {
    pub profile: String,
    pub diagnostics: Vec<Diagnostic>,
}

/// Validation of an expanded design under the reference and search profiles (search compares overrides with
/// kiln-phys's derived values, as `kiln validate --profile search` does); load warnings are kept in both.
pub fn profile_checks(design: &kiln_ir::hw::Design, hw: &HwModel) -> Vec<ProfileCheck> {
    use kiln_ir::hw::validate::validate_priced;
    use kiln_ir::hw::Profile;
    [("reference", Profile::Reference), ("search", Profile::Search)]
        .into_iter()
        .map(|(name, p)| {
            let pricer = (p == Profile::Search).then(|| kiln_phys::pricing::pricer(hw));
            let mut diagnostics = design.warnings.clone();
            diagnostics.extend(validate_priced(&design.doc, hw, p, pricer.as_deref()));
            ProfileCheck {
                profile: name.into(),
                diagnostics,
            }
        })
        .collect()
}

pub struct Placed {
    /// `manifest.floorplan_source`, e.g. `kiln-phys/m3 tier A`.
    pub source: String,
    pub floorplan: Vec<FloorplanRow>,
    pub wires: Vec<WireRow>,
    pub package: Vec<PackageRow>,
    pub summary: Value,
    pub package_mm2: f64,
}

fn name_of<T: Serialize>(x: &T) -> String {
    match serde_json::to_value(x) {
        Ok(Value::String(s)) => s,
        Ok(v) => v.to_string(),
        Err(_) => "?".into(),
    }
}

fn subtree_sum<T: Copy>(
    hw: &HwModel,
    order: &[usize],
    own: Vec<T>,
    add: impl Fn(&mut T, T),
) -> Vec<T> {
    let mut v = own;
    for &i in order.iter().rev() {
        if let Some(p) = hw.nodes[i].parent {
            let x = v[i];
            add(&mut v[p], x);
        }
    }
    v
}

/// Children arranged inside `r` like kiln-phys's hierarchical layout: area-proportional slicing over entity
/// groups (canonical order), arrays of one entity as grids (declared 2-D coordinates when present).
fn arrange(hw: &HwModel, kids: &[usize], area: &[f64], r: Rect) -> Vec<(usize, Rect)> {
    let mut groups: Vec<(&str, Vec<usize>)> = vec![];
    for &k in kids {
        let e = hw.nodes[k].entity.as_str();
        match groups.iter_mut().find(|g| g.0 == e) {
            Some(g) => g.1.push(k),
            None => groups.push((e, vec![k])),
        }
    }
    let areas: Vec<f64> = groups
        .iter()
        .map(|g| g.1.iter().map(|&k| area[k]).sum::<f64>().max(1e-9))
        .collect();
    let mut out = vec![];
    for (gi, gr) in slice_seq(&areas, r).into_iter().enumerate() {
        let members = &groups[gi].1;
        if members.len() == 1 {
            out.push((members[0], gr));
            continue;
        }
        let coords: Vec<&Vec<u32>> = members.iter().map(|&m| &hw.nodes[m].coord).collect();
        let declared = coords.iter().all(|c| c.len() == 2).then(|| {
            (
                coords.iter().map(|c| c[0] as usize).max().unwrap_or(0) + 1,
                coords.iter().map(|c| c[1] as usize).max().unwrap_or(0) + 1,
            )
        });
        let (rows, cols) = match declared {
            Some(d) => d,
            None => grid_dims(members.len(), gr.w(), gr.h()),
        };
        let (cw, chh) = (gr.w() / cols as f64, gr.h() / rows as f64);
        for (i, &m) in members.iter().enumerate() {
            let (ri, ci) = match declared {
                Some(_) => (
                    hw.nodes[m].coord[0] as usize,
                    hw.nodes[m].coord[1] as usize,
                ),
                None => (i / cols, i % cols),
            };
            let x0 = gr.x0 + ci as f64 * cw;
            let y0 = gr.y0 + ri as f64 * chh;
            out.push((m, Rect::new(x0, y0, x0 + cw, y0 + chh)));
        }
    }
    out
}

/// Rows x columns for `n` cells in a `w x h` box with the squarest cells.
fn grid_dims(n: usize, w: f64, h: f64) -> (usize, usize) {
    let mut best = (1, n.max(1));
    let mut score = f64::INFINITY;
    for r in 1..=n.max(1) {
        let c = n.div_ceil(r);
        let s = ((h / r as f64) / (w / c as f64)).ln().abs() + (r * c - n) as f64 / n as f64;
        if s < score - 1e-12 {
            score = s;
            best = (r, c);
        }
    }
    best
}

/// Area-proportional recursive halving of `areas` (in order) over `r`, cutting the longer side.
fn slice_seq(areas: &[f64], r: Rect) -> Vec<Rect> {
    fn go(areas: &[f64], r: Rect, out: &mut Vec<Rect>) {
        if areas.len() <= 1 {
            out.push(r);
            return;
        }
        let total: f64 = areas.iter().sum();
        let (mut acc, mut k) = (0.0, 1);
        for (i, a) in areas.iter().enumerate().take(areas.len() - 1) {
            acc += a;
            k = i + 1;
            if acc >= 0.5 * total {
                break;
            }
        }
        let f = areas[..k].iter().sum::<f64>() / total;
        let (a, b) = if r.w() >= r.h() {
            r.split_v(f)
        } else {
            r.split_h(f)
        };
        go(&areas[..k], a, out);
        go(&areas[k..], b, out);
    }
    let mut out = Vec::with_capacity(areas.len());
    go(areas, r, &mut out);
    out
}

/// Largest axis-aligned rectangle inside `r` overlapping none of `taken`.
fn largest_free(r: Rect, taken: &[Rect]) -> Option<Rect> {
    let mut xs: Vec<f64> = taken
        .iter()
        .flat_map(|t| [t.x0, t.x1])
        .chain([r.x0, r.x1])
        .map(|x| x.clamp(r.x0, r.x1))
        .collect();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let mut best: Option<(f64, Rect)> = None;
    for (i, &x0) in xs.iter().enumerate() {
        for &x1 in &xs[i + 1..] {
            let mut ys: Vec<(f64, f64)> = taken
                .iter()
                .filter(|t| t.x0 < x1 && t.x1 > x0)
                .map(|t| (t.y0, t.y1))
                .collect();
            ys.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut y = r.y0;
            for (y0, y1) in ys.into_iter().chain([(r.y1, r.y1)]) {
                let a = (x1 - x0) * (y0.min(r.y1) - y);
                if a > best.map_or(0.0, |b| b.0) {
                    best = Some((a, Rect::new(x0, y, x1, y0.min(r.y1))));
                }
                y = y.max(y1);
            }
        }
    }
    best.map(|b| b.1)
}

fn under(hw: &HwModel, mut n: usize, root: usize) -> bool {
    loop {
        if n == root {
            return true;
        }
        match hw.nodes[n].parent {
            Some(p) => n = p,
            None => return false,
        }
    }
}

fn shrink(r: Rect, f: f64) -> Rect {
    let (hw_, hh) = (0.5 * r.w() * f, 0.5 * r.h() * f);
    Rect::new(r.cx() - hw_, r.cy() - hh, r.cx() + hw_, r.cy() + hh)
}

fn union(a: Option<Rect>, b: Rect) -> Rect {
    match a {
        None => b,
        Some(a) => Rect::new(a.x0.min(b.x0), a.y0.min(b.y0), a.x1.max(b.x1), a.y1.max(b.y1)),
    }
}

/// Package frame transform: kiln-phys's package frame (y up) to the trace frame (y down, packages laid out
/// side by side).
#[derive(Clone, Copy)]
struct Frame {
    outline: Rect,
    ox: f64,
    oy: f64,
}

impl Frame {
    fn pt(&self, x: f64, y: f64) -> (f64, f64) {
        (x - self.outline.x0 + self.ox, self.outline.y1 - y + self.oy)
    }

    fn rect(&self, r: Rect) -> Rect {
        let (x0, y0) = self.pt(r.x0, r.y1);
        Rect::new(x0, y0, x0 + r.w(), y0 + r.h())
    }
}

fn block_code(hw: &HwModel, n: usize, parts: &[f64; 7]) -> u8 {
    match hw.nodes[n].ix {
        NodeIx::Unit(_) => block::COMPUTE,
        NodeIx::Mem(m) if hw.memories[m].is_stack() => block::HBM,
        NodeIx::Mem(_) => block::SRAM,
        NodeIx::Router(_) | NodeIx::Net(_) => block::NOC,
        NodeIx::Port(_) => block::PHY,
        NodeIx::Block(b) => match &hw.blocks[b].spec.kind {
            BlockKind::Phy(_) => block::PHY,
            BlockKind::MemController(_) | BlockKind::Dma(_) | BlockKind::Sequencer { .. } => {
                block::CONTROL
            }
            BlockKind::Misc { .. } => block::MISC,
        },
        NodeIx::Container(_) => {
            let by = [
                (block::COMPUTE, parts[Part::Datapath as usize]),
                (
                    block::SRAM,
                    parts[Part::Sram as usize] + parts[Part::Rf as usize],
                ),
                (block::CONTROL, parts[Part::Control as usize]),
                (block::NOC, parts[Part::Noc as usize]),
                (block::PHY, parts[Part::Phy as usize]),
                (block::MISC, parts[Part::Misc as usize]),
            ];
            by.iter()
                .fold((block::MISC, 0.0), |b, x| if x.1 > b.1 { *x } else { b })
                .0
        }
    }
}

fn kind_code(k: ChannelKind) -> u8 {
    match k {
        ChannelKind::Feed => 0,
        ChannelKind::MemPort => 1,
        ChannelKind::NocHop => 2,
        ChannelKind::Bus => 3,
        ChannelKind::D2d => 4,
        ChannelKind::Serdes => 5,
        ChannelKind::Vertical => 6,
        ChannelKind::Optical => 7,
        ChannelKind::Near => 8,
        ChannelKind::Host => 9,
    }
}

fn link_source_code(s: LinkSource) -> u8 {
    match s {
        LinkSource::Wire => 0,
        LinkSource::NocHop => 1,
        LinkSource::Phy => 2,
        LinkSource::Bond3d => 3,
        LinkSource::Package => 4,
        LinkSource::Host => 5,
    }
}

fn class_code(c: Option<WireClassId>) -> u8 {
    match c {
        None => 0,
        Some(WireClassId::Local) => 1,
        Some(WireClassId::Intermediate) => 2,
        Some(WireClassId::SemiGlobal) => 3,
        Some(WireClassId::Global) => 4,
    }
}

/// Placed geometry and the design sheet of `hw` from its kiln-phys model; `None` for the legacy model or a
/// design without dies. `resource` maps an instance or channel path to its trace resource row.
pub fn place(
    hw: &HwModel,
    ph: &Phys,
    resource: &dyn Fn(&str) -> Option<u32>,
    checks: &[ProfileCheck],
) -> Option<Placed> {
    let m = ph.m3()?;
    let (fp, ch) = (&m.fp, &m.ch);
    if fp.dies.is_empty() || fp.packages.is_empty() {
        return None;
    }
    let nn = hw.nodes.len();
    let en = |i: usize| hw.nodes[i].enabled;
    let sub = subtree_areas(hw, ch);
    let leak = subtree_sum(
        hw,
        &ch.order,
        ch.nodes.iter().map(|n| n.leak_w).collect(),
        |a, b| *a += b,
    );
    let parts = subtree_sum(
        hw,
        &ch.order,
        ch.nodes.iter().map(|n| n.parts).collect(),
        |a, b| a.iter_mut().zip(b).for_each(|(x, y)| *x += y),
    );
    let live = structural_live(hw);
    let in_unit = unit_subtrees(hw, ch);

    // kiln-phys rectangles (package frame, y up), then units filled in by area inside their placed parent.
    let mut rect: Vec<Option<Rect>> = (0..nn).map(|i| fp.rect[i].filter(|_| en(i))).collect();
    let mut src = vec![source::LAYOUT; nn];
    for d in &fp.dies {
        src[d.node] = source::PLACED;
        for x in d.macros.iter().flat_map(|mm| &mm.nodes) {
            src[*x] = source::PLACED;
        }
    }
    for p in &fp.packages {
        let pn = hw.tree[p.container].node;
        rect[pn] = Some(p.outline).filter(|_| en(pn));
        src[pn] = source::PLACED;
        for (mi, _) in &p.stacks {
            src[hw.memories[*mi].node] = source::PLACED;
        }
    }
    let drawable = |k: usize| {
        en(k)
            && !matches!(hw.nodes[k].ix, NodeIx::Net(_))
            && sub[k] > 0.0
            && (live[k] || in_unit[k])
    };
    let mut force = vec![false; nn];
    for &n in &ch.order {
        let Some(r) = rect[n] else { continue };
        let kids: Vec<usize> = ch.kids[n].iter().copied().filter(|&k| drawable(k)).collect();
        if kids.is_empty() || !(force[n] || kids.iter().any(|&k| rect[k].is_none() && in_unit[k])) {
            continue;
        }
        // kiln-phys's placement (macros, shoreline PHYs) stays put unless its parent moved; the rest goes into the
        // largest free rectangle around it.
        let (keep, kids): (Vec<usize>, Vec<usize>) = kids
            .into_iter()
            .partition(|&k| !force[n] && src[k] == source::PLACED && rect[k].is_some());
        let kept: Vec<Rect> = keep.iter().filter_map(|&k| rect[k]).collect();
        let (space, cap) = match largest_free(r, &kept) {
            Some(f) if !kept.is_empty() => (f, f.w() * f.h()),
            _ => (r, sub[n]),
        };
        let content: f64 = kids.iter().map(|&k| sub[k]).sum();
        let inner = shrink(space, (content / cap.max(1e-9)).clamp(1e-6, 1.0).sqrt());
        for (k, kr) in arrange(hw, &kids, &sub, inner) {
            rect[k] = Some(kr);
            src[k] = source::FILLED;
            force[k] = true;
        }
    }
    // Live-structure blocks with area that nothing placed (no channels: kiln-phys keeps them out of geometry).
    let mut unplaced: Vec<(String, f64)> = (0..nn)
        .filter(|&i| {
            en(i)
                && rect[i].is_none()
                && sub[i] > 0.0
                && !matches!(hw.nodes[i].ix, NodeIx::Net(_) | NodeIx::Router(_))
                && hw.nodes[i].parent.is_some_and(|p| rect[p].is_some())
        })
        .map(|i| (hw.nodes[i].path.clone(), sub[i]))
        .collect();
    unplaced.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    // One frame per package, laid out on a grid (array coordinates when the packages declare 2-D ones).
    let cont_kind = |i: usize| match hw.nodes[i].ix {
        NodeIx::Container(c) => Some(hw.tree[c].kind),
        _ => None,
    };
    let pkg_nodes: Vec<usize> = fp.packages.iter().map(|p| hw.tree[p.container].node).collect();
    let (cell_w, cell_h) = fp
        .packages
        .iter()
        .fold((0.0f64, 0.0f64), |a, p| (a.0.max(p.outline.w()), a.1.max(p.outline.h())));
    let gap = 0.12 * cell_w.max(cell_h);
    let np = fp.packages.len();
    let coords: Vec<&Vec<u32>> = pkg_nodes.iter().map(|&n| &hw.nodes[n].coord).collect();
    let grid_coords = coords.iter().all(|c| c.len() == 2);
    let cols = np.div_ceil((np as f64).sqrt().floor().max(1.0) as usize).max(1);
    let frames: Vec<Frame> = fp
        .packages
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let (r, c) = if grid_coords {
                (coords[k][0] as usize, coords[k][1] as usize)
            } else {
                (k / cols, k % cols)
            };
            Frame {
                outline: p.outline,
                ox: c as f64 * (cell_w + gap),
                oy: r as f64 * (cell_h + gap),
            }
        })
        .collect();
    let mut frame_of: Vec<Option<usize>> = vec![None; nn];
    for &i in &ch.order {
        frame_of[i] = match hw.nodes[i].parent {
            _ if cont_kind(i) == Some(ContainerKind::Package) => pkg_nodes.iter().position(|&p| p == i),
            Some(p) => frame_of[p],
            None => None,
        };
    }
    let mut out_rect: Vec<Option<Rect>> = (0..nn)
        .map(|i| Some(frames[frame_of[i]?].rect(rect[i]?)))
        .collect();
    // Containers above the packages: the box around what they hold.
    for &i in ch.order.iter().rev() {
        if let Some(p) = hw.nodes[i].parent
            && let Some(r) = out_rect[i]
            && frame_of[p].is_none()
            && en(i)
            && en(p)
        {
            out_rect[p] = Some(union(out_rect[p], r));
        }
    }
    for i in 0..nn {
        if frame_of[i].is_none()
            && let Some(r) = out_rect[i]
        {
            let pad = 0.02 * r.w().max(r.h());
            out_rect[i] = Some(Rect::new(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad));
        }
    }

    let die_ix = |i: usize| fp.die_of[i].and_then(|c| fp.dies.iter().position(|d| d.container == c));
    let layer = |i: usize| die_ix(i).map_or(0, |d| fp.dies[d].layer.clamp(0, 255) as u8);
    let die_res = |i: usize| {
        die_ix(i)
            .and_then(|d| resource(&fp.dies[d].path))
            .unwrap_or(NONE_U32)
    };
    let mut floorplan: Vec<FloorplanRow> = (0..nn)
        .filter(|&i| en(i) && !hw.nodes[i].path.is_empty())
        .filter_map(|i| {
            let r = out_rect[i]?;
            Some(FloorplanRow {
                resource: resource(&hw.nodes[i].path)?,
                die: die_res(i),
                layer: layer(i),
                x_um: r.x0,
                y_um: r.y0,
                w_um: r.w(),
                h_um: r.h(),
                poly: None,
                rotation: 0,
                source: src[i],
                block: block_code(hw, i, &parts[i]),
                area_um2: Some(sub[i]),
                leak_w: Some(leak[i]),
            })
        })
        .collect();
    let center = |i: usize| -> Option<(f64, f64)> {
        match out_rect[i] {
            Some(r) if frame_of[i].is_some() => Some((r.cx(), r.cy())),
            _ => frame_of[i].map(|f| frames[f].pt(fp.pos[i].0, fp.pos[i].1)),
        }
    };
    // Routers and footprint-less ports: positions only (wire ends), at kiln-phys's node positions.
    let sites = hw.routers.iter().map(|r| r.node).chain(hw.ports.iter().map(|p| p.node));
    for n in sites {
        if !en(n) || out_rect[n].is_some() {
            continue;
        }
        let (Some((x, y)), Some(res)) = (center(n), resource(&hw.nodes[n].path)) else {
            continue;
        };
        let side = ch.nodes[n].area_um2.max(1.0).sqrt();
        floorplan.push(FloorplanRow {
            resource: res,
            die: die_res(n),
            layer: layer(n),
            x_um: x - 0.5 * side,
            y_um: y - 0.5 * side,
            w_um: side,
            h_um: side,
            poly: None,
            rotation: 0,
            source: source::SITE,
            block: block_code(hw, n, &parts[n]),
            area_um2: Some(sub[n]),
            leak_w: Some(leak[n]),
        });
    }
    floorplan.sort_by_key(|f| f.resource);
    floorplan.dedup_by_key(|f| f.resource);

    let mut wires = vec![];
    for (ci, c) in hw.channels.iter().enumerate() {
        let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
        if !en(s) || !en(d) {
            continue;
        }
        let (Some(ps), Some(pd), Some(rs), Some(rd)) = (
            center(s),
            center(d),
            resource(&hw.nodes[s].path),
            resource(&hw.nodes[d].path),
        ) else {
            continue;
        };
        let lc = &m.links[ci];
        let mut poly = vec![ps.0, ps.1];
        if (ps.0 - pd.0).abs() > 1e-6 && (ps.1 - pd.1).abs() > 1e-6 {
            poly.extend([pd.0, ps.1]);
        }
        poly.extend([pd.0, pd.1]);
        wires.push(WireRow {
            link: resource(&format!("{}-{}", hw.path(c.src), hw.path(c.dst))).unwrap_or(NONE_U32),
            src: rs,
            dst: rd,
            kind: kind_code(c.kind),
            source: link_source_code(lc.source),
            layer: layer(s).max(layer(d)),
            polyline: poly,
            length_um: lc.length_um,
            width_bits: c.width_bits.map(f64::from),
            bw_bps: c.bandwidth.map_or(lc.bw_bytes_per_s, |b| b.0),
            latency_s: lc.latency_s,
            e_j_per_bit: lc.e_j_per_byte / 8.0,
            class: class_code(lc.class),
            pipeline_stages: lc.pipeline_stages,
        });
    }

    let mut package = vec![];
    for (k, p) in fp.packages.iter().enumerate() {
        let r = frames[k].rect(p.outline);
        package.push(PackageRow {
            kind: 0,
            resource: resource(&p.path),
            layer: 0,
            x_um: r.x0,
            y_um: r.y0,
            w_um: r.w(),
            h_um: r.h(),
            label: p.table.clone(),
            value: Some(p.outline.area()),
            value2: Some(p.max_mm2 * 1e6),
        });
    }
    for d in &fp.dies {
        let (Some(o), Some(f)) = (rect[d.node], frame_of[d.node]) else {
            continue;
        };
        let t = (0.012 * o.w().min(o.h())).max(120.0);
        for (ei, e) in d.edges.iter().enumerate() {
            if e.used_um <= 0.0 {
                continue;
            }
            let strip = match ei {
                0 => Rect::new(o.x0, o.y0, o.x1, o.y0 + t),
                1 => Rect::new(o.x1 - t, o.y0, o.x1, o.y1),
                2 => Rect::new(o.x0, o.y1 - t, o.x1, o.y1),
                _ => Rect::new(o.x0, o.y0, o.x0 + t, o.y1),
            };
            let r = frames[f].rect(strip);
            package.push(PackageRow {
                kind: 1,
                resource: resource(&d.path),
                layer: d.layer.clamp(0, 255) as u8,
                x_um: r.x0,
                y_um: r.y0,
                w_um: r.w(),
                h_um: r.h(),
                label: ["S", "E", "N", "W"][ei].into(),
                value: Some(e.used_um),
                value2: Some(e.hbm_used_um),
            });
        }
    }
    for i in 0..nn {
        let harvested = !en(i)
            && hw.nodes[i].parent.is_some_and(en)
            && (matches!(hw.nodes[i].ix, NodeIx::Mem(m) if hw.memories[m].is_stack())
                || cont_kind(i) == Some(ContainerKind::Die));
        if let (true, Some(r), Some(f)) = (harvested, fp.rect[i], frame_of[i]) {
            let r = frames[f].rect(r);
            let owner = std::iter::successors(hw.nodes[i].parent, |&p| hw.nodes[p].parent)
                .find_map(|p| resource(&hw.nodes[p].path));
            package.push(PackageRow {
                kind: 2,
                resource: owner,
                layer: layer(i),
                x_um: r.x0,
                y_um: r.y0,
                w_um: r.w(),
                h_um: r.h(),
                label: hw.nodes[i].path.clone(),
                value: Some(sub[i]),
                value2: None,
            });
        }
    }

    let report = ph.report()?;
    let summary = design_summary(hw, ph, checks, &unplaced);
    Some(Placed {
        source: format!("{} tier {:?}", report.model, report.place_tier),
        floorplan,
        wires,
        package,
        summary,
        package_mm2: report.package_mm2,
    })
}

/// The design sheet (05 §6.1 design-only mode): peaks, memory hierarchy, off-chip memory, interconnect, area
/// and power breakdowns, clocks, validation status and physical findings.
pub fn design_summary(
    hw: &HwModel,
    ph: &Phys,
    checks: &[ProfileCheck],
    unplaced: &[(String, f64)],
) -> Value {
    let s = hw.summary();
    let mut elem: BTreeMap<String, f64> = BTreeMap::new();
    for c in &s.chips {
        for (k, v) in &c.elem_ops {
            *elem.entry(k.clone()).or_default() += v;
        }
    }
    let levels: Vec<Value> = s.chips.first().map_or_else(Vec::new, |c| {
        let chip = hw.index.get(&c.path).map(|&ix| hw.node_of(ix));
        c.levels
            .iter()
            .filter(|l| l.level != u8::MAX)
            .map(|l| {
                let mem = (0..hw.memories.len()).find(|&mi| {
                    hw.levels.get(mi) == Some(&l.level)
                        && hw.nodes[hw.memories[mi].node].enabled
                        && chip.is_none_or(|c| under(hw, hw.memories[mi].node, c))
                });
                let (name, kind) = mem.map_or_else(
                    || (String::new(), String::new()),
                    |mi| {
                        let mm = &hw.memories[mi];
                        let kind = match &mm.spec {
                            MemSpec::OnChip(x) => name_of(&x.kind),
                            MemSpec::Stack(x) => name_of(&x.kind),
                            MemSpec::Local { .. } => "local".into(),
                        };
                        (hw.nodes[mm.node].entity_id.clone(), kind)
                    },
                );
                let read_j_per_b = mem.and_then(|mi| ph.m3().map(|m| m.ch.mems[mi].read_j_per_b));
                json!({"level": l.level, "name": name, "kind": kind, "instances": l.instances,
                    "capacity_b": l.capacity.0 as f64, "bandwidth_bps": l.bandwidth.0,
                    "read_j_per_b": read_j_per_b, "full_bw_w": read_j_per_b.map(|e| e * l.bandwidth.0)})
            })
            .collect()
    });
    let stacks: Vec<usize> = (0..hw.memories.len())
        .filter(|&m| hw.memories[m].is_stack())
        .collect();
    let live_stacks: Vec<usize> = stacks
        .iter()
        .copied()
        .filter(|&m| hw.nodes[hw.memories[m].node].enabled)
        .collect();
    let stack_kind = live_stacks.first().map_or_else(String::new, |&m| match &hw.memories[m].spec {
        MemSpec::Stack(x) => name_of(&x.kind),
        _ => String::new(),
    });
    let offchip = json!({
        "stacks": live_stacks.len(),
        "harvested": stacks.len() - live_stacks.len(),
        "kind": stack_kind,
        "capacity_b": s.offchip_capacity.0 as f64,
        "bandwidth_bps": s.offchip_bandwidth.0,
        "per_stack_bps": live_stacks.first().and_then(|&m| hw.memories[m].bandwidth).map(|b| b.0),
    });
    let nets: Vec<Value> = hw
        .networks
        .iter()
        .enumerate()
        .filter(|(_, n)| hw.nodes[n.node].enabled)
        .map(|(ni, n)| {
            let chans: Vec<f64> = hw
                .channels
                .iter()
                .filter(|c| c.network == Some(ni))
                .map(|c| c.bandwidth.map_or(0.0, |b| b.0))
                .collect();
            json!({"path": hw.nodes[n.node].path, "topology": n.spec.topology.name(),
                "endpoints": n.endpoints.len(), "routers": n.routers.len(), "links": chans.len(),
                "link_bw_bps": chans.iter().copied().fold(0.0, f64::max),
                "width_bits": n.spec.link.width_bits, "phys": n.spec.link.phys.name(), "direct": n.direct})
        })
        .collect();
    let inter: Vec<Value> = s
        .inter_chip
        .iter()
        .map(|n| json!({"path": n.path, "topology": n.topology, "endpoints": n.endpoints,
            "link_bw_bps": n.link_bandwidth.map(|b| b.0)}))
        .collect();
    let mut chan_kinds: BTreeMap<String, usize> = BTreeMap::new();
    for c in &hw.channels {
        *chan_kinds.entry(name_of(&c.kind)).or_default() += 1;
    }

    let rep = ph.report();
    let area = rep.map_or(Value::Null, |r| {
        // Package packing is not monotone in the area corners (fixed-outline dies can pack larger at the optimistic
        // corner): the band is the envelope over both corners and the central placement.
        let corners = [r.package_low_mm2, r.package_high_mm2, r.package_mm2];
        json!({"package_mm2": r.package_mm2, "package_low_mm2": corners.iter().copied().fold(f64::INFINITY, f64::min),
            "package_high_mm2": corners.iter().copied().fold(0.0, f64::max), "package_table": r.package_table,
            "dies": r.dies.iter().map(|d| json!({"path": d.path, "node": d.node, "layer": d.layer,
                "area_mm2": d.area_mm2, "area_low_mm2": d.area_low_mm2, "area_high_mm2": d.area_high_mm2,
                "outline_mm2": d.outline_mm2, "legalized_mm2": d.legalized_mm2, "fixed_outline": d.fixed_outline,
                "parts_mm2": d.parts_mm2, "whitespace_mm2": d.whitespace_mm2, "transistors_b": d.transistors_b,
                "sram_mib": d.sram_mib, "shoreline_limited": d.shoreline_limited, "edges_mm": d.edges_mm,
                "hbm_shoreline_used_mm": d.hbm_shoreline_used_mm,
                "hbm_shoreline_available_mm": d.hbm_shoreline_available_mm})).collect::<Vec<_>>()})
    });

    let power = ph.m3().map_or(Value::Null, |m| {
        let plan = ph.clock_plan(&ClockMode::Nominal);
        let mac_w = ph.peak_compute_w(&plan, None);
        let (mut dram_j, mut phy_j) = (0.0, 0.0);
        for &mi in &live_stacks {
            let bw = ph.mem(mi).bandwidth_bps;
            let e = &m.ch.mems[mi];
            dram_j += bw * e.e_core_j_per_b;
            phy_j += bw * e.e_phy_j_per_b;
        }
        let peak = PhaseEnergy {
            makespan_s: 1.0,
            core_dyn_j: mac_w,
            indep_j: phy_j,
            dram_j,
            activity: 1.0,
            busy: 1.0,
        };
        let idle = PhaseEnergy {
            makespan_s: 1.0,
            ..PhaseEnergy::default()
        };
        let row = |e: &PhaseEnergy| {
            ph.phase_power(e, &plan).map_or(Value::Null, |b| {
                json!({"mac_w": b.dyn_core_w, "phy_w": b.indep_w, "static_w": b.static_w, "clock_w": b.clock_w,
                    "control_w": b.ctrl_w, "dram_w": b.dram_w, "board_fixed_w": b.board_fixed_w,
                    "vr_loss_w": b.vr_loss_w, "chip_w": b.chip_w, "package_w": b.package_w, "board_w": b.board_w,
                    "t_j_c": b.t_j_c, "runaway": b.runaway})
            })
        };
        json!({"tdp_w": m.power.cap_w, "tdp_assumed_w": m.power.assumed_cap_w,
            "cap_level": name_of(&m.power.cap_level), "static_85c_w": ph.static_power_w(),
            "tj_max_c": m.power.tj_max_c, "q_avg_max_w_mm2": m.power.q_avg_max,
            "peak": row(&peak), "idle": row(&idle)})
    });
    let clocks: Vec<Value> = hw
        .clocks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let d = ph.m3().and_then(|m| m.power.domains.get(i));
            let pts = d.map(|d| d.vf.points.clone()).unwrap_or_default();
            json!({"path": c.path, "nominal_hz": c.spec.freq.0, "base_hz": c.spec.base.map(|b| b.0),
                "vf_declared": !c.spec.vf.is_empty(),
                "f_min_hz": pts.first().map(|p| p.0), "f_max_hz": pts.last().map(|p| p.0),
                "v_min": pts.first().map(|p| p.1), "v_max": pts.last().map(|p| p.1),
                "leak_w": d.map(|d| d.leak_w)})
        })
        .collect();
    let diag = |d: &Diagnostic| json!({"code": d.code, "message": d.message, "path": d.path});
    let validation: Vec<Value> = checks
        .iter()
        .map(|c| {
            let errs: Vec<&Diagnostic> = c.diagnostics.iter().filter(|d| d.severity == Severity::Error).collect();
            let warns: Vec<&Diagnostic> = c.diagnostics.iter().filter(|d| d.severity == Severity::Warning).collect();
            json!({"profile": c.profile, "errors": errs.len(), "warnings": warns.len(),
                "first": errs.iter().chain(&warns).take(4).map(|d| diag(d)).collect::<Vec<_>>()})
        })
        .collect();
    json!({
        "name": s.name,
        "family": hw.family,
        "design_hash": s.design_hash,
        "exec_model": name_of(&hw.exec_model),
        "chips": s.chip_count,
        "node": rep.map(|r| r.node.clone()),
        "model": rep.map(|r| r.model.clone()),
        "calibration": rep.map(|r| r.calibration.clone()),
        "peak_ops": s.peak_ops,
        "elem_ops": elem,
        "memory": levels,
        "onchip_capacity_b": s.onchip_capacity.0 as f64,
        "chip_onchip_capacity_b": s.chips.first().map(|c| c.onchip_capacity.0 as f64),
        "offchip": offchip,
        "networks": nets,
        "inter_chip": inter,
        "channels": chan_kinds,
        "area": area,
        "power": power,
        "clocks": clocks,
        "validation": validation,
        "problems": ph.problems().iter().map(diag).collect::<Vec<_>>(),
        "unplaced": unplaced.iter().map(|(p, a)| json!({"path": p, "area_mm2": a / 1e6})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::structure;
    use crate::check::check_trace;
    use crate::container::{read_kiln, write_kiln};
    use crate::trace::Trace;

    fn design(name: &str) -> (HwModel, kiln_ir::hw::Design) {
        let p = format!("{}/../../designs/reference/{name}.json5", env!("CARGO_MANIFEST_DIR"));
        let d = kiln_ir::hw::load_file(&p).expect("loads");
        let (hw, _) = d.expand(&Default::default()).expect("expands");
        (hw, d)
    }

    fn trace(name: &str) -> Trace {
        let (hw, d) = design(name);
        structure(&hw, Some(d.canonical.clone()), Some(name.into()), &profile_checks(&d, &hw))
    }

    /// Every placed block lies inside the nearest placed ancestor (sites excepted: positions only).
    fn contained(t: &Trace) {
        let row: BTreeMap<u32, &FloorplanRow> = t.floorplan.iter().map(|f| (f.resource, f)).collect();
        for f in t.floorplan.iter().filter(|f| f.source != source::SITE) {
            let mut p = t.resources[f.resource as usize].parent;
            while let Some(q) = p {
                if let Some(g) = row.get(&q) {
                    let tol = 1e-6 * g.w_um.max(g.h_um) + 1e-6;
                    assert!(
                        f.x_um >= g.x_um - tol
                            && f.y_um >= g.y_um - tol
                            && f.x_um + f.w_um <= g.x_um + g.w_um + tol
                            && f.y_um + f.h_um <= g.y_um + g.h_um + tol,
                        "{} outside {}",
                        t.resources[f.resource as usize].path,
                        t.resources[q as usize].path
                    );
                    break;
                }
                p = t.resources[q as usize].parent;
            }
        }
    }

    #[test]
    fn reference_designs_place_validate_and_round_trip() {
        for name in ["a100_sxm4_40gb", "h100_sxm5_80gb", "tpu_v6e", "tpu_v5e_2x2", "ember"] {
            let t = trace(name);
            assert!(t.manifest.floorplan_source.as_deref().is_some_and(|s| s.starts_with("kiln-phys")), "{name}");
            assert!(check_trace(&t).is_empty(), "{name}: {:?}", check_trace(&t));
            contained(&t);
            let kinds: Vec<&str> = t.floorplan.iter().map(|f| t.resource_kind(&t.resources[f.resource as usize])).collect();
            for k in ["package", "die", "mem_stack"] {
                assert!(kinds.contains(&k), "{name}: no placed {k}");
            }
            assert!(t.floorplan.iter().any(|f| f.source == source::FILLED), "{name}: units filled in");
            assert!(!t.wires.is_empty() && t.wires.iter().all(|w| w.link != NONE_U32), "{name}: wires");
            let area = &t.scalar("design_summary").unwrap()["area"];
            let (lo, c, hi) = (area["package_low_mm2"].as_f64().unwrap(), area["package_mm2"].as_f64().unwrap(), area["package_high_mm2"].as_f64().unwrap());
            assert!(0.0 < lo && lo <= c && c <= hi, "{name}: package [{lo}, {c}, {hi}]");
            let bytes = write_kiln(&t);
            assert_eq!(bytes, write_kiln(&trace(name)), "{name}: deterministic");
            let mut back = read_kiln(&bytes).unwrap();
            back.manifest.tables.clear();
            assert_eq!(back, t, "{name}: round trip");
        }
    }

    /// kiln-phys's die and macro rectangles survive the unit fill unchanged, and no filled sibling overlaps one.
    fn placed_kept(name: &str, hw: &HwModel, t: &Trace) -> usize {
        let ph = Phys::new(hw);
        let fp = &ph.m3().unwrap().fp;
        let row: BTreeMap<&str, &FloorplanRow> =
            t.floorplan.iter().map(|f| (t.resources[f.resource as usize].path.as_str(), f)).collect();
        let placed = fp.dies.iter().flat_map(|d| d.macros.iter().flat_map(|m| m.nodes.iter().copied()).chain([d.node]));
        let mut n = 0;
        for i in placed.filter(|&i| hw.nodes[i].enabled) {
            let (Some(r), Some(f)) = (fp.rect[i], row.get(hw.nodes[i].path.as_str())) else { continue };
            let path = &hw.nodes[i].path;
            assert_eq!(f.source, source::PLACED, "{name}: {path} relaid");
            assert!((f.w_um - r.w()).abs() < 1e-6 && (f.h_um - r.h()).abs() < 1e-6, "{name}: {path} resized");
            for g in t.floorplan.iter().filter(|g| g.source == source::FILLED) {
                let gp = &t.resources[g.resource as usize].path;
                if g.layer == f.layer && hw.nodes[i].parent.is_some_and(|p| gp.starts_with(&format!("{}.", hw.nodes[p].path))) && !gp.starts_with(&format!("{path}.")) {
                    let ox = (f.x_um + f.w_um).min(g.x_um + g.w_um) - f.x_um.max(g.x_um);
                    let oy = (f.y_um + f.h_um).min(g.y_um + g.h_um) - f.y_um.max(g.y_um);
                    assert!(ox <= 1e-6 || oy <= 1e-6, "{name}: {gp} filled over {path}");
                }
            }
            n += 1;
        }
        n
    }

    /// tpu_v5e with the tensorcore's units, memories and networks directly on the die.
    fn flat_die() -> (HwModel, kiln_ir::hw::Design) {
        let p = format!("{}/../../designs/reference/tpu_v5e.json5", env!("CARGO_MANIFEST_DIR"));
        let t = std::fs::read_to_string(p).unwrap();
        let (a, b) = (t.find("    tensorcore: { kind: \"cluster\", body: {\n").unwrap(), t.find("    v5e_chip:").unwrap());
        let body = &t[a + "    tensorcore: { kind: \"cluster\", body: {\n".len()..b];
        let body = &body[..body.rfind("    } },").unwrap()];
        let t = t
            .replacen("        clusters: [ { id: \"tc\", use: \"tensorcore\" } ],\n", body, 1)
            .replacen("endpoints: [\"tc.vmem\", \"tc.smem\"]", "endpoints: [\"vmem\", \"smem\"]", 1)
            .replacen(&t[a..b], "", 1);
        let d = kiln_ir::hw::Design::from_source(&kiln_ir::hw::FsLoader, None, &t).expect("loads");
        let (hw, _) = d.expand(&Default::default()).expect("expands");
        (hw, d)
    }

    #[test]
    fn unit_fill_keeps_kiln_phys_placement() {
        let (hw, d) = flat_die();
        let t = structure(&hw, Some(d.canonical.clone()), Some("flat".into()), &profile_checks(&d, &hw));
        contained(&t);
        assert!(placed_kept("flat", &hw, &t) >= 11);
        assert!(t.floorplan.iter().any(|f| f.source == source::FILLED && t.resources[f.resource as usize].path == "board.chip.die.mxu0"));
        for name in ["a100_sxm4_40gb", "h100_sxm5_80gb", "tpu_v6e", "tpu_v5e_2x2", "tpu_v4", "ember"] {
            let (hw, d) = design(name);
            let t = structure(&hw, Some(d.canonical.clone()), Some(name.into()), &profile_checks(&d, &hw));
            assert!(placed_kept(name, &hw, &t) > 0, "{name}");
        }
    }

    #[test]
    fn stacks_shoreline_layers_and_frames() {
        let t = trace("a100_sxm4_40gb");
        let die = t.floorplan.iter().find(|f| t.resource_kind(&t.resources[f.resource as usize]) == "die").unwrap();
        // GA100: 826 mm^2 placed outline, HBM shoreline on two edges, one harvested stack.
        assert!((die.w_um * die.h_um / 1e6 - 826.9).abs() < 2.0, "{}", die.w_um * die.h_um / 1e6);
        let shore: Vec<&PackageRow> = t.package_geometry.iter().filter(|g| g.kind == 1).collect();
        assert!(shore.iter().filter(|g| g.value2.unwrap_or(0.0) > 0.0).count() == 2);
        assert_eq!(t.package_geometry.iter().filter(|g| g.kind == 2).count(), 1);
        let hbm = t.floorplan.iter().filter(|f| f.block == block::HBM && f.source == source::PLACED).count();
        assert_eq!(hbm, 5);

        let e = trace("ember");
        let layers: BTreeMap<u8, usize> = e
            .floorplan
            .iter()
            .filter(|f| e.resource_kind(&e.resources[f.resource as usize]) == "die")
            .fold(BTreeMap::new(), |mut m, f| {
                *m.entry(f.layer).or_default() += 1;
                m
            });
        assert_eq!(layers, BTreeMap::from([(0, 4), (1, 4)]));
        let vertical = e.wires.iter().filter(|w| e.enum_name("wires.kind", u32::from(w.kind)) == "vertical").count();
        assert_eq!(vertical, 8);

        // Four packages side by side, none overlapping.
        let m = trace("tpu_v5e_2x2");
        let pk: Vec<&FloorplanRow> = m
            .floorplan
            .iter()
            .filter(|f| m.resource_kind(&m.resources[f.resource as usize]) == "package")
            .collect();
        assert_eq!(pk.len(), 4);
        for (i, a) in pk.iter().enumerate() {
            for b in &pk[i + 1..] {
                let ox = (a.x_um + a.w_um).min(b.x_um + b.w_um) - a.x_um.max(b.x_um);
                let oy = (a.y_um + a.h_um).min(b.y_um + b.h_um) - a.y_um.max(b.y_um);
                assert!(ox <= 0.0 || oy <= 0.0);
            }
        }
    }
}
