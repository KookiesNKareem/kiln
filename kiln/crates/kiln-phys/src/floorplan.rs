//! Floorplan (04 §6): die macros after array collapsing, shoreline assignment, die sizing (§6.2), Tier A/B
//! macro placement, hierarchical layout of every instance inside its macro, router positions, and the package
//! (dies, HBM stacks, interposer outline, §6.3). Coordinates are um in the package frame.

use std::collections::BTreeMap;
use std::time::Instant;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::model::{ContainerKind, HwModel, MemSpec, NodeIx};
use kiln_ir::hw::net::IoKind;
use kiln_ir::hw::phys::{Edge, Outline, Placement};
use serde::Serialize;

use crate::characterize::{Characterized, PhyUse, package_table};
use crate::params::Params;
use crate::place::{self, AnnealStats, Problem, Rect, Tok};
use crate::tables::Tables;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaceTier {
    #[default]
    A,
    B,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Macro {
    /// Top instance nodes (one per array slot; disabled instances included).
    pub nodes: Vec<usize>,
    pub name: String,
    pub area_um2: f64,
    pub power_hint_w: f64,
    /// Edge-pinned PHY macro: (edge, shoreline um, PHY kind).
    pub phy: Option<(Edge, f64, IoKind)>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EdgeUse {
    pub len_um: f64,
    pub used_um: f64,
    pub hbm_used_um: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DieFp {
    pub container: usize,
    pub node: usize,
    pub path: String,
    pub tech: String,
    /// Analytic (Tier A) roll-up: block areas / u_place + seal ring (04 §10; the envelope value, monotone).
    pub area_env_um2: f64,
    pub blocks_um2: f64,
    /// Placed outline (fixed outline, or the sized die incl. shoreline growth).
    pub outline: Rect,
    pub fixed_outline: bool,
    pub shoreline_limited: bool,
    /// S, E, N, W.
    pub edges: [EdgeUse; 4],
    pub macros: Vec<Macro>,
    pub rects: Vec<Rect>,
    pub layer: i32,
    pub legalized_um2: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PackageFp {
    pub container: usize,
    pub path: String,
    pub table: String,
    pub outline: Rect,
    pub stacks: Vec<(usize, Rect)>,
    pub max_mm2: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PlaceStats {
    pub macros: usize,
    pub place_us: f64,
    pub phi_tier_a: f64,
    pub phi: f64,
    pub anneal: Option<AnnealStats>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Floorplan {
    pub tier: PlaceTier,
    pub dies: Vec<DieFp>,
    pub packages: Vec<PackageFp>,
    /// Rectangle of every arena node that occupies area (package frame); `None` for nodes without a footprint.
    pub rect: Vec<Option<Rect>>,
    /// Center of every node (package frame): its rectangle, else the centroid of what it connects.
    pub pos: Vec<(f64, f64)>,
    /// Die container index per node (`None` outside dies, e.g. stacks).
    pub die_of: Vec<Option<usize>>,
    /// Live subtree area per node (`live_subtree_areas`): the geometry of links; dead nodes are 0.
    pub geo_um2: Vec<f64>,
    pub stats: PlaceStats,
    pub problems: Vec<Diagnostic>,
}

pub fn edge_ix(e: Edge) -> usize {
    match e {
        Edge::S => 0,
        Edge::E => 1,
        Edge::N => 2,
        Edge::W => 3,
    }
}

const EDGES: [Edge; 4] = [Edge::S, Edge::E, Edge::N, Edge::W];

fn site_kind_matches(site: IoKind, k: IoKind) -> bool {
    site == k || matches!((site, k), (IoKind::Serdes, IoKind::Pcie) | (IoKind::Pcie, IoKind::Serdes))
}

/// Subtree area of every node (post-order over the arena: children have larger indices).
pub fn subtree_areas(hw: &HwModel, ch: &Characterized) -> Vec<f64> {
    let mut sub: Vec<f64> = ch.nodes.iter().map(|n| n.area_um2).collect();
    for &i in ch.order.iter().rev() {
        if let Some(p) = hw.nodes[i].parent {
            sub[p] += sub[i];
        }
    }
    sub
}

/// Structurally live nodes (04 §6.1, 06 P6): enabled compute units, everything enabled they reach over channels,
/// and the containers holding those. Harvested instances, blocks without channels (misc area) and memories nothing
/// connects to are dead: they count in area but never in the geometry links are measured on, so dead area cannot
/// move a live wire (an optimal placer would park it where it lengthens nothing).
pub fn structural_live(hw: &HwModel) -> Vec<bool> {
    let nn = hw.nodes.len();
    let mut adj = vec![vec![]; nn];
    for c in &hw.channels {
        let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
        if hw.nodes[s].enabled && hw.nodes[d].enabled {
            adj[s].push(d);
            adj[d].push(s);
        }
    }
    let mut live = vec![false; nn];
    let mut st: Vec<usize> = hw.units.iter().map(|u| u.node).filter(|&n| hw.nodes[n].enabled).collect();
    for &n in &st {
        live[n] = true;
    }
    while let Some(x) = st.pop() {
        for &y in &adj[x] {
            if !live[y] {
                live[y] = true;
                st.push(y);
            }
        }
    }
    for i in 0..nn {
        if live[i] {
            let mut p = hw.nodes[i].parent;
            while let Some(q) = p {
                if live[q] {
                    break;
                }
                live[q] = true;
                p = hw.nodes[q].parent;
            }
        }
    }
    live
}

/// Nodes inside a compute unit's subtree (the unit and its local buffers).
pub fn unit_subtrees(hw: &HwModel, ch: &Characterized) -> Vec<bool> {
    let mut in_unit = vec![false; hw.nodes.len()];
    for &i in &ch.order {
        in_unit[i] = matches!(hw.nodes[i].ix, NodeIx::Unit(_)) || hw.nodes[i].parent.is_some_and(|p| in_unit[p]);
    }
    in_unit
}

/// Subtree area counting only live nodes' own area; with `units` given, without compute-unit subtrees and the SIMT
/// control units bring with them (the die arrangement, P6).
pub fn live_subtree_areas(hw: &HwModel, ch: &Characterized, live: &[bool], units: Option<&[bool]>) -> Vec<f64> {
    let own = |i: usize| match units {
        _ if !live[i] => 0.0,
        None => ch.nodes[i].area_um2,
        Some(u) if u[i] => 0.0,
        Some(_) => ch.nodes[i].area_um2 - ch.nodes[i].unit_ctrl_um2,
    };
    // Children summed in value order: the total of a subtree does not depend on sibling order (P4/P6 bit-exactness).
    let mut sub = vec![0.0; hw.nodes.len()];
    for &i in ch.order.iter().rev() {
        let mut k: Vec<f64> = ch.kids[i].iter().map(|&c| sub[c]).collect();
        k.sort_by(f64::total_cmp);
        sub[i] = own(i) + k.iter().sum::<f64>();
    }
    sub
}

/// Children in arrangement order: a structural signature over the arranged (unit-free, live) content, so neither
/// names (P4) nor units and dead blocks (P6) reorder what is placed.
fn arrangement_kids(hw: &HwModel, ch: &Characterized, arr: &[f64], in_unit: &[bool]) -> Vec<Vec<usize>> {
    let nn = hw.nodes.len();
    let fnv = |h: u64, x: u64| (h ^ x).wrapping_mul(0x0000_0100_0000_01B3);
    let mut sig = vec![0u64; nn];
    for &i in ch.order.iter().rev() {
        let n = &hw.nodes[i];
        let kind = match n.ix {
            NodeIx::Container(_) => 1,
            NodeIx::Unit(_) => 2,
            NodeIx::Mem(_) => 3,
            NodeIx::Block(_) => 4,
            NodeIx::Router(_) => 5,
            NodeIx::Port(_) => 6,
            NodeIx::Net(_) => 7,
        };
        let kids: Vec<usize> = ch.kids[i].iter().copied().filter(|&k| !in_unit[k] && arr[k] > 0.0).collect();
        let mut h = fnv(fnv(0xcbf2_9ce4_8422_2325, kind), arr[i].to_bits());
        h = fnv(h, u64::from(n.index));
        for c in &n.coord {
            h = fnv(h, u64::from(*c));
        }
        let mut cs: Vec<u64> = kids.iter().map(|&c| sig[c]).collect();
        cs.sort_unstable();
        for c in cs {
            h = fnv(h, c);
        }
        sig[i] = h;
    }
    (0..nn)
        .map(|i| {
            let mut k: Vec<usize> = ch.kids[i].iter().copied().filter(|&k| !in_unit[k] && arr[k] > 0.0).collect();
            k.sort_by_key(|&c| (sig[c], hw.nodes[c].index, ch.rank[c]));
            k
        })
        .collect()
}

fn grid_dims(n: usize, w: f64, h: f64, declared: Option<(usize, usize)>) -> (usize, usize) {
    if let Some((r, c)) = declared.filter(|(r, c)| r * c >= n) {
        let keep = ((h / r as f64) / (w / c as f64)).ln().abs();
        let flip = ((h / c as f64) / (w / r as f64)).ln().abs();
        return if flip < keep { (c, r) } else { (r, c) };
    }
    let mut best = (1, n.max(1));
    let mut score = f64::INFINITY;
    for r in 1..=n.max(1) {
        let c = n.div_ceil(r);
        let dummy = (r * c - n) as f64;
        let s = ((h / r as f64) / (w / c as f64)).ln().abs() + if dummy > 0.05 * n as f64 { 1.0 + dummy / n as f64 } else { 0.0 };
        if s < score - 1e-12 {
            score = s;
            best = (r, c);
        }
    }
    best
}

/// Lays out the children of `node` inside `r` (area-proportional slicing over entity groups; arrays as grids).
fn layout(hw: &HwModel, akids: &[Vec<usize>], sub: &[f64], node: usize, r: Rect, rect: &mut [Option<Rect>]) {
    rect[node] = Some(r);
    let kids: Vec<usize> = akids[node].iter().copied().filter(|&k| sub[k] > 0.0 && !matches!(hw.nodes[k].ix, NodeIx::Net(_))).collect();
    if kids.is_empty() {
        return;
    }
        let mut groups: Vec<(String, Vec<usize>)> = vec![];
    for &k in &kids {
        let e = &hw.nodes[k].entity;
        match groups.iter_mut().find(|g| &g.0 == e) {
            Some(g) => g.1.push(k),
            None => groups.push((e.clone(), vec![k])),
        }
    }
    let mut areas: Vec<f64> = groups.iter().map(|g| g.1.iter().map(|&k| sub[k]).sum()).collect();
    let content: f64 = areas.iter().sum();
    let total = sub[node].max(1e-9);
    // Shrink to the content share (own logic and nested routers take the rest, spread uniformly).
    let f = (content / total).clamp(1e-6, 1.0).sqrt();
    let inner = Rect::new(r.cx() - 0.5 * r.w() * f, r.cy() - 0.5 * r.h() * f, r.cx() + 0.5 * r.w() * f, r.cy() + 0.5 * r.h() * f);
    areas.iter_mut().for_each(|a| *a = a.max(1e-9));
    let rects = slice_seq(&areas, inner);
    for (gi, (_, members)) in groups.iter().enumerate() {
        let gr = rects[gi];
        if members.len() == 1 {
            layout(hw, akids, sub, members[0], gr, rect);
            continue;
        }
        let coords: Vec<&Vec<u32>> = members.iter().map(|&m| &hw.nodes[m].coord).collect();
        let declared = if coords.iter().all(|c| c.len() == 2) {
            Some((coords.iter().map(|c| c[0] as usize).max().unwrap_or(0) + 1, coords.iter().map(|c| c[1] as usize).max().unwrap_or(0) + 1))
        } else {
            None
        };
        let (rows, cols) = grid_dims(members.len(), gr.w(), gr.h(), declared);
        let transposed = declared.is_some_and(|d| d != (rows, cols));
        for (i, &m) in members.iter().enumerate() {
            let (ri, ci) = match declared {
                Some(_) => {
                    let c = &hw.nodes[m].coord;
                    if transposed { (c[1] as usize, c[0] as usize) } else { (c[0] as usize, c[1] as usize) }
                }
                None => (i / cols, i % cols),
            };
            let (cw, chh) = (gr.w() / cols as f64, gr.h() / rows as f64);
            let cell = Rect::new(gr.x0 + ci as f64 * cw, gr.y0 + ri as f64 * chh, gr.x0 + (ci + 1) as f64 * cw, gr.y0 + (ri + 1) as f64 * chh);
            layout(hw, akids, sub, m, cell, rect);
        }
    }
}

/// Area-proportional recursive halving of `areas` (in order) over `r`, cutting the longer side.
fn slice_seq(areas: &[f64], r: Rect) -> Vec<Rect> {
    let mut out = vec![Rect::default(); areas.len()];
    fn go(areas: &[f64], off: usize, r: Rect, out: &mut [Rect]) {
        if areas.len() == 1 {
            out[off] = r;
            return;
        }
        let total: f64 = areas.iter().sum();
        let mut acc = 0.0;
        let mut k = 1;
        for (i, a) in areas.iter().enumerate().take(areas.len() - 1) {
            acc += a;
            k = i + 1;
            if acc >= 0.5 * total {
                break;
            }
        }
        let fa = areas[..k].iter().sum::<f64>() / total;
        let (a, b) = if r.w() >= r.h() { r.split_v(fa) } else { r.split_h(fa) };
        go(&areas[..k], off, a, out);
        go(&areas[k..], off + k, b, out);
    }
    go(areas, 0, r, &mut out);
    out
}

pub struct FloorplanInput<'a> {
    pub hw: &'a HwModel,
    pub ch: &'a Characterized,
    pub params: &'a Params,
    pub tier: PlaceTier,
    pub seed: u64,
}

/// A memory stack, its footprint (w along the edge, h) and its PHY macro `(die, macro)`.
type StackSite = (usize, f64, f64, Option<(usize, usize)>);

pub fn build(inp: &FloorplanInput) -> Floorplan {
    let t0 = Instant::now();
    let FloorplanInput { hw, ch, params, tier, seed } = *inp;
    let t = Tables::get();
    let nn = hw.nodes.len();
    let sub = subtree_areas(hw, ch);
    let live = structural_live(hw);
    // Geometry (06 P6): dead nodes take no place at all. Die arrangements are computed without compute-unit area
    // (`arr`) and then scaled up by the unit area (sqrt of the live/arranged envelope ratio), so a unit no op can use
    // (an int8 array under a bf16 workload) can only stretch every die-level distance; inside a cluster, wires
    // scale with the cluster's live area (links.rs). Both are monotone in added area.
    let geo = live_subtree_areas(hw, ch, &live, None);
    let in_unit = unit_subtrees(hw, ch);
    let arr = live_subtree_areas(hw, ch, &live, Some(&in_unit));
    let akids = arrangement_kids(hw, ch, &arr, &in_unit);
    // Arrangement rank: preorder over the arranged tree; the rest (units, dead nodes) after it in canonical order.
    let arank: Vec<usize> = {
        let mut r: Vec<usize> = ch.rank.iter().map(|x| x + nn).collect();
        let mut st: Vec<usize> = ch.order.first().copied().into_iter().collect();
        let mut k = 0;
        while let Some(x) = st.pop() {
            r[x] = k;
            k += 1;
            st.extend(akids[x].iter().rev().copied());
        }
        r
    };
    let u_place = params.get("u_place", None);
    let keep = params.get("corner_keepout_mm", None) * 1000.0;
    let tol = params.get("area_overflow_tol", None);
    let aspect = (params.get("aspect_min", None), params.get("aspect_max", None));
    let mut problems = vec![];
    let mut rect: Vec<Option<Rect>> = vec![None; nn];
    let mut die_of: Vec<Option<usize>> = vec![None; nn];
    let mut dies = vec![];
    let mut stats = PlaceStats::default();
    let phy_of_node: BTreeMap<usize, &PhyUse> = ch.phys.iter().map(|p| (p.node, p)).collect();
    // Channel bandwidth between die-level macros (through routers) for the placement weights.
    for ci in 0..hw.tree.len() {
        if hw.tree[ci].kind != ContainerKind::Die {
            continue;
        }
        let dnode = hw.tree[ci].node;
        let tech = ch.tech[dnode].clone();
        let node_t = t.node(&tech).or_else(|| t.node("tsmc_n7")).expect("N7");
        let seal = node_t.seal_ring_um;
        let die_spec = hw.tree[ci].die.clone();
        let mut stack = vec![dnode];
        while let Some(x) = stack.pop() {
            die_of[x] = Some(ci);
            stack.extend(ch.kids[x].iter().copied());
        }
        // Macros: die children grouped by entity (arrays collapse); PHY-bearing subtrees are edge macros per instance.
        let mut macros: Vec<Macro> = vec![];
        let own = ch.nodes[dnode].area_um2;
        if own > 0.0 {
            macros.push(Macro { nodes: vec![dnode], name: "die_misc".into(), area_um2: own, power_hint_w: 0.0, phy: None });
        }
        let phys_in = |n: usize| -> Vec<&PhyUse> {
            let mut v = vec![];
            let mut st = vec![n];
            while let Some(x) = st.pop() {
                if let Some(p) = phy_of_node.get(&x) {
                    v.push(*p);
                }
                st.extend(ch.kids[x].iter().copied());
            }
            v
        };
        let mut noc_um2 = 0.0;
        for &k in &akids[dnode] {
            if arr[k] <= 0.0 {
                continue;
            }
            if matches!(hw.nodes[k].ix, NodeIx::Net(_)) {
                noc_um2 += arr[k];
                continue;
            }
            let pu = phys_in(k);
            let leak: f64 = {
                let mut s = 0.0;
                let mut st = vec![k];
                while let Some(x) = st.pop() {
                    s += ch.nodes[x].leak_w - ch.nodes[x].unit_ctrl_leak_w;
                    st.extend(akids[x].iter().copied());
                }
                s
            };
            if !pu.is_empty() && pu.iter().any(|p| p.shoreline_um > 0.0) {
                let shore: f64 = pu.iter().map(|p| p.shoreline_um).sum();
                macros.push(Macro { nodes: vec![k], name: hw.nodes[k].inst_id.clone(), area_um2: arr[k], power_hint_w: leak, phy: Some((Edge::S, shore, pu[0].kind)) });
                continue;
            }
            let e = &hw.nodes[k].entity;
            match macros.iter_mut().find(|m| m.phy.is_none() && !m.nodes.is_empty() && &hw.nodes[m.nodes[0]].entity == e && m.nodes[0] != dnode) {
                Some(m) => {
                    m.nodes.push(k);
                    m.area_um2 += arr[k];
                    m.power_hint_w += leak;
                }
                None => macros.push(Macro { nodes: vec![k], name: hw.nodes[k].entity_id.clone(), area_um2: arr[k], power_hint_w: leak, phy: None }),
            }
        }
        if noc_um2 > 0.0 {
            macros.push(Macro { nodes: vec![], name: "noc".into(), area_um2: noc_um2, power_hint_w: 0.0, phy: None });
        }
        // Area consumers for diagnostics: every die child's full subtree (dead and unit area included).
        let mut consumers: Vec<(String, f64)> = vec![("die_misc".into(), ch.nodes[dnode].area_um2)];
        for &k in &ch.kids[dnode] {
            let name = &hw.nodes[k].entity_id;
            match consumers.iter_mut().find(|c| &c.0 == name) {
                Some(c) => c.1 += sub[k],
                None => consumers.push((name.clone(), sub[k])),
            }
        }
        // Geometry from live blocks; the area roll-up (envelope, checks, report) from every block incl. dead ones.
        let geo_um2: f64 = macros.iter().map(|m| m.area_um2).sum();
        let blocks_um2 = sub[dnode];
        let dead_um2 = (blocks_um2 - geo[dnode]).max(0.0) / u_place;
        let env_of = |b: f64| ((b / u_place).sqrt() + 2.0 * seal).powi(2);
        let env = env_of(geo_um2);
        let env_full = env_of(blocks_um2);
        // Shoreline sites (01 floorplan) in declaration order, else the greedy rule (04 §6.4 step 1).
        let mut sites: Vec<(Edge, IoKind)> = vec![];
        if let Some(d) = &die_spec {
            for s in &d.floorplan.shoreline {
                for _ in 0..s.rep.instances() {
                    sites.push((s.edge, s.kind));
                }
            }
        }
        let mut phy_order: Vec<usize> = (0..macros.len()).filter(|&i| macros[i].phy.is_some()).collect();
        phy_order.sort_by(|&a, &b| {
            let (sa, sb) = (macros[a].phy.expect("phy").1, macros[b].phy.expect("phy").1);
            sb.total_cmp(&sa).then(arank[macros[a].nodes[0]].cmp(&arank[macros[b].nodes[0]]))
        });
        let mut site_used = vec![false; sites.len()];
        let mut edge_need = [0.0f64; 4];
        let mut hbm_need = [0.0f64; 4];
        let (mut w, mut h, fixed) = match die_spec.as_ref().map(|d| &d.floorplan.outline) {
            Some(Outline::Fixed { w, h }) => (w.0, h.0, true),
            Some(Outline::MaxArea { aspect: Some((lo, _)), .. }) => {
                let a = lo.max(aspect.0).min(1.0);
                ((env / a).sqrt(), (env * a).sqrt(), false)
            }
            _ => (env.sqrt(), env.sqrt(), false),
        };
        // An auto outline stays inside the 26 x 33 mm field when an aspect within bounds allows it.
        if !fixed && w > 26_000.0 && env / 26_000.0 <= 33_000.0 && env / 26_000.0 / 26_000.0 <= aspect.1 {
            w = 26_000.0;
            h = env / w;
        }
        for &mi in &phy_order {
            let (_, shore, kind) = macros[mi].phy.expect("phy");
            let site = sites.iter().enumerate().position(|(si, s)| !site_used[si] && site_kind_matches(s.1, kind));
            let e = match site {
                Some(si) => {
                    site_used[si] = true;
                    sites[si].0
                }
                None => {
                    let lens = [w, h, w, h];
                    let mut best = Edge::S;
                    let mut room = f64::NEG_INFINITY;
                    for e in EDGES {
                        let r = lens[edge_ix(e)] - edge_need[edge_ix(e)];
                        if r > room + 1e-9 {
                            room = r;
                            best = e;
                        }
                    }
                    best
                }
            };
            edge_need[edge_ix(e)] += shore;
            if kind == IoKind::Hbm {
                hbm_need[edge_ix(e)] += shore;
            }
            if let Some(p) = macros[mi].phy.as_mut() {
                p.0 = e;
            }
        }
        let mut shoreline_limited = false;
        let need = |i: usize| if edge_need[i] > 0.0 { edge_need[i] + 2.0 * keep } else { 0.0 };
        // Unit area stretches the arranged die uniformly (an auto outline; a fixed one already holds it).
        let stretch = if fixed { 1.0 } else { (env_of(geo[dnode]) / env).max(1.0).sqrt() };
        if !fixed {
            let need_w = need(0).max(need(2));
            let need_h = need(1).max(need(3));
            if need_w > w {
                w = need_w;
                shoreline_limited = true;
            }
            if need_h > h {
                h = need_h;
                shoreline_limited = true;
            }
            let full = (w * h * stretch * stretch + dead_um2).max(env_full);
            if let Some(Outline::MaxArea { area, .. }) = die_spec.as_ref().map(|d| &d.floorplan.outline)
                && full > area.0 * 1e6 * (1.0 + tol)
            {
                problems.push(overflow(&hw.nodes[dnode].path, full, area.0 * 1e6, &consumers));
            }
        } else {
            if env_full > w * h * (1.0 + tol) {
                problems.push(overflow(&hw.nodes[dnode].path, env_full, w * h, &consumers));
            }
            for (i, e) in EDGES.iter().enumerate() {
                let len = if i.is_multiple_of(2) { w } else { h };
                if need(i) > len * (1.0 + 1e-9) {
                    problems.push(
                        Diagnostic::error("E-PHYS-SHORELINE", format!("{} edge {e:?}: PHY shoreline {:.1} mm (incl. corner keep-outs) exceeds the {:.1} mm edge", hw.nodes[dnode].path, need(i) / 1000.0, len / 1000.0))
                            .at(&hw.nodes[dnode].path)
                            .hint("move PHYs to another edge (floorplan.shoreline), use fewer/denser PHYs, or enlarge the outline"),
                    );
                }
            }
        }
        let reticle = params.get("reticle_mm2", None) * 1e6;
        let stitched = die_spec.as_ref().is_some_and(|d| d.stitched);
        // Reticle on the full die: unit area and dead area grow an auto outline in proportion.
        let grow = if fixed { 1.0 } else { ((w * h * stretch * stretch + dead_um2) / (w * h).max(1.0)).sqrt() };
        let (lo, hi) = (w.min(h) * grow, w.max(h) * grow);
        if !stitched && (lo > 26_000.0 * (1.0 + 1e-9) || hi > 33_000.0 * (1.0 + 1e-9) || env_full > reticle) {
            problems.push(
                Diagnostic::error("E-PHYS-RETICLE", format!("{} is {:.1} x {:.1} mm ({:.0} mm^2 envelope), beyond the 26 x 33 mm reticle", hw.nodes[dnode].path, w * grow / 1000.0, h * grow / 1000.0, env_full / 1e6))
                    .at(&hw.nodes[dnode].path)
                    .hint("split the die into chiplets, cut area, or rebalance PHYs across edges"),
            );
        }
        // PHY strips along the edges (depth = area / shoreline), core macros inside the inset rectangle.
        let outline = Rect::new(0.0, 0.0, w, h);
        let mut rects = vec![Rect::default(); macros.len()];
        let mut depth = [0.0f64; 4];
        let mut cursor = [keep; 4];
        let mut edges = [EdgeUse::default(), EdgeUse::default(), EdgeUse::default(), EdgeUse::default()];
        for (i, e) in edges.iter_mut().enumerate() {
            e.len_um = if i.is_multiple_of(2) { w } else { h };
            e.used_um = edge_need[i];
            e.hbm_used_um = hbm_need[i];
        }
        let scale_e: Vec<f64> = (0..4).map(|i| if need(i) > edges[i].len_um { (edges[i].len_um - 2.0 * keep).max(1.0) / edge_need[i] } else { 1.0 }).collect();
        for &mi in &phy_order {
            let (e, shore, _) = macros[mi].phy.expect("phy");
            let ei = edge_ix(e);
            let len = shore * scale_e[ei];
            let d = (macros[mi].area_um2 / len.max(1.0)).min(0.25 * if ei.is_multiple_of(2) { h } else { w });
            depth[ei] = depth[ei].max(d);
            let c0 = cursor[ei];
            cursor[ei] += len;
            rects[mi] = match e {
                Edge::S => Rect::new(seal + c0, seal, seal + c0 + len, seal + d),
                Edge::N => Rect::new(seal + c0, h - seal - d, seal + c0 + len, h - seal),
                Edge::W => Rect::new(seal, seal + c0, seal + d, seal + c0 + len),
                Edge::E => Rect::new(w - seal - d, seal + c0, w - seal, seal + c0 + len),
            };
        }
        let core_r = Rect::new(seal + depth[3], seal + depth[0], w - seal - depth[1], h - seal - depth[2]);
        let core_ix: Vec<usize> = (0..macros.len()).filter(|&i| macros[i].phy.is_none()).collect();
        let mut prob = Problem {
            area: core_ix.iter().map(|&i| macros[i].area_um2).collect(),
            power: core_ix.iter().map(|&i| macros[i].power_hint_w).collect(),
            edges: vec![],
            terms: vec![],
            region: core_r,
            aspect: (0.5, 2.0),
            lambda_th: 0.1,
        };
        // Weights: declared channel bandwidth between macros (routers contracted onto their endpoints).
        let macro_of = {
            let mut v = vec![usize::MAX; nn];
            for (mi, m) in macros.iter().enumerate() {
                for &top in &m.nodes {
                    let mut st = vec![top];
                    while let Some(x) = st.pop() {
                        v[x] = mi;
                        st.extend(ch.kids[x].iter().copied());
                    }
                }
            }
            v
        };
        let mut pair: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut router_ends: BTreeMap<usize, BTreeMap<usize, f64>> = BTreeMap::new();
        let mut chans: Vec<usize> = (0..hw.channels.len()).collect();
        chans.sort_by_key(|&c| (arank[hw.node_of(hw.channels[c].src)], arank[hw.node_of(hw.channels[c].dst)], c));
        for &ci2 in &chans {
            let chn = &hw.channels[ci2];
            let (s, d) = (hw.node_of(chn.src), hw.node_of(chn.dst));
            let bw = chn.bandwidth.map_or(1e9, |b| b.0) * 1e-9;
            let (ms, md) = (macro_of[s], macro_of[d]);
            match (hw.nodes[s].ix, hw.nodes[d].ix) {
                (NodeIx::Router(r), _) if md != usize::MAX => *router_ends.entry(r).or_default().entry(md).or_default() += bw,
                (_, NodeIx::Router(r)) if ms != usize::MAX => *router_ends.entry(r).or_default().entry(ms).or_default() += bw,
                _ if ms != usize::MAX && md != usize::MAX && ms != md => *pair.entry((ms.min(md), ms.max(md))).or_default() += bw,
                _ => {}
            }
        }
        for ends in router_ends.values() {
            let v: Vec<(usize, f64)> = ends.iter().map(|(a, b)| (*a, *b)).collect();
            let k = v.len().max(2) as f64;
            for i in 0..v.len() {
                for j in i + 1..v.len() {
                    *pair.entry((v[i].0.min(v[j].0), v[i].0.max(v[j].0))).or_default() += v[i].1.min(v[j].1) / (k - 1.0);
                }
            }
        }
        let core_pos: BTreeMap<usize, usize> = core_ix.iter().enumerate().map(|(a, &i)| (i, a)).collect();
        for (&(a, b), &wgt) in &pair {
            match (core_pos.get(&a), core_pos.get(&b)) {
                (Some(&x), Some(&y)) => prob.edges.push((x, y, wgt)),
                (Some(&x), None) => prob.terms.push((x, rects[b].cx(), rects[b].cy(), wgt)),
                (None, Some(&y)) => prob.terms.push((y, rects[a].cx(), rects[a].cy(), wgt)),
                _ => {}
            }
        }
        let (expr, ra): (Vec<Tok>, Vec<Rect>) = if prob.area.is_empty() { (vec![], vec![]) } else { place::tier_a(&prob) };
        stats.phi_tier_a += if ra.is_empty() { 0.0 } else { place::phi(&prob, &ra) };
        let rb = match tier {
            PlaceTier::B if prob.area.len() >= 3 => {
                let moves = (400 * prob.area.len()).min(20_000);
                let (_, r, s) = place::tier_b(&prob, &expr, seed ^ ci as u64, moves);
                stats.anneal = Some(s);
                r
            }
            _ => ra,
        };
        stats.phi += if rb.is_empty() { 0.0 } else { place::phi(&prob, &rb) };
        stats.macros += macros.len();
        for (a, &i) in core_ix.iter().enumerate() {
            rects[i] = rb[a];
        }
        let (outline, w, h) = if stretch > 1.0 {
            for r in rects.iter_mut() {
                *r = Rect::new(r.x0 * stretch, r.y0 * stretch, r.x1 * stretch, r.y1 * stretch);
            }
            for e in edges.iter_mut() {
                e.len_um *= stretch;
            }
            (Rect::new(0.0, 0.0, w * stretch, h * stretch), w * stretch, h * stretch)
        } else {
            (outline, w, h)
        };
        let mut legal = (geo[dnode] - arr[dnode]).max(0.0) / u_place;
        for &i in &core_ix {
            let r = rects[i];
            let asp = if r.w() > 0.0 { r.h() / r.w() } else { 1.0 };
            let infl = if macros[i].nodes.len() > 1 { 1.0 } else { (asp / 2.0).max(0.5 / asp).max(1.0) };
            legal += r.area().max(macros[i].area_um2 / u_place * infl);
        }
        legal += macros.iter().filter(|m| m.phy.is_some()).map(|m| m.area_um2 / u_place).sum::<f64>();
        legal += w * h - (w - 2.0 * seal) * (h - 2.0 * seal) + dead_um2;
        dies.push(DieFp {
            container: ci,
            node: dnode,
            path: hw.nodes[dnode].path.clone(),
            tech,
            area_env_um2: env_full,
            blocks_um2,
            outline,
            fixed_outline: fixed,
            shoreline_limited,
            edges,
            macros,
            rects,
            layer: 0,
            legalized_um2: legal,
        });
    }
    // Package frame: pinned dies at their offsets, others in a row with room for the stacks on their edges;
    // upper-layer dies over their base.
    let mut packages = vec![];
    let mut die_origin: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for pci in 0..hw.tree.len() {
        if hw.tree[pci].kind != ContainerKind::Package {
            continue;
        }
        let pnode = hw.tree[pci].node;
        let pkg = hw.tree[pci].package.clone();
        let table = pkg.as_ref().map_or("organic", |p| package_table(p.substrate.kind)).to_owned();
        let pt = t.package.get(&table).expect("package table");
        let gap = pt.die_gap_mm * 1000.0;
        let hbm_gap = pt.hbm_gap_mm * 1000.0;
        let mut layer_of: BTreeMap<String, i32> = BTreeMap::new();
        if let Some(p) = &pkg {
            for l in &p.layers {
                layer_of.insert(l.id.to_string(), l.index);
            }
        }
        let mut mine: Vec<usize> = (0..dies.len()).filter(|&d| hw.nodes[dies[d].node].parent == Some(pnode)).collect();
        mine.sort_by_key(|&d| arank[dies[d].node]);
        for &d in &mine {
            let spec = hw.tree[dies[d].container].die.as_ref();
            dies[d].layer = spec.and_then(|s| s.layer.as_ref()).and_then(|l| layer_of.get(l.as_str()).copied()).unwrap_or(0);
        }
        // HBM/DRAM stacks: footprint outside the edge of the PHY they attach to, else to the right of the dies.
        let mut used_phy: Vec<(usize, usize)> = vec![];
        let mut mems: Vec<usize> = (0..hw.memories.len()).collect();
        mems.sort_by_key(|&m| arank[hw.memories[m].node]);
        // Pass 1: each stack's footprint and the PHY site it attaches to (channel-bound first, then harvested
        // stacks take free HBM sites); pass 2 places them edge by edge in PHY order, in their die's frame.
        let mut entries: Vec<StackSite> = vec![];
        for &mi in &mems {
            let m = &hw.memories[mi];
            let MemSpec::Stack(s) = &m.spec else { continue };
            if hw.nodes[m.node].parent != Some(pnode) && !is_under(hw, m.node, pnode) {
                continue;
            }
            let kind = serde_json::to_value(s.kind).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_else(|| "custom".into());
            let d = t.dram.get(&kind).or_else(|| t.dram.get("custom")).expect("dram");
            let (fw, fh) = (s.footprint.as_ref().and_then(|f| f.w).map_or(d.footprint_w_mm * 1000.0, |w| w.0), s.footprint.as_ref().and_then(|f| f.h).map_or(d.footprint_h_mm * 1000.0, |h| h.0));
            if fw <= 0.0 || fh <= 0.0 {
                continue;
            }
            let mut stack_chans: Vec<usize> = (0..hw.channels.len()).filter(|&c| hw.node_of(hw.channels[c].src) == m.node || hw.node_of(hw.channels[c].dst) == m.node).collect();
            stack_chans.sort_by_key(|&c| (arank[hw.node_of(hw.channels[c].src)], arank[hw.node_of(hw.channels[c].dst)]));
            let phy = stack_chans.iter().map(|&c| &hw.channels[c]).find_map(|c| {
                let (a, b) = (hw.node_of(c.src), hw.node_of(c.dst));
                let other = if a == m.node { b } else if b == m.node { a } else { return None };
                phy_of_node.get(&other).map(|_| other)
            });
            let bound = phy.and_then(|pn| {
                let di = (0..dies.len()).find(|&d| die_of[pn] == Some(dies[d].container))?;
                let mi2 = dies[di].macros.iter().position(|mm| mm.phy.is_some() && mm.nodes.iter().any(|&top| top == pn || is_under(hw, pn, top)))?;
                Some((di, mi2))
            });
            if let Some(x) = bound {
                used_phy.push(x);
            }
            entries.push((mi, fw, fh, bound));
        }
        for e in entries.iter_mut().filter(|e| e.3.is_none()) {
            e.3 = mine.iter().find_map(|&di| {
                let mi2 = (0..dies[di].macros.len()).find(|&k| dies[di].macros[k].phy.is_some_and(|p| p.2 == IoKind::Hbm) && !used_phy.contains(&(di, k)))?;
                Some((di, mi2))
            });
            if let Some(x) = e.3 {
                used_phy.push(x);
            }
        }
        let along = |e: &StackSite| -> (usize, usize, f64) {
            e.3.map_or((usize::MAX, 0, 0.0), |(di, mi2)| {
                let ei = dies[di].macros[mi2].phy.map_or(1, |p| edge_ix(p.0));
                let r = dies[di].rects[mi2];
                (di, ei, if ei.is_multiple_of(2) { r.cx() } else { r.cy() })
            })
        };
        entries.sort_by(|x, y| {
            let (a, b) = (along(x), along(y));
            a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)).then(arank[hw.memories[x.0].node].cmp(&arank[hw.memories[y.0].node]))
        });
        // Bound stacks in their die's frame; each die's extent with its stacks reserves its place in the row.
        let mut edge_cursor: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut local: Vec<Option<(usize, Rect)>> = vec![None; entries.len()];
        let mut extent: BTreeMap<usize, Rect> = mine.iter().map(|&d| (d, dies[d].outline)).collect();
        for (k, &(mi, fw, fh, place_at)) in entries.iter().enumerate() {
            let Some((di, mi2)) = place_at else { continue };
            let pr = dies[di].rects[mi2];
            let o = dies[di].outline;
            let e = dies[di].macros[mi2].phy.map_or(Edge::E, |p| p.0);
            let ei = edge_ix(e);
            let cur = edge_cursor.entry((di, ei)).or_insert(f64::NEG_INFINITY);
            // Stacks on one edge stack side by side (width along the edge = footprint w).
            let r = match e {
                Edge::W => Rect::new(o.x0 - hbm_gap - fh, (pr.cy() - 0.5 * fw).max(*cur), o.x0 - hbm_gap, (pr.cy() - 0.5 * fw).max(*cur) + fw),
                Edge::E => Rect::new(o.x1 + hbm_gap, (pr.cy() - 0.5 * fw).max(*cur), o.x1 + hbm_gap + fh, (pr.cy() - 0.5 * fw).max(*cur) + fw),
                Edge::S => Rect::new((pr.cx() - 0.5 * fw).max(*cur), o.y0 - hbm_gap - fh, (pr.cx() - 0.5 * fw).max(*cur) + fw, o.y0 - hbm_gap),
                Edge::N => Rect::new((pr.cx() - 0.5 * fw).max(*cur), o.y1 + hbm_gap, (pr.cx() - 0.5 * fw).max(*cur) + fw, o.y1 + hbm_gap + fh),
            };
            *cur = if ei.is_multiple_of(2) { r.x1 + 0.1 * hbm_gap } else { r.y1 + 0.1 * hbm_gap };
            // 04 §6.3: stacks on one side may overhang the die corners by at most overhang_max.
            let span = if ei.is_multiple_of(2) { (r.x0, r.x1, o.x0, o.x1) } else { (r.y0, r.y1, o.y0, o.y1) };
            let over = pt.overhang_max_mm * 1000.0;
            if span.1 > span.3 + over + 1e-6 || span.0 < span.2 - over - 1e-6 {
                let path = &hw.nodes[hw.memories[mi].node].path;
                problems.push(
                    Diagnostic::error("E-PHYS-SHORELINE", format!("{path}: memory stacks along edge {e:?} of {} reach {:.1} mm along a {:.1} mm edge (overhang limit {:.1} mm per corner)", dies[di].path, (span.1 - span.2) / 1000.0, (span.3 - span.2) / 1000.0, over / 1000.0))
                        .at(path)
                        .hint("spread stacks over more edges or use fewer, higher-capacity stacks"),
                );
            }
            local[k] = Some((di, r));
            if let Some(x) = extent.get_mut(&di) {
                *x = Rect::new(x.x0.min(r.x0), x.y0.min(r.y0), x.x1.max(r.x1), x.y1.max(r.y1));
            }
        }
        // A die's west stacks keep the HBM gap from what precedes them, its east stacks from what follows.
        let lead = |d: usize| if extent[&d].x0 < 0.0 { hbm_gap.max(gap) } else { gap };
        let trail = |d: usize| if extent[&d].x1 > dies[d].outline.x1 { hbm_gap.max(gap) } else { gap };
        let pinned = |d: usize| matches!(hw.tree[dies[d].container].die.as_ref().map(|s| &s.placement), Some(Placement::Pinned { .. }));
        // The row of unpinned base dies starts beside the pinned ones.
        let mut x = 0.0f64;
        let mut x_gap = 0.0f64;
        for &d in &mine {
            if let Some(Placement::Pinned { x: px, .. }) = hw.tree[dies[d].container].die.as_ref().map(|s| &s.placement)
                && dies[d].layer == 0
            {
                x = x.max(px.0 + extent[&d].x1);
                x_gap = x_gap.max(trail(d));
            }
        }
        let mut first = x == 0.0 && x_gap == 0.0;
        for &d in &mine {
            let o = match hw.tree[dies[d].container].die.as_ref().map(|s| &s.placement) {
                _ if dies[d].layer > 0 => (f64::NAN, f64::NAN),
                Some(Placement::Pinned { x: px, y: py, .. }) => (px.0, py.0),
                _ => {
                    let start = if first { x } else { x + x_gap.max(lead(d)) };
                    first = false;
                    let o = (start - extent[&d].x0, 0.0);
                    x = o.0 + extent[&d].x1;
                    x_gap = trail(d);
                    o
                }
            };
            die_origin.insert(d, o);
        }
        // Upper-layer dies sit over a base die of the same instance index (else the first base die): centred, or at
        // their pinned offset from it (01 `over`: placement is relative to the die below).
        for &d in &mine {
            if die_origin[&d].0.is_nan() {
                let idx = hw.nodes[dies[d].node].index as usize;
                let bases: Vec<usize> = mine.iter().copied().filter(|&b| dies[b].layer == 0).collect();
                let b = bases.iter().copied().find(|&b| hw.nodes[dies[b].node].index as usize == idx).or_else(|| bases.first().copied()).unwrap_or(d);
                let (bx, by) = die_origin.get(&b).copied().unwrap_or((0.0, 0.0));
                let (bw, bh) = (dies[b].outline.w(), dies[b].outline.h());
                let o = match hw.tree[dies[d].container].die.as_ref().map(|s| &s.placement) {
                    Some(Placement::Pinned { x: px, y: py, .. }) => (bx + px.0, by + py.0),
                    _ => (bx + 0.5 * (bw - dies[d].outline.w()), by + 0.5 * (bh - dies[d].outline.h())),
                };
                die_origin.insert(d, o);
            }
        }
        let mut bbox: Option<Rect> = None;
        let grow = |r: Rect, b: &mut Option<Rect>| {
            *b = Some(match *b {
                None => r,
                Some(o) => Rect::new(o.x0.min(r.x0), o.y0.min(r.y0), o.x1.max(r.x1), o.y1.max(r.y1)),
            });
        };
        for &d in &mine {
            let (ox, oy) = die_origin[&d];
            grow(dies[d].outline.shift(ox, oy), &mut bbox);
        }
        let mut stacks = vec![];
        for (k, &(mi, fw, fh, _)) in entries.iter().enumerate() {
            let m = &hw.memories[mi];
            let r = match local[k] {
                Some((di, r)) => {
                    let (ox, oy) = die_origin.get(&di).copied().unwrap_or((0.0, 0.0));
                    r.shift(ox, oy)
                }
                None => {
                    let b = bbox.unwrap_or_default();
                    Rect::new(b.x1 + hbm_gap, b.y0, b.x1 + hbm_gap + fw, b.y0 + fh)
                }
            };
            rect[m.node] = Some(r);
            grow(r, &mut bbox);
            stacks.push((mi, r));
        }
        // Components of one package layer may not share area (memory stacks sit on the base layer).
        let parts: Vec<(i32, Rect, &str, bool)> = mine
            .iter()
            .map(|&d| (dies[d].layer, dies[d].outline.shift(die_origin[&d].0, die_origin[&d].1), dies[d].path.as_str(), pinned(d)))
            .chain(stacks.iter().map(|&(mi, r)| (0, r, hw.nodes[hw.memories[mi].node].path.as_str(), false)))
            .collect();
        for (i, a) in parts.iter().enumerate() {
            for b in &parts[i + 1..] {
                let o = Rect::new(a.1.x0.max(b.1.x0), a.1.y0.max(b.1.y0), a.1.x1.min(b.1.x1), a.1.y1.min(b.1.y1)).area();
                if a.0 == b.0 && o > 1e-6 * a.1.area().min(b.1.area()) {
                    problems.push(
                        Diagnostic::error("E-PHYS-PACKAGE-OVERLAP", format!("{} and {} overlap by {:.2} mm^2 on package layer {}", a.2, b.2, o / 1e6, a.0))
                            .at(&hw.nodes[pnode].path)
                            .hint(if a.3 || b.3 { "move the pinned dies apart (at least the package die gap) or onto different layers" } else { "spread the memory stacks over more edges or pin the dies apart" }),
                    );
                }
            }
        }
        let mut outline = bbox.unwrap_or_default();
        if outline.area() > pt.max_mm2 * 1e6 * (1.0 + 1e-9) {
            problems.push(
                Diagnostic::error("E-PHYS-PACKAGE-OVERFLOW", format!("{}: package outline {:.0} mm^2 exceeds the {} limit {:.0} mm^2", hw.nodes[pnode].path, outline.area() / 1e6, table, pt.max_mm2))
                    .at(&hw.nodes[pnode].path)
                    .hint("use fewer dies/stacks or a larger package technology (CoWoS-L, organic)"),
            );
        }
        // A declared substrate outline bounds the package, and a fixed one is the package.
        let declared = pkg.as_ref().map(|p| &p.substrate.outline);
        let (bw, bh) = (outline.w(), outline.h());
        let over = match declared {
            Some(Outline::Fixed { w, h }) => {
                let (fw, fh) = (w.0.max(h.0), w.0.min(h.0));
                let fits = bw.max(bh) <= fw * (1.0 + 1e-9) && bw.min(bh) <= fh * (1.0 + 1e-9);
                let (ow, oh) = if bw >= bh { (fw, fh) } else { (fh, fw) };
                outline = Rect::new(outline.x0, outline.y0, outline.x0 + ow.max(bw), outline.y0 + oh.max(bh));
                (!fits).then(|| format!("{:.1} x {:.1} mm fixed substrate", w.0 / 1000.0, h.0 / 1000.0))
            }
            Some(Outline::MaxArea { area, .. }) => (bw * bh > area.0 * 1e6 * (1.0 + 1e-9)).then(|| format!("{:.0} mm^2 substrate limit", area.0)),
            _ => None,
        };
        if let Some(lim) = over {
            problems.push(
                Diagnostic::error("E-PHYS-PACKAGE-OVERFLOW", format!("{}: dies and stacks span {:.1} x {:.1} mm, beyond the declared {lim}", hw.nodes[pnode].path, bw / 1000.0, bh / 1000.0))
                    .at(&hw.nodes[pnode].path)
                    .hint("enlarge the substrate outline or use fewer dies/stacks"),
            );
        }
        packages.push(PackageFp { container: pci, path: hw.nodes[pnode].path.clone(), table, outline, stacks, max_mm2: pt.max_mm2 });
    }
    // Hierarchical layout inside every macro (package frame).
    for (d, die) in dies.iter().enumerate() {
        let (ox, oy) = die_origin.get(&d).copied().unwrap_or((0.0, 0.0));
        rect[die.node] = Some(die.outline.shift(ox, oy));
        for (mi, m) in die.macros.iter().enumerate() {
            let r = die.rects[mi].shift(ox, oy);
            if m.nodes.len() == 1 {
                if m.nodes[0] != die.node {
                    layout(hw, &akids, &arr, m.nodes[0], r, &mut rect);
                }
                continue;
            }
            let coords: Vec<&Vec<u32>> = m.nodes.iter().map(|&x| &hw.nodes[x].coord).collect();
            let declared = if coords.iter().all(|c| c.len() == 2) {
                Some((coords.iter().map(|c| c[0] as usize).max().unwrap_or(0) + 1, coords.iter().map(|c| c[1] as usize).max().unwrap_or(0) + 1))
            } else {
                None
            };
            let (rows, cols) = grid_dims(m.nodes.len(), r.w(), r.h(), declared);
            let transposed = declared.is_some_and(|dd| dd != (rows, cols));
            for (i, &x) in m.nodes.iter().enumerate() {
                let (ri, cj) = match declared {
                    Some(_) => {
                        let c = &hw.nodes[x].coord;
                        if transposed { (c[1] as usize, c[0] as usize) } else { (c[0] as usize, c[1] as usize) }
                    }
                    None => (i / cols, i % cols),
                };
                let (cw, chh) = (r.w() / cols as f64, r.h() / rows as f64);
                layout(hw, &akids, &arr, x, Rect::new(r.x0 + cj as f64 * cw, r.y0 + ri as f64 * chh, r.x0 + (cj + 1) as f64 * cw, r.y0 + (ri + 1) as f64 * chh), &mut rect);
            }
        }
    }
    // Positions: rectangles; routers and other footprint-less nodes at the centroid of their neighbours.
    let mut pos: Vec<(f64, f64)> = rect.iter().map(|r| r.map_or((f64::NAN, f64::NAN), |r| (r.cx(), r.cy()))).collect();
    let mut by_rank: Vec<usize> = (0..nn).collect();
    by_rank.sort_by_key(|&i| arank[i]);
    for _ in 0..3 {
        for &i in &by_rank {
            if !matches!(hw.nodes[i].ix, NodeIx::Router(_)) && rect[i].is_some() {
                continue;
            }
            let (mut sx, mut sy, mut k) = (0.0, 0.0, 0.0);
            let mut outs: Vec<usize> = hw.out_edges[i].iter().map(|&c| hw.node_of(hw.channels[c].dst)).collect();
            outs.sort_by_key(|&o| arank[o]);
            for o in outs {
                if !pos[o].0.is_nan() {
                    sx += pos[o].0;
                    sy += pos[o].1;
                    k += 1.0;
                }
            }
            if k > 0.0 {
                pos[i] = (sx / k, sy / k);
            }
        }
    }
    for i in 0..nn {
        if pos[i].0.is_nan() {
            pos[i] = hw.nodes[i].parent.map_or((0.0, 0.0), |p| pos[p]);
        }
    }
    stats.place_us = t0.elapsed().as_secs_f64() * 1e6;
    Floorplan { tier, dies, packages, rect, pos, die_of, geo_um2: geo, stats, problems }
}

fn is_under(hw: &HwModel, mut n: usize, root: usize) -> bool {
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

fn overflow(path: &str, need: f64, have: f64, consumers: &[(String, f64)]) -> Diagnostic {
    let mut top: Vec<&(String, f64)> = consumers.iter().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let names: Vec<String> = top.iter().take(3).map(|m| format!("{} {:.1} mm^2", m.0, m.1 / 1e6)).collect();
    Diagnostic::error("E-PHYS-AREA-OVERFLOW", format!("{path}: blocks need {:.1} mm^2 (incl. whitespace and seal ring), outline gives {:.1} mm^2 (deficit {:.1} mm^2); largest: {}", need / 1e6, have / 1e6, (need - have) / 1e6, names.join(", ")))
        .at(path)
        .hint("enlarge the outline or remove/shrink the largest consumers")
}
