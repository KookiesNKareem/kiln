//! Link derivation from placement (04 §7.2, §6.3): every channel gets a `LinkCost` from the positions of its
//! endpoints: repeated on-die wires (+ router pipeline and crossbar energy into a router), package traces between
//! die PHYs, SerDes/host PHYs, and vertical bonds.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::model::{Channel, ChannelKind, ContainerKind, HwModel, NodeIx};
use kiln_ir::hw::net::{BondKind, LinkPhys, LinkSpec};

use crate::characterize::{Characterized, PhyUse};
use crate::floorplan::Floorplan;
use crate::params::Params;
use crate::tables::Tables;
use crate::wire::{self, LinkCost, LinkSource};

fn phy_at<'a>(hw: &HwModel, ch: &'a Characterized, mut n: usize) -> Option<&'a PhyUse> {
    if let NodeIx::Port(p) = hw.nodes[n].ix
        && let Some(b) = crate::characterize::bound_phy(hw, &hw.ports[p])
    {
        n = b;
    }
    for _ in 0..3 {
        if let Some(p) = ch.phys.iter().find(|p| p.node == n) {
            return Some(p);
        }
        n = hw.nodes[n].parent?;
    }
    None
}

/// Lowest common ancestor of `a` and `b` when it is a cluster container (below the die), else `None`.
fn cluster_lca(hw: &HwModel, a: usize, b: usize) -> Option<usize> {
    let mut up = vec![];
    let mut x = Some(a);
    while let Some(y) = x {
        up.push(y);
        x = hw.nodes[y].parent;
    }
    let mut x = Some(b);
    while let Some(y) = x {
        if up.contains(&y) {
            return match hw.nodes[y].ix {
                NodeIx::Container(c) if hw.tree[c].kind == ContainerKind::Cluster => Some(y),
                _ => None,
            };
        }
        x = hw.nodes[y].parent;
    }
    None
}

pub(crate) fn bond_table(t: &'static Tables, b: BondKind) -> Option<&'static crate::tables::PhyT> {
    t.phy.get(match b {
        BondKind::Hybrid => "hybrid_bond",
        BondKind::Microbump => "microbump",
        BondKind::Tsv => "tsv",
    })
}

fn die_at(fp: &Floorplan, n: usize) -> Option<usize> {
    fp.die_of[n].and_then(|c| fp.dies.iter().position(|d| d.container == c))
}

fn is_stack(hw: &HwModel, n: usize) -> bool {
    matches!(hw.nodes[n].ix, NodeIx::Mem(m) if hw.memories[m].is_stack())
}

/// Bond kind and declared pitch of a vertical channel: its declared vertical link, else the bond of the upper die's
/// package layer, else hybrid.
fn vertical_bond(hw: &HwModel, fp: &Floorplan, c: &Channel) -> (BondKind, f64) {
    let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
    let declared = link_spec(hw, c).or_else(|| [s, d].into_iter().find_map(|x| match hw.nodes[x].ix {
        NodeIx::Port(p) => hw.ports[p].spec.link.as_ref(),
        _ => None,
    }));
    if let Some(LinkPhys::Vertical(v)) = declared.map(|l| &l.phys) {
        return (v.bond, v.pitch_um.0);
    }
    let layer_bond = |d: usize| {
        let c = fp.dies[d].container;
        let lid = hw.tree[c].die.as_ref()?.layer.as_ref()?;
        let pkg = std::iter::successors(hw.nodes[hw.tree[c].node].parent, |&x| hw.nodes[x].parent).find_map(|x| match hw.nodes[x].ix {
            NodeIx::Container(p) => hw.tree[p].package.as_ref(),
            _ => None,
        })?;
        pkg.layers.iter().find(|l| l.id.as_str() == lid.as_str()).map(|l| (l.bond, l.pitch_um.0))
    };
    [s, d].into_iter().filter(|&n| !is_stack(hw, n)).filter_map(|n| die_at(fp, n)).max_by_key(|&x| fp.dies[x].layer).and_then(layer_bond).unwrap_or((BondKind::Hybrid, 0.0))
}

/// The link spec channel `c` was expanded from: its own (a custom edge's or an endpoint port's), else its network's
/// main link, else its source port's.
fn link_spec<'a>(hw: &'a HwModel, c: &'a Channel) -> Option<&'a LinkSpec> {
    c.link.as_deref().or_else(|| c.network.map(|n| &hw.networks[n].spec.link)).or_else(|| match hw.nodes[hw.node_of(c.src)].ix {
        NodeIx::Port(p) => hw.ports[p].spec.link.as_ref(),
        _ => None,
    })
}

pub fn derive(hw: &HwModel, ch: &Characterized, fp: &Floorplan, params: &Params) -> Vec<LinkCost> {
    let t = Tables::get();
    let s45 = t.datapath.c("s_45_n7");
    let kw = params.get("kappa_e_wire", None);
    let pkg_table = |n: usize| -> &'static crate::tables::PackageT {
        let pk = fp.packages.iter().find(|p| {
            let mut x = Some(n);
            while let Some(y) = x {
                if y == hw.tree[p.container].node {
                    return true;
                }
                x = hw.nodes[y].parent;
            }
            false
        });
        t.package.get(pk.map_or("organic", |p| p.table.as_str())).expect("package table")
    };
    hw.channels
        .iter()
        .map(|c| {
            let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
            let (ps, pd) = (fp.pos[s], fp.pos[d]);
            let len = match cluster_lca(hw, s, d) {
                // Inside a cluster: half the side of the cluster's live area, which dead blocks cannot shrink (P6).
                Some(c) => 0.5 * fp.geo_um2[c].max(0.0).sqrt(),
                None => (ps.0 - pd.0).abs() + (ps.1 - pd.1).abs(),
            };
            let f = c.clock.and_then(|k| hw.clocks.get(k)).map_or(1e9, |k| k.spec.freq.0);
            let bw = c.bandwidth.map_or(0.0, |b| b.0);
            let node = t.node(&ch.tech[s]).or_else(|| t.node("tsmc_n7")).expect("N7");
            let phy_link = |kind_src: LinkSource, extra_len: f64| -> LinkCost {
                let p = phy_at(hw, ch, s).or_else(|| phy_at(hw, ch, d));
                let (e_bit, lat) = p.map_or((5e-12, 50e-9), |p| (p.e_j_per_bit, p.latency_s));
                let pt = pkg_table(s);
                let trace = extra_len.max(0.0);
                LinkCost {
                    class: None,
                    length_um: trace,
                    latency_cycles: ((lat + trace * pt.trace_ps_per_mm * 1e-15) * f).ceil().max(1.0) as u32,
                    latency_s: lat + trace * pt.trace_ps_per_mm * 1e-15,
                    pipeline_stages: 0,
                    bw_bytes_per_s: bw,
                    e_j_per_byte: 8.0 * (e_bit + trace * 1e-3 * pt.trace_e_pj_per_bit_mm * 1e-12),
                    e_j_per_cycle_idle: 0.0,
                    source: kind_src,
                }
            };
            let stack = |x: usize| is_stack(hw, x);
            let mut lc = match c.kind {
                // Stack <-> PHY: the DRAM and PHY energy and DRAM latency are charged at the stack (04 §7.3).
                _ if stack(s) || stack(d) => LinkCost {
                    class: None,
                    length_um: len,
                    latency_cycles: 0,
                    latency_s: 0.0,
                    pipeline_stages: 0,
                    bw_bytes_per_s: bw,
                    e_j_per_byte: 0.0,
                    e_j_per_cycle_idle: 0.0,
                    source: LinkSource::Phy,
                },
                ChannelKind::D2d => phy_link(LinkSource::Package, len),
                ChannelKind::Serdes | ChannelKind::Optical => phy_link(LinkSource::Phy, 0.0),
                ChannelKind::Host => phy_link(LinkSource::Host, 0.0),
                ChannelKind::Vertical => {
                    let pt = bond_table(t, vertical_bond(hw, fp, c).0).expect("bond table");
                    LinkCost {
                        class: None,
                        length_um: 0.0,
                        latency_cycles: 1,
                        latency_s: (pt.latency_ns * 1e-9).max(1.0 / f),
                        pipeline_stages: 0,
                        bw_bytes_per_s: bw,
                        e_j_per_byte: 8.0 * pt.e_pj_per_bit * 1e-12,
                        e_j_per_cycle_idle: 0.0,
                        source: LinkSource::Bond3d,
                    }
                }
                _ => {
                    let width = c.width_bits.map_or(bw * 8.0 / f, f64::from).max(1.0);
                    let mut lc = wire::on_die(node, s45, len.max(1.0), width, f, kw);
                    lc.bw_bytes_per_s = bw;
                    if c.kind == ChannelKind::Near {
                        lc.latency_cycles = 1;
                        lc.latency_s = 1.0 / f;
                    }
                    if link_spec(hw, c).is_some_and(|l| matches!(l.phys, LinkPhys::OnDie { swing: kiln_ir::hw::net::Swing::Low, .. })) {
                        // Low-swing: E = a_t c V_swing V_dd L + e_rx (04 §7.2).
                        let w = node.wire(lc.class.unwrap_or(crate::tables::WireClassId::SemiGlobal));
                        lc.e_j_per_byte = 8.0 * (0.25 * w.c_ff_um * 1e-15 * 0.2 * node.vdd_nom * len + 20e-15) * kw;
                    }
                    if let NodeIx::Router(r) = hw.nodes[d].ix {
                        let ro = &ch.routers[r];
                        lc.latency_cycles += ro.n_pipe;
                        lc.latency_s += f64::from(ro.n_pipe) / f;
                        lc.e_j_per_byte += ro.e_flit_j / (ro.flit_bits / 8.0).max(1.0);
                        lc.source = LinkSource::NocHop;
                    }
                    lc
                }
            };
            // Declared link latency/energy (reference designs) only ever make a link slower or costlier.
            if let Some(l) = link_spec(hw, c) {
                if let Some(x) = l.latency {
                    lc.latency_s = lc.latency_s.max(x.0);
                }
                if let Some(x) = l.energy {
                    lc.e_j_per_byte = lc.e_j_per_byte.max(x.0);
                }
            }
            lc
        })
        .collect()
}

/// 04 §6.3: the vertical links between two footprints need one signal pad per wire (each direction its own), and
/// the bond supplies `overlap / pitch^2 * bond_signal_frac` of them at the coarser of the declared and the table
/// pitch. A vertically attached memory stack sits on its die (its footprint is the overlap).
pub fn bond_problems(hw: &HwModel, fp: &Floorplan) -> Vec<Diagnostic> {
    let t = Tables::get();
    let die_at = |n: usize| die_at(fp, n);
    let stack_at = |n: usize| is_stack(hw, n).then_some(n);
    // (footprint a, footprint b) -> (signals, overlap um^2, pitch um, signal fraction); a footprint is a die index or
    // a stack's arena node (offset past the dies).
    let mut pairs: BTreeMap<(usize, usize), (f64, f64, f64, f64)> = BTreeMap::new();
    for c in hw.channels.iter().filter(|c| c.kind == ChannelKind::Vertical) {
        let (s, d) = (hw.node_of(c.src), hw.node_of(c.dst));
        let at = |n: usize| stack_at(n).map(|x| fp.dies.len() + x).or_else(|| die_at(n));
        let (Some(a), Some(b)) = (at(s), at(d)) else { continue };
        let (bond, pitch) = vertical_bond(hw, fp, c);
        let Some(pt) = bond_table(t, bond) else { continue };
        let pitch = pitch.max(pt.pitch_um.unwrap_or(0.0)).max(1e-3);
        let frac = pt.signal_frac.unwrap_or(0.5);
        let rect = |x: usize| if x < fp.dies.len() { fp.rect[fp.dies[x].node] } else { fp.rect[x - fp.dies.len()] };
        let overlap = match (rect(a), rect(b)) {
            (Some(ra), Some(rb)) if a >= fp.dies.len() || b >= fp.dies.len() => ra.area().min(rb.area()),
            (Some(ra), Some(rb)) => crate::place::Rect::new(ra.x0.max(rb.x0), ra.y0.max(rb.y0), ra.x1.min(rb.x1), ra.y1.min(rb.y1)).area(),
            _ => 0.0,
        };
        let f = c.clock.and_then(|k| hw.clocks.get(k)).map_or(1e9, |k| k.spec.freq.0);
        let signals = f64::from(c.width_bits.unwrap_or(0)).max(c.bandwidth.map_or(0.0, |b| b.0) * 8.0 / f);
        let e = pairs.entry((a.min(b), a.max(b))).or_insert((0.0, overlap, pitch, frac));
        e.0 += signals;
        (e.2, e.3) = (e.2.max(pitch), e.3.min(frac));
    }
    let name = |x: usize| if x < fp.dies.len() { fp.dies[x].path.clone() } else { hw.nodes[x - fp.dies.len()].path.clone() };
    pairs
        .into_iter()
        .filter_map(|((a, b), (signals, overlap, pitch, frac))| {
            let pads = overlap / (pitch * pitch) * frac;
            (signals > pads * (1.0 + 1e-9)).then(|| {
                Diagnostic::error(
                    "E-PHYS-BOND-CAPACITY",
                    format!("{} <-> {}: vertical links need {signals:.0} signals; the {:.2} mm^2 overlap at {pitch} um pitch carries {pads:.0} ({frac} of the pads)", name(a), name(b), overlap / 1e6),
                )
                .at(name(a))
                .hint("narrow the vertical links, enlarge the overlap of the two footprints, or use a finer-pitch bond")
            })
        })
        .collect()
}
