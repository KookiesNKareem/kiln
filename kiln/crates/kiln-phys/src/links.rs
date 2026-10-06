//! Link derivation from placement (04 §7.2, §6.3): every channel gets a `LinkCost` from the positions of its
//! endpoints: repeated on-die wires (+ router pipeline and crossbar energy into a router), package traces between
//! die PHYs, SerDes/host PHYs, and vertical bonds.

use kiln_ir::hw::model::{ChannelKind, ContainerKind, HwModel, NodeIx};
use kiln_ir::hw::net::LinkPhys;

use crate::characterize::{Characterized, PhyUse};
use crate::floorplan::Floorplan;
use crate::params::Params;
use crate::tables::Tables;
use crate::wire::{self, LinkCost, LinkSource};

fn phy_at<'a>(hw: &HwModel, ch: &'a Characterized, mut n: usize) -> Option<&'a PhyUse> {
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
            let stack = |x: usize| matches!(hw.nodes[x].ix, NodeIx::Mem(m) if hw.memories[m].is_stack());
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
                    let pt = t.phy.get("hybrid_bond").expect("bond table");
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
                    if c.network.is_some_and(|n| matches!(hw.networks[n].spec.link.phys, LinkPhys::OnDie { swing: kiln_ir::hw::net::Swing::Low, .. })) {
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
            let spec = c.network.map(|n| &hw.networks[n].spec.link).or_else(|| match hw.nodes[s].ix {
                NodeIx::Port(p) => hw.ports[p].spec.link.as_ref(),
                _ => None,
            });
            if let Some(l) = spec {
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
