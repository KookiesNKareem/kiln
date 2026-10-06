//! E-IR-UNPRICED support (00 decision 3, 01 §18.3): the derived value of every "less is better" override kiln-phys
//! can price, computed on the design with its overrides removed, so `search` validation can allow overrides at
//! >= derived and reject the rest.

use std::sync::Arc;

use kiln_ir::hw::compute::MemOverrides;
use kiln_ir::hw::model::{HwModel, MemSpec, NodeIx};
use kiln_ir::hw::net::Topology;
use kiln_ir::hw::phys::PowerOverride;
use kiln_ir::hw::{PricedField, Pricer};

use crate::characterize::Part;
use crate::floorplan::PlaceTier;
use crate::{Model, Params};

/// The design with every priced override removed (derived values only).
pub fn strip_overrides(hw: &HwModel) -> HwModel {
    let mut h = hw.clone();
    for u in &mut h.units {
        let mut s = (*u.spec).clone();
        s.power = PowerOverride { source: s.power.source.clone(), ..Default::default() };
        u.spec = Arc::new(s);
    }
    for b in &mut h.blocks {
        let mut s = (*b.spec).clone();
        s.power = PowerOverride { source: s.power.source.clone(), ..Default::default() };
        b.spec = Arc::new(s);
    }
    for n in &mut h.networks {
        let mut s = (*n.spec).clone();
        s.router.power = PowerOverride::default();
        let edges = match &mut s.topology {
            Topology::Custom { edges, .. } => edges.iter_mut().filter_map(|e| e.link.as_mut()).collect(),
            _ => vec![],
        };
        for l in std::iter::once(&mut s.link).chain(s.endpoints.iter_mut().filter_map(|e| e.port.as_mut())).chain(edges) {
            l.latency = None;
            l.energy = None;
        }
        n.spec = Arc::new(s);
    }
    for m in &mut h.memories {
        match &mut m.spec {
            MemSpec::OnChip(s) => {
                let mut x = (**s).clone();
                x.overrides = MemOverrides { bandwidth: x.overrides.bandwidth, source: x.overrides.source.clone(), ..Default::default() };
                *s = Arc::new(x);
            }
            MemSpec::Stack(s) => {
                let mut x = (**s).clone();
                x.overrides.energy_per_byte = None;
                x.overrides.power = None;
                *s = Arc::new(x);
            }
            MemSpec::Local { .. } => {}
        }
    }
    h
}

pub struct PhysPricer {
    hw: HwModel,
    m: Model,
}

impl PhysPricer {
    pub fn new(hw: &HwModel) -> PhysPricer {
        let stripped = strip_overrides(hw);
        let m = crate::build_model(&stripped, Params::for_family(hw.family.as_deref()), PlaceTier::A);
        PhysPricer { hw: stripped, m }
    }

    fn net_channels(&self, node: usize) -> Vec<usize> {
        let hw = &self.hw;
        (0..hw.channels.len())
            .filter(|&c| {
                let ch = &hw.channels[c];
                ch.network.is_some_and(|n| hw.networks[n].node == node) || hw.node_of(ch.src) == node || hw.node_of(ch.dst) == node
            })
            .collect()
    }
}

/// Factory for [`kiln_ir::hw::check_priced`].
pub fn pricer(hw: &HwModel) -> Box<dyn Pricer> {
    Box::new(PhysPricer::new(hw))
}

impl Pricer for PhysPricer {
    fn derived(&self, f: &PricedField) -> Option<f64> {
        let hw = &self.hw;
        let ch = &self.m.ch;
        let np = ch.nodes.get(f.node)?;
        let ix = hw.nodes.get(f.node)?.ix;
        let cyc = |c: Option<usize>| c.and_then(|c| hw.clocks.get(c)).map_or(1e9, |c| c.spec.freq.0);
        match (f.field, ix) {
            ("power.area", NodeIx::Net(n)) => Some(hw.networks[n].routers.iter().map(|&r| ch.routers[r].area_um2).sum()),
            ("power.area", NodeIx::Container(_)) => self.m.fp.dies.iter().find(|d| d.node == f.node).map(|d| d.area_env_um2),
            ("power.area", _) => Some(np.area_um2),
            ("power.leakage", NodeIx::Net(n)) => Some(hw.networks[n].routers.iter().map(|&r| ch.nodes[hw.routers[r].node].leak_w).sum()),
            ("power.leakage", _) => Some(np.leak_w),
            ("power.ctrl_ge", NodeIx::Unit(_)) => {
                let t = crate::tables::Tables::get();
                let n = t.node(&ch.tech[f.node])?;
                Some(np.parts[Part::Control as usize] * self.m.params.util(n) / n.a_ge_um2)
            }
            ("power.energy_per_op", NodeIx::Unit(u)) => {
                let k = f.key.as_deref().unwrap_or_default();
                ch.units[u].modes.iter().find(|(m, _)| m.starts_with(k)).map(|x| x.1)
            }
            ("read_energy", NodeIx::Mem(m)) => Some(ch.mems[m].read_j_per_b),
            ("write_energy", NodeIx::Mem(m)) => Some(ch.mems[m].write_j_per_b),
            ("latency", NodeIx::Mem(m)) => Some(ch.mems[m].latency_s * cyc(hw.memories[m].clock)),
            ("leakage", NodeIx::Mem(_)) => Some(np.leak_w),
            ("area", NodeIx::Mem(_)) => Some(np.area_um2),
            ("overrides.energy_per_byte", NodeIx::Mem(m)) => Some(ch.mems[m].read_j_per_b),
            ("overrides.power", NodeIx::Mem(m)) => Some(ch.mems[m].background_w),
            ("link.latency", _) => self.net_channels(f.node).iter().map(|&c| self.m.links[c].latency_s).reduce(f64::max),
            ("link.energy", _) => self.net_channels(f.node).iter().map(|&c| self.m.links[c].e_j_per_byte).reduce(f64::max),
            _ => None,
        }
    }
}
