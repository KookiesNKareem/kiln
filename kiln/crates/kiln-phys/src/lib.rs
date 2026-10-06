//! Physical model (04): technology tables, block area/energy models, floorplan and placers, wire model, area
//! roll-up, power (leakage, clock, DVFS, board), thermal and calibration. [`Phys`] is the interface Tier A
//! consumes: link and memory costs per channel and memory, per-event energies, clock plans and the 03 §4.5 clock
//! solve. [`Phys::new`] builds the M3 model; [`Phys::legacy`] keeps the M1 constants for comparisons.

pub mod blocks;
pub mod calib;
pub mod characterize;
pub mod data;
pub mod floorplan;
pub mod links;
pub mod params;
pub mod place;
pub mod power;
pub mod pricing;
pub mod report;
pub mod sourced;
pub mod tables;
pub mod thermal;
pub mod wire;

use std::sync::Arc;
use std::time::Instant;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::MemKind;
use kiln_ir::hw::model::{ChanIx, ChannelKind, ClockIx, MemIx, MemSpec, ResIx, UnitIx};
use kiln_ir::hw::phys::PowerPolicy;
use serde::{Deserialize, Serialize};

pub use characterize::Characterized;
pub use floorplan::{Floorplan, PlaceTier};
pub use params::{PCorner, Params};
pub use power::{CapDomain, PhaseEnergy, PowerBreakdown, PowerModel, ThermalZone, VfTable};
pub use report::{DieReport, PhysReport};
pub use wire::LinkCost;

pub const MODEL_ID: &str = "kiln-phys/m3";
pub const LEGACY_MODEL_ID: &str = "kiln-phys/legacy-stub-0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// Read or computed from IR fields.
    Derived,
    /// Placeholder constant; uncalibrated.
    Assumed,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkPhys {
    pub bandwidth_bps: f64,
    pub latency_s: f64,
    pub energy_j_per_b: f64,
    pub bandwidth_source: Source,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemPhys {
    pub bandwidth_bps: f64,
    pub latency_s: f64,
    pub read_j_per_b: f64,
    pub write_j_per_b: f64,
    pub dram: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockMode {
    /// Every domain at its nominal (maximum) frequency.
    Nominal,
    /// Every domain at `base` where declared, nominal otherwise.
    Base,
    /// Explicit frequency per clock domain (index = `ClockIx`).
    Fixed(Vec<f64>),
    /// Power-capped DVFS (03 §4.5): solved from a phase power function by [`Phys::solve_clock`].
    PowerCapped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClockPlan {
    pub hz: Vec<f64>,
    /// False when a power-capped plan fell back to nominal because no power model is available.
    pub solved: bool,
    pub throttled: bool,
}

impl ClockPlan {
    pub fn hz(&self, c: Option<ClockIx>) -> Option<f64> {
        c.and_then(|c| self.hz.get(c).copied())
    }
}

/// Phase power in W of every enforced cap ([`Phys::caps`] order) at the given per-domain clocks.
pub type PowerFn<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;

/// An enforced power cap as the clock solve sees it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapSpec {
    pub path: String,
    pub cap_w: f64,
    /// Clocks under its DVFS control, in the order the solve lowers them.
    pub clocks: Vec<ClockIx>,
}

/// A cap whose members' power stays above it at the selected clocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapExcess {
    pub path: String,
    pub power_w: f64,
    pub cap_w: f64,
}

/// The M3 physical model of one design (cached per design hash by the caller's `HwView`).
#[derive(Debug)]
pub struct Model {
    pub params: Params,
    pub ch: Characterized,
    pub fp: Floorplan,
    pub power: PowerModel,
    pub links: Vec<LinkCost>,
    pub report: PhysReport,
    /// Enforced caps, each with its members' power model.
    pub caps: Vec<CapDomain>,
    /// Per-package junction and per-die density checks (04 §9).
    pub zones: Vec<ThermalZone>,
    /// Enabled MAC units: (clock, MACs/cycle, J/MAC at V_nom, arena node) of their highest-power mode.
    pub mac_peak: Vec<(Option<ClockIx>, f64, f64, usize)>,
}

#[derive(Clone, Debug)]
pub struct Phys {
    pub model: &'static str,
    links: Vec<LinkPhys>,
    resources: Vec<LinkPhys>,
    mems: Vec<MemPhys>,
    nominal_hz: Vec<f64>,
    base_hz: Vec<f64>,
    vf_hz: Vec<Vec<f64>>,
    caps: Vec<CapSpec>,
    cap_w: Option<f64>,
    m3: Option<Arc<Model>>,
}

impl PartialEq for Phys {
    fn eq(&self, o: &Self) -> bool {
        self.model == o.model && self.links == o.links && self.mems == o.mems && self.nominal_hz == o.nominal_hz && self.cap_w == o.cap_w
    }
}

const E_MAC_BF16_J: f64 = 0.3e-12;
const E_VEC_OP_J: f64 = 1.0e-12;
const STATIC_FRACTION_OF_CAP: f64 = 0.2;
const DEFAULT_HOP_S: f64 = 1e-9;

fn clocks_of(hw: &HwModel) -> (Vec<f64>, Vec<f64>, Option<f64>) {
    let nominal_hz: Vec<f64> = hw.clocks.iter().map(|c| c.spec.freq.0).collect();
    let base_hz = hw.clocks.iter().map(|c| c.spec.base.map_or(c.spec.freq.0, |b| b.0)).collect();
    let cap_w = hw.power_domains.iter().filter(|p| p.assumed.is_none()).map(|p| p.cap.0).fold(None, |a: Option<f64>, c| Some(a.map_or(c, |a| a + c)));
    (nominal_hz, base_hz, cap_w)
}

fn resources_from(hw: &HwModel, links: &[LinkPhys]) -> Vec<LinkPhys> {
    let mut by_res: Vec<Vec<usize>> = vec![vec![]; hw.resources.len()];
    for (c, ch) in hw.channels.iter().enumerate() {
        if let Some(v) = by_res.get_mut(ch.resource) {
            v.push(c);
        }
    }
    hw.resources
        .iter()
        .enumerate()
        .map(|(r, res)| {
            let chans: Vec<&LinkPhys> = by_res[r].iter().map(|&c| &links[c]).collect();
            let bw = res.capacity.map_or_else(|| chans.iter().map(|l| l.bandwidth_bps).fold(0.0, f64::max), |c| c.0);
            LinkPhys {
                bandwidth_bps: bw,
                latency_s: chans.iter().map(|l| l.latency_s).fold(0.0, f64::max),
                energy_j_per_b: chans.iter().map(|l| l.energy_j_per_b).fold(0.0, f64::max),
                bandwidth_source: Source::Derived,
            }
        })
        .collect()
}

impl Phys {
    /// Legacy-constant model: bandwidths from IR, latencies and energies assumed.
    pub fn legacy(hw: &HwModel) -> Phys {
        let (nominal_hz, base_hz, cap_w) = clocks_of(hw);
        let vf_hz: Vec<Vec<f64>> = hw
            .clocks
            .iter()
            .map(|c| {
                let mut v: Vec<f64> = c.spec.vf.iter().map(|p| p.freq.0).collect();
                v.sort_by(f64::total_cmp);
                v
            })
            .collect();
        let caps = hw
            .power_domains
            .iter()
            .filter(|p| p.assumed.is_none())
            .map(|p| CapSpec {
                path: p.path.clone(),
                cap_w: p.cap.0,
                clocks: if p.policy != PowerPolicy::Dvfs { vec![] } else if p.clocks.is_empty() { (0..vf_hz.len()).filter(|&c| !vf_hz[c].is_empty()).collect() } else { p.clocks.clone() },
            })
            .collect();
        let cycle = |c: Option<ClockIx>| c.and_then(|c| nominal_hz.get(c)).map_or(DEFAULT_HOP_S, |f| 1.0 / f);
        let links: Vec<LinkPhys> = hw
            .channels
            .iter()
            .map(|ch| {
                let (latency_s, energy) = match ch.kind {
                    ChannelKind::Feed => (cycle(ch.clock), 0.05e-12),
                    ChannelKind::MemPort => (0.0, 0.5e-12),
                    ChannelKind::NocHop | ChannelKind::Bus => (2.0 * cycle(ch.clock), 1.0e-12),
                    ChannelKind::D2d | ChannelKind::Vertical => (10e-9, 2.0e-12),
                    ChannelKind::Serdes | ChannelKind::Optical => (100e-9, 40e-12),
                    ChannelKind::Near => (cycle(ch.clock), 0.2e-12),
                    ChannelKind::Host => (500e-9, 80e-12),
                };
                LinkPhys { bandwidth_bps: ch.bandwidth.map_or(0.0, |b| b.0), latency_s, energy_j_per_b: energy, bandwidth_source: Source::Derived }
            })
            .collect();
        let resources = resources_from(hw, &links);
        let mems = hw
            .memories
            .iter()
            .map(|m| {
                let f = m.clock.and_then(|c| nominal_hz.get(c).copied());
                let cyc = |n: f64| f.map_or(n * DEFAULT_HOP_S, |f| n / f);
                let (latency_s, e, dram) = match &m.spec {
                    MemSpec::Stack(_) => (400e-9, 31e-12, true),
                    MemSpec::Local { .. } => (cyc(1.0), 0.05e-12, false),
                    MemSpec::OnChip(mem) => match mem.kind {
                        MemKind::RegisterFile => (cyc(1.0), 0.1e-12, false),
                        MemKind::Fifo => (cyc(1.0), 0.1e-12, false),
                        MemKind::Scratchpad => (cyc(10.0), if m.capacity.0 > 8 << 20 { 1.5e-12 } else { 0.5e-12 }, false),
                        MemKind::Cache => (cyc(200.0), 1.5e-12, false),
                    },
                };
                MemPhys { bandwidth_bps: m.bandwidth.map_or(0.0, |b| b.0), latency_s, read_j_per_b: e, write_j_per_b: e, dram }
            })
            .collect();
        Phys { model: LEGACY_MODEL_ID, links, resources, mems, nominal_hz, base_hz, vf_hz, caps, cap_w, m3: None }
    }

    /// The M3 model with the design family's calibration sets, Tier A placement.
    pub fn new(hw: &HwModel) -> Phys {
        Phys::with(hw, Params::for_family(hw.family.as_deref()), PlaceTier::A)
    }

    pub fn with(hw: &HwModel, params: Params, tier: PlaceTier) -> Phys {
        let model = Arc::new(build_model(hw, params, tier));
        let (nominal_hz, base_hz, cap_w) = clocks_of(hw);
        let links: Vec<LinkPhys> = hw
            .channels
            .iter()
            .zip(&model.links)
            .map(|(c, l)| LinkPhys { bandwidth_bps: c.bandwidth.map_or(0.0, |b| b.0), latency_s: l.latency_s, energy_j_per_b: l.e_j_per_byte, bandwidth_source: Source::Derived })
            .collect();
        let resources = resources_from(hw, &links);
        let mems = hw
            .memories
            .iter()
            .zip(&model.ch.mems)
            .map(|(m, e)| MemPhys { bandwidth_bps: m.bandwidth.map_or(0.0, |b| b.0), latency_s: e.latency_s, read_j_per_b: e.read_j_per_b, write_j_per_b: e.write_j_per_b, dram: e.dram })
            .collect();
        let vf_hz = model.power.domains.iter().map(|d| d.vf.grid()).collect();
        let caps = model.caps.iter().map(|c| CapSpec { path: c.path.clone(), cap_w: c.cap_w, clocks: c.clocks.clone() }).collect();
        Phys { model: MODEL_ID, links, resources, mems, nominal_hz, base_hz, vf_hz, caps, cap_w, m3: Some(model) }
    }

    /// The M3 model (`None` for the legacy stub).
    pub fn m3(&self) -> Option<&Model> {
        self.m3.as_deref()
    }

    pub fn link(&self, c: ChanIx) -> &LinkPhys {
        &self.links[c]
    }

    /// Shared resource (01 §16 `SharedResource`): capacity, worst member latency and energy.
    pub fn resource(&self, r: ResIx) -> &LinkPhys {
        &self.resources[r]
    }

    pub fn mem(&self, m: MemIx) -> &MemPhys {
        &self.mems[m]
    }

    pub fn nominal_hz(&self, c: ClockIx) -> f64 {
        self.nominal_hz[c]
    }

    /// Dynamic energy of one MAC with operand widths `bits_a x bits_b` (legacy: quadratic in width).
    pub fn e_mac_j(&self, bits_a: u32, bits_b: u32) -> f64 {
        E_MAC_BF16_J * f64::from(bits_a) * f64::from(bits_b) / 256.0
    }

    pub fn e_vector_op_j(&self) -> f64 {
        E_VEC_OP_J
    }

    /// Per-event energies of unit `u` for a mapped nest in `mode` (MAC `a*b+acc`, elementwise `dtype@rate`):
    /// `(J per useful MAC, J per padding MAC, J per elementwise op, J per transcendental op)` at V_nom.
    pub fn unit_energies(&self, u: UnitIx, mode: &str, bits: (u32, u32)) -> (f64, f64, f64, f64) {
        match &self.m3 {
            Some(m) => {
                let ue = &m.ch.units[u];
                let mac = m.ch.e_mac(u, mode, bits);
                (mac, mac * ue.idle_frac, m.ch.e_elem(u, mode), ue.e_transc_j)
            }
            None => {
                let mac = self.e_mac_j(bits.0, bits.1);
                (mac, 0.1 * mac, E_VEC_OP_J, E_VEC_OP_J)
            }
        }
    }

    /// `(V(f)/V_nom)^2` for energies charged in clock domain `c` at `plan` by arena node `node` (at its technology's
    /// V_nom; the domain's without a node); 1 for the legacy model, clock-independent resources and DRAM/PHY energies.
    pub fn dyn_scale(&self, c: Option<ClockIx>, node: Option<usize>, plan: &ClockPlan) -> f64 {
        match (&self.m3, c) {
            (Some(m), Some(c)) if c < m.power.domains.len() => {
                let hz = plan.hz.get(c).copied().unwrap_or(self.nominal_hz[c]);
                match node.and_then(|n| m.ch.tech.get(n)) {
                    Some(tech) => m.power.dyn_scale_tech(c, hz, tech),
                    None => m.power.dyn_scale(c, hz),
                }
            }
            _ => 1.0,
        }
    }

    /// Chip static power (legacy: assumed fraction of the cap; M3: leakage at nominal clocks and 85 C).
    pub fn static_power_w(&self) -> f64 {
        match &self.m3 {
            Some(m) => m.power.p_static(&self.nominal_hz, 85.0),
            None => self.cap_w.map_or(0.0, |c| c * STATIC_FRACTION_OF_CAP),
        }
    }

    /// Phase power at `plan` (board, package and die terms; 04 §8).
    pub fn phase_power(&self, e: &PhaseEnergy, plan: &ClockPlan) -> Option<PowerBreakdown> {
        self.m3.as_ref().map(|m| m.power.power(e, &plan.hz))
    }

    /// The enforced caps, summed (reported as the design's TDP).
    pub fn power_cap_w(&self) -> Option<f64> {
        self.cap_w
    }

    /// The enforced caps; each one bounds its own members' power.
    pub fn caps(&self) -> &[CapSpec] {
        &self.caps
    }

    /// The caps `power` (one value per cap, [`Phys::caps`] order) exceeds.
    pub fn cap_excess(&self, power: &[f64]) -> Vec<CapExcess> {
        self.caps
            .iter()
            .zip(power)
            .filter(|(c, p)| p.is_nan() || **p > c.cap_w * (1.0 + 1e-6))
            .map(|(c, &p)| CapExcess { path: c.path.clone(), power_w: p, cap_w: c.cap_w })
            .collect()
    }

    pub fn clock_plan(&self, mode: &ClockMode) -> ClockPlan {
        let hz = match mode {
            ClockMode::Nominal | ClockMode::PowerCapped => self.nominal_hz.clone(),
            ClockMode::Base => self.base_hz.clone(),
            ClockMode::Fixed(v) => v.clone(),
        };
        ClockPlan { hz, solved: !matches!(mode, ClockMode::PowerCapped), throttled: false }
    }

    /// 03 §4.5 hook: for every cap its members' power exceeds, the largest V/f grid point of each clock under its
    /// DVFS control (in order, never above the current one) with the cap met, by bisection (power is monotone in
    /// f). `power` is supplied by the engine; `None` (no power model) keeps nominal clocks and reports
    /// `solved = false`. A cap still exceeded at its clocks' floors is returned with its power (E-MAP-POWER-CAP).
    pub fn solve_clock(&self, power: Option<PowerFn>) -> (ClockPlan, Vec<CapExcess>) {
        let Some(power) = power.filter(|_| !self.caps.is_empty()) else {
            return (self.clock_plan(&ClockMode::PowerCapped), vec![]);
        };
        let mut hz = self.nominal_hz.clone();
        let mut p = power(&hz);
        for (k, cap) in self.caps.iter().enumerate() {
            for &c in &cap.clocks {
                if p[k] <= cap.cap_w {
                    break;
                }
                let table = &self.vf_hz[c];
                let (mut lo, mut hi) = (0usize, table.partition_point(|&f| f <= hz[c] * (1.0 + 1e-12)));
                if hi == 0 {
                    continue;
                }
                while lo + 1 < hi {
                    let mid = (lo + hi) / 2;
                    hz[c] = table[mid];
                    if power(&hz)[k] <= cap.cap_w { lo = mid } else { hi = mid }
                }
                hz[c] = table[lo];
                p = power(&hz);
            }
        }
        let throttled = hz != self.nominal_hz;
        (ClockPlan { hz, solved: true, throttled }, self.cap_excess(&p))
    }

    /// MAC-unit dynamic power at full activity at `plan` (the denominator of the clock-gating activity).
    /// Only the MAC units in `nodes` when given.
    pub fn peak_compute_w(&self, plan: &ClockPlan, nodes: Option<&[bool]>) -> f64 {
        let Some(m) = &self.m3 else { return 0.0 };
        m.mac_peak
            .iter()
            .filter(|x| nodes.is_none_or(|n| n[x.3]))
            .map(|&(c, mpc, e, node)| {
                let f = c.and_then(|c| plan.hz.get(c).copied()).unwrap_or(1e9);
                mpc * f * e * self.dyn_scale(c, Some(node), plan)
            })
            .sum()
    }

    /// Share of a byte's energy at DRAM memory `m` spent in its die-side PHY (the rest is DRAM core + IO on the
    /// board); 0 for on-chip memories.
    pub fn dram_phy_fraction(&self, m: MemIx) -> f64 {
        let Some(x) = self.m3.as_ref().and_then(|x| x.ch.mems.get(m)).filter(|x| x.dram) else { return 0.0 };
        if x.read_j_per_b > 0.0 { x.e_phy_j_per_b / x.read_j_per_b } else { 0.0 }
    }

    pub fn report(&self) -> Option<&PhysReport> {
        self.m3.as_ref().map(|m| &m.report)
    }

    /// E-PHYS-* findings of the model (area overflow, reticle, shoreline, package, missing node tables).
    pub fn problems(&self) -> Vec<Diagnostic> {
        self.m3.as_ref().map_or_else(Vec::new, |m| m.report.problems.clone())
    }
}

fn build_model(hw: &HwModel, params: Params, tier: PlaceTier) -> Model {
    let t0 = Instant::now();
    let ch = characterize::characterize(hw, &params);
    let t_char = t0.elapsed().as_secs_f64() * 1e6;
    let seed = u64::from_str_radix(hw.design_hash.trim_start_matches("hw1-").get(..16).unwrap_or("0"), 16).unwrap_or(0);
    let fp = floorplan::build(&floorplan::FloorplanInput { hw, ch: &ch, params: &params, tier, seed });
    let t1 = Instant::now();
    let links = links::derive(hw, &ch, &fp, &params);
    let t_links = t1.elapsed().as_secs_f64() * 1e6;
    let clocked: Vec<f64> = ch.nodes.iter().map(|n| n.area_um2 - n.parts[characterize::Part::Phy as usize]).collect();
    let dies: Vec<(usize, f64)> = fp.dies.iter().map(|d| (d.node, d.area_env_um2 / 1e6)).collect();
    let pipes = power::pipes(hw, &links);
    let power = PowerModel::build(hw, &ch, &params, &dies, &clocked, &pipes);
    let report = report::build(hw, &ch, &fp, &power, &params, (t_char, fp.stats.place_us, t_links));
    let mac_peak = hw
        .units
        .iter()
        .enumerate()
        .filter(|(_, u)| u.spec.kind.is_mac() && hw.nodes[u.node].enabled)
        .filter_map(|(ui, u)| {
            u.spec
                .precisions
                .iter()
                .map(|m| {
                    let key = match m {
                        kiln_ir::hw::compute::PrecisionMode::Mac { a, b, acc, .. } => format!("{a}*{b}+{acc}"),
                        kiln_ir::hw::compute::PrecisionMode::Elem { dtype, .. } => dtype.to_string(),
                    };
                    (u.spec.kind.ops_per_cycle(m), ch.e_mac(ui, &key, (16, 16)))
                })
                .max_by(|a, b| (a.0 * a.1).total_cmp(&(b.0 * b.1)))
                .map(|(mpc, e)| (u.clock, mpc, e, u.node))
        })
        .collect();
    let caps = CapDomain::build_all(hw, &ch, &params, &dies, &clocked, &pipes);
    let zones = ThermalZone::build_all(hw, &ch, &params, &dies, &clocked, &pipes);
    Model { params, ch, fp, power, links, report, caps, zones, mac_peak }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_ir::hw::{Profile, check_file};

    fn a100() -> HwModel {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../designs/reference/a100_sxm4_40gb.json5");
        check_file(p, Profile::Reference).model.expect("a100 expands")
    }

    #[test]
    fn bandwidths_come_from_ir() {
        let hw = a100();
        for ph in [Phys::legacy(&hw), Phys::new(&hw)] {
            for (i, c) in hw.channels.iter().enumerate() {
                assert_eq!(ph.link(i).bandwidth_bps, c.bandwidth.map_or(0.0, |b| b.0));
            }
            let hbm: f64 = hw.memories.iter().enumerate().filter(|(_, m)| m.is_stack() && hw.nodes[m.node].enabled).map(|(i, _)| ph.mem(i).bandwidth_bps).sum();
            assert!((hbm - 1555.2e9).abs() < 1e6);
            assert!(ph.mem(hw.memories.iter().position(|m| m.is_stack()).unwrap()).dram);
        }
    }

    #[test]
    fn clock_modes() {
        let hw = a100();
        let ph = Phys::legacy(&hw);
        let gpc = hw.clocks.iter().position(|c| c.path.ends_with("gpc_clk")).unwrap();
        assert_eq!(ph.clock_plan(&ClockMode::Nominal).hz[gpc], 1.41e9);
        assert_eq!(ph.clock_plan(&ClockMode::Base).hz[gpc], 1.095e9);
        let (unsolved, _) = ph.solve_clock(None);
        assert!(!unsolved.solved && unsolved.hz[gpc] == 1.41e9);
        let p = |hz: &[f64]| vec![100.0 + 320.0 * hz[gpc] / 1.41e9];
        let (s, over) = ph.solve_clock(Some(&p));
        assert!(s.solved && s.throttled && s.hz[gpc] == 1.29e9 && over.is_empty());
        let lo = |hz: &[f64]| vec![50.0 + hz[gpc] * 0.0];
        assert!(!ph.solve_clock(Some(&lo)).0.throttled);
        // The M3 model bisects a 32-point grid between base and boost: the largest grid clock under the cap.
        let m3 = Phys::new(&hw);
        let (s, _) = m3.solve_clock(Some(&p));
        let want = (400.0 - 100.0) / 320.0 * 1.41e9;
        assert!(s.throttled && s.hz[gpc] <= want && s.hz[gpc] > want - (1.41e9 - 1.095e9) / 31.0, "{}", s.hz[gpc]);
    }
}
