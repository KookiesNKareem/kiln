//! Repeated-wire model (04 §7.1, Bakoglu; Ho, Mai, Horowitz; energy-aware sizing per Banerjee & Mehrotra) and the
//! per-link cost 03 consumes (04 §7.2).

use serde::Serialize;

use crate::tables::{TechNode, WireClassId};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Sizing {
    Delay,
    Energy,
    Custom { k_h: f64, k_s: f64 },
}

impl Sizing {
    fn k(self) -> (f64, f64) {
        match self {
            Sizing::Delay => (1.0, 1.0),
            Sizing::Energy => (0.5, 1.5),
            Sizing::Custom { k_h, k_s } => (k_h, k_s),
        }
    }
}

/// Time of flight in an SiO2-class dielectric: 6.7 ps/mm.
pub const TOF_S_PER_UM: f64 = 6.7e-15;

/// Delay per um of a repeated wire: `sqrt(R0 C0 r c) * (sqrt(0.262 (1 + gamma)) (k_s + 1/k_s) + 0.69 (k_h + 1/k_h))`,
/// which is `2.83 sqrt(R0 C0 r c)` at the delay optimum (gamma = 1); floored at the time of flight.
pub fn tau_s_per_um(n: &TechNode, class: WireClassId, sizing: Sizing) -> f64 {
    let w = n.wire(class);
    let (k_h, k_s) = sizing.k();
    let rc = w.r_ohm_um * w.c_ff_um * 1e-15;
    let base = (n.r0c0_ps * 1e-12 * rc).sqrt();
    let f = (0.262 * (1.0 + n.gamma)).sqrt() * (k_s + 1.0 / k_s) + 0.69 * (k_h + 1.0 / k_h);
    (base * f).max(TOF_S_PER_UM)
}

/// Repeater capacitance per um relative to the wire's: `sqrt(0.38/0.69) sqrt(1 + gamma) k_h / k_s`.
pub fn c_rep_ratio(n: &TechNode, sizing: Sizing) -> f64 {
    let (k_h, k_s) = sizing.k();
    (0.38f64 / 0.69).sqrt() * (1.0 + n.gamma).sqrt() * k_h / k_s
}

/// Unrepeated segment length (um) of the chosen sizing (04 §7.1 design rule).
pub fn segment_um(n: &TechNode, class: WireClassId, sizing: Sizing) -> f64 {
    let w = n.wire(class);
    let (_, k_s) = sizing.k();
    let r0c0 = n.r0c0_ps * 1e-12;
    let s_star = (0.69 * r0c0 * (1.0 + n.gamma) / (0.38 * w.r_ohm_um * w.c_ff_um * 1e-15)).sqrt();
    k_s * s_star
}

/// Dynamic energy per bit of `len_um` of wire at `v` (random data, toggle factor 0.25), before kappa_E_wire.
pub fn e_bit(n: &TechNode, class: WireClassId, sizing: Sizing, len_um: f64, v: f64) -> f64 {
    0.25 * n.wire(class).c_ff_um * 1e-15 * (1.0 + c_rep_ratio(n, sizing)) * len_um * v * v
}

/// Flop energy per bit per cycle (pipeline stage) at V_nom, 8 fJ at 45 nm scaled.
pub fn e_ff_bit(n: &TechNode, s_45: f64) -> f64 {
    8e-15 * s_45 * n.s_e
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkSource {
    Wire,
    NocHop,
    Phy,
    Bond3d,
    Package,
    Host,
}

/// 04 §7.2 output per directional link instance (channel).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LinkCost {
    pub class: Option<WireClassId>,
    pub length_um: f64,
    pub latency_cycles: u32,
    pub latency_s: f64,
    pub pipeline_stages: u32,
    /// Raw wire capacity at the domain clock (before protocol efficiency).
    pub bw_bytes_per_s: f64,
    /// Dynamic energy per byte at V_nom (03 rescales core-domain links by (V/V_nom)^2).
    pub e_j_per_byte: f64,
    /// Pipeline-flop clock energy per byte-time when idle (ungated), J per cycle.
    pub e_j_per_cycle_idle: f64,
    pub source: LinkSource,
}

/// On-die link: picks the class with the least energy (wire + inserted pipeline flops) among intermediate,
/// semi-global and (for <= 64-bit links) global, ties to the lower latency.
pub fn on_die(n: &TechNode, s_45: f64, len_um: f64, width_bits: f64, f_hz: f64, kappa_wire: f64) -> LinkCost {
    let t_avail = ((1.0 - n.f_logic) / f_hz - n.t_setup_clkq_ps * 1e-12).max(1e-12);
    let mut best: Option<(f64, LinkCost)> = None;
    let classes: &[WireClassId] =
        if width_bits <= 64.0 { &[WireClassId::Intermediate, WireClassId::SemiGlobal, WireClassId::Global] } else { &[WireClassId::Intermediate, WireClassId::SemiGlobal] };
    for &class in classes {
        let sizing = Sizing::Energy;
        let t_wire = tau_s_per_um(n, class, sizing) * len_um;
        let stages = ((t_wire / t_avail).ceil() - 1.0).max(0.0);
        let e = e_bit(n, class, sizing, len_um, n.vdd_nom) * kappa_wire + stages * e_ff_bit(n, s_45);
        let lat = 1.0 + stages;
        let lc = LinkCost {
            class: Some(class),
            length_um: len_um,
            latency_cycles: lat as u32,
            latency_s: lat / f_hz,
            pipeline_stages: stages as u32,
            bw_bytes_per_s: width_bits * f_hz / 8.0,
            e_j_per_byte: 8.0 * e,
            e_j_per_cycle_idle: stages * e_ff_bit(n, s_45) * 0.1,
            source: LinkSource::Wire,
        };
        let better = match &best {
            None => true,
            Some((be, b)) => e < *be * (1.0 - 1e-12) || (e <= *be * (1.0 + 1e-12) && lc.latency_s < b.latency_s),
        };
        if better {
            best = Some((e, lc));
        }
    }
    best.expect("at least one class").1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::Tables;

    #[test]
    fn n7_semi_global_matches_04_7_1() {
        let n = Tables::get().node("tsmc_n7").unwrap();
        let d = tau_s_per_um(n, WireClassId::SemiGlobal, Sizing::Delay) * 1e3 * 1e12;
        let e = tau_s_per_um(n, WireClassId::SemiGlobal, Sizing::Energy) * 1e3 * 1e12;
        assert!((95.0..115.0).contains(&d), "delay-optimal {d} ps/mm");
        assert!(e > d * 1.08 && e < d * 1.2, "energy-aware {e} ps/mm vs {d}");
        assert!((c_rep_ratio(n, Sizing::Delay) - 1.05).abs() < 0.02);
        assert!((c_rep_ratio(n, Sizing::Energy) - 0.35).abs() < 0.02);
        let pj_mm = e_bit(n, WireClassId::SemiGlobal, Sizing::Energy, 1000.0, 0.75) * 1e12;
        assert!((0.03..0.05).contains(&pj_mm), "{pj_mm} pJ/bit/mm");
        let g = tau_s_per_um(n, WireClassId::Global, Sizing::Delay) * 1e15;
        assert!((15.0..35.0).contains(&g), "global {g} ps/mm");
    }

    #[test]
    fn longer_links_cost_more_and_pipeline() {
        let n = Tables::get().node("tsmc_n7").unwrap();
        let short = on_die(n, 0.14, 200.0, 512.0, 1.41e9, 1.5);
        let long = on_die(n, 0.14, 20000.0, 512.0, 1.41e9, 1.5);
        assert!(long.e_j_per_byte > 10.0 * short.e_j_per_byte);
        assert!(long.pipeline_stages > 0 && short.pipeline_stages == 0 && long.latency_s > short.latency_s);
        let half = on_die(n, 0.14, 10000.0, 512.0, 1.41e9, 1.5);
        assert!(half.e_j_per_byte < long.e_j_per_byte && half.latency_cycles <= long.latency_cycles);
    }
}
