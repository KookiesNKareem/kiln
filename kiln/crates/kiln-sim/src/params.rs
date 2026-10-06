//! Engine parameters with ranges (03 §9, §9.1). M1 ships two sets: `null` (efficiencies 1, overheads 0;
//! floors and the L3 subset) and `assumed-v0` (generic priors with plausible bands, nothing fitted).

use std::collections::BTreeMap;

use kiln_ir::common::content_hash;
use kiln_ir::hw::types::ExecModel;
use kiln_trace::Corner;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub lower: f64,
    pub central: f64,
    pub upper: f64,
}

/// Which end of the range lowers throughput (03 §9.1 `pess_dir`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PessDir {
    Lower,
    Upper,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub key: BTreeMap<String, String>,
    pub unit: String,
    pub range: Range,
    pub pess_dir: PessDir,
    pub status: String,
    pub basis: String,
    /// Table-valued parameters (telemetry-derived operating points `[x, value]`); the range then scales it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<Vec<[f64; 2]>>,
}

impl Param {
    pub fn at(&self, c: Corner) -> f64 {
        let (lo, hi) = (self.range.lower, self.range.upper);
        match (c, self.pess_dir) {
            (Corner::Central, _) => self.range.central,
            (Corner::Low, PessDir::Lower) | (Corner::High, PessDir::Upper) => lo,
            (Corner::Low, PessDir::Upper) | (Corner::High, PessDir::Lower) => hi,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamSet {
    pub id: String,
    pub kind: String,
    pub params: Vec<Param>,
    /// Hash of the calibration-set file this was resolved from (06 §3.6); provenance carries it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_hash: Option<String>,
    /// Parameters that fell back to a wildcard or assumed prior (06 §3.1 rule 2 `extrapolated`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extrapolated: Vec<String>,
}

/// Sustained clock of power-capped domains as a function of MAC-pipe activity at nominal clock, from measured
/// telemetry operating points (no power model; 03 §4.5 stand-in until 04's model is calibrated).
#[derive(Clone, Debug, PartialEq)]
pub struct ClockOp {
    pub clocks: Vec<usize>,
    /// `(activity, hz)` knots, activity ascending; linear between knots, clamped outside.
    pub points: Vec<(f64, f64)>,
    pub scale: f64,
}

impl ClockOp {
    pub fn hz(&self, activity: f64) -> f64 {
        let p = &self.points;
        let f = match p.iter().position(|q| q.0 >= activity) {
            _ if p.is_empty() => return f64::INFINITY,
            Some(0) => p[0].1,
            None => p[p.len() - 1].1,
            Some(i) => {
                let (a, b) = (p[i - 1], p[i]);
                a.1 + (b.1 - a.1) * (activity - a.0) / (b.0 - a.0).max(1e-12)
            }
        };
        f * self.scale
    }
}

/// Point values the engine runs with (one corner).
#[derive(Clone, Debug, PartialEq)]
pub struct SimParams {
    pub eta_dram: f64,
    pub t_launch: f64,
    pub t_min_kernel: f64,
    pub t_gap: f64,
    pub t_dispatch: f64,
    pub t_program: f64,
    pub t_sync: f64,
    pub rho_max: f64,
    pub contention_scale: f64,
    /// DRAM pipeline ramp (fill + drain) charged once per DRAM resource a segment streams through.
    pub t_dram_ramp: f64,
    /// Compute-pipe efficiency per unit template entity path (`unit_eff`, 03 §9).
    pub unit_eff: Vec<(String, f64)>,
    pub clock_op: Option<ClockOp>,
    /// Bandwidth multipliers per resource (shadow-price probes).
    pub cap_scale: Vec<(u32, f64)>,
}

impl SimParams {
    pub fn null() -> Self {
        SimParams {
            eta_dram: 1.0,
            t_launch: 0.0,
            t_min_kernel: 0.0,
            t_gap: 0.0,
            t_dispatch: 0.0,
            t_program: 0.0,
            t_sync: 0.0,
            rho_max: 0.95,
            contention_scale: 0.0,
            t_dram_ramp: 0.0,
            unit_eff: vec![],
            clock_op: None,
            cap_scale: vec![],
        }
    }
}

pub fn exec_key(m: ExecModel) -> &'static str {
    match m {
        ExecModel::HostLaunched => "host_launched",
        ExecModel::DeviceQueued => "device_queued",
        ExecModel::StaticDataflow => "static_dataflow",
    }
}

impl ParamSet {
    pub fn null() -> Self {
        ParamSet { id: "null".into(), kind: "null".into(), params: vec![], set_hash: None, extrapolated: vec![] }
    }

    /// Generic priors (06 §3.2 plausible bands; assumed, not fitted). `dram` keys the DRAM residual.
    pub fn assumed(exec: ExecModel, dram: &str) -> Self {
        let p = |name: &str, key: (&str, &str), unit: &str, r: (f64, f64, f64), pess: PessDir| Param {
            name: name.into(),
            key: BTreeMap::from([(key.0.to_string(), key.1.to_string())]),
            unit: unit.into(),
            range: Range { lower: r.0, central: r.1, upper: r.2 },
            pess_dir: pess,
            status: "assumed".into(),
            basis: "band".into(),
            table: None,
        };
        let e = ("exec_model", exec_key(exec));
        let us = 1e-6;
        let mut params = vec![p("eta_res", ("dram_kind", dram), "1", (0.85, 0.91, 0.97), PessDir::Lower)];
        params.extend(match exec {
            ExecModel::HostLaunched => vec![
                p("t_launch", e, "s", (2.0 * us, 5.0 * us, 10.0 * us), PessDir::Upper),
                p("t_min_kernel", e, "s", (1.0 * us, 2.5 * us, 5.0 * us), PessDir::Upper),
                p("t_gap", e, "s", (0.5 * us, 1.0 * us, 2.0 * us), PessDir::Upper),
            ],
            ExecModel::DeviceQueued => vec![p("t_dispatch", e, "s", (1.0 * us, 2.0 * us, 5.0 * us), PessDir::Upper)],
            ExecModel::StaticDataflow => vec![
                p("t_program", e, "s", (5.0 * us, 20.0 * us, 50.0 * us), PessDir::Upper),
                p("t_sync", ("exec_model", exec_key(exec)), "s", (0.5 * us, 1.0 * us, 3.0 * us), PessDir::Upper),
            ],
        });
        params.push(p("contention_scale", ("engine", "tier_a"), "1", (0.5, 1.0, 1.5), PessDir::Upper));
        ParamSet { id: "assumed-v0".into(), kind: "assumed".into(), params, set_hash: None, extrapolated: vec![] }
    }

    pub fn hash(&self) -> String {
        self.set_hash.clone().unwrap_or_else(|| content_hash("cal1-", &serde_json::to_value(self).expect("params serialize")))
    }

    pub fn at(&self, c: Corner) -> SimParams {
        let mut s = SimParams::null();
        for p in &self.params {
            let v = p.at(c);
            match p.name.as_str() {
                "eta_res" => s.eta_dram = v,
                "t_launch" => s.t_launch = v,
                "t_min_kernel" => s.t_min_kernel = v,
                "t_gap" => s.t_gap = v,
                "t_dispatch" => s.t_dispatch = v,
                "t_program" => s.t_program = v,
                "t_sync" => s.t_sync = v,
                "rho_max" => s.rho_max = v,
                "contention_scale" => s.contention_scale = v,
                "t_dram_ramp" => s.t_dram_ramp = v,
                "unit_eff" => {
                    if let Some(t) = p.key.get("unit_template") {
                        s.unit_eff.push((t.clone(), v));
                    }
                }
                "f_cap_op" => {
                    let clocks = p.key.get("clock_ix").and_then(|c| c.parse().ok()).into_iter().collect();
                    let points = p.table.iter().flatten().map(|q| (q[0], q[1])).collect();
                    s.clock_op = Some(ClockOp { clocks, points, scale: v });
                }
                _ => {}
            }
        }
        s
    }

    /// Central values with one parameter moved to its pessimistic end (sensitivity intervals).
    pub fn perturbed(&self, i: usize, c: Corner) -> SimParams {
        let mut set = self.clone();
        let v = set.params[i].at(c);
        set.params[i].range.central = v;
        set.at(Corner::Central)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_follow_pessimistic_direction() {
        let s = ParamSet::assumed(ExecModel::HostLaunched, "hbm2");
        let (lo, c, hi) = (s.at(Corner::Low), s.at(Corner::Central), s.at(Corner::High));
        assert!(lo.eta_dram < c.eta_dram && c.eta_dram < hi.eta_dram);
        assert!(lo.t_gap > c.t_gap && c.t_gap > hi.t_gap);
        assert_eq!(ParamSet::null().at(Corner::Low), SimParams::null());
        assert_ne!(s.hash(), ParamSet::null().hash());
    }
}
