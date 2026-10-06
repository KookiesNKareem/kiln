//! Calibrated physical parameters (04 §12.1) resolved for one design: fitted values from the generic set (and the
//! platform set of a reference chip's family), else the registry / node-table prior; corners per 04 §3.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::tables::{CalibEntry, Tables};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PCorner {
    #[default]
    Central,
    /// Every parameter at the end of its range that makes area, power and energy smaller.
    Optimistic,
    /// Every parameter at the end that makes them larger.
    Pessimistic,
}

/// Parameters whose larger value is optimistic (smaller area/power); every other registered one is pessimistic.
const LARGER_IS_OPTIMISTIC: [&str; 3] = ["util_std_cell", "u_place", "eta_vr"];
/// Shape parameters without a pessimistic direction (kept central at every corner).
const NO_DIRECTION: [&str; 2] = ["v_th", "alpha"];

/// Node where the area group fits `util_std_cell`; other nodes transfer from it (`Params::util`).
pub const UTIL_REF_NODE: &str = "tsmc_n7";

/// Node-scoped parameters and their node-table path.
fn node_path(name: &str) -> Option<&'static str> {
    Some(match name {
        "util_std_cell" => "logic.util_std_cell",
        "p_ll" => "leakage.p_ll_w_mm2",
        "p_ls" => "sram.p_ls_mw_per_mib",
        "v_th" => "dvfs.v_th",
        "alpha" => "dvfs.alpha",
        _ => return None,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Params {
    pub corner: PCorner,
    /// `(name, key)` -> fitted entry (key "" when unscoped).
    fitted: BTreeMap<(String, String), CalibEntry>,
    /// Calibration-set ids in application order (generic, then platform).
    pub set_ids: Vec<String>,
    /// Explicit values (calibration fits and tests), highest precedence, at every corner.
    pub fixed: BTreeMap<(String, String), f64>,
}

impl Params {
    /// Generic fitted set plus, for a reference design whose `family` names a platform set, that set.
    pub fn for_family(family: Option<&str>) -> Params {
        let t = Tables::get();
        let mut p = Params::default();
        let generic = t.calib.iter().filter(|c| c.scope == "generic");
        let platform = t.calib.iter().filter(|c| c.scope == "platform" && family.is_some() && c.platform.as_deref() == family);
        for c in generic.chain(platform) {
            p.set_ids.push(c.id.clone());
            for e in &c.params {
                p.fitted.insert((e.name.clone(), e.key.clone().unwrap_or_default()), e.clone());
            }
        }
        p
    }

    /// Priors only (no calibration set).
    pub fn priors() -> Params {
        Params::default()
    }

    pub fn at(&self, corner: PCorner) -> Params {
        Params { corner, ..self.clone() }
    }

    pub fn with(mut self, name: &str, key: Option<&str>, v: f64) -> Params {
        self.fixed.insert((name.to_owned(), key.unwrap_or_default().to_owned()), v);
        self
    }

    /// `(central, lower, upper)` of a parameter.
    pub fn band(&self, name: &str, key: Option<&str>) -> (f64, f64, f64) {
        let t = Tables::get();
        let k = key.unwrap_or_default().to_owned();
        if let Some(e) = self.fitted.get(&(name.to_owned(), k.clone())) {
            let (lo, hi) = (e.value * (-e.sigma).exp(), e.value * e.sigma.exp());
            return (e.value, lo.max(e.bounds[0]), hi.min(e.bounds[1]));
        }
        let prior = if let Some(path) = node_path(name) {
            key.and_then(|n| t.node(n)).and_then(|n| n.sourced.get(path).cloned())
        } else if name == "e_dram_core" {
            let kind = key.unwrap_or("hbm2");
            t.dram.get(kind).map(|d| crate::sourced::Sourced {
                v: d.e_core_pj_per_bit,
                q: crate::sourced::Quality::Asm,
                c: crate::sourced::Conf::M,
                src: None,
                bounds: Some([2.0, 6.0]),
                range: None,
                fit: true,
                note: None,
            })
        } else {
            t.params.get(name).map(|r| r.prior.clone())
        };
        let s = prior.unwrap_or_else(|| panic!("phys parameter {name}/{k} has no prior"));
        let (lo, hi) = s.band();
        (s.v, lo, hi)
    }

    pub fn get(&self, name: &str, key: Option<&str>) -> f64 {
        if let Some(v) = self.fixed.get(&(name.to_owned(), key.unwrap_or_default().to_owned())) {
            return *v;
        }
        let (c, lo, hi) = self.band(name, key);
        if NO_DIRECTION.contains(&name) {
            return c;
        }
        let opt_is_high = LARGER_IS_OPTIMISTIC.contains(&name);
        match (self.corner, opt_is_high) {
            (PCorner::Central, _) => c,
            (PCorner::Optimistic, true) | (PCorner::Pessimistic, false) => hi,
            (PCorner::Optimistic, false) | (PCorner::Pessimistic, true) => lo,
        }
    }

    /// Std-cell utilization on node `n` (04 §4.1, §12.1): the node's own fitted or fixed value when it has one,
    /// else the reference node's (N7, where the area group fits it) times the node's sourced density realization.
    /// The generic 0.65 prior on an unfitted node would otherwise make it less dense than the fitted reference
    /// for no physical reason.
    pub fn util(&self, n: &crate::tables::TechNode) -> f64 {
        let key = ("util_std_cell".to_owned(), n.id.clone());
        if n.id == UTIL_REF_NODE || self.fixed.contains_key(&key) || self.fitted.contains_key(&key) {
            return self.get("util_std_cell", Some(&n.id));
        }
        self.get("util_std_cell", Some(UTIL_REF_NODE)) * n.density_realization
    }

    pub fn fitted(&self) -> impl Iterator<Item = &CalibEntry> {
        self.fitted.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_move_in_the_declared_direction() {
        let p = Params::priors();
        let (c, lo, hi) = p.band("kappa_pe", None);
        assert!(lo < c && c < hi);
        assert_eq!(p.at(PCorner::Pessimistic).get("kappa_pe", None), hi);
        assert_eq!(p.at(PCorner::Pessimistic).get("u_place", None), p.band("u_place", None).1);
        assert_eq!(p.at(PCorner::Optimistic).get("util_std_cell", Some("tsmc_n7")), p.band("util_std_cell", Some("tsmc_n7")).2);
        assert_eq!(p.get("p_ll", Some("tsmc_n7")), 0.04);
        assert_eq!(p.clone().with("kappa_pe", None, 2.0).get("kappa_pe", None), 2.0);
    }
}
