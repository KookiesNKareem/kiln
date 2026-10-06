//! 06 §6.3 standard MAP-Elites descriptors with fixed ranges.

use std::collections::BTreeMap;

use kiln_ir::hw::HwModel;
use kiln_trace::Corner;
use kiln_trace::result::{EvalResult, Feature};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Descriptor {
    pub name: &'static str,
    pub unit: &'static str,
    pub low: f64,
    pub high: f64,
    pub log: bool,
    /// Number of values (3 for `energy_split`).
    pub dims: usize,
}

const fn d(name: &'static str, unit: &'static str, low: f64, high: f64, log: bool) -> Descriptor {
    Descriptor {
        name,
        unit,
        low,
        high,
        log,
        dims: 1,
    }
}

pub const DESCRIPTORS: [Descriptor; 12] = [
    d("die_mm2_total", "mm^2", 10.0, 8.0 * 858.0, false),
    d("power_w", "W", 5.0, 8.0 * 1000.0, false),
    d("onchip_bytes", "B", 1048576.0, 17179869184.0, true),
    d("peak_flops_bf16", "FLOP/s", 1e12, 1e17, true),
    d("machine_balance", "FLOP/B", 1.0, 4096.0, true),
    d("n_chips", "count", 1.0, 1024.0, true),
    d("compute_tiles", "count", 1.0, 65536.0, true),
    d("near_mem_flop_frac", "1", 0.0, 1.0, false),
    d("memory_levels", "count", 1.0, 8.0, false),
    Descriptor {
        dims: 3,
        ..d("energy_split", "1", 0.0, 1.0, false)
    },
    d("bound_frac_mem", "1", 0.0, 1.0, false),
    d("score_rel_width", "1", 0.0, 2.0, false),
];

pub fn descriptor(name: &str) -> Option<&'static Descriptor> {
    DESCRIPTORS.iter().find(|d| d.name == name)
}

impl Descriptor {
    /// Position in `[0, 1]` on the fixed range (log scale where marked), clamped.
    pub fn normalize(&self, x: f64) -> f64 {
        let t = if self.log {
            (x.max(f64::MIN_POSITIVE).ln() - self.low.ln()) / (self.high.ln() - self.low.ln())
        } else {
            (x - self.low) / (self.high - self.low)
        };
        if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) }
    }
}

pub const ENERGY_SPLIT_KEYS: [&str; 3] = [
    "energy_split_compute",
    "energy_split_memory",
    "energy_split_interconnect",
];

/// Standard descriptors in `[0, 1]` on their fixed ranges; `energy_split` expands to three scalar keys.
pub fn normalized(features: &BTreeMap<String, Feature>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for d in &DESCRIPTORS {
        match features.get(d.name) {
            Some(Feature::Scalar(x)) => {
                out.insert(d.name.to_string(), d.normalize(*x));
            }
            Some(Feature::Vector(v)) if d.dims == 3 => {
                for (k, x) in ENERGY_SPLIT_KEYS.iter().zip(v) {
                    out.insert(k.to_string(), d.normalize(*x));
                }
            }
            _ => {}
        }
    }
    out
}

const BF16: &str = "bf16*bf16";

/// The design's off-chip (mem stack interface) bandwidth and capacity, which the matched envelope pins (06 §6.4).
pub fn offchip(r: &mut EvalResult, m: &HwModel) {
    let s = m.summary();
    r.features.insert(
        crate::fitness::OFFCHIP_BW.into(),
        Feature::Scalar(s.offchip_bandwidth.0),
    );
    r.features.insert(
        crate::fitness::OFFCHIP_BYTES.into(),
        Feature::Scalar(s.offchip_capacity.0 as f64),
    );
}

/// Fills standard descriptors not already set by the engine; drops ones not in `only` when given.
pub fn fill(r: &mut EvalResult, model: Option<&HwModel>, only: Option<&[String]>) {
    let mut put = |k: &str, v: Feature| {
        r.features.entry(k.into()).or_insert(v);
    };
    if let Some(m) = model {
        let s = m.summary();
        let bf16 = s
            .peak_ops
            .iter()
            .filter(|(k, _)| k.starts_with(BF16) && !k.ends_with(":sparse"))
            .map(|(_, v)| *v)
            .fold(0.0, f64::max);
        put("onchip_bytes", Feature::Scalar(s.onchip_capacity.0 as f64));
        put("peak_flops_bf16", Feature::Scalar(bf16));
        if s.offchip_bandwidth.0 > 0.0 {
            put(
                "machine_balance",
                Feature::Scalar(bf16 / s.offchip_bandwidth.0),
            );
        }
        put("n_chips", Feature::Scalar(s.chip_count as f64));
        let tiles = m
            .units
            .iter()
            .filter(|u| m.nodes[u.node].enabled && u.spec.kind.is_mac())
            .count();
        put("compute_tiles", Feature::Scalar(tiles as f64));
        let levels = s.chips.iter().map(|c| c.levels.len()).max().unwrap_or(0);
        put("memory_levels", Feature::Scalar(levels as f64));
    }
    if let Some(p) = r.physical.clone() {
        put(
            "die_mm2_total",
            Feature::Scalar(p.die_mm2.values().map(|d| d.central).sum()),
        );
        put("power_w", Feature::Scalar(p.tdp_w));
    }
    let central: Vec<_> = r
        .sim
        .iter()
        .filter(|s| s.corner == Corner::Central)
        .collect();
    if !central.is_empty() {
        let (c, m, l) = central.iter().fold((0.0, 0.0, 0.0), |(c, m, l), s| {
            let e = &s.energy;
            (
                c + e.compute_j + e.nmp_j,
                m + e.memory_j.values().sum::<f64>(),
                l + e.link_j.values().sum::<f64>(),
            )
        });
        let t = c + m + l;
        if t > 0.0 {
            put("energy_split", Feature::Vector(vec![c / t, m / t, l / t]));
        }
    }
    let decode: Vec<_> = r
        .phases
        .iter()
        .filter(|p| p.phase.as_str().contains("decode"))
        .collect();
    let (num, den) = decode.iter().fold((0.0, 0.0), |(n, d), p| {
        let mem: f64 = p
            .bound_breakdown
            .iter()
            .filter(|(k, _)| k.starts_with("mem:"))
            .map(|(_, v)| v)
            .sum();
        (n + mem * p.time_s.central, d + p.time_s.central)
    });
    if den > 0.0 {
        put("bound_frac_mem", Feature::Scalar(num / den));
    }
    if let Some(si) = r.score_interval {
        put("score_rel_width", Feature::Scalar(si.rel_width()));
    }
    if let Some(only) = only {
        r.features.retain(|k, _| only.iter().any(|o| o == k));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::result_with;

    #[test]
    fn normalize_fixed_ranges() {
        let d = descriptor("onchip_bytes").unwrap();
        assert_eq!(d.normalize(1048576.0), 0.0);
        assert!((d.normalize(2f64.powi(27)) - 0.5).abs() < 1e-12);
        assert_eq!(d.normalize(1e30), 1.0);
        assert_eq!(
            descriptor("near_mem_flop_frac")
                .unwrap()
                .normalize(f64::NAN),
            0.0
        );
    }

    #[test]
    fn fills_design_and_result_features() {
        let design = kiln_ir::hw::load_file(
            crate::inputs::default_designs_dir().join("reference/a100_sxm4_40gb.json5"),
        )
        .unwrap();
        let (model, _) = design.expand(&Default::default()).unwrap();
        let mut r = result_with(&[("decode_b1", 80.0), ("prefill_b1", 1000.0)], 826.0, 400.0);
        fill(&mut r, Some(&model), None);
        let f = |k: &str| match &r.features[k] {
            Feature::Scalar(x) => *x,
            Feature::Vector(_) => panic!(),
        };
        assert!(
            (f("peak_flops_bf16") - 312e12).abs() / 312e12 < 0.01,
            "{}",
            f("peak_flops_bf16")
        );
        assert_eq!(f("n_chips"), 1.0);
        assert_eq!(f("die_mm2_total"), 826.0);
        assert!((f("bound_frac_mem") - 0.8).abs() < 1e-12);
        assert!(f("compute_tiles") >= 108.0);
        fill(&mut r, None, Some(&["n_chips".into()]));
        assert_eq!(r.features.len(), 1);
    }
}
