//! Area roll-up and reporting (04 §10): per die envelope area with its corner band, area parts, shoreline,
//! transistor estimate, package outline, envelope (cap, cap level, thermal limits), E-PHYS problems, timings.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::phys::CapLevel;
use serde::Serialize;

use crate::characterize::{self, Characterized, PARTS, Part};
use crate::floorplan::{self, Floorplan, FloorplanInput, PlaceTier};
use crate::params::{PCorner, Params};
use crate::power::PowerModel;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DieReport {
    pub path: String,
    pub node: String,
    /// Envelope (analytic Tier A roll-up) at the central, optimistic and pessimistic parameter corners, mm^2.
    pub area_mm2: f64,
    pub area_low_mm2: f64,
    pub area_high_mm2: f64,
    /// Placed outline (declared fixed outline, or the sized die incl. shoreline growth), mm^2.
    pub outline_mm2: f64,
    pub legalized_mm2: f64,
    pub fixed_outline: bool,
    pub shoreline_limited: bool,
    pub parts_mm2: BTreeMap<String, f64>,
    pub whitespace_mm2: f64,
    /// S, E, N, W: (edge length, used) mm.
    pub edges_mm: [(f64, f64); 4],
    pub hbm_shoreline_used_mm: f64,
    pub hbm_shoreline_available_mm: f64,
    pub transistors_b: f64,
    pub sram_mib: f64,
    pub layer: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhysReport {
    pub model: String,
    pub data_hash: String,
    pub calibration: Vec<String>,
    pub dies: Vec<DieReport>,
    pub package_mm2: f64,
    pub package_low_mm2: f64,
    pub package_high_mm2: f64,
    pub package_table: String,
    pub tdp_w: Option<f64>,
    /// Unpublished cap: (nominal, lo, hi) W, reported, never enforced (04 §8.1).
    pub tdp_assumed_w: Option<(f64, f64, f64)>,
    pub cap_level: CapLevel,
    pub tj_max_c: f64,
    pub q_avg_max_w_mm2: f64,
    pub q_block_max_w_mm2: f64,
    pub node: String,
    pub problems: Vec<Diagnostic>,
    /// Characterize, place and link-derivation wall times, microseconds (excluded from hashes).
    pub timing_us: (f64, f64, f64),
    pub place_tier: PlaceTier,
    pub macros: usize,
}

pub fn die_parts(_hw: &HwModel, ch: &Characterized, die_node: usize) -> [f64; 7] {
    let mut parts = [0.0; 7];
    let mut st = vec![die_node];
    while let Some(x) = st.pop() {
        for (i, p) in ch.nodes[x].parts.iter().enumerate() {
            parts[i] += p;
        }
        st.extend(ch.kids[x].iter().copied());
    }
    parts
}

fn corner_areas(hw: &HwModel, params: &Params, c: PCorner) -> (Vec<f64>, f64) {
    let p = params.at(c);
    let ch = characterize::characterize(hw, &p);
    let fp = floorplan::build(&FloorplanInput { hw, ch: &ch, params: &p, tier: PlaceTier::A, seed: 0 });
    (fp.dies.iter().map(|d| d.area_env_um2 / 1e6).collect(), fp.packages.iter().map(|x| x.outline.area()).sum::<f64>() / 1e6)
}

pub fn build(hw: &HwModel, ch: &Characterized, fp: &Floorplan, pw: &PowerModel, params: &Params, timing_us: (f64, f64, f64)) -> PhysReport {
    let (lo, pkg_lo) = corner_areas(hw, params, PCorner::Optimistic);
    let (hi, pkg_hi) = corner_areas(hw, params, PCorner::Pessimistic);
    let dies: Vec<DieReport> = fp
        .dies
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let parts = die_parts(hw, ch, d.node);
            let mut ge = 0.0;
            let mut bits = 0.0;
            let mut st = vec![d.node];
            while let Some(x) = st.pop() {
                ge += ch.nodes[x].ge;
                bits += ch.nodes[x].sram_bits;
                st.extend(ch.kids[x].iter().copied());
            }
            let blocks: f64 = parts.iter().sum();
            let hbm_avail: f64 = [1usize, 3].iter().map(|&e| d.edges[e].len_um).sum::<f64>() / 1000.0;
            DieReport {
                path: d.path.clone(),
                node: d.tech.clone(),
                area_mm2: d.area_env_um2 / 1e6,
                area_low_mm2: lo.get(i).copied().unwrap_or(d.area_env_um2 / 1e6).min(d.area_env_um2 / 1e6),
                area_high_mm2: hi.get(i).copied().unwrap_or(d.area_env_um2 / 1e6).max(d.area_env_um2 / 1e6),
                outline_mm2: d.outline.area() / 1e6,
                legalized_mm2: d.legalized_um2 / 1e6,
                fixed_outline: d.fixed_outline,
                shoreline_limited: d.shoreline_limited,
                parts_mm2: PARTS.iter().map(|p| (format!("{p:?}").to_lowercase(), parts[*p as usize] / 1e6)).collect(),
                whitespace_mm2: (d.area_env_um2 - blocks) / 1e6,
                edges_mm: [0, 1, 2, 3].map(|e| (d.edges[e].len_um / 1000.0, d.edges[e].used_um / 1000.0)),
                hbm_shoreline_used_mm: d.edges.iter().map(|e| e.hbm_used_um).sum::<f64>() / 1000.0,
                hbm_shoreline_available_mm: hbm_avail.max(d.edges.iter().map(|e| e.hbm_used_um).sum::<f64>() / 1000.0),
                transistors_b: (4.0 * ge + 6.0 * bits) / 1e9,
                sram_mib: bits / 8.0 / 1048576.0,
                layer: d.layer,
            }
        })
        .collect();
    let mut problems = ch.problems.clone();
    problems.extend(fp.problems.iter().cloned());
    problems.extend(crate::power::vf_problems(hw, ch));
    let node = dies.first().map_or_else(String::new, |d| d.node.clone());
    let _ = Part::Datapath;
    PhysReport {
        model: crate::MODEL_ID.into(),
        data_hash: crate::data::data_hash().into(),
        calibration: params.set_ids.clone(),
        package_mm2: fp.packages.iter().map(|p| p.outline.area()).sum::<f64>() / 1e6,
        package_low_mm2: pkg_lo,
        package_high_mm2: pkg_hi,
        package_table: fp.packages.first().map_or_else(String::new, |p| p.table.clone()),
        dies,
        tdp_w: pw.cap_w,
        tdp_assumed_w: pw.assumed_cap_w,
        cap_level: pw.cap_level,
        tj_max_c: pw.tj_max_c,
        q_avg_max_w_mm2: pw.q_avg_max,
        q_block_max_w_mm2: params.get("q_block_max", None),
        node,
        problems,
        timing_us,
        place_tier: fp.tier,
        macros: fp.stats.macros,
    }
}
