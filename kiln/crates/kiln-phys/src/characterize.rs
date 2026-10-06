//! Per-instance area, leakage and per-event energies of an expanded design (04 §5): `characterize(hw, params)`.
//! Areas include kappa factors and std-cell utilization; energies are at the node's V_nom (the clock solve
//! rescales by (V/V_nom)^2); disabled (harvested) instances are placed and counted in area, gated for leakage.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::{ComputeKind, Dataflow, MemImpl, MemKind, PortDir, PrecisionMode};
use kiln_ir::hw::model::{ContainerKind, HwModel, MemSpec, NodeIx};
use kiln_ir::hw::net::{IoKind, LinkPhys, SerdesSpec};
use kiln_ir::hw::types::{BlockKind, ExecModel};
use kiln_ir::precision::Precision;
use serde::Serialize;

use crate::blocks::{self, Lane, Ports, Router, Sram};
use crate::params::Params;
use crate::tables::{PhyT, Tables, TechNode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Datapath,
    Sram,
    Rf,
    Control,
    Noc,
    Phy,
    Misc,
}

pub const PARTS: [Part; 7] = [Part::Datapath, Part::Sram, Part::Rf, Part::Control, Part::Noc, Part::Phy, Part::Misc];

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct NodePhys {
    /// Own area (children excluded), um^2.
    pub area_um2: f64,
    pub parts: [f64; 7],
    /// Leakage at V_nom, 85 C (0 for harvested instances: gated).
    pub leak_w: f64,
    pub flops: f64,
    pub sram_bits: f64,
    /// Gate-equivalent count (transistor estimate: 4 per GE).
    pub ge: f64,
    /// Control-logic gate equivalents (switching while a phase runs, `alpha_ctrl`).
    pub ctrl_ge: f64,
    /// Part of the own area that exists because of compute units (SIMT sub-core control): with the units' own area,
    /// what the die arrangement leaves out (floorplan.rs, P6).
    pub unit_ctrl_um2: f64,
    pub unit_ctrl_leak_w: f64,
}

/// Per-op energies of one compute unit at V_nom (J).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct UnitEnergy {
    /// `(mode key prefix, J per MAC)` for MAC modes (`bf16*bf16+fp32`) and `(dtype, J per op)` for elementwise.
    pub modes: Vec<(String, f64)>,
    pub idle_frac: f64,
    pub e_elem_j: f64,
    pub e_transc_j: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct MemEnergy {
    pub read_j_per_b: f64,
    pub write_j_per_b: f64,
    pub latency_s: f64,
    pub dram: bool,
    /// DRAM: core + IO energy per byte (board level) and PHY energy per byte (die level).
    pub e_core_j_per_b: f64,
    pub e_phy_j_per_b: f64,
    pub background_w: f64,
}

/// A PHY instance (PHY block or I/O port) for shoreline and links.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhyUse {
    pub node: usize,
    pub table: String,
    pub kind: IoKind,
    pub units: f64,
    pub shoreline_um: f64,
    pub area_um2: f64,
    pub e_j_per_bit: f64,
    pub latency_s: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Characterized {
    /// Children of every node in canonical (path) order, and the canonical preorder and rank of the arena: every
    /// reduction and placement iterates in this order so results do not depend on declaration order (P4/M6).
    pub kids: Vec<Vec<usize>>,
    pub order: Vec<usize>,
    pub rank: Vec<usize>,
    pub nodes: Vec<NodePhys>,
    /// Technology node id per arena node.
    pub tech: Vec<String>,
    pub units: Vec<UnitEnergy>,
    pub mems: Vec<MemEnergy>,
    pub routers: Vec<Router>,
    pub phys: Vec<PhyUse>,
    pub problems: Vec<Diagnostic>,
}

fn ports_of(m: &kiln_ir::hw::compute::Memory) -> (Ports, f64) {
    let mut p = Ports::default();
    let mut widest = 0u32;
    let banks = m.banks.max(1);
    for ps in &m.ports {
        let c = if ps.per_bank { ps.count } else { ps.count.div_ceil(banks) }.max(1);
        match ps.dir {
            PortDir::Read => p.read += c,
            PortDir::Write => p.write += c,
            _ => p.rw += c,
        }
        widest = widest.max(ps.width_bits);
    }
    if m.ports.is_empty() {
        p.rw = 1;
    }
    let width = |ps: &kiln_ir::hw::compute::PortSpec| if ps.per_bank { ps.width_bits } else { ps.width_bits / banks };
    let bank_word = m.ports.iter().map(width).max().unwrap_or(m.word_bits).max(m.word_bits);
    (p, f64::from(bank_word))
}

fn bits_of(p: &kiln_ir::precision::PrecisionSpec) -> f64 {
    f64::from(p.precision.element_bits())
}

struct Ctx<'a> {
    t: &'static Tables,
    p: &'a Params,
    lanes: BTreeMap<(Precision, Precision, Precision, u32, u32), Lane>,
}

impl Ctx<'_> {
    fn lane(&mut self, a: Precision, b: Precision, acc: Precision, l: f64, extra: f64) -> Lane {
        let k = (a, b, acc, l as u32, extra as u32);
        *self.lanes.entry(k).or_insert_with(|| blocks::mac_lane(&self.t.datapath, a, b, acc, l, extra))
    }
}

/// Holding-register bits per PE for the supported stationarity modes (04 §4.9).
fn dataflow_bits(modes: &[Dataflow], b_a: f64, b_w: f64, w_acc: f64, dp: &crate::tables::Datapath) -> f64 {
    let mut bits = 0.0;
    let mut extra_mux = 0.0;
    let all = modes.contains(&Dataflow::Any);
    let has = |d: Dataflow| all || modes.contains(&d);
    let mut n = 0;
    for (d, w) in [(Dataflow::WeightStationary, b_w), (Dataflow::OutputStationary, w_acc), (Dataflow::InputStationary, b_a)] {
        if has(d) {
            bits += w;
            n += 1;
            if n > 1 {
                extra_mux += w;
            }
        }
    }
    bits + extra_mux * dp.c("a_mux_um2") / dp.c("a_ff_um2")
}

fn phy_table(t: &Tables, kind: IoKind, serdes: Option<&SerdesSpec>, substrate_d2d: &str, dram: Option<&str>) -> &'static str {
    let _ = t;
    match kind {
        IoKind::Hbm => match dram {
            Some("hbm3") => "hbm3",
            Some("hbm3e") => "hbm3e",
            Some("hbm4") => "hbm4",
            Some("hbm2e") => "hbm2e",
            _ => "hbm2",
        },
        IoKind::Lpddr => "lpddr",
        IoKind::Pcie => "pcie",
        IoKind::Optical => "optical",
        IoKind::Vertical => "hybrid_bond",
        IoKind::D2d => {
            if substrate_d2d == "ucie_std" {
                "ucie_std"
            } else {
                "ucie_adv"
            }
        }
        IoKind::Serdes => match serdes {
            Some(s) if matches!(s.protocol, kiln_ir::hw::net::SerdesProtocol::Pcie) => "pcie",
            Some(s) if s.lane_rate_bits_per_s.0 > 112e9 => "serdes_224g",
            Some(s) if s.lane_rate_bits_per_s.0 > 56e9 => "serdes_112g",
            _ => "serdes_56g",
        },
    }
}

/// Substrate kind -> package table id.
pub fn package_table(kind: kiln_ir::hw::phys::SubstrateKind) -> &'static str {
    use kiln_ir::hw::phys::SubstrateKind::*;
    match kind {
        SiliconInterposer => "cowos_s",
        RdlInterposer => "cowos_l",
        Bridge => "emib",
        Organic | None => "organic",
    }
}

/// A characterized unit template: its spec, own-area result, leakage and energies.
type UnitCached = (std::sync::Arc<kiln_ir::hw::compute::ComputeUnit>, NodePhys, f64, UnitEnergy);

pub fn characterize(hw: &HwModel, params: &Params) -> Characterized {
    let t = Tables::get();
    let mut problems = vec![];
    let nn = hw.nodes.len();
    // Technology per node: nearest die container, stack logic die, else the design's tech.
    let sys_tech = hw.tree.first().and_then(|c| c.tech.clone()).unwrap_or_else(|| "tsmc_n7".into());
    let mut tech: Vec<String> = vec![String::new(); nn];
    for i in 0..nn {
        let own = match hw.nodes[i].ix {
            NodeIx::Container(c) if hw.tree[c].kind == ContainerKind::Die => hw.tree[c].tech.clone(),
            NodeIx::Mem(m) => match &hw.memories[m].spec {
                MemSpec::Stack(s) => s.logic_die.as_ref().map(|l| l.tech.as_ref().map_or_else(|| "dram_logic_1y".to_owned(), |t| t.node().to_owned())),
                _ => None,
            },
            _ => None,
        };
        tech[i] = own.unwrap_or_else(|| hw.nodes[i].parent.map_or_else(|| sys_tech.clone(), |p| tech[p].clone()));
    }
    let mut missing: BTreeMap<String, ()> = BTreeMap::new();
    let node_of = |i: usize, missing: &mut BTreeMap<String, ()>| -> &'static TechNode {
        t.node(&tech[i]).unwrap_or_else(|| {
            missing.insert(tech[i].clone(), ());
            t.node("tsmc_n7").expect("N7 table")
        })
    };
    let mut c = Ctx { t, p: params, lanes: BTreeMap::new() };
    let dp = &t.datapath;
    let mut nodes = vec![NodePhys::default(); nn];
    let mut units = vec![UnitEnergy::default(); hw.units.len()];
    let mut mems = vec![MemEnergy::default(); hw.memories.len()];
    let mut routers = vec![Router::default(); hw.routers.len()];
    let mut phys = vec![];
    let pkg_of = |mut n: usize| -> Option<usize> {
        loop {
            if let NodeIx::Container(ci) = hw.nodes[n].ix
                && hw.tree[ci].kind == ContainerKind::Package
            {
                return Some(ci);
            }
            n = hw.nodes[n].parent?;
        }
    };
    let substrate_d2d = |n: usize| -> String {
        pkg_of(n)
            .and_then(|ci| hw.tree[ci].package.as_ref())
            .map(|p| t.package.get(package_table(p.substrate.kind)).map_or_else(|| "ucie_adv".into(), |x| x.d2d_default.clone()))
            .unwrap_or_else(|| "ucie_adv".into())
    };
    // Node-scoped parameters once per table (params.get allocates its keys).
    let npar: BTreeMap<&str, (f64, f64, f64)> =
        t.nodes.keys().map(|id| (id.as_str(), (t.node(id).map_or(0.65, |n| params.util(n)), params.get("p_ll", Some(id)), params.get("p_ls", Some(id))))).collect();
    let util_of = |n: &TechNode| npar.get(n.id.as_str()).map_or(0.65, |x| x.0);
    let p_ll_of = |n: &TechNode| npar.get(n.id.as_str()).map_or(0.04, |x| x.1);
    let p_ls_of = |n: &TechNode| npar.get(n.id.as_str()).map_or(1.0, |x| x.2);
    let kpe = params.get("kappa_pe", None);
    let kctrl = params.get("kappa_ctrl", None);
    let ke_dp = params.get("kappa_e_dp", None);
    let ke_sram = params.get("kappa_e_sram", None);
    let ke_rf = params.get("kappa_e_rf", None);
    let ke_wire = params.get("kappa_e_wire", None);
    let ksram = params.get("kappa_sram_area", None);
    let misc_block = params.get("misc_block_mm2", None);
    let cache_pipe = params.get("cache_pipeline_cycles", None);
    let a_misc = params.get("a_misc_mm2", None);
    let mm_a = dp.c("mm_area");
    let mm_e = dp.c("mm_energy");
    let fma32 = blocks::fma_lane(dp, Precision::Fp32);

    // Instances of one template share their spec (same Arc) and node: characterized once.
    let mut unit_cache: BTreeMap<(String, String), UnitCached> = BTreeMap::new();
    for (ui, u) in hw.units.iter().enumerate() {
        let n = node_of(u.node, &mut missing);
        let enabled = hw.nodes[u.node].enabled;
        let key = (hw.nodes[u.node].entity.clone(), n.id.clone());
        if let Some((_, acc, leak, ue)) = unit_cache.get(&key).filter(|c| std::sync::Arc::ptr_eq(&c.0, &u.spec) || *c.0 == *u.spec) {
            let np = &mut nodes[u.node];
            for (i, p) in acc.parts.iter().enumerate() {
                np.parts[i] += p;
            }
            np.area_um2 += acc.area_um2;
            np.flops += acc.flops;
            np.ge += acc.ge;
            np.sram_bits += acc.sram_bits;
            np.leak_w += if enabled { *leak } else { 0.0 };
            units[ui] = ue.clone();
            continue;
        }
        let mut acc = NodePhys::default();
        let mut acc_leak = 0.0;
        let util = util_of(n);
        let s = &u.spec;
        let dp_area_45;
        let mut ff = 0.0;
        let mut ue = UnitEnergy { idle_frac: dp.c("idle_gated"), ..Default::default() };
        let (fma_a, fma_e) = blocks::scale(dp, n, fma32);
        let n_modes = s.precisions.len().max(1) as f64;
        let mm_e_k = if n_modes > 1.0 { 1.0 + mm_e } else { 1.0 };
        let mut ctrl_extra = 0.0;
        match &s.kind {
            ComputeKind::Matrix(mx) => {
                let l = match mx.geometry {
                    kiln_ir::hw::compute::Geometry::Mma { k, .. } => f64::from(k).min(dp.c("max_dot_len")),
                    kiln_ir::hw::compute::Geometry::Spatial { ref dims } => dims.get("k").map_or(1.0, |&k| f64::from(k).min(dp.c("max_dot_len"))),
                    _ => 1.0,
                };
                let flows = mx.dataflow.as_ref().map_or_else(|| vec![mx.geometry.default_dataflow()], |d| d.all());
                let mut widest: f64 = 0.0;
                for m in &s.precisions {
                    if let PrecisionMode::Mac { a, b, acc, .. } = m {
                        let extra = if l <= 1.0 { dataflow_bits(&flows, bits_of(a), bits_of(b), bits_of(acc), dp) } else { 0.0 };
                        let lane = c.lane(a.precision, b.precision, acc.precision, l, extra);
                        let lanes = s.kind.ops_per_cycle(m);
                        widest = widest.max(lanes * lane.area_um2);
                        ff = f64::max(ff, lanes * lane.ff_bits);
                        let (_, e) = blocks::scale(dp, n, lane);
                        ue.modes.push((format!("{a}*{b}+{acc}"), e * ke_dp * mm_e_k));
                    }
                }
                dp_area_45 = widest * (1.0 + mm_a * (n_modes - 1.0));
            }
            ComputeKind::Cim(cim) => {
                // 04 §4.11: bit-cell array + weight-set mux + bit products + adder tree / ADC + shift-accumulate.
                let (pr, cols, cb, bi) = (f64::from(cim.active_rows()), f64::from(cim.cols), f64::from(cim.cell_bits), f64::from(cim.input_bits_per_cycle));
                let sets = f64::from(cim.weight_sets.max(1));
                let arr_bits = f64::from(cim.rows) * cols * cb * sets;
                let mut arr = blocks::sram(n, dp, arr_bits, cols * cb, 1, Ports { read: 1, write: 1, rw: 0 }, (cb / 2.0).max(1.0));
                arr.area_um2 *= ksram;
                let a_pp = dp.c("a_pp_small_um2");
                let (a_add, e_add) = (dp.c("a_add_um2"), dp.c("e_add_fj") * 1e-3);
                let (a_ff, e_ff) = (dp.c("a_ff_um2"), dp.c("e_ff_fj") * 1e-3);
                let n_pp = pr * cols * cb * bi;
                let w_min = s.precisions.iter().filter_map(|m| if let PrecisionMode::Mac { b, .. } = m { Some(bits_of(b)) } else { None }).fold(f64::INFINITY, f64::min);
                let cpw = (w_min / cb).ceil().max(1.0);
                let w_acc = 32.0;
                let tree_w = cb + bi + (pr.log2().ceil()) / 2.0;
                let (a_tree, e_tree) = match cim.style {
                    kiln_ir::hw::compute::CimStyle::Digital => (a_add * cols * (pr - 1.0) * tree_w, e_add * cols * (pr - 1.0) * tree_w),
                    kiln_ir::hw::compute::CimStyle::Analog => {
                        let ab = f64::from(cim.adc_bits.unwrap_or_else(|| cim.boundary_adc_bits()));
                        (cols * 40.0 * 2f64.powf(ab / 2.0) * 45.0, cols * 2f64.powf(ab) * 0.005)
                    }
                };
                let a_sacc = (a_add + a_ff) * (cols / cpw) * w_acc;
                let a_mux = if sets > 1.0 { dp.c("a_mux_um2") * cols * cb * sets.log2().ceil() } else { 0.0 };
                let e_cycle = dp.c("alpha_dp") * (dp.c("e_pp_fj") * 1e-3 * n_pp + e_tree + e_ff * cols * w_acc / cpw);
                dp_area_45 = a_pp * n_pp + a_tree + a_sacc + a_mux;
                ff = cols * w_acc / cpw;
                let scale_e = 1e-12 * dp.c("s_45_n7") * n.s_e * ke_dp;
                for m in &s.precisions {
                    if let PrecisionMode::Mac { a, b, acc, .. } = m {
                        let macs = s.kind.ops_per_cycle(m).max(1e-9);
                        ue.modes.push((format!("{a}*{b}+{acc}"), e_cycle * scale_e / macs));
                    }
                }
                acc.parts[Part::Sram as usize] += arr.area_um2;
                acc.area_um2 += arr.area_um2;
                acc.sram_bits += arr_bits;
                acc_leak += arr.leak_w * p_ls_of(n) / n.p_ls_mw_per_mib;
            }
            ComputeKind::Vector(v) => {
                let base = f64::from(v.lanes) * f64::from(v.sublanes.max(1));
                let mut widest: f64 = 0.0;
                for m in &s.precisions {
                    if let PrecisionMode::Elem { dtype, rate } = m {
                        let lane = blocks::fma_lane(dp, dtype.precision);
                        let lanes = base * rate;
                        widest = widest.max(lanes * lane.area_um2);
                        ff = f64::max(ff, lanes * lane.ff_bits);
                        let (_, e) = blocks::scale(dp, n, lane);
                        ue.modes.push((dtype.to_string(), e * ke_dp * mm_e_k));
                    }
                }
                dp_area_45 = widest * (1.0 + mm_a * (n_modes - 1.0));
            }
            ComputeKind::Special(sp) => {
                dp_area_45 = f64::from(sp.lanes) * dp.c("sfu_area") * fma32.area_um2;
                ff = f64::from(sp.lanes) * fma32.ff_bits;
            }
            ComputeKind::Scalar(sc) => {
                dp_area_45 = f64::from(sc.issue_width) * fma32.area_um2;
                ff = f64::from(sc.issue_width) * fma32.ff_bits;
                ctrl_extra = dp.c("scalar_core_ge") * n.a_ge_um2;
            }
        }
        ue.e_elem_j = ue.modes.iter().find(|(k, _)| k == "fp32").map_or(fma_e * ke_dp, |x| x.1);
        ue.e_transc_j = dp.c("sfu_energy") * fma_e * ke_dp;
        let _ = fma_a;
        let (a_dp, _) = blocks::scale(dp, n, Lane { area_um2: dp_area_45, ..Default::default() });
        let a_dp = kpe * a_dp / util;
        let a_ctrl = s.power.ctrl_ge.map_or(kctrl * a_dp, |g| g * n.a_ge_um2 / util) + ctrl_extra / util;
        let derived = a_dp + a_ctrl;
        let area = s.power.area.map_or(derived, |a| (a.0 * 1e6).max(derived)).max(s.footprint.as_ref().and_then(|f| f.area).map_or(0.0, |a| a.0 * 1e6));
        for (k, e) in &s.power.energy_per_op {
            if let Some(x) = ue.modes.iter_mut().find(|(m, _)| m.starts_with(k.as_str())) {
                x.1 = x.1.max(e.0);
            }
        }
        let scale = if derived > 0.0 { area / derived } else { 1.0 };
        acc.parts[Part::Datapath as usize] += a_dp * scale;
        acc.parts[Part::Control as usize] += a_ctrl * scale;
        acc.area_um2 += area;
        acc.flops += ff;
        acc.ge += area * util / n.a_ge_um2;
        acc_leak += s.power.leakage.map_or(area * 1e-6 * p_ll_of(n), |l| l.0);
        let np = &mut nodes[u.node];
        for (i, p) in acc.parts.iter().enumerate() {
            np.parts[i] += p;
        }
        np.area_um2 += acc.area_um2;
        np.flops += acc.flops;
        np.ge += acc.ge;
        np.sram_bits += acc.sram_bits;
        np.leak_w += if enabled { acc_leak } else { 0.0 };
        units[ui] = ue.clone();
        unit_cache.insert(key, (u.spec.clone(), acc, acc_leak, ue));
    }

    // SIMT sub-cores (04 §5 ctrl_ge): a cluster whose MAC/vector units feed from a register file inside it, under
    // a host-launched execution model, pays warp scheduling, operand collection, LSU and TEX control.
    let mut op_ctrl_ge = vec![0.0; nn];
    if hw.exec_model == ExecModel::HostLaunched {
        let simt = params.get("simt_ctrl_ge", None);
        let simt_op = params.get("simt_operand_ge_per_bit", None);
        let mut simt_cluster = vec![false; nn];
        op_ctrl_ge = vec![0.0; nn];
        for u in &hw.units {
            let Some(cn) = hw.nodes[u.node].parent else { continue };
            if matches!(u.spec.kind, ComputeKind::Matrix(_) | ComputeKind::Vector(_))
                && u.feeds.values().any(|f| hw.nodes[hw.memories[f.mem].node].parent == Some(cn) && hw.memories[f.mem].onchip_kind() == Some(MemKind::RegisterFile))
            {
                simt_cluster[cn] = true;
            }
        }
        // Harvested (disabled) instances have no resolved feeds but are placed and counted in area (04 §5): a
        // cluster is SIMT when any instance of its template entity is.
        let simt_entities: std::collections::BTreeSet<&str> = (0..nn).filter(|&i| simt_cluster[i]).map(|i| hw.nodes[i].entity.as_str()).collect();
        for ci in 0..hw.tree.len() {
            let cn = hw.tree[ci].node;
            let is_simt = hw.tree[ci].kind == ContainerKind::Cluster && (simt_cluster[cn] || simt_entities.contains(hw.nodes[cn].entity.as_str()));
            if is_simt {
                let n = node_of(cn, &mut missing);
                let util = util_of(n);
                // Matrix-operand delivery (register-bank reads, operand staging and the collector-to-tensor-core
                // network) grows with the operand bits the sub-core's matrix units consume per cycle.
                let op_bits: f64 = hw.nodes[cn]
                    .children
                    .iter()
                    .filter_map(|&k| if let NodeIx::Unit(ui) = hw.nodes[k].ix { Some(&hw.units[ui].spec) } else { None })
                    .filter(|s| matches!(s.kind, ComputeKind::Matrix(_)))
                    .map(|s| {
                        s.precisions
                            .iter()
                            .filter_map(|m| if let PrecisionMode::Mac { a, b, .. } = m { Some(s.kind.ops_per_cycle(m) * (bits_of(a) + bits_of(b))) } else { None })
                            .fold(0.0, f64::max)
                    })
                    .sum();
                let ge = simt + simt_op * op_bits;
                let a = ge * n.a_ge_um2 / util;
                op_ctrl_ge[cn] += simt_op * op_bits;
                let np = &mut nodes[cn];
                np.parts[Part::Control as usize] += a;
                np.area_um2 += a;
                np.unit_ctrl_um2 += a;
                np.ge += ge;
                np.flops += ge / 10.0;
                if hw.nodes[cn].enabled {
                    np.leak_w += a * 1e-6 * p_ll_of(n);
                    np.unit_ctrl_leak_w += a * 1e-6 * p_ll_of(n);
                }
            }
        }
    }

    for (mi, m) in hw.memories.iter().enumerate() {
        let n = node_of(m.node, &mut missing);
        let enabled = hw.nodes[m.node].enabled;
        let f = m.clock.and_then(|c| hw.clocks.get(c)).map_or(1e9, |c| c.spec.freq.0);
        let cyc = 1.0 / f;
        let p_ls_k = p_ls_of(n) / n.p_ls_mw_per_mib;
        match &m.spec {
            MemSpec::Stack(s) => {
                let kind = serde_json::to_value(s.kind).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_else(|| "custom".into());
                let d = t.dram.get(&kind).or_else(|| t.dram.get("custom")).expect("custom DRAM table");
                let e_core = params.get("e_dram_core", Some(&kind)) * 1e-12 * 8.0;
                let table = phy_table(t, IoKind::Hbm, None, "", Some(&kind));
                let e_phy = t.phy.get(table).map_or(0.6, |p| p.e_pj_per_bit) * 1e-12 * 8.0;
                let e = s.overrides.energy_per_byte.map_or(e_core + e_phy, |o| o.0.max(e_core + e_phy));
                mems[mi] = MemEnergy {
                    read_j_per_b: e,
                    write_j_per_b: e,
                    latency_s: 400e-9,
                    dram: true,
                    e_core_j_per_b: e - e_phy,
                    e_phy_j_per_b: e_phy,
                    background_w: if enabled { s.overrides.power.map_or(d.background_w, |p| p.0.max(d.background_w)) } else { 0.0 },
                };
            }
            MemSpec::Local { buffer, .. } => {
                let bits = buffer.capacity.0 as f64 * 8.0;
                let r = if bits <= 32768.0 {
                    blocks::flop_array(n, bits, 64.0, Ports { read: 1, write: 1, rw: 0 }, false)
                } else {
                    let mut r = blocks::sram(n, &t.datapath, bits, 256.0, 1, Ports { read: 1, write: 1, rw: 0 }, 1.0);
                    r.area_um2 *= ksram;
                    r
                };
                let np = &mut nodes[m.node];
                let part = if bits <= 32768.0 { Part::Rf } else { Part::Sram };
                np.parts[part as usize] += r.area_um2;
                np.area_um2 += r.area_um2;
                np.sram_bits += bits;
                if bits <= 32768.0 {
                    np.flops += bits;
                }
                np.leak_w += if enabled { r.leak_w * p_ls_k + if bits <= 32768.0 { r.area_um2 * 1e-6 * p_ll_of(n) } else { 0.0 } } else { 0.0 };
                mems[mi] = MemEnergy { read_j_per_b: 8.0 * r.e_read_bit * ke_rf, write_j_per_b: 8.0 * r.e_write_bit * ke_rf, latency_s: cyc, ..Default::default() };
            }
            MemSpec::OnChip(spec) => {
                let bits = m.capacity.0 as f64 * 8.0;
                let (ports, word) = ports_of(spec);
                let imp = match spec.implementation {
                    MemImpl::Auto => match spec.kind {
                        MemKind::RegisterFile if bits <= 16384.0 => MemImpl::Flop,
                        MemKind::RegisterFile => MemImpl::SramBanked,
                        MemKind::Fifo if bits <= 32768.0 => MemImpl::Flop,
                        _ => MemImpl::SramMacro,
                    },
                    i => i,
                };
                let cell_k = match (imp, spec.bitcell) {
                    (MemImpl::Edram, _) => 0.6,
                    (MemImpl::Mram, _) => 0.8,
                    (_, Some(kiln_ir::hw::compute::Bitcell::Hc)) => n.hc_factor,
                    _ => 1.0,
                };
                let mut tag_bits = 0.0;
                if let (MemKind::Cache, Some(cs)) = (spec.kind, &spec.cache) {
                    let lines = m.capacity.0 as f64 / (cs.line.0 as f64).max(1.0);
                    let sets = (lines / f64::from(cs.ways.max(1))).max(1.0);
                    tag_bits = lines * ((48.0 - sets.log2() - (cs.line.0 as f64).max(1.0).log2()).max(8.0) + 4.0);
                }
                let r: Sram = match imp {
                    MemImpl::Flop | MemImpl::LatchArray => blocks::flop_array(n, bits, word, ports, imp == MemImpl::LatchArray),
                    _ => {
                        let mut r = blocks::sram(n, &t.datapath, bits + tag_bits, word, spec.banks, ports, cell_k);
                        r.area_um2 *= ksram;
                        r
                    }
                };
                let flop = matches!(imp, MemImpl::Flop | MemImpl::LatchArray);
                let ke = if spec.kind == MemKind::RegisterFile { ke_rf } else { ke_sram };
                let o = &spec.overrides;
                let area = o.area.map_or(r.area_um2, |a| (a.0 * 1e6).max(r.area_um2));
                let np = &mut nodes[m.node];
                let part = if spec.kind == MemKind::RegisterFile || flop { Part::Rf } else { Part::Sram };
                np.parts[part as usize] += area;
                np.area_um2 += area;
                np.sram_bits += bits;
                if flop {
                    np.flops += bits;
                } else {
                    // Macro periphery (decoders, sense amps, drivers, control) is logic: it counts toward the transistor
                    // estimate (the bit cells are counted per bit).
                    np.ge += (area - bits * n.cell_um2).max(0.0) * util_of(n) / n.a_ge_um2;
                }
                let leak = if flop { r.area_um2 * 1e-6 * p_ll_of(n) } else { r.leak_w * p_ls_k };
                let leak = o.leakage.map_or(leak, |l| l.0.max(leak));
                // Gating (04 §4.7) discounts only idle windows; a characterized memory leaks at its active rate.
                np.leak_w += if enabled { leak } else { 0.0 };
                let pipe = if spec.kind == MemKind::Cache { cache_pipe } else { 0.0 };
                let lat = ((r.t_acc_s * f).ceil() + 1.0 + pipe) * cyc;
                let lat = o.latency.map_or(lat, |l| (l.0 * cyc).max(lat));
                mems[mi] = MemEnergy {
                    read_j_per_b: o.read_energy.map_or(8.0 * r.e_read_bit * ke, |e| e.0.max(8.0 * r.e_read_bit * ke)),
                    write_j_per_b: o.write_energy.map_or(8.0 * r.e_write_bit * ke, |e| e.0.max(8.0 * r.e_write_bit * ke)),
                    latency_s: lat,
                    ..Default::default()
                };
            }
        }
    }

    for (ri, ro) in hw.routers.iter().enumerate() {
        let net = &hw.networks[ro.net];
        let n = node_of(ro.node, &mut missing);
        let util = util_of(n);
        let chans = hw.out_edges[ro.node].len().max(2) as u32;
        let rs = &net.spec.router;
        let p = rs.radix.unwrap_or(chans).max(2);
        let w = f64::from(net.spec.flit_bits.or(net.spec.link.width_bits).unwrap_or(128));
        let pipe = rs.pipeline.map_or(2, |c| c.0 as u32);
        let mut r = match net.spec.topology {
            kiln_ir::hw::net::Topology::Bus { .. } => blocks::bus(n, chans, w, util),
            _ => blocks::router(n, p, rs.vcs.unwrap_or(2), rs.input_buffer_flits.unwrap_or(4), w, pipe, util),
        };
        // 04 §7.4: an allocator slower than the clock period takes extra stages, or rejects a declared pipeline.
        let clock = net.clock.or_else(|| hw.out_edges[ro.node].iter().find_map(|&c| hw.channels[c].clock));
        let period = 1.0 / clock.and_then(|c| hw.clocks.get(c)).map_or(1e9, |c| c.spec.freq.0);
        if r.t_cycle_min_s > period * (1.0 + 1e-9) {
            // The undeclared two-stage router with the allocator spread over as many stages as it needs.
            let need = (r.t_cycle_min_s / period).ceil() as u32 + 1;
            match rs.pipeline {
                Some(_) if pipe >= need => {}
                Some(_) if hw.nodes[ro.node].enabled => problems.push(
                    Diagnostic::error(
                        "E-PHYS-ROUTER-TIMING",
                        format!("{}: router allocation needs {:.1} ps, the clock period is {:.1} ps: the declared {pipe}-stage pipeline cannot close timing", hw.nodes[ro.node].path, r.t_cycle_min_s * 1e12, period * 1e12),
                    )
                    .at(&hw.nodes[ro.node].path)
                    .hint(format!("declare router.pipeline >= {need} or lower the network clock")),
                ),
                Some(_) => {}
                None => r.n_pipe = need,
            }
        }
        let area = rs.power.area.map_or(r.area_um2, |a| (a.0 * 1e6).max(r.area_um2));
        let np = &mut nodes[ro.node];
        np.parts[Part::Noc as usize] += area;
        np.area_um2 += area;
        np.flops += r.ff_bits;
        np.leak_w += if hw.nodes[ro.node].enabled { rs.power.leakage.map_or(r.logic_um2 * 1e-6 * p_ll_of(n), |l| l.0) } else { 0.0 };
        routers[ri] = Router { e_flit_j: r.e_flit_j * ke_wire, ..r };
    }

    let a_ge_n7 = t.node("tsmc_n7").map_or(0.0439, |x| x.a_ge_um2);
    // Logic area per GE relative to N7 (misc logic quoted in N7 mm^2 scales like every other logic block).
    let logic_rel = |n: &TechNode| (n.a_ge_um2 / util_of(n)) / (a_ge_n7 / t.node("tsmc_n7").map_or(0.65, &util_of));
    let add_phy = |node: usize, kind: IoKind, units_n: f64, table: &str, nodes: &mut Vec<NodePhys>, phys: &mut Vec<PhyUse>, n: &TechNode, enabled: bool| {
        let Some(pt): Option<&PhyT> = t.phy.get(table) else { return 0.0 };
        let area = pt.area_mm2_n7 * 1e6 * units_n * (n.a_ge_um2 / a_ge_n7).sqrt();
        let np = &mut nodes[node];
        np.parts[Part::Phy as usize] += area;
        np.area_um2 += area;
        np.leak_w += if enabled { pt.leak_w * units_n } else { 0.0 };
        phys.push(PhyUse {
            node,
            table: table.to_owned(),
            kind,
            units: units_n,
            shoreline_um: pt.shoreline_mm * 1000.0 * units_n,
            area_um2: area,
            e_j_per_bit: pt.e_pj_per_bit * 1e-12,
            latency_s: pt.latency_ns * 1e-9,
        });
        area
    };
    let stack_kind = |mi: usize| hw.memories[mi].dram_kind().and_then(|k| serde_json::to_value(k).ok()).and_then(|v| v.as_str().map(String::from));
    for b in &hw.blocks {
        let n = node_of(b.node, &mut missing);
        let util = util_of(n);
        let enabled = hw.nodes[b.node].enabled;
        let (area, part, counted) = match &b.spec.kind {
            BlockKind::Phy(ps) => {
                // The stack this PHY serves (channel to a stack memory) picks the HBM generation.
                let dram = hw.out_edges[b.node]
                    .iter()
                    .map(|&ch| hw.channels[ch].dst)
                    .chain(hw.channels.iter().filter(|ch| hw.node_of(ch.dst) == b.node).map(|ch| ch.src))
                    .find_map(|x| if let NodeIx::Mem(mi) = x { stack_kind(mi) } else { None });
                let table = phy_table(t, ps.for_kind, None, &substrate_d2d(b.node), dram.as_deref());
                let units_n = match ps.for_kind {
                    IoKind::Hbm => f64::from(ps.lanes.unwrap_or(1024)) / if dram.as_deref() == Some("hbm4") { 2048.0 } else { 1024.0 },
                    IoKind::Lpddr => f64::from(ps.lanes.unwrap_or(64)) / 64.0,
                    _ => f64::from(ps.lanes.unwrap_or(1)),
                };
                let a = add_phy(b.node, ps.for_kind, units_n, table, &mut nodes, &mut phys, n, enabled);
                (a, Part::Phy, a)
            }
            BlockKind::MemController(_) => (0.0, Part::Misc, 0.0),
            BlockKind::Dma(_) => (dp.c("dma_ge") * n.a_ge_um2 / util, Part::Control, 0.0),
            BlockKind::Sequencer { .. } => (dp.c("sequencer_ge") * n.a_ge_um2 / util, Part::Control, 0.0),
            BlockKind::Misc { .. } => (misc_block * 1e6 * logic_rel(n), Part::Misc, 0.0),
        };
        let declared = b.spec.footprint.as_ref().and_then(|f| f.area).map(|a| a.0 * 1e6).or(b.spec.power.area.map(|a| a.0 * 1e6));
        let area = match (&b.spec.kind, declared) {
            (BlockKind::Misc { .. }, Some(d)) => d,
            (_, Some(d)) => d.max(area),
            _ => area,
        } - counted;
        let np = &mut nodes[b.node];
        np.parts[part as usize] += area;
        np.area_um2 += area;
        if enabled {
            np.leak_w += b.spec.power.leakage.map_or(area * 1e-6 * p_ll_of(n), |l| l.0);
        }
    }
    for p in &hw.ports {
        let n = node_of(p.node, &mut missing);
        let serdes = p.spec.link.as_ref().and_then(|l| if let LinkPhys::Serdes(s) = &l.phys { Some(s) } else { None });
        let table = phy_table(t, p.spec.kind, serdes, &substrate_d2d(p.node), None);
        let units_n = match (&p.spec.kind, p.spec.link.as_ref().map(|l| &l.phys)) {
            (_, Some(LinkPhys::Serdes(s))) => f64::from(s.lanes),
            (_, Some(LinkPhys::Optical(o))) => f64::from(o.lanes),
            (_, Some(LinkPhys::D2d(d))) => f64::from(d.modules),
            (IoKind::Pcie, _) => 16.0,
            (IoKind::Vertical, _) => 0.0,
            _ => 1.0,
        };
        add_phy(p.node, p.spec.kind, units_n, table, &mut nodes, &mut phys, n, hw.nodes[p.node].enabled);
    }

    // Chip-level misc (fuses, PLLs, management, test) once per compute die.
    for ci in 0..hw.tree.len() {
        let cn = hw.tree[ci].node;
        if hw.tree[ci].kind == ContainerKind::Die && hw.tree[ci].die.as_ref().is_none_or(|d| d.role == kiln_ir::hw::types::DieRole::Compute) {
            let n = node_of(cn, &mut missing);
            let a = a_misc * 1e6 * logic_rel(n);
            let a = hw.tree[ci].die.as_ref().and_then(|d| d.power.area).map_or(a, |x| (x.0 * 1e6).max(a));
            let np = &mut nodes[cn];
            np.parts[Part::Misc as usize] += a;
            np.area_um2 += a;
            np.leak_w += a * 1e-6 * p_ll_of(n);
        }
    }
    for (i, np) in nodes.iter_mut().enumerate() {
        let n = node_of(i, &mut missing);
        np.ctrl_ge = np.parts[Part::Control as usize] * util_of(n) / n.a_ge_um2 - op_ctrl_ge[i];
        if np.ge == 0.0 {
            let logic = np.parts[Part::Datapath as usize] + np.parts[Part::Control as usize] + np.parts[Part::Noc as usize] + np.parts[Part::Misc as usize];
            np.ge = logic * util_of(n) / n.a_ge_um2;
        }
    }
    for k in missing.keys() {
        problems.push(
            Diagnostic::error("E-PHYS-NODE-MISSING-FIELD", format!("technology node '{k}' has no kiln-phys table yet; N7 values used"))
                .hint("open-PDK and predictive nodes are built by the data-build step (04 §3); use a TSMC node"),
        );
    }
    let _ = &c.p;
    // Canonical order without names or declaration order (P4): a structural signature per subtree (node kind,
    // own area, instance index and coordinates, sorted child signatures); siblings sort by (signature, index), and
    // only structural twins fall back to arena order.
    let fnv = |h: u64, x: u64| (h ^ x).wrapping_mul(0x0000_0100_0000_01B3);
    let mut sig = vec![0u64; nn];
    for i in (0..nn).rev() {
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
        let mut h = fnv(0xcbf2_9ce4_8422_2325, kind);
        h = fnv(h, nodes[i].area_um2.to_bits());
        h = fnv(h, u64::from(n.index));
        for c in &n.coord {
            h = fnv(h, u64::from(*c));
        }
        let mut cs: Vec<u64> = n.children.iter().map(|&c| sig[c]).collect();
        cs.sort_unstable();
        for c in cs {
            h = fnv(h, c);
        }
        sig[i] = h;
    }
    let kids: Vec<Vec<usize>> = hw
        .nodes
        .iter()
        .map(|n| {
            let mut k = n.children.clone();
            k.sort_by_key(|&c| (sig[c], hw.nodes[c].index, c));
            k
        })
        .collect();
    let mut order = Vec::with_capacity(nn);
    let mut st = vec![0usize];
    while let Some(x) = st.pop() {
        order.push(x);
        st.extend(kids[x].iter().rev().copied());
    }
    let mut rank = vec![0; nn];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    phys.sort_by_key(|p| rank[p.node]);
    Characterized { kids, order, rank, nodes, tech, units, mems, routers, phys, problems }
}

impl Characterized {
    /// Energy per MAC of unit `u` in `mode` (`a*b+acc`), else the closest mode by operand bits.
    pub fn e_mac(&self, u: usize, mode: &str, bits: (u32, u32)) -> f64 {
        let ue = &self.units[u];
        if let Some((_, e)) = ue.modes.iter().find(|(k, _)| !mode.is_empty() && k.starts_with(mode)) {
            return *e;
        }
        let want = f64::from(bits.0 * bits.1);
        ue.modes
            .iter()
            .filter(|(k, _)| k.contains('*'))
            .min_by(|a, b| (bits_product(&a.0) - want).abs().total_cmp(&(bits_product(&b.0) - want).abs()))
            .map_or(0.0, |x| x.1)
    }

    /// Energy per elementwise op of unit `u` in vector mode `mode` (`fp32@1` or `fp32`).
    pub fn e_elem(&self, u: usize, mode: &str) -> f64 {
        let ue = &self.units[u];
        let dt = mode.split('@').next().unwrap_or(mode);
        ue.modes.iter().find(|(k, _)| k == dt).map_or(ue.e_elem_j, |x| x.1)
    }
}

fn bits_product(key: &str) -> f64 {
    let (a, rest) = key.split_once('*').unwrap_or((key, ""));
    let b = rest.split('+').next().unwrap_or("");
    let bits = |s: &str| Precision::from_name(s).map_or(16.0, |p| f64::from(p.element_bits()));
    bits(a) * bits(b)
}
