//! Typed views of the data files (04 §2-§4): technology nodes, datapath anchors, DRAM kinds, PHYs, packages,
//! the calibration parameter registry and fitted calibration sets. Parsed once per process.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::data;
use crate::sourced::{Sourced, check_sourced};

/// Flattened `a.b.c` path -> sourced value of one data file.
pub type SourcedMap = BTreeMap<String, Sourced>;

fn flatten(v: &Value) -> SourcedMap {
    fn walk(path: &str, v: &Value, out: &mut SourcedMap) {
        if let Value::Object(o) = v {
            if o.contains_key("v") && o.contains_key("q") {
                if let Ok(s) = serde_json::from_value::<Sourced>(v.clone()) {
                    out.insert(path.to_owned(), s);
                }
                return;
            }
            for (k, x) in o {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                walk(&p, x, out);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk("", v, &mut out);
    out
}

fn req(m: &SourcedMap, file: &str, key: &str) -> Result<f64, Diagnostic> {
    m.get(key).map(|s| s.v).ok_or_else(|| {
        Diagnostic::error("E-PHYS-NODE-MISSING-FIELD", format!("{file}: required field {key} missing")).at(format!("{file}:{key}"))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct WireClass {
    pub layers: f64,
    pub pitch_nm: f64,
    pub r_ohm_um: f64,
    pub c_ff_um: f64,
    pub routing_frac: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireClassId {
    Local,
    Intermediate,
    SemiGlobal,
    Global,
}

impl WireClassId {
    pub const ALL: [WireClassId; 4] = [WireClassId::Local, WireClassId::Intermediate, WireClassId::SemiGlobal, WireClassId::Global];
    pub fn key(self) -> &'static str {
        match self {
            WireClassId::Local => "local",
            WireClassId::Intermediate => "intermediate",
            WireClassId::SemiGlobal => "semi_global",
            WireClassId::Global => "global",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TechNode {
    pub id: String,
    pub a_ge_um2: f64,
    pub util_std_cell: f64,
    /// Realized logic density of this node's products relative to N7's, over the ratio of published peak densities
    /// (04 §4.1): 1 at N7; scales the N7-fitted `util_std_cell` on nodes without their own fit.
    pub density_realization: f64,
    pub vdd_nom: f64,
    pub s_e: f64,
    pub c_ge_ff: f64,
    pub cell_um2: f64,
    pub hc_factor: f64,
    pub cell_aspect: f64,
    pub n_hp: f64,
    pub n_wp: f64,
    pub a_fix_um2: f64,
    pub g_ctrl: f64,
    pub t_s0_ns: f64,
    pub t_s1_ns: f64,
    pub p_ls_mw_per_mib: f64,
    pub a_bit_flop_um2: f64,
    pub a_bit_latch_um2: f64,
    pub k_w: f64,
    pub e_rf0_fj: f64,
    pub wires: [WireClass; 4],
    pub r0c0_ps: f64,
    pub gamma: f64,
    pub t_setup_clkq_ps: f64,
    pub f_logic: f64,
    pub p_ll_w_mm2: f64,
    pub k_v: f64,
    pub k_t: f64,
    pub v_th: f64,
    pub alpha: f64,
    pub v_min: f64,
    pub v_max: f64,
    pub c_clk_flop_ff: f64,
    pub c_clk_area_pf_mm2: f64,
    pub d0_per_cm2: f64,
    pub wafer_usd: f64,
    pub seal_ring_um: f64,
    #[serde(skip)]
    pub sourced: SourcedMap,
}

impl TechNode {
    fn load(file: &str, v: &Value) -> Result<TechNode, Diagnostic> {
        let m = flatten(v);
        let g = |k: &str| req(&m, file, k);
        let wire = |c: &str| -> Result<WireClass, Diagnostic> {
            Ok(WireClass {
                layers: g(&format!("wires.{c}.layers"))?,
                pitch_nm: g(&format!("wires.{c}.pitch_nm"))?,
                r_ohm_um: g(&format!("wires.{c}.r_ohm_um"))?,
                c_ff_um: g(&format!("wires.{c}.c_ff_um"))?,
                routing_frac: g(&format!("wires.{c}.routing_frac"))?,
            })
        };
        Ok(TechNode {
            id: v["id"].as_str().unwrap_or_default().to_owned(),
            a_ge_um2: g("logic.nand2_area_um2")?,
            util_std_cell: g("logic.util_std_cell")?,
            density_realization: m.get("logic.density_realization").map_or(1.0, |s| s.v),
            vdd_nom: g("logic.vdd_nom_v")?,
            s_e: g("logic.s_e")?,
            c_ge_ff: g("logic.c_ge_ff")?,
            cell_um2: g("sram.bitcell_hd_um2")?,
            hc_factor: g("sram.hc_factor")?,
            cell_aspect: g("sram.aspect_w_over_h")?,
            n_hp: g("sram.n_hp")?,
            n_wp: g("sram.n_wp")?,
            a_fix_um2: g("sram.a_fix_um2")?,
            g_ctrl: g("sram.g_ctrl")?,
            t_s0_ns: g("sram.t_s0_ns")?,
            t_s1_ns: g("sram.t_s1_ns")?,
            p_ls_mw_per_mib: g("sram.p_ls_mw_per_mib")?,
            a_bit_flop_um2: g("rf.a_bit_flop_um2")?,
            a_bit_latch_um2: g("rf.a_bit_latch_um2")?,
            k_w: g("rf.k_w")?,
            e_rf0_fj: g("rf.e_rf0_fj_per_bit")?,
            wires: [wire("local")?, wire("intermediate")?, wire("semi_global")?, wire("global")?],
            r0c0_ps: g("driver.r0c0_ps")?,
            gamma: g("driver.gamma")?,
            t_setup_clkq_ps: g("driver.t_setup_clkq_ps")?,
            f_logic: g("driver.f_logic")?,
            p_ll_w_mm2: g("leakage.p_ll_w_mm2")?,
            k_v: g("leakage.k_v")?,
            k_t: g("leakage.k_t")?,
            v_th: g("dvfs.v_th")?,
            alpha: g("dvfs.alpha")?,
            v_min: g("dvfs.v_min")?,
            v_max: g("dvfs.v_max")?,
            c_clk_flop_ff: g("clock.c_clk_flop_ff")?,
            c_clk_area_pf_mm2: g("clock.c_clk_area_pf_mm2")?,
            d0_per_cm2: g("misc.d0_per_cm2")?,
            wafer_usd: g("misc.wafer_usd")?,
            seal_ring_um: g("misc.seal_ring_um")?,
            sourced: m,
        })
    }

    pub fn wire(&self, c: WireClassId) -> WireClass {
        self.wires[c as usize]
    }
}

/// 45 nm datapath anchors and coefficients (04 §4.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Datapath {
    pub v: SourcedMap,
    pub formats: BTreeMap<String, (f64, f64)>,
    coeffs: BTreeMap<String, f64>,
    anchors: BTreeMap<String, (f64, f64)>,
}

impl Datapath {
    fn new(v: SourcedMap, formats: BTreeMap<String, (f64, f64)>) -> Datapath {
        let coeffs = v.iter().filter_map(|(k, s)| k.strip_prefix("coeffs.").map(|c| (c.to_owned(), s.v))).collect();
        let mut anchors: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for (k, s) in &v {
            if let Some(rest) = k.strip_prefix("anchors.")
                && let Some((op, f)) = rest.split_once('.')
            {
                let e = anchors.entry(op.to_owned()).or_default();
                if f == "a_um2" { e.0 = s.v } else { e.1 = s.v }
            }
        }
        Datapath { v, formats, coeffs, anchors }
    }

    pub fn c(&self, k: &str) -> f64 {
        *self.coeffs.get(k).unwrap_or_else(|| panic!("datapath coefficient {k} (checked at load)"))
    }

    pub fn anchor(&self, op: &str) -> (f64, f64) {
        self.anchors.get(op).copied().unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DramKindT {
    pub e_core_pj_per_bit: f64,
    pub e_act_nj: f64,
    pub footprint_w_mm: f64,
    pub footprint_h_mm: f64,
    pub background_w: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhyT {
    pub id: String,
    pub unit: String,
    pub shoreline_mm: f64,
    pub e_pj_per_bit: f64,
    pub area_mm2_n7: f64,
    pub latency_ns: f64,
    pub leak_w: f64,
    pub pitch_um: Option<f64>,
    pub signal_frac: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PackageT {
    pub id: String,
    pub max_mm2: f64,
    pub die_gap_mm: f64,
    pub hbm_gap_mm: f64,
    pub trace_e_pj_per_bit_mm: f64,
    pub trace_ps_per_mm: f64,
    pub overhang_max_mm: f64,
    pub d2d_default: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamReg {
    pub role: String,
    pub scope: String,
    pub unit: String,
    pub prior: Sourced,
    pub group: String,
}

/// One fitted value in a calibration set (`calib/<platform>-<date>.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibEntry {
    pub name: String,
    /// Node id for node-scoped parameters, DRAM kind for `e_dram_core`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub value: f64,
    /// Log-space posterior sigma (Laplace); the range is `value * exp(+-sigma)`.
    pub sigma: f64,
    pub prior: f64,
    pub bounds: [f64; 2],
    #[serde(default)]
    pub at_bound: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibSet {
    pub schema: String,
    pub id: String,
    pub scope: String,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default)]
    pub params: Vec<CalibEntry>,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub targets_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<Value>,
}

pub struct Tables {
    pub nodes: BTreeMap<String, TechNode>,
    pub aliases: BTreeMap<String, String>,
    pub datapath: Datapath,
    pub dram: BTreeMap<String, DramKindT>,
    pub phy: BTreeMap<String, PhyT>,
    pub package: BTreeMap<String, PackageT>,
    pub params: BTreeMap<String, ParamReg>,
    pub calib: Vec<CalibSet>,
    pub sources: serde_json::Map<String, Value>,
    /// Problems found at load (bare numbers, unknown sources, missing fields); empty in a valid build.
    pub problems: Vec<Diagnostic>,
}

fn parse(path: &str) -> Value {
    serde_json::from_str(data::file(path).unwrap_or("{}")).unwrap_or(Value::Null)
}

impl Tables {
    pub fn get() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(Tables::load)
    }

    fn load() -> Tables {
        let mut problems = vec![];
        let sources = parse("sources.json")["sources"].as_object().cloned().unwrap_or_default();
        for (p, t) in data::FILES {
            match serde_json::from_str::<Value>(t) {
                Ok(v) if *p != "sources.json" && !p.starts_with("calib/") => check_sourced(p, &v, &sources, &mut problems),
                Ok(_) => {}
                Err(e) => problems.push(Diagnostic::error("E-PHYS-UNSOURCED", format!("{p}: not JSON: {e}"))),
            }
        }
        let mut nodes = BTreeMap::new();
        let mut dram = BTreeMap::new();
        let mut phy = BTreeMap::new();
        let mut package = BTreeMap::new();
        let mut calib = vec![];
        for (p, _) in data::FILES {
            let v = parse(p);
            let id = p.rsplit('/').next().unwrap_or(p).trim_end_matches(".json").to_owned();
            let m = flatten(&v);
            let g = |k: &str| req(&m, p, k);
            if p.starts_with("tech/") && id != "aliases" {
                match TechNode::load(p, &v) {
                    Ok(n) => {
                        nodes.insert(id, n);
                    }
                    Err(e) => problems.push(e),
                }
            } else if p.starts_with("dram/") {
                let r = (|| -> Result<DramKindT, Diagnostic> {
                    Ok(DramKindT {
                        e_core_pj_per_bit: g("e_core_pj_per_bit")?,
                        e_act_nj: g("e_act_nj")?,
                        footprint_w_mm: g("footprint_w_mm")?,
                        footprint_h_mm: g("footprint_h_mm")?,
                        background_w: g("background_w")?,
                    })
                })();
                match r {
                    Ok(d) => {
                        dram.insert(id, d);
                    }
                    Err(e) => problems.push(e),
                }
            } else if p.starts_with("phy/") {
                let r = (|| -> Result<PhyT, Diagnostic> {
                    Ok(PhyT {
                        id: id.clone(),
                        unit: v["unit"].as_str().unwrap_or("unit").to_owned(),
                        shoreline_mm: g("shoreline_mm")?,
                        e_pj_per_bit: g("e_pj_per_bit")?,
                        area_mm2_n7: g("area_mm2_n7")?,
                        latency_ns: g("latency_ns")?,
                        leak_w: g("leak_w")?,
                        pitch_um: m.get("pitch_um").map(|s| s.v),
                        signal_frac: m.get("signal_frac").map(|s| s.v),
                    })
                })();
                match r {
                    Ok(x) => {
                        phy.insert(id, x);
                    }
                    Err(e) => problems.push(e),
                }
            } else if p.starts_with("package/") {
                let r = (|| -> Result<PackageT, Diagnostic> {
                    Ok(PackageT {
                        id: id.clone(),
                        max_mm2: g("max_mm2")?,
                        die_gap_mm: g("die_gap_mm")?,
                        hbm_gap_mm: g("hbm_gap_mm")?,
                        trace_e_pj_per_bit_mm: g("trace_e_pj_per_bit_mm")?,
                        trace_ps_per_mm: g("trace_ps_per_mm")?,
                        overhang_max_mm: g("overhang_max_mm")?,
                        d2d_default: v["d2d_default"].as_str().unwrap_or("ucie_adv").to_owned(),
                    })
                })();
                match r {
                    Ok(x) => {
                        package.insert(id, x);
                    }
                    Err(e) => problems.push(e),
                }
            } else if p.starts_with("calib/") {
                match serde_json::from_value::<CalibSet>(v) {
                    Ok(c) => calib.push(c),
                    Err(e) => problems.push(Diagnostic::error("E-PHYS-CALIB-BOUND", format!("{p}: malformed calibration set: {e}"))),
                }
            }
        }
        let aliases = parse("tech/aliases.json")["aliases"]
            .as_object()
            .map(|o| o.iter().filter_map(|(k, v)| v["to"].as_str().map(|t| (k.clone(), t.to_owned()))).collect())
            .unwrap_or_default();
        let dpv = parse("datapath.json");
        let mut formats = BTreeMap::new();
        if let Some(o) = dpv["formats"].as_object() {
            for (k, f) in o {
                formats.insert(k.clone(), (f["m"]["v"].as_f64().unwrap_or(0.0), f["e"]["v"].as_f64().unwrap_or(0.0)));
            }
        }
        let datapath = Datapath::new(flatten(&dpv), formats);
        for k in [
            "a_pp_small_um2", "a_pp_large_um2", "e_pp_fj", "a_add_um2", "e_add_fj", "a_mux_um2", "e_mux_fj", "a_cmp_um2", "a_ff_um2",
            "e_ff_fj", "guard_bits", "fp_mult_exp", "fp_add_exp", "fp_add_e_exp", "alpha_dp", "a_ge45_um2", "s_45_n7", "mm_area",
            "mm_energy", "idle_gated", "idle_ungated", "sfu_area", "sfu_energy", "cast_energy", "mx_block", "max_dot_len",
            "scalar_core_ge", "dma_ge", "sequencer_ge", "simt_ctrl_ge",
        ] {
            if !datapath.v.contains_key(&format!("coeffs.{k}")) {
                problems.push(Diagnostic::error("E-PHYS-NODE-MISSING-FIELD", format!("datapath.json: coeffs.{k} missing")));
            }
        }
        let params: BTreeMap<String, ParamReg> = parse("params.json")["params"]
            .as_object()
            .map(|o| o.iter().filter_map(|(k, v)| serde_json::from_value(v.clone()).ok().map(|p| (k.clone(), p))).collect())
            .unwrap_or_default();
        Tables { nodes, aliases, datapath, dram, phy, package, params, calib, sources, problems }
    }

    /// Node table for an IR `tech` name (aliases resolved); `None` for nodes whose table is not built yet.
    pub fn node(&self, name: &str) -> Option<&TechNode> {
        let id = self.aliases.get(name).map_or(name, String::as_str);
        self.nodes.get(id)
    }

    pub fn resolve(&self, name: &str) -> String {
        self.aliases.get(name).cloned().unwrap_or_else(|| name.to_owned())
    }

    pub fn param_prior(&self, name: &str) -> f64 {
        self.params.get(name).map_or_else(|| panic!("registered phys parameter {name}"), |p| p.prior.v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_data_files_load_sourced() {
        let t = Tables::get();
        assert!(t.problems.is_empty(), "{:#?}", t.problems);
        for n in ["tsmc_n16", "tsmc_n7", "tsmc_n5", "tsmc_n4", "tsmc_n3e", "tsmc_n2", "dram_logic_1y"] {
            assert!(t.nodes.contains_key(n), "{n}");
        }
        assert_eq!(t.node("nvidia_4n").unwrap().id, "tsmc_n4");
        assert_eq!(t.node("tsmc_n12").unwrap().id, "tsmc_n16");
        let n7 = t.node("tsmc_n7").unwrap();
        assert!((n7.a_ge_um2 - 4.0 / 91.2).abs() < 1e-4);
        assert_eq!(n7.cell_um2, 0.027);
        assert!(t.dram.contains_key("hbm2") && t.phy.contains_key("serdes_56g") && t.package.contains_key("cowos_s"));
        assert!(t.params.contains_key("kappa_e_wire"));
        assert_eq!(crate::data::data_hash().len(), 6 + 32);
    }

    #[test]
    fn nodes_order_by_density() {
        let t = Tables::get();
        let a = |n: &str| t.node(n).unwrap().a_ge_um2;
        assert!(a("tsmc_n16") > a("tsmc_n7") && a("tsmc_n7") > a("tsmc_n5") && a("tsmc_n5") > a("tsmc_n4") && a("tsmc_n4") > a("tsmc_n3e"));
        let s = |n: &str| t.node(n).unwrap().s_e;
        assert!(s("tsmc_n16") > s("tsmc_n7") && s("tsmc_n7") > s("tsmc_n5") && s("tsmc_n5") > s("tsmc_n3e"));
    }
}
