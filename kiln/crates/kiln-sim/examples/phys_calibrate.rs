//! kiln-phys calibration (04 §12): area group on published die areas (A100 826 mm^2, GP100 610 mm^2, TPU v4 < 600,
//! TPU v4i < 400 with its CMEM/MXU fractions, v6e within the reticle; TPU transistor counts as weak terms), power
//! group on our A100 NVML telemetry (fit set: idle, gemm 8192^3, decode_b1, decode_b32; held out: gemm 16384^3,
//! decode_b8, prefill_b1); H100 and V100 held out for area. Writes `kiln-phys/data/calib/{generic,a100_40gb}-<date>.json`
//! with `WRITE=1`; kiln-sim's `phys_calibration` test reruns `run` and compares it with the committed sets.
//!
//! `cargo run -p kiln-sim --release --example phys_calibrate`

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kiln_ir::bench::BenchOp;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::{Profile, check_file};
use kiln_map::HwView;
use kiln_phys::calib::{self, Free};
use kiln_phys::{ClockMode, ClockPlan, Params, Phys, PlaceTier};
use kiln_sim::result::OFF_DIE;
use kiln_sim::{CalibSet, Prepared, SimOptions, simulate_bench_op, simulate_member};
use kiln_trace::IntervalMethod;
use serde_json::{Value, json};

const DATE: &str = "2026-10-05";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn design(name: &str) -> HwModel {
    let p = root().join(format!("designs/reference/{name}.json5"));
    check_file(&p, Profile::Full).model.unwrap_or_else(|| panic!("{name} expands"))
}

/// One power record: what was run, at what clock and temperature, measured board power and time per run.
struct Rec {
    name: &'static str,
    fit: bool,
    work: Work,
    mhz: f64,
    temp_c: f64,
    power_w: f64,
    t_meas_s: f64,
}

enum Work {
    Idle,
    Gemm(u64),
    Step(&'static str),
}

/// Energy components of a record at kappa = 1 (J per run, already at V(f)) and the run's MAC activity.
#[derive(Clone, Copy, Debug, Default)]
struct Comp {
    dp: f64,
    rf: f64,
    sram: f64,
    wire: f64,
    dram: f64,
    fixed: f64,
    activity: f64,
}

fn steady(meas: &Value, rec: &str) -> (f64, f64, f64, f64) {
    let r = meas["records"].as_array().unwrap().iter().find(|r| r["name"] == rec).unwrap_or_else(|| panic!("{rec}"));
    let cols: Vec<&str> = r["series_cols"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    let ti = cols.iter().position(|c| *c == "temp_c").unwrap();
    let (t0, t1) = (r["t_load_start"].as_f64().unwrap(), r["t_load_end"].as_f64().unwrap());
    let temps: Vec<f64> = r["series"].as_array().unwrap().iter().filter(|s| s[0].as_f64().unwrap() > t0 + 0.5 && s[0].as_f64().unwrap() < t1).map(|s| s[ti].as_f64().unwrap()).collect();
    let pre: Vec<f64> = r["series"].as_array().unwrap().iter().filter(|s| s[0].as_f64().unwrap() < t0).map(|s| s[2].as_f64().unwrap()).collect();
    let temp = temps.iter().sum::<f64>() / temps.len().max(1) as f64;
    let idle = pre.iter().sum::<f64>() / pre.len().max(1) as f64;
    (r["steady"]["power_w"].as_f64().unwrap(), r["load_s"].as_f64().unwrap() / r["load_calls"].as_f64().unwrap(), temp, idle)
}

fn seq(meas: &Value, rec: &str) -> (f64, f64, f64) {
    let r = meas["records"].as_array().unwrap().iter().find(|r| r["name"] == rec).unwrap_or_else(|| panic!("{rec}"));
    let g = &r["modes"]["graph"];
    (g["clock"]["power_w"].as_f64().unwrap(), g["median_s"].as_f64().unwrap(), g["clock"]["temp_c"].as_f64().unwrap())
}

fn records() -> Vec<Rec> {
    let dir = root().join("../calibration/measurements");
    let load = |f: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(dir.join(f)).unwrap()).unwrap() };
    let (m1, m2, s1, s2) = (load("a100_2026-10-05_micro.json"), load("a100_2026-10-05_micro_r2.json"), load("a100_2026-10-05_seq.json"), load("a100_2026-10-05_seq_r2.json"));
    let avg2 = |a: f64, b: f64| 0.5 * (a + b);
    let mut out = vec![];
    let (p1, t1, c1, i1) = steady(&m1, "power_step/gemm_8192");
    let (p2, t2, c2, i2) = steady(&m2, "power_step/gemm_8192");
    out.push(Rec { name: "gemm_8192", fit: true, work: Work::Gemm(8192), mhz: avg2(1275.0, 1290.0), temp_c: avg2(c1, c2), power_w: avg2(p1, p2), t_meas_s: avg2(t1, t2) });
    let (q1, u1, d1, j1) = steady(&m1, "power_step/gemm_16384");
    let (q2, u2, d2, j2) = steady(&m2, "power_step/gemm_16384");
    out.push(Rec { name: "gemm_16384", fit: false, work: Work::Gemm(16384), mhz: avg2(1245.0, 1290.0), temp_c: avg2(d1, d2), power_w: avg2(q1, q2), t_meas_s: avg2(u1, u2) });
    let idle = (i1 + i2 + j1 + j2) / 4.0;
    out.push(Rec { name: "idle", fit: true, work: Work::Idle, mhz: 1410.0, temp_c: 35.0, power_w: idle, t_meas_s: 1.0 });
    for (ph, fit) in [("decode_b1", true), ("decode_b8", false), ("decode_b32", true), ("prefill_b1", false)] {
        let (pa, ta, ca) = seq(&s1, &format!("seq/{ph}/step_sdpa"));
        let (pb, tb, cb) = seq(&s2, &format!("seq/{ph}/step_sdpa"));
        let mhz = if ph == "prefill_b1" { avg2(1350.0, 1365.0) } else { 1410.0 };
        let w: &'static str = Box::leak(format!("llama3_8b:{ph}").into_boxed_str());
        out.push(Rec { name: Box::leak(ph.to_string().into_boxed_str()), fit, work: Work::Step(w), mhz, temp_c: avg2(ca, cb), power_w: avg2(pa, pb), t_meas_s: avg2(ta, tb) });
    }
    out
}

/// Energy components of `rec` on design view `v` (built at kappa_E = 1) at the record's clock.
fn components(p: &Prepared, rec: &Rec, cal: &Arc<CalibSet>) -> Comp {
    let gpc = p.view.hw.clocks.iter().position(|c| c.path.ends_with("gpc_clk")).unwrap();
    let mut hz: Vec<f64> = p.view.hw.clocks.iter().map(|c| c.spec.freq.0).collect();
    hz[gpc] = rec.mhz * 1e6;
    let opts = SimOptions { interval: IntervalMethod::None, shadow_prices: false, calibration: Some(cal.clone()), clock: ClockMode::Fixed(hz.clone()), layer_scope_fallback: true, ..SimOptions::default() };
    let run = match &rec.work {
        Work::Idle => return Comp::default(),
        Work::Gemm(n) => simulate_bench_op(p, &BenchOp::gemm(*n, *n, *n, false), &opts).unwrap(),
        Work::Step(w) => simulate_member(p, &kiln_wl::zoo::workload(w).unwrap(), &opts).unwrap().0,
    };
    let c = &run.central;
    let v = &p.view;
    let plan = ClockPlan { hz: c.clocks.iter().map(|x| x.hz).collect(), solved: true, throttled: false };
    let mut out = Comp { dp: c.energy.compute_j + c.energy.padding_j, ..Default::default() };
    let rf_levels: Vec<String> = v
        .hw
        .memories
        .iter()
        .enumerate()
        .filter(|(_, m)| m.onchip_kind() == Some(kiln_ir::hw::compute::MemKind::RegisterFile))
        .map(|(i, _)| format!("l{}", v.hw.levels[i]))
        .collect();
    let dram_levels: Vec<String> = v.resources.iter().filter(|r| r.class == kiln_map::hwview::ResClass::Dram).map(|r| format!("l{}", r.level.unwrap_or(0))).collect();
    for (k, x) in &c.energy.memory_j {
        if dram_levels.contains(k) {
            out.dram += x;
        } else if rf_levels.contains(k) {
            out.rf += x;
        } else {
            out.sram += x;
        }
    }
    let phy = kiln_sim::result::dram_phy_j(v, &c.resources);
    out.dram -= phy;
    out.fixed += phy;
    for (k, x) in &c.energy.link_j {
        if OFF_DIE.contains(&k.as_str()) { out.fixed += x } else { out.wire += x }
    }
    // Per run of the measured kernel: whole-step results are per step already; the bench GEMM is one call.
    let pe = kiln_sim::result::phase_energy(v, &c.energy, &c.resources, c.makespan_s, &plan);
    out.activity = pe.activity;
    eprintln!("  {:<12} sim {:.3e} s vs meas {:.3e} s (ratio {:.3}), activity {:.2}; W at meas time: dp {:.1} rf {:.1} sram {:.1} wire {:.1} dram {:.1} fixed {:.1}; mem {:?} link {:?}", rec.name, c.makespan_s, rec.t_meas_s, c.makespan_s / rec.t_meas_s, out.activity, out.dp / rec.t_meas_s, out.rf / rec.t_meas_s, out.sram / rec.t_meas_s, out.wire / rec.t_meas_s, out.dram / rec.t_meas_s, out.fixed / rec.t_meas_s, c.energy.memory_j.iter().map(|(k, v)| format!("{k}:{:.1}", v / rec.t_meas_s)).collect::<Vec<_>>(), c.energy.link_j.iter().map(|(k, v)| format!("{k}:{:.1}", v / rec.t_meas_s)).collect::<Vec<_>>());
    out
}

struct PowerFit {
    names: Vec<&'static str>,
}

fn main() {
    let (g, a) = run();
    if std::env::var("WRITE").is_ok() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kiln-phys/data/calib");
        std::fs::write(dir.join(format!("generic-{DATE}.json")), serde_json::to_string_pretty(&g).unwrap() + "\n").unwrap();
        std::fs::write(dir.join(format!("a100_40gb-{DATE}.json")), serde_json::to_string_pretty(&a).unwrap() + "\n").unwrap();
        println!("wrote calibration sets (rebuild to embed them)");
    }
}

/// Both fits from the data in the tree: the generic set (area group and power-group family parameters) and the A100
/// platform set.
pub fn run() -> (kiln_phys::tables::CalibSet, kiln_phys::tables::CalibSet) {
    // ---- area group ------------------------------------------------------------------------------------------
    let a100 = design("a100_sxm4_40gb");
    let v4 = design("tpu_v4");
    let h100 = design("h100_sxm5_80gb");
    let v5e = design("tpu_v5e");
    let v4i = design("tpu_v4i");
    let v6e = design("tpu_v6e");
    let v100 = design("v100_sxm2_32gb");
    let p100 = design("p100_sxm2_16gb");
    let free_a = vec![
        Free::registered("util_std_cell", Some("tsmc_n7")),
        Free::registered("u_place", None),
        Free::registered("kappa_pe", None),
        Free::registered("kappa_ctrl", None),
        Free::registered("a_misc_mm2", None),
        Free::registered("simt_ctrl_ge", None),
        Free::registered("simt_operand_ge_per_bit", None),
        Free::registered("kappa_sram_area", None),
    ];
    let base = Params::priors();
    let xs = 0.25;
    let area_resid = |vals: &[f64]| -> Vec<f64> {
        let p = calib::with_values(&base, &free_a, vals);
        let a = calib::area_of(&a100, &p);
        let t = calib::area_of(&v4, &p);
        let i = calib::area_of(&v4i, &p);
        let g = calib::area_of(&p100, &p);
        let x6 = calib::area_of(&v6e, &p);
        // GPU transistor counts are checks, not terms (04 §12.3).
        let hinge = |x: f64, lo: f64, hi: f64| if x < lo { (x / lo).ln() } else if x > hi { (x / hi).ln() } else { 0.0 };
        vec![
            (a.die_mm2 / 826.0).ln() / 0.05,
            (g.die_mm2 / 610.0).ln() / 0.05,
            hinge(x6.die_mm2, 0.0, 858.0) / 0.05,
            hinge(t.die_mm2, 0.0, 600.0) / 0.05,
            hinge(i.die_mm2, 0.0, 400.0) / 0.05,
            (i.entities.get("cmem").copied().unwrap_or(0.0) - 0.28) / 0.04,
            (i.entities.get("mxu").copied().unwrap_or(0.0) - 0.11) / 0.04,
            (t.transistors_b / 22.0).ln() / xs,
            (i.transistors_b / 16.0).ln() / xs,
        ]
    };
    let fa = calib::fit(&free_a, &area_resid, 50);
    let pa = calib::with_values(&base, &free_a, &fa.values);
    println!("== area group (MAP, {} iterations, cost {:.3})", fa.iterations, fa.cost);
    for (k, f) in free_a.iter().enumerate() {
        println!("  {:<16} {:<9} prior {:>10.4}  fit {:>10.4}  sigma_post {:.3}  bounds {:?}{}", f.name, f.key.clone().unwrap_or_default(), f.prior, fa.values[k], fa.sigma_post[k], f.bounds, if fa.at_bound[k] { "  AT BOUND" } else { "" });
    }
    let mut area_rows = vec![];
    for (chip, hw, target, lo, hi, use_) in [
        ("a100_40gb", &a100, 826.0, 826.0, 826.0, "fit"),
        ("p100_sxm2", &p100, 610.0, 610.0, 610.0, "fit"),
        ("tpu_v4", &v4, 600.0, 0.0, 600.0, "fit"),
        ("tpu_v4i", &v4i, 400.0, 0.0, 400.0, "fit"),
        ("h100_sxm5", &h100, 814.0, 814.0, 814.0, "held_out"),
        ("v100_sxm2", &v100, 815.0, 815.0, 815.0, "held_out"),
        ("tpu_v5e", &v5e, 325.0, 300.0, 350.0, "weak_held_out"),
        ("tpu_v6e", &v6e, 858.0, 0.0, 858.0, "reticle"),
    ] {
        let a = calib::area_of(hw, &pa);
        let prior = calib::area_of(hw, &base);
        let err = if a.die_mm2 < lo { a.die_mm2 / lo - 1.0 } else if a.die_mm2 > hi { a.die_mm2 / hi - 1.0 } else { 0.0 };
        println!(
            "  {chip:<10} [{use_}] area {:.1} mm^2 (prior {:.1}) target {} -> {:+.1}%  transistors {:.1} B  fractions {}",
            a.die_mm2,
            prior.die_mm2,
            if lo == hi { format!("{target:.0}") } else if lo == 0.0 { format!("< {hi:.0}") } else { format!("[{lo:.0}, {hi:.0}]") },
            100.0 * err,
            a.transistors_b,
            a.fractions.iter().map(|(k, v)| format!("{k} {:.0}%", 100.0 * v)).collect::<Vec<_>>().join(" ")
        );
        if chip == "tpu_v4i" {
            println!("             CMEM {:.1}% (target 28 +- 4), MXU {:.1}% (target 11 +- 4)", 100.0 * a.entities.get("cmem").copied().unwrap_or(0.0), 100.0 * a.entities.get("mxu").copied().unwrap_or(0.0));
        }
        area_rows.push(json!({"chip": chip, "use": use_, "area_mm2": a.die_mm2, "target": [lo, hi], "err": err, "transistors_b": a.transistors_b, "fractions": a.fractions}));
    }

    // ---- power group -----------------------------------------------------------------------------------------
    // Components at kappa_E = 1 (other parameters at the area fit), engine timing from the A100 platform set.
    let unit_e = pa
        .clone()
        .with("kappa_e_dp", None, 1.0)
        .with("kappa_e_sram", None, 1.0)
        .with("kappa_e_rf", None, 1.0)
        .with("kappa_e_wire", None, 1.0)
        .with("kappa_clk", None, 1.0)
        .with("e_dram_core", Some("hbm2"), 3.9);
    let hw = Arc::new(a100.clone());
    let phys = Phys::with(&hw, unit_e.clone(), PlaceTier::A);
    let p = Prepared { design: Arc::new(kiln_ir::hw::load_file(root().join("designs/reference/a100_sxm4_40gb.json5")).unwrap()), view: Arc::new(HwView::with_phys(hw.clone(), phys).unwrap()), published_reference: true };
    let cal = Arc::new(CalibSet::by_id("platform-a100_40gb").unwrap());
    let recs = records();
    println!("== power group: components (A100, platform-a100_40gb timing, kappa_E = 1)");
    let comps: Vec<Comp> = recs.iter().map(|r| components(&p, r, &cal)).collect();
    // Leakage and clock are linear in p_ll, p_ls and kappa_clk at fixed V and T: probe the model at three points.
    let gpc = hw.clocks.iter().position(|c| c.path.ends_with("gpc_clk")).unwrap();
    let static_at = |pp: &Params, r: &Rec| -> (f64, f64) {
        let ph = Phys::with(&hw, pp.clone(), PlaceTier::A);
        let m = ph.m3().unwrap();
        let mut hz: Vec<f64> = hw.clocks.iter().map(|c| c.spec.freq.0).collect();
        hz[gpc] = r.mhz * 1e6;
        (m.power.p_static(&hz, r.temp_c), m.power.dram_background_w)
    };
    let p_ll0 = unit_e.get("p_ll", Some("tsmc_n7"));
    let p_ls0 = unit_e.get("p_ls", Some("tsmc_n7"));
    let lin: Vec<(f64, f64, f64, f64, f64, f64)> = recs
        .iter()
        .map(|r| {
            let (s0, bg) = static_at(&unit_e, r);
            let (s1, _) = static_at(&unit_e.clone().with("p_ll", Some("tsmc_n7"), 2.0 * p_ll0), r);
            let (s2, _) = static_at(&unit_e.clone().with("p_ls", Some("tsmc_n7"), 2.0 * p_ls0), r);
            let m = p.view.phys.m3().unwrap();
            let mut hz: Vec<f64> = hw.clocks.iter().map(|c| c.spec.freq.0).collect();
            hz[gpc] = r.mhz * 1e6;
            let act = comps[recs.iter().position(|x| x.name == r.name).unwrap()].activity;
            let clk = m.power.p_clock(&hz, act);
            let ctrl = m.power.p_ctrl(&hz, if matches!(r.work, Work::Idle) { 0.0 } else { 1.0 });
            (s1 - s0, s2 - s0, s0 - (s1 - s0) - (s2 - s0), clk, bg, ctrl)
        })
        .collect();
    let free_p = vec![
        Free::registered("kappa_e_dp", None),
        Free::registered("kappa_e_sram", None),
        Free::registered("kappa_e_rf", None),
        Free::registered("kappa_e_wire", None),
        Free::registered("kappa_clk", None),
        Free::registered("p_ll", Some("tsmc_n7")),
        Free::registered("e_dram_core", Some("hbm2")),
        Free::registered("p_board_w", None),
        Free::registered("alpha_ctrl", None),
    ];
    let eta = unit_e.get("eta_vr", None);
    let a0 = unit_e.get("alpha_ctrl", None);
    let predict = |vals: &[f64], i: usize| -> f64 {
        let (c, r, l) = (&comps[i], &recs[i], &lin[i]);
        let (kdp, ks, krf, kw, kclk, pll, edram, board) = (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
        let t = r.t_meas_s;
        let dyn_core = (kdp * c.dp + krf * c.rf + ks * c.sram + kw * c.wire) / t;
        let stat = l.0 * pll / p_ll0 + l.1 + l.2;
        let clk = kclk * l.3 + vals[8] / a0 * l.5;
        let dram = edram / 3.9 * c.dram / t + l.4;
        dyn_core + c.fixed / t + stat + clk + dram + board + (1.0 / eta - 1.0) * (dyn_core + stat + clk)
    };
    let fit_ix: Vec<usize> = (0..recs.len()).filter(|&i| recs[i].fit).collect();
    let pf = PowerFit { names: recs.iter().map(|r| r.name).collect() };
    let power_resid = |vals: &[f64]| -> Vec<f64> { fit_ix.iter().map(|&i| (predict(vals, i) / recs[i].power_w).ln() / 0.05).collect() };
    let fp = calib::fit(&free_p, &power_resid, 50);
    println!("== power group (MAP, {} iterations, cost {:.3})", fp.iterations, fp.cost);
    for (k, f) in free_p.iter().enumerate() {
        println!("  {:<16} {:<9} prior {:>8.4}  fit {:>8.4}  sigma_post {:.3}  bounds {:?}{}", f.name, f.key.clone().unwrap_or_default(), f.prior, fp.values[k], fp.sigma_post[k], f.bounds, if fp.at_bound[k] { "  AT BOUND" } else { "" });
    }
    let mut power_rows = vec![];
    for (i, r) in recs.iter().enumerate() {
        let pr = predict(&fp.values, i);
        let p0 = predict(&free_p.iter().map(|f| f.prior).collect::<Vec<_>>(), i);
        println!("  {:<12} [{}] at {:.0} MHz {:.0} C: predicted {:.1} W (prior {:.1}) measured {:.1} W -> {:+.1}%", pf.names[i], if r.fit { "fit" } else { "held out" }, r.mhz, r.temp_c, pr, p0, r.power_w, 100.0 * (pr / r.power_w - 1.0));
        power_rows.push(json!({"record": r.name, "fit": r.fit, "mhz": r.mhz, "temp_c": r.temp_c, "predicted_w": pr, "measured_w": r.power_w, "err": pr / r.power_w - 1.0}));
    }
    let mut generic = calib::entries(&free_a, &fa);
    let mut platform = vec![];
    for e in calib::entries(&free_p, &fp) {
        if e.name == "p_board_w" { platform.push(e) } else { generic.push(e) }
    }
    let report = json!({"date": DATE, "area": {"free": free_a, "cost": fa.cost, "rows": area_rows}, "power": {"free": free_p, "cost": fp.cost, "rows": power_rows},
        "note": "04 §12.3: three chips cannot identify every parameter; node- and family-scoped values are prior-driven (dual reporting). Power residuals use measured run time (energy calibration isolated from timing)."});
    let g = calib::calib_set(&format!("phys-generic-{DATE}"), "generic", None, generic, report.clone());
    let a = calib::calib_set(&format!("phys-a100_40gb-{DATE}"), "platform", Some("a100_40gb"), platform, json!({"date": DATE, "see": format!("phys-generic-{DATE}")}));
    (g, a)
}
