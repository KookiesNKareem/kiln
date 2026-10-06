//! Envelope rules of an evaluation (04 §8, §10, §17): what turns physical findings into residuals, power caps,
//! V/f legality and thermal runaway.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use kiln_ir::hw::{Design, Profile};
use kiln_sim::{CalibSet, Prepared, SimOptions};
use kiln_trace::IntervalMethod;
use kiln_trace::result::{EvalResult, Status};
use kiln_trace::sim::ResourceKind;
use serde_json::{Value, json};

fn canonical(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference").join(format!("{name}.json5"));
    kiln_ir::hw::load_file(&p).unwrap().canonical
}

fn prepared(v: Value) -> Prepared {
    Prepared::load(Design::from_value(v).unwrap(), Profile::Full).unwrap()
}

fn opts() -> SimOptions {
    let cal = Arc::new(CalibSet::by_id("generic-v2").expect("generic set"));
    SimOptions { interval: IntervalMethod::None, shadow_prices: false, calibration: Some(cal), ..SimOptions::default() }
}

fn eval(p: &Prepared, w: &str) -> EvalResult {
    kiln_sim::evaluate_prepared(p, &[kiln_wl::zoo::workload(w).unwrap()], &opts())
}

fn codes(r: &EvalResult) -> Vec<String> {
    r.errors.iter().map(|e| e.diag.code.clone()).collect()
}

/// 04 §17 rule 7: physical findings are residuals only on a published chip of the shipped reference set; a
/// candidate cannot buy that status with a `die_area` claim.
#[test]
fn residual_status_comes_from_the_reference_set_not_from_claims() {
    let h100 = eval(&common::reference("h100_sxm5_80gb.json5"), "llama3_8b:decode_b8");
    assert_eq!(h100.status, Status::Ok, "{:?}", codes(&h100));
    assert!(h100.score > 0.0 && h100.warnings.iter().any(|w| w.diag.code == "W-PHYS-RESIDUAL"));

    let mut v = canonical("h100_sxm5_80gb");
    v["name"] = Value::from("candidate");
    let r = eval(&prepared(v), "llama3_8b:decode_b8");
    assert_eq!(r.status, Status::Envelope, "{:?}", codes(&r));
    assert_eq!(r.score, 0.0);
    assert!(codes(&r).contains(&"E-PHYS-RETICLE".to_string()) && !r.warnings.iter().any(|w| w.diag.code == "W-PHYS-RESIDUAL"));
}

/// A small tiled accelerator (one die, four tiles, one HBM3 stack), edited by `f`.
fn tiles(f: impl Fn(&mut Value)) -> Prepared {
    let mut doc = serde_json::json!({
        "schema": "kiln.hw/1.0", "name": "tiles", "tech": "tsmc_n7",
        "clocks": [ { "id": "clk", "freq": 1.0e9 } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "clk",
                "clusters": [ { "id": "tile", "layout": { "grid": [2, 2] },
                    "units": [ { "id": "mxu", "kind": "matrix", "geometry": { "systolic": { "rows": 64, "cols": 64 } },
                                 "precisions": ["bf16*bf16+fp32"], "local": [ { "id": "w", "holds": "b", "capacity": 8192 } ],
                                 "feeds": { "a": "sram", "b": "sram", "o": "sram" } },
                               { "id": "vpu", "kind": "vector", "lanes": 64, "precisions": ["fp32@1"], "feeds": { "any": "sram" } } ],
                    "memories": [ { "id": "sram", "kind": "scratchpad", "capacity": 1048576, "banks": 4,
                                    "ports": [ { "dir": "rw", "width_bits": 512 } ] } ] } ],
                "networks": [ { "id": "noc", "topology": { "type": "mesh", "dims": [2, 2] },
                                "endpoints": [ { "select": "tile*.sram", "at": "layout" } ], "link": "256b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "24GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "6.4Gbps", "attach": "die.noc" } ] } }
    });
    f(&mut doc);
    let d = Design::from_source(&kiln_ir::hw::MemLoader::default(), None, &doc.to_string()).expect("parses");
    Prepared::load(d, Profile::Full).unwrap()
}

/// 04 §8: a design without a stable junction temperature is rejected (E-PHYS-THERMAL-RUNAWAY), not scored at the
/// power of an arbitrary iteration.
#[test]
fn thermal_runaway_rejects_the_design() {
    let cool = eval(&tiles(|_| {}), "llama3_8b:decode_b1");
    assert!(!codes(&cool).contains(&"E-PHYS-THERMAL-RUNAWAY".to_string()), "{:?}", codes(&cool));
    let hot = eval(&tiles(|v| v["system"]["package"]["dies"][0]["clusters"][0]["units"][0]["power"] = json!({ "leakage": "60W" })), "llama3_8b:decode_b1");
    assert_eq!(hot.status, Status::Envelope, "{:?}", codes(&hot));
    assert!(codes(&hot).contains(&"E-PHYS-THERMAL-RUNAWAY".to_string()) && hot.score == 0.0, "{:?}", codes(&hot));
}

/// 03 §4.5: a throttled domain moves its on-chip memory ports and on-die links at the solved clock (a 512-bit
/// link at half the clock carries half the bytes per second); DRAM channels keep their pin rate.
#[test]
fn throttled_domains_scale_synchronous_bandwidth() {
    let p = common::mesh(common::Params { gr: 2, gc: 2, rows: 32, cols: 32, lanes: 32, sram_kib: 512, pin_gbps: 6 }, common::A, common::Extra::None);
    let (model, sc) = common::tiny(true, 4);
    let prog = common::program(&model, &sc, 3);
    let f = p.view.phys.nominal_hz(0);
    let at = |hz: f64| common::run(&p, &prog, &SimOptions { clock: kiln_phys::ClockMode::Fixed(vec![hz]), ..common::quick() }).central;
    let (full, half) = (at(f), at(0.5 * f));
    assert_eq!(full.resources.len(), half.resources.len());
    let mut checked = 0;
    for (r, h) in full.resources.iter().zip(&half.resources) {
        assert_eq!((&h.resource, h.bytes), (&r.resource, r.bytes));
        // On this design every on-chip port and on-die link runs on the one clock; the stack and its PHY link do not.
        let synchronous = match r.kind {
            ResourceKind::ComputeUnit | ResourceKind::Sequencer => continue,
            ResourceKind::DramChannel => false,
            _ => r.class.as_deref() != Some("memport"),
        };
        let want = if synchronous { 2.0 } else { 1.0 };
        assert!((h.busy_s / r.busy_s / want - 1.0).abs() < 1e-9, "{}: busy {} -> {} at half clock", r.resource, r.busy_s, h.busy_s);
        checked += usize::from(synchronous);
    }
    assert!(checked > 0);
}

/// 04 §8: a cap is checked at the clocks the phase runs at, also when there is no lower V/f point to throttle to.
#[test]
fn an_exceeded_cap_is_reported_without_throttling() {
    let capped = |cap: &'static str| {
        tiles(move |v| {
            v["clocks"][0] = json!({ "id": "clk", "freq": "1GHz", "base": "1GHz" });
            v["power"] = json!([{ "id": "tdp", "members": "board.chip", "cap": cap, "policy": "dvfs", "clocks": ["clk"] }]);
        })
    };
    let ok = eval(&capped("2000W"), "llama3_8b:decode_b1");
    assert!(!codes(&ok).contains(&"E-MAP-POWER-CAP".to_string()) && ok.score > 0.0, "{:?}", codes(&ok));
    let over = eval(&capped("10W"), "llama3_8b:decode_b1");
    assert_eq!(over.status, Status::Envelope, "{:?}", codes(&over));
    assert!(codes(&over).contains(&"E-MAP-POWER-CAP".to_string()) && over.score == 0.0);
}

/// 04 §8: each cap bounds its own members; a generous cap on another die does not lend allowance.
#[test]
fn caps_are_enforced_per_domain() {
    let two = |die_cap: &'static str| {
        tiles(move |v| {
            let pkg = &mut v["system"]["package"];
            let mut d2 = pkg["dies"][0].clone();
            d2["id"] = json!("die2");
            pkg["dies"].as_array_mut().unwrap().push(d2);
            let mut s2 = pkg["mem_stacks"][0].clone();
            s2["id"] = json!("hbm2");
            s2["attach"] = json!("die2.noc");
            pkg["mem_stacks"].as_array_mut().unwrap().push(s2);
            v["power"] = json!([
                { "id": "die_cap", "members": "board.chip.die", "cap": die_cap, "policy": "fixed" },
                { "id": "spare", "members": "board.chip.die2", "cap": "5000W", "policy": "fixed" },
            ]);
        })
    };
    let ok = eval(&two("1000W"), "llama3_8b:decode_b1");
    assert!(!codes(&ok).contains(&"E-MAP-POWER-CAP".to_string()), "{:?}", codes(&ok));
    let r = eval(&two("5W"), "llama3_8b:decode_b1");
    assert_eq!(r.status, Status::Envelope, "{:?}", codes(&r));
    let e = r.errors.iter().find(|e| e.diag.code == "E-MAP-POWER-CAP").expect("die cap exceeded");
    assert!(e.diag.message.contains("die_cap") && !r.errors.iter().any(|e| e.diag.message.contains("spare")), "{:?}", e.diag.message);
}

/// 03 §4.5: compute energy scales with the voltage of the clock domain that executed it, not the first capped
/// domain's.
#[test]
fn compute_energy_scales_with_its_own_domain_voltage() {
    let at = |volts: &'static str| {
        let p = tiles(move |v| {
            v["clocks"] = json!([
                { "id": "slow", "freq": "1GHz", "base": "1GHz", "vf": [{ "freq": "1GHz", "voltage": "0.6V" }] },
                { "id": "clk", "freq": "1GHz", "base": "1GHz", "vf": [{ "freq": "1GHz", "voltage": volts }] },
            ]);
            v["power"] = json!([{ "id": "tdp", "members": "board.chip", "cap": "5000W", "policy": "dvfs", "clocks": ["slow", "clk"] }]);
        });
        let (model, sc) = common::tiny(true, 4);
        common::run(&p, &common::program(&model, &sc, 3), &common::quick()).central.energy.compute_j
    };
    let (nominal, high) = (at("0.75V"), at("0.9V"));
    assert!((high / nominal - (0.9f64 / 0.75).powi(2)).abs() < 1e-9, "{nominal} J at 0.75 V, {high} J at 0.9 V");
}

/// 04 §4.7: a 0 V operating point (free compute, leakage and clock power at full speed) is rejected at evaluation.
#[test]
fn zero_voltage_is_rejected() {
    let r = eval(&tiles(|v| v["clocks"][0] = json!({ "id": "clk", "freq": "1GHz", "base": "1GHz", "vf": [{ "freq": "1GHz", "voltage": "0V" }] })), "llama3_8b:decode_b1");
    assert_eq!(r.status, Status::Envelope, "{:?}", codes(&r));
    assert!(codes(&r).contains(&"E-PHYS-VF-VOLTAGE".to_string()) && r.score == 0.0);
}

/// 04 §8: a die-scoped cap carries the die-side PHY's share of DRAM traffic energy (the stack outside the die
/// keeps the DRAM core share); a design cannot pass a die cap by leaving its PHY power on the stack.
#[test]
fn die_caps_carry_the_dram_phy_energy() {
    use kiln_sim::engine::Engine;
    use kiln_sim::result::{Assembly, assemble, phase_energy, phase_energy_within};
    use kiln_trace::Corner;

    let p = tiles(|v| v["power"] = json!([{ "id": "die_cap", "members": "board.chip.die", "cap": "5000W", "policy": "fixed" }]));
    let (model, sc) = common::tiny(true, 4);
    let prog = common::program(&model, &sc, 3);
    let run = common::run(&p, &prog, &common::quick());
    let params = run.params.at(Corner::Central);
    let out = Engine { view: &p.view, g: &run.graph, params: &params, clocks: &run.clocks }.run(prog.window.map(|w| w.0 / 2));
    let a = Assembly {
        view: &p.view,
        prog: &prog,
        graph: &run.graph,
        phase: run.central.phase.clone(),
        scope: run.central.scope,
        corner: Corner::Central,
        provenance: run.central.provenance.clone(),
        params: &params,
        clocks: &run.clocks,
        trace_ops: false,
    };
    let r = assemble(&a, &out);
    let whole = phase_energy(&p.view, &r.energy, &r.resources, r.makespan_s, &run.clocks);
    let die = phase_energy_within(&a, &out, r.makespan_s, &p.view.phys.m3().unwrap().caps[0].nodes);
    assert!(whole.indep_j > 0.0 && whole.dram_j > 0.0);
    assert!((die.indep_j / whole.indep_j - 1.0).abs() < 1e-9, "die-side PHY {} J of {} J", die.indep_j, whole.indep_j);
    assert_eq!(die.dram_j, 0.0);
}

/// 04 §8: each DRAM resource's traffic splits between its stack and its die-side PHY by its own energies; another
/// stack's energy override does not move PHY power off the die.
#[test]
fn dram_phy_share_is_per_stack() {
    use kiln_sim::engine::Engine;
    use kiln_sim::result::{Assembly, assemble, phase_energy, phase_energy_within};
    use kiln_trace::Corner;

    let at = |override_j: Option<f64>| {
        let p = tiles(move |v| {
            let pkg = &mut v["system"]["package"];
            let mut s2 = pkg["mem_stacks"][0].clone();
            s2["id"] = json!("hbm2");
            if let Some(e) = override_j {
                s2["overrides"] = json!({ "energy_per_byte": e });
            }
            pkg["mem_stacks"].as_array_mut().unwrap().push(s2);
            v["power"] = json!([{ "id": "die_cap", "members": "board.chip.die", "cap": "5000W", "policy": "fixed" }]);
        });
        let (model, sc) = common::tiny(true, 4);
        let prog = common::program(&model, &sc, 3);
        let run = common::run(&p, &prog, &common::quick());
        let params = run.params.at(Corner::Central);
        let out = Engine { view: &p.view, g: &run.graph, params: &params, clocks: &run.clocks }.run(prog.window.map(|w| w.0 / 2));
        let a = Assembly {
            view: &p.view,
            prog: &prog,
            graph: &run.graph,
            phase: run.central.phase.clone(),
            scope: run.central.scope,
            corner: Corner::Central,
            provenance: run.central.provenance.clone(),
            params: &params,
            clocks: &run.clocks,
            trace_ops: false,
        };
        let r = assemble(&a, &out);
        let dram: Vec<(String, f64)> = r.resources.iter().filter(|x| x.kind == ResourceKind::DramChannel).map(|x| (x.resource.to_string(), x.bytes)).collect();
        let whole = phase_energy(&p.view, &r.energy, &r.resources, r.makespan_s, &run.clocks);
        let die = phase_energy_within(&a, &out, r.makespan_s, &p.view.phys.m3().unwrap().caps[0].nodes);
        (dram, whole, die)
    };
    let (base, over) = (at(None), at(Some(3.6e-9)));
    assert_eq!(base.0, over.0);
    assert!(base.0.len() == 2 && base.0.iter().all(|x| x.1 > 0.0), "{:?}", base.0);
    assert!(over.1.dram_j > base.1.dram_j);
    for (b, o) in [(base.1.indep_j, over.1.indep_j), (base.2.indep_j, over.2.indep_j)] {
        assert!(b > 0.0 && (o / b - 1.0).abs() < 1e-9, "die-side PHY {o} J with the override vs {b} J without");
    }
}
