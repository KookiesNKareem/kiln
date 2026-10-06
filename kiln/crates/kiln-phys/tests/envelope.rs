//! Envelope checks of the physical model (04 §7.2 low-swing links, §7.4 router timing, §6.3 package layout,
//! §8 thermal runaway, §4.7 V/f limits).

use kiln_ir::hw::model::{ChannelKind, NodeIx};
use kiln_ir::hw::{Design, HwModel, MemLoader, Profile, check};
use kiln_phys::Phys;
use serde_json::{Value, json};

fn mesh(link: Value, clock: Value) -> Value {
    json!({
        "schema": "kiln.hw/1.0", "name": "env", "tech": "tsmc_n7",
        "clocks": [ clock ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "clk",
                "clusters": [ { "id": "tile", "layout": { "grid": [2, 2] },
                    "units": [ { "id": "mxu", "kind": "matrix", "geometry": { "systolic": { "rows": 32, "cols": 32 } },
                                 "precisions": ["bf16*bf16+fp32"], "local": [ { "id": "w", "holds": "b", "capacity": 2048 } ], "feeds": { "a": "sram", "b": "sram", "o": "sram" } } ],
                    "memories": [ { "id": "sram", "kind": "scratchpad", "capacity": 262144, "banks": 4,
                                    "ports": [ { "dir": "rw", "width_bits": 256 } ] } ] } ],
                "networks": [ { "id": "noc", "topology": { "type": "mesh", "dims": [2, 2] },
                                "endpoints": [ { "select": "tile*.sram", "at": "layout" } ], "link": link } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "6.4Gbps", "attach": "die.noc" } ] } }
    })
}

fn model(v: &Value) -> HwModel {
    let d = Design::from_source(&MemLoader::default(), None, &serde_json::to_string(v).unwrap()).expect("parses");
    let r = check(d, Profile::Full, &Default::default());
    assert!(!r.has_errors(), "{:?}", r.errors().collect::<Vec<_>>());
    r.model.expect("model")
}

fn clk(freq: f64) -> Value {
    json!({ "id": "clk", "freq": freq })
}

fn codes(ph: &Phys) -> Vec<String> {
    ph.problems().into_iter().map(|d| d.code).collect()
}

/// 04 §7.2: a low-swing hop replaces only the wire energy; the router it enters still costs its traversal.
#[test]
fn low_swing_hops_keep_router_energy() {
    let hw = model(&mesh(json!({ "width_bits": 128, "phys": { "type": "on_die", "swing": "low" } }), clk(1e9)));
    let ph = Phys::new(&hw);
    let m = ph.m3().unwrap();
    let node = kiln_phys::tables::Tables::get().node("tsmc_n7").unwrap();
    let kw = m.params.get("kappa_e_wire", None);
    let mut n = 0;
    for (i, c) in hw.channels.iter().enumerate() {
        let NodeIx::Router(r) = hw.nodes[hw.node_of(c.dst)].ix else { continue };
        if c.kind != ChannelKind::NocHop {
            continue;
        }
        let ro = &m.ch.routers[r];
        let router = ro.e_flit_j / (ro.flit_bits / 8.0).max(1.0);
        let l = &m.links[i];
        let w = node.wire(l.class.expect("on-die wire class"));
        let wire = 8.0 * (0.25 * w.c_ff_um * 1e-15 * 0.2 * node.vdd_nom * l.length_um + 20e-15) * kw;
        assert!((l.e_j_per_byte - (wire + router)).abs() <= 1e-9 * (wire + router), "channel {i}: {} J/B, want wire {wire} + router {router}", l.e_j_per_byte);
        n += 1;
    }
    assert!(n > 0, "no hop into a router");
}

fn with_router(mut v: Value, router: Value) -> Value {
    v["system"]["package"]["dies"][0]["networks"][0]["router"] = router;
    v
}

/// 04 §7.4: a router whose allocator needs longer than the clock period is pipelined deeper, or its declared
/// pipeline is rejected.
#[test]
fn router_timing_is_enforced() {
    let ok = Phys::new(&model(&with_router(mesh(json!("128b"), clk(1e9)), json!({ "pipeline": 2 }))));
    assert!(!codes(&ok).contains(&"E-PHYS-ROUTER-TIMING".to_string()));
    let m = ok.m3().unwrap();
    assert!(m.ch.routers.iter().all(|r| r.t_cycle_min_s > 50e-12 && r.n_pipe == 2), "{:?}", m.ch.routers);

    let fast = Phys::new(&model(&with_router(mesh(json!("128b"), clk(20e9)), json!({ "pipeline": 2 }))));
    assert!(codes(&fast).contains(&"E-PHYS-ROUTER-TIMING".to_string()), "{:?}", codes(&fast));

    let auto = Phys::new(&model(&mesh(json!("128b"), clk(20e9))));
    assert!(!codes(&auto).contains(&"E-PHYS-ROUTER-TIMING".to_string()));
    for r in &auto.m3().unwrap().ch.routers {
        let stages_needed = (r.t_cycle_min_s * 20e9).ceil() as u32;
        assert!(r.n_pipe >= 2 + stages_needed - 1, "{} stages for a {:.0} ps allocator at 50 ps", r.n_pipe, r.t_cycle_min_s * 1e12);
    }

    // Declaring the depth automatic pipelining picks closes timing; one stage less does not.
    let depth = auto.m3().unwrap().ch.routers.iter().map(|r| r.n_pipe).max().unwrap();
    let declared = |n: u32| Phys::new(&model(&with_router(mesh(json!("128b"), clk(20e9)), json!({ "pipeline": n }))));
    let deep = declared(depth);
    assert!(!codes(&deep).contains(&"E-PHYS-ROUTER-TIMING".to_string()), "{:?}", deep.problems());
    assert!(deep.m3().unwrap().ch.routers.iter().all(|r| r.n_pipe == depth));
    assert!(codes(&declared(depth - 1)).contains(&"E-PHYS-ROUTER-TIMING".to_string()));
}

fn two_dies(second: Value) -> Value {
    let mut v = mesh(json!("128b"), clk(1e9));
    let die = v["system"]["package"]["dies"][0].clone();
    let mut a = die.clone();
    a["placement"] = json!({ "mode": "pinned", "x": 0, "y": 0 });
    let mut b = die;
    b["id"] = json!("die2");
    b["placement"] = second;
    v["system"]["package"]["dies"] = json!([a, b]);
    let mut hbm2 = v["system"]["package"]["mem_stacks"][0].clone();
    hbm2["id"] = json!("hbm2");
    hbm2["attach"] = json!("die2.noc");
    v["system"]["package"]["mem_stacks"].as_array_mut().unwrap().push(hbm2);
    v
}

/// 04 §6.3: pinned dies of one package layer may not overlap each other or the memory stacks.
#[test]
fn pinned_dies_may_not_overlap() {
    let overlap = |p: &Phys| codes(p).iter().filter(|c| *c == "E-PHYS-PACKAGE-OVERLAP").count();
    let same = Phys::new(&model(&two_dies(json!({ "mode": "pinned", "x": 0, "y": 0 }))));
    assert!(overlap(&same) >= 1, "{:?}", codes(&same));
    let w = same.m3().unwrap().fp.dies[0].outline.w();
    let apart = Phys::new(&model(&two_dies(json!({ "mode": "pinned", "x": format!("{}um", 3.0 * w), "y": 0 }))));
    assert_eq!(overlap(&apart), 0, "{:?}", apart.problems());
    let auto = Phys::new(&model(&two_dies(json!({ "mode": "auto" }))));
    assert_eq!(overlap(&auto), 0, "{:?}", auto.problems());
}

/// 04 §4.7: a declared V/f point or boost voltage outside the node's voltage range (0 V makes compute free) is
/// rejected.
#[test]
fn vf_voltages_are_within_the_node_range() {
    let vf = |base: &str, pts: Value| json!({ "id": "clk", "freq": "1GHz", "base": base, "vf": pts });
    let bad = |c: Value| codes(&Phys::new(&model(&mesh(json!("128b"), c)))).contains(&"E-PHYS-VF-VOLTAGE".to_string());
    assert!(bad(vf("1GHz", json!([{ "freq": "1GHz", "voltage": "0V" }]))));
    assert!(bad(vf("1GHz", json!([{ "freq": "1GHz", "voltage": "1.2V" }]))));
    assert!(bad(json!({ "id": "clk", "freq": "1GHz", "voltage": "0.1V" })));
    assert!(!bad(vf("0.8GHz", json!([{ "freq": "0.8GHz", "voltage": "0.7V" }, { "freq": "1GHz", "voltage": "0.8V" }]))));
    assert!(!bad(clk(1e9)));
}

/// 03 §4.5: the solve lowers only the clocks under the cap's DVFS control; a cap still exceeded at their floor is
/// reported instead of throttling other domains.
#[test]
fn the_solve_lowers_only_controlled_clocks() {
    let hw = kiln_ir::hw::check_file(format!("{}/../../designs/reference/a100_sxm4_40gb.json5", env!("CARGO_MANIFEST_DIR")), Profile::Reference).model.unwrap();
    let ph = Phys::new(&hw);
    let gpc = hw.clocks.iter().position(|c| c.path.ends_with("gpc_clk")).unwrap();
    let hbm = hw.clocks.iter().position(|c| c.path.ends_with("hbm_clk")).unwrap();
    assert_eq!(ph.caps().iter().map(|c| c.clocks.clone()).collect::<Vec<_>>(), vec![vec![gpc]]);
    let (plan, over) = ph.solve_clock(Some(&|hz: &[f64]| vec![300.0 + 200.0 * hz[gpc] / 1.41e9 + 100.0 * hz[hbm] / 1.215e9]));
    assert_eq!(plan.hz[gpc], 1.095e9, "gpc at its floor");
    assert_eq!(plan.hz[hbm], 1.215e9, "hbm_clk is not under the cap's control");
    assert_eq!(over.len(), 1);
    assert!((over[0].power_w - (300.0 + 200.0 * 1.095 / 1.41 + 100.0)).abs() < 1e-6 && over[0].cap_w == 400.0, "{over:?}");
}

/// An N7 die on `clk` and an N16 die on `c16` (`shared`: both dies on the one clock `c16` declares).
fn n7_n16(c16: Value, shared: bool) -> HwModel {
    let mut v = two_dies(json!({ "mode": "auto" }));
    v["clocks"] = json!([clk(1e9), c16]);
    let die2 = &mut v["system"]["package"]["dies"][1];
    die2["tech"] = json!("tsmc_n16");
    if shared {
        let mut c = v["clocks"][1].clone();
        c["id"] = json!("clk");
        v["clocks"] = json!([c]);
    } else {
        die2["default_clock"] = json!("c16");
    }
    model(&v)
}

fn die_unit<'a>(hw: &'a HwModel, die: &str) -> &'a kiln_ir::hw::model::UnitInst {
    hw.units.iter().find(|u| hw.nodes[u.node].path.contains(&format!(".{die}."))).unwrap()
}

/// 04 §4.7: each clock domain is validated and scaled with the technology of the blocks it clocks, not the first
/// die's; a domain shared by two technologies satisfies both ranges and scales each block at its own V_nom.
#[test]
fn clock_domains_use_their_own_technology() {
    let c16 = |v: &str| json!({ "id": "c16", "freq": "1GHz", "base": "1GHz", "vf": [{ "freq": "1GHz", "voltage": v }] });
    let bad = |hw: &HwModel| codes(&Phys::new(hw)).contains(&"E-PHYS-VF-VOLTAGE".to_string());
    assert!(bad(&n7_n16(c16("0.6V"), false)), "0.6 V is below N16's 0.65 V minimum");
    assert!(!bad(&n7_n16(c16("1.0V"), false)), "1.0 V is within N16's range");
    assert!(bad(&n7_n16(c16("0.6V"), true)) && bad(&n7_n16(c16("1.0V"), true)), "a shared domain satisfies both nodes");

    let hw = n7_n16(c16("0.8V"), false);
    let ph = Phys::new(&hw);
    let plan = ph.clock_plan(&kiln_phys::ClockMode::Nominal);
    let u16 = die_unit(&hw, "die2");
    assert!((ph.dyn_scale(u16.clock, Some(u16.node), &plan) - 1.0).abs() < 1e-12, "N16 at its own 0.8 V nominal");

    let hw = n7_n16(c16("0.8V"), true);
    let ph = Phys::new(&hw);
    let plan = ph.clock_plan(&kiln_phys::ClockMode::Nominal);
    let (u7, u16) = (die_unit(&hw, "die"), die_unit(&hw, "die2"));
    assert_eq!(u7.clock, u16.clock);
    assert!((ph.dyn_scale(u16.clock, Some(u16.node), &plan) - 1.0).abs() < 1e-12);
    assert!((ph.dyn_scale(u7.clock, Some(u7.node), &plan) - (0.8f64 / 0.75).powi(2)).abs() < 1e-12);
}

/// 04 §4.7: power gating cuts only an idle block's leakage; characterization charges a gated SRAM its full
/// active leakage, so the flag alone buys no power headroom.
#[test]
fn power_gating_is_not_a_static_leakage_discount() {
    let leak = |gated: bool| {
        let mut v = mesh(json!("128b"), clk(1e9));
        v["system"]["package"]["dies"][0]["clusters"][0]["memories"][0]["power_gated"] = json!(gated);
        let ph = Phys::new(&model(&v));
        let m = ph.m3().unwrap();
        (m.ch.nodes.iter().map(|n| n.leak_w).sum::<f64>(), m.power.p_static(&[1e9], 85.0))
    };
    let (open, gated) = (leak(false), leak(true));
    assert!(open.0 > 0.0);
    assert_eq!(open, gated);
}

fn thermal_package(id: &str, power: Option<Value>) -> Value {
    let mut p = mesh(json!("128b"), clk(1e9))["system"]["package"].clone();
    p["id"] = json!(id);
    if let Some(pw) = power {
        p["power"] = pw;
    }
    p
}

fn two_packages(b_cooling: &str) -> HwModel {
    let mut v = mesh(json!("128b"), clk(1e9));
    v["system"] = json!({ "boards": [ { "id": "board", "packages": [
        thermal_package("a", Some(json!({ "cap": "1000W" }))),
        thermal_package("b", Some(json!({ "cap": "1000W", "thermal": { "tj_max_c": 95.0, "cooling": b_cooling } }))),
    ] } ] });
    model(&v)
}

/// 04 §9: a cap's thermal limits come from the packages holding its members; another package's liquid cooling
/// does not cool them, and the whole design is held to its worst package.
#[test]
fn cooling_is_resolved_per_package() {
    let (air, liquid) = (Phys::new(&two_packages("air")), Phys::new(&two_packages("liquid_cold_plate")));
    let (ma, ml) = (air.m3().unwrap(), liquid.m3().unwrap());
    let p = &ml.params;
    let a = |m: &kiln_phys::Model| m.caps.iter().find(|c| c.path.contains(".a.")).unwrap().power.clone();
    let b = |m: &kiln_phys::Model| m.caps.iter().find(|c| c.path.contains(".b.")).unwrap().power.clone();
    assert_eq!(a(ml).r_ja_k_mm2_w, p.get("r_ja_k_mm2_w", None));
    assert_eq!(a(ml).q_avg_max, p.get("q_avg_max_air", None));
    assert_eq!(a(ml).tj_max_c, p.get("tj_max_c", None));
    assert_eq!(a(ml), a(ma));
    assert_eq!(b(ml).r_ja_k_mm2_w, p.get("r_ja_liquid_k_mm2_w", None));
    assert_eq!(b(ml).q_avg_max, p.get("q_avg_max_liquid", None));
    assert_eq!(b(ml).tj_max_c, 95.0);
    assert_eq!(ml.power.r_ja_k_mm2_w, p.get("r_ja_k_mm2_w", None));
    assert_eq!(ml.power.q_avg_max, p.get("q_avg_max_air", None));
    assert_eq!(ml.power.tj_max_c, 95.0f64.min(p.get("tj_max_c", None)));
}

/// 01 §12, 04 §4.7: a declared `theta_ja` (K/W) replaces the area-normalized default junction-to-ambient
/// resistance.
#[test]
fn declared_theta_ja_is_used() {
    let mut v = mesh(json!("128b"), clk(1e9));
    v["system"]["package"]["power"] = json!({ "cap": "1000W", "thermal": { "tj_max_c": 105.0, "theta_ja": 1.0 } });
    let ph = Phys::new(&model(&v));
    let m = ph.m3().unwrap();
    let e = kiln_phys::PhaseEnergy { makespan_s: 1.0, core_dyn_j: 20.0, ..Default::default() };
    for pm in [&m.power, &m.caps[0].power] {
        let b = pm.power(&e, &[1e9]);
        assert!((b.t_j_c - (pm.t_inlet_c + b.package_w * 1.0)).abs() < 0.01, "{b:?}");
    }
}
