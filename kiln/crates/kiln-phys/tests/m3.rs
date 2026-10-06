//! M3 acceptance on the physical side (06 §2.6 L6 area, §11 P13 monotonicity, E-IR-UNPRICED pricing, placer
//! speed). Power and clock-under-cap need the engine and live in kiln-sim/tests/phys_m3.rs.

use kiln_ir::hw::{Design, HwModel, Profile, check, check_file, check_priced, load_file};
use kiln_phys::calib::area_of;
use kiln_phys::place::{self, Problem, Rect};
use kiln_phys::{Params, Phys};
use serde_json::Value;

fn path(name: &str) -> String {
    format!("{}/../../designs/reference/{name}.json5", env!("CARGO_MANIFEST_DIR"))
}

fn model(name: &str) -> HwModel {
    check_file(path(name), Profile::Full).model.expect("expands")
}

fn from_value(v: Value) -> HwModel {
    let r = check(Design::from_value(v).expect("typed"), Profile::Full, &Default::default());
    assert!(!r.has_errors(), "{:?}", r.errors().collect::<Vec<_>>());
    r.model.expect("model")
}

#[test]
fn l6_die_areas_within_tolerance() {
    // 06 §2.6: +-15% of published; the A100 and P100 are in the fit set (04 §12.3 +-5%), the TPUs are published only as
    // upper bounds, the H100 is held out (04 §12.3 +-12%). The V100 is a documented held-out residual (04 §17).
    for (name, lo, hi, tol) in [("a100_sxm4_40gb", 826.0, 826.0, 0.05), ("p100_sxm2_16gb", 610.0, 610.0, 0.05), ("tpu_v4", 0.0, 600.0, 0.0), ("tpu_v4i", 0.0, 400.0, 0.0), ("h100_sxm5_80gb", 814.0, 814.0, 0.12)] {
        let hw = model(name);
        let a = area_of(&hw, &Params::for_family(hw.family.as_deref())).die_mm2;
        assert!(a >= lo * (1.0 - tol) && a <= hi * (1.0 + tol), "{name}: {a:.1} mm^2 vs [{lo}, {hi}] +-{tol}");
    }
}

#[test]
fn every_reference_design_characterizes() {
    for name in ["a100_sxm4_40gb", "p100_sxm2_16gb", "h100_sxm5_80gb", "h100_pcie_80gb", "v100_sxm2_32gb", "tpu_v4", "tpu_v4i", "tpu_v5e", "tpu_v5e_2x2", "tpu_v6e", "ember"] {
        let hw = model(name);
        let ph = Phys::new(&hw);
        let rep = ph.report().expect("m3 report");
        assert!(rep.dies.iter().all(|d| d.area_mm2 > 0.0 && d.area_low_mm2 <= d.area_mm2 && d.area_mm2 <= d.area_high_mm2), "{name}");
        assert!(rep.package_mm2 > 0.0 && ph.static_power_w() > 0.0, "{name}");
        for (i, c) in hw.channels.iter().enumerate() {
            let l = ph.link(i);
            assert!(l.latency_s >= 0.0 && l.energy_j_per_b >= 0.0 && l.energy_j_per_b.is_finite(), "{name} channel {i} {:?}", c.kind);
        }
        assert!(rep.problems.iter().all(|d| d.code.starts_with("E-PHYS-")), "{name}: {:?}", rep.problems);
        // The fit-set chips fit their outlines, edges and packages; the held-out GPUs (V100 +26%, H100 +8% vs their
        // published dies, 04 §12.3) and the ember example overflow and are reported (as residuals on published dies).
        if !["v100_sxm2_32gb", "h100_sxm5_80gb", "h100_pcie_80gb", "ember"].contains(&name) {
            assert!(rep.problems.is_empty(), "{name}: {:?}", rep.problems.iter().map(|d| &d.message).collect::<Vec<_>>());
        }
    }
}

fn canonical(name: &str) -> Value {
    load_file(path(name)).expect("loads").canonical
}

fn visit(v: &mut Value, f: &mut dyn FnMut(&mut serde_json::Map<String, Value>)) {
    match v {
        Value::Object(o) => {
            f(o);
            for x in o.values_mut() {
                visit(x, f);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| visit(x, f)),
        _ => {}
    }
}

/// P13: envelope area and static power are monotone non-decreasing in component count and size.
#[test]
fn p13_area_and_static_power_monotone() {
    let base = canonical("tpu_v4");
    let measure = |v: Value| {
        let hw = from_value(v);
        let ph = Phys::new(&hw);
        let a: f64 = ph.report().unwrap().dies.iter().map(|d| d.area_mm2).sum();
        (a, ph.static_power_w())
    };
    let (a0, s0) = measure(base.clone());
    // More MXUs.
    let mut more = base.clone();
    visit(&mut more, &mut |o| {
        if o.get("kind").and_then(Value::as_str) == Some("matrix") {
            let c = o.get("count").and_then(Value::as_u64).unwrap_or(1);
            o.insert("count".into(), Value::from(c + 1));
        }
    });
    // Larger memories.
    let mut bigger = base.clone();
    visit(&mut bigger, &mut |o| {
        if o.get("kind").and_then(Value::as_str) == Some("scratchpad")
            && let Some(c) = o.get("capacity").and_then(Value::as_u64)
        {
            o.insert("capacity".into(), Value::from(c * 2));
        }
    });
    // An extra vector unit.
    let mut extra = base.clone();
    let mut done = false;
    visit(&mut extra, &mut |o| {
        if done {
            return;
        }
        if let Some(Value::Array(units)) = o.get_mut("units")
            && let Some(f) = units.first().and_then(|u| u.get("feeds")).cloned()
        {
            units.push(serde_json::json!({"id": "p13_extra", "kind": "vector", "lanes": 64, "precisions": ["fp32@1"], "feeds": f}));
            done = true;
        }
    });
    for (what, v) in [("more MXUs", more), ("bigger memories", bigger), ("extra unit", extra)] {
        let (a, s) = measure(v);
        assert!(a > a0 && s > s0, "{what}: area {a0:.2} -> {a:.2}, static {s0:.3} -> {s:.3}");
    }
}

#[test]
fn search_prices_overrides_against_derived_values() {
    // 00 decision 3: a memory energy override >= the kiln-phys derived value is allowed under `search`, below is
    // unpriced (E-IR-1101); without a pricer it stays unverifiable.
    let mut v = canonical("tpu_v4");
    v.as_object_mut().unwrap().remove("family");
    let with_energy = |v: &Value, e: f64| {
        let mut v = v.clone();
        visit(&mut v, &mut |o| {
            if o.get("id").and_then(Value::as_str) == Some("cmem") {
                o.insert("overrides".into(), serde_json::json!({"read_energy": e}));
            }
        });
        v
    };
    let codes = |v: Value, priced: bool| -> Vec<String> {
        let d = Design::from_value(v).unwrap();
        let r = if priced { check_priced(d, Profile::Search, &Default::default(), Some(&kiln_phys::pricing::pricer)) } else { check(d, Profile::Search, &Default::default()) };
        r.diagnostics.iter().filter(|d| d.code.starts_with("E-")).map(|d| d.code.clone()).collect()
    };
    assert!(codes(with_energy(&v, 1.0), false).contains(&"E-IR-1101".to_string()), "unverifiable without kiln-phys");
    assert!(codes(with_energy(&v, 1.0), true).is_empty(), "1 J/B is far above any derived SRAM energy");
    assert!(codes(with_energy(&v, 1e-18), true).contains(&"E-IR-1101".to_string()), "below derived is unpriced");
}

#[test]
fn tier_a_placer_is_fast_at_2000_macros() {
    let n = 2000;
    let p = Problem {
        area: (0..n).map(|i| 1.0 + (i % 7) as f64).collect(),
        power: (0..n).map(|i| (i % 5) as f64).collect(),
        edges: (1..n).map(|i| (i - 1, i, 1.0)).chain((0..n / 2).map(|i| (i, n - 1 - i, 0.5))).collect(),
        terms: vec![(0, 0.0, 0.0, 5.0), (n - 1, 100.0, 100.0, 5.0)],
        region: Rect::new(0.0, 0.0, 100.0, 100.0),
        aspect: (0.5, 2.0),
        lambda_th: 0.0,
    };
    let t = std::time::Instant::now();
    let (_, r) = place::tier_a(&p);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(r.len(), n);
    let total: f64 = r.iter().map(Rect::area).sum();
    assert!((total - 1e4).abs() < 1e-6);
    // 04 §6.4 targets 2 ms; the bound here leaves room for debug builds and loaded CI machines.
    assert!(ms < if cfg!(debug_assertions) { 2000.0 } else { 200.0 }, "{ms} ms");
}

#[test]
fn halving_a_d2d_span_cuts_link_latency_and_energy_by_the_closed_form() {
    // T2 M4 (06 §12.2): two pinned dies, one d2d link; halving the offset of die b shortens the package trace.
    let p = format!("{}/../../corpus/trust/designs/m4_two_die.json5", env!("CARGO_MANIFEST_DIR"));
    let v = load_file(&p).unwrap().canonical;
    let mut moved = v.clone();
    let x = moved.pointer_mut("/system/boards/0/packages/0/dies/1/placement/x").unwrap();
    *x = Value::from(x.as_f64().unwrap() * 0.5);
    let link = |v: Value| {
        let hw = from_value(v);
        let ph = Phys::new(&hw);
        let c = hw.channels.iter().position(|c| c.kind == kiln_ir::hw::model::ChannelKind::D2d).expect("d2d channel");
        (ph.m3().unwrap().links[c], *ph.link(c))
    };
    let ((lc0, l0), (lc1, l1)) = (link(v), link(moved));
    let dl = lc0.length_um - lc1.length_um;
    assert!((dl - 20_000.0).abs() < 1.0, "trace shortens by the 20 mm offset change: {dl}");
    let t = kiln_phys::tables::Tables::get();
    let pk = &t.package["cowos_s"];
    let dt = l0.latency_s - l1.latency_s;
    let de = l0.energy_j_per_b - l1.energy_j_per_b;
    assert!((dt - dl * pk.trace_ps_per_mm * 1e-15).abs() <= 0.01 * dt, "{dt}");
    assert!((de - 8.0 * dl * 1e-3 * pk.trace_e_pj_per_bit_mm * 1e-12).abs() <= 0.01 * de, "{de}");
}

/// P6 on the physical side (06 §2.2, 04 §6.1): dead area (a misc block or a memory nothing connects to) leaves every
/// link bit-identical; a reachable unit no op of another precision can use only lengthens links.
#[test]
fn p6_dead_area_never_shortens_a_link() {
    let links = |v: Value| {
        let hw = from_value(v);
        let ph = Phys::new(&hw);
        let mut out = std::collections::BTreeMap::new();
        for (i, c) in hw.channels.iter().enumerate() {
            let k = (hw.nodes[hw.node_of(c.src)].path.clone(), hw.nodes[hw.node_of(c.dst)].path.clone());
            let n = out.keys().filter(|(a, b, _): &&(String, String, usize)| (a, b) == (&k.0, &k.1)).count();
            let l = ph.link(i);
            out.insert((k.0, k.1, n), (l.latency_s, l.energy_j_per_b));
        }
        out
    };
    for name in ["a100_sxm4_40gb", "tpu_v4", "tpu_v5e"] {
        let base = canonical(name);
        let l0 = links(base.clone());
        let on_die = |f: &mut dyn FnMut(&mut serde_json::Map<String, Value>)| {
            let mut v = base.clone();
            let mut done = false;
            visit(&mut v, &mut |o| {
                if !done && o.contains_key("default_clock") {
                    f(o);
                    done = true;
                }
            });
            v
        };
        let misc = on_die(&mut |o| {
            let b = o.entry("blocks").or_insert_with(|| Value::Array(vec![]));
            b.as_array_mut().unwrap().push(serde_json::json!({"id": "p6_dummy", "kind": {"type": "misc"}, "footprint": {"area": 4.0e-5}}));
        });
        let orphan = on_die(&mut |o| {
            let m = o.entry("memories").or_insert_with(|| Value::Array(vec![]));
            m.as_array_mut().unwrap().push(serde_json::json!({"id": "p6_orphan", "kind": "scratchpad", "capacity": 4194304, "ports": [{"dir": "rw", "width_bits": 256}]}));
        });
        for (what, v) in [("misc block", misc), ("orphan memory", orphan)] {
            assert_eq!(links(v), l0, "{name}: {what} moved a live link");
        }
        let mut idle = base.clone();
        let mut done = false;
        visit(&mut idle, &mut |o| {
            if done {
                return;
            }
            if let Some(Value::Array(units)) = o.get_mut("units")
                && let Some(f) = units.iter().find(|u| u.get("kind").and_then(Value::as_str) == Some("matrix")).and_then(|u| u.get("feeds")).cloned()
            {
                units.push(serde_json::json!({"id": "p6_int4", "kind": "matrix", "geometry": {"systolic": {"rows": 64, "cols": 64}},
                                              "precisions": ["int4*int4+int32"], "local": [{"id": "w", "holds": "b", "capacity": 8192}], "feeds": f}));
                done = true;
            }
        });
        assert!(done, "{name}: no matrix unit to sit beside");
        let l1 = links(idle);
        for (k, (lat, e)) in &l0 {
            let (lat1, e1) = l1[k];
            assert!(lat1 >= *lat && e1 >= *e, "{name}: unused int4 unit shortened {k:?}: {lat} -> {lat1}, {e} -> {e1}");
        }
    }
}

/// 04 §8.1: an unpublished (assumed) cap is reported with its range and never throttles; a published one does.
#[test]
fn assumed_caps_are_reported_not_enforced() {
    for (name, enforced, assumed) in [("tpu_v5e", None, Some((200.0, 120.0, 250.0))), ("tpu_v6e", None, Some((450.0, 300.0, 700.0))), ("a100_sxm4_40gb", Some(400.0), None)] {
        let ph = Phys::new(&model(name));
        assert_eq!(ph.power_cap_w(), enforced, "{name}");
        assert_eq!(ph.report().unwrap().tdp_assumed_w, assumed, "{name}");
        let (plan, _) = ph.solve_clock(Some(&|_: &[f64]| vec![1e6; ph.caps().len()]));
        assert_eq!(plan.throttled, enforced.is_some(), "{name}: a 1 MW phase throttles only under a published cap");
    }
    let mut v = canonical("tpu_v5e");
    visit(&mut v, &mut |o| {
        if let Some(a) = o.get_mut("assumed").and_then(Value::as_object_mut) {
            a.insert("hi".into(), Value::from("150W"));
        }
    });
    let r = check(Design::from_value(v).unwrap(), Profile::Full, &Default::default());
    assert!(r.diagnostics.iter().any(|d| d.code == "E-IR-0908"), "nominal 200 W above hi 150 W");
    let r = check(Design::from_value(canonical("tpu_v5e")).unwrap(), Profile::Search, &Default::default());
    assert!(r.diagnostics.iter().any(|d| d.code == "E-IR-1106"), "a searched design cannot carry an unpublished cap");
}
