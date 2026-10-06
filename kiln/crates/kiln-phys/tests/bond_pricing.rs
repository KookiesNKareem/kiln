//! Declared vertical bonds reach link costing and are priced against their bond table (E-IR-UNPRICED).

use kiln_ir::hw::{Design, HwModel, Profile, check, check_priced, load_file};
use kiln_phys::Phys;
use kiln_phys::pricing::pricer;
use serde_json::Value;

fn ember() -> Value {
    load_file(format!("{}/../../designs/reference/ember.json5", env!("CARGO_MANIFEST_DIR"))).expect("loads").canonical
}

fn visit(v: &mut Value, f: &mut dyn FnMut(&mut serde_json::Map<String, Value>)) {
    match v {
        Value::Object(o) => {
            f(o);
            o.values_mut().for_each(|x| visit(x, f));
        }
        Value::Array(a) => a.iter_mut().for_each(|x| visit(x, f)),
        _ => {}
    }
}

fn set_vertical(v: &mut Value, key: &str, x: Value) {
    visit(v, &mut |o| {
        if o.get("type") == Some(&Value::from("vertical")) {
            o.insert(key.into(), x.clone());
        }
    });
}

fn bond_errors(v: Value) -> Vec<String> {
    let r = check_priced(Design::from_value(v).unwrap(), Profile::Search, &Default::default(), Some(&pricer));
    r.errors().filter(|d| d.message.contains("bond pitch") || d.message.contains("signals")).map(|d| d.code.clone()).collect()
}

#[test]
fn package_level_vertical_bonds_reach_link_costing() {
    let energy = |v: Value| {
        let hw: HwModel = check(Design::from_value(v).unwrap(), Profile::Full, &Default::default()).model.unwrap();
        let ph = Phys::new(&hw);
        hw.channels.iter().enumerate().filter(|(_, c)| c.kind == kiln_ir::hw::model::ChannelKind::Vertical).map(|(i, _)| ph.link(i).energy_j_per_b).collect::<Vec<_>>()
    };
    let hybrid = energy(ember());
    let mut tsv = ember();
    set_vertical(&mut tsv, "bond", Value::from("tsv"));
    let tsv = energy(tsv);
    assert_eq!(hybrid.len(), 8);
    assert!(hybrid.iter().zip(&tsv).all(|(h, t)| (h - 0.4e-12).abs() < 1e-18 && (t - 0.8e-12).abs() < 1e-18), "{hybrid:?} {tsv:?}");
}

#[test]
fn vertical_bond_pitches_finer_than_their_table_are_unpriced() {
    assert_eq!(bond_errors(ember()), Vec::<String>::new());
    let mut coarse = ember();
    set_vertical(&mut coarse, "pitch_um", Value::from(12.0));
    assert_eq!(bond_errors(coarse), Vec::<String>::new(), "a coarser pitch is a cost-neutral de-rate");
    let mut fine = ember();
    set_vertical(&mut fine, "pitch_um", Value::from(5.0));
    let e = bond_errors(fine);
    assert!(!e.is_empty() && e.iter().all(|c| c == "E-IR-1101"), "{e:?}");
    let mut signals = ember();
    set_vertical(&mut signals, "signals", Value::from(100000));
    let e = bond_errors(signals);
    assert!(!e.is_empty() && e.iter().all(|c| c == "E-IR-1104"), "{e:?}");
}

#[test]
fn layer_bonds_of_vertically_attached_stacks_are_priced() {
    let mini = |pitch: f64| -> Value {
        let src = format!(
            r#"{{ schema: "kiln.hw/1.0", name: "mini", tech: "tsmc_n7",
              clocks: [ {{ id: "clk", freq: "1GHz" }} ],
              system: {{ package: {{ id: "chip",
                layers: [ {{ id: "base", index: 0, bond: "microbump", pitch_um: {pitch} }} ],
                dies: [ {{ id: "die", default_clock: "clk", layer: "base",
                  units: [ {{ id: "v", kind: "vector", lanes: 16, precisions: ["fp32@1"], feeds: {{ any: "hbm" }} }} ] }} ],
                mem_stacks: [ {{ id: "hbm", kind: "hbm3", capacity: "8GiB", io_width_bits: 1024, pin_rate_bits_per_s: "6.4Gbps",
                                 attach: {{ die: "die" }} }} ] }} }} }}"#
        );
        Design::from_source(&kiln_ir::hw::MemLoader::default(), None, &src).unwrap().canonical
    };
    assert_eq!(bond_errors(mini(40.0)), Vec::<String>::new());
    assert_eq!(bond_errors(mini(20.0)), vec!["E-IR-1101".to_string()]);
}
