//! Golden whole-step numbers (06 §5.3). `KILN_BLESS=1 cargo test -p kiln-sim --test golden` rewrites them;
//! a change is a model change and is reviewed as one.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::*;
use kiln_sim::SimOptions;
use kiln_trace::IntervalMethod;

const REL: f64 = 1e-12;

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/reference_m1.json")
}

#[test]
fn reference_whole_steps_match_goldens() {
    let opts = SimOptions { interval: IntervalMethod::Corners, shadow_prices: false, layer_scope_fallback: true, ..SimOptions::default() };
    let mut got: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let cases: [(&str, &[&str]); 3] = [
        ("tpu_v5e.json5", &["prefill_b1", "decode_b1", "decode_b8", "decode_b32"]),
        ("tpu_v6e.json5", &["prefill_b1", "decode_b1", "decode_b8", "decode_b32"]),
        ("a100_sxm4_40gb.json5", &["decode_b1"]),
    ];
    for (d, phases) in cases {
        let p = reference(d);
        for ph in phases {
            let m = kiln_wl::zoo::workload(&format!("llama3_8b:{ph}")).unwrap();
            let (run, _) = kiln_sim::simulate_member(&p, &m, &opts).unwrap();
            let c = &run.central;
            let e = got.entry(format!("{d}/{ph}")).or_default();
            e.insert("makespan_s".into(), c.makespan_s);
            e.insert("t_a2_s".into(), c.t_a2_s);
            e.insert("t_a0_s".into(), c.t_a0_s);
            e.insert("energy_j".into(), c.energy.total_j);
            e.insert("low_s".into(), run.time.high);
            e.insert("high_s".into(), run.time.low);
        }
    }
    if std::env::var("KILN_BLESS").is_ok() {
        std::fs::write(path(), serde_json::to_string_pretty(&got).unwrap() + "\n").unwrap();
        return;
    }
    let want: BTreeMap<String, BTreeMap<String, f64>> = serde_json::from_str(&std::fs::read_to_string(path()).expect("goldens exist")).unwrap();
    assert_eq!(want.keys().collect::<Vec<_>>(), got.keys().collect::<Vec<_>>());
    for (k, w) in &want {
        for (f, x) in w {
            let y = got[k][f];
            assert!((x - y).abs() <= REL * x.abs(), "{k}.{f}: golden {x}, now {y}");
        }
    }
}
