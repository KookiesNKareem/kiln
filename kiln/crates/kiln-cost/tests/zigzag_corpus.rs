//! L3 differential test vs ZigZag 3.8.5 (06 §2.3), replaying `tests/data/zigzag_corpus.json` produced by
//! `kiln/oracles/zigzag_diff.py`: fixed mapping, `zigzag_compat` accounting; per-level access counts must match
//! exactly, latency and energy within 1%, unless the case matches an entry of `oracles/known_diffs.json`.

use std::path::PathBuf;

use kiln_cost::{CostOptions, Mapping, OpNest, UnitTemplate, evaluate};
use serde_json::Value;

const TOL: f64 = 0.01;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn known_diffs() -> Vec<(String, String)> {
    let p = root().join("../../oracles/known_diffs.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).expect("known_diffs.json")).expect("json");
    v["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|e| (e["case_prefix"].as_str().unwrap_or("").to_owned(), e["metric"].as_str().unwrap_or("").to_owned()))
        .collect()
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b { 0.0 } else { (a - b).abs() / b.abs().max(1e-30) }
}

#[test]
fn zigzag_fixed_mapping_corpus() {
    let corpus: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("tests/data/zigzag_corpus.json")).expect("corpus")).expect("json");
    assert_eq!(corpus["version"], "3.8.5", "corpus must come from the pinned oracle");
    let known = known_diffs();
    let allowed = |name: &str, metric: &str| known.iter().any(|(p, m)| name.starts_with(p.as_str()) && m == metric);
    let opts = CostOptions { zigzag_compat: true, ..Default::default() };
    let cases = corpus["cases"].as_array().expect("cases");
    let (mut count_ok, mut count_n, mut lat_ok, mut en_ok) = (0, 0, 0, 0);
    let (mut worst_lat, mut worst_en) = ((0.0f64, String::new()), (0.0f64, String::new()));
    let mut failures = vec![];
    let mut report = vec![];
    for c in cases {
        let name = c["name"].as_str().expect("name");
        let unit: UnitTemplate = serde_json::from_value(c["unit"].clone()).expect("unit");
        let nest: OpNest = serde_json::from_value(c["nest"].clone()).expect("nest");
        let mapping: Mapping = serde_json::from_value(c["mapping"].clone()).expect("mapping");
        let zz = &c["zigzag"];
        let e = evaluate(&unit, &nest, &mapping, &opts).unwrap_or_else(|d| panic!("{name}: {d}"));
        let mut counts_match = true;
        for (si, op) in ["I", "W", "O"].iter().enumerate() {
            let chain = &unit.chains[si].levels;
            for (j, lv) in zz["counts"][op].as_array().expect("counts").iter().enumerate() {
                let a = e.accesses.iter().find(|a| a.operand == si && a.level == chain[j]).expect("access row");
                let mine = [("to_low", a.to_low), ("from_high", a.from_high), ("to_high", a.to_high), ("from_low", a.from_low)];
                for (k, v) in mine {
                    count_n += 1;
                    if lv[k].as_u64() == Some(v) {
                        count_ok += 1;
                    } else {
                        counts_match = false;
                        if !allowed(name, "counts") {
                            failures.push(format!("{name}: {op} level {j} {k}: kiln {v} vs zigzag {}", lv[k]));
                        }
                    }
                }
            }
        }
        let lat = e.latency_s * unit.clock_hz;
        let zl = zz["latency_cycles"].as_f64().expect("lat");
        let rl = rel(lat, zl);
        if rl <= TOL {
            lat_ok += 1;
        } else if !allowed(name, "latency") {
            failures.push(format!(
                "{name}: latency kiln {lat} (stall {} on {} off {}) vs zigzag {zl} (stall {} on {} off {})",
                e.stall_cycles,
                "-",
                e.fill_drain_cycles,
                zz["stall"],
                zz["onloading"],
                zz["offloading"]
            ));
        }
        if rl > worst_lat.0 {
            worst_lat = (rl, name.to_owned());
        }
        let ze = zz["energy_j"].as_f64().expect("energy");
        let re = rel(e.energy.total_j, ze);
        if re <= TOL {
            en_ok += 1;
        } else if !allowed(name, "energy") {
            failures.push(format!("{name}: energy kiln {} vs zigzag {ze}", e.energy.total_j));
        }
        if re > worst_en.0 {
            worst_en = (re, name.to_owned());
        }
        report.push(serde_json::json!({
            "name": name, "counts_match": counts_match, "latency_rel_err": rl, "energy_rel_err": re,
            "kiln_latency_cycles": lat, "zigzag_latency_cycles": zl, "kiln_energy_j": e.energy.total_j, "zigzag_energy_j": ze,
        }));
    }
    let n = cases.len();
    println!(
        "zigzag 3.8.5 fixed-mapping corpus: {n} cases; access counts {count_ok}/{count_n} exact; latency within 1%: {lat_ok}/{n} \
         (worst {:.3}% {}); energy within 1%: {en_ok}/{n} (worst {:.3}% {})",
        worst_lat.0 * 100.0,
        worst_lat.1,
        worst_en.0 * 100.0,
        worst_en.1
    );
    let out = root().join("../../target/zigzag_diff_report.json");
    let _ = std::fs::write(out, serde_json::to_string_pretty(&report).unwrap_or_default());
    assert!(failures.is_empty(), "{} disagreements:\n{}", failures.len(), failures.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}
