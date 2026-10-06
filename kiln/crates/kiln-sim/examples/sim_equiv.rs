//! Tier A equivalence harness: maps and simulates the reference designs over the standard suite and writes
//! (`snapshot`) or compares (`check`) mapping hashes and full central/low/high results, so performance work on
//! kiln-map / kiln-cost / the engine can be shown to leave results unchanged. Numbers compare within `--rel`
//! (default 0: bit-identical); search and cache counters are excluded.
//! `cargo run --release -p kiln-sim --example sim_equiv -- snapshot|check <file> [--rel 1e-12] [design.json5...]`

use std::collections::BTreeMap;

use kiln_ir::hw::Profile;
use kiln_sim::{Prepared, SimOptions};
use kiln_trace::{IntervalMethod, TraceLevel};
use serde_json::Value;

const DESIGNS: [&str; 6] = ["a100_sxm4_40gb.json5", "v100_sxm2_32gb.json5", "h100_sxm5_80gb.json5", "tpu_v4.json5", "tpu_v5e.json5", "tpu_v6e.json5"];

fn fp16(mut m: kiln_wl::zoo::SuiteMember) -> kiln_wl::zoo::SuiteMember {
    use kiln_ir::wl::{ElemType, ModelSrc};
    if let ModelSrc::Full(model) = &mut m.doc.model {
        let model = model.as_mut();
        let tensors = model.tensors.values_mut().chain(model.graphs.values_mut().flat_map(|g| g.tensors.values_mut()));
        for t in tensors.filter(|t| t.dtype == ElemType::BF16) {
            t.dtype = ElemType::from(kiln_ir::precision::Precision::Fp16);
        }
    }
    m
}

fn strip(r: &kiln_trace::sim::SimResult) -> Value {
    let mut v = serde_json::to_value(r).expect("result");
    v["provenance"] = Value::Null;
    v["cost_model"] = Value::Null;
    v
}

/// Largest relative difference between two JSON trees (`INFINITY` on a structural or text difference).
fn diff(a: &Value, b: &Value, path: &str, worst: &mut (f64, String)) {
    fn set(worst: &mut (f64, String), d: f64, p: &str) {
        if d > worst.0 {
            *worst = (d, p.to_string());
        }
    }
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(0.0), y.as_f64().unwrap_or(0.0));
            if x != y {
                set(worst, (x - y).abs() / x.abs().max(y.abs()), path);
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                diff(p, q, &format!("{path}[{i}]"), worst);
            }
        }
        (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
            for (k, p) in x {
                match y.get(k) {
                    Some(q) => diff(p, q, &format!("{path}.{k}"), worst),
                    None => set(worst, f64::INFINITY, &format!("{path}.{k}")),
                }
            }
        }
        _ if a == b => {}
        _ => set(worst, f64::INFINITY, path),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, file) = (args[0].as_str(), args[1].clone());
    let mut rel = 0.0;
    let mut designs: Vec<String> = vec![];
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        if a == "--rel" {
            rel = it.next().and_then(|x| x.parse().ok()).expect("--rel <x>");
        } else {
            designs.push(a.clone());
        }
    }
    if designs.is_empty() {
        designs = DESIGNS.iter().map(|s| s.to_string()).collect();
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference");
    let configs = [
        ("w3", SimOptions { window: 3, interval: IntervalMethod::Corners, shadow_prices: true, trace: TraceLevel::Ops, layer_scope_fallback: true, ..SimOptions::default() }),
        ("w1", SimOptions { window: 1, interval: IntervalMethod::None, shadow_prices: false, layer_scope_fallback: true, ..SimOptions::default() }),
    ];
    let mut got: BTreeMap<String, Value> = BTreeMap::new();
    for f in &designs {
        let p = Prepared::from_file(&root.join(f), Profile::Reference).expect("design");
        let no_bf16 = p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None) == 0.0;
        for m in kiln_wl::zoo::suite("standard").expect("suite") {
            let m = if no_bf16 { fp16(m) } else { m };
            for (tag, opts) in &configs {
                let key = format!("{f}/{}/{tag}", m.scenario);
                let v = match kiln_sim::simulate_member(&p, &m, opts) {
                    Ok((run, _)) => serde_json::json!({
                        "mapping": run.mapping.hash(),
                        "central": strip(&run.central),
                        "low": run.low.as_ref().map(strip),
                        "high": run.high.as_ref().map(strip),
                    }),
                    Err(e) => Value::String(format!("{e:?}")),
                };
                eprintln!("{key}");
                got.insert(key, v);
            }
        }
    }
    if mode == "snapshot" {
        std::fs::write(&file, serde_json::to_string(&got).expect("json")).expect("write");
        return;
    }
    let want: BTreeMap<String, Value> = serde_json::from_str(&std::fs::read_to_string(&file).expect("snapshot")).expect("json");
    let mut bad = 0;
    let mut max_rel = 0.0f64;
    for (k, g) in &got {
        let Some(w) = want.get(k) else {
            println!("MISSING {k}");
            continue;
        };
        if g["mapping"] != w["mapping"] {
            bad += 1;
            println!("MAPPING {k}");
            continue;
        }
        let mut worst = (0.0, String::new());
        diff(w, g, "", &mut worst);
        max_rel = max_rel.max(worst.0);
        if worst.0 > rel {
            bad += 1;
            println!("DIFF {k}: {:.3e} at {}", worst.0, worst.1);
        }
    }
    println!("{} cases, {bad} over tolerance {rel:e}, max relative difference {max_rel:.3e}", got.len());
    std::process::exit(i32::from(bad > 0));
}
