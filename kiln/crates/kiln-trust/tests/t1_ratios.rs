//! T1 published-ratio corpus (06 §12.1): provenance, arithmetic and grading rules.

use std::collections::BTreeMap;

use kiln_ir::hw::{Design, Profile};
use kiln_trace::sim::Scope;
use serde_json::Value;

const CHIPS: [&str; 8] =
    ["v100_sxm2_32gb", "a100_sxm4_40gb", "a100_sxm4_80gb", "h100_sxm5_80gb", "h100_pcie_80gb", "tpu_v4", "tpu_v5e", "tpu_v6e"];

fn corpus() -> Value {
    let text = std::fs::read_to_string(kiln_trust::corpus_path("published_ratios.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or_else(|| panic!("{k} missing in {v}"))
}

#[test]
fn entries_are_sourced_graded_and_consistent() {
    let c = corpus();
    assert_eq!(s(&c, "schema"), "kiln.trust.published_ratios/1");
    let entries = c["entries"].as_array().unwrap();
    assert!(entries.len() >= 10);
    let mut ids: Vec<&str> = entries.iter().map(|e| s(e, "id")).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), entries.len(), "unique ids");
    for e in entries {
        let id = s(e, "id");
        let (num, den) = (s(e, "numerator"), s(e, "denominator"));
        assert!(CHIPS.contains(&num) && CHIPS.contains(&den) && num != den, "{id}");
        let ratio = e["ratio"].as_f64().unwrap();
        assert!(ratio > 0.0, "{id}");
        let (grade, tol) = (s(e, "grade"), e["tolerance"].as_f64().unwrap());
        assert!(matches!((grade, tol), ("G3", 0.20) | ("G4", 0.25)), "{id}: 06 §12.1 tolerance per grade");
        let fit = [num, den].iter().any(|c| c.starts_with("a100") || *c == "tpu_v5e");
        assert_eq!(e["fit_device"].as_bool(), Some(fit), "{id}: A100 and v5e are generic-set fit devices");
        assert!(matches!(s(&e["workload"], "benchmark_class"), "llm" | "non_llm"), "{id}");
        assert!(e["comparable"].is_boolean() && e["confounders"].is_array(), "{id}");
        assert!(matches!(s(e, "confidence"), "high" | "medium" | "low"), "{id}");
        if grade == "G4" {
            let (n, d) = (&e["numerator_source"], &e["denominator_source"]);
            for src in [n, d] {
                assert!(s(src, "url").starts_with("https://"), "{id}");
                assert!(!s(src, "round").is_empty(), "{id}");
                let per = src["value"].as_f64().unwrap() / src["accelerators"].as_f64().unwrap();
                assert!((per / src["per_accelerator"].as_f64().unwrap() - 1.0).abs() < 1e-4, "{id}: per-accelerator arithmetic");
            }
            let r = n["per_accelerator"].as_f64().unwrap() / d["per_accelerator"].as_f64().unwrap();
            assert!((r / ratio - 1.0).abs() < 1e-3, "{id}: ratio {ratio} vs {r}");
        } else {
            assert!(s(&e["source"], "url").starts_with("https://") && !s(&e["source"], "quote").is_empty(), "{id}");
        }
    }
}

#[test]
fn comparable_llm_entries_exist() {
    let c = corpus();
    let usable: Vec<&Value> =
        c["entries"].as_array().unwrap().iter().filter(|e| e["comparable"] == true && e["workload"]["benchmark_class"] == "llm").collect();
    assert!(usable.len() >= 5, "comparable LLM entries: {}", usable.len());
    assert!(c["rejected"].as_array().is_some_and(|r| !r.is_empty()));
}

/// Generation shape per corpus model: the only configuration the corpus states for GPT-J is the TRT-LLM one
/// (batch 64, 128 in / 128 out); JetStream Llama-2-70B is 1024 in / 1024 out with batch unstated (32 assumed).
fn shape(model: &str) -> Option<(&'static str, u64, u64, u64)> {
    match model {
        m if m.starts_with("gpt-j") => Some(("gptj_6b", 64, 128, 128)),
        "llama-2-70b" => Some(("llama2_70b", 32, 1024, 1024)),
        _ => None,
    }
}

/// A100 SXM4 80GB from the 40GB reference: five active HBM2e stacks of 16 GiB at 2,039 GB/s (pub:ds).
fn a100_80gb() -> Design {
    let c = kiln_trust::load("a100_sxm4_40gb").unwrap().canonical;
    let mut v = kiln_trust::transform::scale_offchip_bandwidth(&c, 2039.0 / 1555.2).unwrap();
    kiln_trust::transform::visit_mut(&mut v, &mut |o| {
        if let Some(Value::Array(stacks)) = o.get_mut("mem_stacks") {
            for s in stacks.iter_mut().filter_map(Value::as_object_mut) {
                s.insert("kind".into(), "hbm2e".into());
                s.insert("capacity".into(), Value::from(16u64 << 30));
            }
        }
    });
    let r = kiln_trust::check_value(v, Profile::Full);
    assert!(!r.has_errors(), "{:#?}", r.errors().collect::<Vec<_>>());
    r.design.unwrap()
}

fn design(chip: &str) -> Design {
    if chip == "a100_sxm4_80gb" { a100_80gb() } else { kiln_trust::load(chip).unwrap() }
}

/// Seconds for one whole generation `prefill(B, P) + G x decode(B, kv = P + G/2)` on one chip. A phase whose
/// resident set does not fit is scored at `scope: layer` (02 §12.5); it is scaled by the layer count and flagged.
fn generation(chip: &str, preset: &str, b: u64, p: u64, g: u64) -> Result<(f64, bool), String> {
    let d = &design(chip);
    let layers = f64::from(kiln_wl::zoo::preset(preset).unwrap().n_layers);
    let mut layer_scope = false;
    let mut t = |w: String| -> Result<f64, String> {
        let (ph, _, _) = kiln_trust::run_phase(d, &w)?;
        let l = ph.scope == Scope::Layer;
        layer_scope |= l;
        Ok(ph.time_s.central * if l { layers } else { 1.0 })
    };
    let pre = t(format!("{preset}:prefill_b{b}_s{p}"))?;
    let dec = t(format!("{preset}:decode_b{b}_kv{}", p + g / 2))?;
    println!(
        "{chip:<15} {preset} b{b} p{p} g{g}: prefill {pre:.4e} s, decode step {dec:.4e} s, generation {:.4} s{}",
        pre + g as f64 * dec,
        if layer_scope { " (layer scope x L)" } else { "" }
    );
    Ok((pre + g as f64 * dec, layer_scope))
}

/// T1 dry run (uncalibrated, generic parameter set): predicted single-chip generation-throughput ratios vs the
/// published per-accelerator ratios, with the 06 §12.1 grade tolerance. Reports; asserts only that LLM entries
/// with a zoo model evaluate.
#[test]
fn t1_backtest_against_engine() {
    let c = corpus();
    let mut cache: BTreeMap<String, Result<(f64, bool), String>> = BTreeMap::new();
    let mut rows = vec![];
    for e in c["entries"].as_array().unwrap().iter().filter(|e| e["workload"]["benchmark_class"] == "llm") {
        let (id, num, den) = (s(e, "id"), s(e, "numerator"), s(e, "denominator"));
        let Some((preset, b, p, g)) = shape(s(&e["workload"], "model").trim_end_matches("/MLPerf")) else {
            println!("{id}: model {} has no zoo preset; skipped", e["workload"]["model"]);
            continue;
        };
        let mut gen_time =
            |chip: &str| cache.entry(format!("{chip}/{preset}")).or_insert_with(|| generation(chip, preset, b, p, g)).clone();
        let (tn, tden) = (gen_time(num), gen_time(den));
        let ((tn, ln), (td, ld)) = match (tn, tden) {
            (Ok(a), Ok(b)) => (a, b),
            (x, y) if [&x, &y].iter().any(|r| r.as_ref().is_err_and(|e| e.contains("E-MAP-CAP-001"))) => {
                rows.push(format!("{id:<56} {num}/{den} {preset} b{b} p{p} g{g}: INFEASIBLE on one chip (model + KV do not fit; never extrapolated)"));
                continue;
            }
            (a, b) => panic!("{id}: {:?} / {:?}", a.err(), b.err()),
        };
        let predicted = td / tn;
        let published = e["ratio"].as_f64().unwrap();
        let tol = e["tolerance"].as_f64().unwrap();
        let err = predicted / published - 1.0;
        let verdict = if err.abs() <= tol { "PASS" } else { "FAIL" };
        let flags = [(ln, num), (ld, den)].iter().filter(|(l, _)| *l).map(|(_, c)| format!("{c} layer-scope")).collect::<Vec<_>>();
        rows.push(format!(
            "{id:<56} {num}/{den} {preset} b{b} p{p} g{g}: predicted {predicted:.3} published {published:.3} err {:+.1}% tol {:.0}% {verdict}{}{} [{} {}]",
            100.0 * err,
            100.0 * tol,
            if e["comparable"] == true { "" } else { " (comparable=false)" },
            if flags.is_empty() { String::new() } else { format!(" ({})", flags.join(", ")) },
            s(e, "grade"),
            s(e, "confidence"),
        ));
    }
    let a40 = generation("a100_sxm4_40gb", "gptj_6b", 64, 128, 128).unwrap().0;
    let a80 = cache["a100_sxm4_80gb/gptj_6b"].as_ref().unwrap().0;
    println!("a100 80GB/40GB gptj generation speedup {:.3} (reference only; corpus pairs use the 80GB part)", a40 / a80);
    for r in &rows {
        println!("{r}");
    }
    assert!(rows.len() >= 6, "{rows:#?}");
}
