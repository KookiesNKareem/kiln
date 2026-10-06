//! Search-equivalence harness: runs [`cost`] on a fixed query corpus and writes (`snapshot`) or compares
//! (`check`) every chosen mapping and cost, so search speedups can be shown to leave results unchanged.
//! Corpus: the 48 Llama golden tiles (default, kiln-map full and quick budgets), the ZigZag corpus nests
//! (default and `zigzag_compat`), and optionally a JSON-lines query dump (`tpl`/`unit`/`nest`/`objective`/
//! `options` per line, `unit` given once per `tpl`).
//! A dump comes from any workload run with `KILN_COST_DUMP=<file>` (every cold query of a [`CostCache`]).
//! `REPS=<n>` repeats the corpus for timing (best of n); `ONLY=<prefix>` restricts it (`llama`, `zigzag`, `dump`);
//! `BUDGET=<n>` overrides every query's evaluation budget (stresses truncation); `CACHE=1` answers every query
//! through one [`CostCache`] (its cross-query class memo included) instead of [`cost`].
//! `cargo run --release -p kiln-cost --example search_equiv -- snapshot|check <file> [dump.jsonl]`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use kiln_cost::*;
use kiln_ir::hw::{HwModel, Profile, check_file};
use kiln_ir::wl::KernelClass;
use serde_json::Value;

const FULL: CostOptions = CostOptions {
    ragged: RaggedPolicy::Split,
    budget: SearchBudget { top_k_spatial: 8, max_evals_per_spatial: 20_000, stop_at_floor: true },
    zigzag_compat: false,
};
const QUICK: CostOptions = CostOptions { budget: SearchBudget { top_k_spatial: 2, max_evals_per_spatial: 300, stop_at_floor: true }, ..FULL };

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model(name: &str) -> HwModel {
    check_file(root().join("../../designs/reference").join(name), Profile::Reference).model.expect("reference design")
}

fn unit_ix(hw: &HwModel, suffix: &str) -> usize {
    hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with(suffix)).expect("unit")
}

/// The golden test's 48 tiles (tests/golden.rs): weight GEMMs of decode_b8 and prefill_b1 on an A100 SM and a v5e TensorCore.
fn llama_tiles() -> Vec<(String, UnitTemplate, OpNest)> {
    let a100 = model("a100_sxm4_40gb.json5");
    let sm = UnitTemplate::from_hw(&a100, unit_ix(&a100, "gpc0_0.tpc0.sm0.smsp0.tc"), &TemplateOptions { gang: 4, bw_assumed: Some(1555e9 / 108.0), ..Default::default() })
        .expect("sm");
    let tpu = model("tpu_v5e.json5");
    let tc = UnitTemplate::from_hw(&tpu, unit_ix(&tpu, "tc.mxu0"), &TemplateOptions { gang: 4, ..Default::default() }).expect("tc");
    let level = |t: &UnitTemplate, s: &str| t.levels.iter().position(|l| l.name.contains(s)).expect("level");
    let targets = [("a100-sm", level(&sm, "l2p"), sm, Some((256u64, 128u64))), ("v5e-tc", level(&tc, "vmem"), tc, None)];
    let mut out = vec![];
    for w in ["llama3_8b:decode_b8", "llama3_8b:prefill_b1"] {
        let m = kiln_wl::zoo::workload(w).expect("workload");
        let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
        for n in &lg.nodes {
            for (ki, k) in n.lowered.kernels.iter().enumerate() {
                if k.class != KernelClass::Contraction || !["proj", "qkv", "gate_up", "down"].iter().any(|s| n.path.ends_with(s)) {
                    continue;
                }
                let nest = OpNest::from_lowered(n, ki).expect("nest");
                for (name, act, unit, tile) in &targets {
                    let mut sizes: Vec<u64> = nest.dims.iter().map(|d| d.size).collect();
                    let uses = |role, d: usize| nest.operands.iter().any(|o| o.role == role && o.axes.iter().any(|a| a.terms.iter().any(|&(x, _)| x == d)));
                    if let Some((cm, cn)) = tile {
                        for (d, s) in sizes.iter_mut().enumerate() {
                            if nest.dims[d].kind != LoopKind::Parallel {
                                continue;
                            }
                            let (a, b) = (uses(kiln_ir::hw::compute::OperandRole::A, d), uses(kiln_ir::hw::compute::OperandRole::B, d));
                            if a && !b {
                                *s = (*s).min(*cm);
                            } else if b && !a {
                                *s = (*s).min(*cn);
                            }
                        }
                    }
                    let mut x = nest.tile(&sizes, None);
                    let room = unit.levels[*act].capacity_bytes / 4;
                    let bytes = |o: &NestOperand| o.axes.iter().map(|a| sizes[a.terms[0].0]).product::<u64>() * 2;
                    let fits = x.operands.iter().filter(|o| o.role != kiln_ir::hw::compute::OperandRole::B).map(bytes).sum::<u64>() <= room;
                    for o in &mut x.operands {
                        if fits && o.role != kiln_ir::hw::compute::OperandRole::B {
                            if o.is_output {
                                o.sink = Some(*act);
                            } else {
                                o.source = Some(*act);
                            }
                        }
                    }
                    out.push((format!("{name}/{w}/{}", n.path), unit.clone(), x));
                }
            }
        }
    }
    out
}

struct Q {
    name: String,
    unit: usize,
    nest: OpNest,
    objective: Objective,
    options: CostOptions,
}

fn corpus(dump: Option<&str>) -> (Vec<UnitTemplate>, Vec<Q>) {
    let mut units: Vec<UnitTemplate> = vec![];
    let mut qs = vec![];
    for (name, u, n) in llama_tiles() {
        units.push(u);
        for (tag, o) in [("default", CostOptions::default()), ("full", FULL), ("quick", QUICK)] {
            qs.push(Q { name: format!("llama/{name}/{tag}"), unit: units.len() - 1, nest: n.clone(), objective: Objective::Latency, options: o });
        }
    }
    let zz: Value = serde_json::from_str(&std::fs::read_to_string(root().join("tests/data/zigzag_corpus.json")).expect("corpus")).expect("json");
    for c in zz["cases"].as_array().expect("cases") {
        let name = c["name"].as_str().expect("name");
        units.push(serde_json::from_value(c["unit"].clone()).expect("unit"));
        let nest: OpNest = serde_json::from_value(c["nest"].clone()).expect("nest");
        for (tag, o) in [("default", CostOptions::default()), ("compat", CostOptions { zigzag_compat: true, ..Default::default() })] {
            qs.push(Q { name: format!("zigzag/{name}/{tag}"), unit: units.len() - 1, nest: nest.clone(), objective: Objective::Latency, options: o });
        }
    }
    if let Some(path) = dump {
        let mut tpl: BTreeMap<String, usize> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for (i, line) in std::fs::read_to_string(path).expect("dump").lines().enumerate() {
            let v: Value = serde_json::from_str(line).expect("dump line");
            let h = v["tpl"].as_str().expect("tpl").to_string();
            if !v["unit"].is_null() && !tpl.contains_key(&h) {
                units.push(serde_json::from_value(v["unit"].clone()).expect("unit"));
                tpl.insert(h.clone(), units.len() - 1);
            }
            if !seen.insert(format!("{h}|{}|{}|{}", v["nest"], v["objective"], v["options"])) {
                continue;
            }
            qs.push(Q {
                name: format!("dump/{i}"),
                unit: tpl[&h],
                nest: serde_json::from_value(v["nest"].clone()).expect("nest"),
                objective: serde_json::from_value(v["objective"].clone()).expect("objective"),
                options: serde_json::from_value(v["options"].clone()).expect("options"),
            });
        }
    }
    (units, qs)
}

/// Everything a caller sees except the search effort counters (evaluated / pruned).
fn result(r: Result<CostEntry, kiln_ir::common::Diagnostic>) -> String {
    match r {
        Ok(mut e) => {
            e.search.evaluated = 0;
            e.search.pruned = 0;
            serde_json::to_string(&e).expect("entry")
        }
        Err(d) => format!("ERR {} {}", d.code, d.message),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, file) = (args[0].as_str(), &args[1]);
    let (units, qs) = corpus(args.get(2).map(String::as_str));
    let reps: usize = std::env::var("REPS").ok().and_then(|x| x.parse().ok()).unwrap_or(1);
    let only = std::env::var("ONLY").unwrap_or_default();
    let cached = std::env::var_os("CACHE").is_some();
    let budget: Option<u64> = std::env::var("BUDGET").ok().and_then(|x| x.parse().ok());
    let mut qs: Vec<Q> = qs.into_iter().filter(|q| q.name.starts_with(&only)).collect();
    if let Some(b) = budget {
        qs.iter_mut().for_each(|q| q.options.budget.max_evals_per_spatial = b);
    }
    let hashes: Vec<String> = units.iter().map(UnitTemplate::hash).collect();
    let mut by_src: BTreeMap<&str, f64> = BTreeMap::new();
    let mut got: Vec<(String, String)> = vec![];
    for rep in 0..reps {
        let cache = CostCache::new();
        let mut this: BTreeMap<&str, f64> = BTreeMap::new();
        for q in &qs {
            let t = Instant::now();
            let cq = CostQuery { unit: &units[q.unit], nest: &q.nest, objective: q.objective, options: q.options };
            let r = if cached { cache.query_hashed(&cq, &hashes[q.unit]).map(|e| (*e).clone()) } else { cost(&cq) };
            *this.entry(q.name.split('/').next().expect("src")).or_default() += t.elapsed().as_secs_f64() * 1e3;
            if rep == 0 {
                got.push((q.name.clone(), result(r)));
            }
        }
        for (k, v) in this {
            let e = by_src.entry(k).or_insert(f64::INFINITY);
            *e = e.min(v);
        }
    }
    for (s, ms) in &by_src {
        println!("{s:<8} {ms:9.1} ms (best of {reps})");
    }
    if mode == "snapshot" {
        let text: String = got.iter().map(|(n, r)| format!("{n}\t{r}\n")).collect();
        std::fs::write(file, text).expect("write");
        return;
    }
    let want: BTreeMap<String, String> =
        std::fs::read_to_string(file).expect("snapshot").lines().map(|l| l.split_once('\t').map(|(a, b)| (a.to_string(), b.to_string())).expect("line")).collect();
    let mut bad = 0;
    for (n, r) in &got {
        match want.get(n) {
            Some(w) if w == r => {}
            Some(_) => {
                bad += 1;
                if bad <= 10 {
                    println!("MISMATCH {n}");
                }
            }
            None => println!("MISSING {n}"),
        }
    }
    println!("{bad} mismatches of {} queries", got.len());
    std::process::exit(i32::from(bad > 0));
}
