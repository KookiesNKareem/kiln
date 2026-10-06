//! Kernels per layer the default A100 stack recipe issues for a zoo preset (out-of-sample check, 2026-10-05).
//! `cargo run --release -p kiln-sim --example oos_kernels -- <preset> [n_layers]`

use std::collections::BTreeMap;
use std::path::PathBuf;

use kiln_ir::common::Id;
use kiln_ir::hw::Profile;
use kiln_ir::wl::WorkloadDoc;
use kiln_sim::{Prepared, SimOptions, simulate_member};
use kiln_trace::IntervalMethod;
use kiln_wl::zoo::{SuiteMember, build_model, preset, scenario};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("gptj_6b");
    let mut cfg = preset(name).unwrap();
    if let Some(l) = args.get(1) {
        cfg.n_layers = l.parse().unwrap();
    }
    let p = Prepared::from_file(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/a100_sxm4_40gb.json5"), Profile::Reference).unwrap();
    let opts = SimOptions { interval: IntervalMethod::None, shadow_prices: false, layer_scope_fallback: true, ..SimOptions::default() };
    let mut out = serde_json::Map::new();
    for phase in ["decode_b1", "decode_b8", "decode_b32", "prefill_b1"] {
        let mut doc = WorkloadDoc::new(Id::new(name).unwrap(), build_model(&cfg).unwrap());
        let sid = Id::new(phase).unwrap();
        doc.scenarios.insert(sid.clone(), scenario(phase).unwrap());
        let m = SuiteMember { name: format!("{name}:{phase}"), doc, scenario: sid };
        let (run, prog) = match simulate_member(&p, &m, &opts) {
            Ok(x) => x,
            Err(e) => {
                println!("{phase}: {:?}", e.iter().map(|d| d.code.clone()).collect::<Vec<_>>());
                continue;
            }
        };
        let g = &run.graph;
        let (w, l) = prog.window.unwrap();
        let mid = Some(w / 2);
        let mut kernels = vec![];
        for (gi, grp) in g.groups.iter().enumerate() {
            if grp.iteration != mid {
                continue;
            }
            let mut nodes: Vec<String> = grp.ops.iter().map(|&o| {
                let n = &prog.nodes[prog.ops[g.ops[o as usize].op].node];
                format!("{}[{}]", n.op_name, n.role.as_deref().unwrap_or("-"))
            }).collect();
            nodes.dedup();
            if grp.tasks.1 > grp.tasks.0 {
                kernels.push(format!("{} <- {}", grp.label, nodes.join("+")));
            }
            for k in g.stack.iter().filter(|k| k.group as usize == gi) {
                kernels.push(format!("  +{} ({:.0} B{})", k.name, k.bytes, if k.onchip { ", on-chip" } else { "" }));
            }
        }
        let (a, b) = kiln_sim::stack::kernel_counts(g, mid);
        let (pa, pb) = kiln_sim::stack::kernel_counts(g, None);
        println!("== {name} L={} {phase}: per layer {} (groups {a} + stack {b}); outside layers {} ; step total {}; makespan {:.6} s",
                 cfg.n_layers, a + b, pa + pb, (pa + pb) as u64 + l * (a + b) as u64, run.central.makespan_s);
        for k in &kernels {
            println!("   {k}");
        }
        let mut by: BTreeMap<String, usize> = BTreeMap::new();
        for k in g.stack.iter().filter(|k| g.groups[k.group as usize].iteration == mid) {
            *by.entry(k.name.clone()).or_default() += 1;
        }
        out.insert(phase.into(), serde_json::json!({"per_layer": a + b, "groups": a, "stack_extra": b, "outside_layers": pa + pb,
            "step_total": (pa + pb) as u64 + l * (a + b) as u64, "layers": l, "kernels": kernels, "makespan_s": run.central.makespan_s}));
    }
    if let Ok(path) = std::env::var("OOS_JSON") {
        std::fs::write(path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    }
}
