//! Memory writes are priced at the memory's write energy, reads at its read energy.

mod common;

use std::path::PathBuf;

use common::*;
use kiln_ir::hw::{Design, Profile};
use kiln_sim::Prepared;
use serde_json::Value;

/// TPU v5e with its vmem's write energy overridden to `e` (None: derived).
fn v5e(e: Option<&str>) -> Prepared {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/tpu_v5e.json5");
    let mut v = kiln_ir::hw::load_file(&p).unwrap().canonical;
    fn edit(v: &mut Value, e: &str) -> bool {
        match v {
            Value::Array(xs) => xs.iter_mut().any(|x| edit(x, e)),
            Value::Object(m) if m.get("id") == Some(&Value::from("vmem")) => {
                m.insert("overrides".into(), serde_json::json!({ "write_energy": e, "source": "test" }));
                true
            }
            Value::Object(m) => m.values_mut().any(|x| edit(x, e)),
            _ => false,
        }
    }
    if let Some(e) = e {
        assert!(edit(&mut v, e), "v5e declares vmem");
    }
    Prepared::load(Design::from_value(v).unwrap(), Profile::Full).unwrap()
}

#[test]
fn writes_cost_write_energy() {
    let (model, sc) = tiny(false, 4);
    let prog = program(&model, &sc, 3);
    let mem = |p: &Prepared| -> (f64, f64) {
        let r = run(p, &prog, &quick()).central;
        let vmem: f64 = r.resources.iter().filter(|x| x.resource.as_str().ends_with("vmem")).map(|x| x.energy_j).sum();
        (vmem, r.energy.memory_j.values().sum())
    };
    let (base, cheap, dear) = (mem(&v5e(None)), mem(&v5e(Some("1pJ/B"))), mem(&v5e(Some("100pJ/B"))));
    assert_eq!(base, cheap, "an override below the derived energy keeps it");
    assert!(dear.0 > 2.0 * base.0, "vmem writes at 100 pJ/B: {} vs {} J", dear.0, base.0);
    assert!(dear.1 - base.1 > 0.99 * (dear.0 - base.0), "the memory energy total follows: {dear:?} vs {base:?}");
}

/// 08 §F: a software-stack kernel's writes land in `wbytes` (priced at write energy), not only in its byte count.
#[test]
fn stack_kernel_writes_are_writes() {
    let p = reference("a100_sxm4_40gb.json5");
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b1").unwrap();
    let opts = kiln_sim::SimOptions { interval: kiln_trace::IntervalMethod::None, shadow_prices: false, layer_scope_fallback: true, ..Default::default() };
    let (run, prog) = kiln_sim::simulate_member(&p, &m, &opts).unwrap();
    assert!(!run.graph.stack.is_empty());
    let params = run.params.at(kiln_trace::Corner::Central);
    let mid = prog.window.map(|(w, _)| w / 2);
    let totals = |g: &kiln_map::lower::TaskGraph| {
        let out = kiln_sim::engine::Engine { view: &p.view, g, params: &params, clocks: &run.clocks }.run(mid);
        let sum = |x: &[Vec<f64>; 3]| x.iter().flatten().sum::<f64>();
        (sum(&out.bytes), sum(&out.wbytes))
    };
    let mut bare = run.graph.clone();
    bare.stack.clear();
    let (with, without) = (totals(&run.graph), totals(&bare));
    let (db, dw) = (with.0 - without.0, with.1 - without.1);
    assert!(db > 0.0 && dw > 0.0 && dw < db, "stack kernels add {db} B of traffic, {dw} B of it written");
    let written: f64 = run.graph.stack.iter().map(|k| k.wbytes).sum();
    assert!((dw - written).abs() <= 1e-9 * written, "{dw} vs {written}");
}
