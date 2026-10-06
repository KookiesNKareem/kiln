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
