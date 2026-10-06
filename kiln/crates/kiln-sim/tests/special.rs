//! Special-function units (01 §5.4): transcendentals a design's special units implement run there, beside the
//! vector work; a special unit with nothing to do changes nothing.

mod common;

use std::path::PathBuf;

use common::*;
use kiln_ir::hw::{Design, Profile};
use kiln_ir::wl::{CacheState, EvalMode, PhaseKind, Scenario, SeqBatch};
use kiln_sim::Prepared;
use serde_json::Value;

/// TPU v5e with its EUP edited by `f` (`None` removes it).
fn v5e(f: impl Fn(&mut Value) -> bool) -> Prepared {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/tpu_v5e.json5");
    let mut v = kiln_ir::hw::load_file(&p).unwrap().canonical;
    assert!(edit(&mut v, &f), "v5e declares an eup");
    Prepared::load(Design::from_value(v).unwrap(), Profile::Full).unwrap()
}

fn edit(v: &mut Value, f: &impl Fn(&mut Value) -> bool) -> bool {
    match v {
        Value::Array(xs) => {
            if let Some(i) = xs.iter().position(|u| u["id"] == "eup" && u["kind"] == "special") {
                if !f(&mut xs[i]) {
                    xs.remove(i);
                }
                return true;
            }
            xs.iter_mut().any(|x| edit(x, f))
        }
        Value::Object(m) => m.values_mut().any(|x| edit(x, f)),
        _ => false,
    }
}

fn makespan(p: &Prepared, model: &kiln_ir::wl::Model, sc: &Scenario) -> f64 {
    run(p, &program(model, sc, 3), &quick()).central.makespan_s
}

#[test]
fn transcendentals_run_on_special_units() {
    let none = v5e(|_| false);
    let eup = v5e(|_| true);
    // Same unit, functions this workload never uses (Llama has no erf).
    let idle = v5e(|u| {
        u["functions"] = serde_json::json!(["erf"]);
        true
    });
    let (model, prefill) = tiny(false, 4);
    let (t_none, t_eup, t_idle) = (makespan(&none, &model, &prefill), makespan(&eup, &model, &prefill), makespan(&idle, &model, &prefill));
    assert_eq!(t_idle, t_none, "an idle special unit changes nothing");
    assert!(t_eup < t_none * 0.99, "softmax, SiLU and rsqrt use the EUP: {t_eup} vs {t_none}");

    // No transcendentals at all: an isolated GEMM is identical with or without the EUP.
    let gemm = kiln_wl::zoo::gemm_model(1024, 1024, 1024);
    let mut sc = Scenario::snapshot(PhaseKind::Custom, SeqBatch::uniform(1024, 1, 1));
    sc.eval_mode = EvalMode::Isolated { cache: CacheState::Cold, launch: false };
    let (_, lg, _) = kiln_wl::evaluate_snapshot(&gemm, &sc).unwrap();
    let prog = kiln_map::Program::isolated(&lg).unwrap();
    let t = |p: &Prepared| run(p, &prog, &quick()).central.makespan_s;
    assert_eq!(t(&eup), t(&none));
}

#[test]
fn softmax_splits_between_the_vpu_and_the_eup() {
    let none = v5e(|_| false);
    let eup = v5e(|_| true);
    let (model, prefill) = tiny(false, 4);
    let prog = program(&model, &prefill, 3);
    let opts = kiln_sim::SimOptions { trace: kiln_trace::TraceLevel::Ops, ..quick() };
    let (a, b) = (run(&none, &prog, &opts), run(&eup, &prog, &opts));
    let busy = |r: &kiln_sim::PhaseRun, unit: &str| -> f64 {
        r.central.resources.iter().filter(|x| x.resource.as_str().ends_with(unit)).map(|x| x.busy_s).sum()
    };
    assert_eq!(busy(&a, "eup"), 0.0);
    assert!(busy(&b, "eup") > 0.0, "the EUP takes transcendental work");
    assert!(busy(&b, "vpu") < busy(&a, "vpu"), "the VPU sheds it: {} vs {}", busy(&b, "vpu"), busy(&a, "vpu"));
    // v5e's VPU emulates exp at 1/8 of 1024 lanes, as fast as the 128-lane EUP: the two split it evenly.
    let exps: Vec<&str> = prog.ops.iter().filter(|o| o.id.contains(".attn.") && o.body().exp > 0).map(|o| o.id.as_str()).collect();
    assert!(!exps.is_empty());
    let exp = |r: &kiln_sim::PhaseRun| r.central.ops.iter().filter(|o| exps.contains(&o.op.as_str())).map(|o| o.time_s()).sum::<f64>();
    assert!(exp(&b) < 0.75 * exp(&a), "{} vs {}", exp(&b), exp(&a));
}
