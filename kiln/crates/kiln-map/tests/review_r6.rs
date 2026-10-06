//! Regressions from the r6 cost/map review: units run only the op classes they declare.

use std::sync::Arc;

use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;
use kiln_map::heuristic::placement;
use kiln_map::lower::Lowerer;
use kiln_map::{HwView, KilnCost, NestCost, NestQuery, Pool, Program, UnitCostModel};

fn variant(name: &str, edits: &[(&str, &str)]) -> HwView {
    let mut src = std::fs::read_to_string(format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"))).expect("design");
    for (from, to) in edits {
        assert!(src.contains(from), "{from}");
        src = src.replace(from, to);
    }
    let dir = std::env::temp_dir().join(format!("kiln-map-r6-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = dir.join(format!("{}-{name}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::write(&p, src).expect("write");
    let r = check_file(&p, Profile::Reference);
    HwView::new(Arc::new(r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics)))).expect("view")
}

fn program(workload: &str) -> Program {
    let m = kiln_wl::zoo::workload(workload).expect("workload");
    let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
    Program::whole_step(m.model(), &lg, 1).expect("program")
}

fn cost_on(v: &HwView, prog: &Program, oi: usize, pool: Pool) -> Result<NestCost, kiln_ir::common::Diagnostic> {
    let op = &prog.ops[oi];
    let s = Lowerer::slices(op, &placement(op, 0, vec![], 1)).swap_remove(0);
    let u = &v.units[v.pool(pool)[0]];
    let q = NestQuery { prog, op, slice: &s, points: kiln_map::geom::slice_points(op, &s), hw: &v.hw, unit: u.unit, level_caps: &[], level_mems: &[], level_bw: &[], residency: &[], gang: u.members.len() as u32, quick: true, partial: false };
    KilnCost::new().cost(&q)
}

const V5E_VPU: &str = "{ id: \"vpu\", kind: \"vector\", lanes: 128, sublanes: 8,";

#[test]
fn vector_units_run_only_declared_op_classes() {
    let prog = program("llama3_8b:decode_b1");
    let add = prog.ops.iter().position(|o| o.class() == KernelClass::Map && o.body().vector() > 0 && o.body().transcendental() == 0).expect("an elementwise map");
    let exp = prog.ops.iter().position(|o| o.class() != KernelClass::Contraction && o.body().exp > 0 && o.body().log == 0).expect("an exp");
    let base = variant("tpu_v5e.json5", &[]);
    let cvt_only = variant("tpu_v5e.json5", &[(V5E_VPU, &format!("{V5E_VPU} ops: [\"convert\"],"))]);
    cost_on(&base, &prog, add, Pool::Vector).expect("the default vector op set holds elementwise");
    let e = cost_on(&cvt_only, &prog, add, Pool::Vector).expect_err("a convert-only unit runs no adds");
    assert_eq!(e.code, "E-MAP-OP-004", "{e:?}");
    // Without `transcendental` the exps all go to the EUP, never the VPU.
    let ops = "\"elementwise\", \"reduction\", \"convert\", \"permute\", \"gather_scatter\", \"scan\", \"sort_topk\"";
    let no_transc = variant("tpu_v5e.json5", &[(V5E_VPU, &format!("{V5E_VPU} ops: [{ops}],"))]);
    let c = cost_on(&no_transc, &prog, exp, Pool::Vector).expect("the EUP implements exp");
    let b = cost_on(&base, &prog, exp, Pool::Vector).expect("base");
    assert!(c.special_cycles[0] >= b.special_cycles[0], "{c:?} vs {b:?}");
}

#[test]
fn matrix_units_need_a_contraction_op_class() {
    let prog = program("llama3_8b:decode_b1");
    let mm = prog.ops.iter().position(|o| o.class() == KernelClass::Contraction).expect("a matmul");
    let v = variant("tpu_v5e.json5", &[("id: \"mxu\",", "id: \"mxu\", ops: [\"collective_reduce\"],")]);
    let e = cost_on(&v, &prog, mm, Pool::Mac).expect_err("no matmul class");
    assert_eq!(e.code, "E-MAP-OP-004", "{e:?}");
}
