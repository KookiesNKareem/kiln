//! Regressions from the round-2 review of kiln-wl lowering.

use kiln_ir::wl::*;
use kiln_wl::lower::MoeRows;
use kiln_wl::zoo::{self, preset};
use kiln_wl::{NodeCtx, evaluate_snapshot, lower};

const BF: ElemType = ElemType::BF16;

fn t(shape: &[u64], class: TensorClass) -> TypeInfo {
    TypeInfo::new(shape.to_vec(), BF, class)
}

fn grouped(
    eq: &str,
    inputs: Vec<TypeInfo>,
    output: TypeInfo,
    moe_rows: Option<MoeRows>,
) -> Result<kiln_wl::Lowered, kiln_ir::common::Diagnostic> {
    let ins: Vec<String> = (0..inputs.len()).map(|i| format!("in{i}")).collect();
    let ins: Vec<&str> = ins.iter().map(String::as_str).collect();
    let node = Node::new(
        "ge",
        Op::GroupedEinsum(EinsumAttrs {
            eq: eq.into(),
            accum: None,
        }),
        &ins,
        &["out"],
    );
    let seqs = SeqBatch::uniform(1, 1, 1);
    lower(&NodeCtx {
        node: &node,
        inputs,
        outputs: vec![output],
        seqs: &seqs,
        moe_rows,
        group_size: None,
    })
}

#[test]
fn repeat_count_must_equal_stack_depth() {
    let mut m = zoo::build_model(&preset("llama3_8b").unwrap()).unwrap();
    let s = zoo::scenario("decode_b1").unwrap();
    assert!(evaluate_snapshot(&m, &s).is_ok());
    let w = m
        .tensors
        .values_mut()
        .find(|t| t.class == TensorClass::Weight && t.stack.is_some())
        .unwrap();
    w.stack = Some(1u64.into());
    let e = evaluate_snapshot(&m, &s).unwrap_err();
    assert_eq!(e[0].code, "E-WL-SHAPE-001", "{e:?}");
}

#[test]
fn no_drop_grouped_einsum_rejects_loads_above_capacity() {
    let act = TensorClass::Activation;
    let ins = || vec![t(&[2, 1, 8], act), t(&[2, 8, 8], TensorClass::Weight)];
    let rows = |drop_overflow| {
        Some(MoeRows {
            rows: vec![2, 2],
            drop_overflow,
        })
    };
    let e = grouped("ecd,end->ecn", ins(), t(&[2, 1, 8], act), rows(false)).unwrap_err();
    assert_eq!(e.code, "E-WL-SHAPE-001");
    assert!(grouped("ecd,end->ecn", ins(), t(&[2, 1, 8], act), rows(true)).is_ok());
}

#[test]
fn malformed_grouped_equations_are_diagnosed() {
    let act = TensorClass::Activation;
    for (eq, out) in [("ek,ek->", t(&[], act)), ("ek->ek", t(&[2, 8], act))] {
        let e = grouped(eq, vec![t(&[2, 8], act), t(&[2, 8], act)], out, None).unwrap_err();
        assert!(e.code.starts_with("E-WL-"), "{eq}: {e:?}");
    }
}
