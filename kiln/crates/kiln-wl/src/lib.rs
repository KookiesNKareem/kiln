//! Workload transforms (spec 02): binding, lowering to affine kernels, counting, model zoo and suites,
//! scenario expansion, bench export and harness compatibility.

pub mod bench;
pub mod convert;
pub mod count;
pub mod expand;
pub mod graph;
pub mod legacy;
pub mod lower;
pub mod partition;
pub mod stack;
pub mod zoo;

use kiln_ir::common::Diagnostic;
use kiln_ir::wl::{Model, PhaseInstance, Scenario, TensorClass, validate_model};

pub use graph::{
    Binding, LowerOpts, LoweredGraph, LoweredNode, StepStats, bind, lower_graph, stats,
};
pub use lower::{Lowered, NodeCtx, cost_hint, lower};

/// Validates, binds and lowers the single phase instance of a snapshot scenario, with whole-step stats.
pub fn evaluate_snapshot(
    model: &Model,
    scenario: &Scenario,
) -> Result<(Binding, LoweredGraph, StepStats), Vec<Diagnostic>> {
    let diags: Vec<Diagnostic> = validate_model(model)
        .into_iter()
        .filter(|d| d.severity == kiln_ir::common::Severity::Error)
        .collect();
    if !diags.is_empty() {
        return Err(diags);
    }
    let inst = expand::expand(scenario, model).map_err(|e| vec![e])?;
    let [inst] = inst.as_slice() else {
        return Err(vec![Diagnostic::error(
            "E-WL-SCN-001",
            "expected a snapshot scenario (one phase instance)",
        )]);
    };
    evaluate_instance(model, scenario, inst)
}

/// Binds and lowers one phase instance of a validated model, with whole-step stats.
pub fn evaluate_instance(
    model: &Model,
    scenario: &Scenario,
    inst: &PhaseInstance,
) -> Result<(Binding, LoweredGraph, StepStats), Vec<Diagnostic>> {
    let b = bind(model, &inst.seqs, &inst.bindings)?;
    let opts = LowerOpts {
        routing: scenario.routing.as_ref(),
        group_size: None,
    };
    let lg = lower_graph(model, &b, &opts).map_err(|e| vec![e])?;
    let st = stats(model, &b, &lg).map_err(|e| vec![e])?;
    Ok((b, lg, st))
}

/// Parameter count: elements of every weight tensor, times its stack depth, under a binding.
pub fn param_count(model: &Model, b: &Binding) -> Result<u64, Diagnostic> {
    model
        .tensors
        .iter()
        .filter(|(_, t)| t.class == TensorClass::Weight)
        .try_fold(0u64, |acc, (id, t)| {
            let ti = graph::bind_type(t, b, id.as_str())?;
            let stack = t
                .stack
                .as_ref()
                .map_or(Ok(1), |e| graph::eval_dim(e, b, id.as_str()))?;
            Ok(acc + ti.numel() as u64 * stack)
        })
}
