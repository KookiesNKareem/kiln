//! Software-stack recipes on a lowered task graph (08 §F): each node's non-primary kernels are charged at
//! the group holding the node's first modelled op, with their traffic served on chip when small enough.

use kiln_ir::common::Diagnostic;
use kiln_ir::wl::TypeInfo;
use kiln_map::HwView;
use kiln_map::lower::TaskGraph;
use kiln_map::program::{Program, ProgramKind};
use kiln_wl::stack::{GroupKernel, NodeShape, Stack};

/// Replaces `g.stack` with the recipe's extra kernels. Isolated programs keep every intermediate off chip.
pub fn attach(view: &HwView, prog: &Program, g: &mut TaskGraph, stack: &Stack) -> Result<(), Diagnostic> {
    let onchip_cap = match prog.kind {
        ProgramKind::WholeStep => view.shared_onchip().map(|s| view.groups[s].capacity as f64 * stack.onchip_fraction),
        ProgramKind::Isolated => None,
    };
    let mut done = vec![false; prog.nodes.len()];
    let mut out = vec![];
    for (gi, grp) in g.groups.iter().enumerate() {
        for &o in &grp.ops {
            let node = prog.ops[g.ops[o as usize].op].node;
            if std::mem::replace(&mut done[node], true) {
                continue;
            }
            let n = &prog.nodes[node];
            let info = |ts: &[usize]| -> Vec<TypeInfo> { ts.iter().map(|&t| prog.tensors[t].info()).collect() };
            let (inputs, outputs) = (info(&n.inputs), info(&n.outputs));
            let shape = NodeShape { op: &n.op_name, role: n.role.as_deref(), inputs: &inputs, outputs: &outputs };
            for k in stack.extra_kernels(&shape).map_err(|e| e.at(n.path.clone()))? {
                let onchip = onchip_cap.is_some_and(|c| k.footprint_b <= c);
                out.push(GroupKernel { group: gi as u32, bytes: k.bytes(), name: k.name, kind: k.kind, onchip });
            }
        }
    }
    g.stack = out;
    Ok(())
}

/// Kernels one iteration (`None`: prologue and epilogue) issues: groups with modelled work, and stack extras.
pub fn kernel_counts(g: &TaskGraph, iteration: Option<u32>) -> (usize, usize) {
    let groups = g.groups.iter().filter(|x| x.iteration == iteration && x.tasks.1 > x.tasks.0).count();
    let extra = g.stack.iter().filter(|k| g.groups[k.group as usize].iteration == iteration).count();
    (groups, extra)
}
