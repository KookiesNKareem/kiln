//! `explain_run` (03 §10, 00 decision 7): the single producer of LLM-readable bottleneck text.

use std::fmt::Write;

use kiln_trace::sim::{Binding, BindingClass, SimResult};

fn pct(x: f64, of: f64) -> f64 {
    if of > 0.0 { 100.0 * x / of } else { 0.0 }
}

fn class_name(c: BindingClass) -> &'static str {
    match c {
        BindingClass::Compute => "compute",
        BindingClass::Dram => "off-chip memory bandwidth",
        BindingClass::Link => "interconnect links",
        BindingClass::Port => "on-chip memory ports",
        BindingClass::Dependency => "dependency chains",
        BindingClass::Overhead => "launch/sync overheads",
        BindingClass::Nmp => "near-memory units",
        BindingClass::Contention => "contention",
        BindingClass::PipelineBubble => "pipeline bubbles",
    }
}

fn binding_text(b: &Binding) -> String {
    match b {
        Binding::Overhead { overhead } => format!("overhead ({overhead:?})"),
        _ => match b.resource() {
            Some(r) => format!("{} on {}", class_name(b.class()), r.as_str()),
            None => class_name(b.class()).to_string(),
        },
    }
}

/// Deterministic summary of a result, or of one op when `op` is given; at most `max_items` list entries.
pub fn explain_run(r: &SimResult, op: Option<&str>, max_items: usize) -> String {
    let mut s = String::new();
    if let Some(id) = op {
        let Some(o) = r.op(id) else { return format!("op {id} is not in this result (trace level `ops` lists ops)") };
        let _ = write!(s, "{id}: {:.3} us, bound by {}", o.time_s() * 1e6, binding_text(&o.binding));
        if let Some(ru) = &o.runner_up {
            let _ = write!(s, "; runner-up {} at {:.0}% of it", binding_text(&ru.binding), 100.0 * ru.ratio);
        }
        if o.macs_issued > 0 {
            let _ = write!(s, "; MAC utilization of issued slots {:.0}%", pct(o.macs_useful as f64, o.macs_issued as f64));
        }
        for f in o.floors.iter().take(max_items) {
            let _ = write!(s, "; {:?} floor {:.3} us", f.kind, f.seconds * 1e6);
        }
        return s;
    }
    let t = r.makespan_s;
    let _ = write!(s, "{} ({:?}, tier {:?}, {:?} corner): {:.3} ms", r.phase, r.scope, r.tier, r.corner, t * 1e3);
    let _ = write!(s, "; floors A0 {:.3} ms, A2 {:.3} ms ({:.0}% of time)", r.t_a0_s * 1e3, r.t_a2_s * 1e3, pct(r.t_a2_s, t));
    let mut parts: Vec<_> = r.bottleneck.time_by_binding.iter().collect();
    parts.sort_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(b.0)));
    let shares: Vec<String> = parts.iter().take(max_items).map(|(c, x)| format!("{} {:.0}%", class_name(**c), pct(**x, t))).collect();
    let _ = write!(s, ". Time by limiter: {}", shares.join(", "));
    let tops: Vec<String> = r
        .bottleneck
        .top_resources
        .iter()
        .take(max_items)
        .map(|x| format!("{} {:.0}%{}", x.resource.as_str(), 100.0 * x.utilization, if x.shadow_price > 0.5 { " (binding)" } else { "" }))
        .collect();
    if !tops.is_empty() {
        let _ = write!(s, ". Busiest resources: {}", tops.join(", "));
    }
    let mut ops: Vec<_> = r.ops.iter().collect();
    ops.sort_by(|a, b| b.time_s().total_cmp(&a.time_s()).then(a.op.cmp(&b.op)));
    let slow: Vec<String> = ops.iter().take(max_items).map(|o| format!("{} {:.1} us ({})", o.op.as_str(), o.time_s() * 1e6, binding_text(&o.binding))).collect();
    if !slow.is_empty() {
        let _ = write!(s, ". Slowest ops: {}", slow.join("; "));
    }
    let fails: Vec<String> = r.invariants.failures().map(|c| format!("{:?} {}", c.id, c.message)).collect();
    if !fails.is_empty() {
        let _ = write!(s, ". INVARIANT FAILURES (simulator bug): {}", fails.join("; "));
    }
    s.push('.');
    s
}
