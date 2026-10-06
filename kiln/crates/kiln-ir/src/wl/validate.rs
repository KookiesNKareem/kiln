//! Structural validation of a workload (02 §7.2, §15). Shape inference (`E-WL-SHAPE-001`) and binding errors
//! (`E-WL-SYM-*`, `E-WL-DIM-001`) need bound values and live in `kiln-wl::bind`.

use std::collections::{BTreeMap, BTreeSet};

use super::doc::{Model, ModelSrc, SymKind, WorkloadDoc};
use super::graph::Graph;
use super::op::Op;
use super::scenario::ScenarioMode;
use super::tensor::{TensorClass, TensorDecl};
use crate::common::{Diagnostic, Id};

pub fn validate_doc(doc: &WorkloadDoc) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if let ModelSrc::Full(m) = &doc.model {
        out.extend(validate_model(m));
    }
    for (id, s) in &doc.scenarios {
        let path = format!("scenarios.{id}");
        if let ScenarioMode::Snapshot { seqs, .. } = &s.mode
            && let Err(e) = seqs.check()
        {
            out.push(Diagnostic {
                path: Some(format!(
                    "{path}.mode.snapshot.seqs.{}",
                    e.path.clone().unwrap_or_default()
                )),
                ..e
            });
        }
        if let ScenarioMode::Static {
            batch,
            prompt_len,
            gen_len,
            prefix_cached,
            ..
        } = s.mode
            && (batch == 0 || prompt_len == 0 || gen_len == 0 || prefix_cached >= prompt_len)
        {
            out.push(
                Diagnostic::error("E-WL-SCN-001", "static scenario needs batch, prompt_len, gen_len ≥ 1 and prefix_cached < prompt_len")
                    .at(path),
            );
        }
    }
    out
}

pub fn validate_model(m: &Model) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let check_decl = |path: String, t: &TensorDecl, out: &mut Vec<Diagnostic>| {
        if let Err(e) = t.dtype.check() {
            out.push(e.at(path.clone()));
        }
        let (mut sizes, mut segs) = (Vec::new(), Vec::new());
        for e in t.shape.iter().chain(t.stack.iter()) {
            e.size_symbols(&mut sizes);
            e.segment_symbols(&mut segs);
        }
        let refs = sizes
            .into_iter()
            .map(|s| (s, SymKind::Size))
            .chain(segs.into_iter().map(|s| (s, SymKind::Segments)));
        for (s, want) in refs {
            let msg = match m.symbols.get(&s) {
                None => format!("tensor uses undeclared symbol {s:?}"),
                Some(d) if d.kind != want => {
                    format!("symbol {s:?} is declared {:?} but used as {want:?}", d.kind)
                }
                Some(_) => continue,
            };
            out.push(
                Diagnostic::error("E-WL-SYM-001", msg)
                    .at(path.clone())
                    .hint("declare size symbols under model.symbols; segments symbols appear only as sum(seqs.f)/max(seqs.f)"),
            );
        }
    };
    for (id, t) in &m.tensors {
        let path = format!("tensors.{id}");
        check_decl(path.clone(), t, &mut out);
        if !t.class.is_model_level() {
            out.push(
                Diagnostic::error(
                    "E-WL-CLS-001",
                    format!("model-level tensor has class {:?}", t.class),
                )
                .at(path)
                .hint("only weight, kv_cache and constant tensors are declared at model level"),
            );
        }
    }
    if !m.graphs.contains_key(&m.entry.forward) {
        out.push(
            Diagnostic::error(
                "E-WL-REF-001",
                format!("entry graph {:?} not found", m.entry.forward.as_str()),
            )
            .at("entry.forward"),
        );
    }
    for (gid, g) in &m.graphs {
        out.extend(validate_graph(m, gid, g, &check_decl));
    }
    out
}

fn validate_graph(
    m: &Model,
    gid: &Id,
    g: &Graph,
    check_decl: &dyn Fn(String, &TensorDecl, &mut Vec<Diagnostic>),
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let gp = format!("graphs.{gid}");
    for (id, t) in &g.tensors {
        check_decl(format!("{gp}.tensors.{id}"), t, &mut out);
        if matches!(t.class, TensorClass::Weight | TensorClass::Constant) {
            out.push(
                Diagnostic::error(
                    "E-WL-CLS-001",
                    format!("{:?} tensor declared inside a graph", t.class),
                )
                .at(format!("{gp}.tensors.{id}"))
                .hint("declare weights and constants at model level and pass them as params"),
            );
        }
    }
    let known =
        |t: &Id| g.tensors.contains_key(t) || g.params.contains(t) || m.tensors.contains_key(t);
    let decl = |t: &Id| g.tensors.get(t).or_else(|| m.tensors.get(t));
    let mut ids = BTreeSet::new();
    let mut producers: BTreeMap<&Id, Vec<&Id>> = BTreeMap::new();
    for n in &g.nodes {
        let np = format!("{gp}.nodes.{}", n.id);
        if !ids.insert(&n.id) {
            out.push(
                Diagnostic::error(
                    "E-WL-ID-001",
                    format!("duplicate node id {:?}", n.id.as_str()),
                )
                .at(np.clone()),
            );
        }
        for t in n.inputs.iter().chain(&n.outputs) {
            if !known(t) {
                out.push(
                    Diagnostic::error("E-WL-REF-001", format!("tensor {:?} is not declared", t.as_str()))
                        .at(np.clone())
                        .hint("declare it in the graph's tensors, list it in params, or declare it at model level"),
                );
            }
        }
        for t in &n.outputs {
            producers.entry(t).or_default().push(&n.id);
            if let Some(d) = decl(t)
                && matches!(
                    d.class,
                    TensorClass::Weight | TensorClass::Constant | TensorClass::Input
                )
            {
                out.push(
                    Diagnostic::error(
                        "E-WL-CLS-001",
                        format!("node writes {:?} tensor {:?}", d.class, t.as_str()),
                    )
                    .at(np.clone()),
                );
            }
        }
        match &n.op {
            Op::Repeat(r) => {
                match m.graphs.get(&r.body) {
                    None => out.push(
                        Diagnostic::error(
                            "E-WL-REF-001",
                            format!("repeat body {:?} not found", r.body.as_str()),
                        )
                        .at(np.clone()),
                    ),
                    Some(body) => {
                        for s in &r.stacked {
                            if m.tensors.get(&s.outer).is_none_or(|t| t.stack.is_none()) {
                                out.push(
                                    Diagnostic::error(
                                        "E-WL-REF-001",
                                        format!(
                                            "stacked outer {:?} is not a model tensor with `stack`",
                                            s.outer.as_str()
                                        ),
                                    )
                                    .at(np.clone()),
                                );
                            }
                            if !body.params.contains(&s.param) {
                                out.push(
                                    Diagnostic::error(
                                        "E-WL-REF-001",
                                        format!(
                                            "stacked param {:?} is not a body param",
                                            s.param.as_str()
                                        ),
                                    )
                                    .at(np.clone()),
                                );
                            }
                        }
                    }
                }
                if r.body == *gid {
                    out.push(
                        Diagnostic::error("E-WL-DAG-001", "repeat body is its own graph")
                            .at(np.clone()),
                    );
                }
            }
            Op::Call(c) if !m.graphs.contains_key(&c.graph) => {
                out.push(
                    Diagnostic::error(
                        "E-WL-REF-001",
                        format!("called graph {:?} not found", c.graph.as_str()),
                    )
                    .at(np.clone()),
                );
            }
            _ => {}
        }
    }
    for (t, ps) in &producers {
        if ps.len() > 1 {
            let names: Vec<&str> = ps.iter().map(|p| p.as_str()).collect();
            out.push(
                Diagnostic::error(
                    "E-WL-CLS-001",
                    format!(
                        "tensor {:?} has {} producers {names:?}",
                        t.as_str(),
                        ps.len()
                    ),
                )
                .at(format!("{gp}.tensors.{t}"))
                .hint("SSA: write a new version with alias_of for in-place updates"),
            );
        }
    }
    for (id, t) in &g.tensors {
        if t.class == TensorClass::Activation
            && !producers.contains_key(id)
            && !g.params.contains(id)
        {
            out.push(
                Diagnostic::error(
                    "E-WL-CLS-001",
                    format!("activation {:?} has no producer", id.as_str()),
                )
                .at(format!("{gp}.tensors.{id}")),
            );
        }
    }
    for r in &g.results {
        if !producers.contains_key(r) && !g.params.contains(r) {
            out.push(
                Diagnostic::error(
                    "E-WL-REF-001",
                    format!("result {:?} is never produced", r.as_str()),
                )
                .at(format!("{gp}.results")),
            );
        }
    }
    let mut alias_users: BTreeMap<&Id, &Id> = BTreeMap::new();
    for (id, t) in &g.tensors {
        let Some(base) = &t.alias_of else { continue };
        let ap = format!("{gp}.tensors.{id}");
        match decl(base).or_else(|| g.params.contains(base).then_some(t)) {
            None => out.push(
                Diagnostic::error(
                    "E-WL-ALIAS-001",
                    format!("alias_of {:?} is not declared", base.as_str()),
                )
                .at(ap.clone()),
            ),
            Some(b) if b.class != t.class || b.shape != t.shape => out.push(
                Diagnostic::error(
                    "E-WL-ALIAS-001",
                    format!(
                        "alias {:?} differs from {:?} in class or shape",
                        id.as_str(),
                        base.as_str()
                    ),
                )
                .at(ap.clone()),
            ),
            Some(_) => {}
        }
        if let Some(prev) = alias_users.insert(base, id) {
            out.push(
                Diagnostic::error(
                    "E-WL-ALIAS-001",
                    format!(
                        "{:?} and {:?} are both versions of {:?}",
                        prev.as_str(),
                        id.as_str(),
                        base.as_str()
                    ),
                )
                .at(ap)
                .hint("alias chains must be linear: version the latest alias instead"),
            );
        }
    }
    if let Some(cycle) = find_cycle(g) {
        out.push(
            Diagnostic::error("E-WL-DAG-001", format!("cycle: {}", cycle.join(" -> "))).at(gp),
        );
    }
    out
}

fn find_cycle(g: &Graph) -> Option<Vec<String>> {
    let order = g.topo_order();
    let pos: Vec<usize> = (0..g.nodes.len())
        .map(|i| {
            order
                .iter()
                .position(|&o| o == i)
                .expect("all nodes ordered")
        })
        .collect();
    for (i, n) in g.nodes.iter().enumerate() {
        for t in &n.inputs {
            if let Some(p) = g.nodes.iter().position(|nd| nd.outputs.contains(t))
                && pos[p] >= pos[i]
            {
                return Some(vec![
                    g.nodes[p].id.to_string(),
                    t.to_string(),
                    n.id.to_string(),
                ]);
            }
        }
    }
    None
}
