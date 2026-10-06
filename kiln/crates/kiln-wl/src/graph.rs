//! Binding (02 §2.3), whole-graph lowering with `repeat`/`call`, and step statistics (02 §9.7 `stats`).

use indexmap::IndexMap;
use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::wl::*;

use crate::count::{NodeCost, Traffic, node_cost};
use crate::lower::{self, Lowered, MoeRows, NodeCtx};

/// Every symbol resolved to an integer for one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub seqs: SeqBatch,
    pub values: IndexMap<String, u64>,
}

impl DimEnv for Binding {
    fn sym(&self, name: &str) -> Option<u64> {
        self.values.get(name).copied()
    }

    fn seg(&self, _sym: &str, agg: SegAgg, field: SegField) -> Option<u64> {
        let v = self.seqs.segments.iter().map(|s| match field {
            SegField::Count => s.count,
            SegField::QLen => s.q_len,
            SegField::KvLen => s.kv_len,
            SegField::CountXQ => s.count * s.q_len,
            SegField::CountXKv => s.count * s.kv_len,
        });
        Some(match agg {
            SegAgg::Sum => v.sum(),
            SegAgg::Max => v.max().unwrap_or(0),
        })
    }
}

/// Binds symbols in priority order: explicit bindings, values derived from the sequence batch (`T`, `N`,
/// defaults for `kv_cap` and `slots`), then symbol defaults. `T` and `N` are always derived.
pub fn bind(
    model: &Model,
    seqs: &SeqBatch,
    explicit: &IndexMap<String, u64>,
) -> Result<Binding, Vec<Diagnostic>> {
    seqs.check().map_err(|e| vec![e])?;
    let mut b = Binding {
        seqs: seqs.canonical(),
        values: explicit.clone(),
    };
    b.values.insert("T".into(), seqs.tokens());
    b.values.insert("N".into(), seqs.seqs());
    b.values.entry("kv_cap".into()).or_insert(seqs.max_kv());
    b.values.entry("slots".into()).or_insert(seqs.seqs());
    let mut errs = Vec::new();
    for _ in 0..model.symbols.len() {
        for (name, d) in &model.symbols {
            if d.kind == SymKind::Size
                && !b.values.contains_key(name)
                && let Some(v) = d
                    .default
                    .as_ref()
                    .and_then(|e| e.eval(&b).ok())
                    .and_then(|r| r.to_u64())
            {
                b.values.insert(name.clone(), v);
            }
        }
    }
    for (name, d) in &model.symbols {
        let Some(&v) = b.values.get(name) else {
            continue;
        };
        let bad = d.min.is_some_and(|m| v < m)
            || d.max.is_some_and(|m| v > m)
            || d.divisible_by.is_some_and(|k| k == 0 || v % k != 0);
        if bad {
            errs.push(
                Diagnostic::error(
                    "E-WL-SYM-002",
                    format!(
                        "symbol {name} = {v} violates min {:?} / max {:?} / divisible_by {:?}",
                        d.min, d.max, d.divisible_by
                    ),
                )
                .at(format!("symbols.{name}")),
            );
        }
    }
    let decls = model
        .tensors
        .iter()
        .map(|(id, t)| (format!("tensors.{id}"), t))
        .chain(model.graphs.iter().flat_map(|(g, gr)| {
            gr.tensors
                .iter()
                .map(move |(id, t)| (format!("graphs.{g}.tensors.{id}"), t))
        }));
    let mut unbound: IndexMap<String, Vec<String>> = IndexMap::new();
    for (path, t) in decls {
        let mut syms = Vec::new();
        t.shape
            .iter()
            .chain(&t.stack)
            .for_each(|e| e.size_symbols(&mut syms));
        for s in syms.into_iter().filter(|s| !b.values.contains_key(s)) {
            unbound.entry(s).or_default().push(path.clone());
        }
    }
    for (s, users) in unbound {
        errs.push(
            Diagnostic::error(
                "E-WL-SYM-001",
                format!("symbol {s} is unbound; used by {}", users.join(", ")),
            )
            .hint("bind it in the scenario's `bindings` or give the symbol a default"),
        );
    }
    if errs.is_empty() { Ok(b) } else { Err(errs) }
}

pub fn eval_dim(e: &DimExpr, b: &Binding, what: &str) -> Result<u64, Diagnostic> {
    let r = e.eval(b).map_err(|err| match err {
        EvalError::Unbound(s) => {
            Diagnostic::error("E-WL-SYM-001", format!("symbol {s} is unbound in {what}"))
        }
        EvalError::DivByZero => {
            Diagnostic::error("E-WL-DIM-001", format!("division by zero in {what}: {e}"))
        }
        EvalError::Overflow => Diagnostic::error("E-WL-DIM-001", format!("{what}: {e} overflows")),
    })?;
    r.to_u64().ok_or_else(|| {
        Diagnostic::error(
            "E-WL-DIM-001",
            format!("{what}: {e} = {r} is not a non-negative integer"),
        )
        .hint(format!("symbol values: {:?}", b.values))
    })
}

pub fn bind_type(decl: &TensorDecl, b: &Binding, what: &str) -> Result<TypeInfo, Diagnostic> {
    let shape = decl
        .shape
        .iter()
        .map(|e| eval_dim(e, b, what))
        .collect::<Result<Vec<_>, _>>()?;
    let ti = TypeInfo::new(shape, decl.dtype, decl.class);
    if ti.checked_footprint().is_none() {
        return Err(Diagnostic::error(
            "E-WL-DIM-001",
            format!("{what}: shape {:?} overflows the element or byte count", ti.shape),
        ));
    }
    Ok(ti)
}

/// Where a tensor visible in some scope ultimately lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    /// Model-level tensor id, or the id in the outermost graph that declares it.
    pub root: Id,
    pub class: TensorClass,
    /// A per-iteration slice of a stacked tensor (distinct data each iteration).
    pub stacked: bool,
    /// The declaration's `upcast_ok` (02 §3): may a contraction widen it into a wider mode.
    pub upcast_ok: bool,
}

#[derive(Clone, Debug)]
pub struct LoweredNode {
    /// Dotted path: enclosing repeat/call node ids, then this node's id.
    pub path: String,
    pub node: Node,
    /// Product of enclosing repeat counts.
    pub multiplicity: u64,
    pub inputs: Vec<TypeInfo>,
    pub outputs: Vec<TypeInfo>,
    pub origins: Vec<(Id, Origin)>,
    pub lowered: Lowered,
    pub cost: CostHint,
    pub hint: CostHint,
    pub traffic: Vec<Traffic>,
}

#[derive(Clone, Debug, Default)]
pub struct LoweredGraph {
    pub nodes: Vec<LoweredNode>,
    /// Active sequences (`N`) of the binding the graph was lowered for.
    pub seqs: Option<u64>,
}

#[derive(Clone, Debug, Default)]
struct Scope {
    vars: IndexMap<Id, (TypeInfo, Origin)>,
    /// Results of a called graph, bound to the caller's outputs: their data lives where the caller's does.
    results: IndexMap<Id, (TypeInfo, Origin)>,
}

impl Scope {
    fn get(&self, t: &Id, path: &str) -> Result<&(TypeInfo, Origin), Diagnostic> {
        self.vars.get(t).ok_or_else(|| {
            Diagnostic::error("E-WL-REF-001", format!("tensor {t} is not visible"))
                .at(path.to_string())
        })
    }
}

pub struct LowerOpts<'a> {
    pub routing: Option<&'a RoutingModel>,
    pub group_size: Option<u64>,
}

/// Lowers the entry graph for one binding. Repeat bodies are lowered once with their multiplicity.
pub fn lower_graph(
    model: &Model,
    b: &Binding,
    opts: &LowerOpts,
) -> Result<LoweredGraph, Diagnostic> {
    let mut vars = IndexMap::new();
    for (id, t) in &model.tensors {
        let ti = bind_type(t, b, &format!("tensors.{id}"))?;
        vars.insert(
            id.clone(),
            (
                ti,
                Origin {
                    root: id.clone(),
                    class: t.class,
                    stacked: false,
                    upcast_ok: t.upcast_ok,
                },
            ),
        );
    }
    let entry = &model.entry.forward;
    let g = model.graphs.get(entry).ok_or_else(|| {
        Diagnostic::error("E-WL-REF-001", format!("entry graph {entry} not found"))
    })?;
    for p in &g.params {
        if !g.tensors.contains_key(p) && !vars.contains_key(p) {
            return Err(Diagnostic::error(
                "E-WL-REF-001",
                format!("entry param {p} is not declared"),
            )
            .at(format!("graphs.{entry}")));
        }
    }
    let mut w = Walker {
        model,
        b,
        opts,
        out: LoweredGraph {
            seqs: Some(b.seqs.seqs()),
            ..LoweredGraph::default()
        },
    };
    w.walk(g, Scope { vars, ..Scope::default() }, "", 1)?;
    Ok(w.out)
}

struct Walker<'a> {
    model: &'a Model,
    b: &'a Binding,
    opts: &'a LowerOpts<'a>,
    out: LoweredGraph,
}

impl Walker<'_> {
    fn walk(
        &mut self,
        g: &Graph,
        mut scope: Scope,
        prefix: &str,
        mult: u64,
    ) -> Result<(), Diagnostic> {
        let (model, b) = (self.model, self.b);
        for (id, t) in &g.tensors {
            let ti = bind_type(t, b, &format!("{prefix}tensors.{id}"))?;
            if let Some((caller, o)) = scope.results.get(id) {
                if (&caller.shape, caller.dtype) != (&ti.shape, ti.dtype) {
                    return Err(Diagnostic::error(
                        "E-WL-SHAPE-001",
                        format!(
                            "result {id} is {:?} {:?}, the caller's output is {:?} {:?}",
                            ti.dtype, ti.shape, caller.dtype, caller.shape
                        ),
                    )
                    .at(format!("{prefix}tensors.{id}")));
                }
                let o = o.clone();
                scope.vars.insert(id.clone(), (ti, o));
                continue;
            }
            if g.params.contains(id)
                && let Some((bound, _)) = scope.vars.get(id)
            {
                if (&bound.shape, bound.dtype) != (&ti.shape, ti.dtype) {
                    return Err(Diagnostic::error(
                        "E-WL-SHAPE-001",
                        format!(
                            "param {id} is declared {:?} {:?}, the caller passes {:?} {:?}",
                            ti.dtype, ti.shape, bound.dtype, bound.shape
                        ),
                    )
                    .at(format!("{prefix}tensors.{id}")));
                }
                continue;
            }
            let origin = match t.alias_of.as_ref().and_then(|a| scope.vars.get(a)) {
                Some((_, o)) => Origin { upcast_ok: o.upcast_ok && t.upcast_ok, ..o.clone() },
                None => Origin {
                    root: id.clone(),
                    class: t.class,
                    stacked: false,
                    upcast_ok: t.upcast_ok,
                },
            };
            scope.vars.insert(id.clone(), (ti, origin));
        }
        for i in g.topo_order() {
            let node = &g.nodes[i];
            let path = format!("{prefix}{}", node.id);
            match &node.op {
                Op::Repeat(r) => {
                    let count = eval_dim(&r.count, b, &format!("{path}.count"))?;
                    let body = model.graphs.get(&r.body).ok_or_else(|| {
                        Diagnostic::error(
                            "E-WL-REF-001",
                            format!("repeat body {} not found", r.body),
                        )
                        .at(path.clone())
                    })?;
                    let mut inner = IndexMap::new();
                    for s in &r.stacked {
                        let (ti, o) = scope.get(&s.outer, &path)?.clone();
                        if let Some(depth) = model
                            .tensors
                            .get(&o.root)
                            .and_then(|t| t.stack.as_ref())
                            .filter(|_| !o.stacked)
                        {
                            let depth = eval_dim(depth, b, &format!("tensors.{}.stack", o.root))?;
                            if depth != count {
                                return Err(Diagnostic::error(
                                    "E-WL-SHAPE-001",
                                    format!(
                                        "stacked {} has stack {depth}, repeat runs {count} iterations",
                                        s.outer
                                    ),
                                )
                                .at(path.clone())
                                .hint("a stacked tensor holds one slice per iteration (02 §7.3)"));
                            }
                        }
                        inner.insert(s.param.clone(), (ti, Origin { stacked: true, ..o }));
                    }
                    for c in &r.carry {
                        inner.insert(c.param.clone(), scope.get(&c.init, &path)?.clone());
                    }
                    for (outer, param) in &r.broadcast {
                        inner.insert(param.clone(), scope.get(outer, &path)?.clone());
                    }
                    let mut results = IndexMap::new();
                    for c in &r.carry {
                        let out = scope.get(&c.out, &path)?;
                        let same = |what: &str, t: &TypeInfo| -> Result<(), Diagnostic> {
                            if (&t.shape, t.dtype) == (&out.0.shape, out.0.dtype) {
                                return Ok(());
                            }
                            Err(Diagnostic::error(
                                "E-WL-SHAPE-001",
                                format!(
                                    "carry {what} is {:?} {:?}, its out {} is {:?} {:?}",
                                    t.dtype, t.shape, c.out, out.0.dtype, out.0.shape
                                ),
                            )
                            .at(path.clone()))
                        };
                        same("init", &scope.get(&c.init, &path)?.0)?;
                        match inner.get(&c.yield_) {
                            Some((y, _)) if body.params.contains(&c.yield_) => same("yield", y)?,
                            _ if body.tensors.contains_key(&c.yield_) => {
                                results.insert(c.yield_.clone(), out.clone());
                            }
                            _ => {
                                return Err(Diagnostic::error(
                                    "E-WL-REF-001",
                                    format!("carry yield {} is not a tensor of body {}", c.yield_, r.body),
                                )
                                .at(path.clone()));
                            }
                        }
                    }
                    let mult = mult.checked_mul(count).ok_or_else(|| {
                        Diagnostic::error(
                            "E-WL-DIM-001",
                            format!("repeat multiplicity {mult} x {count} overflows u64"),
                        )
                        .at(path.clone())
                    })?;
                    self.walk(
                        body,
                        Scope { vars: inner, results },
                        &format!("{path}."),
                        mult,
                    )?;
                }
                Op::Call(c) => {
                    let body = model.graphs.get(&c.graph).ok_or_else(|| {
                        Diagnostic::error(
                            "E-WL-REF-001",
                            format!("called graph {} not found", c.graph),
                        )
                        .at(path.clone())
                    })?;
                    if (body.params.len(), body.results.len())
                        != (node.inputs.len(), node.outputs.len())
                    {
                        return Err(Diagnostic::error(
                            "E-WL-REF-001",
                            format!(
                                "call passes {} inputs and {} outputs; graph {} takes {} params and {} results",
                                node.inputs.len(),
                                node.outputs.len(),
                                c.graph,
                                body.params.len(),
                                body.results.len()
                            ),
                        )
                        .at(path.clone()));
                    }
                    let mut inner = Scope::default();
                    for (p, a) in body.params.iter().zip(&node.inputs) {
                        inner.vars.insert(p.clone(), scope.get(a, &path)?.clone());
                    }
                    for (r, o) in body.results.iter().zip(&node.outputs) {
                        if !body.params.contains(r) {
                            inner.results.insert(r.clone(), scope.get(o, &path)?.clone());
                        }
                    }
                    self.walk(body, inner, &format!("{path}."), mult)?;
                }
                _ => {
                    let n = self.lower_node(g, &scope, node, path, mult)?;
                    self.out.nodes.push(n);
                }
            }
        }
        Ok(())
    }

    fn lower_node(
        &self,
        g: &Graph,
        scope: &Scope,
        node: &Node,
        path: String,
        mult: u64,
    ) -> Result<LoweredNode, Diagnostic> {
        let (model, b, opts) = (self.model, self.b, self.opts);
        let at = |d: Diagnostic| Diagnostic {
            path: Some(path.clone()),
            ..d
        };
        let mut origins = Vec::new();
        let mut types = |ids: &[Id]| -> Result<Vec<TypeInfo>, Diagnostic> {
            ids.iter()
                .map(|t| {
                    let (ti, o) = scope.get(t, &path)?;
                    origins.push((t.clone(), o.clone()));
                    Ok(ti.clone())
                })
                .collect()
        };
        let inputs = types(&node.inputs)?;
        let outputs = types(&node.outputs)?;
        let moe_rows = match &node.op {
            Op::GroupedEinsum(_) => moe_rows(model, b, opts, g, node).map_err(at)?,
            _ => None,
        };
        let ctx = NodeCtx {
            node,
            inputs,
            outputs,
            seqs: &b.seqs,
            moe_rows,
            group_size: opts.group_size,
        };
        let lowered = lower::lower(&ctx).map_err(at)?;
        let hint = lower::cost_hint(&ctx).map_err(at)?;
        let NodeCost { cost, traffic } = node_cost(&ctx, &lowered).map_err(at)?;
        Ok(LoweredNode {
            path,
            node: node.clone(),
            multiplicity: mult,
            inputs: ctx.inputs,
            outputs: ctx.outputs,
            origins,
            lowered,
            cost,
            hint,
            traffic,
        })
    }
}

/// Kept rows per expert under the scenario's routing model, from the dispatch feeding this grouped einsum.
/// Uniform routing distributes `T·k` assignments as evenly as integers allow (remainder to the lowest ids).
fn moe_rows(
    model: &Model,
    b: &Binding,
    opts: &LowerOpts,
    g: &Graph,
    node: &Node,
) -> Result<Option<MoeRows>, Diagnostic> {
    let mut cur = node.inputs.first();
    let mut dispatch = None;
    while let Some(t) = cur {
        let Some(p) = g.nodes.iter().find(|n| n.outputs.contains(t)) else {
            break;
        };
        if let Op::MoeDispatch(a) = &p.op {
            dispatch = Some(a);
            break;
        }
        cur = p.inputs.first();
    }
    let Some(d) = dispatch else { return Ok(None) };
    match opts
        .routing
        .or(model.routing.as_ref())
        .unwrap_or(&RoutingModel::Uniform)
    {
        RoutingModel::Uniform => {}
        other => {
            return Err(Diagnostic::error(
                "E-WL-OP-001",
                format!("routing model {other:?} is not supported by the M0 lowering"),
            )
            .hint("use uniform"));
        }
    }
    let (e, tk) = (u64::from(d.n_experts), b.seqs.tokens() * u64::from(d.top_k));
    let mut rows: Vec<u64> = (0..e).map(|i| tk / e + u64::from(i < tk % e)).collect();
    if let (Some(cf), DropPolicy::DropOverflow) = (&d.capacity_factor, d.drop_policy) {
        let c = (cf.eval(b).map_err(|_| {
            Diagnostic::error("E-WL-DIM-001", "capacity_factor must be a constant")
        })? * Rational::int(tk.into()))
        .checked_div(Rational::int(e.into()))
        .map(Rational::ceil)
        .and_then(Rational::to_u64)
        .unwrap_or(0);
        rows.iter_mut().for_each(|r| *r = (*r).min(c));
    }
    Ok(Some(MoeRows {
        rows,
        drop_overflow: d.drop_policy == DropPolicy::DropOverflow,
    }))
}

/// Resident model state by class (footprints at bound `kv_cap`/`slots`, times `stack`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resident {
    pub weights: u128,
    pub kv_cache: u128,
    pub constants: u128,
}

/// Whole-step totals. Compulsory bytes count model state and host I/O only: activations are internal to a
/// whole step (02 §12.5); per-node compulsory bytes (isolated mode) are in each node's `cost`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    pub useful: CostHint,
    pub weight_read: u128,
    pub kv_read: u128,
    pub kv_written: u128,
    pub constant_read: u128,
    pub io_read: u128,
    pub io_written: u128,
    pub resident: Resident,
}

impl StepStats {
    pub fn compulsory_bytes(&self) -> u128 {
        self.weight_read
            + self.kv_read
            + self.kv_written
            + self.constant_read
            + self.io_read
            + self.io_written
    }
}

pub fn stats(model: &Model, b: &Binding, lg: &LoweredGraph) -> Result<StepStats, Diagnostic> {
    let mut s = StepStats::default();
    let mut per_root: std::collections::BTreeMap<Id, (TensorClass, u128, u128, u128)> =
        Default::default();
    for n in &lg.nodes {
        s.useful = s.useful + n.cost * u128::from(n.multiplicity);
        for t in &n.traffic {
            let Some((_, o)) = n.origins.iter().find(|(id, _)| id == &t.tensor) else {
                continue;
            };
            if o.class == TensorClass::Activation {
                continue;
            }
            let m = if o.stacked {
                u128::from(n.multiplicity)
            } else {
                1
            };
            let e = per_root.entry(o.root.clone()).or_insert((o.class, 0, 0, m));
            e.1 = e.1.max(t.read);
            e.2 = e.2.max(t.written);
            e.3 = e.3.max(m);
        }
    }
    for (class, r, w, m) in per_root.into_values() {
        let (r, w) = (r * m, w * m);
        match class {
            TensorClass::Weight => s.weight_read += r,
            TensorClass::KvCache => (s.kv_read, s.kv_written) = (s.kv_read + r, s.kv_written + w),
            TensorClass::Constant => s.constant_read += r,
            TensorClass::Input | TensorClass::Output => {
                (s.io_read, s.io_written) = (s.io_read + r, s.io_written + w)
            }
            TensorClass::Activation => {}
        }
    }
    s.resident = resident(model, b)?;
    Ok(s)
}

/// Bytes of model-level state, stacked tensors at their full stack (E-WL-DIM-001 beyond `u128`).
pub fn resident(model: &Model, b: &Binding) -> Result<Resident, Diagnostic> {
    let mut r = Resident::default();
    for (id, t) in &model.tensors {
        let ti = bind_type(t, b, &format!("tensors.{id}"))?;
        let stack = t
            .stack
            .as_ref()
            .map_or(Ok(1), |e| eval_dim(e, b, &format!("tensors.{id}.stack")))?;
        let slot = match t.class {
            TensorClass::Weight => &mut r.weights,
            TensorClass::KvCache => &mut r.kv_cache,
            _ => &mut r.constants,
        };
        *slot = ti
            .footprint()
            .checked_mul(u128::from(stack))
            .and_then(|bytes| slot.checked_add(bytes))
            .ok_or_else(|| {
                Diagnostic::error(
                    "E-WL-DIM-001",
                    format!("tensors.{id}: resident bytes over the stack of {stack} overflow"),
                )
            })?;
    }
    Ok(r)
}
