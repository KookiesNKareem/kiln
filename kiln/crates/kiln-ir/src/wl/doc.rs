//! Workload documents, canonical form and hashes (02 §1, §11.5).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::Meta;
use super::dim::DimExpr;
use super::graph::Graph;
use super::plan::ParallelPlan;
use super::scenario::{RoutingModel, Scenario, SeqBatch};
use super::tensor::TensorDecl;
use crate::common::{Id, content_hash};

pub const SCHEMA_VERSION: &str = "0.1";
pub const HASH_PREFIX: &str = "wl1-";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymKind {
    Size,
    Segments,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolDecl {
    pub kind: SymKind,
    #[serde(default)]
    pub default: Option<DimExpr>,
    #[serde(default)]
    pub min: Option<u64>,
    #[serde(default)]
    pub max: Option<u64>,
    #[serde(default)]
    pub divisible_by: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

impl SymbolDecl {
    pub fn size(default: Option<u64>) -> Self {
        Self {
            kind: SymKind::Size,
            default: default.map(DimExpr::int),
            min: None,
            max: None,
            divisible_by: None,
            doc: None,
        }
    }

    pub fn segments() -> Self {
        Self {
            kind: SymKind::Segments,
            ..Self::size(None)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryPoints {
    pub forward: Id,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub symbols: IndexMap<String, SymbolDecl>,
    pub graphs: IndexMap<Id, Graph>,
    pub entry: EntryPoints,
    pub tensors: IndexMap<Id, TensorDecl>,
    #[serde(default)]
    pub routing: Option<RoutingModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZooRef {
    pub preset: String,
    #[serde(default)]
    pub overrides: serde_json::Map<String, Value>,
}

/// `model` is either a full model or `{"zoo": {...}}` sugar, expanded by `kiln-wl` before hashing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSrc {
    Zoo { zoo: ZooRef },
    Full(Box<Model>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadDoc {
    pub kiln_workload: String,
    pub id: Id,
    pub model: ModelSrc,
    #[serde(default)]
    pub scenarios: IndexMap<Id, Scenario>,
    #[serde(default)]
    pub plans: IndexMap<Id, ParallelPlan>,
    #[serde(default, skip_serializing_if = "Meta::is_empty")]
    pub meta: Meta,
}

impl WorkloadDoc {
    pub fn new(id: Id, model: Model) -> Self {
        Self {
            kiln_workload: SCHEMA_VERSION.into(),
            id,
            model: ModelSrc::Full(Box::new(model)),
            scenarios: IndexMap::new(),
            plans: IndexMap::new(),
            meta: Meta::default(),
        }
    }

    /// The expanded model, if the document carries one (zoo sugar needs `kiln-wl` expansion first).
    pub fn expanded(&self) -> Option<&Model> {
        match &self.model {
            ModelSrc::Full(m) => Some(m),
            ModelSrc::Zoo { .. } => None,
        }
    }
}

fn to_value<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).expect("IR types serialize to JSON")
}

/// Canonical JSON value of a model (02 §11.5): defaults materialized, nodes in canonical topological order,
/// maps sorted (by `canonical_json`), `meta` and `doc` removed. Dim expressions are canonical by construction.
pub fn canonical_model(model: &Model) -> Value {
    let mut m = model.clone();
    m.symbols.values_mut().for_each(|s| s.doc = None);
    m.tensors
        .values_mut()
        .for_each(|t| t.meta = Meta::default());
    for g in m.graphs.values_mut() {
        g.tensors
            .values_mut()
            .for_each(|t| t.meta = Meta::default());
        let order = g.topo_order();
        let mut nodes: Vec<_> = order.into_iter().map(|i| g.nodes[i].clone()).collect();
        nodes.iter_mut().for_each(|n| n.meta = Meta::default());
        g.nodes = nodes;
    }
    to_value(&m)
}

pub fn model_hash(model: &Model) -> String {
    content_hash(HASH_PREFIX, &canonical_model(model))
}

pub fn scenario_hash(s: &Scenario) -> String {
    let mut s = s.clone();
    if let super::scenario::ScenarioMode::Snapshot { seqs, .. } = &mut s.mode {
        *seqs = seqs.canonical();
    }
    content_hash(HASH_PREFIX, &to_value(&s))
}

pub fn plan_hash(p: &ParallelPlan) -> String {
    content_hash(HASH_PREFIX, &to_value(p))
}

pub fn binding_hash(seqs: &SeqBatch, bindings: &IndexMap<String, u64>) -> String {
    content_hash(
        HASH_PREFIX,
        &json!({"seqs": to_value(&seqs.canonical()), "bindings": to_value(bindings)}),
    )
}

/// `H(model_hash, scenario_hash, plan_hash or "none")`.
pub fn workload_hash(model: &Model, scenario: &Scenario, plan: Option<&ParallelPlan>) -> String {
    let plan = plan.map_or_else(|| "none".to_string(), plan_hash);
    content_hash(
        HASH_PREFIX,
        &json!({"model": model_hash(model), "scenario": scenario_hash(scenario), "plan": plan}),
    )
}
