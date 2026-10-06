//! Workload IR types (spec 02): pure data, serde, canonical hashing and structural validation.
//! Transforms (binding, lowering, partition, zoo) live in `kiln-wl`.

pub mod dim;
pub mod doc;
pub mod dtype;
pub mod graph;
pub mod kernel;
pub mod op;
pub mod plan;
pub mod scenario;
pub mod tensor;
pub mod validate;

pub use dim::{DimEnv, DimExpr, EvalError, Rational, SegAgg, SegField};
pub use doc::{
    EntryPoints, HASH_PREFIX, Model, ModelSrc, SCHEMA_VERSION, SymKind, SymbolDecl, WorkloadDoc,
    ZooRef, binding_hash, canonical_model, model_hash, plan_hash, scenario_hash, workload_hash,
};
pub use dtype::{ElemType, Scaling};
pub use graph::{Graph, Region, RegionKind};
pub use kernel::{
    Access, Combiner, CostHint, DiffConstraint, DimKind, Domain, IndexExpr, Kernel, KernelClass,
    LoopDim, Operand, ScalarBody, SegmentDomain, Term, Work,
};
pub use op::*;
pub use plan::{
    ParallelPlan, PipelinePlan, PpSchedule, ShardAnnot, ShardSpec, StageSplit, Template,
};
pub use scenario::*;
pub use tensor::{InitHint, Layout, LayoutSpec, Sparsity, TensorClass, TensorDecl, TypeInfo};
pub use validate::{validate_doc, validate_model};

/// Free-form metadata, excluded from hashes.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Meta(pub indexmap::IndexMap<String, serde_json::Value>);

impl Meta {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
