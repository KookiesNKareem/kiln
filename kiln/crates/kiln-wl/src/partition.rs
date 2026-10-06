//! Partition pass interface (02 §9.7). M0 implements only the trivial single-rank plan; sharding propagation
//! and collective insertion are deferred.

use indexmap::IndexMap;
use kiln_ir::common::Diagnostic;
use kiln_ir::wl::{Model, ParallelPlan, PipelinePlan, plan_hash};

use crate::graph::{Binding, Resident, resident};

#[derive(Clone, Debug)]
pub struct StageProgram {
    pub stage: u32,
    /// Per-rank logical graph (collectives inserted) and its binding.
    pub model: Model,
    pub binding: Binding,
    pub layers: std::ops::Range<u32>,
    pub resident_bytes_per_rank: Resident,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankVariation {
    pub rank: u32,
    pub expert_rows: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct PartitionedProgram {
    pub plan_hash: String,
    pub mesh: IndexMap<String, u32>,
    pub stages: Vec<StageProgram>,
    pub pipeline: Option<PipelinePlan>,
    pub per_rank_variation: Vec<RankVariation>,
}

pub fn partition(
    model: &Model,
    b: &Binding,
    plan: &ParallelPlan,
) -> Result<PartitionedProgram, Diagnostic> {
    let Some(ranks) = plan.ranks() else {
        return Err(Diagnostic::error("E-WL-DIM-001", "mesh rank count exceeds u64"));
    };
    if ranks != 1 || plan.pipeline.is_some() || !plan.shardings.is_empty() {
        return Err(Diagnostic::error(
            "E-WL-OP-001",
            "multi-rank partitioning is deferred past M0",
        )
        .hint("use a mesh of size 1"));
    }
    let layers = b.values.get("L").copied().unwrap_or(0) as u32;
    let resident = resident(model, b)?;
    Ok(PartitionedProgram {
        plan_hash: plan_hash(plan),
        mesh: plan.mesh.clone(),
        stages: vec![StageProgram {
            stage: 0,
            model: model.clone(),
            binding: b.clone(),
            layers: 0..layers,
            resident_bytes_per_rank: resident,
        }],
        pipeline: None,
        per_rank_variation: vec![],
    })
}
