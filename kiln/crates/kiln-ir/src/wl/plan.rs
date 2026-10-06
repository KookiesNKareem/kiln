//! Parallel plans (02 §9.2-9.5). Types only; `kiln-wl::partition` consumes them.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Template {
    pub tp: Option<String>,
    pub sp: bool,
    pub dp: Option<String>,
    pub pp: Option<String>,
    pub ep: Option<Vec<String>>,
    pub cp: Option<String>,
    pub vocab_parallel: bool,
    pub attn_dp: bool,
}

/// One entry per logical dim: replicated (empty) or sharded over mesh axes (major to minor).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardSpec {
    pub dims: Vec<Vec<String>>,
    #[serde(default)]
    pub partial: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardAnnot {
    /// Dotted tensor path, e.g. `block.wqkv` or `w_qkv`.
    pub tensor: String,
    pub spec: ShardSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageSplit {
    Even,
    Explicit { layers_per_stage: Vec<u32> },
    Balanced,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PpSchedule {
    Gpipe,
    Interleaved { r#virtual: u32 },
    InferenceRr,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelinePlan {
    pub axis: String,
    pub split: StageSplit,
    pub microbatches: u32,
    pub schedule: PpSchedule,
    #[serde(default)]
    pub embed_stage: u32,
    #[serde(default)]
    pub head_stage: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelPlan {
    pub mesh: IndexMap<String, u32>,
    #[serde(default)]
    pub template: Option<Template>,
    #[serde(default)]
    pub shardings: Vec<ShardAnnot>,
    #[serde(default)]
    pub pipeline: Option<PipelinePlan>,
    #[serde(default)]
    pub allow_uneven: bool,
}

impl ParallelPlan {
    pub fn ranks(&self) -> u64 {
        self.mesh.values().map(|&v| u64::from(v)).product()
    }
}
