use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Tier {
    A,
    B,
}

/// 05 §3.3; ordered so that `level >= TraceLevel::Summary` reads as in 06 §6.3.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TraceLevel {
    None,
    #[default]
    Summary,
    Ops,
    Full,
}

/// 06 §10 trust levels, in increasing order.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    #[default]
    Uncalibrated,
    CalibratedSingleChip,
    CalibratedPhysical,
    Audited,
    CalibratedMultiChip,
}

/// 00 contract + 03 §10 + 06 §6.3: everything needed to reproduce a result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub kiln_version: String,
    pub git_hash: String,
    pub design_hash: String,
    pub workload_hash: String,
    pub calibration_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options_hash: Option<String>,
    pub tier: Tier,
    #[serde(default)]
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub trust_level: TrustLevel,
    /// Whole-step repeat window `w` (03 §4.9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_bytes: Option<u64>,
    /// Search and engine flags that affect metrics (03 §10 "search flags").
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub flags: BTreeMap<String, String>,
}

impl Provenance {
    /// Provenance of a trace with no recorded run (schemas, empty and synthetic traces).
    pub fn unknown(tier: Tier) -> Self {
        Provenance {
            kiln_version: crate::KILN_VERSION.into(),
            git_hash: "unknown".into(),
            design_hash: String::new(),
            workload_hash: String::new(),
            calibration_hash: String::new(),
            calibration_id: None,
            mapping_hash: None,
            options_hash: None,
            tier,
            seeds: vec![],
            trust_level: TrustLevel::Uncalibrated,
            window: None,
            chunk_bytes: None,
            flags: BTreeMap::new(),
        }
    }
}
