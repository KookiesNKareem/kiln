//! Floorplan data, clocks, power and technology references (01 §11-§13).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::net::{BondKind, IoKind};
use super::quantity::{Hz, Joules, Mm2, Um, Volts, Watts, Cycles};
use super::types::{ClockRef, Ref, Replication, Selector};
use crate::common::Id;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Footprint {
    #[serde(default)]
    pub area: Option<Mm2>,
    #[serde(default)]
    pub w: Option<Um>,
    #[serde(default)]
    pub h: Option<Um>,
    #[serde(default)]
    pub aspect: Option<(f64, f64)>,
    #[serde(default)]
    pub utilization: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Placement {
    #[default]
    Auto,
    Pinned {
        x: Um,
        y: Um,
        #[serde(default)]
        rot: Rotation,
    },
    Region {
        x0: Um,
        y0: Um,
        x1: Um,
        y1: Um,
    },
    Edge {
        edge: Edge,
        #[serde(default)]
        offset: Option<Um>,
    },
    Array,
    Site {
        site: Ref,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rotation {
    #[default]
    R0,
    R90,
    R180,
    R270,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Edge {
    N,
    E,
    S,
    W,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DieFloorplan {
    #[serde(default)]
    pub outline: Outline,
    #[serde(default)]
    pub shoreline: Vec<ShorelineSite>,
    #[serde(default)]
    pub keepouts: Vec<Rect>,
    #[serde(default)]
    pub reticle_limit: Option<Mm2>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rect {
    pub x0: Um,
    pub y0: Um,
    pub x1: Um,
    pub y1: Um,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outline {
    #[default]
    Auto,
    Fixed {
        w: Um,
        h: Um,
    },
    MaxArea {
        area: Mm2,
        #[serde(default)]
        aspect: Option<(f64, f64)>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShorelineSite {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub edge: Edge,
    #[serde(default)]
    pub offset: Option<Um>,
    #[serde(default)]
    pub length: Option<Um>,
    pub kind: IoKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Substrate {
    #[serde(default)]
    pub kind: SubstrateKind,
    #[serde(default)]
    pub outline: Outline,
    #[serde(default)]
    pub sites: Vec<PackageSite>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubstrateKind {
    #[default]
    Organic,
    SiliconInterposer,
    RdlInterposer,
    Bridge,
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageSite {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub x: Um,
    pub y: Um,
    #[serde(default)]
    pub rot: Rotation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackLayer {
    pub id: Id,
    pub index: i32,
    pub bond: BondKind,
    pub pitch_um: Um,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockDomain {
    pub id: Id,
    pub freq: Hz,
    #[serde(default)]
    pub base: Option<Hz>,
    #[serde(default)]
    pub voltage: Option<Volts>,
    #[serde(default)]
    pub vf: Vec<VfPoint>,
    #[serde(default)]
    pub crossing_latency: Option<Cycles>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VfPoint {
    pub freq: Hz,
    pub voltage: Volts,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerDomain {
    pub id: Id,
    pub members: Selector,
    pub cap: Watts,
    #[serde(default)]
    pub policy: PowerPolicy,
    #[serde(default)]
    pub clocks: Vec<ClockRef>,
    #[serde(default)]
    pub idle: Option<Watts>,
    #[serde(default)]
    pub thermal: Option<ThermalSpec>,
    /// An unpublished cap: the plausible range `[lo, hi]` (01 §12, 04 §8.1). Assumed caps never throttle a central
    /// evaluation; `cap` is the nominal value reported beside the range.
    #[serde(default)]
    pub assumed: Option<CapRange>,
}

/// The plausible range of an assumed (unpublished) power cap.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapRange {
    pub lo: Watts,
    pub hi: Watts,
    #[serde(default)]
    pub basis: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPolicy {
    #[default]
    Dvfs,
    Fixed,
    DutyCycle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalSpec {
    pub tj_max_c: f64,
    #[serde(default)]
    pub cooling: CoolingClass,
    #[serde(default)]
    pub theta_ja: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoolingClass {
    #[default]
    Air,
    AirHp,
    LiquidColdPlate,
    Immersion,
}

/// Inline power cap on a board or package (a `PowerDomain` minus `id`/`members`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerCap {
    pub cap: Watts,
    #[serde(default)]
    pub policy: PowerPolicy,
    #[serde(default)]
    pub clocks: Vec<ClockRef>,
    #[serde(default)]
    pub idle: Option<Watts>,
    #[serde(default)]
    pub thermal: Option<ThermalSpec>,
    /// None => the level of the entity carrying the cap.
    #[serde(default)]
    pub level: Option<CapLevel>,
    /// An unpublished cap (see `PowerDomain::assumed`).
    #[serde(default)]
    pub assumed: Option<CapRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapLevel {
    Board,
    Package,
    Die,
}

/// Per-entity cost overrides; `reference` profile only, with a `source` citation (01 §18.3).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerOverride {
    #[serde(default)]
    pub area: Option<Mm2>,
    #[serde(default)]
    pub energy_per_op: BTreeMap<String, Joules>,
    #[serde(default)]
    pub leakage: Option<Watts>,
    #[serde(default)]
    pub ctrl_ge: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
}

impl PowerOverride {
    pub fn is_set(&self) -> bool {
        self.area.is_some() || !self.energy_per_op.is_empty() || self.leakage.is_some() || self.ctrl_ge.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum TechRef {
    Name(String),
    Detailed {
        node: String,
        #[serde(default)]
        variant: Option<String>,
        #[serde(default)]
        vdd: Option<Volts>,
        #[serde(default)]
        metal_stack: Option<String>,
    },
}

impl TechRef {
    pub fn node(&self) -> &str {
        match self {
            Self::Name(n) | Self::Detailed { node: n, .. } => n,
        }
    }
}

/// Node names 04's technology table defines (04 §2, `kiln-phys/data/tech`) plus aliases, and `tsmc_n12`
/// which 01's ember example uses for a PIM logic die. kiln-phys owns the real table.
pub const KNOWN_TECH: &[&str] = &[
    "tsmc_n16",
    "tsmc_n12",
    "tsmc_n7",
    "tsmc_n5",
    "tsmc_n4",
    "tsmc_n4p",
    "nvidia_4n",
    "tsmc_n3e",
    "tsmc_n2",
    "asap7",
    "sky130",
    "gf180mcu",
    "ihp_sg13g2",
    "dram_logic_1y",
    "dram_1b",
];
