//! Document structure, containers, replication and blocks (01 §3, §4, §14.1).
//!
//! These are the typed, post-authoring structs (pipeline step 6). Authoring-only constructs (`params`,
//! `templates`, `imports`, `extends`, `set`, `use`/`with`, string shorthands) are consumed on the untyped value
//! tree by `author` before these types are deserialized, so they do not appear here.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::compute::{ComputeUnit, DmaSpec, MemControllerSpec, MemStack, Memory};
use super::net::{AddressMap, IoPort, LinkDecl, LinkSpec, Network, PhySpec, PortExport, Switch};
use super::phys::{
    ClockDomain, DieFloorplan, Footprint, Placement, PowerCap, PowerDomain, PowerOverride, StackLayer, Substrate,
    TechRef,
};
use super::quantity::{Bytes, BytesPerSec, Seconds, Watts};
use crate::common::Id;

pub type Ref = String;
pub type Selector = String;
pub type Path = String;
pub type ClockRef = String;
pub type LayerRef = String;

pub const SCHEMA_CURRENT: &str = "kiln.hw/1.0";

pub(crate) fn one() -> u32 {
    1
}
pub(crate) fn one_f() -> f64 {
    1.0
}
pub(crate) fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwDoc {
    pub schema: String,
    pub name: Id,
    #[serde(default)]
    pub meta: Meta,
    pub tech: TechRef,
    #[serde(default)]
    pub exec_model: ExecModel,
    #[serde(default)]
    pub family: Option<Id>,
    #[serde(default)]
    pub clocks: Vec<ClockDomain>,
    #[serde(default)]
    pub power: Vec<PowerDomain>,
    pub system: System,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub citations: IndexMap<String, String>,
    #[serde(default)]
    pub claims: Vec<Claim>,
    #[serde(default)]
    pub notes: IndexMap<Path, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    /// e.g. `peak_ops.bf16`, `offchip_bw`, `onchip_bytes.cache`, `die_area`, `tdp`.
    pub metric: String,
    #[serde(default)]
    pub scope: Path,
    pub value: f64,
    #[serde(default = "default_rel_tol")]
    pub rel_tol: f64,
    pub source: String,
}

fn default_rel_tol() -> f64 {
    0.02
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecModel {
    #[default]
    HostLaunched,
    DeviceQueued,
    StaticDataflow,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct System {
    #[serde(default)]
    pub hosts: Vec<Host>,
    #[serde(default)]
    pub boards: Vec<Board>,
    #[serde(default)]
    pub switches: Vec<Switch>,
    #[serde(default)]
    pub networks: Vec<Network>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Board {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub packages: Vec<Package>,
    #[serde(default)]
    pub switches: Vec<Switch>,
    #[serde(default)]
    pub networks: Vec<Network>,
    #[serde(default)]
    pub host_links: Vec<HostLink>,
    #[serde(default)]
    pub power: Option<PowerCap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub dies: Vec<Die>,
    #[serde(default)]
    pub mem_stacks: Vec<MemStack>,
    #[serde(default)]
    pub links: Vec<LinkDecl>,
    #[serde(default)]
    pub networks: Vec<Network>,
    #[serde(default)]
    pub ports: Vec<PortExport>,
    #[serde(default)]
    pub substrate: Substrate,
    #[serde(default)]
    pub layers: Vec<StackLayer>,
    #[serde(default)]
    pub address_map: Vec<AddressMap>,
    #[serde(default)]
    pub exec_model: Option<ExecModel>,
    #[serde(default)]
    pub power: Option<PowerCap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Die {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    #[serde(default)]
    pub tech: Option<TechRef>,
    #[serde(default)]
    pub role: DieRole,
    #[serde(flatten)]
    pub contents: Contents,
    #[serde(default)]
    pub ports: Vec<IoPort>,
    #[serde(default)]
    pub clocks: Vec<ClockDomain>,
    #[serde(default)]
    pub default_clock: Option<ClockRef>,
    #[serde(default)]
    pub floorplan: DieFloorplan,
    #[serde(default)]
    pub layer: Option<LayerRef>,
    #[serde(default)]
    pub over: Option<Ref>,
    #[serde(default)]
    pub placement: Placement,
    #[serde(default)]
    pub stitched: bool,
    #[serde(default)]
    pub power: PowerOverride,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DieRole {
    #[default]
    Compute,
    Io,
    MemoryBase,
    Sram,
    InterposerActive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cluster {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    #[serde(flatten)]
    pub contents: Contents,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub placement: Placement,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contents {
    #[serde(default)]
    pub clusters: Vec<Cluster>,
    #[serde(default)]
    pub units: Vec<ComputeUnit>,
    #[serde(default)]
    pub memories: Vec<Memory>,
    #[serde(default)]
    pub networks: Vec<Network>,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Host {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub mem_capacity: Bytes,
    pub mem_bandwidth: BytesPerSec,
    #[serde(default)]
    pub launch_overhead: Seconds,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostLink {
    pub host: Ref,
    pub to: Selector,
    pub link: LinkSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub kind: BlockKind,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub placement: Placement,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default)]
    pub power: PowerOverride,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BlockKind {
    MemController(MemControllerSpec),
    Phy(PhySpec),
    Dma(DmaSpec),
    Sequencer {
        issue_overhead: Seconds,
    },
    Misc {
        #[serde(default)]
        power: Option<Watts>,
    },
}

/// Replication fields every entity accepts (01 §14.1), flattened into the entity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replication {
    #[serde(default)]
    pub count: Option<u32>,
    #[serde(default)]
    pub layout: Layout,
    #[serde(default)]
    pub vary: Vec<Variation>,
    #[serde(default)]
    pub disabled: Vec<Selector>,
    #[serde(default)]
    pub harvest_power: HarvestPower,
    /// Routers of disabled instances stay in mesh/torus networks (01 §12); None => true.
    #[serde(default)]
    pub keep_routers: Option<bool>,
}

impl Replication {
    /// Instance count implied by `count` and `layout` (the agreement check is E-IR-0213); saturates at
    /// `u64::MAX`, which no expansion budget admits.
    pub fn instances(&self) -> u64 {
        match (&self.layout, self.count) {
            (_, Some(n)) => u64::from(n),
            (Layout::Grid { grid, .. }, None) => grid_product(grid),
            (Layout::Ring { ring }, None) => u64::from(*ring),
            (Layout::Explicit { coords }, None) => coords.len() as u64,
            (Layout::Linear, None) => 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Layout {
    #[default]
    Linear,
    Grid {
        grid: Vec<u32>,
        #[serde(default)]
        order: GridOrder,
        #[serde(default)]
        gap: Option<super::quantity::Um>,
    },
    Ring {
        ring: u32,
    },
    Explicit {
        coords: Vec<Vec<i32>>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridOrder {
    #[default]
    RowMajor,
    ColMajor,
    Snake,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarvestPower {
    #[default]
    Gated,
    Leak,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variation {
    pub select: Selector,
    pub set: BTreeMap<String, Value>,
}

/// Instance ids and layout coordinates in layout order (01 §15): `sm` x2 -> `sm0`, `sm1`; grid -> `t1_2`;
/// a single instance keeps the bare id.
pub fn instance_ids(id: &str, rep: &Replication) -> Vec<(String, Vec<u32>)> {
    match &rep.layout {
        Layout::Grid { grid, order, .. } => grid_coords(grid, *order)
            .into_iter()
            .map(|c| {
                let name = c.iter().map(u32::to_string).collect::<Vec<_>>().join("_");
                (format!("{id}{name}"), c)
            })
            .collect(),
        _ => {
            let n = rep.instances();
            if n == 1 {
                vec![(id.to_owned(), vec![])]
            } else {
                (0..n).map(|i| (format!("{id}{i}"), vec![])).collect()
            }
        }
    }
}

/// Instances of a grid, saturating at `u64::MAX`.
pub fn grid_product(grid: &[u32]) -> u64 {
    grid.iter().try_fold(1u64, |a, &g| a.checked_mul(u64::from(g))).unwrap_or(u64::MAX)
}

fn grid_coords(grid: &[u32], order: GridOrder) -> Vec<Vec<u32>> {
    let total = grid_product(grid);
    let row_major = |mut i: u64| -> Vec<u32> {
        let mut c = vec![0u32; grid.len()];
        for d in (0..grid.len()).rev() {
            let g = u64::from(grid[d].max(1));
            c[d] = (i % g) as u32;
            i /= g;
        }
        c
    };
    (0..total)
        .map(|i| match order {
            GridOrder::RowMajor => row_major(i),
            GridOrder::ColMajor => {
                let mut c = vec![0u32; grid.len()];
                let mut i = i;
                for (d, &g) in grid.iter().enumerate() {
                    let g = u64::from(g.max(1));
                    c[d] = (i % g) as u32;
                    i /= g;
                }
                c
            }
            GridOrder::Snake => {
                let mut c = row_major(i);
                if let [.., r, last] = c.as_mut_slice()
                    && *r % 2 == 1
                {
                    *last = grid[grid.len() - 1] - 1 - *last;
                }
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_naming() {
        let rep = |count, layout| Replication { count, layout, ..Default::default() };
        let names = |r: &Replication| instance_ids("x", r).into_iter().map(|(n, _)| n).collect::<Vec<_>>();
        assert_eq!(names(&rep(None, Layout::Linear)), ["x"]);
        assert_eq!(names(&rep(Some(3), Layout::Linear)), ["x0", "x1", "x2"]);
        let grid = |order| Layout::Grid { grid: vec![2, 3], order, gap: None };
        assert_eq!(names(&rep(None, grid(GridOrder::RowMajor))), ["x0_0", "x0_1", "x0_2", "x1_0", "x1_1", "x1_2"]);
        assert_eq!(names(&rep(None, grid(GridOrder::ColMajor)))[..3], ["x0_0", "x1_0", "x0_1"]);
        assert_eq!(names(&rep(None, grid(GridOrder::Snake)))[3..], ["x1_2", "x1_1", "x1_0"]);
        assert_eq!(rep(None, grid(GridOrder::RowMajor)).instances(), 6);
    }
}
