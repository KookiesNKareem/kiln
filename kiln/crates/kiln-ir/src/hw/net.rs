//! Interconnect: networks, routers, links, PHYs, ports, switches, address maps (01 §8.4, §10).

use serde::{Deserialize, Serialize};

use super::phys::{Footprint, PowerOverride};
use super::quantity::{BitsPerSec, Bytes, BytesPerSec, Cycles, Hz, JoulesPerByte, Seconds, Um, Watts};
use super::types::{ClockRef, Ref, Replication, Selector, one, yes};
use crate::common::Id;
use crate::precision::Precision;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub topology: Topology,
    pub endpoints: Vec<EndpointBinding>,
    #[serde(default)]
    pub router: RouterSpec,
    pub link: LinkSpec,
    #[serde(default)]
    pub routing: Routing,
    #[serde(default)]
    pub features: NetFeatures,
    #[serde(default)]
    pub flit_bits: Option<u32>,
    #[serde(default)]
    pub clock: Option<ClockRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Topology {
    Bus {
        #[serde(default)]
        arbitration: Arbitration,
    },
    Crossbar {
        #[serde(default)]
        speedup: Option<f64>,
    },
    P2p,
    Ring {
        #[serde(default = "yes")]
        bidirectional: bool,
        #[serde(default = "one")]
        rings: u32,
    },
    Mesh {
        dims: Vec<u32>,
    },
    Torus {
        dims: Vec<u32>,
        #[serde(default)]
        wrap: Option<Vec<bool>>,
    },
    Tree {
        arity: u32,
        levels: u32,
    },
    FatTree {
        arity: u32,
        levels: u32,
        #[serde(default)]
        taper: Option<f64>,
    },
    Star {
        center: Ref,
    },
    Hierarchical {
        levels: Vec<Ref>,
        gateways: Vec<GatewaySpec>,
    },
    Custom {
        routers: u32,
        edges: Vec<CustomEdge>,
    },
}

impl Network {
    /// Every link spec the network declares: the main link, endpoint ports and custom-graph edge links.
    pub fn link_specs(&self) -> impl Iterator<Item = &LinkSpec> {
        let edges = match &self.topology {
            Topology::Custom { edges, .. } => edges.as_slice(),
            _ => &[],
        };
        std::iter::once(&self.link).chain(self.endpoints.iter().filter_map(|e| e.port.as_ref())).chain(edges.iter().filter_map(|e| e.link.as_ref()))
    }
}

impl Topology {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bus { .. } => "bus",
            Self::Crossbar { .. } => "crossbar",
            Self::P2p => "p2p",
            Self::Ring { .. } => "ring",
            Self::Mesh { .. } => "mesh",
            Self::Torus { .. } => "torus",
            Self::Tree { .. } => "tree",
            Self::FatTree { .. } => "fat_tree",
            Self::Star { .. } => "star",
            Self::Hierarchical { .. } => "hierarchical",
            Self::Custom { .. } => "custom",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arbitration {
    #[default]
    RoundRobin,
    FixedPriority,
}

/// Not specified by 01; minimal form: a node (router, switch or endpoint) joining the listed level networks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewaySpec {
    pub node: Ref,
    pub joins: Vec<Ref>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomEdge {
    pub a: u32,
    pub b: u32,
    #[serde(default)]
    pub link: Option<LinkSpec>,
    #[serde(default = "one")]
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointBinding {
    pub select: Selector,
    #[serde(default)]
    pub at: RouterBinding,
    #[serde(default)]
    pub port: Option<LinkSpec>,
    #[serde(default)]
    pub multiplicity: Option<u32>,
    #[serde(default)]
    pub ports: Option<Selector>,
}

/// Authored as `"auto"`, `"layout"`, `{index: [[..]]}`, `{per_router: n}`, `{router: [..]}`,
/// `{layout_offset: [..]}`; the canonical form is externally tagged.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RouterBinding {
    #[default]
    Auto,
    Layout,
    Index(Vec<Vec<u32>>),
    Concentrated {
        per_router: u32,
    },
    Fixed {
        router: Vec<u32>,
    },
    LayoutOffset(Vec<i32>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterSpec {
    #[serde(default)]
    pub radix: Option<u32>,
    #[serde(default)]
    pub pipeline: Option<Cycles>,
    #[serde(default)]
    pub input_buffer_flits: Option<u32>,
    #[serde(default)]
    pub vcs: Option<u32>,
    #[serde(default)]
    pub footprint: Option<Footprint>,
    #[serde(default)]
    pub power: PowerOverride,
}

/// `source` is an addition to 01 §10.3 (citation for `latency`/`energy`/`bandwidth` overrides, E-IR-1103).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkSpec {
    #[serde(default)]
    pub width_bits: Option<u32>,
    #[serde(default)]
    pub clock: Option<ClockRef>,
    #[serde(default = "yes")]
    pub full_duplex: bool,
    #[serde(default = "one")]
    pub count: u32,
    #[serde(default)]
    pub phys: LinkPhys,
    #[serde(default)]
    pub latency: Option<Seconds>,
    #[serde(default)]
    pub energy: Option<JoulesPerByte>,
    #[serde(default)]
    pub bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub source: Option<String>,
}

impl Default for LinkSpec {
    fn default() -> Self {
        Self {
            width_bits: None,
            clock: None,
            full_duplex: true,
            count: 1,
            phys: LinkPhys::default(),
            latency: None,
            energy: None,
            bandwidth: None,
            source: None,
        }
    }
}

impl LinkSpec {
    /// Structural per-direction bandwidth of one link (01 §10.3) given the clock that applies to it.
    pub fn derived_bandwidth(&self, clock: Option<Hz>) -> Option<BytesPerSec> {
        let bits_per_s = match &self.phys {
            LinkPhys::OnDie { .. } | LinkPhys::Vertical(_) => f64::from(self.width_bits?) * clock?.0,
            LinkPhys::D2d(d) => d.phy_bits_per_s(),
            LinkPhys::Serdes(s) => {
                f64::from(s.lanes) * s.lane_rate_bits_per_s.0 * s.encoding_efficiency.unwrap_or(1.0)
            }
            LinkPhys::Optical(o) => f64::from(o.lanes) * o.lane_rate_bits_per_s.0,
        };
        Some(BytesPerSec(bits_per_s / 8.0))
    }

    pub fn effective_bandwidth(&self, clock: Option<Hz>) -> Option<BytesPerSec> {
        self.bandwidth.or_else(|| self.derived_bandwidth(clock))
    }

    pub fn has_overrides(&self) -> bool {
        self.latency.is_some() || self.energy.is_some() || self.bandwidth.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinkPhys {
    OnDie {
        #[serde(default)]
        metal: MetalClass,
        #[serde(default)]
        repeated: Option<bool>,
        #[serde(default)]
        swing: Swing,
        #[serde(default)]
        pipelined: Option<bool>,
        #[serde(default)]
        sizing: Option<WireSizing>,
    },
    D2d(D2dSpec),
    Serdes(SerdesSpec),
    Vertical(VerticalSpec),
    Optical(OpticalSpec),
}

impl Default for LinkPhys {
    fn default() -> Self {
        Self::OnDie { metal: MetalClass::Auto, repeated: None, swing: Swing::Full, pipelined: None, sizing: None }
    }
}

impl LinkPhys {
    pub fn name(&self) -> &'static str {
        match self {
            Self::OnDie { .. } => "on_die",
            Self::D2d(_) => "d2d",
            Self::Serdes(_) => "serdes",
            Self::Vertical(_) => "vertical",
            Self::Optical(_) => "optical",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalClass {
    #[default]
    Auto,
    Local,
    Intermediate,
    Global,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Swing {
    #[default]
    Full,
    Low,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum WireSizing {
    Delay,
    Energy,
    Custom { k_h: f64, k_s: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct D2dSpec {
    pub standard: D2dStandard,
    pub modules: u32,
    #[serde(default)]
    pub lanes_per_module: Option<u32>,
    pub pin_rate_bits_per_s: BitsPerSec,
    #[serde(default)]
    pub bump_pitch_um: Option<Um>,
    #[serde(default)]
    pub reach: Option<Um>,
}

impl D2dSpec {
    pub fn lanes_per_module(&self) -> u32 {
        self.lanes_per_module.unwrap_or(match self.standard {
            D2dStandard::UcieStandard => 16,
            _ => 64,
        })
    }

    pub fn phy_bits_per_s(&self) -> f64 {
        f64::from(self.modules) * f64::from(self.lanes_per_module()) * self.pin_rate_bits_per_s.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum D2dStandard {
    UcieStandard,
    UcieAdvanced,
    Bow,
    Aib,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerdesSpec {
    pub protocol: SerdesProtocol,
    pub lanes: u32,
    pub lane_rate_bits_per_s: BitsPerSec,
    #[serde(default)]
    pub encoding_efficiency: Option<f64>,
    #[serde(default)]
    pub fec_latency: Option<Seconds>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerdesProtocol {
    NvlinkLike,
    IciLike,
    Pcie,
    Ethernet,
    Infiniband,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerticalSpec {
    pub bond: BondKind,
    pub pitch_um: Um,
    #[serde(default)]
    pub signals: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpticalSpec {
    pub lanes: u32,
    pub lane_rate_bits_per_s: BitsPerSec,
    #[serde(default)]
    pub switch_latency: Option<Seconds>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BondKind {
    Hybrid,
    Microbump,
    Tsv,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Routing {
    #[default]
    Default,
    DimensionOrder {
        order: Vec<u32>,
    },
    MinimalAdaptive,
    Table {
        routes: Vec<RouteEntry>,
    },
}

/// Not specified by 01: an explicit route as a router sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteEntry {
    pub src: u32,
    pub dst: u32,
    pub path: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetFeatures {
    #[serde(default)]
    pub multicast: bool,
    #[serde(default)]
    pub broadcast: bool,
    #[serde(default)]
    pub in_network_reduce: Vec<Precision>,
    #[serde(default)]
    pub ordered: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoPort {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub kind: IoKind,
    #[serde(default)]
    pub phy: Option<Ref>,
    pub internal: Ref,
    #[serde(default)]
    pub link: Option<LinkSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoKind {
    D2d,
    Hbm,
    Serdes,
    Pcie,
    Lpddr,
    Optical,
    Vertical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkDecl {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub a: Ref,
    pub b: Ref,
    pub link: LinkSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortExport {
    pub id: Id,
    pub from: Ref,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhySpec {
    pub for_kind: IoKind,
    #[serde(default)]
    pub lanes: Option<u32>,
    #[serde(default)]
    pub shoreline: Option<Um>,
    #[serde(default)]
    pub site: Option<Ref>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Switch {
    pub id: Id,
    #[serde(flatten)]
    pub rep: Replication,
    pub radix: u32,
    pub port: LinkSpec,
    #[serde(default)]
    pub latency: Option<Seconds>,
    #[serde(default)]
    pub in_network_reduce: Vec<Precision>,
    #[serde(default)]
    pub reduce_bandwidth: Option<BytesPerSec>,
    #[serde(default)]
    pub power: Option<Watts>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddressMap {
    pub id: Id,
    pub targets: Selector,
    pub granule: Bytes,
    #[serde(default)]
    pub hash: InterleaveHash,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterleaveHash {
    #[default]
    Linear,
    XorFold,
}
