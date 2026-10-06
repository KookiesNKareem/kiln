//! Expanded hardware model (01 §16): instances, channels, shared resources, and the queries engines use.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use serde::Serialize;

use super::compute::{ComputeKind, ComputeUnit, DramKind, LocalBuffer, MemKind, MemStack, Memory, OperandRole, PrecisionMode};
use super::net::{AddressMap, IoPort, Network, Switch};
use super::phys::{CapLevel, ClockDomain, PowerPolicy};
use super::quantity::{Bytes, BytesPerSec, Hz, Seconds, Watts};
use super::select::{Seg, Selector, Start, glob, index_matches};
use super::types::{Block, Die, ExecModel, Host, Package};
use crate::common::Diagnostic;
use crate::op_class::OpClass;
use crate::precision::Precision;

pub type ContIx = usize;
pub type UnitIx = usize;
pub type MemIx = usize;
pub type BlockIx = usize;
pub type RouterIx = usize;
pub type PortIx = usize;
pub type NetIx = usize;
pub type ChanIx = usize;
pub type ResIx = usize;
pub type ClockIx = usize;
pub type PdIx = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum NodeIx {
    Container(ContIx),
    Unit(UnitIx),
    Mem(MemIx),
    Block(BlockIx),
    Router(RouterIx),
    Port(PortIx),
    Net(NetIx),
}

/// One instance in the expanded tree; every typed instance points at its node.
#[derive(Clone, Debug, Serialize)]
pub struct InstNode {
    pub path: String,
    /// Template-level entity path (instance suffixes stripped), used to fold repeated diagnostics.
    pub entity: String,
    pub entity_id: String,
    pub inst_id: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub index: u32,
    pub count: u32,
    /// Layout coordinates (grid) of this instance; empty for linear layouts.
    pub coord: Vec<u32>,
    pub ix: NodeIx,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainerKind {
    System,
    Host,
    Board,
    Package,
    Die,
    Cluster,
    Switch,
}

#[derive(Clone, Debug, Serialize)]
pub struct Container {
    pub node: usize,
    pub kind: ContainerKind,
    pub clock: Option<ClockIx>,
    pub tech: Option<String>,
    pub exec_model: Option<ExecModel>,
    pub host: Option<Arc<Host>>,
    pub switch: Option<Arc<Switch>>,
    /// Die instance spec (floorplan, layer, role, power overrides) for kiln-phys; not part of `kiln expand` output.
    #[serde(skip)]
    pub die: Option<Arc<Die>>,
    /// Package instance spec (substrate, stacks, layers) for kiln-phys.
    #[serde(skip)]
    pub package: Option<Arc<Package>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeedInst {
    pub mem: MemIx,
    pub read: Option<ChanIx>,
    pub write: Option<ChanIx>,
    pub via: Option<NetIx>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NearInst {
    pub mem: MemIx,
    /// Index of the granule (bank, pseudo-channel, ...) of `mem` this instance is bound to.
    pub slice: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnitInst {
    pub node: usize,
    pub spec: Arc<ComputeUnit>,
    pub ops: Vec<OpClass>,
    pub feeds: BTreeMap<OperandRole, FeedInst>,
    pub local: Vec<MemIx>,
    pub near: Option<NearInst>,
    pub clock: Option<ClockIx>,
    pub container: ContIx,
}

#[derive(Clone, Debug, Serialize)]
pub enum MemSpec {
    OnChip(Arc<Memory>),
    Stack(Arc<MemStack>),
    Local { buffer: LocalBuffer, unit: UnitIx },
}

#[derive(Clone, Debug, Serialize)]
pub struct MemInst {
    pub node: usize,
    pub spec: MemSpec,
    pub capacity: Bytes,
    pub clock: Option<ClockIx>,
    pub container: ContIx,
    /// Structural peak bandwidth: ports x clock (on-chip) or io width x pin rate (stack), or the override.
    pub bandwidth: Option<BytesPerSec>,
    pub bandwidth_derived: Option<BytesPerSec>,
    pub backing: Vec<MemIx>,
}

impl MemInst {
    pub fn is_stack(&self) -> bool {
        matches!(self.spec, MemSpec::Stack(_))
    }
    pub fn is_local(&self) -> bool {
        matches!(self.spec, MemSpec::Local { .. })
    }
    pub fn onchip_kind(&self) -> Option<MemKind> {
        match &self.spec {
            MemSpec::OnChip(m) => Some(m.kind),
            _ => None,
        }
    }
    pub fn dram_kind(&self) -> Option<DramKind> {
        match &self.spec {
            MemSpec::Stack(s) => Some(s.kind),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BlockInst {
    pub node: usize,
    pub spec: Arc<Block>,
    pub container: ContIx,
    pub clock: Option<ClockIx>,
    /// Created by kiln (stack attach shorthands), reported by `kiln expand` so authors can pin it.
    pub synthesized: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouterInst {
    pub node: usize,
    pub net: NetIx,
    pub coord: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PortInst {
    pub node: usize,
    pub spec: Arc<IoPort>,
    pub container: ContIx,
}

#[derive(Clone, Debug, Serialize)]
pub struct NetInst {
    pub node: usize,
    pub spec: Arc<Network>,
    pub endpoints: Vec<NodeIx>,
    pub routers: Vec<RouterIx>,
    pub clock: Option<ClockIx>,
    /// Direct network: selected packages/dies act as routers and their ports carry the links.
    pub direct: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Feed,
    MemPort,
    NocHop,
    Bus,
    D2d,
    Serdes,
    Vertical,
    Optical,
    Near,
    Host,
}

#[derive(Clone, Debug, Serialize)]
pub struct Channel {
    pub src: NodeIx,
    pub dst: NodeIx,
    pub kind: ChannelKind,
    pub width_bits: Option<u32>,
    pub clock: Option<ClockIx>,
    pub resource: ResIx,
    pub network: Option<NetIx>,
    /// Structural per-direction bandwidth (width x clock, PHY lanes x rate, or override); kiln-phys refines.
    pub bandwidth: Option<BytesPerSec>,
    /// `bandwidth` without overrides; None when kiln-ir cannot derive it (e.g. an on-die link with no clock).
    pub bandwidth_derived: Option<BytesPerSec>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResKind {
    Bus,
    Port,
    LinkDir,
    RouterXbar,
    Dma,
    MemBankGroup,
}

#[derive(Clone, Debug, Serialize)]
pub struct SharedResource {
    pub kind: ResKind,
    pub capacity: Option<BytesPerSec>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClockInst {
    pub path: String,
    pub spec: ClockDomain,
}

#[derive(Clone, Debug, Serialize)]
pub struct PowerDomainInst {
    pub path: String,
    pub cap: Watts,
    pub policy: PowerPolicy,
    pub members: Vec<ContIx>,
    /// Empty = every clock of the members.
    pub clocks: Vec<ClockIx>,
    pub idle: Option<Watts>,
    pub level: Option<CapLevel>,
    /// `Some` for an unpublished cap (never throttles; 04 §8.1).
    pub assumed: Option<super::phys::CapRange>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AddressMapInst {
    pub path: String,
    pub spec: AddressMap,
    pub targets: Vec<NodeIx>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HwModel {
    pub design_hash: String,
    pub schema: String,
    pub name: String,
    pub exec_model: ExecModel,
    pub family: Option<String>,
    pub nodes: Vec<InstNode>,
    pub tree: Vec<Container>,
    pub units: Vec<UnitInst>,
    pub memories: Vec<MemInst>,
    pub blocks: Vec<BlockInst>,
    pub routers: Vec<RouterInst>,
    pub ports: Vec<PortInst>,
    pub networks: Vec<NetInst>,
    pub channels: Vec<Channel>,
    pub resources: Vec<SharedResource>,
    pub clocks: Vec<ClockInst>,
    pub power_domains: Vec<PowerDomainInst>,
    pub address_maps: Vec<AddressMapInst>,
    pub index: BTreeMap<String, NodeIx>,
    /// Derived memory level per `MemIx` (16.2 `level`); `u8::MAX` = unreachable from any unit.
    pub levels: Vec<u8>,
    /// Outgoing channels per node (arena index).
    pub out_edges: Vec<Vec<ChanIx>>,
}

/// Per-chip and system totals (16.2 `summary`, 02 §16 asks a-e).
#[derive(Clone, Debug, Default, Serialize)]
pub struct HwSummary {
    pub name: String,
    pub design_hash: String,
    pub exec_model: Option<ExecModel>,
    pub chips: Vec<ChipSummary>,
    pub chip_count: usize,
    pub inter_chip: Vec<NetSummary>,
    pub host_launch_overhead: Option<Seconds>,
    /// Dense MAC-mode peaks summed over chips, ops/s (FLOP = 2 x MAC).
    pub peak_ops: BTreeMap<String, f64>,
    pub offchip_capacity: Bytes,
    pub offchip_bandwidth: BytesPerSec,
    pub onchip_capacity: Bytes,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ChipSummary {
    pub path: String,
    /// Mode key (`bf16*bf16+fp32`, `...:sparse` for structured-sparse peaks) -> ops/s (2 x MAC/s).
    pub peak_ops: BTreeMap<String, f64>,
    /// `(dtype, class)` key `fp32:elementwise` -> elementwise ops/s on vector/scalar/special units.
    pub elem_ops: BTreeMap<String, f64>,
    pub levels: Vec<LevelSummary>,
    pub offchip_capacity: Bytes,
    pub offchip_bandwidth: BytesPerSec,
    pub onchip_capacity: Bytes,
    pub exec_model: ExecModel,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LevelSummary {
    pub level: u8,
    pub instances: usize,
    pub capacity: Bytes,
    pub bandwidth: BytesPerSec,
}

#[derive(Clone, Debug, Serialize)]
pub struct NetSummary {
    pub path: String,
    pub topology: String,
    pub endpoints: usize,
    pub link_bandwidth: Option<BytesPerSec>,
}

impl HwModel {
    pub fn node(&self, ix: NodeIx) -> &InstNode {
        &self.nodes[self.node_of(ix)]
    }

    pub fn node_of(&self, ix: NodeIx) -> usize {
        match ix {
            NodeIx::Container(i) => self.tree[i].node,
            NodeIx::Unit(i) => self.units[i].node,
            NodeIx::Mem(i) => self.memories[i].node,
            NodeIx::Block(i) => self.blocks[i].node,
            NodeIx::Router(i) => self.routers[i].node,
            NodeIx::Port(i) => self.ports[i].node,
            NodeIx::Net(i) => self.networks[i].node,
        }
    }

    pub fn path(&self, ix: NodeIx) -> &str {
        &self.node(ix).path
    }

    pub fn enabled(&self, ix: NodeIx) -> bool {
        self.node(ix).enabled
    }

    pub fn clock_hz(&self, c: Option<ClockIx>) -> Option<Hz> {
        c.map(|c| self.clocks[c].spec.freq)
    }

    pub fn level(&self, mem: MemIx) -> u8 {
        self.levels[mem]
    }

    pub fn feed_memories(&self, unit: UnitIx) -> Vec<MemIx> {
        let mut v: Vec<MemIx> = self.units[unit].feeds.values().map(|f| f.mem).collect();
        v.extend(self.units[unit].near.as_ref().map(|n| n.mem));
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Nodes data may pass through without being staged (routers, ports, controllers, PHYs, switches...).
    fn pass_through(&self, n: usize) -> bool {
        !matches!(self.nodes[n].ix, NodeIx::Unit(_) | NodeIx::Mem(_)) && self.nodes[n].enabled
    }

    /// Deterministic shortest channel path from `src` to `dst` (BFS in channel order). Intermediate hops are
    /// never memories or units. Dimension-order routing for mesh/torus is a later milestone (03).
    pub fn route(&self, src: NodeIx, dst: NodeIx) -> Option<Vec<ChanIx>> {
        let (s, t) = (self.node_of(src), self.node_of(dst));
        if s == t {
            return Some(vec![]);
        }
        let mut prev: Vec<Option<ChanIx>> = vec![None; self.nodes.len()];
        let mut seen = vec![false; self.nodes.len()];
        seen[s] = true;
        let mut q = VecDeque::from([s]);
        while let Some(n) = q.pop_front() {
            for &c in &self.out_edges[n] {
                let m = self.node_of(self.channels[c].dst);
                if seen[m] || !self.nodes[m].enabled {
                    continue;
                }
                seen[m] = true;
                prev[m] = Some(c);
                if m == t {
                    let mut path = vec![];
                    let mut cur = t;
                    while let Some(c) = prev[cur] {
                        path.push(c);
                        cur = self.node_of(self.channels[c].src);
                    }
                    path.reverse();
                    return Some(path);
                }
                if self.pass_through(m) {
                    q.push_back(m);
                }
            }
        }
        None
    }

    /// Memories reachable from `starts` through routed channels, staging through memories on the way.
    pub fn reachable_mems(&self, starts: &[usize]) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let mut q: VecDeque<usize> = starts.iter().copied().collect();
        for &s in starts {
            seen[s] = true;
        }
        while let Some(n) = q.pop_front() {
            for &c in &self.out_edges[n] {
                let m = self.node_of(self.channels[c].dst);
                if !seen[m] && self.nodes[m].enabled && !matches!(self.nodes[m].ix, NodeIx::Unit(_)) {
                    seen[m] = true;
                    q.push_back(m);
                }
            }
        }
        seen
    }

    pub fn reachable_from(&self, mem: MemIx) -> Vec<MemIx> {
        let seen = self.reachable_mems(&[self.memories[mem].node]);
        (0..self.memories.len()).filter(|&m| m != mem && seen[self.memories[m].node]).collect()
    }

    /// Structural peak at nominal clocks for `units`, ops/s (2 x MAC/s for MAC modes, lane-ops/s otherwise).
    /// `mode` matches [`super::compute::PrecisionMode::key`] prefixes such as `bf16*bf16+fp32` or `fp32`.
    pub fn peak_ops(&self, units: &[UnitIx], mode: &str, class: OpClass) -> f64 {
        self.peak_ops_where(units, class, |m| m.key().starts_with(mode))
    }

    fn peak_ops_where(&self, units: &[UnitIx], class: OpClass, pick: impl Fn(&PrecisionMode) -> bool) -> f64 {
        units
            .iter()
            .filter(|&&u| self.nodes[self.units[u].node].enabled && self.units[u].ops.contains(&class))
            .map(|&u| {
                let ui = &self.units[u];
                let f = self.clock_hz(ui.clock).map_or(0.0, |h| h.0);
                let k = &ui.spec.kind;
                let per_cycle = ui
                    .spec
                    .precisions
                    .iter()
                    .filter(|m| pick(m))
                    .map(|m| k.ops_per_cycle(m))
                    .fold(0.0, f64::max);
                let per_op = if k.is_mac() { 2.0 } else { 1.0 };
                per_op * per_cycle * k.class_rate(class) * f
            })
            .sum()
    }

    /// Dense peak for square input precision `p` (both operands `p`), whole system or below `scope`.
    pub fn peak_ops_for(&self, p: Precision, scope: Option<&[usize]>) -> f64 {
        let units: Vec<UnitIx> = (0..self.units.len())
            .filter(|&u| self.units[u].spec.kind.is_mac() && scope.is_none_or(|s| self.under(self.units[u].node, s)))
            .collect();
        self.peak_ops_where(&units, OpClass::Matmul, |m| matches!(m, PrecisionMode::Mac { a, b, .. } if a.precision == p && b.precision == p))
    }

    fn under(&self, mut n: usize, roots: &[usize]) -> bool {
        loop {
            if roots.contains(&n) {
                return true;
            }
            match self.nodes[n].parent {
                Some(p) => n = p,
                None => return false,
            }
        }
    }

    pub fn offchip_bandwidth(&self, scope: Option<&[usize]>) -> BytesPerSec {
        BytesPerSec(
            self.memories
                .iter()
                .filter(|m| m.is_stack() && self.nodes[m.node].enabled && scope.is_none_or(|s| self.under(m.node, s)))
                .filter_map(|m| m.bandwidth.map(|b| b.0))
                .sum(),
        )
    }

    pub fn offchip_capacity(&self, scope: Option<&[usize]>) -> Bytes {
        Bytes(self.mem_sum(scope, |m| m.is_stack()))
    }

    /// On-chip bytes (memories plus unit-local buffers), optionally only of one memory kind.
    pub fn onchip_capacity(&self, scope: Option<&[usize]>, kind: Option<MemKind>) -> Bytes {
        Bytes(self.mem_sum(scope, |m| match kind {
            Some(k) => m.onchip_kind() == Some(k),
            None => !m.is_stack(),
        }))
    }

    fn mem_sum(&self, scope: Option<&[usize]>, pred: impl Fn(&MemInst) -> bool) -> u64 {
        self.memories
            .iter()
            .filter(|m| pred(m) && self.nodes[m.node].enabled && scope.is_none_or(|s| self.under(m.node, s)))
            .map(|m| m.capacity.0)
            .sum()
    }

    /// Resolves an absolute selector (no leading `/` needed) to instances.
    pub fn instances(&self, selector: &str) -> Result<Vec<NodeIx>, Diagnostic> {
        let sel = Selector::parse(selector.trim_start_matches('/'))?;
        let sel = Selector { start: Start::Root, ..sel };
        Ok(resolve_in(&self.nodes, 0, &sel).into_iter().map(|n| self.nodes[n].ix).collect())
    }

    pub fn summary(&self) -> HwSummary {
        let chips: Vec<usize> = self
            .tree
            .iter()
            .filter(|c| c.kind == ContainerKind::Package && self.nodes[c.node].enabled)
            .map(|c| c.node)
            .collect();
        let mut s = HwSummary {
            name: self.name.clone(),
            design_hash: self.design_hash.clone(),
            exec_model: Some(self.exec_model),
            chip_count: chips.len(),
            offchip_capacity: self.offchip_capacity(None),
            offchip_bandwidth: self.offchip_bandwidth(None),
            onchip_capacity: self.onchip_capacity(None, None),
            ..Default::default()
        };
        for &chip in &chips {
            let scope = [chip];
            let mut c = ChipSummary {
                path: self.nodes[chip].path.clone(),
                offchip_capacity: self.offchip_capacity(Some(&scope)),
                offchip_bandwidth: self.offchip_bandwidth(Some(&scope)),
                onchip_capacity: self.onchip_capacity(Some(&scope), None),
                exec_model: self.tree.iter().find(|t| t.node == chip).and_then(|t| t.exec_model).unwrap_or(self.exec_model),
                ..Default::default()
            };
            for u in &self.units {
                if !self.nodes[u.node].enabled || !self.under(u.node, &scope) {
                    continue;
                }
                let f = self.clock_hz(u.clock).map_or(0.0, |h| h.0);
                // Modes and sparsity entries are alternatives: a unit contributes its best of each, once.
                let (mut peak, mut elem) = (BTreeMap::<String, f64>::new(), BTreeMap::<String, f64>::new());
                let best = |map: &mut BTreeMap<String, f64>, k: String, v: f64| {
                    let e = map.entry(k).or_default();
                    *e = e.max(v);
                };
                for m in &u.spec.precisions {
                    let ops = u.spec.kind.ops_per_cycle(m) * f;
                    if u.spec.kind.is_mac() {
                        let key = mode_key(m);
                        if let ComputeKind::Matrix(mx) = &u.spec.kind {
                            for sp in &mx.sparsity {
                                best(&mut peak, format!("{key}:sparse"), 2.0 * ops * sp.speedup);
                            }
                        }
                        best(&mut peak, key, 2.0 * ops);
                    } else {
                        for &class in &u.ops {
                            let key = format!("{}:{}", mode_key(m), serde_json::to_value(class).unwrap_or_default().as_str().unwrap_or(""));
                            best(&mut elem, key, ops * u.spec.kind.class_rate(class));
                        }
                    }
                }
                for (k, v) in peak {
                    *c.peak_ops.entry(k).or_default() += v;
                }
                for (k, v) in elem {
                    *c.elem_ops.entry(k).or_default() += v;
                }
            }
            let mut levels: BTreeMap<u8, LevelSummary> = BTreeMap::new();
            for (mi, m) in self.memories.iter().enumerate() {
                if !self.nodes[m.node].enabled || !self.under(m.node, &scope) {
                    continue;
                }
                let l = levels.entry(self.levels[mi]).or_insert_with(|| LevelSummary { level: self.levels[mi], ..Default::default() });
                l.instances += 1;
                l.capacity.0 += m.capacity.0;
                l.bandwidth.0 += m.bandwidth.map_or(0.0, |b| b.0);
            }
            c.levels = levels.into_values().collect();
            for (k, v) in &c.peak_ops {
                *s.peak_ops.entry(k.clone()).or_default() += v;
            }
            s.chips.push(c);
        }
        for n in &self.networks {
            let mut pkgs: Vec<usize> =
                n.endpoints.iter().filter_map(|&e| chips.iter().copied().find(|&c| self.under(self.node_of(e), &[c]))).collect();
            pkgs.sort_unstable();
            pkgs.dedup();
            if n.direct || pkgs.len() > 1 {
                s.inter_chip.push(NetSummary {
                    path: self.nodes[n.node].path.clone(),
                    topology: n.spec.topology.name().to_owned(),
                    endpoints: n.endpoints.len(),
                    link_bandwidth: n.spec.link.effective_bandwidth(self.clock_hz(n.clock)),
                });
            }
        }
        s.host_launch_overhead = self
            .tree
            .iter()
            .filter_map(|c| c.host.as_ref().map(|h| h.launch_overhead))
            .fold(None, |acc: Option<Seconds>, x| Some(Seconds(acc.map_or(x.0, |a| a.0.max(x.0)))));
        s
    }
}

fn mode_key(m: &super::compute::PrecisionMode) -> String {
    match m {
        super::compute::PrecisionMode::Mac { a, b, acc, .. } => format!("{a}*{b}+{acc}"),
        super::compute::PrecisionMode::Elem { dtype, .. } => dtype.to_string(),
    }
}

/// Matches one selector segment against the children of `scope`.
pub(crate) fn match_children(nodes: &[InstNode], scope: usize, name: &str, index: Option<&[Vec<super::select::Span>]>) -> Vec<usize> {
    let kids = &nodes[scope].children;
    let exact_entity = name
        .strip_suffix('*')
        .filter(|p| !p.contains('*') && kids.iter().any(|&k| nodes[k].entity_id == *p));
    kids.iter()
        .copied()
        .filter(|&k| {
            let n = &nodes[k];
            let named = match exact_entity {
                Some(p) => n.entity_id == p,
                None => n.entity_id == name || glob(name, &n.inst_id),
            };
            named && index.is_none_or(|ax| n.entity_id == name.trim_end_matches('*') && index_matches(ax, n.index, &n.coord))
        })
        .collect()
}

fn descendants(nodes: &[InstNode], n: usize, out: &mut Vec<usize>) {
    out.push(n);
    for &c in &nodes[n].children {
        descendants(nodes, c, out);
    }
}

fn walk_segs(nodes: &[InstNode], start: usize, segs: &[Seg]) -> Vec<usize> {
    let mut frontier = vec![start];
    for seg in segs {
        let mut next = vec![];
        match seg {
            Seg::AnyDepth => {
                for &f in &frontier {
                    descendants(nodes, f, &mut next);
                }
            }
            Seg::Pat { name, index } => {
                for &f in &frontier {
                    next.extend(match_children(nodes, f, name, index.as_deref()));
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        frontier = next;
    }
    frontier
}

/// Resolves a parsed selector from `scope` (the container of the referencing entity): lexical lookup walks up
/// until the first segment matches (nearest wins), `^.` starts at ancestors, `/` at the root.
pub(crate) fn resolve_in(nodes: &[InstNode], scope: usize, sel: &Selector) -> Vec<usize> {
    match sel.start {
        Start::Root => walk_segs(nodes, 0, &sel.segs),
        Start::Up(k) => {
            let mut s = Some(scope);
            for _ in 0..k {
                s = s.and_then(|x| nodes[x].parent);
            }
            s.map_or_else(Vec::new, |s| walk_segs(nodes, s, &sel.segs))
        }
        Start::Lexical => {
            let mut s = Some(scope);
            while let Some(cur) = s {
                let first = walk_segs(nodes, cur, &sel.segs[..1]);
                if !first.is_empty() {
                    return walk_segs(nodes, cur, &sel.segs);
                }
                s = nodes[cur].parent;
            }
            vec![]
        }
    }
}
