//! Expansion pipeline steps 8-11 (01 §15): instances, references, synthesized entities, routers and channels.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::author::apply_field;
use super::compute::{
    ComputeUnit, LogicDie, MemControllerSpec, MemStack, Memory, NearBinding, NearGranularity, StackAttach,
};
use super::diag::{Diags, from_value};
use super::model::*;
use super::net::{EndpointBinding, IoKind, LinkDecl, LinkPhys, LinkSpec, Network, PhySpec, RouterBinding, Topology};
use super::phys::{PowerCap, Placement, PowerOverride};
use super::quantity::BytesPerSec;
use super::select::{InstVars, Seg, Selector, Vars, glob, index_matches, interpolate, max_index};
use super::types::{
    Block, BlockKind, Board, Cluster, Contents, Die, HostLink, HwDoc, Layout, Package, Replication, instance_ids,
};
use crate::common::Diagnostic;

#[derive(Clone, Debug)]
pub struct ExpandOptions {
    /// Resource guard (E-IR-0210), not a modelling cap.
    pub max_instances: u64,
}

impl Default for ExpandOptions {
    fn default() -> Self {
        Self { max_instances: 50_000_000 }
    }
}

/// Router-to-router link: `(a, b, dimension or edge index, link override)`.
type RouterLink = (usize, usize, usize, Option<LinkSpec>);

#[derive(Clone)]
struct Cx {
    scope: usize,
    cont: ContIx,
    die_clocks: IndexMap<String, ClockIx>,
    clock: Option<ClockIx>,
    key: String,
}

#[derive(Clone, Copy)]
enum Gran {
    Mem { banks: u32 },
    Stack { channels: u32, pcs: u64, banks_per_pc: u32 },
}

impl Gran {
    fn of(m: &MemSpec) -> Self {
        match m {
            MemSpec::OnChip(m) => Self::Mem { banks: m.banks },
            MemSpec::Stack(s) => Self::stack(s),
            MemSpec::Local { .. } => Self::Mem { banks: 1 },
        }
    }

    fn stack(s: &MemStack) -> Self {
        Self::Stack { channels: s.channel_count(), pcs: s.pseudo_channels(), banks_per_pc: s.banks_per_pseudo_channel() }
    }

    fn granules(self, g: NearGranularity) -> u64 {
        match (self, g) {
            (_, NearGranularity::PerInstance | NearGranularity::PerStack) => 1,
            (Self::Mem { banks }, NearGranularity::PerBank) => banks.into(),
            (Self::Mem { banks }, NearGranularity::PerBankGroup) => banks.div_ceil(4).into(),
            (Self::Mem { .. }, _) => 1,
            (Self::Stack { channels, .. }, NearGranularity::PerChannel) => channels.into(),
            (Self::Stack { pcs, .. }, NearGranularity::PerPseudoChannel) => pcs,
            (Self::Stack { pcs, .. }, NearGranularity::PerBankGroup) => pcs.saturating_mul(4),
            (Self::Stack { pcs, banks_per_pc, .. }, NearGranularity::PerBank) => pcs.saturating_mul(banks_per_pc.into()),
        }
    }
}

struct Inst {
    id: String,
    index: u32,
    count: u32,
    coord: Vec<u32>,
}

struct B<'a> {
    doc: &'a HwDoc,
    m: HwModel,
    d: Diags,
    max: u64,
    used: u64,
    global_clocks: IndexMap<String, ClockIx>,
    disabled: Vec<(usize, String, String)>,
    scopes: Vec<Vec<(String, Gran)>>,
    links: Vec<(usize, Arc<LinkDecl>)>,
    host_links: Vec<(usize, HostLink)>,
    addr_maps: Vec<(usize, super::net::AddressMap)>,
    exports: Vec<(usize, super::net::PortExport)>,
    caps: Vec<(ContIx, PowerCap)>,
    auto_eps: BTreeMap<NetIx, Vec<usize>>,
}

fn join(a: &str, b: &str) -> String {
    if a.is_empty() { b.to_owned() } else { format!("{a}.{b}") }
}

/// Expands a typed document into an [`HwModel`]. `Ok` carries warnings; `Err` all diagnostics of the stage.
pub fn expand(doc: &HwDoc, design_hash: &str, opts: &ExpandOptions) -> Result<(HwModel, Vec<Diagnostic>), Vec<Diagnostic>> {
    let m = HwModel {
        design_hash: design_hash.to_owned(),
        schema: doc.schema.clone(),
        name: doc.name.to_string(),
        exec_model: doc.exec_model,
        family: doc.family.as_ref().map(ToString::to_string),
        nodes: vec![],
        tree: vec![],
        units: vec![],
        memories: vec![],
        blocks: vec![],
        routers: vec![],
        ports: vec![],
        networks: vec![],
        channels: vec![],
        resources: vec![],
        clocks: vec![],
        power_domains: vec![],
        address_maps: vec![],
        index: BTreeMap::new(),
        levels: vec![],
        out_edges: vec![],
    };
    let mut b = B {
        doc,
        m,
        d: Diags::default(),
        max: opts.max_instances,
        used: 0,
        global_clocks: IndexMap::new(),
        disabled: vec![],
        scopes: vec![],
        links: vec![],
        host_links: vec![],
        addr_maps: vec![],
        exports: vec![],
        caps: vec![],
        auto_eps: BTreeMap::new(),
    };
    b.system();
    if b.d.has_errors() {
        return Err(b.d.into_vec());
    }
    b.graph();
    if b.d.has_errors() {
        return Err(b.d.into_vec());
    }
    Ok((b.m, b.d.into_vec()))
}

impl B<'_> {
    /// Charges `n` synthesized instances against the expansion budget (E-IR-0210); false once it is exceeded.
    fn charge(&mut self, n: u64, key: &str) -> bool {
        self.used = self.used.saturating_add(n);
        if self.used > self.max {
            self.d.push_inst(
                "budget",
                Diagnostic::error("E-IR-0210", format!("expansion budget of {} instances exceeded at '{key}' (count {n})", self.max))
                    .at(key)
                    .hint("reduce the largest counts or raise ExpandOptions::max_instances"),
            );
        }
        self.used <= self.max
    }

    fn add_node(&mut self, parent: Option<usize>, key: &str, id: &str, inst: &Inst, ix: NodeIx) -> usize {
        let path = match parent {
            Some(p) => join(&self.m.nodes[p].path, &inst.id),
            None => String::new(),
        };
        let n = self.m.nodes.len();
        if self.m.index.contains_key(&path) {
            self.d.push(
                Diagnostic::error("E-IR-0106", format!("duplicate instance path '{path}'"))
                    .at(&path)
                    .hint("ids must be unique within a scope, including instance names like 'hbm0' from 'hbm' x N"),
            );
        } else {
            self.m.index.insert(path.clone(), ix);
        }
        let enabled = parent.is_none_or(|p| self.m.nodes[p].enabled);
        self.m.nodes.push(InstNode {
            path,
            entity: key.to_owned(),
            entity_id: id.to_owned(),
            inst_id: inst.id.clone(),
            parent,
            children: vec![],
            index: inst.index,
            count: inst.count,
            coord: inst.coord.clone(),
            ix,
            enabled,
        });
        if let Some(p) = parent {
            self.m.nodes[p].children.push(n);
        }
        n
    }

    fn replicate<T: Clone + Serialize + DeserializeOwned>(
        &mut self,
        cx: &Cx,
        id: &str,
        rep: &Replication,
        spec: &T,
    ) -> Vec<(Inst, Arc<T>)> {
        let key = join(&cx.key, id);
        let n = rep.instances();
        if let (Layout::Grid { grid, .. }, Some(c)) = (&rep.layout, rep.count) {
            let prod = super::types::grid_product(grid);
            if prod != u64::from(c) {
                self.d.push(
                    Diagnostic::error("E-IR-0213", format!("'{id}': count {c} disagrees with layout grid {grid:?} ({prod})"))
                        .at(&key)
                        .hint("omit count when a grid is given, or make product(grid) == count"),
                );
                return vec![];
            }
        }
        if n > 1 && id.ends_with(|c: char| c.is_ascii_digit()) {
            self.d.push(
                Diagnostic::error("E-IR-0105", format!("'{id}' has count {n}; instances would be named {id}0..{id}{}", n - 1))
                    .at(&key)
                    .hint(format!("rename to '{}'", id.trim_end_matches(|c: char| c.is_ascii_digit()))),
            );
            return vec![];
        }
        if !self.charge(n, &key) {
            return vec![];
        }
        let ids = instance_ids(id, rep);
        let specs = self.vary(id, &ids, rep, spec, &key);
        for s in &rep.disabled {
            self.disabled.push((cx.scope, s.clone(), key.clone()));
        }
        ids.into_iter()
            .zip(specs)
            .enumerate()
            .map(|(k, ((inst, coord), spec))| (Inst { id: inst, index: k as u32, count: n as u32, coord }, spec))
            .collect()
    }

    /// One spec per instance in `ids`, with the `vary` patches applied.
    fn vary<T: Clone + Serialize + DeserializeOwned>(
        &mut self,
        id: &str,
        ids: &[(String, Vec<u32>)],
        rep: &Replication,
        spec: &T,
        key: &str,
    ) -> Vec<Arc<T>> {
        let base = Arc::new(spec.clone());
        let mut specs = vec![base; ids.len()];
        for v in &rep.vary {
            let hits = self.select_own(id, ids, &v.select, key);
            for k in hits {
                let mut val = serde_json::to_value(&*specs[k]).expect("entity serializes");
                for (field, value) in &v.set {
                    if let Err(e) = apply_field(&mut val, field, value.clone(), &mut Vec::new()) {
                        self.d.push(e.at(key));
                    }
                }
                match from_value::<T>(val, key) {
                    Ok(t) => specs[k] = Arc::new(t),
                    Err(e) => self.d.push(e),
                }
            }
        }
        specs
    }

    /// `vary` selectors name instances of the entity itself.
    fn select_own(&mut self, id: &str, ids: &[(String, Vec<u32>)], sel: &str, key: &str) -> Vec<usize> {
        let parsed = match Selector::parse(sel) {
            Ok(s) => s,
            Err(e) => {
                self.d.push(e.at(key));
                return vec![];
            }
        };
        let [Seg::Pat { name, index }] = parsed.segs.as_slice() else {
            self.d.push(
                Diagnostic::error("E-IR-0211", format!("vary selector {sel:?} must name instances of '{id}'")).at(key),
            );
            return vec![];
        };
        let hits: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(k, (inst, coord))| {
                (name == id || glob(name, inst)) && index.as_ref().is_none_or(|ax| index_matches(ax, *k as u32, coord))
            })
            .map(|(k, _)| k)
            .collect();
        if hits.is_empty() {
            self.d.push(
                Diagnostic::error("E-IR-0211", format!("vary selector {sel:?} matches no instance of '{id}' (count {})", ids.len()))
                    .at(key),
            );
        }
        hits
    }

    fn clock(&mut self, cx: &Cx, r: Option<&str>, at: &str) -> Option<ClockIx> {
        let Some(r) = r else { return cx.clock };
        let found = cx.die_clocks.get(r).or_else(|| self.global_clocks.get(r)).copied();
        if found.is_none() {
            let mut known: Vec<&String> = cx.die_clocks.keys().collect();
            known.extend(self.global_clocks.keys());
            self.d.push_inst(
                at,
                Diagnostic::error("E-IR-0901", format!("unknown clock '{r}'")).at(at).hint(format!("known clocks: {known:?}")),
            );
        }
        found
    }

    fn container(&mut self, parent: usize, key: &str, id: &str, inst: &Inst, kind: ContainerKind) -> (usize, ContIx) {
        let ci = self.m.tree.len();
        let node = self.add_node(Some(parent), key, id, inst, NodeIx::Container(ci));
        self.m.tree.push(Container { node, kind, clock: None, tech: None, exec_model: None, host: None, switch: None, die: None, package: None });
        (node, ci)
    }

    fn sub(&self, cx: &Cx, scope: usize, cont: ContIx, id: &str) -> Cx {
        Cx { scope, cont, die_clocks: cx.die_clocks.clone(), clock: cx.clock, key: join(&cx.key, id) }
    }

    fn system(&mut self) {
        let root = Inst { id: String::new(), index: 0, count: 1, coord: vec![] };
        let node = self.add_node(None, "", "", &root, NodeIx::Container(0));
        self.m.tree.push(Container {
            node,
            kind: ContainerKind::System,
            clock: None,
            tech: Some(self.doc.tech.node().to_owned()),
            exec_model: Some(self.doc.exec_model),
            host: None,
            switch: None,
            die: None,
            package: None,
        });
        for c in &self.doc.clocks {
            self.global_clocks.insert(c.id.to_string(), self.m.clocks.len());
            self.m.clocks.push(ClockInst { path: c.id.to_string(), spec: c.clone() });
        }
        let cx = Cx { scope: 0, cont: 0, die_clocks: IndexMap::new(), clock: None, key: String::new() };
        let sys = &self.doc.system;
        for h in &sys.hosts {
            for (inst, spec) in self.replicate(&cx, h.id.as_str(), &h.rep, h) {
                let (_, ci) = self.container(0, &join(&cx.key, h.id.as_str()), h.id.as_str(), &inst, ContainerKind::Host);
                self.m.tree[ci].host = Some(spec);
            }
        }
        for b in &sys.boards {
            self.board(b, &cx);
        }
        self.switches(&sys.switches, &cx);
        for n in &sys.networks {
            self.network(n, &cx);
        }
    }

    fn switches(&mut self, sw: &[super::net::Switch], cx: &Cx) {
        for s in sw {
            for (inst, spec) in self.replicate(cx, s.id.as_str(), &s.rep, s) {
                let (_, ci) = self.container(cx.scope, &join(&cx.key, s.id.as_str()), s.id.as_str(), &inst, ContainerKind::Switch);
                self.m.tree[ci].switch = Some(spec);
            }
        }
    }

    fn board(&mut self, b: &Board, cx: &Cx) {
        for (inst, spec) in self.replicate(cx, b.id.as_str(), &b.rep, b) {
            let (node, ci) = self.container(cx.scope, &join(&cx.key, b.id.as_str()), b.id.as_str(), &inst, ContainerKind::Board);
            let cx2 = self.sub(cx, node, ci, b.id.as_str());
            for p in &spec.packages {
                self.package(p, &cx2);
            }
            self.switches(&spec.switches, &cx2);
            for n in &spec.networks {
                self.network(n, &cx2);
            }
            for hl in &spec.host_links {
                self.host_links.push((node, hl.clone()));
            }
            if let Some(cap) = &spec.power {
                self.caps.push((ci, cap.clone()));
            }
        }
    }

    fn package(&mut self, p: &Package, cx: &Cx) {
        for (inst, spec) in self.replicate(cx, p.id.as_str(), &p.rep, p) {
            let (node, ci) =
                self.container(cx.scope, &join(&cx.key, p.id.as_str()), p.id.as_str(), &inst, ContainerKind::Package);
            self.m.tree[ci].exec_model = spec.exec_model;
            self.m.tree[ci].package = Some(spec.clone());
            let cx2 = self.sub(cx, node, ci, p.id.as_str());
            self.scopes.push(
                spec.mem_stacks
                    .iter()
                    .map(|s| (s.id.to_string(), Gran::stack(s)))
                    .collect(),
            );
            for d in &spec.dies {
                self.die(d, &cx2);
            }
            for s in &spec.mem_stacks {
                self.mem_stack(s, &cx2);
            }
            for n in &spec.networks {
                self.network(n, &cx2);
            }
            self.scopes.pop();
            for l in &spec.links {
                self.links.push((node, Arc::new(l.clone())));
            }
            for a in &spec.address_map {
                self.addr_maps.push((node, a.clone()));
            }
            for e in &spec.ports {
                self.exports.push((node, e.clone()));
            }
            if let Some(cap) = &spec.power {
                self.caps.push((ci, cap.clone()));
            }
        }
    }

    fn die(&mut self, d: &Die, cx: &Cx) {
        for (inst, spec) in self.replicate(cx, d.id.as_str(), &d.rep, d) {
            let key = join(&cx.key, d.id.as_str());
            let (node, ci) = self.container(cx.scope, &key, d.id.as_str(), &inst, ContainerKind::Die);
            let tech = spec.tech.as_ref().unwrap_or(&self.doc.tech).node().to_owned();
            self.m.tree[ci].tech = Some(tech);
            self.m.tree[ci].die = Some(spec.clone());
            let mut cx2 = self.sub(cx, node, ci, d.id.as_str());
            cx2.die_clocks = IndexMap::new();
            for c in &spec.clocks {
                cx2.die_clocks.insert(c.id.to_string(), self.m.clocks.len());
                self.m.clocks.push(ClockInst { path: join(&self.m.nodes[node].path, c.id.as_str()), spec: c.clone() });
            }
            cx2.clock = self.clock(&cx2, spec.default_clock.as_deref(), &key);
            self.m.tree[ci].clock = cx2.clock;
            self.contents(&spec.contents, &cx2);
            for p in &spec.ports {
                for (inst, pspec) in self.replicate(&cx2, p.id.as_str(), &p.rep, p) {
                    let pi = self.m.ports.len();
                    let pnode = self.add_node(Some(node), &join(&cx2.key, p.id.as_str()), p.id.as_str(), &inst, NodeIx::Port(pi));
                    self.m.ports.push(PortInst { node: pnode, spec: pspec, container: ci });
                }
            }
        }
    }

    fn contents(&mut self, c: &Contents, cx: &Cx) {
        self.scopes.push(c.memories.iter().map(|m| (m.id.to_string(), Gran::Mem { banks: m.banks })).collect());
        for cl in &c.clusters {
            self.cluster(cl, cx);
        }
        for u in &c.units {
            self.unit(u, cx);
        }
        for m in &c.memories {
            self.memory(m, cx);
        }
        for n in &c.networks {
            self.network(n, cx);
        }
        for b in &c.blocks {
            self.block(b, cx, false);
        }
        self.scopes.pop();
    }

    fn cluster(&mut self, cl: &Cluster, cx: &Cx) {
        for (inst, spec) in self.replicate(cx, cl.id.as_str(), &cl.rep, cl) {
            let key = join(&cx.key, cl.id.as_str());
            let (node, ci) = self.container(cx.scope, &key, cl.id.as_str(), &inst, ContainerKind::Cluster);
            let mut cx2 = self.sub(cx, node, ci, cl.id.as_str());
            cx2.clock = self.clock(cx, spec.clock.as_deref(), &key);
            self.m.tree[ci].clock = cx2.clock;
            self.contents(&spec.contents, &cx2);
        }
    }

    fn near_granules(&self, near: &NearBinding) -> Option<u64> {
        let segs = super::select::split_segments(&near.memory);
        let last = segs.last().copied().unwrap_or_default();
        let name: String = last.split(['[', '{']).next().unwrap_or(last).to_owned();
        let gran = self.scopes.iter().rev().flat_map(|s| s.iter()).find(|(id, _)| *id == name)?.1;
        Some(gran.granules(near.granularity))
    }

    fn unit(&mut self, u: &ComputeUnit, cx: &Cx) {
        let key = join(&cx.key, u.id.as_str());
        let mut rep = u.rep.clone();
        if let Some(near) = &u.near
            && let Some(g) = self.near_granules(near)
            && rep.count.is_none()
            && matches!(rep.layout, Layout::Linear)
        {
            let Ok(g) = u32::try_from(g) else {
                self.d.push_inst(
                    "budget",
                    Diagnostic::error("E-IR-0210", format!("'{key}' implies {g} near units, beyond a u32 count")).at(&key),
                );
                return;
            };
            rep.count = Some(g);
        }
        for (inst, spec) in self.replicate(cx, u.id.as_str(), &rep, u) {
            let clock = self.clock(cx, spec.clock.as_deref(), &key);
            let ui = self.m.units.len();
            let node = self.add_node(Some(cx.scope), &key, u.id.as_str(), &inst, NodeIx::Unit(ui));
            let ops = spec.ops.clone().unwrap_or_else(|| spec.kind.default_ops());
            let mut local = vec![];
            for lb in &spec.local {
                let mi = self.m.memories.len();
                let li = Inst { id: lb.id.to_string(), index: 0, count: 1, coord: vec![] };
                let lnode = self.add_node(Some(node), &join(&key, lb.id.as_str()), lb.id.as_str(), &li, NodeIx::Mem(mi));
                self.m.memories.push(MemInst {
                    node: lnode,
                    spec: MemSpec::Local { buffer: lb.clone(), unit: ui },
                    capacity: lb.capacity,
                    clock,
                    container: cx.cont,
                    bandwidth: None,
                    bandwidth_derived: None,
                    backing: vec![],
                });
                local.push(mi);
            }
            self.m.units.push(UnitInst {
                node,
                spec,
                ops,
                feeds: BTreeMap::new(),
                local,
                near: None,
                clock,
                container: cx.cont,
            });
        }
    }

    fn memory(&mut self, m: &Memory, cx: &Cx) {
        let key = join(&cx.key, m.id.as_str());
        for (inst, spec) in self.replicate(cx, m.id.as_str(), &m.rep, m) {
            let clock = self.clock(cx, spec.clock.as_deref(), &key);
            let mi = self.m.memories.len();
            let node = self.add_node(Some(cx.scope), &key, m.id.as_str(), &inst, NodeIx::Mem(mi));
            let derived = self.m.clock_hz(clock).map(|f| BytesPerSec(spec.port_bits_per_cycle() as f64 * f.0 / 8.0));
            self.m.memories.push(MemInst {
                node,
                capacity: spec.capacity,
                clock,
                container: cx.cont,
                bandwidth: spec.overrides.bandwidth.or(derived),
                bandwidth_derived: derived,
                backing: vec![],
                spec: MemSpec::OnChip(spec),
            });
        }
    }

    fn mem_stack(&mut self, s: &MemStack, cx: &Cx) {
        let key = join(&cx.key, s.id.as_str());
        for (inst, spec) in self.replicate(cx, s.id.as_str(), &s.rep, s) {
            let clock = spec.clock.as_deref().and_then(|r| self.clock(cx, Some(r), &key));
            let mi = self.m.memories.len();
            let node = self.add_node(Some(cx.scope), &key, s.id.as_str(), &inst, NodeIx::Mem(mi));
            self.m.memories.push(MemInst {
                node,
                capacity: spec.capacity,
                clock,
                container: cx.cont,
                bandwidth: Some(spec.bandwidth()),
                bandwidth_derived: Some(spec.derived_bandwidth()),
                backing: vec![],
                spec: MemSpec::Stack(spec.clone()),
            });
            if let Some(LogicDie { contents, .. }) = &spec.logic_die {
                let cx2 = Cx { scope: node, cont: cx.cont, die_clocks: IndexMap::new(), clock: None, key: key.clone() };
                self.contents(contents, &cx2);
            }
        }
    }

    fn network(&mut self, n: &Network, cx: &Cx) {
        let key = join(&cx.key, n.id.as_str());
        for (inst, spec) in self.replicate(cx, n.id.as_str(), &n.rep, n) {
            let clock = self.clock(cx, spec.clock.as_deref(), &key);
            let ni = self.m.networks.len();
            let node = self.add_node(Some(cx.scope), &key, n.id.as_str(), &inst, NodeIx::Net(ni));
            self.m.networks.push(NetInst { node, spec, endpoints: vec![], routers: vec![], clock, direct: false });
        }
    }

    fn block(&mut self, b: &Block, cx: &Cx, synthesized: bool) -> Vec<usize> {
        let key = join(&cx.key, b.id.as_str());
        let mut out = vec![];
        for (inst, spec) in self.replicate(cx, b.id.as_str(), &b.rep, b) {
            let clock = self.clock(cx, spec.clock.as_deref(), &key);
            let bi = self.m.blocks.len();
            let node = self.add_node(Some(cx.scope), &key, b.id.as_str(), &inst, NodeIx::Block(bi));
            self.m.blocks.push(BlockInst { node, spec, container: cx.cont, clock, synthesized });
            out.push(bi);
        }
        out
    }

    // ---- references -------------------------------------------------------------------------------------

    fn vars(&self, n: usize) -> Vars {
        let node = &self.m.nodes[n];
        let own = InstVars { i: node.index, n: node.count, coord: node.coord.clone() };
        let mut up = vec![];
        let mut p = node.parent;
        while let Some(pn) = p {
            let a = &self.m.nodes[pn];
            if a.count > 1 {
                up.push(InstVars { i: a.index, n: a.count, coord: a.coord.clone() });
            }
            p = a.parent;
        }
        Vars { own, up }
    }

    fn resolve(&mut self, scope: usize, vars: &Vars, s: &str, at: &str, none_code: &str) -> Vec<usize> {
        let r = interpolate(s, vars).and_then(|s2| Selector::parse(&s2).map(|sel| (s2, sel)));
        let (s2, sel) = match r {
            Ok(x) => x,
            Err(e) => {
                self.d.push_inst(at, e.at(at));
                return vec![];
            }
        };
        let found = resolve_in(&self.m.nodes, scope, &sel);
        if found.is_empty() {
            let hint = self.near_paths(scope, &sel);
            self.d.push_inst(
                &format!("{at}#{s}"),
                Diagnostic::error(none_code, format!("{s2:?} matches nothing")).at(at).hint(hint),
            );
        }
        found
    }

    fn near_paths(&self, scope: usize, sel: &Selector) -> String {
        let first = match sel.segs.first() {
            Some(Seg::Pat { name, .. }) => name.trim_end_matches('*').to_owned(),
            _ => String::new(),
        };
        let mut cands: Vec<String> = vec![];
        let mut s = Some(scope);
        while let Some(cur) = s {
            for &c in &self.m.nodes[cur].children {
                let n = &self.m.nodes[c];
                if !cands.contains(&n.entity_id) && (first.is_empty() || n.entity_id.contains(&first[..first.len().min(2)])) {
                    cands.push(n.entity_id.clone());
                }
            }
            s = self.m.nodes[cur].parent;
        }
        cands.truncate(8);
        format!("visible ids from this scope include {cands:?}; use ^. or an absolute /path if needed")
    }

    fn resolve_one(&mut self, scope: usize, vars: &Vars, s: &str, at: &str) -> Option<usize> {
        let r = self.resolve(scope, vars, s, at, "E-IR-0206");
        match r.as_slice() {
            [one] => Some(*one),
            [] => None,
            many => {
                let cands: Vec<&str> = many.iter().take(4).map(|&n| self.m.nodes[n].path.as_str()).collect();
                self.d.push_inst(
                    &format!("{at}#{s}"),
                    Diagnostic::error("E-IR-0207", format!("reference {s:?} is ambiguous ({} matches: {cands:?}...)", many.len()))
                        .at(at)
                        .hint("use an indexed selector like name[{i}], '^.' or an absolute /path"),
                );
                None
            }
        }
    }

    fn parent(&self, n: usize) -> usize {
        self.m.nodes[n].parent.unwrap_or(0)
    }

    fn disable(&mut self, n: usize) {
        self.m.nodes[n].enabled = false;
        for c in self.m.nodes[n].children.clone() {
            self.disable(c);
        }
    }

    // ---- graph ------------------------------------------------------------------------------------------

    fn graph(&mut self) {
        for (scope, sel, key) in std::mem::take(&mut self.disabled) {
            let parsed = Selector::parse(&sel);
            let found = self.resolve(scope, &Vars::default(), &sel, &key, "E-IR-0205");
            if found.is_empty()
                && let Ok(p) = parsed
                && let Some(Seg::Pat { index: Some(ax), .. }) = p.segs.first()
            {
                let max = max_index(ax);
                self.d.push(Diagnostic::error("E-IR-0211", format!("disabled {sel:?}: index {max:?} out of range")).at(&key));
            }
            for n in found {
                self.disable(n);
            }
        }
        for (pkg, e) in std::mem::take(&mut self.exports) {
            let at = join(&self.m.nodes[pkg].path, e.id.as_str());
            if let Some(t) = self.resolve_one(pkg, &Vars::default(), &e.from, &at) {
                let ix = self.m.nodes[t].ix;
                let inst = Inst { id: e.id.to_string(), index: 0, count: 1, coord: vec![] };
                let key = self.m.nodes[pkg].entity.clone();
                self.add_node(Some(pkg), &join(&key, e.id.as_str()), e.id.as_str(), &inst, ix);
            }
        }
        self.auto_endpoints();
        self.stacks();
        for ni in 0..self.m.networks.len() {
            if self.m.nodes[self.m.networks[ni].node].enabled {
                self.build_net(ni);
            }
        }
        self.feeds();
        self.explicit_links();
        self.out_edges();
        self.backing();
        self.out_edges();
        self.address_and_power();
        self.levels();
    }

    fn net_of(&mut self, from: usize, r: &str, at: &str) -> Option<NetIx> {
        let scope = self.parent(from);
        let vars = self.vars(from);
        let n = self.resolve_one(scope, &vars, r, at)?;
        match self.m.nodes[n].ix {
            NodeIx::Net(ni) => Some(ni),
            _ => {
                self.d.push_inst(at, Diagnostic::error("E-IR-0206", format!("{r:?} is not a network")).at(at));
                None
            }
        }
    }

    fn auto_endpoints(&mut self) {
        for pi in 0..self.m.ports.len() {
            let node = self.m.ports[pi].node;
            if !self.m.nodes[node].enabled {
                continue;
            }
            let at = self.m.nodes[node].path.clone();
            let spec = self.m.ports[pi].spec.clone();
            let scope = self.parent(node);
            let vars = self.vars(node);
            let Some(t) = self.resolve_one(scope, &vars, &spec.internal, &at) else { continue };
            match self.m.nodes[t].ix {
                NodeIx::Net(ni) => self.auto_eps.entry(ni).or_default().push(node),
                target @ (NodeIx::Mem(_) | NodeIx::Block(_)) => {
                    let l = spec.link.clone().unwrap_or_default();
                    self.link(NodeIx::Port(pi), target, &l, ChannelKind::MemPort, None, None, None);
                }
                _ => self.d.push(
                    Diagnostic::error("E-IR-0701", format!("port internal {:?} is not a network, memory or block", spec.internal))
                        .at(&at),
                ),
            }
        }
        for bi in 0..self.m.blocks.len() {
            let node = self.m.blocks[bi].node;
            if !self.m.nodes[node].enabled {
                continue;
            }
            let at = self.m.nodes[node].path.clone();
            let nets = match &self.m.blocks[bi].spec.kind {
                BlockKind::MemController(mc) => mc.endpoint_of.clone(),
                BlockKind::Dma(d) => d.endpoint_of.clone(),
                _ => continue,
            };
            for r in nets {
                if let Some(ni) = self.net_of(node, &r, &at) {
                    self.auto_eps.entry(ni).or_default().push(node);
                }
            }
        }
    }

    fn synth_block(&mut self, scope: usize, cont: ContIx, id: String, kind: BlockKind) -> (usize, BlockIx) {
        let bi = self.m.blocks.len();
        let key = join(&self.m.nodes[scope].entity, &id);
        let inst = Inst { id: id.clone(), index: 0, count: 1, coord: vec![] };
        let node = self.add_node(Some(scope), &key, &id, &inst, NodeIx::Block(bi));
        let clock = self.m.tree[cont].clock;
        let spec = Block {
            id: crate::common::Id::new(id).expect("synthesized ids are valid"),
            rep: Replication::default(),
            kind,
            footprint: None,
            placement: Placement::Auto,
            clock: None,
            power: PowerOverride::default(),
        };
        self.m.blocks.push(BlockInst { node, spec: Arc::new(spec), container: cont, clock, synthesized: true });
        (node, bi)
    }

    fn stacks(&mut self) {
        let mut serves: BTreeMap<MemIx, Vec<BlockIx>> = BTreeMap::new();
        for bi in 0..self.m.blocks.len() {
            let node = self.m.blocks[bi].node;
            if let BlockKind::MemController(MemControllerSpec { serves: s, .. }) = &self.m.blocks[bi].spec.kind {
                let s = s.clone();
                let at = self.m.nodes[node].path.clone();
                let (scope, vars) = (self.parent(node), self.vars(node));
                if let Some(t) = self.resolve_one(scope, &vars, &s, &at)
                    && let NodeIx::Mem(mi) = self.m.nodes[t].ix
                {
                    serves.entry(mi).or_default().push(bi);
                }
            }
        }
        for mi in 0..self.m.memories.len() {
            let MemSpec::Stack(spec) = &self.m.memories[mi].spec else { continue };
            let spec = spec.clone();
            let node = self.m.memories[mi].node;
            if !self.m.nodes[node].enabled {
                continue;
            }
            let at = self.m.nodes[node].path.clone();
            let (scope, vars) = (self.parent(node), self.vars(node));
            let bw = spec.bandwidth();
            let stack_link = LinkSpec {
                width_bits: Some(spec.io_width_bits),
                bandwidth: Some(bw),
                ..LinkSpec::default()
            };
            let stack_ix = NodeIx::Mem(mi);
            let inst_id = self.m.nodes[node].inst_id.clone();
            let first = self.m.channels.len();
            match &spec.attach {
                Some(StackAttach::Phys { phys }) => {
                    let mut phy_nodes = vec![];
                    for r in phys {
                        if let Some(p) = self.resolve_one(scope, &vars, r, &at) {
                            phy_nodes.push(p);
                        }
                    }
                    for &p in &phy_nodes {
                        let pix = self.m.nodes[p].ix;
                        self.link(stack_ix, pix, &stack_link, ChannelKind::MemPort, None, None, None);
                        let pparent = self.parent(p);
                        for &bi in serves.get(&mi).into_iter().flatten() {
                            if self.parent(self.m.blocks[bi].node) == pparent || phy_nodes.len() == 1 {
                                self.link(NodeIx::Block(bi), pix, &stack_link, ChannelKind::MemPort, None, None, None);
                            }
                        }
                    }
                }
                Some(StackAttach::Controllers { controllers }) => {
                    for r in controllers {
                        let Some(c) = self.resolve_one(scope, &vars, r, &at) else { continue };
                        let NodeIx::Block(bi) = self.m.nodes[c].ix else { continue };
                        let cont = self.m.blocks[bi].container;
                        let (_, phy) = self.synth_block(
                            self.parent(c),
                            cont,
                            format!("{inst_id}_phy"),
                            BlockKind::Phy(PhySpec { for_kind: IoKind::Hbm, lanes: Some(spec.io_width_bits), shoreline: None, site: None }),
                        );
                        self.link(stack_ix, NodeIx::Block(phy), &stack_link, ChannelKind::MemPort, None, None, None);
                        self.link(NodeIx::Block(bi), NodeIx::Block(phy), &stack_link, ChannelKind::MemPort, None, None, None);
                    }
                }
                Some(StackAttach::Network { network }) => {
                    let Some(n) = self.resolve_one(scope, &vars, network, &at) else { continue };
                    let NodeIx::Net(ni) = self.m.nodes[n].ix else {
                        self.d.push(Diagnostic::error("E-IR-0206", format!("attach {network:?} is not a network")).at(&at));
                        continue;
                    };
                    let host = self.parent(n);
                    let cont = self.cont_of(host);
                    let (mc_node, mc) = self.synth_block(
                        host,
                        cont,
                        format!("{inst_id}_mc"),
                        BlockKind::MemController(MemControllerSpec {
                            serves: format!("/{}", self.m.nodes[node].path),
                            channels: None,
                            width_bits: None,
                            queue_depth: None,
                            scheduler: Default::default(),
                            endpoint_of: vec![format!("/{}", self.m.nodes[n].path)],
                        }),
                    );
                    let (_, phy) = self.synth_block(
                        host,
                        cont,
                        format!("{inst_id}_phy"),
                        BlockKind::Phy(PhySpec { for_kind: IoKind::Hbm, lanes: Some(spec.io_width_bits), shoreline: None, site: None }),
                    );
                    self.auto_eps.entry(ni).or_default().push(mc_node);
                    self.link(stack_ix, NodeIx::Block(phy), &stack_link, ChannelKind::MemPort, None, None, None);
                    self.link(NodeIx::Block(mc), NodeIx::Block(phy), &stack_link, ChannelKind::MemPort, None, None, None);
                }
                Some(StackAttach::Vertical { die }) => {
                    if let Some(d) = self.resolve_one(scope, &vars, die, &at) {
                        let dix = self.m.nodes[d].ix;
                        self.link(stack_ix, dix, &stack_link, ChannelKind::Vertical, None, None, None);
                    }
                }
                None => {}
            }
            for c in &mut self.m.channels[first..] {
                c.bandwidth_derived = Some(spec.derived_bandwidth());
            }
        }
    }

    fn cont_of(&self, mut n: usize) -> ContIx {
        loop {
            if let NodeIx::Container(c) = self.m.nodes[n].ix {
                return c;
            }
            n = self.parent(n);
        }
    }

    fn res(&mut self, kind: ResKind, capacity: Option<BytesPerSec>) -> ResIx {
        self.m.resources.push(SharedResource { kind, capacity });
        self.m.resources.len() - 1
    }

    #[allow(clippy::too_many_arguments)]
    fn chan(
        &mut self,
        src: NodeIx,
        dst: NodeIx,
        kind: ChannelKind,
        width_bits: Option<u32>,
        clock: Option<ClockIx>,
        resource: ResIx,
        network: Option<NetIx>,
        bandwidth: Option<BytesPerSec>,
    ) -> ChanIx {
        self.m.channels.push(Channel { src, dst, kind, width_bits, clock, resource, network, bandwidth, bandwidth_derived: bandwidth });
        self.m.channels.len() - 1
    }

    /// A link's own clock, resolved from its network, else from the innermost node enclosing both endpoints.
    fn link_clock(&mut self, r: &str, a: NodeIx, b: NodeIx, net: Option<NetIx>) -> Option<ClockIx> {
        let scope = match net {
            Some(ni) => self.m.nodes[self.m.networks[ni].node].path.clone(),
            None => {
                let up = |mut n: usize| {
                    let mut v = vec![n];
                    while let Some(p) = self.m.nodes[n].parent {
                        v.push(p);
                        n = p;
                    }
                    v
                };
                let (ua, ub) = (up(self.m.node_of(a)), up(self.m.node_of(b)));
                ua.into_iter().find(|n| ub.contains(n)).map(|n| self.m.nodes[n].path.clone()).unwrap_or_default()
            }
        };
        let found = self.m.link_clock(r, &scope);
        if found.is_none() {
            self.d.push(Diagnostic::error("E-IR-0901", format!("unknown link clock '{r}'")).at(scope));
        }
        found
    }

    /// One bidirectional link (`count` parallel copies); `shared` puts every direction on one resource (bus).
    #[allow(clippy::too_many_arguments)]
    fn link(
        &mut self,
        a: NodeIx,
        b: NodeIx,
        l: &LinkSpec,
        default_kind: ChannelKind,
        clock: Option<ClockIx>,
        net: Option<NetIx>,
        shared: Option<ResIx>,
    ) -> Vec<ChanIx> {
        self.link_dir(a, b, l, default_kind, clock, net, shared, true)
    }

    /// [`Self::link`], or only its `a -> b` channels when `both` is false.
    #[allow(clippy::too_many_arguments)]
    fn link_dir(
        &mut self,
        a: NodeIx,
        b: NodeIx,
        l: &LinkSpec,
        default_kind: ChannelKind,
        clock: Option<ClockIx>,
        net: Option<NetIx>,
        shared: Option<ResIx>,
        both: bool,
    ) -> Vec<ChanIx> {
        let kind = match l.phys {
            _ if !matches!(default_kind, ChannelKind::NocHop | ChannelKind::Bus) => default_kind,
            LinkPhys::OnDie { .. } => default_kind,
            LinkPhys::D2d(_) => ChannelKind::D2d,
            LinkPhys::Serdes(_) => ChannelKind::Serdes,
            LinkPhys::Vertical(_) => ChannelKind::Vertical,
            LinkPhys::Optical(_) => ChannelKind::Optical,
        };
        let clock = match &l.clock {
            Some(r) => self.link_clock(r, a, b, net),
            None => clock,
        };
        let bw = l.effective_bandwidth(self.m.clock_hz(clock));
        let mut out = vec![];
        for _ in 0..l.count.max(1) {
            let (r1, r2) = match shared {
                Some(r) => (r, r),
                None if l.full_duplex && both => (self.res(ResKind::LinkDir, bw), self.res(ResKind::LinkDir, bw)),
                None => {
                    let r = self.res(ResKind::LinkDir, bw);
                    (r, r)
                }
            };
            out.push(self.chan(a, b, kind, l.width_bits, clock, r1, net, bw));
            if both {
                out.push(self.chan(b, a, kind, l.width_bits, clock, r2, net, bw));
            }
        }
        let derived = l.derived_bandwidth(self.m.clock_hz(clock));
        for &c in &out {
            self.m.channels[c].bandwidth_derived = derived;
        }
        out
    }

    fn router(&mut self, ni: NetIx, name: String, coord: Vec<u32>) -> NodeIx {
        let ri = self.m.routers.len();
        let nnode = self.m.networks[ni].node;
        let key = join(&self.m.nodes[nnode].entity, &name);
        let inst = Inst { id: name.clone(), index: self.m.networks[ni].routers.len() as u32, count: 1, coord: coord.clone() };
        let node = self.add_node(Some(nnode), &key, &name, &inst, NodeIx::Router(ri));
        self.m.routers.push(RouterInst { node, net: ni, coord });
        self.m.networks[ni].routers.push(ri);
        NodeIx::Router(ri)
    }

    fn layout_coord(&self, mut n: usize) -> Vec<u32> {
        loop {
            if !self.m.nodes[n].coord.is_empty() {
                return self.m.nodes[n].coord.clone();
            }
            match self.m.nodes[n].parent {
                Some(p) => n = p,
                None => return vec![self.m.nodes[n].index],
            }
        }
    }

    fn build_net(&mut self, ni: NetIx) {
        let spec = self.m.networks[ni].spec.clone();
        let nnode = self.m.networks[ni].node;
        let at = self.m.nodes[nnode].path.clone();
        let scope = self.parent(nnode);
        let vars = self.vars(nnode);
        let clock = self.m.networks[ni].clock;

        struct Group {
            nodes: Vec<usize>,
            at: RouterBinding,
            port: LinkSpec,
            mult: u32,
            ports: Option<String>,
        }
        let mut groups = vec![];
        for EndpointBinding { select, at: binding, port, multiplicity, ports } in &spec.endpoints {
            let found: Vec<usize> = self
                .resolve(scope, &vars, select, &at, "E-IR-0205")
                .into_iter()
                .filter(|&n| self.m.nodes[n].enabled)
                .collect();
            groups.push(Group {
                nodes: found,
                at: binding.clone(),
                port: port.clone().unwrap_or_else(|| spec.link.clone()),
                mult: multiplicity.unwrap_or(1),
                ports: ports.clone(),
            });
        }
        if let Some(mut auto) = self.auto_eps.get(&ni).cloned() {
            auto.retain(|n| self.m.nodes[*n].enabled && !groups.iter().any(|g| g.nodes.contains(n)));
            groups.push(Group { nodes: auto, at: RouterBinding::Auto, port: spec.link.clone(), mult: 1, ports: None });
        }
        let mut seen = std::collections::BTreeSet::new();
        for g in &mut groups {
            g.nodes.retain(|&n| {
                let ok = seen.insert(n);
                if !ok {
                    self.d.push_inst(
                        &at,
                        Diagnostic::error("E-IR-0702", format!("'{}' bound twice to network '{at}'", self.m.nodes[n].path))
                            .at(&at)
                            .hint("bind it once and use `multiplicity` for several ports"),
                    );
                }
                ok
            });
            for &n in &g.nodes {
                let ok = match self.m.nodes[n].ix {
                    NodeIx::Mem(_) | NodeIx::Unit(_) | NodeIx::Block(_) | NodeIx::Port(_) => true,
                    NodeIx::Container(c) => {
                        matches!(self.m.tree[c].kind, ContainerKind::Package | ContainerKind::Die | ContainerKind::Switch | ContainerKind::Host)
                    }
                    _ => false,
                };
                if !ok {
                    self.d.push_inst(
                        &at,
                        Diagnostic::error("E-IR-0701", format!("endpoint '{}' is not connectable", self.m.nodes[n].path))
                            .at(&at)
                            .hint("endpoints are memories, units (via feeds), blocks, ports, packages/dies (direct networks), switches, hosts"),
                    );
                }
            }
        }
        let direct = groups.iter().any(|g| g.ports.is_some());
        self.m.networks[ni].direct = direct;
        self.m.networks[ni].endpoints = groups.iter().flat_map(|g| g.nodes.iter().map(|&n| self.m.nodes[n].ix)).collect();
        let total: usize = groups.iter().map(|g| g.nodes.len()).sum();
        let too_small = |msg: String| Diagnostic::error("E-IR-0703", msg).at(&at);
        let endpoint_chans = groups
            .iter()
            .map(|g| (g.nodes.len() as u64).saturating_mul(u64::from(g.mult)).saturating_mul(u64::from(g.port.count.max(1))))
            .fold(0u64, u64::saturating_add);
        if !self.charge(synthesized(&spec, total as u64).saturating_add(endpoint_chans), &at) {
            return;
        }
        let chan0 = self.m.channels.len();

        let (dims, rlinks): (Vec<u32>, Vec<RouterLink>) = match &spec.topology {
            Topology::Bus { .. } | Topology::Crossbar { .. } => {
                let r = self.router(ni, "r0".into(), vec![]);
                let shared = matches!(spec.topology, Topology::Bus { .. })
                    .then(|| self.res(ResKind::Bus, spec.link.effective_bandwidth(self.m.clock_hz(clock))));
                let kind = if shared.is_some() { ChannelKind::Bus } else { ChannelKind::NocHop };
                for g in &groups {
                    for &n in &g.nodes {
                        for _ in 0..g.mult {
                            self.link(self.m.nodes[n].ix, r, &g.port, kind, clock, Some(ni), shared);
                        }
                    }
                }
                self.radix_check(ni, &spec, &at, chan0);
                return;
            }
            Topology::P2p => {
                let eps: Vec<usize> = groups.iter().flat_map(|g| g.nodes.clone()).collect();
                if eps.len() != 2 {
                    self.d.push(too_small(format!("p2p network joins exactly 2 instances, got {}", eps.len())));
                    return;
                }
                let (a, b) = (self.m.nodes[eps[0]].ix, self.m.nodes[eps[1]].ix);
                self.link(a, b, &spec.link, ChannelKind::NocHop, clock, Some(ni), None);
                return;
            }
            Topology::Star { center } => {
                let Some(c) = self.resolve_one(scope, &vars, center, &at) else { return };
                let cix = self.m.nodes[c].ix;
                for g in &groups {
                    for &n in &g.nodes {
                        self.link(self.m.nodes[n].ix, cix, &g.port, ChannelKind::NocHop, clock, Some(ni), None);
                    }
                }
                return;
            }
            Topology::Hierarchical { .. } => {
                self.d.push(
                    Diagnostic::warning("W-IR-0799", "hierarchical topology is not expanded into channels in M0")
                        .at(&at)
                        .hint("model each level as its own network sharing gateway endpoints"),
                );
                return;
            }
            Topology::Ring { rings, .. } => {
                let r = if direct { total } else { total.max(1) } as u32;
                let links = grid_links(&[r], &[true]);
                let l = LinkSpec { count: spec.link.count.saturating_mul(*rings), ..spec.link.clone() };
                (vec![r], links.into_iter().map(|(a, b, d)| (a, b, d, Some(l.clone()))).collect())
            }
            Topology::Mesh { dims } => (dims.clone(), grid_links(dims, &vec![false; dims.len()]).into_iter().map(|(a, b, d)| (a, b, d, None)).collect()),
            Topology::Torus { dims, wrap } => {
                let w = wrap.clone().unwrap_or_else(|| vec![true; dims.len()]);
                (dims.clone(), grid_links(dims, &w).into_iter().map(|(a, b, d)| (a, b, d, None)).collect())
            }
            Topology::Custom { routers, edges } => {
                let mut links = vec![];
                for e in edges {
                    if e.a >= *routers || e.b >= *routers {
                        self.d.push(
                            Diagnostic::error("E-IR-0705", format!("custom edge {}-{} references a router >= {routers}", e.a, e.b))
                                .at(&at),
                        );
                        continue;
                    }
                    for _ in 0..e.count.max(1) {
                        links.push((e.a as usize, e.b as usize, links.len(), e.link.clone()));
                    }
                }
                if !connected(*routers as usize, links.iter().map(|&(a, b, ..)| (a, b))) {
                    self.d.push(Diagnostic::error("E-IR-0705", "custom topology graph is disconnected").at(&at));
                }
                (vec![*routers], links)
            }
            Topology::Tree { arity, levels } | Topology::FatTree { arity, levels, .. } => {
                let fat = matches!(spec.topology, Topology::FatTree { .. });
                let (a, l) = (u64::from((*arity).max(1)), (*levels).max(1));
                let mut starts = vec![0u64];
                for lv in 0..l {
                    starts.push(starts[lv as usize] + a.pow(lv));
                }
                let total_r = starts[l as usize];
                let mut links = vec![];
                for lv in 1..l {
                    for j in 0..a.pow(lv) {
                        let child = (starts[lv as usize] + j) as usize;
                        let parent = (starts[lv as usize - 1] + j / a) as usize;
                        let cnt = if fat { a.pow(l - 1 - lv) as u32 } else { 1 };
                        let ls = LinkSpec { count: spec.link.count.saturating_mul(cnt), ..spec.link.clone() };
                        links.push((child, parent, lv as usize, Some(ls)));
                    }
                }
                (vec![total_r as u32], links)
            }
        };

        let n_routers: usize = dims.iter().map(|&d| d as usize).product();
        let to_linear = |c: &[i64]| -> Option<usize> {
            if c.len() != dims.len() {
                return None;
            }
            let mut idx = 0usize;
            for (d, &x) in c.iter().enumerate() {
                if x < 0 || x >= i64::from(dims[d]) {
                    return None;
                }
                idx = idx * dims[d] as usize + x as usize;
            }
            Some(idx)
        };
        let leaf_start = match &spec.topology {
            Topology::Tree { arity, levels } | Topology::FatTree { arity, levels, .. } => {
                n_routers - (u64::from((*arity).max(1)).pow((*levels).max(1) - 1) as usize)
            }
            _ => 0,
        };
        let mut placement: Vec<(usize, usize, u32, LinkSpec)> = vec![];
        let mut k_global = 0usize;
        for g in &groups {
            for (k, &n) in g.nodes.iter().enumerate() {
                let coord: Option<Vec<i64>> = match &g.at {
                    RouterBinding::Auto => {
                        let span = n_routers - leaf_start;
                        Some(vec![(leaf_start + k_global * span / total.max(1)) as i64])
                    }
                    RouterBinding::Layout => Some(self.layout_coord(n).iter().map(|&x| i64::from(x)).collect()),
                    RouterBinding::LayoutOffset(off) => Some(
                        self.layout_coord(n).iter().zip(off.iter().chain(std::iter::repeat(&0))).map(|(&x, &o)| i64::from(x) + i64::from(o)).collect(),
                    ),
                    RouterBinding::Index(list) => list.get(k).map(|c| c.iter().map(|&x| i64::from(x)).collect()),
                    RouterBinding::Concentrated { per_router } => Some(vec![(k / (*per_router).max(1) as usize) as i64]),
                    RouterBinding::Fixed { router } => Some(router.iter().map(|&x| i64::from(x)).collect()),
                };
                k_global += 1;
                let lin = coord.as_ref().and_then(|c| {
                    if c.len() == 1 && dims.len() > 1 { usize::try_from(c[0]).ok().filter(|&x| x < n_routers) } else { to_linear(c) }
                });
                match lin {
                    Some(r) => placement.push((n, r, g.mult, g.port.clone())),
                    None => self.d.push_inst(
                        &at,
                        too_small(format!(
                            "endpoint '{}' maps to router {coord:?}, outside {} {dims:?}",
                            self.m.nodes[n].path,
                            spec.topology.name()
                        ))
                        .hint("enlarge the topology or fix the endpoint binding"),
                    ),
                }
            }
        }

        if direct {
            self.direct_links(ni, &spec, &at, &groups.iter().filter(|g| g.ports.is_some()).flat_map(|g| g.nodes.iter().map(move |&n| (n, g.ports.clone().unwrap_or_default()))).collect::<Vec<_>>(), &placement, &rlinks, n_routers, clock);
            return;
        }
        let names: Vec<String> = (0..n_routers)
            .map(|r| {
                if dims.len() > 1 {
                    let c = delinearize(r, &dims);
                    format!("r{}", c.iter().map(u32::to_string).collect::<Vec<_>>().join("_"))
                } else {
                    format!("r{r}")
                }
            })
            .collect();
        let routers: Vec<NodeIx> = names
            .into_iter()
            .enumerate()
            .map(|(r, name)| {
                let coord = if dims.len() > 1 { delinearize(r, &dims) } else { vec![r as u32] };
                self.router(ni, name, coord)
            })
            .collect();
        let both = !matches!(spec.topology, Topology::Ring { bidirectional: false, .. });
        for (a, b, _, l) in &rlinks {
            let l = l.clone().unwrap_or_else(|| spec.link.clone());
            self.link_dir(routers[*a], routers[*b], &l, ChannelKind::NocHop, clock, Some(ni), None, both);
        }
        for (n, r, mult, port) in placement {
            for _ in 0..mult {
                self.link(self.m.nodes[n].ix, routers[r], &port, ChannelKind::NocHop, clock, Some(ni), None);
            }
        }
        self.radix_check(ni, &spec, &at, chan0);
    }

    /// A declared radix must cover every channel a router of `ni` drives (all created from `chan0` on): kiln-phys
    /// prices the declared radix, and derives it from exactly these channels when it is omitted.
    fn radix_check(&mut self, ni: NetIx, spec: &Network, at: &str, chan0: usize) {
        let Some(r) = spec.router.radix else { return };
        let mut out: BTreeMap<usize, usize> = BTreeMap::new();
        for c in &self.m.channels[chan0..] {
            if let NodeIx::Router(ri) = c.src
                && self.m.routers[ri].net == ni
            {
                *out.entry(ri).or_default() += 1;
            }
        }
        let degree = out.into_values().max().unwrap_or(0);
        if degree > r as usize {
            self.d.push(
                Diagnostic::error("E-IR-0704", format!("router radix {r} < required degree {degree}"))
                    .at(at)
                    .hint("raise router.radix, lower concentration, or omit radix to derive it"),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn direct_links(
        &mut self,
        ni: NetIx,
        spec: &Network,
        at: &str,
        entities: &[(usize, String)],
        placement: &[(usize, usize, u32, LinkSpec)],
        rlinks: &[RouterLink],
        n_routers: usize,
        clock: Option<ClockIx>,
    ) {
        let mut at_router: Vec<Option<usize>> = vec![None; n_routers];
        for &(n, r, ..) in placement {
            if at_router[r].replace(n).is_some() {
                self.d.push(Diagnostic::error("E-IR-0703", format!("two entities map to router {r}")).at(at));
            }
        }
        let mut slots: Vec<Vec<(usize, u8, usize)>> = vec![vec![]; n_routers];
        for (li, &(a, b, d, _)) in rlinks.iter().enumerate() {
            slots[a].push((d, 0, li));
            slots[b].push((d, 1, li));
        }
        for s in &mut slots {
            s.sort_unstable();
        }
        let mut port_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (r, ent) in at_router.iter().enumerate() {
            let Some(e) = *ent else { continue };
            let Some((_, sel)) = entities.iter().find(|(n, _)| *n == e) else { continue };
            let epath = self.m.nodes[e].path.clone();
            let ports: Vec<usize> = match Selector::parse(sel) {
                Ok(s) => resolve_in(&self.m.nodes, e, &Selector { start: super::select::Start::Up(0), ..s })
                    .into_iter()
                    .filter(|&p| self.m.nodes[p].enabled)
                    .collect(),
                Err(err) => {
                    self.d.push(err.at(at));
                    continue;
                }
            };
            if ports.len() < slots[r].len() {
                self.d.push_inst(
                    at,
                    Diagnostic::error(
                        "E-IR-0710",
                        format!("'{epath}' has {} '{sel}' ports; {} needs {}", ports.len(), spec.topology.name(), slots[r].len()),
                    )
                    .at(at)
                    .hint("add ports to the die or use a smaller topology"),
                );
                continue;
            }
            if ports.len() > slots[r].len() {
                self.d.push_inst(
                    &format!("{at}#0726"),
                    Diagnostic::warning("W-IR-0726", format!("'{epath}' has {} ports, {} used", ports.len(), slots[r].len())).at(at),
                );
            }
            for (k, &(_, side, li)) in slots[r].iter().enumerate() {
                port_of.insert((li, side as usize), ports[k]);
            }
        }
        for (li, (_, _, _, l)) in rlinks.iter().enumerate() {
            let (Some(&pa), Some(&pb)) = (port_of.get(&(li, 0)), port_of.get(&(li, 1))) else { continue };
            let mut l = l.clone().unwrap_or_else(|| spec.link.clone());
            l.count = 1;
            let (a, b) = (self.m.nodes[pa].ix, self.m.nodes[pb].ix);
            let both = !matches!(spec.topology, Topology::Ring { bidirectional: false, .. });
            self.link_dir(a, b, &l, ChannelKind::NocHop, clock, Some(ni), None, both);
        }
    }

    fn feeds(&mut self) {
        for ui in 0..self.m.units.len() {
            let node = self.m.units[ui].node;
            if !self.m.nodes[node].enabled {
                continue;
            }
            let spec = self.m.units[ui].spec.clone();
            let at = self.m.nodes[node].path.clone();
            let key = self.m.nodes[node].entity.clone();
            let (scope, vars) = (self.parent(node), self.vars(node));
            let clock = self.m.units[ui].clock;
            let f = self.m.clock_hz(clock);
            for (role, feed) in &spec.feeds {
                let Some(t) = self.resolve_one(scope, &vars, &feed.from, &at) else { continue };
                let NodeIx::Mem(mi) = self.m.nodes[t].ix else {
                    self.d.push_inst(
                        &key,
                        Diagnostic::error("E-IR-0305", format!("feed '{role:?}' from {:?} is not a memory", feed.from)).at(&at),
                    );
                    continue;
                };
                let via = match &feed.via {
                    Some(v) => self.net_of(node, v, &at),
                    None => None,
                };
                let (mut read, mut write) = (None, None);
                if feed.via.is_none() {
                    let widest = match &self.m.memories[mi].spec {
                        MemSpec::OnChip(m) => Some(m.widest_port_bits()),
                        MemSpec::Stack(s) => Some(s.io_width_bits),
                        MemSpec::Local { .. } => None,
                    };
                    let width = match (feed.width_bits, widest) {
                        (Some(w), Some(p)) => Some(w.min(p)),
                        (w, p) => w.or(p),
                    };
                    let bw = width.zip(f).map(|(w, f)| BytesPerSec(f64::from(w) * f.0 / 8.0));
                    let r = self.res(ResKind::Port, bw);
                    read = Some(self.chan(NodeIx::Mem(mi), NodeIx::Unit(ui), ChannelKind::Feed, width, clock, r, None, bw));
                    if role.is_output() {
                        let w = self.res(ResKind::Port, bw);
                        write = Some(self.chan(NodeIx::Unit(ui), NodeIx::Mem(mi), ChannelKind::Feed, width, clock, w, None, bw));
                    }
                }
                self.m.units[ui].feeds.insert(*role, FeedInst { mem: mi, read, write, via });
            }
            if let Some(near) = &spec.near
                && let Some(t) = self.resolve_one(scope, &vars, &near.memory, &at)
            {
                match self.m.nodes[t].ix {
                    NodeIx::Mem(mi) => {
                        let bw = near.internal_bandwidth;
                        let r = self.res(ResKind::MemBankGroup, bw);
                        let rc = self.chan(NodeIx::Mem(mi), NodeIx::Unit(ui), ChannelKind::Near, None, clock, r, None, bw);
                        let w = self.res(ResKind::MemBankGroup, bw);
                        let wc = self.chan(NodeIx::Unit(ui), NodeIx::Mem(mi), ChannelKind::Near, None, clock, w, None, bw);
                        for c in [rc, wc] {
                            self.m.channels[c].bandwidth_derived = None;
                        }
                        let g = Gran::of(&self.m.memories[mi].spec).granules(near.granularity);
                        let (count, slice) = (self.m.nodes[node].count, self.m.nodes[node].index);
                        if u64::from(count) != g {
                            self.d.push_inst(
                                &key,
                                Diagnostic::error("E-IR-0607", format!("'{key}' count {count} != {g} granules of its bound memory"))
                                    .at(&at)
                                    .hint("one instance per bound granule: omit count and layout when binding a memory template by id"),
                            );
                        }
                        self.m.units[ui].near = Some(NearInst { mem: mi, slice });
                    }
                    _ => self.d.push_inst(
                        &key,
                        Diagnostic::error("E-IR-0601", format!("near.memory {:?} is not a memory or mem stack", near.memory))
                            .at(&at),
                    ),
                }
            }
            for li in self.m.units[ui].local.clone() {
                let MemSpec::Local { buffer, .. } = &self.m.memories[li].spec else { continue };
                if let Some(r) = buffer.refill_from.clone()
                    && let Some(t) = self.resolve_one(scope, &vars, &r, &at)
                    && let NodeIx::Mem(src) = self.m.nodes[t].ix
                {
                    self.m.memories[li].backing.push(src);
                }
            }
        }
    }

    fn explicit_links(&mut self) {
        for (pkg, l) in std::mem::take(&mut self.links) {
            let key = join(&self.m.nodes[pkg].entity, l.id.as_str());
            let n = l.rep.instances();
            let base = self.vars(pkg);
            let specs = self.vary(l.id.as_str(), &instance_ids(l.id.as_str(), &l.rep), &l.rep, &*l, &key);
            for (i, l) in (0..n as u32).zip(specs) {
                let vars = Vars {
                    own: InstVars { i, n: n as u32, coord: vec![] },
                    up: [InstVars { i: self.m.nodes[pkg].index, n: self.m.nodes[pkg].count, coord: self.m.nodes[pkg].coord.clone() }]
                        .into_iter()
                        .chain(base.up.clone())
                        .collect(),
                };
                let a = self.resolve_one(pkg, &vars, &l.a, &key);
                let b = self.resolve_one(pkg, &vars, &l.b, &key);
                if let (Some(a), Some(b)) = (a, b) {
                    let (ax, bx) = (self.m.nodes[a].ix, self.m.nodes[b].ix);
                    self.link(ax, bx, &l.link, ChannelKind::NocHop, None, None, None);
                }
            }
        }
        for (board, hl) in std::mem::take(&mut self.host_links) {
            let at = self.m.nodes[board].path.clone();
            let vars = self.vars(board);
            let Some(h) = self.resolve_one(board, &vars, &hl.host, &at) else { continue };
            let targets = self.resolve(board, &vars, &hl.to, &at, "E-IR-0205");
            let hix = self.m.nodes[h].ix;
            for t in targets {
                if self.m.nodes[t].enabled {
                    let tix = self.m.nodes[t].ix;
                    self.link(hix, tix, &hl.link, ChannelKind::Host, None, None, None);
                }
            }
        }
    }

    fn out_edges(&mut self) {
        let mut out = vec![vec![]; self.m.nodes.len()];
        for (c, ch) in self.m.channels.iter().enumerate() {
            out[self.m.node_of(ch.src)].push(c);
        }
        self.m.out_edges = out;
    }

    fn backing(&mut self) {
        for mi in 0..self.m.memories.len() {
            let MemSpec::OnChip(spec) = &self.m.memories[mi].spec else { continue };
            let Some(sel) = spec.backing.clone() else { continue };
            let node = self.m.memories[mi].node;
            if !self.m.nodes[node].enabled {
                continue;
            }
            let at = self.m.nodes[node].path.clone();
            let (scope, vars) = (self.parent(node), self.vars(node));
            let targets: Vec<MemIx> = self
                .resolve(scope, &vars, &sel, &at, "E-IR-0205")
                .into_iter()
                .filter_map(|t| match self.m.nodes[t].ix {
                    NodeIx::Mem(m) => Some(m),
                    _ => None,
                })
                .collect();
            let reach = self.m.reachable_mems(&[node]);
            for &t in &targets {
                if !reach[self.m.memories[t].node] {
                    let l = LinkSpec::default();
                    let clock = self.m.memories[mi].clock;
                    self.link(NodeIx::Mem(mi), NodeIx::Mem(t), &l, ChannelKind::MemPort, clock, None, None);
                }
            }
            self.m.memories[mi].backing = targets;
        }
    }

    fn address_and_power(&mut self) {
        for (pkg, am) in std::mem::take(&mut self.addr_maps) {
            let at = join(&self.m.nodes[pkg].path, am.id.as_str());
            let vars = self.vars(pkg);
            let targets: Vec<NodeIx> =
                self.resolve(pkg, &vars, &am.targets, &at, "E-IR-0205").into_iter().map(|n| self.m.nodes[n].ix).collect();
            self.m.address_maps.push(AddressMapInst { path: at, spec: am, targets });
        }
        for pd in &self.doc.power {
            let at = pd.id.to_string();
            let sel = format!("/{}", pd.members.trim_start_matches('/'));
            let members: Vec<ContIx> = self
                .resolve(0, &Vars::default(), &sel, &at, "E-IR-0205")
                .into_iter()
                .filter_map(|n| match self.m.nodes[n].ix {
                    NodeIx::Container(c) => Some(c),
                    _ => None,
                })
                .collect();
            let clocks = pd.clocks.iter().filter_map(|c| self.global_clocks.get(c.as_str()).copied()).collect();
            for c in pd.clocks.iter().filter(|c| !self.global_clocks.contains_key(c.as_str())) {
                self.d.push(Diagnostic::error("E-IR-0901", format!("power domain clock '{c}' is not a global clock")).at(&at));
            }
            self.m.power_domains.push(PowerDomainInst {
                path: at,
                cap: pd.cap,
                policy: pd.policy,
                members,
                clocks,
                idle: pd.idle,
                level: None,
                assumed: pd.assumed.clone(),
            });
        }
        for (ci, cap) in std::mem::take(&mut self.caps) {
            let path = join(&self.m.nodes[self.m.tree[ci].node].path, "power");
            let clocks = cap.clocks.iter().filter_map(|c| self.global_clocks.get(c.as_str()).copied()).collect();
            self.m.power_domains.push(PowerDomainInst {
                path,
                cap: cap.cap,
                policy: cap.policy,
                members: vec![ci],
                clocks,
                idle: cap.idle,
                level: cap.level,
                assumed: cap.assumed.clone(),
            });
        }
    }

    fn levels(&mut self) {
        let nm = self.m.memories.len();
        let mut lv = vec![u8::MAX; nm];
        let mut frontier = vec![];
        for (mi, m) in self.m.memories.iter().enumerate() {
            if m.is_local() {
                lv[mi] = 0;
            }
        }
        for u in &self.m.units {
            if !self.m.nodes[u.node].enabled {
                continue;
            }
            for f in u.feeds.values() {
                if !self.m.memories[f.mem].is_stack() && lv[f.mem] == u8::MAX {
                    lv[f.mem] = 1;
                    frontier.push(f.mem);
                }
            }
        }
        let mut level = 1u8;
        while !frontier.is_empty() && level < u8::MAX - 1 {
            let mut seen = vec![false; self.m.nodes.len()];
            let mut q: VecDeque<usize> = frontier.iter().map(|&m| self.m.memories[m].node).collect();
            for &n in &q {
                seen[n] = true;
            }
            let mut next = vec![];
            while let Some(n) = q.pop_front() {
                for &c in &self.m.out_edges[n] {
                    let t = self.m.node_of(self.m.channels[c].dst);
                    if seen[t] || !self.m.nodes[t].enabled {
                        continue;
                    }
                    seen[t] = true;
                    match self.m.nodes[t].ix {
                        NodeIx::Mem(mi) => {
                            if lv[mi] == u8::MAX && !self.m.memories[mi].is_stack() {
                                lv[mi] = level + 1;
                                next.push(mi);
                            }
                        }
                        NodeIx::Unit(_) => {}
                        _ => q.push_back(t),
                    }
                }
            }
            frontier = next;
            level += 1;
        }
        let deepest = lv.iter().copied().filter(|&l| l != u8::MAX).max().unwrap_or(0);
        for (mi, m) in self.m.memories.iter().enumerate() {
            if m.is_stack() {
                lv[mi] = deepest + 1;
            }
        }
        self.m.levels = lv;
    }
}

/// Routers plus router-to-router channels a network topology synthesizes, saturating (charged to the budget).
fn synthesized(spec: &Network, endpoints: u64) -> u64 {
    let count = u64::from(spec.link.count.max(1));
    match &spec.topology {
        Topology::Bus { .. } | Topology::Crossbar { .. } => 1,
        Topology::P2p | Topology::Star { .. } | Topology::Hierarchical { .. } => 0,
        Topology::Ring { rings, .. } => endpoints.saturating_mul(1 + count.saturating_mul(u64::from(*rings))),
        Topology::Mesh { dims } | Topology::Torus { dims, .. } => {
            let r = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(u64::from(d))).unwrap_or(u64::MAX);
            r.saturating_mul(1 + count.saturating_mul(dims.len() as u64))
        }
        Topology::Custom { routers, edges } => edges.iter().fold(u64::from(*routers), |a, e| {
            let per = u64::from(e.link.as_ref().map_or(spec.link.count, |l| l.count).max(1));
            a.saturating_add(u64::from(e.count.max(1)).saturating_mul(per))
        }),
        Topology::Tree { arity, levels } | Topology::FatTree { arity, levels, .. } => {
            let (a, l) = (u64::from((*arity).max(1)), (*levels).max(1));
            let routers = (0..l).fold(0u64, |s, lv| s.saturating_add(a.saturating_pow(lv)));
            routers.saturating_add(u64::from(l - 1).saturating_mul(a.saturating_pow(l - 1)).saturating_mul(count))
        }
    }
}

fn delinearize(mut i: usize, dims: &[u32]) -> Vec<u32> {
    let mut c = vec![0u32; dims.len()];
    for d in (0..dims.len()).rev() {
        let g = dims[d].max(1) as usize;
        c[d] = (i % g) as u32;
        i /= g;
    }
    c
}

/// Neighbor links of a mesh/torus in router order: `(a, b, dim)` with `b` one step `+dim` from `a`. A size-2
/// wrapped dimension yields two parallel links between the pair; size-1 dimensions have none (01 §10.1).
fn grid_links(dims: &[u32], wrap: &[bool]) -> Vec<(usize, usize, usize)> {
    let total: usize = dims.iter().map(|&d| d as usize).product();
    let mut out = vec![];
    for a in 0..total {
        let c = delinearize(a, dims);
        for d in 0..dims.len() {
            if dims[d] <= 1 {
                continue;
            }
            let mut nb = c.clone();
            if c[d] + 1 < dims[d] {
                nb[d] = c[d] + 1;
            } else if wrap.get(d).copied().unwrap_or(false) {
                nb[d] = 0;
            } else {
                continue;
            }
            let b = nb.iter().zip(dims).fold(0usize, |acc, (&x, &g)| acc * g as usize + x as usize);
            out.push((a, b, d));
        }
    }
    out
}

fn connected(n: usize, edges: impl Iterator<Item = (usize, usize)>) -> bool {
    if n <= 1 {
        return true;
    }
    let mut adj = vec![vec![]; n];
    for (a, b) in edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut seen = vec![false; n];
    let mut stack = vec![0];
    seen[0] = true;
    while let Some(x) = stack.pop() {
        for &y in &adj[x] {
            if !seen[y] {
                seen[y] = true;
                stack.push(y);
            }
        }
    }
    seen.into_iter().all(|s| s)
}
