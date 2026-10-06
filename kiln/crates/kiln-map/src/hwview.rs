//! What the mapper and engines see of an expanded design: unit pools, memory chains (feed memory up to the
//! off-chip home), interleaved memory groups, the resource table (03 §4.1) and routed transfer profiles over
//! directional link instances (03 §3.5, L1/L4: every link a transfer crosses carries busy time).

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::{ComputeKind, OperandRole};
use kiln_ir::hw::model::{ChanIx, ChannelKind, ClockIx, MemIx, MemSpec, NodeIx, UnitIx};
use kiln_phys::Phys;
use kiln_trace::sim::ResourceKind;

pub type GroupIx = usize;
pub type ResId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pool {
    Mac,
    Vector,
}

/// Interleaved memory instances acting as one level for placement (an HBM stack set, L2 slices, one SRAM).
#[derive(Clone, Debug, PartialEq)]
pub struct MemGroup {
    pub name: String,
    pub mems: Vec<MemIx>,
    pub level: u8,
    pub offchip: bool,
    pub capacity: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnitInfo {
    pub unit: UnitIx,
    pub pool: Pool,
    pub path: String,
    /// Entity path; units with equal templates share it (cost-cache key).
    pub template: String,
    /// Memory groups from the feed memory (index 0) up to the top of the hierarchy.
    pub chain: Vec<GroupIx>,
    /// Leading chain levels that are single on-chip instances (unit-local tiling applies there).
    pub private: usize,
    /// Interned `template` (cost-cache key).
    pub template_ix: u32,
    pub compute: ResId,
    /// Feed channel resources by operand role (reads into the unit, writes out of it).
    pub feed_in: Vec<(OperandRole, ResId)>,
    pub feed_out: Vec<(OperandRole, ResId)>,
    /// Chain index of each role's feed memory (0 unless roles read from different memories).
    pub feed_at: Vec<(OperandRole, usize)>,
    pub clock: Option<ClockIx>,
    /// Gang (03 §2.2): units of one template sharing every chain level above their own feed memories (the
    /// tensor cores, or the FP32 ALUs, of one SM) are mapped as one unit. The lead lists every member, itself first; members point at the lead.
    pub members: Vec<usize>,
    pub lead: usize,
    /// Compute resources of the special-function units this vector unit hands transcendentals to
    /// ([`crate::cost::special_units`]); empty for MAC units.
    pub special: Vec<ResId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResClass {
    Compute,
    Link,
    Mem,
    Dram,
    Sequencer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    pub path: String,
    pub kind: ResourceKind,
    pub class: ResClass,
    /// B/s for links and memories, 1 for compute and the sequencer (their demands are cycles or seconds).
    pub capacity: f64,
    pub clock: Option<ClockIx>,
    /// Energy per byte read (moved, for links).
    pub energy_j_per_b: f64,
    /// Energy per byte written: a memory's own, `energy_j_per_b` elsewhere.
    pub write_j_per_b: f64,
    pub link_class: Option<String>,
    pub level: Option<u8>,
}

/// Per-link and per-memory share of a transfer's bytes from one group to another, after interleaving over
/// the instances of both groups and splitting over equal-hop paths in proportion to bandwidth (ECMP).
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub src: GroupIx,
    pub dst: GroupIx,
    pub entries: Vec<(ResId, f64)>,
    /// Of `entries`, the shares that are writes into memory instances (priced at their write energy).
    pub writes: Vec<(ResId, f64)>,
    pub latency_s: f64,
    /// The part of `latency_s` that is cycles of a clock domain (on-die links, on-chip memories), by domain
    /// ascending, in seconds at nominal clocks; the rest (DRAM, PHYs) is clock independent.
    pub lat_clk: Vec<(ClockIx, f64)>,
    pub hops: usize,
}

pub struct HwView {
    pub hw: Arc<HwModel>,
    pub phys: Phys,
    pub units: Vec<UnitInfo>,
    pub groups: Vec<MemGroup>,
    pub group_of: Vec<Option<GroupIx>>,
    pub offchip: Option<GroupIx>,
    pub resources: Vec<Resource>,
    pub res_of_mem: Vec<ResId>,
    pub res_of_shared: Vec<ResId>,
    pub sequencer: ResId,
    unit_index: BTreeMap<String, usize>,
    in_edges: Vec<Vec<ChanIx>>,
    bfs: Mutex<BTreeMap<usize, Arc<Vec<u32>>>>,
    profiles: Mutex<BTreeMap<(GroupIx, GroupIx), Arc<Profile>>>,
    res_ids: std::sync::OnceLock<Vec<kiln_ir::common::Id>>,
    res_div: std::sync::OnceLock<Vec<f64>>,
    res_bytes: std::sync::OnceLock<Vec<bool>>,
}

impl std::fmt::Debug for HwView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwView").field("design", &self.hw.design_hash).field("units", &self.units.len()).finish()
    }
}

const UNREACHED: u32 = u32::MAX;

/// Channel fractions, latency, hop count, clocked latency components (along the slowest path).
type RouteSplit = (Vec<(ChanIx, f64)>, f64, usize, Vec<(ClockIx, f64)>);

/// Adds `s` seconds of clock `c` to latency components `v`.
pub fn add_lat_clk(v: &mut Vec<(ClockIx, f64)>, c: ClockIx, s: f64) {
    if s <= 0.0 {
        return;
    }
    match v.binary_search_by_key(&c, |x| x.0) {
        Ok(i) => v[i].1 += s,
        Err(i) => v.insert(i, (c, s)),
    }
}

impl HwView {
    pub fn new(hw: Arc<HwModel>) -> Result<HwView, Diagnostic> {
        let phys = Phys::new(&hw);
        Self::with_phys(hw, phys)
    }

    pub fn with_phys(hw: Arc<HwModel>, phys: Phys) -> Result<HwView, Diagnostic> {
        let mut in_edges = vec![vec![]; hw.nodes.len()];
        for (c, ch) in hw.channels.iter().enumerate() {
            in_edges[hw.node_of(ch.dst)].push(c);
        }
        let mut v = HwView {
            phys,
            units: vec![],
            groups: vec![],
            group_of: vec![None; hw.memories.len()],
            offchip: None,
            resources: vec![],
            res_of_mem: vec![],
            res_of_shared: vec![],
            sequencer: 0,
            unit_index: BTreeMap::new(),
            in_edges,
            bfs: Mutex::new(BTreeMap::new()),
            profiles: Mutex::new(BTreeMap::new()),
            res_ids: std::sync::OnceLock::new(),
            res_div: std::sync::OnceLock::new(),
            res_bytes: std::sync::OnceLock::new(),
            hw,
        };
        v.build_resources();
        v.build_units()?;
        v.unit_index = v.units.iter().enumerate().map(|(i, u)| (u.path.clone(), i)).collect();
        Ok(v)
    }

    fn build_resources(&mut self) {
        let hw = self.hw.clone();
        for r in 0..hw.resources.len() {
            let chans: Vec<usize> = (0..hw.channels.len()).filter(|&c| hw.channels[c].resource == r).collect();
            let first = chans.first().copied();
            let path = first.map_or_else(
                || format!("res{r}"),
                |c| format!("{}->{}", hw.path(hw.channels[c].src), hw.path(hw.channels[c].dst)),
            );
            let lp = self.phys.resource(r);
            let class = first.map(|c| format!("{:?}", hw.channels[c].kind).to_lowercase());
            let kind = match first.map(|c| hw.channels[c].kind) {
                Some(ChannelKind::MemPort) => ResourceKind::MemPort,
                _ => ResourceKind::Link,
            };
            self.res_of_shared.push(self.resources.len() as ResId);
            self.resources.push(Resource {
                path,
                kind,
                class: ResClass::Link,
                capacity: lp.bandwidth_bps,
                clock: first.and_then(|c| hw.channels[c].clock),
                energy_j_per_b: lp.energy_j_per_b,
                write_j_per_b: lp.energy_j_per_b,
                link_class: class,
                level: None,
            });
        }
        for (m, mi) in hw.memories.iter().enumerate() {
            let mp = self.phys.mem(m);
            self.res_of_mem.push(self.resources.len() as ResId);
            self.resources.push(Resource {
                path: hw.nodes[mi.node].path.clone(),
                kind: if mp.dram { ResourceKind::DramChannel } else { ResourceKind::MemPort },
                class: if mp.dram { ResClass::Dram } else { ResClass::Mem },
                capacity: mp.bandwidth_bps,
                clock: mi.clock,
                energy_j_per_b: mp.read_j_per_b,
                write_j_per_b: mp.write_j_per_b,
                link_class: None,
                level: Some(hw.levels[m]),
            });
        }
        self.sequencer = self.resources.len() as ResId;
        self.resources.push(Resource {
            path: "sequencer".into(),
            kind: ResourceKind::Sequencer,
            class: ResClass::Sequencer,
            capacity: 1.0,
            clock: None,
            energy_j_per_b: 0.0,
            write_j_per_b: 0.0,
            link_class: None,
            level: None,
        });
    }

    fn build_units(&mut self) -> Result<(), Diagnostic> {
        let hw = self.hw.clone();
        for (u, ui) in hw.units.iter().enumerate() {
            if !hw.nodes[ui.node].enabled || ui.feeds.is_empty() || ui.near.is_some() {
                continue;
            }
            let pool = match &ui.spec.kind {
                ComputeKind::Matrix(_) | ComputeKind::Cim(_) => Pool::Mac,
                ComputeKind::Vector(_) => Pool::Vector,
                _ => continue,
            };
            let first = [OperandRole::A, OperandRole::In, OperandRole::Any]
                .iter()
                .find_map(|r| ui.feeds.get(r))
                .or_else(|| ui.feeds.values().next())
                .map(|f| f.mem)
                .expect("feeds nonempty");
            // Roles fed from different memories (Hopper's wgmma reads a and b from shared memory, accumulators
            // stay in registers): the chain starts at the feed memory whose chain holds every other feed.
            let mut feed_mems: Vec<MemIx> = ui.feeds.values().map(|f| f.mem).collect();
            feed_mems.sort_unstable();
            feed_mems.dedup();
            let mut chain = self.chain_from(first);
            for &m in &feed_mems {
                if self.holds(&chain, &feed_mems) {
                    break;
                }
                let c = self.chain_from(m);
                if self.holds(&c, &feed_mems) {
                    chain = c;
                }
            }
            let feed_at: Vec<(OperandRole, usize)> =
                ui.feeds.iter().map(|(r, f)| (*r, self.group_of[f.mem].and_then(|g| chain.iter().position(|&c| c == g)).unwrap_or(0))).collect();
            let private = chain
                .iter()
                .take_while(|&&g| self.groups[g].mems.len() == 1 && !self.groups[g].offchip)
                .count()
                .min(chain.len().saturating_sub(1));
            let template = hw.nodes[ui.node].entity.clone();
            let template_ix = self.units.iter().find(|x| x.template == template).map_or(self.units.len() as u32, |x| x.template_ix);
            let res_of_chan = |c: Option<ChanIx>| c.map(|c| self.res_of_shared[hw.channels[c].resource]);
            let feed_in = ui.feeds.iter().filter_map(|(r, f)| res_of_chan(f.read).map(|x| (*r, x))).collect();
            let feed_out = ui.feeds.iter().filter_map(|(r, f)| res_of_chan(f.write).map(|x| (*r, x))).collect();
            let compute = self.resources.len() as ResId;
            let node = &hw.nodes[ui.node];
            self.resources.push(Resource {
                path: node.path.clone(),
                kind: ResourceKind::ComputeUnit,
                class: ResClass::Compute,
                capacity: 1.0,
                clock: ui.clock,
                energy_j_per_b: 0.0,
                write_j_per_b: 0.0,
                link_class: None,
                level: None,
            });
            self.units.push(UnitInfo {
                unit: u,
                pool,
                path: node.path.clone(),
                template,
                chain,
                private,
                template_ix,
                compute,
                feed_in,
                feed_out,
                feed_at,
                clock: ui.clock,
                members: vec![self.units.len()],
                lead: self.units.len(),
                special: vec![],
            });
        }
        self.build_specials();
        self.build_gangs();
        if self.units.is_empty() {
            return Err(Diagnostic::error("E-MAP-HW-001", "design has no enabled matrix or vector unit with feeds")
                .hint("add a compute unit with `feeds` to a die"));
        }
        self.offchip = self.groups.iter().position(|g| g.offchip);
        Ok(())
    }

    /// One compute resource per special-function unit some vector unit hands work to (appended after the
    /// mapped units' resources, so designs without them keep their resource ids).
    fn build_specials(&mut self) {
        let hw = self.hw.clone();
        let mut res: BTreeMap<usize, ResId> = BTreeMap::new();
        for i in 0..self.units.len() {
            if self.units[i].pool != Pool::Vector {
                continue;
            }
            let mut ids = vec![];
            for s in crate::cost::special_units(&hw, self.units[i].unit) {
                let id = *res.entry(s).or_insert_with(|| {
                    let ui = &hw.units[s];
                    self.resources.push(Resource {
                        path: hw.nodes[ui.node].path.clone(),
                        kind: ResourceKind::ComputeUnit,
                        class: ResClass::Compute,
                        capacity: 1.0,
                        clock: ui.clock,
                        energy_j_per_b: 0.0,
                write_j_per_b: 0.0,
                        link_class: None,
                        level: None,
                    });
                    (self.resources.len() - 1) as ResId
                });
                ids.push(id);
            }
            self.units[i].special = ids;
        }
    }

    /// Units of one template whose chains agree above the feed level and whose feeds are partly or fully private
    /// form a gang; units that share their feed memory too (TPU MXUs on one vreg file) stay separate.
    /// The level above the feeds must be one on-chip instance (the SM's L1): tiles that only meet at an
    /// interleaved level or off chip (ember's 254 tiles over a mesh NoC) stay separate units the mapper splits over.
    fn build_gangs(&mut self) {
        let mut by: BTreeMap<(u32, Vec<GroupIx>), Vec<usize>> = BTreeMap::new();
        let private = |g: GroupIx| self.groups[g].mems.len() == 1 && !self.groups[g].offchip;
        for (i, u) in self.units.iter().enumerate().filter(|(_, u)| u.chain.len() > 1 && private(u.chain[1])) {
            by.entry((u.template_ix, u.chain[1..].to_vec())).or_default().push(i);
        }
        for members in by.into_values().filter(|m| m.len() > 1) {
            let mut feeds: Vec<GroupIx> = members.iter().map(|&i| self.units[i].chain[0]).collect();
            feeds.sort_unstable();
            feeds.dedup();
            let (g, k) = (members.len(), feeds.len());
            let even = feeds.iter().all(|&f| members.iter().filter(|&&i| self.units[i].chain[0] == f).count() * k == g);
            if k == 1 || !even {
                continue;
            }
            for &i in &members {
                self.units[i].lead = members[0];
            }
            let lead = members[0];
            self.units[lead].members = members;
        }
    }

    fn group_for(&mut self, mut mems: Vec<MemIx>) -> GroupIx {
        mems.sort_unstable();
        mems.dedup();
        if let Some(g) = self.groups.iter().position(|g| g.mems == mems) {
            return g;
        }
        let hw = &self.hw;
        let m0 = mems[0];
        let g = self.groups.len();
        let name = if mems.len() == 1 { hw.nodes[hw.memories[m0].node].path.clone() } else { hw.nodes[hw.memories[m0].node].entity.clone() };
        self.groups.push(MemGroup {
            name,
            level: hw.levels[m0],
            offchip: hw.memories[m0].is_stack(),
            capacity: mems.iter().map(|&m| hw.memories[m].capacity.0).sum(),
            mems: mems.clone(),
        });
        for m in mems {
            self.group_of[m].get_or_insert(g);
        }
        g
    }

    fn holds(&self, chain: &[GroupIx], mems: &[MemIx]) -> bool {
        mems.iter().all(|&m| self.group_of[m].is_some_and(|g| chain.contains(&g)))
    }

    /// Feed memory, then `backing` where declared, else the nearest directly routed memories one level up.
    fn chain_from(&mut self, base: MemIx) -> Vec<GroupIx> {
        let hw = self.hw.clone();
        let mut chain = vec![self.group_for(vec![base])];
        let mut cur = vec![base];
        for _ in 0..16 {
            let m = cur[0];
            let mi = &hw.memories[m];
            let backing: Vec<MemIx> = mi.backing.iter().copied().filter(|&b| hw.nodes[hw.memories[b].node].enabled).collect();
            let next = if !backing.is_empty() {
                backing
            } else {
                let dist = self.dist_from(mi.node);
                let lvl = hw.levels[m];
                let cand: Vec<MemIx> = (0..hw.memories.len())
                    .filter(|&x| {
                        let n = hw.memories[x].node;
                        x != m && hw.nodes[n].enabled && !hw.memories[x].is_local() && dist[n] != UNREACHED && hw.levels[x] > lvl && hw.levels[x] != u8::MAX
                    })
                    .collect();
                let Some(best) = cand.iter().map(|&x| hw.levels[x]).min() else { break };
                // Of several memories one level up, the one holding the most (ember's 3D-stacked L3, not the
                // CIM macros' activation buffers on the same NoC); ties keep the first declared.
                let mut by_entity: Vec<(&str, u64)> = vec![];
                for &x in cand.iter().filter(|&&x| hw.levels[x] == best) {
                    let e = hw.nodes[hw.memories[x].node].entity.as_str();
                    match by_entity.iter_mut().find(|(n, _)| *n == e) {
                        Some((_, c)) => *c += hw.memories[x].capacity.0,
                        None => by_entity.push((e, hw.memories[x].capacity.0)),
                    }
                }
                let entity = by_entity.iter().fold(by_entity[0], |b, &x| if x.1 > b.1 { x } else { b }).0.to_owned();
                cand.into_iter().filter(|&x| hw.levels[x] == best && hw.nodes[hw.memories[x].node].entity == entity).collect()
            };
            let g = self.group_for(next.clone());
            if chain.contains(&g) {
                break;
            }
            chain.push(g);
            cur = next;
        }
        chain
    }

    fn pass_through(&self, n: usize) -> bool {
        !matches!(self.hw.nodes[n].ix, NodeIx::Unit(_) | NodeIx::Mem(_)) && self.hw.nodes[n].enabled
    }

    /// Hop distances from `src` (a node arena index), expanding only through pass-through nodes.
    fn dist_from(&self, src: usize) -> Arc<Vec<u32>> {
        if let Some(d) = self.bfs.lock().expect("bfs cache").get(&src) {
            return d.clone();
        }
        let hw = &self.hw;
        let mut dist = vec![UNREACHED; hw.nodes.len()];
        dist[src] = 0;
        let mut q = VecDeque::from([src]);
        while let Some(n) = q.pop_front() {
            if n != src && !self.pass_through(n) {
                continue;
            }
            for &c in &hw.out_edges[n] {
                let m = hw.node_of(hw.channels[c].dst);
                if dist[m] == UNREACHED && hw.nodes[m].enabled {
                    dist[m] = dist[n] + 1;
                    q.push_back(m);
                }
            }
        }
        let d = Arc::new(dist);
        self.bfs.lock().expect("bfs cache").insert(src, d.clone());
        d
    }

    /// Channel fractions of a unit flow from node `s` to node `t` over all equal-hop paths, split at each
    /// branch in proportion to link bandwidth. `None` when unreachable.
    fn route_split(&self, s: usize, t: usize) -> Option<RouteSplit> {
        if s == t {
            return Some((vec![], 0.0, 0, vec![]));
        }
        let hw = &self.hw;
        let dist = self.dist_from(s);
        if dist[t] == UNREACHED {
            return None;
        }
        let mut on = vec![false; hw.nodes.len()];
        on[t] = true;
        let mut q = VecDeque::from([t]);
        let mut layers: Vec<Vec<usize>> = vec![vec![]; dist[t] as usize + 1];
        layers[dist[t] as usize].push(t);
        while let Some(n) = q.pop_front() {
            for &c in &self.in_edges[n] {
                let p = hw.node_of(hw.channels[c].src);
                if !on[p] && dist[p] != UNREACHED && dist[p] + 1 == dist[n] && (p == s || self.pass_through(p)) {
                    on[p] = true;
                    layers[dist[p] as usize].push(p);
                    q.push_back(p);
                }
            }
        }
        let mut flow = vec![0.0f64; hw.nodes.len()];
        let mut lat = vec![0.0f64; hw.nodes.len()];
        let mut via: Vec<Option<ChanIx>> = vec![None; hw.nodes.len()];
        flow[s] = 1.0;
        let mut out = vec![];
        for layer in &mut layers {
            layer.sort_unstable();
            for &n in layer.iter() {
                if n == t {
                    continue;
                }
                let next: Vec<ChanIx> = hw.out_edges[n]
                    .iter()
                    .copied()
                    .filter(|&c| {
                        let m = hw.node_of(hw.channels[c].dst);
                        on[m] && dist[m] == dist[n] + 1
                    })
                    .collect();
                let bw = |c: ChanIx| self.phys.link(c).bandwidth_bps.max(1.0);
                let total: f64 = next.iter().map(|&c| bw(c)).sum();
                for c in next {
                    let m = hw.node_of(hw.channels[c].dst);
                    let f = flow[n] * bw(c) / total;
                    flow[m] += f;
                    let l = lat[n] + self.phys.link(c).latency_s;
                    if via[m].is_none() || l > lat[m] {
                        (lat[m], via[m]) = (l, Some(c));
                    }
                    out.push((c, f));
                }
            }
        }
        let mut clk = vec![];
        let mut n = t;
        while let Some(c) = via[n] {
            if let Some(k) = self.clocked_link(c) {
                add_lat_clk(&mut clk, k, self.phys.link(c).latency_s);
            }
            n = hw.node_of(hw.channels[c].src);
        }
        Some((out, lat[t], dist[t] as usize, clk))
    }

    /// The clock domain whose cycles a channel's latency counts: on-die links; PHY, package and host links are
    /// clock independent.
    fn clocked_link(&self, c: ChanIx) -> Option<ClockIx> {
        let ch = &self.hw.channels[c];
        match ch.kind {
            ChannelKind::D2d | ChannelKind::Serdes | ChannelKind::Optical | ChannelKind::Host | ChannelKind::Vertical => None,
            _ => ch.clock,
        }
    }

    /// Transfer profile from group `src` to group `dst` (cached per design). From a group to itself: bytes a unit
    /// writes into the group in place (interleaved over its instances, all writes).
    pub fn profile(&self, src: GroupIx, dst: GroupIx) -> Result<Arc<Profile>, Diagnostic> {
        if let Some(p) = self.profiles.lock().expect("profile cache").get(&(src, dst)) {
            return Ok(p.clone());
        }
        if src == dst {
            let mems = &self.groups[src].mems;
            let entries: Vec<(ResId, f64)> = mems.iter().map(|&m| (self.res_of_mem[m], 1.0 / mems.len() as f64)).collect();
            let p = Arc::new(Profile { src, dst, writes: entries.clone(), entries, latency_s: 0.0, lat_clk: vec![], hops: 0 });
            self.profiles.lock().expect("profile cache").insert((src, dst), p.clone());
            return Ok(p);
        }
        let hw = &self.hw;
        let (gs, gd) = (&self.groups[src], &self.groups[dst]);
        let coherent = |g: &MemGroup| {
            g.mems.len() > 1
                && g.mems.iter().all(|&m| matches!(&hw.memories[m].spec, MemSpec::OnChip(x) if x.cache.as_ref().is_some_and(|c| !c.coherent_with.is_empty())))
        };
        let route = |a: MemIx, b: MemIx| {
            self.route_split(hw.memories[a].node, hw.memories[b].node).ok_or_else(|| {
                Diagnostic::error("E-MAP-ROUTE-001", format!("no route from {} to {}", hw.nodes[hw.memories[a].node].path, hw.nodes[hw.memories[b].node].path))
                    .hint("connect the memories through a network, or place the tensor elsewhere")
            })
        };
        // Interleaved groups spread a transfer over every pair; a coherent (replicating) cache serves reads from its
        // nearest copies (pub: A100 L2 partitions keep coherent copies for their own GPCs). Fills and writes go
        // to the address's home slice, interleaved.
        let mut pairs: Vec<(MemIx, MemIx, f64, RouteSplit)> = vec![];
        let cs = coherent(gs);
        let (outer, inner) = if cs { (&gd.mems, &gs.mems) } else { (&gs.mems, &gd.mems) };
        for &x in outer.iter() {
            let mut rs: Vec<(MemIx, RouteSplit)> = vec![];
            for &y in inner.iter() {
                let (a, b) = if cs { (y, x) } else { (x, y) };
                rs.push((y, route(a, b)?));
            }
            if cs {
                let best = rs.iter().map(|r| r.1.2).min().unwrap_or(0);
                rs.retain(|r| r.1.2 == best);
            }
            let w = 1.0 / (outer.len() * rs.len()) as f64;
            for (y, r) in rs {
                let (a, b) = if cs { (y, x) } else { (x, y) };
                pairs.push((a, b, w, r));
            }
        }
        let mut acc: BTreeMap<ResId, f64> = BTreeMap::new();
        let mut wacc: BTreeMap<ResId, f64> = BTreeMap::new();
        let (mut latency, mut hops, mut lat_clk, mut first) = (0.0f64, 0usize, vec![], true);
        for (a, b, w, (chans, l, h, mut clk)) in pairs {
            *acc.entry(self.res_of_mem[a]).or_default() += w;
            *acc.entry(self.res_of_mem[b]).or_default() += w;
            *wacc.entry(self.res_of_mem[b]).or_default() += w;
            let mut pair: BTreeMap<ResId, f64> = BTreeMap::new();
            for (c, f) in chans {
                let r = self.res_of_shared[hw.channels[c].resource];
                let e = pair.entry(r).or_default();
                if hw.channels[c].kind == ChannelKind::Bus {
                    *e = e.max(f);
                } else {
                    *e += f;
                }
            }
            for (r, f) in pair {
                *acc.entry(r).or_default() += f * w;
            }
            let ml = self.phys.mem(a).latency_s;
            if first || l + ml > latency {
                if let Some(k) = hw.memories[a].clock.filter(|_| !hw.memories[a].is_stack()) {
                    add_lat_clk(&mut clk, k, ml);
                }
                (latency, lat_clk, first) = (l + ml, clk, false);
            }
            hops = hops.max(h);
        }
        let p = Arc::new(Profile { src, dst, entries: acc.into_iter().collect(), writes: wacc.into_iter().collect(), latency_s: latency, lat_clk, hops });
        self.profiles.lock().expect("profile cache").insert((src, dst), p.clone());
        Ok(p)
    }

    /// MAC operand pairs the mappable MAC units run, with their aggregate MAC/s at nominal clocks: what
    /// kiln-wl's convert pass needs to insert explicit converts before contractions no mode runs.
    pub fn mac_modes(&self, dequantize: bool) -> kiln_wl::convert::MacModes {
        let mut modes: Vec<kiln_wl::convert::MacMode> = vec![];
        for u in self.units.iter().filter(|u| u.pool == Pool::Mac) {
            let spec = &self.hw.units[u.unit].spec;
            for m in &spec.precisions {
                if let kiln_ir::hw::compute::PrecisionMode::Mac { a, b, acc, .. } = m {
                    let r = spec.kind.ops_per_cycle(m) * self.clock_hz(u.clock);
                    let (a, b, acc) = (a.precision, b.precision, acc.precision);
                    match modes.iter_mut().find(|x| x.a == a && x.b == b && x.acc == acc) {
                        Some(x) => x.rate += r,
                        None => modes.push(kiln_wl::convert::MacMode { a, b, acc, rate: r }),
                    }
                }
            }
        }
        kiln_wl::convert::MacModes { modes, dequantize }
    }

    /// Mappable units of a pool: gang leads and ungrouped units.
    pub fn pool(&self, pool: Pool) -> Vec<usize> {
        (0..self.units.len()).filter(|&i| self.units[i].pool == pool && self.units[i].lead == i).collect()
    }

    pub fn unit_of(&self, unit: UnitIx) -> Option<usize> {
        self.units.iter().position(|u| u.unit == unit)
    }

    pub fn unit_by_path(&self, path: &str) -> Option<usize> {
        self.unit_index.get(path).copied()
    }

    /// Group whose instances are exactly `mems` (a transfer is interleaved over the group's instances, so a
    /// subset of a group is no home).
    pub fn group_by_mems(&self, mems: &[MemIx]) -> Option<GroupIx> {
        let mut s = mems.to_vec();
        s.sort_unstable();
        s.dedup();
        self.groups.iter().position(|g| g.mems == s)
    }

    pub fn group_by_paths(&self, paths: &[String]) -> Option<GroupIx> {
        let mems: Option<Vec<MemIx>> = paths
            .iter()
            .map(|p| self.hw.memories.iter().position(|m| &self.hw.nodes[m.node].path == p))
            .collect();
        self.group_by_mems(&mems?)
    }

    pub fn group_paths(&self, g: GroupIx) -> Vec<String> {
        self.groups[g].mems.iter().map(|&m| self.hw.nodes[self.hw.memories[m].node].path.clone()).collect()
    }

    /// Path of groups from `src` to unit `u`'s feed group: up to the first group shared with `u`'s chain,
    /// then down that chain. `src` outside every chain routes straight to the nearest chain group.
    pub fn stage_path(&self, src: GroupIx, src_chain: Option<&[GroupIx]>, u: usize) -> Vec<GroupIx> {
        let chain = &self.units[u].chain;
        if let Some(i) = chain.iter().position(|&g| g == src) {
            return chain[..=i].iter().rev().copied().collect();
        }
        if let Some(sc) = src_chain {
            for (j, g) in sc.iter().enumerate() {
                if let Some(i) = chain.iter().position(|x| x == g) {
                    let mut p: Vec<GroupIx> = sc[..=j].to_vec();
                    p.extend(chain[..i].iter().rev());
                    return p;
                }
            }
        }
        let top = chain.len().saturating_sub(1).min(1);
        let mut p = vec![src];
        p.extend(chain[..=top].iter().rev());
        p
    }

    /// Activation home: the largest group every unit's chain shares, below the off-chip level.
    pub fn shared_onchip(&self) -> Option<GroupIx> {
        let first = &self.units.first()?.chain;
        first
            .iter()
            .copied()
            .filter(|&g| !self.groups[g].offchip && self.units.iter().all(|u| u.chain.contains(&g)))
            .max_by_key(|&g| (self.groups[g].capacity, std::cmp::Reverse(g)))
    }

    /// Result ids of the resources ([`sanitize`]d paths), computed once.
    pub fn resource_ids(&self) -> &[kiln_ir::common::Id] {
        self.res_ids.get_or_init(|| {
            self.resources
                .iter()
                .map(|r| kiln_ir::common::Id::new(sanitize(&r.path)).unwrap_or_else(|_| kiln_ir::common::Id::new("unnamed").expect("valid")))
                .collect()
        })
    }

    /// Per resource, what a demand divides by for seconds at nominal clocks: the clock (Hz) of a compute
    /// resource, the capacity (B/s, at least 1) of any other. Computed once.
    pub fn resource_divisors(&self) -> &[f64] {
        self.res_div.get_or_init(|| {
            self.resources.iter().map(|r| if r.class == ResClass::Compute { self.clock_hz(r.clock) } else { r.capacity.max(1.0) }).collect()
        })
    }

    /// Per resource, whether its demands are bytes (not compute cycles or sequencer seconds). Computed once.
    pub fn resource_carries_bytes(&self) -> &[bool] {
        self.res_bytes.get_or_init(|| self.resources.iter().map(|r| !matches!(r.class, ResClass::Compute | ResClass::Sequencer)).collect())
    }

    pub fn clock_hz(&self, c: Option<ClockIx>) -> f64 {
        c.map_or(1.0e9, |c| self.phys.nominal_hz(c))
    }
}

/// Resource paths contain `->` and `[`; result ids follow `[a-z0-9_.-]+` (00 conventions).
pub fn sanitize(path: &str) -> String {
    let s: String = path
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') { c } else { '_' })
        .collect();
    s.replace("-_", "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_ir::hw::{Profile as VProfile, check_file};

    pub(crate) fn load(name: &str) -> Arc<HwModel> {
        let p = format!("{}/../../designs/reference/{name}", env!("CARGO_MANIFEST_DIR"));
        Arc::new(check_file(p, VProfile::Reference).model.expect("design expands"))
    }

    #[test]
    fn a100_chain_and_profiles() {
        let v = HwView::new(load("a100_sxm4_40gb.json5")).unwrap();
        let mac = v.pool(Pool::Mac);
        assert_eq!(mac.len(), 108, "one gang of 4 tensor cores per SM");
        assert_eq!(v.units[mac[0]].members.len(), 4);
        let ch = &v.units[mac[0]].chain;
        let names: Vec<&str> = ch.iter().map(|&g| v.groups[g].name.as_str()).collect();
        assert_eq!(ch.len(), 4, "{names:?}");
        assert_eq!(v.groups[ch[2]].mems.len(), 80);
        assert!(v.groups[ch[3]].offchip && v.groups[ch[3]].mems.len() == 5);
        let p = v.profile(ch[3], ch[2]).unwrap();
        let sum_mem: f64 = p.entries.iter().filter(|(r, _)| v.resources[*r as usize].class == ResClass::Dram).map(|e| e.1).sum();
        assert!((sum_mem - 1.0).abs() < 1e-12);
        let mc_links = p.entries.iter().filter(|(r, _)| v.resources[*r as usize].path.contains(".mc")).count();
        assert!(mc_links >= 10, "ECMP must use both controllers of every stack: {mc_links}");
        assert_eq!(v.shared_onchip(), Some(ch[2]));
    }

    /// On-die links must not cap off-chip bandwidth below the stacks' (published) aggregate when every unit pulls.
    #[test]
    fn fabric_does_not_cap_offchip() {
        for f in [
            "a100_sxm4_40gb.json5",
            "v100_sxm2_32gb.json5",
            "h100_sxm5_80gb.json5",
            "h100_pcie_80gb.json5",
            "tpu_v4.json5",
            "tpu_v5e.json5",
            "tpu_v6e.json5",
        ] {
            let v = HwView::new(load(f)).unwrap();
            let mut below: Vec<GroupIx> = v.pool(Pool::Mac).iter().map(|&u| v.units[u].chain[v.units[u].chain.len() - 2]).collect();
            below.sort_unstable();
            below.dedup();
            let top = *v.units[v.pool(Pool::Mac)[0]].chain.last().unwrap();
            assert!(v.groups[top].offchip, "{f}");
            let dram: f64 = v.groups[top].mems.iter().map(|&m| v.resources[v.res_of_mem[m] as usize].capacity).sum();
            for dir in [false, true] {
                let links: f64 = below
                    .iter()
                    .map(|&g| {
                        let p = if dir { v.profile(g, top) } else { v.profile(top, g) }.unwrap();
                        let worst = p.entries.iter().filter(|e| v.resources[e.0 as usize].class == ResClass::Link).map(|&(r, x)| x / v.resources[r as usize].capacity).fold(0.0, f64::max);
                        1.0 / worst
                    })
                    .sum();
                assert!(links >= dram * (1.0 - 1e-9), "{f}: links carry {:.1} GB/s vs {:.1} GB/s DRAM", links / 1e9, dram / 1e9);
            }
        }
    }

    #[test]
    fn tpu_chain() {
        let v = HwView::new(load("tpu_v5e.json5")).unwrap();
        let mac = v.pool(Pool::Mac);
        assert_eq!(mac.len(), 4);
        let names: Vec<String> = v.units[mac[0]].chain.iter().map(|&g| v.groups[g].name.clone()).collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(names[1].ends_with("vmem"));
        assert_eq!(v.pool(Pool::Vector).len(), 1);
    }
}
