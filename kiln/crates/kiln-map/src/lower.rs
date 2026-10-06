//! Mapping -> task graph (03 §1 `LoweredGraph`, §3.5, §4.8): compute slices, staged transfers derived from
//! footprint overlap (never all-to-all), K-split combines and write-backs, each with per-resource demands.
//! Tier A and Tier B consume the same graph.

use std::collections::BTreeMap;
use std::sync::Arc;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::{ComputeKind, OperandRole};
use kiln_ir::hw::model::ClockIx;
use kiln_ir::op_class::OpClass;

use crate::cost::{NestCost, NestQuery, UnitCostModel};
use crate::geom::{BoxIndex, Slice, TBox, footprint, slice_points, uncovered};
use crate::hwview::{GroupIx, HwView, Pool, Profile, ResId, add_lat_clk};
use crate::mapping::{ExecGroup, GroupKind, LaunchKind, Lifetime, Mapping, OpPlacement, Target};
use crate::program::{POp, Program};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    /// Direct demand on one resource: cycles for compute units (at that unit's clock), bytes otherwise.
    Res(ResId, f64),
    /// Bytes spread over a routed transfer profile (index into [`TaskGraph::profiles`]).
    Prof(u32, f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskKind {
    Compute,
    Transfer,
    Reduce,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub kind: TaskKind,
    pub op: u32,
    pub group: u32,
    /// Latency at nominal clocks; `lat_clk` indexes [`TaskGraph::lat_clk`], its clocked part.
    pub lat_s: f64,
    pub lat_clk: (u32, u32),
    pub dem: (u32, u32),
    pub pred: (u32, u32),
    pub bytes: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpStats {
    pub op: usize,
    pub group: u32,
    pub useful_macs: u128,
    pub issued_macs: u128,
    pub vec_ops: u128,
    pub transc_ops: u128,
    pub slices: u32,
    pub units: u32,
    /// Footprint bytes the op reads from or writes to off-chip homes (I2 compulsory traffic).
    pub compulsory_offchip: u128,
    /// Dense peak MAC/s of the units the op occupies, at nominal clocks (I1).
    pub peak_macs_per_s: f64,
    pub e_compute_j: f64,
    pub padding_j: f64,
    /// `(view unit, compute J, padding J)` per slice: where the energy is spent (its clock domain and container).
    pub e_by_unit: Vec<(usize, f64, f64)>,
    pub mode: String,
    /// Bytes delivered into each memory group.
    pub level_bytes: BTreeMap<GroupIx, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TGroup {
    pub label: String,
    pub ops: Vec<u32>,
    pub tasks: (u32, u32),
    pub barrier_after: bool,
    pub launch: LaunchKind,
    pub iteration: Option<u32>,
    pub fused: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TaskGraph {
    pub tasks: Vec<Task>,
    pub demands: Vec<Amount>,
    /// Clocked latency components `(clock, seconds at its nominal frequency)` of the tasks (03 §4.5: they
    /// stretch when the domain throttles).
    pub lat_clk: Vec<(ClockIx, f64)>,
    pub preds: Vec<u32>,
    pub groups: Vec<TGroup>,
    pub ops: Vec<OpStats>,
    pub profiles: Vec<Arc<Profile>>,
    /// Software-stack kernels issued beyond each group's modelled ones, sorted by group (empty until a
    /// recipe is attached, 08 §F).
    pub stack: Vec<kiln_wl::stack::GroupKernel>,
    /// Workload node of every program op (indexed like [`Task::op`]).
    pub op_node: Vec<u32>,
    pub cost_hits: u64,
    pub cost_misses: u64,
}

impl TaskGraph {
    pub fn demands_of(&self, t: &Task) -> &[Amount] {
        &self.demands[t.dem.0 as usize..t.dem.1 as usize]
    }

    pub fn preds_of(&self, t: &Task) -> &[u32] {
        &self.preds[t.pred.0 as usize..t.pred.1 as usize]
    }

    pub fn lat_clk_of(&self, t: &Task) -> &[(ClockIx, f64)] {
        &self.lat_clk[t.lat_clk.0 as usize..t.lat_clk.1 as usize]
    }
}

type StageKey = (GroupIx, usize, TBox);
/// Unit template, op signature, slice extents, segment, live points and per-operand footprint sizes (masked or
/// param segments), resource share (interned), cost mode (full, quick, floor), operand residency (chain index). A slice's cost depends on its
/// position only through those (see [`UnitCostModel`]).
type CostKey = (u32, u32, Vec<u64>, u32, Vec<u64>, u32, u8, Vec<u8>);
/// Unit template, gang size, op signature, slice extents, segment, live points (masked or param segments).
type RoofKey = (u32, u32, u32, Vec<u64>, u32, u128);

/// A unit's share of the levels it holds in common with the other units an op runs on: capacity of each private
/// chain level, and read bandwidth (B/s) of each shared on-chip level above them.
#[derive(Clone, Debug, PartialEq)]
pub struct Share {
    caps: Vec<u64>,
    bw: Vec<(usize, f64)>,
    /// Interned by value (cost-cache key).
    id: u32,
}

/// Producers `(slice, unit, task)` of each `(operand, output box)`.
/// New private locations and writers of one tensor, committed together.
type NewLocs = (Vec<(TBox, (usize, u32))>, Vec<(TBox, u32)>);
type OutSets = BTreeMap<(usize, TBox), Vec<(usize, usize, u32)>>;

/// The local estimate of one op's tasks (03 §4.2 A1/L over the op alone), accumulated task by task in creation
/// order: busy time per resource (direct demands as they come, profile bytes spread at the end in profile
/// order) and the longest dependency chain within the op.
#[derive(Default)]
struct Est {
    t0: usize,
    busy: Vec<f64>,
    touched: Vec<u32>,
    pbytes: Vec<f64>,
    ptouched: Vec<u32>,
    /// Per task since `t0`: start, finish, latency, is a reduce.
    st: Vec<f64>,
    fin: Vec<f64>,
    lat: Vec<f64>,
    reduce: Vec<bool>,
    lmax: f64,
}

impl Est {
    fn start(t0: usize, busy: Vec<f64>, pbytes: Vec<f64>) -> Est {
        Est { t0, busy, pbytes, ..Est::default() }
    }

    /// Adds task `t0 + n` (its preds are tasks of the same op or earlier ops, which do not count).
    fn add(&mut self, v: &HwView, pmax: &[f64], reduce: bool, lat_s: f64, dem: &[Amount], preds: &[u32]) {
        let div = v.resource_divisors();
        let mut dur = 0.0f64;
        for a in dem {
            match *a {
                Amount::Res(r, x) => {
                    let s = x / div[r as usize];
                    if self.busy[r as usize] == 0.0 {
                        self.touched.push(r);
                    }
                    self.busy[r as usize] += s;
                    dur = dur.max(s);
                }
                Amount::Prof(p, x) => {
                    if self.pbytes.len() <= p as usize {
                        self.pbytes.resize(p as usize + 1, 0.0);
                    }
                    if self.pbytes[p as usize] == 0.0 {
                        self.ptouched.push(p);
                    }
                    self.pbytes[p as usize] += x;
                    dur = dur.max(x * pmax[p as usize]);
                }
            }
        }
        let (mut start, mut floor) = (0.0f64, 0.0f64);
        for &p in preds.iter().filter(|&&p| p as usize >= self.t0) {
            let i = p as usize - self.t0;
            if !self.reduce[i] && !reduce {
                start = start.max(self.st[i] + self.lat[i]);
                floor = floor.max(self.fin[i]);
            } else {
                start = start.max(self.fin[i]);
            }
        }
        let fin = (start + dur).max(floor) + lat_s;
        self.st.push(start);
        self.fin.push(fin);
        self.lat.push(lat_s);
        self.reduce.push(reduce);
        self.lmax = self.lmax.max(fin);
    }

    /// The estimate; returns the zeroed scratch buffers.
    fn finish(mut self, v: &HwView, profiles: &[Arc<Profile>]) -> (f64, Vec<f64>, Vec<f64>) {
        self.ptouched.sort_unstable();
        self.ptouched.dedup();
        for &p in &self.ptouched {
            let x = std::mem::take(&mut self.pbytes[p as usize]);
            for &(r, f) in &profiles[p as usize].entries {
                if self.busy[r as usize] == 0.0 {
                    self.touched.push(r);
                }
                self.busy[r as usize] += x * f / v.resources[r as usize].capacity.max(1.0);
            }
        }
        let mut m = self.lmax;
        for &r in &self.touched {
            m = m.max(self.busy[r as usize]);
            self.busy[r as usize] = 0.0;
        }
        (m, self.busy, self.pbytes)
    }
}

/// Incremental lowering state; ops can be lowered speculatively (`commit = false`) for candidate scoring.
pub struct Lowerer<'a> {
    pub prog: &'a Program,
    pub view: &'a HwView,
    pub cost: &'a dyn UnitCostModel,
    pub g: TaskGraph,
    /// Profile id per `(from, to)` group pair, dense (`u32::MAX` = not yet used).
    prof_ids: Vec<u32>,
    scratch: Vec<f64>,
    pscratch: Vec<f64>,
    /// Per profile: max over entries of share / capacity (seconds per byte on its slowest resource).
    pmax: Vec<f64>,
    private: Vec<BoxIndex<(usize, u32)>>,
    writers: Vec<BoxIndex<u32>>,
    touched: Vec<usize>,
    staged: BTreeMap<StageKey, u32>,
    homes: Vec<Vec<(TBox, GroupIx)>>,
    lifetimes: Vec<Lifetime>,
    cost_cache: BTreeMap<CostKey, Arc<NestCost>>,
    last_cost: Option<(CostKey, Arc<NestCost>)>,
    shares: BTreeMap<(u32, Vec<usize>), Arc<Share>>,
    share_ids: BTreeMap<Vec<u64>, u32>,
    /// Shares of the op being lowered, by unit template (its unit set is fixed).
    op_shares: Vec<(u32, Arc<Share>)>,
    /// Transfer profiles fetched from the view, with their slowest-resource seconds per byte, by group pair.
    pcache: Vec<Option<(Arc<Profile>, f64)>>,
    roof_cache: BTreeMap<RoofKey, f64>,
    dscratch: Vec<Amount>,
    counted: std::collections::BTreeSet<usize>,
    cur_group: u32,
    /// Speculative lowering costs slices with [`UnitCostModel::cost_floor`]: the estimate is a lower bound.
    pub floor: bool,
    /// Speculative lowering: tasks are folded into the estimate as they are created, never stored.
    virt: Option<Est>,
    /// Staging paths by (source group or unit, destination unit, source is a unit's feed).
    paths: BTreeMap<(usize, usize, bool), Arc<[GroupIx]>>,
    /// Live points of masked slices by (op, slice).
    points: BTreeMap<usize, BTreeMap<Slice, u128>>,
}

/// Vector work beside a MAC unit: the vector unit (view index) and its compute resources, cycles on each, the bytes
/// it moves through its feeds and feed memories, energy per operation.
struct VectorWork {
    unit: usize,
    units: Vec<ResId>,
    cycles: f64,
    feed: Vec<Amount>,
    /// Bytes written into each feed memory group.
    writes: Vec<(GroupIx, f64)>,
    e_op: f64,
}

impl VectorWork {
    fn charge(&self, st: &mut OpStats, ops: f64) {
        st.e_compute_j += ops * self.e_op;
        st.e_by_unit.push((self.unit, ops * self.e_op, 0.0));
    }

}

/// Effects of lowering one op, applied on commit.
#[derive(Default)]
struct Effects {
    level: Vec<f64>,
    private: Vec<(usize, TBox, usize, u32)>,
    writers: Vec<(usize, TBox, u32)>,
    counted: Vec<usize>,
    stats: OpStats,
}

fn region_box(lo: &[u64], hi: &[u64]) -> TBox {
    let mut b = TBox::EMPTY;
    for (l, h) in lo.iter().zip(hi) {
        b.push(*l as i64, *h as i64, h - l);
    }
    b
}

/// Where slice `s` starts relative to the blocks its footprints depend on: per operand axis under a floor division
/// or of a block-scaled input, its start modulo that block (empty when the op has no such axis).
fn alignment(prog: &Program, op: &POp, s: &Slice) -> Vec<u64> {
    use kiln_ir::wl::IndexExpr;
    let seg = &op.segs[s.seg as usize];
    let param = |p: &str| seg.params.iter().find(|(n, _)| n == p).map_or(0, |x| x.1);
    fn start(op: &POp, s: &Slice, param: &dyn Fn(&str) -> i64, e: &IndexExpr) -> Option<(i64, u64)> {
        match e {
            IndexExpr::Affine { terms, offset } => Some((
                offset + terms.iter().map(|t| t.coeff * t.param.as_deref().map_or(1, param) * t.dim.as_deref().and_then(|d| op.dim_ix(d)).map_or(1, |d| s.lo[d] as i64)).sum::<i64>(),
                1,
            )),
            IndexExpr::FloorDiv { inner, by } => start(op, s, param, inner).map(|(x, m)| (x, m.saturating_mul((*by).max(1)))),
            IndexExpr::Indirect { .. } => None,
        }
    }
    let mut out = vec![];
    for o in &op.operands {
        let block = if o.access.writes() { 1 } else { kiln_wl::convert::operand_spec(&prog.tensors[o.tensor].dtype).block_size().map_or(1, u64::from) };
        for e in &o.index {
            if let Some((x, m)) = start(op, s, &param, e) {
                let m = m.saturating_mul(block);
                if m > 1 {
                    out.push(x.rem_euclid(m as i64) as u64);
                }
            }
        }
    }
    out
}

/// Per-op lowering context: which groups the op's units share (capacity shares, private-hop folding).
struct OpCtx {
    sharers: BTreeMap<GroupIx, usize>,
}

impl<'a> Lowerer<'a> {
    pub fn new(prog: &'a Program, view: &'a HwView, mapping: &Mapping, cost: &'a dyn UnitCostModel) -> Result<Self, Diagnostic> {
        let n = prog.tensors.len();
        let mut homes = vec![vec![]; n];
        let mut lifetimes = vec![Lifetime::Streamed; n];
        for (i, t) in prog.tensors.iter().enumerate() {
            let Some(tp) = mapping.tensors.get(&t.id) else { continue };
            lifetimes[i] = tp.lifetime.clone();
            for h in &tp.home {
                let g = view.group_by_paths(&h.mems).ok_or_else(|| {
                    Diagnostic::error("E-MAP-VAL-011", format!("home of {} names unknown memories or not exactly one memory group", t.id)).at(t.id.clone())
                })?;
                homes[i].push((region_box(&h.region.lo, &h.region.hi), g));
            }
        }
        Ok(Lowerer {
            prog,
            view,
            cost,
            g: TaskGraph { op_node: prog.ops.iter().map(|o| o.node as u32).collect(), ..TaskGraph::default() },
            prof_ids: vec![u32::MAX; view.groups.len() * view.groups.len()],
            scratch: vec![0.0; view.resources.len()],
            pscratch: vec![],
            pmax: vec![],
            private: vec![BoxIndex::default(); n],
            writers: vec![BoxIndex::default(); n],
            touched: vec![],
            staged: BTreeMap::new(),
            homes,
            lifetimes,
            cost_cache: BTreeMap::new(),
            last_cost: None,
            shares: BTreeMap::new(),
            share_ids: BTreeMap::new(),
            op_shares: vec![],
            pcache: vec![None; view.groups.len() * view.groups.len()],
            roof_cache: BTreeMap::new(),
            dscratch: vec![],
            counted: Default::default(),
            cur_group: 0,
            floor: false,
            virt: None,
            paths: BTreeMap::new(),
            points: BTreeMap::new(),
        })
    }

    fn prof(&mut self, from: GroupIx, to: GroupIx) -> Result<(u32, f64), Diagnostic> {
        let slot = from * self.view.groups.len() + to;
        let id = self.prof_ids[slot];
        if id != u32::MAX {
            return Ok((id, self.g.profiles[id as usize].latency_s));
        }
        let (p, pmax) = match &self.pcache[slot] {
            Some(x) => x.clone(),
            None => {
                let p = self.view.profile(from, to)?;
                let m = p.entries.iter().map(|&(r, f)| f / self.view.resources[r as usize].capacity.max(1.0)).fold(0.0, f64::max);
                self.pcache[slot] = Some((p.clone(), m));
                (p, m)
            }
        };
        let id = self.g.profiles.len() as u32;
        let lat = p.latency_s;
        self.pmax.push(pmax);
        self.g.profiles.push(p);
        self.prof_ids[slot] = id;
        Ok((id, lat))
    }

    /// [`HwView::stage_path`] from group `src` (or from unit `src`'s feed along its chain) to unit `u`, cached.
    fn path(&mut self, src: usize, from_unit: bool, u: usize) -> Arc<[GroupIx]> {
        let view = self.view;
        self.paths
            .entry((src, u, from_unit))
            .or_insert_with(|| {
                if from_unit {
                    let c = &view.units[src].chain;
                    view.stage_path(c[0], Some(c), u).into()
                } else {
                    view.stage_path(src, None, u).into()
                }
            })
            .clone()
    }

    /// Live points of slice `s` of op `oi` (masked segments counted once per slice).
    fn slice_points(&mut self, oi: usize, s: &Slice) -> u128 {
        let op = &self.prog.ops[oi];
        if s.is_empty() || op.segs[s.seg as usize].cons.is_empty() {
            return slice_points(op, s);
        }
        let m = self.points.entry(oi).or_default();
        if let Some(&x) = m.get(s) {
            return x;
        }
        let x = slice_points(op, s);
        m.insert(s.clone(), x);
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn push_task(&mut self, kind: TaskKind, op: usize, lat_s: f64, lat_clk: &[(ClockIx, f64)], dem: &[Amount], preds: &[u32], bytes: f64) -> u32 {
        if let Some(e) = &mut self.virt {
            e.add(self.view, &self.pmax, kind == TaskKind::Reduce, lat_s, dem, preds);
            return (e.t0 + e.st.len() - 1) as u32;
        }
        let l0 = self.g.lat_clk.len() as u32;
        self.g.lat_clk.extend_from_slice(lat_clk);
        let d0 = self.g.demands.len() as u32;
        self.g.demands.extend_from_slice(dem);
        let p0 = self.g.preds.len();
        self.g.preds.extend_from_slice(preds);
        self.g.preds[p0..].sort_unstable();
        let mut w = p0;
        for r in p0..self.g.preds.len() {
            if w == p0 || self.g.preds[r] != self.g.preds[w - 1] {
                self.g.preds[w] = self.g.preds[r];
                w += 1;
            }
        }
        self.g.preds.truncate(w);
        let p0 = p0 as u32;
        self.g.tasks.push(Task {
            kind,
            op: op as u32,
            group: self.cur_group,
            lat_s,
            lat_clk: (l0, self.g.lat_clk.len() as u32),
            dem: (d0, self.g.demands.len() as u32),
            pred: (p0, self.g.preds.len() as u32),
            bytes,
        });
        (self.g.tasks.len() - 1) as u32
    }

    /// One task moving `bytes` along consecutive group hops `path` (no staging dedup). A hop leaving `src`'s or
    /// entering `dst`'s feed level is spread over that unit's gang members' feed levels.
    #[allow(clippy::too_many_arguments)]
    fn route_task(
        &mut self,
        op: usize,
        path: &[GroupIx],
        bytes: f64,
        preds: &[u32],
        fx: &mut Effects,
        src: Option<usize>,
        dst: Option<usize>,
    ) -> Result<Option<u32>, Diagnostic> {
        if path.len() < 2 {
            return Ok(None);
        }
        let view = self.view;
        let gang = |u: Option<usize>, g: GroupIx| u.filter(|&u| view.units[u].members.len() > 1 && view.units[u].chain[0] == g);
        let mut dem = std::mem::take(&mut self.dscratch);
        dem.clear();
        let mut lat = 0.0;
        let mut clk = vec![];
        for w in path.windows(2).take(16) {
            let (a, b) = (gang(src, w[0]), gang(dst, w[1]));
            let (mut hop, mut hop_p) = (0.0f64, u32::MAX);
            match a.or(b) {
                Some(u) => {
                    let share = bytes / view.units[u].members.len() as f64;
                    for &m in &view.units[u].members {
                        let g = view.units[m].chain[0];
                        let (pid, l) = if a.is_some() { self.prof(g, w[1])? } else { self.prof(w[0], g)? };
                        dem.push(Amount::Prof(pid, share));
                        if hop_p == u32::MAX || l > hop {
                            (hop, hop_p) = (l, pid);
                        }
                        fx.level[if a.is_some() { w[1] } else { g }] += share;
                    }
                }
                None => {
                    let (pid, l) = self.prof(w[0], w[1])?;
                    dem.push(Amount::Prof(pid, bytes));
                    (hop, hop_p) = (l, pid);
                    fx.level[w[1]] += bytes;
                }
            }
            lat += hop;
            if let Some(p) = self.g.profiles.get(hop_p as usize) {
                p.lat_clk.iter().for_each(|&(c, x)| add_lat_clk(&mut clk, c, x));
            }
        }
        let t = self.push_task(TaskKind::Transfer, op, lat, &clk, &dem, preds, bytes);
        self.dscratch = dem;
        Ok(Some(t))
    }

    /// Slices of `op` under placement `p`, segment-major then split axes row-major.
    pub fn slices(op: &POp, p: &OpPlacement) -> Vec<Slice> {
        let mut out = vec![];
        for (si, seg) in op.segs.iter().enumerate() {
            let mut cur = vec![Slice::whole(si as u32, seg)];
            for ax in &p.split {
                let Some(d) = op.dim_ix(&ax.dim) else { continue };
                let mut next = vec![];
                for s in &cur {
                    let mut lo = s.lo[d];
                    for &part in &ax.parts {
                        let mut t = s.clone();
                        t.lo[d] = lo;
                        t.hi[d] = (lo + part).min(s.hi[d]);
                        lo += part;
                        next.push(t);
                    }
                }
                cur = next;
            }
            out.extend(cur);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn nest_cost(&mut self, oi: usize, op: &POp, sig: u32, s: &Slice, fps: &[TBox], u: usize, share: &Share, quick: bool) -> Result<Arc<NestCost>, Diagnostic> {
        let seg = &op.segs[s.seg as usize];
        let ext: Vec<u64> = (0..s.lo.len()).map(|d| s.extent(d)).collect();
        let pos = if seg.cons.is_empty() && seg.params.is_empty() {
            vec![]
        } else {
            let pts = self.slice_points(oi, s);
            let mut v = vec![(pts >> 64) as u64, pts as u64];
            v.extend(fps.iter().flat_map(|b| {
                let e = b.elems();
                [(e >> 64) as u64, e as u64]
            }));
            v
        };
        let mut pos = pos;
        pos.extend(alignment(self.prog, op, s));
        let caps = &share.caps;
        let mode = if self.floor { 2 } else { u8::from(quick) };
        let at = self.residency(oi, op, u);
        let key = (self.view.units[u].template_ix, sig, ext, s.seg, pos, share.id, mode, at.clone());
        if let Some((k, c)) = &self.last_cost
            && *k == key
        {
            self.g.cost_hits += u64::from(!self.floor);
            return Ok(c.clone());
        }
        if let Some(c) = self.cost_cache.get(&key) {
            self.g.cost_hits += u64::from(!self.floor);
            let c = c.clone();
            self.last_cost = Some((key, c.clone()));
            return Ok(c);
        }
        self.g.cost_misses += u64::from(!self.floor);
        let ui = &self.view.units[u];
        let mems: Vec<usize> = ui.chain[..caps.len()].iter().map(|&g| self.view.groups[g].mems[0]).collect();
        let residency: Vec<Option<usize>> = at.iter().map(|&i| ui.chain.get(usize::from(i)).map(|&g| self.view.groups[g].mems[0])).collect();
        let points = self.slice_points(oi, s);
        let q = NestQuery {
            prog: self.prog,
            op,
            slice: s,
            points,
            hw: &self.view.hw,
            unit: ui.unit,
            level_caps: caps,
            level_mems: &mems,
            level_bw: &share.bw,
            residency: &residency,
            gang: ui.members.len() as u32,
            quick,
        };
        let c = Arc::new(match self.floor.then(|| self.cost.cost_floor(&q)).flatten() {
            Some(Ok(c)) => c,
            _ => self.cost.cost(&q)?,
        });
        self.cost_cache.insert(key.clone(), c.clone());
        self.last_cost = Some((key, c.clone()));
        Ok(c)
    }

    /// Per operand of op `oi`, the index in unit `u`'s chain of the memory it is read from (or written to) when
    /// the op starts (ends): its home, or the level a home outside the chain enters it at; `u8::MAX` for the top of
    /// the chain (also private tensors, which may stage through any level).
    fn residency(&self, oi: usize, op: &POp, u: usize) -> Vec<u8> {
        let chain = &self.view.units[u].chain;
        let top = chain.len().saturating_sub(1);
        (0..op.operands.len())
            .map(|i| {
                let t = self.prog.root(self.prog.converted_from(oi, i).map_or(op.operands[i].tensor, |c| c.0));
                let at = self.homes[t].iter().map(|&(_, g)| chain.iter().position(|&c| c == g).unwrap_or(top.min(1))).max();
                at.filter(|&x| x < top).map_or(u8::MAX, |x| x.min(254) as u8)
            })
            .collect()
    }

    fn pool_units(p: &OpPlacement, units: &[usize]) -> Vec<usize> {
        let mut v: Vec<usize> = p.slice_to_unit.iter().map(|&i| units[i as usize]).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Units of the op on each private-or-feed chain group.
    fn sharers(view: &HwView, pool_units: &[usize]) -> BTreeMap<GroupIx, usize> {
        let mut m = BTreeMap::new();
        for &u in pool_units {
            for &g in &view.units[u].chain[..view.units[u].private.max(1)] {
                *m.entry(g).or_default() += 1;
            }
        }
        m
    }

    /// `u`'s [`Share`] when the op runs on `pool_units` (cached per template and unit set). The bandwidth share
    /// of a shared on-chip level is the rate each unit gets when every unit using it reads from it at once.
    fn share(&mut self, u: usize, pool_units: &[usize]) -> Result<Arc<Share>, Diagnostic> {
        let view = self.view;
        let ui = &view.units[u];
        if let Some((_, s)) = self.op_shares.iter().find(|x| x.0 == ui.template_ix) {
            return Ok(s.clone());
        }
        let key = (ui.template_ix, pool_units.to_vec());
        if let Some(s) = self.shares.get(&key) {
            self.op_shares.push((ui.template_ix, s.clone()));
            return Ok(s.clone());
        }
        let sharers = Self::sharers(view, pool_units);
        let caps: Vec<u64> = ui.chain[..ui.private].iter().map(|g| view.groups[*g].capacity / sharers.get(g).copied().unwrap_or(1).max(1) as u64).collect();
        let mut bw = vec![];
        for i in ui.private.max(1)..ui.chain.len() {
            let g = ui.chain[i];
            if view.groups[g].offchip {
                break;
            }
            let mut load: BTreeMap<ResId, f64> = BTreeMap::new();
            for &s in pool_units.iter().filter(|&&s| view.units[s].chain.get(i) == Some(&g)) {
                for &(r, f) in &view.profile(g, view.units[s].chain[i - 1])?.entries {
                    *load.entry(r).or_default() += f / view.resources[r as usize].capacity.max(1.0);
                }
            }
            let worst = load.values().copied().fold(0.0, f64::max);
            if worst > 0.0 {
                bw.push((view.groups[g].mems[0], 1.0 / worst));
            }
        }
        let vk: Vec<u64> = caps.iter().copied().chain(bw.iter().flat_map(|&(m, b)| [m as u64, b.to_bits()])).collect();
        let n = self.share_ids.len() as u32;
        let id = *self.share_ids.entry(vk).or_insert(n);
        let s = Arc::new(Share { caps, bw, id });
        self.shares.insert(key, s.clone());
        self.op_shares.push((ui.template_ix, s.clone()));
        Ok(s)
    }

    fn feed_res(&self, u: usize, oi: usize, op: &POp, write: bool) -> Option<ResId> {
        self.feed(u, oi, op, write).map(|x| x.1)
    }

    /// Role and channel resource the unit reads operand `oi` through (or writes it through).
    fn feed(&self, u: usize, oi: usize, op: &POp, write: bool) -> Option<(OperandRole, ResId)> {
        let ui = &self.view.units[u];
        let roles: &[OperandRole] = match (&op.mac, write) {
            (Some(m), false) if oi == m.a => &[OperandRole::A],
            (Some(m), false) if oi == m.b => &[OperandRole::B],
            (Some(m), false) if oi == m.out => &[OperandRole::C, OperandRole::O],
            (Some(_), true) => &[OperandRole::O, OperandRole::C],
            (_, false) => &[OperandRole::In, OperandRole::Any, OperandRole::A],
            (_, true) => &[OperandRole::Out, OperandRole::Any, OperandRole::O],
        };
        let list = if write { &ui.feed_out } else { &ui.feed_in };
        roles.iter().find_map(|r| list.iter().find(|(x, _)| x == r)).or(list.first()).copied()
    }

    /// Chain index of the memory unit `u` reads operand `oi` from (or writes it to).
    fn feed_level(&self, u: usize, oi: usize, op: &POp, write: bool) -> usize {
        let ui = &self.view.units[u];
        self.feed(u, oi, op, write).and_then(|(r, _)| ui.feed_at.iter().find(|x| x.0 == r)).map_or(0, |x| x.1)
    }

    /// Busiest unit's compute seconds under placement `p` from the structural (roofline) unit model: a cheap
    /// candidate pre-screen; no tasks are created and the configured cost model is not queried.
    pub fn compute_estimate(&mut self, oi_prog: usize, p: &OpPlacement, units: &[usize]) -> Result<f64, Diagnostic> {
        let view = self.view;
        let op = &self.prog.ops[oi_prog];
        let pool_units = Self::pool_units(p, units);
        self.op_shares.clear();
        let sig = self.prog.signature(oi_prog);
        let mut per_unit: Vec<f64> = vec![0.0; view.units.len()];
        let mut last: Option<(RoofKey, f64)> = None;
        for (si, s) in Self::slices(op, p).iter().enumerate() {
            if s.is_empty() {
                continue;
            }
            let u = units[p.slice_to_unit[si] as usize];
            let ui = &view.units[u];
            // Roofline cycles depend on the template, gang, op, extents and (masked/param segments) live points only.
            let seg = &op.segs[s.seg as usize];
            let pts = if seg.cons.is_empty() && seg.params.is_empty() { 0 } else { self.slice_points(oi_prog, s) };
            let share = self.share(u, &pool_units)?;
            let same = |k: &RoofKey| {
                k.0 == ui.template_ix && k.1 == ui.members.len() as u32 && k.4 == s.seg && k.5 == pts && k.3.iter().enumerate().all(|(d, &e)| e == s.extent(d))
            };
            if let Some((k, c)) = &last
                && same(k)
            {
                per_unit[u] += *c / view.clock_hz(ui.clock);
                continue;
            }
            let key = (ui.template_ix, ui.members.len() as u32, sig, (0..s.lo.len()).map(|d| s.extent(d)).collect(), s.seg, pts);
            let cycles = match self.roof_cache.get(&key) {
                Some(&c) => c,
                None => {
                    let mems: Vec<usize> = ui.chain[..share.caps.len()].iter().map(|&g| view.groups[g].mems[0]).collect();
                    let points = self.slice_points(oi_prog, s);
                    let c = crate::cost::RooflineCost.cost(&NestQuery {
                        prog: self.prog,
                        op,
                        slice: s,
                        points,
                        hw: &view.hw,
                        unit: ui.unit,
                        level_caps: &share.caps,
                        level_mems: &mems,
                        level_bw: &share.bw,
                        residency: &[],
                        gang: ui.members.len() as u32,
                        quick: true,
                    })?;
                    let sfu = ui.special.first().map_or(0.0, |&r| c.special_cycles / view.clock_hz(view.resources[r as usize].clock));
                    let t = (c.cycles / view.clock_hz(ui.clock)).max(sfu) * view.clock_hz(ui.clock);
                    self.roof_cache.insert(key.clone(), t);
                    t
                }
            };
            last = Some((key, cycles));
            per_unit[u] += cycles / view.clock_hz(ui.clock);
        }
        Ok(per_unit.into_iter().fold(0.0, f64::max))
    }

    /// Lowers one op; returns its local estimate `max(A1_op, L_op)` (seconds, peak efficiencies).
    pub fn lower_op(&mut self, oi_prog: usize, p: &OpPlacement, units: &[usize], commit: bool) -> Result<f64, Diagnostic> {
        let prog = self.prog;
        let view = self.view;
        let op = &prog.ops[oi_prog];
        let sig = prog.signature(oi_prog);
        let (t0, d0, p0) = (self.g.tasks.len(), self.g.demands.len(), self.g.preds.len());
        let nprof = self.g.profiles.len();
        let mut fx = Effects { level: vec![0.0; view.groups.len()], ..Effects::default() };
        if self.virt.take().is_some() || self.scratch.len() != view.resources.len() {
            // An earlier speculative lowering failed midway: its buffers are not clean.
            self.scratch = vec![0.0; view.resources.len()];
            self.pscratch.clear();
        }
        if !commit {
            self.virt = Some(Est::start(t0, std::mem::take(&mut self.scratch), std::mem::take(&mut self.pscratch)));
        }
        let slices = Self::slices(op, p);
        let pool_units = Self::pool_units(p, units);
        self.op_shares.clear();
        let cx = OpCtx { sharers: Self::sharers(view, &pool_units) };
        let mut local_staged: BTreeMap<StageKey, u32> = BTreeMap::new();
        let mut out_sets: OutSets = BTreeMap::new();
        let mut fps: Vec<TBox> = Vec::with_capacity(op.operands.len());
        // Operands a fused convert produces: read from its source, converted beside the MAC unit.
        let conv: Vec<(usize, usize, usize)> = (0..op.operands.len()).filter_map(|oi| prog.converted_from(oi_prog, oi).map(|(t, c)| (oi, t, c))).collect();
        let src = |oi: usize| conv.iter().find(|x| x.0 == oi).map_or(op.operands[oi].tensor, |x| x.1);
        for oi in 0..op.operands.len() {
            let tensor = src(oi);
            let root = prog.root(tensor);
            if self.homes[root].iter().any(|(_, g)| view.groups[*g].offchip) && !self.counted.contains(&root) && !fx.counted.contains(&root) {
                let k = &op.kernel;
                let numel: u128 = prog.tensors[tensor].shape.iter().map(|&d| u128::from(d)).product();
                let el = k.operand_footprint(&k.operands[oi]).unwrap_or(0).min(numel);
                fx.stats.compulsory_offchip += prog.tensors[tensor].bytes(el);
                fx.counted.push(root);
            }
        }
        for (si, s) in slices.iter().enumerate() {
            if s.is_empty() {
                continue;
            }
            let u = units[p.slice_to_unit[si] as usize];
            let ui = &view.units[u];
            let share = self.share(u, &pool_units)?;
            fps.clear();
            fps.extend(op.operands.iter().enumerate().map(|(oi, o)| footprint(op, oi, s, &prog.tensors[o.tensor].shape)));
            let nc = self.nest_cost(oi_prog, op, sig, s, &fps, u, &share, !commit)?;
            let mut deps: Vec<u32> = vec![];
            let mut fold: Vec<Amount> = vec![];
            let mut fold_lat = (0.0f64, u32::MAX);
            for (oi, _) in op.operands.iter().enumerate().filter(|(_, o)| o.access.reads()) {
                let b = fps[oi];
                if b.elems() == 0 {
                    continue;
                }
                self.read(oi_prog, oi, src(oi), &b, u, &nc, &cx, &mut local_staged, &mut deps, &mut fold, &mut fold_lat, &mut fx)?;
            }
            let f = view.clock_hz(ui.clock);
            let share = 1.0 / ui.members.len() as f64;
            let mut dem: Vec<Amount> = ui.members.iter().map(|&m| Amount::Res(view.units[m].compute, nc.cycles)).collect();
            if nc.special_cycles > 0.0 {
                dem.extend(ui.members.iter().flat_map(|&m| view.units[m].special.iter().map(|&r| Amount::Res(r, nc.special_cycles))));
            }
            for &(oi, t, c) in &conv {
                let elems = fps[oi].elems() as f64;
                let to = prog.tensors[op.operands[oi].tensor].dtype.scalar;
                let b = prog.ops[c].body();
                let work = |k: &ComputeKind| elems * (f64::from(b.cvt) / k.class_rate(OpClass::Convert) + f64::from(b.mul) / k.class_rate(OpClass::Elementwise));
                let io = (prog.tensors[t].bytes(fps[oi].elems()) as f64, prog.tensors[op.operands[oi].tensor].bytes(fps[oi].elems()) as f64);
                let vw = self.vector_cycles(u, &prog.ops[c], "convert a contraction operand", &[prog.tensors[t].dtype.scalar, to], work, io)?;
                dem.extend(self.vector_demands(&vw)?);
                let ops = elems * f64::from(b.cvt + b.mul);
                fx.stats.vec_ops += ops as u128;
                vw.charge(&mut fx.stats, ops);
            }
            dem.extend(fold);
            for &m in &ui.members {
                // Feed bytes per chain level the unit reads or writes operands at (index 0 unless roles are fed
                // from different memories).
                let mut feed_total = [[0.0f64; 2]; 4];
                for (oi, o) in op.operands.iter().enumerate() {
                    for w in [false, true] {
                        // Outputs are read back through the feed for their partial sums.
                        let bytes = share * if w || o.access.reads() { nc.feed_bytes[oi] } else { nc.readback.get(oi).copied().unwrap_or(0.0) };
                        if (if w { o.access.writes() } else { o.access.reads() || bytes > 0.0 })
                            && let Some(r) = self.feed_res(m, oi, op, w)
                        {
                            dem.push(Amount::Res(r, bytes));
                            feed_total[self.feed_level(m, oi, op, w).min(3)][usize::from(w)] += bytes;
                        }
                    }
                }
                let chain = &view.units[m].chain;
                for (l, &[rd, wr]) in feed_total.iter().enumerate() {
                    let g = chain[l.min(chain.len() - 1)];
                    if l == 0 || rd > 0.0 {
                        dem.push(Amount::Res(view.res_of_mem[view.groups[g].mems[0]], rd));
                    }
                    if wr > 0.0 {
                        dem.push(Amount::Prof(self.prof(g, g)?.0, wr));
                    }
                }
                // Partial sums spilled above the feed move between the levels of the unit's chain while it computes.
                for (ent, up, down) in &nc.spill {
                    let Some(i) = chain.iter().position(|&g| self.group_entity(g) == ent.as_str()).filter(|&i| i > 0) else { continue };
                    for (from, to, b) in [(chain[i - 1], chain[i], *up), (chain[i], chain[i - 1], *down)] {
                        if b > 0.0 {
                            let (pid, _) = self.prof(from, to)?;
                            dem.push(Amount::Prof(pid, b * share));
                            fx.level[to] += b * share;
                        }
                    }
                }
            }
            let mut clk = vec![];
            if let Some(c) = ui.clock {
                add_lat_clk(&mut clk, c, nc.fill_cycles / f);
            }
            if let Some(p) = self.g.profiles.get(fold_lat.1 as usize) {
                p.lat_clk.iter().for_each(|&(c, x)| add_lat_clk(&mut clk, c, x));
            }
            let mut ct = self.push_task(TaskKind::Compute, oi_prog, nc.fill_cycles / f + fold_lat.0, &clk, &dem, &deps, 0.0);
            // A contraction's own vector work (MX scale passes, the final down-conversion) runs on the vector unit
            // beside its MAC unit once the slice's partial results exist; the slice's output waits for it.
            if op.mac.is_some() && nc.vec_ops > 0 {
                let (cvt, rest) = (nc.cvt_ops as f64, (nc.vec_ops - nc.cvt_ops) as f64);
                let work = |k: &ComputeKind| cvt / k.class_rate(OpClass::Convert) + rest / k.class_rate(OpClass::Elementwise);
                let acc = op.kernel.accum.unwrap_or(kiln_ir::precision::Precision::Fp32);
                // Each operation reads an accumulator; conversions write the result, scale passes an accumulator.
                let (acc_b, out_b) = (f64::from(acc.element_bits()) / 8.0, op.mac.as_ref().map_or(0.0, |m| prog.tensors[op.operands[m.out].tensor].dtype.elem_bits() as f64 / 8.0));
                let io = ((cvt + rest) * acc_b, cvt * out_b + rest * acc_b);
                let vw = self.vector_cycles(u, op, "run the contraction's conversions and scale passes", &[acc], work, io)?;
                let vdem = self.vector_demands(&vw)?;
                ct = self.push_task(TaskKind::Compute, oi_prog, 0.0, &[], &vdem, &[ct], 0.0);
                vw.charge(&mut fx.stats, nc.vec_ops as f64);
            }
            let st = &mut fx.stats;
            st.useful_macs += nc.useful_macs;
            st.issued_macs += nc.issued_macs;
            st.vec_ops += nc.vec_ops;
            st.transc_ops += nc.transc_ops;
            st.slices += 1;
            if st.mode.is_empty() {
                st.mode.clone_from(&nc.mode);
            }
            let (e_mac, e_pad, e_elem, e_transc) = view.phys.unit_energies(ui.unit, &nc.mode, nc.bits);
            // A contraction's vector work is charged on the vector unit that runs it.
            let own_vec = if op.mac.is_some() { 0.0 } else { nc.vec_ops as f64 };
            let (e, pad) = (nc.useful_macs as f64 * e_mac + own_vec * e_elem + nc.transc_ops as f64 * e_transc, (nc.issued_macs - nc.useful_macs) as f64 * e_pad);
            st.e_compute_j += e;
            st.padding_j += pad;
            st.e_by_unit.push((u, e, pad));
            for (oi, _) in op.operands.iter().enumerate().filter(|(_, o)| o.access.writes()) {
                out_sets.entry((oi, fps[oi])).or_default().push((si, u, ct));
            }
        }
        for ((oi, b), producers) in out_sets {
            let o = &op.operands[oi];
            let root = prog.root(o.tensor);
            let (_, owner, mut last) = producers[0];
            if producers.len() > 1 {
                let acc = op.kernel.accum.map_or(4.0, |a| f64::from(a.element_bits()) / 8.0);
                let part = b.elems() as f64 * acc;
                let mut preds = vec![last];
                for &(_, u, ct) in &producers[1..] {
                    if u == owner {
                        preds.push(ct);
                        continue;
                    }
                    let path = self.path(u, true, owner);
                    preds.push(self.route_task(oi_prog, &path, part, &[ct], &mut fx, Some(u), Some(owner))?.unwrap_or(ct));
                }
                let n = (producers.len() - 1) as f64 * b.elems() as f64;
                let acc = op.kernel.accum.unwrap_or(kiln_ir::precision::Precision::Fp32);
                // Each add reads two partial sums and writes one.
                let acc_b = f64::from(acc.element_bits()) / 8.0;
                let vw = self.vector_cycles(owner, op, "add the partial sums of a split reduction", &[acc], |k| n / k.class_rate(OpClass::Elementwise), (2.0 * n * acc_b, n * acc_b))?;
                fx.stats.vec_ops += n as u128;
                vw.charge(&mut fx.stats, n);
                let dem = self.vector_demands(&vw)?;
                last = self.push_task(TaskKind::Reduce, oi_prog, 0.0, &[], &dem, &preds, 0.0);
            }
            if self.lifetimes[root] == Lifetime::Private {
                fx.private.push((root, b, owner, last));
                continue;
            }
            let aliased = root != o.tensor;
            for hi in 0..self.homes[root].len() {
                let (region, g) = (&self.homes[root][hi].0, self.homes[root][hi].1);
                // Each home receives the part of the slice's output it holds.
                let x = if aliased || region.n != b.n || region.covers(&b) {
                    b
                } else {
                    match region.intersect(&b) {
                        Some(x) => x,
                        None => continue,
                    }
                };
                let bytes = prog.tensors[o.tensor].bytes(x.elems()) as f64;
                let mut path = self.path(g, false, owner).to_vec();
                path.reverse();
                let w = self.route_task(oi_prog, &path, bytes, &[last], &mut fx, Some(owner), None)?.unwrap_or(last);
                fx.writers.push((root, x, w));
            }
        }
        let est = self.estimate(t0);
        fx.stats.op = oi_prog;
        fx.stats.group = self.cur_group;
        fx.stats.units = pool_units.len() as u32;
        fx.stats.peak_macs_per_s = pool_units
            .iter()
            .map(|&u| {
                let ui = &view.units[u];
                let k = &view.hw.units[ui.unit].spec;
                let mode = k.precisions.iter().map(|m| k.kind.ops_per_cycle(m)).fold(0.0, f64::max);
                if k.kind.is_mac() { mode * view.clock_hz(ui.clock) * ui.members.len() as f64 } else { 0.0 }
            })
            .sum();
        if commit {
            fx.stats.level_bytes = fx.level.iter().enumerate().filter(|(_, b)| **b > 0.0).map(|(g, b)| (g, *b)).collect();
            let mut by_t: BTreeMap<usize, NewLocs> = BTreeMap::new();
            for (t, b, u, task) in fx.private {
                by_t.entry(t).or_default().0.push((b, (u, task)));
            }
            for (t, b, task) in fx.writers {
                by_t.entry(t).or_default().1.push((b, task));
            }
            self.staged.extend(local_staged);
            for (t, (p, w)) in by_t {
                if self.private[t].is_empty() && self.writers[t].is_empty() {
                    self.touched.push(t);
                }
                // Staged copies of a region just written are stale: later readers fetch it again after the write.
                let stale = |x: &TBox| p.iter().map(|e| &e.0).chain(w.iter().map(|e| &e.0)).any(|b| b.n != x.n || b.intersect(x).is_some());
                self.staged.retain(|k, _| prog.root(k.1) != t || !stale(&k.2));
                if !p.is_empty() {
                    self.private[t].extend(p);
                }
                if !w.is_empty() {
                    self.writers[t].extend(w);
                }
            }
            self.counted.extend(fx.counted);
            self.g.ops.push(fx.stats);
        } else {
            self.g.tasks.truncate(t0);
            self.g.demands.truncate(d0);
            self.g.preds.truncate(p0);
            if self.g.profiles.len() > nprof {
                for id in self.prof_ids.iter_mut().filter(|id| **id != u32::MAX && (**id as usize) >= nprof) {
                    *id = u32::MAX;
                }
                self.g.profiles.truncate(nprof);
                self.pmax.truncate(nprof);
            }
        }
        Ok(est)
    }

    /// Compute resources and per-resource cycles of vector work beside MAC unit `u` (a fused convert's, a
    /// contraction's own conversions and scale passes, a split reduction's combine): on the vector unit (gang)
    /// sharing its feed memory, in its mode for `dtypes` ([`crate::cost::vector_mode`]) over `work` (lane
    /// operations at the unit's class rates); an error when the design has no such unit or mode. Also the bytes
    /// `(read, written)` it moves through its feeds and feed memories, spread over its gang, and the unit's energy
    /// per operation in that mode.
    fn vector_cycles(
        &self,
        u: usize,
        op: &POp,
        what: &str,
        dtypes: &[kiln_ir::precision::Precision],
        work: impl Fn(&ComputeKind) -> f64,
        (rd, wr): (f64, f64),
    ) -> Result<VectorWork, Diagnostic> {
        let view = self.view;
        let v = self.reducer(u).ok_or_else(|| self.no_vector_unit(u, op, what))?;
        let vu = &view.units[view.units[v].lead];
        let spec = &view.hw.units[vu.unit].spec;
        let (dt, rate) = crate::cost::vector_mode(&view.hw, vu.unit, dtypes).map_err(|d| d.at(op.id.clone()))?;
        let lanes = spec.kind.base_ops_per_cycle() as f64 * rate * vu.members.len() as f64;
        let e_op = view.phys.unit_energies(vu.unit, dt.name(), (dt.element_bits(), dt.element_bits())).2;
        let share = 1.0 / vu.members.len() as f64;
        let (mut feed, mut writes) = (vec![], vec![]);
        for &m in &vu.members {
            let mi = &view.units[m];
            let port = |list: &[(OperandRole, ResId)], roles: &[OperandRole]| roles.iter().find_map(|r| list.iter().find(|x| x.0 == *r)).or(list.first()).map(|x| x.1);
            if let Some(r) = port(&mi.feed_in, &[OperandRole::In, OperandRole::Any]) {
                feed.push(Amount::Res(r, rd * share));
            }
            if let Some(r) = port(&mi.feed_out, &[OperandRole::Out, OperandRole::Any]) {
                feed.push(Amount::Res(r, wr * share));
            }
            feed.push(Amount::Res(view.res_of_mem[view.groups[mi.chain[0]].mems[0]], rd * share));
            writes.push((mi.chain[0], wr * share));
        }
        Ok(VectorWork { unit: view.units[v].lead, units: vu.members.iter().map(|&m| view.units[m].compute).collect(), cycles: (work(&spec.kind) / lanes).ceil(), feed, writes, e_op })
    }

    /// Demands of vector work: its compute cycles, feed traffic and feed-memory writes.
    fn vector_demands(&mut self, vw: &VectorWork) -> Result<Vec<Amount>, Diagnostic> {
        let mut d: Vec<Amount> = vw.units.iter().map(|&r| Amount::Res(r, vw.cycles)).chain(vw.feed.iter().copied()).collect();
        for &(g, b) in vw.writes.iter().filter(|w| w.1 > 0.0) {
            d.push(Amount::Prof(self.prof(g, g)?.0, b));
        }
        Ok(d)
    }

    fn no_vector_unit(&self, u: usize, op: &POp, what: &str) -> Diagnostic {
        let path = &self.view.hw.nodes[self.view.hw.units[self.view.units[u].unit].node].path;
        Diagnostic::error("E-MAP-OP-004", format!("no vector unit reads the feed memory of {path} to {what}"))
            .at(op.id.clone())
            .hint("add a vector unit fed from the MAC unit's feed memory, or avoid the split / conversion")
    }

    fn group_entity(&self, g: GroupIx) -> &str {
        let hw = &self.view.hw;
        &hw.nodes[hw.memories[self.view.groups[g].mems[0]].node].entity
    }

    /// The vector unit that reads MAC unit `owner`'s feed memory, where its operands and partial results live.
    fn reducer(&self, owner: usize) -> Option<usize> {
        let v = self.view;
        let feed = v.units[owner].chain[0];
        v.pool(Pool::Vector).into_iter().find(|&x| v.units[x].chain[0] == feed)
    }

    #[allow(clippy::too_many_arguments)]
    fn read(
        &mut self,
        op_ix: usize,
        oi: usize,
        tensor: usize,
        b: &TBox,
        u: usize,
        nc: &NestCost,
        cx: &OpCtx,
        local: &mut BTreeMap<StageKey, u32>,
        deps: &mut Vec<u32>,
        fold: &mut Vec<Amount>,
        fold_lat: &mut (f64, u32),
        fx: &mut Effects,
    ) -> Result<(), Diagnostic> {
        let prog = self.prog;
        let view = self.view;
        let root = prog.root(tensor);
        let aliased = root != tensor;
        let t = &prog.tensors[tensor];
        let ui = &view.units[u];
        let chain = &ui.chain;
        let fl = self.feed_level(u, oi, &prog.ops[op_ix], false);
        if self.lifetimes[root] == Lifetime::Private {
            // A floor estimate only bounds the estimate from below, so it may leave tasks out: transfers between
            // the units of fused ops (quadratic in slices for misaligned splits) are skipped.
            if self.floor {
                return Ok(());
            }
            let srcs: Vec<(TBox, usize, u32)> = self.private[root]
                .overlapping(b)
                .filter_map(|(pb, (pu, task))| {
                    let x = if aliased || pb.n != b.n { Some(*b) } else { pb.intersect(b) };
                    x.map(|x| (x, *pu, *task))
                })
                .collect();
            let whole = aliased || srcs.iter().any(|s| s.0 == *b);
            let gap: u128 = if whole { 0 } else { uncovered(b, srcs.iter().map(|s| &s.0)).iter().map(TBox::elems).sum() };
            if srcs.is_empty() || gap > 0 {
                return Err(Diagnostic::error("E-MAP-VAL-017", format!("{} reads {} elements of private tensor {} no earlier op of its span wrote", prog.ops[op_ix].id, if srcs.is_empty() { b.elems() } else { gap }, prog.tensors[root].id))
                    .at(prog.tensors[root].id.clone())
                    .hint("give the tensor a home, or fuse its producers into the same span"));
            }
            for (x, pu, task) in srcs {
                if pu == u {
                    deps.push(task);
                    continue;
                }
                let path = self.path(pu, true, u);
                let id = self.route_task(op_ix, &path, t.bytes(x.elems()) as f64, &[task], fx, Some(pu), Some(u))?.unwrap_or(task);
                deps.push(id);
            }
            return Ok(());
        }
        for hi in 0..self.homes[root].len() {
            let (region, g) = (&self.homes[root][hi].0, self.homes[root][hi].1);
            let x = if aliased || region.n != b.n || region.covers(b) {
                *b
            } else {
                match region.intersect(b) {
                    Some(x) => x,
                    None => continue,
                }
            };
            // Writers are earlier ops' tasks: they only order this op's tasks, which a speculative estimate ignores.
            let wdeps: Vec<u32> = if self.virt.is_some() { vec![] } else { self.writers[root].overlapping(&x).map(|w| w.1).collect() };
            let mut hops = [(0usize, 0usize, None::<usize>); 16];
            let nh = match chain.iter().position(|&c| c == g) {
                Some(i) => {
                    for h in 0..i.min(16) {
                        hops[h] = (chain[i - h], chain[i - h - 1], Some(i - h - 1));
                    }
                    i.min(16)
                }
                None => {
                    let p = self.path(g, false, u);
                    for (h, w) in p.windows(2).take(16).enumerate() {
                        hops[h] = (w[0], w[1], chain.iter().position(|&c| c == w[1]));
                    }
                    p.len().saturating_sub(1).min(16)
                }
            };
            // Hops end at the memory the unit reads this operand from (below it only other roles' feeds).
            let nh = hops[..nh].iter().position(|h| h.2.is_some_and(|l| l < fl)).unwrap_or(nh);
            let base = t.bytes(x.elems()) as f64;
            let mut prev: Option<u32> = None;
            for &(from, to, li) in &hops[..nh] {
                let w = [from, to];
                let mult = li.filter(|&l| l < ui.private).map_or(1.0, |l| nc.reread.get(l).and_then(|r| r.get(oi)).copied().unwrap_or(1.0));
                let bytes = base * mult;
                if li == Some(fl) && cx.sharers.get(&to).copied().unwrap_or(1) <= 1 {
                    let share = bytes / ui.members.len() as f64;
                    for &m in &ui.members {
                        let to = view.units[m].chain[fl];
                        let (pid, lat) = self.prof(w[0], to)?;
                        fold.push(Amount::Prof(pid, share));
                        if fold_lat.1 == u32::MAX || lat > fold_lat.0 {
                            *fold_lat = (lat, pid);
                        }
                        fx.level[to] += share;
                    }
                    continue;
                }
                let key = (to, tensor, x);
                if let Some(&id) = local.get(&key).or_else(|| self.staged.get(&key)) {
                    prev = Some(id);
                    continue;
                }
                let one = [prev.unwrap_or(0)];
                let preds: &[u32] = if prev.is_some() { &one } else { &wdeps };
                let id = self.route_task(op_ix, &w[..2], bytes, preds, fx, None, Some(u))?.expect("one hop");
                local.insert(key, id);
                prev = Some(id);
            }
            match prev {
                Some(p) => deps.push(p),
                None => deps.extend(wdeps),
            }
        }
        Ok(())
    }

    /// `max(sum of demands per resource, longest dependency chain)` over tasks appended since `t0`.
    fn estimate(&mut self, t0: usize) -> f64 {
        let e = match self.virt.take() {
            Some(e) => e,
            None => {
                let mut e = Est::start(t0, std::mem::take(&mut self.scratch), std::mem::take(&mut self.pscratch));
                for t in &self.g.tasks[t0..] {
                    e.add(self.view, &self.pmax, t.kind == TaskKind::Reduce, t.lat_s, self.g.demands_of(t), self.g.preds_of(t));
                }
                e
            }
        };
        let (m, busy, pbytes) = e.finish(self.view, &self.g.profiles);
        self.scratch = busy;
        self.pscratch = pbytes;
        m
    }

    pub fn begin_group(&mut self, label: String, g: &ExecGroup, iteration: Option<u32>) {
        self.staged.clear();
        self.counted.clear();
        if self.g.groups.last().is_none_or(|p| p.barrier_after) {
            for t in std::mem::take(&mut self.touched) {
                self.writers[t].clear();
                self.private[t].clear();
            }
        }
        self.cur_group = self.g.groups.len() as u32;
        let n = self.g.tasks.len() as u32;
        self.g.groups.push(TGroup {
            label,
            ops: vec![],
            tasks: (n, n),
            barrier_after: g.barrier_after,
            launch: g.launch,
            iteration,
            fused: matches!(g.kind, GroupKind::Fused { .. }),
        });
    }

    pub fn end_group(&mut self) {
        let n = self.g.tasks.len() as u32;
        let first = self.g.ops.iter().position(|o| o.group == self.cur_group).unwrap_or(self.g.ops.len());
        let gr = self.g.groups.last_mut().expect("group open");
        gr.tasks.1 = n;
        gr.ops = (first as u32..self.g.ops.len() as u32).collect();
    }

    pub fn units_of(&self, mapping: &Mapping, op: &str) -> Result<Vec<usize>, Diagnostic> {
        let set = mapping.set_of(op).ok_or_else(|| Diagnostic::error("E-MAP-VAL-003", "op has no unit set").at(op.to_string()))?;
        set.units
            .iter()
            .map(|p| self.view.unit_by_path(p).ok_or_else(|| Diagnostic::error("E-MAP-VAL-001", format!("unknown unit {p}")).at(op.to_string())))
            .collect()
    }

    pub fn finish(self) -> TaskGraph {
        self.g
    }
}

/// Deterministic lowering of a validated mapping (03 §3.2 steps 6-9).
pub fn lower(prog: &Program, view: &HwView, mapping: &Mapping, cost: &dyn UnitCostModel) -> Result<TaskGraph, Diagnostic> {
    let mut l = Lowerer::new(prog, view, mapping, cost)?;
    let mut sets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for g in &mapping.groups {
        let first = g.ops.first().and_then(|o| prog.op(o));
        let iteration = first.and_then(|i| prog.iteration_of_op(i));
        let label = first.map_or_else(String::new, |i| prog.nodes[prog.ops[i].node].path.clone());
        l.begin_group(label, g, iteration);
        for o in &g.ops {
            let i = prog.op(o).ok_or_else(|| Diagnostic::error("E-MAP-VAL-012", "unknown op in group").at(o.clone()))?;
            if !prog.placed(i) {
                continue;
            }
            let p = mapping.ops.get(o).ok_or_else(|| Diagnostic::error("E-MAP-VAL-002", "op has no placement").at(o.clone()))?;
            let set = match p.target {
                Target::Units { set } => set,
                Target::Host => return Err(Diagnostic::error("E-MAP-VAL-004", "host execution is unmodelled in v0").at(o.clone())),
            };
            let units = match sets.get(&set) {
                Some(u) => u.clone(),
                None => {
                    let u = l.units_of(mapping, o)?;
                    sets.insert(set, u.clone());
                    u
                }
            };
            l.lower_op(i, p, &units, true)?;
        }
        l.end_group();
    }
    Ok(l.finish())
}
