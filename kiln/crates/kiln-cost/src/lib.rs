//! Intra-unit loop-nest cost model (03 §2): a Rust-native port of ZigZag/LOMA concepts.
//!
//! [`cost`] searches spatial unrollings, loop orders, memory allocations and double buffering for one
//! [`OpNest`] on one [`UnitTemplate`]; [`evaluate`] costs a fixed [`Mapping`]; [`CostCache`] memoizes by
//! (template hash, exact shape class, objective, options).

pub mod energy;
mod model;
mod nest;
mod persist;
mod search;
mod template;
mod temporal;
mod types;

pub use persist::MODEL_HASH;
pub use search::accumulates;
pub use types::*;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::PrecisionSpec;
use kiln_ir::wl::Kernel;

use model::{ClassEval, Prep, bytes, evaluate_class};

/// Objective value, tie-break value, and the per-class results of one spatial candidate.
type Candidate = (f64, f64, Vec<(TileClass, ClassEval)>);

/// Searches the mapping space and returns the best entry under `query.objective`.
pub fn cost(query: &CostQuery) -> Result<CostEntry, Diagnostic> {
    cost_with(query, None)
}

/// The unit as the selected mode sees it: a `k` axis spanning `k_pack` elements per lane for packed modes.
fn packed(unit: &UnitTemplate, mode: usize) -> std::borrow::Cow<'_, UnitTemplate> {
    let pack = unit.modes[mode].k_pack;
    if pack <= 1 {
        return std::borrow::Cow::Borrowed(unit);
    }
    let mut u = unit.clone();
    for a in u.axes.iter_mut().filter(|a| a.name == "k") {
        a.size *= pack;
    }
    std::borrow::Cow::Owned(u)
}

fn cost_with(query: &CostQuery, memo: Option<(&ClassMemo, &CostKey)>) -> Result<CostEntry, Diagnostic> {
    let (unit, nest, opts) = (query.unit, query.nest, &query.options);
    let mode = search::select_mode(unit, nest)?;
    let unit = &*packed(unit, mode);
    let p = Prep::new(unit, nest, mode, opts.zigzag_compat)?;
    let fl = floors_with(&p)?;
    let floor_e = nest.useful_points() as f64 * f64::from(nest.macs_per_point) * unit.modes[mode].e_mac_j;
    let floor_pair = (fl.compute_cycles as f64, floor_e);
    let ragged = if opts.zigzag_compat { RaggedPolicy::Pad } else { opts.ragged };
    let mut stats = SearchStats::default();
    let cands = search::spatial_candidates(unit, nest, opts.budget.top_k_spatial);
    stats.spatial_candidates = cands.len() as u64;
    let mut best: Option<Candidate> = None;
    let mut first_err = None;
    let full: Vec<u64> = nest.dims.iter().map(|d| d.size).collect();
    let stop = (opts.budget.stop_at_floor && query.objective == Objective::Latency).then(|| port_floor(&p, fl.compute_cycles));
    'cand: for sp in &cands {
        if let (Some(f), Some(b)) = (stop, &best)
            && b.2.iter().map(|x| (x.1.issue + x.1.stall) as f64).sum::<f64>() <= f
        {
            break;
        }
        let mut parts = vec![];
        let bound = best.as_ref().map(|b| b.0);
        for (sizes, csp) in search::classes(nest, sp, ragged) {
            let stop = stop.filter(|_| sizes == full);
            let cfg = temporal::SearchCfg { obj: query.objective, budget: opts.budget, bound, floors: floor_pair, stop };
            let found = match memo {
                Some((m, key)) => m.search(key, &p, &sizes, &csp, &cfg, &mut stats),
                None => temporal::search_class(&p, &sizes, &csp, &cfg, &mut stats, &mut 0),
            };
            match found {
                Ok(Some(r)) => parts.push((TileClass { sizes, count: 1, spatial: csp, temporal: r.tm }, r.eval)),
                Ok(None) => continue 'cand,
                Err(e) => {
                    first_err.get_or_insert(e);
                    continue 'cand;
                }
            }
        }
        let cyc: f64 = parts.iter().map(|x| x.1.total_f).sum();
        let en: f64 = parts.iter().map(|x| x.1.energy.total_j).sum();
        let v = search::objective_value(query.objective, cyc, en, floor_pair.0, floor_pair.1);
        let tie = if query.objective == Objective::Energy { cyc } else { en };
        if best.as_ref().is_none_or(|b| v.total_cmp(&b.0).then(tie.total_cmp(&b.1)).is_lt()) {
            best = Some((v, tie, parts));
        }
    }
    let Some((_, _, parts)) = best else {
        return Err(first_err.unwrap_or_else(|| {
            Diagnostic::error("E-COST-INFEASIBLE", format!("no feasible mapping of the nest on {}", unit.name)).at(unit.name.clone())
        }));
    };
    Ok(assemble(&p, parts, fl, stats))
}

/// Costs a fixed mapping (differential testing, kiln-map re-costing at parameter corners).
pub fn evaluate(unit: &UnitTemplate, nest: &OpNest, mapping: &Mapping, options: &CostOptions) -> Result<CostEntry, Diagnostic> {
    let mode = search::select_mode(unit, nest)?;
    let unit = &*packed(unit, mode);
    let p = Prep::new(unit, nest, mode, options.zigzag_compat)?;
    let fl = floors_with(&p)?;
    if mapping.classes.is_empty() {
        return Err(Diagnostic::error("E-COST-MAPPING", "mapping has no tile classes"));
    }
    let mut parts = vec![];
    for c in &mapping.classes {
        if c.sizes.len() != nest.dims.len() {
            return Err(Diagnostic::error("E-COST-MAPPING", "tile class sizes do not match the nest dims"));
        }
        let mut ev = evaluate_class(&p, &c.sizes, &c.spatial, &c.temporal, true)?;
        if c.count > 1 {
            scale(&mut ev, c.count);
        }
        parts.push((c.clone(), ev));
    }
    Ok(assemble(&p, parts, fl, SearchStats::default()))
}

/// Compute and per-level bandwidth floors of a nest on a unit, independent of mapping (03 §8 I1/I2).
pub fn floors(unit: &UnitTemplate, nest: &OpNest) -> Result<Floors, Diagnostic> {
    let mode = search::select_mode(unit, nest)?;
    let unit = &*packed(unit, mode);
    floors_with(&Prep::new(unit, nest, mode, false)?)
}

/// A lower bound on the cycles of every mapping of a nest on a unit (the search's latency floor plus pipeline
/// fill and drain): what [`cost`] returns is never below it.
pub fn latency_floor(unit: &UnitTemplate, nest: &OpNest) -> Result<u64, Diagnostic> {
    let mode = search::select_mode(unit, nest)?;
    let unit = &*packed(unit, mode);
    let p = Prep::new(unit, nest, mode, false)?;
    let fl = floors_with(&p)?;
    Ok(port_floor(&p, fl.compute_cycles).floor() as u64 + unit.pipeline.fill + unit.pipeline.drain)
}

/// Per operand (index = nest operand) and level: lower bounds of the bytes [`cost`] reports read from and
/// written to that level for it, under any mapping (every element of an operand crosses each level of its
/// chain once: read out of it toward the array for inputs, and written into it from below for outputs and
/// from above for inputs below their home).
pub fn access_floors(unit: &UnitTemplate, nest: &OpNest) -> Result<Vec<Vec<(u64, u64)>>, Diagnostic> {
    let mode = search::select_mode(unit, nest)?;
    let unit = &*packed(unit, mode);
    let p = Prep::new(unit, nest, mode, false)?;
    let full: Vec<u64> = nest.dims.iter().map(|d| d.size).collect();
    let mut out = vec![vec![(0u64, 0u64); unit.levels.len()]; nest.operands.len()];
    for (si, s) in p.streams.iter().enumerate().filter(|(_, s)| !s.scale) {
        let b = bytes(s.footprint(&full), s.bits);
        let chain = &p.chains[si];
        for (j, &l) in chain.iter().enumerate() {
            let x = &mut out[s.operand][l];
            if s.is_output {
                x.1 = b;
            } else {
                x.0 = b;
                if j + 1 < chain.len() {
                    x.1 = b;
                }
            }
        }
    }
    Ok(out)
}

fn floors_with(p: &Prep) -> Result<Floors, Diagnostic> {
    let unit = p.unit;
    let m = &unit.modes[p.mode];
    let work = p.nest.useful_points() as f64 * p.ops_per_point as f64;
    let compute_cycles = (work / m.macs_per_cycle).ceil() as u64;
    let full: Vec<u64> = p.nest.dims.iter().map(|d| d.size).collect();
    let mut compulsory = vec![0u64; unit.levels.len()];
    for (si, s) in p.streams.iter().enumerate() {
        let b = bytes(s.footprint(&full), s.bits);
        for &l in &p.chains[si] {
            compulsory[l] += b;
        }
    }
    let bandwidth_cycles = unit
        .levels
        .iter()
        .zip(&compulsory)
        .map(|(lv, &c)| {
            let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(unit.axes[a].size)).product::<u64>().max(1);
            let bw: f64 = lv.ports.iter().map(|p| p.bytes_per_cycle).sum::<f64>() * n_inst as f64;
            if bw > 0.0 { c as f64 / bw } else { 0.0 }
        })
        .collect();
    Ok(Floors { compute_cycles, compulsory_bytes: compulsory, bandwidth_cycles })
}

/// Latency floor by port direction: every compulsory input byte is read out of each level of its chain and
/// written into every level below its home (outputs the reverse), through ports able to carry that direction.
fn port_floor(p: &Prep, compute: u64) -> f64 {
    let unit = p.unit;
    let full: Vec<u64> = p.nest.dims.iter().map(|d| d.size).collect();
    let (mut rd, mut wr) = (vec![0u64; unit.levels.len()], vec![0u64; unit.levels.len()]);
    for (si, s) in p.streams.iter().enumerate() {
        let b = bytes(s.footprint(&full), s.bits);
        let chain = &p.chains[si];
        for (j, &l) in chain.iter().enumerate() {
            let top = j + 1 == chain.len();
            let (out, inn) = if s.is_output { (&mut wr, &mut rd) } else { (&mut rd, &mut wr) };
            out[l] += b;
            if !top {
                inn[l] += b;
            }
        }
    }
    unit.levels.iter().enumerate().fold(compute as f64, |acc, (l, lv)| {
        let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(unit.axes[a].size)).product::<u64>().max(1);
        let bw = |d: PortDir| lv.ports.iter().filter(|x| x.dir == d).map(|x| x.bytes_per_cycle).sum::<f64>() * n_inst as f64;
        let (r, w, x) = (bw(PortDir::Read), bw(PortDir::Write), bw(PortDir::ReadWrite));
        let t = |bytes: u64, cap: f64| if bytes == 0 { 0.0 } else if cap > 0.0 { bytes as f64 / cap } else { f64::INFINITY };
        acc.max(t(rd[l], r + x)).max(t(wr[l], w + x)).max(t(rd[l] + wr[l], r + w + x))
    })
}

fn scale(ev: &mut ClassEval, k: u64) {
    ev.issue *= k;
    ev.stall *= k;
    ev.onload *= k;
    ev.offload *= k;
    ev.fill_drain *= k;
    ev.total_f *= k as f64;
    ev.useful *= k;
    ev.issued *= k;
    ev.vector_ops *= k;
    ev.conversion_ops *= k;
    for a in &mut ev.accesses {
        a.to_low *= k;
        a.from_high *= k;
        a.to_high *= k;
        a.from_low *= k;
        a.read_bytes *= k;
        a.write_bytes *= k;
        a.dir_bytes.iter_mut().for_each(|b| *b *= k);
    }
    let e = &mut ev.energy;
    let kf = k as f64;
    e.mac_j *= kf;
    e.idle_mac_j *= kf;
    e.vector_j *= kf;
    e.conversion_j *= kf;
    e.levels_j.iter_mut().for_each(|x| *x *= kf);
    e.total_j *= kf;
}

fn assemble(p: &Prep, parts: Vec<(TileClass, ClassEval)>, floors: Floors, search: SearchStats) -> CostEntry {
    let unit = p.unit;
    let mut accesses: Vec<LevelAccess> = vec![];
    let mut energy = EnergyBreakdown { levels_j: vec![0.0; unit.levels.len()], ..Default::default() };
    let (mut issue, mut stall, mut fd, mut useful, mut issued, mut vec_ops, mut conv) = (0, 0, 0, 0, 0, 0, 0);
    let mut total_f = 0.0;
    let mut limiter_src: Option<(u64, Limiter)> = None;
    let mut classes = vec![];
    for (tc, ev) in parts {
        issue += ev.issue;
        stall += ev.stall;
        fd += ev.onload + ev.offload + ev.fill_drain;
        useful += ev.useful;
        issued += ev.issued;
        vec_ops += ev.vector_ops;
        conv += ev.conversion_ops;
        total_f += ev.total_f;
        for a in &ev.accesses {
            match accesses.iter_mut().find(|x| x.level == a.level && x.operand == a.operand) {
                Some(x) => {
                    x.to_low += a.to_low;
                    x.from_high += a.from_high;
                    x.to_high += a.to_high;
                    x.from_low += a.from_low;
                    x.read_bytes += a.read_bytes;
                    x.write_bytes += a.write_bytes;
                    for (y, b) in x.dir_bytes.iter_mut().zip(a.dir_bytes) {
                        *y += b;
                    }
                }
                None => accesses.push(a.clone()),
            }
        }
        energy.mac_j += ev.energy.mac_j;
        energy.idle_mac_j += ev.energy.idle_mac_j;
        energy.vector_j += ev.energy.vector_j;
        energy.conversion_j += ev.energy.conversion_j;
        for (x, y) in energy.levels_j.iter_mut().zip(&ev.energy.levels_j) {
            *x += y;
        }
        energy.total_j += ev.energy.total_j;
        if let Some(l) = ev.limiter.clone()
            && limiter_src.as_ref().is_none_or(|(c, _)| ev.cycles() > *c)
        {
            limiter_src = Some((ev.cycles(), l));
        }
        classes.push(tc);
    }
    let cycles = if p.compat { total_f.ceil() as u64 } else { issue + stall + fd };
    let m = &unit.modes[p.mode];
    let limiter = if issue >= stall && issue >= fd {
        Limiter::Compute
    } else if stall >= fd {
        match limiter_src {
            Some((_, l @ Limiter::Port { .. })) => l,
            _ => Limiter::Port { level: 0, port: 0 },
        }
    } else {
        Limiter::FillDrain
    };
    CostEntry {
        cycles,
        latency_s: total_f / unit.clock_hz,
        issue_cycles: issue,
        stall_cycles: stall,
        fill_drain_cycles: fd,
        useful_macs: useful,
        issued_macs: issued,
        vector_ops: vec_ops,
        conversion_ops: conv,
        mode: p.mode,
        accesses,
        energy,
        energy_source: unit.energy_source,
        spatial_util: if issued > 0 { useful as f64 / issued as f64 } else { 0.0 },
        utilization: if cycles > 0 { useful as f64 / (cycles as f64 * m.macs_per_cycle) } else { 0.0 },
        limiter,
        floors,
        mapping: Mapping { classes },
        search,
    }
}

impl UnitTemplate {
    /// Builds the template of `unit` (and its gang) from the expanded hardware model.
    pub fn from_hw(hw: &HwModel, unit: usize, options: &TemplateOptions) -> Result<UnitTemplate, Diagnostic> {
        template::from_hw(hw, unit, options)
    }

    /// `ut1-` content hash over the canonical JSON of the template.
    pub fn hash(&self) -> String {
        template::hash(self)
    }
}

impl OpNest {
    /// Converts a lowered kernel. `dtypes[i]` is the storage precision of `kernel.operands[i]`; `roles`
    /// overrides the derived roles (contraction: first read `A`, second read `B`, write `O`).
    pub fn from_kernel(kernel: &Kernel, dtypes: &[PrecisionSpec], roles: Option<&[OperandRole]>) -> Result<OpNest, Diagnostic> {
        OpNest::from_kernel_impl(kernel, dtypes, roles)
    }

    /// Converts kernel `kernel` of a lowered graph node, taking operand dtypes from the node's bound inputs,
    /// outputs and temporaries.
    pub fn from_lowered(node: &kiln_wl::LoweredNode, kernel: usize) -> Result<OpNest, Diagnostic> {
        let k = node.lowered.kernels.get(kernel).ok_or_else(|| {
            Diagnostic::error("E-COST-NEST", format!("node {} has no kernel {kernel}", node.path)).at(node.path.clone())
        })?;
        let n = &node.node;
        let dtypes = k
            .operands
            .iter()
            .map(|o| {
                let ti = n
                    .inputs
                    .iter()
                    .position(|t| *t == o.tensor)
                    .map(|i| &node.inputs[i])
                    .or_else(|| n.outputs.iter().position(|t| *t == o.tensor).map(|i| &node.outputs[i]))
                    .or_else(|| node.lowered.temps.iter().find(|t| t.0 == o.tensor).map(|t| &t.1));
                ti.map(|t| PrecisionSpec::new(t.dtype.scalar)).ok_or_else(|| {
                    Diagnostic::error("E-COST-NEST", format!("kernel {} operand {} has no bound type", k.id, o.tensor))
                        .at(node.path.clone())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        OpNest::from_kernel(k, &dtypes, None)
    }

    /// Nest operand of each costed stream ([`LevelAccess::operand`] indexes streams: the operands, then the
    /// MX/block scale streams of the inputs that have them).
    pub fn stream_operands(&self) -> Vec<usize> {
        nest::streams(self, 0).iter().map(|s| s.operand).collect()
    }

    /// The same nest restricted to a tile: `sizes[d]` replaces the size of dim `d`. `points` is rescaled only
    /// for box domains; masked tiles pass their exact count.
    pub fn tile(&self, sizes: &[u64], points: Option<u64>) -> OpNest {
        self.tile_impl(sizes, points)
    }
}

/// Memo key (03 §2.8): exact shape class with names stripped, so renaming dims or tensors hits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CostKey {
    pub unit_template: String,
    pub shape: String,
    pub objective: Objective,
    pub options: CostOptions,
}

impl CostKey {
    pub fn new(query: &CostQuery) -> CostKey {
        CostKey::with_template_hash(query, query.unit.hash())
    }

    /// As [`CostKey::new`] with the template hash precomputed by the caller (`UnitTemplate::hash`).
    pub fn with_template_hash(query: &CostQuery, unit_template: String) -> CostKey {
        let mut n = query.nest.clone();
        n.dims.iter_mut().for_each(|d| d.name = String::new());
        n.operands.iter_mut().for_each(|o| o.tensor = String::new());
        // Dim names still matter where a template axis restricts dims by name; keep the class instead.
        let classes: Vec<Option<&str>> = (0..query.nest.dims.len()).map(|d| nest::dim_class(query.nest, d)).collect();
        let named: Vec<&str> = query
            .nest
            .dims
            .iter()
            .map(|d| if query.unit.axes.iter().any(|a| a.allowed.contains(&d.name)) { d.name.as_str() } else { "" })
            .collect();
        let shape = serde_json::to_string(&(&n, classes, named)).expect("nest serializes");
        CostKey { unit_template, shape, objective: query.objective, options: query.options }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub truncated: u64,
}

type Slot = Result<Arc<CostEntry>, Diagnostic>;

/// Template, shape, objective, ZigZag compat, tile class sizes, spatial mapping, bound and stop (bits).
type ClassKey = (String, String, Objective, bool, Vec<u64>, SpatialMapping, Option<u64>, Option<u64>);

struct ClassMemoEntry {
    found: Result<Option<temporal::ClassResult>, Diagnostic>,
    stats: SearchStats,
    evals: u64,
    budget: u64,
}

/// Class searches shared by the queries of one cache: the full search of a shape repeats the quick search's
/// first spatial candidates exactly (same class, spatial mapping, bound), as do queries differing only in budget.
/// A stored search answers a budget it provably does not depend on: it finished within that budget, or it was
/// truncated at exactly that budget.
#[derive(Default)]
struct ClassMemo {
    map: std::sync::Mutex<BTreeMap<ClassKey, ClassMemoEntry>>,
}

impl std::fmt::Debug for ClassMemo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassMemo").finish_non_exhaustive()
    }
}

/// Entries kept before the class memo starts over (each holds one class evaluation).
const CLASS_MEMO_CAP: usize = 100_000;

impl ClassMemo {
    fn search(
        &self,
        key: &CostKey,
        p: &Prep,
        sizes: &[u64],
        csp: &SpatialMapping,
        cfg: &temporal::SearchCfg,
        stats: &mut SearchStats,
    ) -> Result<Option<temporal::ClassResult>, Diagnostic> {
        let k: ClassKey = (
            key.unit_template.clone(),
            key.shape.clone(),
            key.objective,
            key.options.zigzag_compat,
            sizes.to_vec(),
            csp.clone(),
            cfg.bound.map(f64::to_bits),
            cfg.stop.map(f64::to_bits),
        );
        let budget = cfg.budget.max_evals_per_spatial;
        if let Some(e) = self.map.lock().expect("class memo").get(&k)
            && (if e.stats.truncated { e.budget == budget } else { e.evals <= budget })
        {
            stats.evaluated += e.stats.evaluated;
            stats.pruned += e.stats.pruned;
            stats.truncated |= e.stats.truncated;
            return e.found.clone();
        }
        let mut local = SearchStats::default();
        let mut evals = 0;
        let found = temporal::search_class(p, sizes, csp, cfg, &mut local, &mut evals);
        stats.evaluated += local.evaluated;
        stats.pruned += local.pruned;
        stats.truncated |= local.truncated;
        let mut m = self.map.lock().expect("class memo");
        if m.len() >= CLASS_MEMO_CAP {
            m.clear();
        }
        m.insert(k, ClassMemoEntry { found: found.clone(), stats: local, evals, budget });
        found
    }
}

/// Thread-safe memo of [`cost`] results. Insertion races are harmless: the search is a pure function of the key.
/// Optionally backed by a file shared across processes and designs ([`CostCache::open`], [`CostCache::save`]).
#[derive(Debug, Default)]
pub struct CostCache {
    /// Entry and its last-use tick.
    map: RwLock<BTreeMap<CostKey, (Slot, AtomicU64)>>,
    classes: ClassMemo,
    tick: AtomicU64,
    file: Option<(std::path::PathBuf, u64)>,
    hits: AtomicU64,
    misses: AtomicU64,
    truncated: AtomicU64,
}

impl CostCache {
    pub fn new() -> CostCache {
        CostCache::default()
    }

    /// Default size bound of a cache file.
    pub const DEFAULT_FILE_BYTES: u64 = 64 << 20;

    /// A cache backed by the file at `path`: its entries for this model version ([`MODEL_HASH`]) are loaded now
    /// (a missing or foreign file starts empty), and [`CostCache::save`] writes back at most `max_bytes`, the
    /// most recently used entries first.
    pub fn open(path: impl Into<std::path::PathBuf>, max_bytes: u64) -> CostCache {
        let path = path.into();
        let mut map = BTreeMap::new();
        let mut last = 0;
        for (k, r, t) in persist::read(&path) {
            last = last.max(t);
            map.insert(k, (r.map(Arc::new), AtomicU64::new(t)));
        }
        CostCache { map: RwLock::new(map), tick: AtomicU64::new(last + 1), file: Some((path, max_bytes)), ..CostCache::default() }
    }

    /// Writes the entries to the backing file, merged with entries other processes saved meanwhile; no-op for a
    /// cache without a file.
    pub fn save(&self) -> std::io::Result<()> {
        let Some((path, max_bytes)) = &self.file else { return Ok(()) };
        let entries: Vec<(CostKey, String, u64)> = self
            .map
            .read()
            .expect("cache lock")
            .iter()
            .map(|(k, (v, t))| {
                let t = t.load(Ordering::Relaxed);
                (k.clone(), persist::line(k, v, t), t)
            })
            .collect();
        persist::write(path, entries, *max_bytes)
    }

    pub fn query(&self, query: &CostQuery) -> Result<Arc<CostEntry>, Diagnostic> {
        self.query_key(query, CostKey::new(query))
    }

    /// As [`CostCache::query`] with the template hash precomputed once per template (saves hashing it per call).
    pub fn query_hashed(&self, query: &CostQuery, template_hash: &str) -> Result<Arc<CostEntry>, Diagnostic> {
        self.query_key(query, CostKey::with_template_hash(query, template_hash.to_owned()))
    }

    fn query_key(&self, query: &CostQuery, key: CostKey) -> Result<Arc<CostEntry>, Diagnostic> {
        if let Some((v, t)) = self.map.read().expect("cache lock").get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            t.store(self.tick.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
            return v.clone();
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(path) = std::env::var_os("KILN_COST_DUMP") {
            dump(&path, query, &key.unit_template);
        }
        let v = cost_with(query, Some((&self.classes, &key))).map(Arc::new);
        if v.as_ref().is_ok_and(|e| e.search.truncated) {
            self.truncated.fetch_add(1, Ordering::Relaxed);
        }
        let t = AtomicU64::new(self.tick.fetch_add(1, Ordering::Relaxed));
        self.map.write().expect("cache lock").entry(key).or_insert_with(|| (v.clone(), t));
        v
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            truncated: self.truncated.load(Ordering::Relaxed),
        }
    }

    pub fn len(&self) -> usize {
        self.map.read().expect("cache lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Appends a cold query to the JSON-lines file `KILN_COST_DUMP` names (the template once per hash), for the
/// `search_equiv` harness that replays real workloads' queries against search changes.
fn dump(path: &std::ffi::OsStr, query: &CostQuery, template_hash: &str) {
    use std::io::Write;
    static DUMPED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut dumped = DUMPED.lock().expect("dump lock");
    let unit = if dumped.iter().any(|h| h == template_hash) {
        serde_json::Value::Null
    } else {
        dumped.push(template_hash.to_owned());
        serde_json::to_value(query.unit).unwrap_or_default()
    };
    let line = serde_json::json!({"tpl": template_hash, "unit": unit, "nest": query.nest, "objective": query.objective, "options": query.options});
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}
