//! Evaluation of one tile class under a fixed spatial + temporal mapping (03 §2.4-§2.6).
//!
//! Access counts follow ZigZag's loop-relevance algebra: an operand tile at a level is refilled once per
//! iteration of the loops above it, except that irrelevant loops directly above a level boundary keep the tile
//! stationary (ZigZag's merge-down). Counts are exact integers; bytes are `ceil(elements * bits / 8)`.

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::compute::OperandRole;

use crate::nest::{Stream, err};
use crate::types::*;

pub(crate) const MAX_DIMS: usize = 16;

/// Everything about a (unit, nest) pair that does not depend on the mapping.
pub(crate) struct Prep<'a> {
    pub unit: &'a UnitTemplate,
    pub nest: &'a OpNest,
    pub streams: Vec<Stream>,
    /// Per stream, the template levels it stages through (innermost first), truncated at its source/sink.
    pub chains: Vec<Vec<LevelIx>>,
    /// Per stream, per chain level: axes served (shared) by that level or any level below it.
    pub served: Vec<Vec<u64>>,
    /// Per stream, per chain level, per direction: port index on the level.
    pub ports: Vec<Vec<[Option<usize>; 4]>>,
    pub mode: usize,
    pub ops_per_point: u64,
    /// Work units per lane per cycle in the chosen mode.
    pub lane_rate: f64,
    pub compat: bool,
    /// Scale streams of the block-scaled inputs the mode does not apply in-array (a vector pass each).
    pub mx_emulated: Vec<usize>,
    pub needs_conversion: bool,
}

pub(crate) fn dir_ix(d: Direction) -> usize {
    match d {
        Direction::ToLow => 0,
        Direction::FromHigh => 1,
        Direction::ToHigh => 2,
        Direction::FromLow => 3,
    }
}

const DIRS: [Direction; 4] = [Direction::ToLow, Direction::FromHigh, Direction::ToHigh, Direction::FromLow];

fn reads(d: Direction) -> bool {
    matches!(d, Direction::ToLow | Direction::ToHigh)
}

pub(crate) fn port_for(level: &MemLevel, role: OperandRole, dir: Direction) -> Option<usize> {
    let explicit = level.ports.iter().position(|p| p.serves.iter().any(|&(r, d)| d == dir && (r == role || r == OperandRole::Any)));
    explicit.or_else(|| {
        level.ports.iter().position(|p| {
            p.serves.is_empty()
                && match p.dir {
                    PortDir::ReadWrite => true,
                    PortDir::Read => reads(dir),
                    PortDir::Write => !reads(dir),
                }
        })
    })
}

pub(crate) fn chain_for(unit: &UnitTemplate, role: OperandRole) -> Option<&OperandChain> {
    let find = |r: OperandRole| unit.chains.iter().find(|c| c.role == r);
    find(role)
        .or_else(|| match role {
            OperandRole::A | OperandRole::B | OperandRole::C => find(OperandRole::In),
            OperandRole::O => find(OperandRole::Out).or_else(|| find(OperandRole::C)),
            OperandRole::In => find(OperandRole::A),
            OperandRole::Out => find(OperandRole::O),
            OperandRole::Any => None,
        })
        .or_else(|| find(OperandRole::Any))
}

impl<'a> Prep<'a> {
    pub fn new(unit: &'a UnitTemplate, nest: &'a OpNest, mode: usize, compat: bool) -> Result<Prep<'a>, Diagnostic> {
        nest.validate()?;
        if nest.dims.len() > MAX_DIMS {
            return Err(err("E-COST-NEST", format!("more than {MAX_DIMS} loop dims")));
        }
        let m = unit.modes.get(mode).ok_or_else(|| err("E-COST-MODE", format!("mode {mode} out of range")))?;
        let acc_bits = unit.psum_precision.unwrap_or(m.acc).precision.element_bits().max(nest.accum.map_or(0, |a| a.element_bits()));
        let mut streams = crate::nest::streams(nest, acc_bits);
        for s in &mut streams {
            if let Some(&(_, n)) = unit.operand_run.iter().find(|x| x.0 == s.role) {
                s.run = n;
            }
        }
        if compat {
            streams.retain(|s| !s.scale);
        }
        let mx_emulated = if m.mx_native { vec![] } else { (0..streams.len()).filter(|&i| streams[i].scale).collect() };
        let mut chains = Vec::with_capacity(streams.len());
        let mut served = Vec::with_capacity(streams.len());
        let mut ports = Vec::with_capacity(streams.len());
        for s in &streams {
            let o = &nest.operands[s.operand];
            let ch = chain_for(unit, s.role).ok_or_else(|| {
                err("E-COST-CHAIN", format!("unit {} has no memory chain for operand role {:?} ({})", unit.name, s.role, o.tensor))
                    .hint("add a feed for this role to the unit")
            })?;
            let mut levels = ch.levels.clone();
            if levels.is_empty() {
                return Err(err("E-COST-CHAIN", format!("unit {}: empty chain for role {:?}", unit.name, s.role)));
            }
            let end = if o.is_output { o.sink } else { o.source };
            if let Some(lv) = end {
                let pos = levels.iter().position(|&l| l == lv).ok_or_else(|| {
                    err("E-COST-RESIDENCY", format!("operand {} resides at level {lv}, which is not on its chain {:?}", o.tensor, levels))
                })?;
                levels.truncate(pos + 1);
            }
            let all = (1u64 << unit.axes.len()) - 1;
            let mut acc = 0u64;
            let mut sv = vec![];
            for &l in &levels {
                let inst = unit.levels[l].instance_axes.iter().fold(0u64, |m, &a| m | (1 << a));
                acc |= all & !inst;
                sv.push(acc);
            }
            let mut pv = vec![];
            for &l in &levels {
                let mut row = [None; 4];
                for d in DIRS {
                    row[dir_ix(d)] = port_for(&unit.levels[l], s.role, d);
                }
                pv.push(row);
            }
            chains.push(levels);
            served.push(sv);
            ports.push(pv);
        }
        let ops_per_point = u64::from(match nest.kind {
            NestKind::Contraction => nest.macs_per_point,
            _ => nest.vector_ops_per_point,
        });
        let lanes: f64 = unit.axes.iter().map(|a| f64::from(a.size)).product();
        let out = &nest.operands[nest.operands.iter().position(|o| o.is_output).unwrap_or(0)];
        let needs_conversion = !compat && !unit.fused_down_conversion && out.dtype.precision.element_bits() != acc_bits && nest.kind == NestKind::Contraction;
        Ok(Prep {
            unit,
            nest,
            streams,
            chains,
            served,
            ports,
            mode,
            ops_per_point,
            lane_rate: m.macs_per_cycle / lanes,
            compat,
            mx_emulated,
            needs_conversion,
        })
    }
}

/// Result of one tile class.
#[derive(Clone, Debug, Default)]
pub(crate) struct ClassEval {
    pub issue: u64,
    pub stall: u64,
    pub onload: u64,
    pub offload: u64,
    pub fill_drain: u64,
    /// Exact (possibly fractional) total in ZigZag-compatible mode.
    pub total_f: f64,
    pub useful: u64,
    pub issued: u64,
    pub vector_ops: u64,
    pub conversion_ops: u64,
    pub accesses: Vec<LevelAccess>,
    pub energy: EnergyBreakdown,
    pub limiter: Option<Limiter>,
}

impl ClassEval {
    pub fn cycles(&self) -> u64 {
        self.issue + self.stall + self.onload + self.offload + self.fill_drain
    }
}

struct Act {
    stream: usize,
    j: usize,
    dir: Direction,
    level: LevelIx,
    port: Option<usize>,
    period: u64,
    count: u64,
    /// Elements per instance per period.
    amount: u64,
    bits: u32,
    inst: u64,
    buffered: bool,
    /// Elements per instance of the startup (inputs) or final-drain (outputs) chunk: the level-0 tile.
    chunk: u64,
}

pub(crate) fn bits_ceil_div(amount: u64, bits: u32, bytes_per_cycle: f64) -> u64 {
    let num = u128::from(amount) * u128::from(bits);
    let den = bytes_per_cycle * 8.0;
    if den <= 0.0 {
        return 0;
    }
    if den.fract() == 0.0 && den < 1.8e19 {
        let d = den as u128;
        return num.div_ceil(d) as u64;
    }
    (num as f64 / den).ceil() as u64
}

pub(crate) fn bytes(elems: u64, bits: u32) -> u64 {
    (u128::from(elems) * u128::from(bits)).div_ceil(8) as u64
}

pub(crate) fn infeasible(level: &MemLevel, need: u64, cap: u64, culprit: Option<&str>) -> Diagnostic {
    let mut d = err(
        "E-COST-INFEASIBLE",
        format!("tiles need {need} B per instance of level {} but it holds {cap} B", level.name),
    )
    .at(level.name.clone());
    d = match culprit {
        Some(dim) => d.hint(format!("move {dim} inside or split {dim} across units")),
        None => d.hint("use smaller tiles or a larger level"),
    };
    d
}

/// Mapping-independent quantities of one spatial mapping of one tile class.
pub(crate) struct SpatialCtx {
    pub axis_unroll: Vec<u64>,
    /// Temporal iterations per dim.
    pub t: [u64; MAX_DIMS],
    pub full_ext: [u64; MAX_DIMS],
    /// Per stream, per chain level: spatial extent per dim over the axes served at or below that level.
    pub sp: Vec<Vec<[u64; MAX_DIMS]>>,
    /// Per stream, per chain level: instances of the level the mapping uses (unrolls on unserved axes).
    pub inst: Vec<Vec<u64>>,
}

impl SpatialCtx {
    pub fn new(p: &Prep, sizes: &[u64], spatial: &SpatialMapping) -> Result<SpatialCtx, Diagnostic> {
        let unit = p.unit;
        let nd = p.nest.dims.len();
        let na = unit.axes.len();
        if spatial.axes.len() > na {
            return Err(err("E-COST-MAPPING", format!("spatial mapping binds {} axes, unit has {na}", spatial.axes.len())));
        }
        if sizes.len() != nd {
            return Err(err("E-COST-MAPPING", "tile class sizes do not match the nest dims"));
        }
        let mut ad = vec![[1u64; MAX_DIMS]; na];
        let mut unroll = [1u64; MAX_DIMS];
        let mut axis_unroll = vec![1u64; na];
        for (a, binds) in spatial.axes.iter().enumerate() {
            for &(d, u) in binds {
                if d >= nd || u == 0 {
                    return Err(err("E-COST-MAPPING", format!("bad spatial binding ({d}, {u}) on axis {a}")));
                }
                ad[a][d] *= u;
                unroll[d] *= u;
                axis_unroll[a] *= u;
            }
            if axis_unroll[a] > u64::from(unit.axes[a].size) {
                return Err(err(
                    "E-COST-MAPPING",
                    format!("axis {} unrolls {} > size {}", unit.axes[a].name, axis_unroll[a], unit.axes[a].size),
                ));
            }
        }
        let mut t = [1u64; MAX_DIMS];
        let mut full_ext = [1u64; MAX_DIMS];
        for d in 0..nd {
            t[d] = sizes[d].div_ceil(unroll[d]);
            full_ext[d] = t[d] * unroll[d];
        }
        let mut sp = vec![];
        let mut inst = vec![];
        for served in &p.served {
            sp.push(
                served
                    .iter()
                    .map(|&mask| {
                        let mut e = [1u64; MAX_DIMS];
                        for (_, row) in ad.iter().enumerate().filter(|(a, _)| mask & (1 << a) != 0) {
                            for d in 0..nd {
                                e[d] *= row[d];
                            }
                        }
                        e
                    })
                    .collect(),
            );
            inst.push(served.iter().map(|&mask| (0..na).filter(|&a| mask & (1 << a) == 0).map(|a| axis_unroll[a]).product()).collect());
        }
        Ok(SpatialCtx { axis_unroll, t, full_ext, sp, inst })
    }
}

/// Prefix products of one loop order: `cyc[k]` cycles covered by loops `[0, k)`, `tp[k][d]` temporal extent of
/// dim `d` within them.
pub(crate) struct OrderCtx {
    pub cyc: Vec<u64>,
    pub tp: Vec<[u64; MAX_DIMS]>,
}

impl OrderCtx {
    pub fn new(p: &Prep, sc: &SpatialCtx, loops: &[TemporalLoop]) -> Result<OrderCtx, Diagnostic> {
        let nd = p.nest.dims.len();
        let nl = loops.len();
        let mut cyc = vec![1u64; nl + 1];
        let mut tp = vec![[1u64; MAX_DIMS]; nl + 1];
        for (k, l) in loops.iter().enumerate() {
            if l.dim >= nd || l.factor == 0 {
                return Err(err("E-COST-MAPPING", format!("bad temporal loop {l:?}")));
            }
            cyc[k + 1] = cyc[k] * l.factor;
            tp[k + 1] = tp[k];
            tp[k + 1][l.dim] *= l.factor;
        }
        if let Some(d) = (0..nd).find(|&d| tp[nl][d] != sc.t[d]) {
            return Err(err(
                "E-COST-MAPPING",
                format!("dim {}: temporal product {} != {} (size / spatial unroll)", p.nest.dims[d].name, tp[nl][d], sc.t[d]),
            ));
        }
        Ok(OrderCtx { cyc, tp })
    }

    pub fn ext(&self, k: usize, sp: &[u64; MAX_DIMS]) -> [u64; MAX_DIMS] {
        let mut e = self.tp[k];
        for d in 0..MAX_DIMS {
            e[d] *= sp[d];
        }
        e
    }
}

pub(crate) fn evaluate_class(
    p: &Prep,
    sizes: &[u64],
    spatial: &SpatialMapping,
    tm: &TemporalMapping,
    check_capacity: bool,
) -> Result<ClassEval, Diagnostic> {
    let sc = SpatialCtx::new(p, sizes, spatial)?;
    let oc = OrderCtx::new(p, &sc, &tm.loops)?;
    evaluate_with(p, sizes, &sc, &oc, tm, check_capacity, true)
}

pub(crate) fn evaluate_with(
    p: &Prep,
    sizes: &[u64],
    sc: &SpatialCtx,
    oc: &OrderCtx,
    tm: &TemporalMapping,
    check_capacity: bool,
    detail: bool,
) -> Result<ClassEval, Diagnostic> {
    let unit = p.unit;
    let loops = &tm.loops;
    let nl = loops.len();
    if tm.alloc.len() != p.streams.len() {
        return Err(err("E-COST-MAPPING", format!("alloc has {} streams, nest has {}", tm.alloc.len(), p.streams.len())));
    }
    let cyc = &oc.cyc;
    let ttot = cyc[nl];
    let full_ext = sc.full_ext;
    let axis_unroll = &sc.axis_unroll;

    let mut ev = ClassEval::default();
    let m = &unit.modes[p.mode];
    let useful_points = class_points(p.nest, sizes);
    ev.useful = useful_points * p.ops_per_point;
    ev.issue = if p.compat {
        ttot
    } else {
        ((ttot as f64) * (p.ops_per_point as f64) / p.lane_rate).ceil() as u64
    };
    ev.issued = if p.compat { ttot * axis_unroll.iter().product::<u64>() } else { (ev.issue as f64 * m.macs_per_cycle).round() as u64 };
    ev.issued = ev.issued.max(ev.useful);

    let mut acts: Vec<Act> = Vec::with_capacity(p.streams.len() * 8);
    // Per level, the bytes resident in each capacity share ([`MemLevel::share`]).
    let mut level_use: Vec<Vec<((usize, u64), u64)>> = vec![vec![]; unit.levels.len()];
    let mut level_culprit: Vec<Option<usize>> = vec![None; unit.levels.len()];
    ev.energy.levels_j = vec![0.0; unit.levels.len()];
    // ZigZag-compat side tables, per stream per chain level.
    let mut zz = ZzTables::default();

    for (si, s) in p.streams.iter().enumerate() {
        let chain = &p.chains[si];
        let n = chain.len();
        let alloc = &tm.alloc[si];
        if alloc.len() != n || alloc.windows(2).any(|w| w[0] > w[1]) || alloc[n - 1] != nl {
            return Err(err(
                "E-COST-MAPPING",
                format!("stream {si}: alloc {alloc:?} must be non-decreasing over {n} levels and end at {nl}"),
            ));
        }
        let db = tm.double_buffer.get(si);
        let dbuf = |j: usize| db.and_then(|v| v.get(j)).copied().unwrap_or(false);
        let (b, s0) = canonical_bounds(s, loops, alloc);
        let inst = &sc.inst[si];
        let sp = &sc.sp[si];
        let f: Vec<u64> = (0..n).map(|j| s.footprint(&oc.ext(b[j], &sp[j]))).collect();
        let dn: Vec<u64> = (0..n).map(|j| s.footprint(&oc.ext(if j == 0 { 0 } else { b[j - 1] }, &sp[j]))).collect();
        let per_dn: Vec<u64> = (0..n).map(|j| if j == 0 { cyc[s0] } else { cyc[b[j - 1]] }).collect();
        let per_up: Vec<u64> = (0..n).map(|j| cyc[b[j]]).collect();
        let total_dn = |j: usize| dn[j] * inst[j] * (ttot / per_dn[j]);
        let total_up = |j: usize| f[j] * inst[j] * (ttot / per_up[j]);

        let mut counts = vec![[0u64; 4]; n];
        let mut bits = vec![[s.bits; 4]; n];
        if s.is_output {
            let osize = s.footprint(&full_ext);
            for (j, c) in counts.iter_mut().enumerate() {
                c[3] = total_dn(j);
                c[0] = c[3].saturating_sub(osize);
                if j + 1 < n {
                    c[2] = total_up(j);
                }
            }
            for j in 0..n.saturating_sub(1) {
                counts[j][1] = counts[j + 1][0];
            }
            for (c, bt) in counts.iter().zip(bits.iter_mut()) {
                let here = if c[0] > 0 { s.acc_bits } else { s.bits };
                let above = if c[1] > 0 { s.acc_bits } else { s.bits };
                *bt = [here, above, above, here];
            }
        } else {
            for (j, c) in counts.iter_mut().enumerate() {
                c[0] = total_dn(j);
                if j + 1 < n {
                    c[1] = total_up(j);
                }
            }
            if p.compat {
                let bw0 = p.ports[si][0][0].map_or(0.0, |pt| unit.levels[chain[0]].ports[pt].bytes_per_cycle * 8.0);
                let boost0 = s.footprint(&oc.ext(0, &sp[0]));
                if bw0 < (boost0 * u64::from(s.bits)) as f64 {
                    counts[0][0] = dn[0] * inst[0] * ttot;
                }
            }
        }

        // A level the stream only passes through holds no tile: its transfers overlap compute exactly when the
        // level below them is buffered (native model only; ZigZag compat keeps its own rule).
        let mut buf = vec![false; n];
        for j in 0..n {
            let through = !p.compat && j > 0 && j + 1 < n && b[j] == b[j - 1];
            buf[j] = if through { buf[j - 1] } else { dbuf(j) };
        }
        for j in 0..n {
            let l = chain[j];
            let lv = &unit.levels[l];
            let c = counts[j];
            let bt = bits[j];
            let mut a = LevelAccess { level: l, operand: si, to_low: c[0], from_high: c[1], to_high: c[2], from_low: c[3], ..Default::default() };
            a.dir_bytes = [0, 1, 2, 3].map(|d| bytes(c[d], bt[d]));
            // 03 §2.4: of an output's writes up, those read back are partial sums at accumulator precision; the final
            // pass carries the result at output precision.
            let partial = |d: usize| if d == 3 { c[0] } else { c[1] };
            let split = |d: usize| s.is_output && !p.compat && (d == 2 || d == 3) && partial(d) > 0 && bt[d] != s.bits;
            for d in [2, 3] {
                if split(d) {
                    a.dir_bytes[d] = bytes(partial(d), bt[d]) + bytes(c[d].saturating_sub(partial(d)), s.bits);
                }
            }
            a.read_bytes = a.dir_bytes[0] + a.dir_bytes[2];
            a.write_bytes = a.dir_bytes[1] + a.dir_bytes[3];
            if !p.compat {
                ev.energy.levels_j[l] += a.read_bytes as f64 * lv.e_read_j_per_b + a.write_bytes as f64 * lv.e_write_j_per_b;
            }
            if detail {
                ev.accesses.push(a);
            }

            let tile_bits = if s.is_output && c[0] > 0 { s.acc_bits } else { s.bits };
            let mult = if dbuf(j) { 2 } else { 1 };
            let streams_through = j > 0 && j + 1 < n && b[j] == b[j - 1];
            if !streams_through {
                let (sh, need) = (lv.share(s.role), bytes(f[j], tile_bits) * mult);
                match level_use[l].iter_mut().find(|x| x.0 == sh) {
                    Some(x) => x.1 += need,
                    None => level_use[l].push((sh, need)),
                }
            }
            if level_culprit[l].is_none() && j + 1 < n {
                level_culprit[l] = loops.get(b[j].saturating_sub(1)).map(|x| x.dim);
            }

            let down_buf = j == 0 || buf[j - 1];
            let chunk = if n > 1 { dn[1] } else { dn[0] };
            let mut push = |dir: Direction, period: u64, amount: u64, buffered: bool| {
                let d = dir_ix(dir);
                let count = ttot / period;
                let act = |count: u64, bits: u32| Act { stream: si, j, dir, level: l, port: p.ports[si][j][d], period, count, amount, bits, inst: inst[j], buffered, chunk };
                if split(d) {
                    // The final pass first (the drain carries it), then the partial-sum passes.
                    let fin = ((u128::from(count) * u128::from(c[d].saturating_sub(partial(d)))).div_ceil(u128::from(c[d].max(1))) as u64).min(count);
                    acts.push(act(fin, s.bits));
                    acts.push(act(count - fin, bt[d]));
                } else {
                    acts.push(act(count, bt[d]));
                }
            };
            if s.is_output {
                push(Direction::FromLow, per_dn[j], dn[j], down_buf);
                if c[0] > 0 {
                    push(Direction::ToLow, per_dn[j], dn[j], down_buf);
                }
                if j + 1 < n {
                    push(Direction::ToHigh, per_up[j], f[j], buf[j]);
                    if c[1] > 0 {
                        push(Direction::FromHigh, per_up[j], f[j], buf[j]);
                    }
                }
            } else {
                push(Direction::ToLow, per_dn[j], dn[j], down_buf);
                if j + 1 < n {
                    push(Direction::FromHigh, per_up[j], f[j], buf[j]);
                }
            }
        }
        if p.compat {
            zz.push_stream(s, loops, &b, &f, &counts);
        }
    }

    if check_capacity && !p.compat {
        for (l, uses) in level_use.iter().enumerate() {
            let lv = &unit.levels[l];
            let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(unit.axes[a].size)).product::<u64>().max(1);
            for &((_, cap), need) in uses {
                if need > cap / n_inst {
                    let dim = level_culprit[l].map(|d| p.nest.dims[d].name.clone());
                    return Err(infeasible(lv, need, cap / n_inst, dim.as_deref()));
                }
            }
        }
    }

    let real = |a: &Act| -> u64 {
        a.port.map_or(0, |pt| bits_ceil_div(a.amount, a.bits, unit.levels[a.level].ports[pt].bytes_per_cycle))
    };

    if p.compat {
        zz_latency(p, &acts, &real, &mut zz, ttot, &mut ev);
        zz_energy(p, &acts, &mut ev);
    } else {
        native_latency(p, &acts, ttot, &mut ev);
    }

    ev.energy.mac_j = ev.useful as f64 * m.e_mac_j;
    if !p.compat {
        ev.energy.idle_mac_j = (ev.issued - ev.useful) as f64 * m.e_mac_j * unit.e_mac_idle_ratio;
        // 03 §2.7: one pass per (output element, reduction block), tail blocks included.
        let (mut ext, mut red_ext) = ([1u64; MAX_DIMS], [1u64; MAX_DIMS]);
        for (d, x) in p.nest.dims.iter().enumerate() {
            ext[d] = sizes[d];
            if x.kind == LoopKind::Reduction {
                red_ext[d] = sizes[d];
            }
        }
        let out = p.streams.iter().find(|s| s.is_output).expect("validated: one output");
        let blocks: u64 = p.mx_emulated.iter().map(|&i| p.streams[i].footprint(&red_ext)).sum();
        let box_pts: u64 = sizes.iter().product();
        let mx_ops = (u128::from(out.footprint(&ext) * blocks) * u128::from(useful_points) / u128::from(box_pts.max(1))) as u64;
        ev.vector_ops = useful_points * u64::from(p.nest.vector_ops_per_point) * u64::from(p.nest.kind == NestKind::Contraction) + mx_ops;
        if p.needs_conversion {
            let out = p.streams.iter().find(|s| s.is_output).expect("validated: one output");
            ev.conversion_ops = out.footprint(&full_ext);
        }
        ev.energy.vector_j = ev.vector_ops as f64 * unit.e_vector_op_j;
        ev.energy.conversion_j = ev.conversion_ops as f64 * unit.e_vector_op_j;
    }
    ev.energy.total_j = ev.energy.mac_j
        + ev.energy.idle_mac_j
        + ev.energy.vector_j
        + ev.energy.conversion_j
        + ev.energy.levels_j.iter().sum::<f64>();
    if !p.compat {
        ev.total_f = ev.cycles() as f64;
    }
    Ok(ev)
}

/// Effective level boundaries of a stream: irrelevant loops directly above a boundary keep the lower tile
/// stationary, so they belong to the lower level (ZigZag's merge-down). Also returns the number of leading
/// irrelevant loops at level 0 (input operands stay on the array's inputs across them).
pub(crate) fn canonical_bounds(s: &Stream, loops: &[TemporalLoop], alloc: &[usize]) -> (Vec<usize>, usize) {
    let nl = loops.len();
    let n = alloc.len();
    let mut b = alloc.to_vec();
    for j in 0..n.saturating_sub(1) {
        let mut x = if j > 0 { b[j].max(b[j - 1]) } else { b[j] };
        while x < nl && !s.relevant(loops[x].dim) {
            x += 1;
        }
        b[j] = x;
    }
    let s0 = if s.is_output { 0 } else { stationary_run(s, &loops[..b[0]]) };
    (b, s0)
}

/// Innermost loops of `loops` an input stays in the array over: irrelevant to it, at most `run` iterations.
pub(crate) fn stationary_run(s: &Stream, loops: &[TemporalLoop]) -> usize {
    let mut it = 1u64;
    loops
        .iter()
        .take_while(|l| {
            it = it.saturating_mul(l.factor);
            !s.relevant(l.dim) && it <= s.run
        })
        .count()
}

pub(crate) fn class_points(nest: &OpNest, sizes: &[u64]) -> u64 {
    let b: u64 = sizes.iter().product();
    match nest.points {
        Some(pts) if nest.box_points() > 0 => (u128::from(b) * u128::from(pts) / u128::from(nest.box_points())) as u64,
        _ => b,
    }
}

/// Port cycles of one act: whole sub-port transactions spread over the port's lanes.
fn lane_cycles(level: &MemLevel, pt: usize, amount: u64, bits: u32) -> f64 {
    let port = &level.ports[pt];
    let lanes = port.lanes.max(1);
    bits_ceil_div(amount, bits, port.bytes_per_cycle / f64::from(lanes)) as f64 / f64::from(lanes)
}

fn native_latency(p: &Prep, acts: &[Act], ttot: u64, ev: &mut ClassEval) {
    let unit = p.unit;
    // Act periods count temporal iterations; one iteration lasts `issue / ttot` cycles in the chosen mode.
    let cyc_per_iter = ev.issue as f64 / ttot.max(1) as f64;
    // Per port: unbuffered busy, buffered busy, rigid per-period excess (cycles).
    let mut ports: Vec<((LevelIx, usize), f64, f64, f64)> = vec![];
    for a in acts {
        let Some(pt) = a.port else { continue };
        let r = lane_cycles(&unit.levels[a.level], pt, a.amount, a.bits);
        let n = a.count as f64;
        let key = (a.level, pt);
        let i = match ports.iter().position(|e| e.0 == key) {
            Some(i) => i,
            None => {
                ports.push((key, 0.0, 0.0, 0.0));
                ports.len() - 1
            }
        };
        if a.buffered {
            ports[i].2 += r * n;
            ports[i].3 += (r - a.period as f64 * cyc_per_iter).max(0.0) * n;
        } else {
            ports[i].1 += r * n;
        }
    }
    let t = ev.issue as f64;
    let mut best: Option<((LevelIx, usize), f64)> = None;
    for &(key, bu, bb, rigid) in &ports {
        let st = bu + rigid.max(bb - t);
        if best.is_none_or(|(_, s)| st > s) {
            best = Some((key, st));
        }
    }
    ev.stall = best.map_or(0, |b| (b.1 - 1e-9).ceil().max(0.0) as u64);

    // Startup and final drain: the first level-0 tile must travel down every level before compute starts, the
    // last one up after it ends; everything else streams under compute and is already in the port busy times.
    let edge_cost = |want_out: bool| -> u64 {
        let mut worst = 0;
        for si in (0..p.streams.len()).filter(|&si| p.streams[si].is_output == want_out) {
            let n = p.chains[si].len();
            let mut t = 0;
            for j in 0..n.saturating_sub(1) {
                let (lo, hi) = if want_out { (Direction::ToHigh, Direction::FromLow) } else { (Direction::FromHigh, Direction::ToLow) };
                let find = |jj: usize, d: Direction| acts.iter().find(|a| a.stream == si && a.j == jj && a.dir == d);
                let chunk_cycles = |a: Option<&Act>| {
                    a.and_then(|a| a.port.map(|pt| bits_ceil_div(a.chunk, a.bits, unit.levels[a.level].ports[pt].bytes_per_cycle)))
                        .unwrap_or(0)
                };
                t += chunk_cycles(find(j, lo)).max(chunk_cycles(find(j + 1, hi))) + unit.levels[p.chains[si][j + 1]].latency_cycles;
            }
            worst = worst.max(t);
        }
        worst
    };
    ev.onload = edge_cost(false);
    ev.offload = edge_cost(true);

    let pl = unit.pipeline;
    let out_tiles = p
        .streams
        .iter()
        .position(|s| s.is_output)
        .map_or(1, |si| acts.iter().filter(|a| a.stream == si && a.j == 0 && a.dir == Direction::ToHigh).map(|a| a.count).sum::<u64>().max(1));
    ev.fill_drain = pl.fill + pl.drain + pl.issue_overhead * out_tiles;

    let fd = ev.onload + ev.offload + ev.fill_drain;
    ev.limiter = Some(if ev.issue >= ev.stall && ev.issue >= fd {
        Limiter::Compute
    } else if ev.stall >= fd {
        let (level, port) = best.map_or((0, 0), |b| b.0);
        Limiter::Port { level, port }
    } else {
        Limiter::FillDrain
    });
}

// ------------------------------------------------------------------------------------- ZigZag 3.8.5 compat

/// Per (stream, chain level) quantities ZigZag's latency model reads.
#[derive(Default)]
struct ZzTables {
    /// Per stream: per level tile footprint per instance (arch level j+1 data, elements).
    tile: Vec<Vec<u64>>,
    /// Per stream: product of relevant loops at the top of each level (top_r_loop_size, mem level j).
    top_r: Vec<Vec<u64>>,
    /// Per stream: product of irrelevant loops at the top of each level.
    top_ir: Vec<Vec<u64>>,
    /// Per stream: bits of the level-j tile as ZigZag's data_precision_dict sees it (arch j+1).
    tile_bits: Vec<Vec<u32>>,

}

impl ZzTables {
    fn push_stream(&mut self, s: &Stream, loops: &[TemporalLoop], b: &[usize], f: &[u64], counts: &[[u64; 4]]) {
        let n = b.len();
        let mut top_r = vec![1u64; n];
        let mut top_ir = vec![1u64; n];
        for j in 0..n {
            let lo = if j == 0 { 0 } else { b[j - 1] };
            for k in (lo..b[j]).rev() {
                if !s.relevant(loops[k].dim) {
                    break;
                }
                top_r[j] *= loops[k].factor;
            }
            for k in (lo..b[j]).rev() {
                if s.relevant(loops[k].dim) {
                    break;
                }
                top_ir[j] *= loops[k].factor;
            }
        }
        let tile_bits = (0..n)
            .map(|j| {
                if !s.is_output {
                    s.bits
                } else if counts.get(j + 1).is_some_and(|c| c[0] > 0) {
                    s.acc_bits
                } else {
                    s.bits
                }
            })
            .collect();
        self.tile.push(f.to_vec());
        self.top_r.push(top_r);
        self.top_ir.push(top_ir);
        self.tile_bits.push(tile_bits);
    }
}

fn zz_latency(p: &Prep, acts: &[Act], real: &dyn Fn(&Act) -> u64, zz: &mut ZzTables, ttot: u64, ev: &mut ClassEval) {
    let unit = p.unit;
    let ns = p.streams.len();
    // Double-buffer heuristic (calc_double_buffer_flag): operands visited output first, then inputs in order.
    let mut order: Vec<usize> = (0..ns).filter(|&s| p.streams[s].is_output).collect();
    order.extend((0..ns).filter(|&s| !p.streams[s].is_output));
    let cap_bits = |l: LevelIx| {
        let lv = &unit.levels[l];
        let n_inst: u64 = lv.instance_axes.iter().map(|&a| u64::from(unit.axes[a].size)).product::<u64>().max(1);
        (lv.capacity_bytes / n_inst) as f64 * 8.0
    };
    let indiv: Vec<Vec<f64>> = (0..ns)
        .map(|s| {
            (0..p.chains[s].len())
                .map(|j| {
                    let bits = zz.tile[s][j] * u64::from(zz.tile_bits[s][j]);
                    (bits / zz.top_r[s][j]) as f64 / cap_bits(p.chains[s][j])
                })
                .collect()
        })
        .collect();
    let members = |l: LevelIx| -> Vec<(usize, usize)> {
        (0..ns).flat_map(|t| p.chains[t].iter().enumerate().filter(move |&(_, &x)| x == l).map(move |(k, _)| (t, k))).collect()
    };
    let mut shared: Vec<Vec<f64>> = (0..ns)
        .map(|s| p.chains[s].iter().map(|&l| members(l).iter().map(|&(t, k)| indiv[t][k]).sum()).collect())
        .collect();
    let mut dbl: Vec<Vec<bool>> = (0..ns).map(|s| vec![false; p.chains[s].len() + 1]).collect();
    for &s in &order {
        for j in 0..p.chains[s].len() {
            if shared[s][j] <= 0.5 {
                dbl[s][j + 1] = true;
            } else if indiv[s][j] <= 1.0 - shared[s][j] {
                dbl[s][j + 1] = true;
                let add = indiv[s][j];
                for (t, k) in members(p.chains[s][j]) {
                    shared[t][k] += add;
                }
            }
        }
    }
    let top_ir_of = |s: usize, arch: usize| if arch == 0 { 1 } else { zz.top_ir[s][arch - 1] };
    let allowed = |a: &Act| -> u64 {
        let (flag, scale) = match a.dir {
            Direction::ToLow | Direction::FromLow => (dbl[a.stream][a.j], top_ir_of(a.stream, a.j)),
            _ => (dbl[a.stream][a.j + 1], top_ir_of(a.stream, a.j + 1)),
        };
        if flag { a.period } else { (a.period as f64 / scale as f64) as u64 }
    };

    // Per physical port: stall/slack combine with memory-updating-window union.
    let mut keys: Vec<(LevelIx, usize)> = vec![];
    for a in acts {
        if let Some(pt) = a.port
            && !keys.contains(&(a.level, pt))
        {
            keys.push((a.level, pt));
        }
    }
    let mut ss_list: Vec<f64> = vec![0.0];
    for &key in &keys {
        let pa: Vec<(i128, i128, u64, u64)> = acts
            .iter()
            .filter(|a| a.port.map(|pt| (a.level, pt)) == Some(key) && a.count > 0)
            .map(|a| {
                let r = real(a) as i128;
                let al = allowed(a) as i128;
                (r, al, a.period, a.count)
            })
            .collect();
        match pa.len() {
            0 => {}
            1 => {
                let (r, al, _, c) = pa[0];
                ss_list.push(((r - al) * (c as i128 - 1)) as f64);
            }
            _ => {
                let union = muw_union(&pa);
                let (mut pos, mut neg, mut muw) = (0i128, 0i128, 0i128);
                for &(r, al, _, c) in &pa {
                    let ss = (r - al) * (c as i128 - 1);
                    if ss > 0 {
                        pos += ss;
                    } else {
                        neg += ss;
                    }
                    muw += al * (c as i128 - 1);
                }
                ss_list.push((pos + 0.max(neg + muw - union)) as f64);
            }
        }
    }
    let stall = ss_list.iter().copied().fold(f64::MIN, f64::max);

    // Onloading / offloading (calc_data_loading_latency).
    let total_comp = ttot as f64 + stall;
    let mut load: Vec<LoadEntry> = vec![];
    for &key in &keys {
        let on_port: Vec<&Act> = acts.iter().filter(|a| a.port.map(|pt| (a.level, pt)) == Some(key)).collect();
        let lv = &unit.levels[key.0];
        let port = &lv.ports[key.1];
        let bw_bits = port.bytes_per_cycle * 8.0;
        let inputs_served: Vec<usize> = {
            let mut v: Vec<usize> = on_port.iter().filter(|a| !p.streams[a.stream].is_output).map(|a| p.streams[a.stream].operand).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let shared2 = inputs_served.len() > 1;
        let cand: Vec<&&Act> = on_port
            .iter()
            .filter(|a| !(p.streams[a.stream].is_output && matches!(a.dir, Direction::ToLow | Direction::FromHigh)))
            .filter(|a| a.count > 0)
            .collect();
        let req_bw: f64 = cand
            .iter()
            .filter(|a| a.count > 1)
            .map(|a| ((a.amount * u64::from(a.bits)) as f64 / a.period as f64).floor())
            .sum();
        let ones: Vec<&&&Act> = cand.iter().filter(|a| a.count == 1).collect();
        let mins: Vec<f64> = ones
            .iter()
            .map(|a| ((zz.tile[a.stream][0] * u64::from(zz.tile_bits[a.stream][0])) as f64 / bw_bits).ceil())
            .collect();
        let totals: Vec<f64> = ones.iter().map(|a| real(a) as f64).collect();
        let surplus = (((bw_bits - req_bw) / bw_bits) * total_comp).max(0.0);
        let rem = reduce_balanced(&totals, &mins, surplus);
        for (a, r) in ones.iter().zip(rem) {
            load.push(((a.stream, a.j, a.dir), (r, shared2)));
        }
        for a in cand.iter().filter(|a| a.count > 1) {
            load.push(((a.stream, a.j, a.dir), (real(a) as f64, shared2)));
        }
    }
    let get = |s: usize, j: usize, d: Direction| -> (f64, bool) {
        load.iter().rev().find(|e| e.0 == (s, j, d)).map_or((0.0, false), |e| e.1)
    };
    let inputs: Vec<usize> = (0..ns).filter(|&s| !p.streams[s].is_output).collect();
    let mut parts: Vec<(f64, f64, f64)> = vec![];
    for &s in &inputs {
        let (mut ind, mut half, mut sh) = (0.0, 0.0, 0.0);
        for j in 0..p.chains[s].len().saturating_sub(1) {
            let e1 = get(s, j, Direction::FromHigh);
            let e2 = get(s, j + 1, Direction::ToLow);
            let longest = e1.0.max(e2.0);
            if e1.1 && e2.1 {
                sh += longest;
            } else if !e1.1 && !e2.1 {
                ind += longest;
            } else {
                half = longest;
            }
        }
        parts.push((ind, half, sh));
    }
    let onload = match parts.as_slice() {
        [] => 0.0,
        [(ind, _, _)] => *ind,
        [a, b] => {
            let p1 = a.2 + (b.2 + b.1 + b.0).max(a.1 + a.0);
            let p2 = b.2 + (a.2 + a.1 + a.0).max(b.1 + b.0);
            p1.min(p2)
        }
        many => many.iter().map(|x| x.0 + x.1 + x.2).fold(0.0, f64::max),
    };
    let mut offload = 0.0;
    for s in (0..ns).filter(|&s| p.streams[s].is_output) {
        for j in 0..p.chains[s].len().saturating_sub(1) {
            offload += get(s, j, Direction::ToHigh).0.max(get(s, j + 1, Direction::FromLow).0);
        }
    }
    ev.stall = stall.max(0.0).round() as u64;
    ev.onload = onload.ceil() as u64;
    ev.offload = offload.ceil() as u64;
    ev.total_f = ttot as f64 + stall + onload + offload;
    let fd = onload + offload;
    ev.limiter = Some(if ttot as f64 >= stall && ttot as f64 >= fd {
        Limiter::Compute
    } else if stall >= fd {
        Limiter::Port { level: keys.first().map_or(0, |k| k.0), port: 0 }
    } else {
        Limiter::FillDrain
    });
}

/// `((stream, level, direction), (cycles, port shared by two inputs))`.
type LoadEntry = ((usize, usize, Direction), (f64, bool));

fn muw_union(pa: &[(i128, i128, u64, u64)]) -> i128 {
    for &(_, al, per, c) in pa {
        if per as i128 == al {
            return al * c as i128;
        }
    }
    let (mut maxp, mut maxi) = (0u64, 0usize);
    for (i, &(_, _, per, _)) in pa.iter().enumerate() {
        if per > maxp {
            maxp = per;
            maxi = i;
        }
    }
    let mp = maxp as usize;
    let mut used = vec![false; mp];
    for &(_, al, per, _) in pa {
        let per = per as usize;
        if per == 0 {
            continue;
        }
        for blk in 0..mp / per {
            for x in 0..(al as usize).min(per) {
                used[blk * per + x] = true;
            }
        }
    }
    let union = used.iter().filter(|&&u| u).count() as i128;
    union * pa[maxi].3 as i128
}

fn reduce_balanced(c: &[f64], m: &[f64], mut s: f64) -> Vec<f64> {
    let mut out = c.to_vec();
    if s <= 0.0 || c.is_empty() {
        return out;
    }
    let mut idx: Vec<usize> = (0..c.len()).collect();
    idx.sort_by(|&a, &b| c[b].total_cmp(&c[a]));
    let mut cs: Vec<f64> = idx.iter().map(|&i| c[i]).collect();
    let ms: Vec<f64> = idx.iter().map(|&i| m[i]).collect();
    for i in 0..cs.len() {
        let maxr = (cs[i] - ms[i]) * (i + 1) as f64;
        if s >= maxr {
            let r = cs[i] - ms[i];
            for x in cs.iter_mut().take(i + 1) {
                *x -= r;
            }
            s -= maxr;
        } else {
            let r = s / (i + 1) as f64;
            for x in cs.iter_mut().take(i + 1) {
                *x -= r;
            }
            break;
        }
    }
    for (k, &i) in idx.iter().enumerate() {
        out[i] = cs[k];
    }
    out
}

/// ZigZag word-access energy: `ceil(amount * bits / bw) * periods * instances` accesses per direction, each
/// costing `e_per_byte * port_bytes_per_cycle` (r_cost / w_cost per word).
fn zz_energy(p: &Prep, acts: &[Act], ev: &mut ClassEval) {
    for a in acts {
        let Some(pt) = a.port else { continue };
        if a.count == 0 || a.amount == 0 || a.bits == 0 {
            continue;
        }
        let lv = &p.unit.levels[a.level];
        let bw = lv.ports[pt].bytes_per_cycle;
        let words = (bits_ceil_div(a.amount, a.bits, bw) * a.count * a.inst) as f64;
        let e = if reads(a.dir) { lv.e_read_j_per_b } else { lv.e_write_j_per_b };
        ev.energy.levels_j[a.level] += words * e * bw;
    }
}
