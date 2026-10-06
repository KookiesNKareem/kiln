//! `UnitTemplate` construction from the expanded hardware model (01 §5, §7, §16) and content hashing.

use kiln_ir::common::{Diagnostic, content_hash};
use kiln_ir::hw::HwModel;
use kiln_ir::hw::compute::{ComputeKind, Geometry, MemKind, OperandPolicy, OperandRole, PortDir as IrPortDir, PrecisionMode};
use kiln_ir::hw::model::{MemSpec, NodeIx};
use kiln_ir::precision::{PrecisionKind, PrecisionSpec};

use crate::energy;
use crate::nest::err;
use crate::types::*;

/// Bandwidth given to unit-local buffers, which have no declared ports: they never limit the array.
const LOCAL_BYTES_PER_CYCLE: f64 = (1u64 << 40) as f64;

fn axis(name: &str, size: u32, allowed: &[&str]) -> SpatialAxis {
    SpatialAxis { name: name.into(), size, allowed: allowed.iter().map(|s| s.to_string()).collect() }
}

fn axes_of(kind: &ComputeKind) -> Vec<SpatialAxis> {
    match kind {
        ComputeKind::Matrix(m) => match &m.geometry {
            Geometry::Systolic { rows, cols } => vec![axis("rows", *rows, &["k"]), axis("cols", *cols, &["n"])],
            Geometry::Mma { m, n, k } => vec![axis("m", *m, &["m"]), axis("n", *n, &["n"]), axis("k", *k, &["k"])],
            Geometry::OuterProduct { rows, cols } => vec![axis("rows", *rows, &["m"]), axis("cols", *cols, &["n"])],
            Geometry::Spatial { dims } => dims.iter().map(|(d, &s)| axis(d, s, &[d.as_str()])).collect(),
        },
        ComputeKind::Cim(c) => vec![axis("rows", c.active_rows(), &["k"]), axis("cols", c.cols, &["n"])],
        ComputeKind::Vector(v) => {
            let mut a = vec![axis("lanes", v.lanes, &[])];
            if v.sublanes > 1 {
                a.push(axis("sublanes", v.sublanes, &[]));
            }
            a
        }
        ComputeKind::Scalar(s) => vec![axis("issue", s.issue_width, &[])],
        ComputeKind::Special(s) => vec![axis("lanes", s.lanes, &[])],
    }
}

fn is_block(p: PrecisionSpec) -> bool {
    matches!(p.precision.kind(), PrecisionKind::Mx | PrecisionKind::BlockFloat)
}

/// Units forming the gang of `unit`: `gang` enabled siblings of the same entity under the nearest ancestor
/// holding at least that many.
fn gang_of(hw: &HwModel, unit: usize, gang: u32) -> Result<Vec<usize>, Diagnostic> {
    if gang <= 1 {
        return Ok(vec![unit]);
    }
    let me = &hw.nodes[hw.units[unit].node];
    let mut anc = me.parent;
    while let Some(a) = anc {
        let mut v: Vec<usize> = (0..hw.units.len())
            .filter(|&u| {
                let n = &hw.nodes[hw.units[u].node];
                n.enabled && n.entity == me.entity && under(hw, hw.units[u].node, a)
            })
            .collect();
        if v.len() >= gang as usize {
            let pos = v.iter().position(|&u| u == unit).unwrap_or(0);
            let start = pos / gang as usize * gang as usize;
            v = v.into_iter().skip(start).take(gang as usize).collect();
            if v.len() == gang as usize {
                return Ok(v);
            }
        }
        anc = hw.nodes[a].parent;
    }
    Err(err("E-COST-GANG", format!("no ancestor of {} holds {gang} units of its kind", me.path)))
}

fn under(hw: &HwModel, mut n: usize, root: usize) -> bool {
    loop {
        if n == root {
            return true;
        }
        match hw.nodes[n].parent {
            Some(p) => n = p,
            None => return false,
        }
    }
}

fn feed_mem(hw: &HwModel, unit: usize, role: OperandRole) -> Option<usize> {
    let f = &hw.units[unit].feeds;
    let pick = |r: OperandRole| f.get(&r).map(|x| x.mem);
    match role {
        OperandRole::O => pick(OperandRole::O).or_else(|| pick(OperandRole::C)),
        OperandRole::In => pick(OperandRole::In).or_else(|| pick(OperandRole::A)),
        OperandRole::Out => pick(OperandRole::Out).or_else(|| pick(OperandRole::O)),
        r => pick(r),
    }
    .or_else(|| pick(OperandRole::Any))
}

fn local_mem(hw: &HwModel, unit: usize, role: OperandRole) -> Option<usize> {
    hw.units[unit].local.iter().copied().find(|&m| match &hw.memories[m].spec {
        MemSpec::Local { buffer, .. } => buffer.holds == role || buffer.holds == OperandRole::Any,
        _ => false,
    })
}

/// Route bandwidths from one memory to another, B/s (None = not derivable).
#[derive(Clone, Copy)]
struct RouteBw {
    bottleneck: Option<f64>,
    first: Option<f64>,
    last: Option<f64>,
}

fn min_opt(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, y) => x.or(y),
    }
}

/// Memories directly routed from `cur` (through routers, ports, controllers, never another memory) along the
/// BFS route, which is the deterministic route of 01 §16.2.
fn routed_mems(hw: &HwModel, cur: usize) -> Vec<(usize, RouteBw)> {
    let start = hw.memories[cur].node;
    let mut seen: Vec<Option<RouteBw>> = vec![None; hw.nodes.len()];
    seen[start] = Some(RouteBw { bottleneck: None, first: None, last: None });
    let mut q = std::collections::VecDeque::from([start]);
    let mut out = vec![];
    while let Some(n) = q.pop_front() {
        let here = seen[n].expect("visited");
        for &c in &hw.out_edges[n] {
            let ch = &hw.channels[c];
            let m = hw.node_of(ch.dst);
            if seen[m].is_some() || !hw.nodes[m].enabled {
                continue;
            }
            let bw = ch.bandwidth.map(|b| b.0);
            let r = RouteBw { bottleneck: min_opt(here.bottleneck, bw), first: if n == start { bw } else { here.first }, last: bw };
            seen[m] = Some(r);
            match hw.nodes[m].ix {
                NodeIx::Mem(mi) => out.push((mi, r)),
                NodeIx::Unit(_) => {}
                _ => q.push_back(m),
            }
        }
    }
    out
}

/// Aggregate bandwidth (B/s) from `src` to the `dst` group: the sum of per-route bottlenecks, bounded by the
/// hop every route shares at `src` (first hop going out, last hop coming in).
fn group_bw(hw: &HwModel, cur: usize, group: &[usize]) -> (Option<f64>, Option<f64>) {
    let out = routed_mems(hw, cur);
    let (mut up, mut up_first) = (None, None);
    for &(m, r) in &out {
        if group.contains(&m) {
            up = Some(up.unwrap_or(0.0) + r.bottleneck.unwrap_or(f64::INFINITY));
            up_first = min_opt(up_first, r.first);
        }
    }
    let (mut down, mut down_last) = (None, None);
    for &g in group {
        if let Some(&(_, r)) = routed_mems(hw, g).iter().find(|x| x.0 == cur) {
            down = Some(down.unwrap_or(0.0) + r.bottleneck.unwrap_or(f64::INFINITY));
            down_last = min_opt(down_last, r.last);
        }
    }
    let fin = |x: Option<f64>| x.filter(|v| v.is_finite());
    (fin(min_opt(down, down_last)), fin(min_opt(up, up_first)))
}

/// The next staging level out from `cur`: its declared `backing` (as kiln-map's chains), else the lowest
/// derived level above `cur`'s among directly routed memories (off-chip stacks last), with every routed instance
/// of that entity aggregated (address-interleaved slices or stacks).
fn next_level(hw: &HwModel, cur: usize, home: Option<usize>) -> Option<Vec<usize>> {
    let reach = routed_mems(hw, cur);
    if let Some(h) = home
        && reach.iter().any(|r| r.0 == h)
    {
        return Some(vec![h]);
    }
    let backing: Vec<usize> = hw.memories[cur].backing.iter().copied().filter(|&b| hw.nodes[hw.memories[b].node].enabled).collect();
    if !backing.is_empty() {
        // Routed order first (it names the level), then any backing memory reached only through the miss path.
        let mut v: Vec<usize> = reach.iter().map(|r| r.0).filter(|m| backing.contains(m)).collect();
        let rest: Vec<usize> = backing.into_iter().filter(|m| !v.contains(m)).collect();
        v.extend(rest);
        return Some(v);
    }
    let lc = hw.level(cur);
    let best = reach
        .iter()
        .map(|r| r.0)
        .filter(|&m| hw.level(m) > lc && hw.level(m) != u8::MAX)
        .min_by_key(|&m| (hw.memories[m].is_stack(), hw.level(m), m))?;
    let ent = &hw.nodes[hw.memories[best].node].entity;
    Some(reach.iter().filter(|r| &hw.nodes[hw.memories[r.0].node].entity == ent).map(|r| r.0).collect())
}

struct Builder<'a> {
    hw: &'a HwModel,
    f_unit: f64,
    levels: Vec<MemLevel>,
    keys: Vec<Vec<usize>>,
}

impl Builder<'_> {
    /// Limits a level's read ports to the route bandwidth toward the unit (`down`, B/s) and its write ports to
    /// the route back (`up`); read-write ports to the larger of the two.
    fn cap(&mut self, l: LevelIx, down: Option<f64>, up: Option<f64>) {
        let f = self.f_unit;
        let ports = &mut self.levels[l].ports;
        let rw = match (down, up) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
        for (dir, limit) in [(PortDir::Read, down), (PortDir::Write, up), (PortDir::ReadWrite, rw)] {
            let Some(limit) = limit.map(|x| x / f) else { continue };
            let total: f64 = ports.iter().filter(|p| p.dir == dir).map(|p| p.bytes_per_cycle).sum();
            if total > limit && total > 0.0 {
                ports.iter_mut().filter(|p| p.dir == dir).for_each(|p| p.bytes_per_cycle *= limit / total);
            }
        }
    }

    fn level(&mut self, mems: Vec<usize>, instance_axes: Vec<usize>, copies: u32) -> LevelIx {
        let mut key = mems.clone();
        key.sort_unstable();
        if let Some(i) = self.keys.iter().position(|k| *k == key) {
            return i;
        }
        let hw = self.hw;
        let rep = mems[0];
        let mi = &hw.memories[rep];
        let f_mem = hw.clock_hz(mi.clock).map_or(self.f_unit, |h| h.0);
        let per_cycle = |bits_per_mem_cycle: f64| bits_per_mem_cycle / 8.0 * f_mem / self.f_unit;
        let capacity: u64 = mems.iter().map(|&m| hw.memories[m].capacity.0).sum::<u64>() * u64::from(copies);
        let path = hw.nodes[mi.node].path.clone();
        let name = if mems.len() > 1 { format!("{path}[x{}]", mems.len()) } else { path };
        let (ports, e_r, double_buffer, latency) = match &mi.spec {
            MemSpec::OnChip(m) => {
                let per_inst = mems.len() as f64;
                let mut ports: Vec<MemPort> = m
                    .ports
                    .iter()
                    .map(|p| {
                        let lanes = p.count * if p.per_bank { m.banks } else { 1 };
                        MemPort {
                            dir: match p.dir {
                                IrPortDir::Read => PortDir::Read,
                                IrPortDir::Write => PortDir::Write,
                                IrPortDir::Rw => PortDir::ReadWrite,
                            },
                            bytes_per_cycle: per_cycle(f64::from(lanes) * f64::from(p.width_bits)) * per_inst,
                            serves: vec![],
                            lanes: lanes * mems.len() as u32,
                        }
                    })
                    .collect();
                if let Some(bw) = m.overrides.bandwidth {
                    let total: f64 = ports.iter().map(|p| p.bytes_per_cycle).sum();
                    let target = bw.0 / self.f_unit * per_inst;
                    if total > 0.0 {
                        ports.iter_mut().for_each(|p| p.bytes_per_cycle *= target / total);
                    }
                }
                let e = m
                    .overrides
                    .read_energy
                    .map_or_else(|| energy::e_read_onchip(Some(m.kind), mi.capacity.0), |e| e.0);
                // Declared in memory cycles; the template counts unit cycles.
                let lat = m.overrides.latency.map_or(0, |c| (c.0 * self.f_unit / f_mem * (1.0 - 1e-12)).ceil() as u64);
                (ports, e, true, lat)
            }
            MemSpec::Stack(_) => {
                let bw: f64 = mems.iter().filter_map(|&x| hw.memories[x].bandwidth.map(|b| b.0)).sum();
                let p = MemPort { dir: PortDir::ReadWrite, bytes_per_cycle: bw / self.f_unit, serves: vec![], lanes: 1 };
                (vec![p], energy::E_DRAM_PER_BYTE, true, 0)
            }
            MemSpec::Local { buffer, .. } => {
                let p = MemPort { dir: PortDir::ReadWrite, bytes_per_cycle: LOCAL_BYTES_PER_CYCLE, serves: vec![], lanes: 1 };
                (vec![p], energy::e_read_onchip(Some(MemKind::RegisterFile), buffer.capacity.0), buffer.double_buffered, 0)
            }
        };
        let partitions = match &mi.spec {
            MemSpec::OnChip(m) => match &m.operands {
                OperandPolicy::Partitioned { parts } => parts.iter().map(|(&r, b)| (r, b.0 * mems.len() as u64 * u64::from(copies))).collect(),
                _ => vec![],
            },
            _ => vec![],
        };
        let e_w = match &mi.spec {
            MemSpec::OnChip(m) => m.overrides.write_energy.map_or(e_r * energy::WRITE_RATIO, |e| e.0),
            _ => e_r * energy::WRITE_RATIO,
        };
        self.levels.push(MemLevel {
            name,
            mem: Some(rep),
            capacity_bytes: capacity,
            ports,
            double_buffer,
            instance_axes,
            e_read_j_per_b: e_r,
            e_write_j_per_b: e_w,
            latency_cycles: latency,
            external: false,
            partitions,
        });
        self.keys.push(key);
        self.levels.len() - 1
    }
}

pub(crate) fn from_hw(hw: &HwModel, unit: usize, opts: &TemplateOptions) -> Result<UnitTemplate, Diagnostic> {
    let u = hw.units.get(unit).ok_or_else(|| err("E-COST-UNIT", format!("unit index {unit} out of range")))?;
    let path = hw.nodes[u.node].path.clone();
    let f_unit = hw
        .clock_hz(u.clock)
        .map(|h| h.0)
        .filter(|&f| f > 0.0)
        .ok_or_else(|| err("E-COST-CLOCK", format!("unit {path} has no clock")).at(path.clone()))?;
    let gang = gang_of(hw, unit, opts.gang)?;
    let g = gang.len() as u32;
    let kind = &u.spec.kind;
    let mut axes = axes_of(kind);
    let roles: &[OperandRole] =
        if kind.is_mac() { &[OperandRole::A, OperandRole::B, OperandRole::O] } else { &[OperandRole::In, OperandRole::Out] };
    // Members sharing their feed memories (two tensor cores on one register file) split the gang into an outer
    // axis over distinct feeds (the feed levels' instances) and an inner axis sharing them.
    let mut feed_sets: Vec<Vec<Option<usize>>> = gang.iter().map(|&x| roles.iter().map(|&r| feed_mem(hw, x, r)).collect()).collect();
    feed_sets.sort();
    feed_sets.dedup();
    let k1 = feed_sets.len() as u32;
    if k1 > 1 && k1 < g && (!g.is_multiple_of(k1) || feed_sets.iter().any(|f| gang.iter().filter(|&&x| roles.iter().map(|&r| feed_mem(hw, x, r)).collect::<Vec<_>>() == *f).count() as u32 != g / k1)) {
        return Err(err("E-COST-GANG", format!("gang of {g} units of {path} shares feeds unevenly")));
    }
    let (feed_axis, sub_axis) = match g {
        1 => (None, None),
        _ if k1 == 1 || k1 == g => {
            axes.push(axis("gang", g, &[]));
            (Some(axes.len() - 1), None)
        }
        _ => {
            axes.push(axis("gang", k1, &[]));
            axes.push(axis("gang_sub", g / k1, &[]));
            (Some(axes.len() - 2), Some(axes.len() - 1))
        }
    };
    let member_axes: Vec<usize> = feed_axis.into_iter().chain(sub_axis).collect();
    let modes = u
        .spec
        .precisions
        .iter()
        .map(|pm| {
            let mpc = kind.ops_per_cycle(pm) * f64::from(g);
            match *pm {
                PrecisionMode::Mac { a, b, acc, out, .. } => MacMode {
                    a,
                    b,
                    acc,
                    out,
                    macs_per_cycle: mpc,
                    e_mac_j: energy::e_mac(a.precision.max(b.precision)),
                    mx_native: is_block(a) || is_block(b),
                    k_pack: match kind {
                        ComputeKind::Matrix(m) if matches!(m.geometry, Geometry::Mma { .. }) => {
                            (16 / a.precision.element_bits().max(b.precision.element_bits()).max(1)).max(1)
                        }
                        _ => 1,
                    },
                },
                PrecisionMode::Elem { dtype, .. } => MacMode {
                    a: dtype,
                    b: dtype,
                    acc: dtype,
                    out: None,
                    macs_per_cycle: mpc,
                    e_mac_j: energy::E_VECTOR_OP,
                    mx_native: false,
                    k_pack: 1,
                },
            }
        })
        .collect();
    let mut b = Builder { hw, f_unit, levels: vec![], keys: vec![] };
    let mut chains = vec![];
    for &role in roles {
        let mut lv: Vec<LevelIx> = vec![];
        let per_member = |f: &dyn Fn(usize) -> Option<usize>| -> Option<Vec<usize>> { gang.iter().map(|&x| f(x)).collect() };
        if let Some(locals) = per_member(&|x| local_mem(hw, x, role)) {
            lv.push(b.level(vec![locals[0]], member_axes.clone(), g));
        }
        let Some(feeds) = per_member(&|x| feed_mem(hw, x, role)) else {
            if lv.is_empty() {
                continue;
            }
            chains.push(OperandChain { role, levels: lv });
            continue;
        };
        let distinct = {
            let mut f = feeds.clone();
            f.sort_unstable();
            f.dedup();
            f.len() as u32
        };
        let (ia, copies) = match (distinct, sub_axis) {
            (1, _) => (vec![], 1),
            (d, None) if d == g => (member_axes.clone(), g),
            (d, Some(_)) if d == k1 => (feed_axis.into_iter().collect(), k1),
            _ => return Err(err("E-COST-GANG", format!("gang of {g} units of {path}: {role:?} feeds shared unevenly"))),
        };
        let mut cur = feeds[0];
        let mut fan = 1usize;
        lv.push(b.level(vec![cur], ia, copies));
        while opts.home != Some(cur) && !hw.memories[cur].is_stack() {
            let Some(next) = next_level(hw, cur, opts.home) else { break };
            let (down, up) = group_bw(hw, cur, &next);
            let n = next.len();
            cur = next[0];
            let l = b.level(next, vec![], 1);
            b.cap(l, down.map(|x| x * fan as f64), up.map(|x| x * fan as f64));
            fan = n;
            lv.push(l);
        }
        chains.push(OperandChain { role, levels: lv });
    }
    if chains.is_empty() {
        return Err(err("E-COST-CHAIN", format!("unit {path} has no operand feeds")).at(path));
    }
    for c in &chains {
        let top = *c.levels.last().expect("non-empty chain");
        b.levels[top].external = true;
        if let Some(bw) = opts.bw_assumed {
            b.levels[top].ports = vec![MemPort { dir: PortDir::ReadWrite, bytes_per_cycle: bw / f_unit, serves: vec![], lanes: 1 }];
        }
    }
    let cyc = |c: Option<kiln_ir::hw::quantity::Cycles>| c.map(|c| c.0.round() as u64);
    let pl = &u.spec.pipeline;
    let fill_default = match kind {
        ComputeKind::Matrix(m) => match m.geometry {
            Geometry::Systolic { rows, cols } => (u64::from(rows) + u64::from(cols)).saturating_sub(1),
            _ => 0,
        },
        _ => 0,
    };
    Ok(UnitTemplate {
        name: if g > 1 { format!("{path}[gang {g}]") } else { path },
        clock_hz: f_unit,
        axes,
        modes,
        levels: b.levels,
        chains,
        pipeline: PipelineCycles {
            fill: cyc(pl.fill).unwrap_or(fill_default),
            drain: cyc(pl.drain).unwrap_or(0),
            issue_overhead: pl.issue_overhead.0.round() as u64,
        },
        psum_precision: None,
        fused_down_conversion: false,
        e_mac_idle_ratio: energy::IDLE_MAC_RATIO,
        e_vector_op_j: energy::E_VECTOR_OP,
        energy_source: EnergySource::Uncalibrated,
        operand_run: match kind {
            ComputeKind::Matrix(m) => m.operand_run.iter().map(|(&r, &n)| (r, u64::from(n.max(1)))).collect(),
            _ => vec![],
        },
    })
}

pub(crate) fn hash(t: &UnitTemplate) -> String {
    content_hash("ut1-", &serde_json::to_value(t).expect("template serializes"))
}
