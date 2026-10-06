//! Assembles a Tier A [`SimResult`] (03 §10) from one engine run: whole-step extrapolation (03 §4.9),
//! attribution that sums exactly to the makespan, per-op and per-resource results, energy, floors.

use std::collections::BTreeMap;

use kiln_ir::common::Id;
use kiln_ir::hw::model::NodeIx;
use kiln_ir::hw::types::BlockKind;
use kiln_map::hwview::{HwView, ResClass};
use kiln_map::lower::TaskGraph;
use kiln_map::mapping::LaunchKind;
use kiln_map::program::Program;
use kiln_phys::ClockPlan;
use kiln_trace::sim::{
    Binding, BindingClass, Bottleneck, CalibrationContribution, ClockSample, CostModelSummary, EnergyBreakdown, Floor,
    FloorKind, GroupKind, GroupResult, InvariantReport, LevelBytes, OpResult, OverheadKind, PowerSummary, ResourceResult,
    Scope, SimResult, Target,
};
use kiln_trace::{Corner, Provenance, Tier};

use crate::engine::{MID, OTHER, PE, RunOut, sanitize};
use crate::params::SimParams;

/// Weights that turn window quantities into the reported scope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    pub pe: f64,
    pub mid: f64,
    pub other: f64,
    pub step_overhead: f64,
}

impl Weights {
    /// `scope: step`: window plus `(L - w)` copies of the middle iteration; `scope: layer`: middle iteration
    /// plus a `1/L` share of prologue, epilogue and per-step overheads; `scope: op` and windowless graphs: 1.
    pub fn new(scope: Scope, window: Option<(u32, u64)>) -> Weights {
        match (scope, window) {
            (Scope::Layer, Some((_, l))) => Weights { pe: 1.0 / l as f64, mid: 1.0, other: 0.0, step_overhead: 1.0 / l as f64 },
            (Scope::Op, _) => Weights { pe: 1.0, mid: 1.0, other: 1.0, step_overhead: 0.0 },
            (_, Some((w, l))) => Weights { pe: 1.0, mid: 1.0 + (l - u64::from(w)) as f64, other: 1.0, step_overhead: 1.0 },
            (_, None) => Weights { pe: 1.0, mid: 1.0, other: 1.0, step_overhead: 1.0 },
        }
    }

    pub fn of(&self, class: usize) -> f64 {
        [self.pe, self.mid, self.other][class]
    }
}

pub fn iteration_class(it: Option<u32>, mid: Option<u32>) -> usize {
    match it {
        None => PE,
        i if i == mid => MID,
        _ => OTHER,
    }
}

pub struct Assembly<'a> {
    pub view: &'a HwView,
    pub prog: &'a Program,
    pub graph: &'a TaskGraph,
    pub phase: Id,
    pub scope: Scope,
    pub corner: Corner,
    pub provenance: Provenance,
    pub params: &'a SimParams,
    pub clocks: &'a ClockPlan,
    pub trace_ops: bool,
}

pub fn step_overhead(a: &Assembly) -> (f64, OverheadKind) {
    match a.graph.groups.first().map(|g| g.launch) {
        Some(LaunchKind::HostLaunch) => (a.params.t_launch, OverheadKind::Launch),
        Some(LaunchKind::StaticProgram) => (a.params.t_program, OverheadKind::Program),
        _ => (0.0, OverheadKind::Dispatch),
    }
}

fn id(s: &str) -> Id {
    Id::new(sanitize(s)).unwrap_or_else(|_| Id::new("unnamed").expect("valid"))
}

pub fn assemble(a: &Assembly, run: &RunOut) -> SimResult {
    let w = Weights::new(a.scope, a.prog.window);
    let wseg = |s: &crate::engine::SegOut| w.of(iteration_class(s.iteration, run.mid_iteration));
    let (ovh, _) = step_overhead(a);
    let mut makespan = 0.0;
    let (mut t_a2, mut t_a0) = (0.0, 0.0);
    let mut by: BTreeMap<BindingClass, f64> = BTreeMap::new();
    for s in &run.segs {
        let k = wseg(s);
        makespan += k * s.time_est;
        t_a2 += k * s.time_floor;
        t_a0 += k * s.a0;
        *by.entry(s.binding.class()).or_default() += k * s.core_est();
        if s.contention > 0.0 {
            *by.entry(BindingClass::Contention).or_default() += k * s.contention;
        }
        if s.overhead + s.stack_overhead > 0.0 {
            *by.entry(BindingClass::Overhead).or_default() += k * (s.overhead + s.stack_overhead);
        }
        if s.stack_dram > 0.0 {
            *by.entry(BindingClass::Dram).or_default() += k * s.stack_dram;
        }
        if s.stack_onchip > 0.0 {
            *by.entry(BindingClass::Port).or_default() += k * s.stack_onchip;
        }
    }
    makespan += w.step_overhead * ovh;
    t_a2 += w.step_overhead * ovh;
    if ovh > 0.0 && w.step_overhead > 0.0 {
        *by.entry(BindingClass::Overhead).or_default() += w.step_overhead * ovh;
    }
    by.retain(|_, v| *v > 0.0);
    let v = a.view;
    let comb = |x: &[Vec<f64>; 3], r: usize| w.pe * x[PE][r] + w.mid * x[MID][r] + w.other * x[OTHER][r];
    let mut energy = EnergyBreakdown::default();
    let mut resources = vec![];
    for (r, node) in resource_nodes(v).into_iter().enumerate() {
        let (busy, bytes) = (comb(&run.busy, r), comb(&run.bytes, r));
        if busy <= 0.0 {
            continue;
        }
        let res = &v.resources[r];
        let e = bytes * res.energy_j_per_b * dyn_scale(v, res, node, a.clocks);
        resources.push(ResourceResult {
            resource: v.resource_ids()[r].clone(),
            kind: res.kind,
            busy_s: busy,
            stall_s: 0.0,
            bytes,
            macs: 0,
            energy_j: e,
            utilization: if makespan > 0.0 { busy / makespan } else { 0.0 },
            peak_queue: None,
            class: res.link_class.clone(),
        });
    }
    (energy.memory_j, energy.link_j, _) = resource_energy(a, run, None);
    let seg_of_group: Vec<usize> = {
        let mut m = vec![0usize; a.graph.groups.len()];
        for (si, s) in run.segs.iter().enumerate() {
            m[s.groups.0..s.groups.1].iter_mut().for_each(|x| *x = si);
        }
        m
    };
    let dram_bw: f64 = v.resources.iter().filter(|r| r.class == ResClass::Dram).map(|r| r.capacity).sum();
    let mut span: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for (t, task) in a.graph.tasks.iter().enumerate() {
        let e = span.entry(task.op as usize).or_insert((f64::INFINITY, 0.0));
        e.0 = e.0.min(run.task_start[t]);
        e.1 = e.1.max(run.task_end[t]);
    }
    let mut ops = vec![];
    for st in &a.graph.ops {
        let s = &run.segs[seg_of_group[st.group as usize]];
        let k = wseg(s);
        let (ec, ep) = op_energy(v, st, a.clocks, None);
        energy.compute_j += k * ec;
        energy.padding_j += k * ep;
        if !a.trace_ops && a.scope != Scope::Op {
            continue;
        }
        let (start, end) = span.get(&st.op).copied().unwrap_or((0.0, 0.0));
        let end = end.max(if start.is_finite() { start } else { 0.0 });
        let op = &a.prog.ops[st.op];
        let mut floors = vec![];
        if st.peak_macs_per_s > 0.0 {
            floors.push(Floor { kind: FloorKind::Compute, path: None, seconds: st.useful_macs as f64 / st.peak_macs_per_s });
        }
        if st.compulsory_offchip > 0 && dram_bw > 0.0 {
            floors.push(Floor { kind: FloorKind::MemoryLevel, path: Some("offchip".into()), seconds: st.compulsory_offchip as f64 / dram_bw });
        }
        ops.push(OpResult {
            op: id(&op.id),
            layer: a.prog.iteration_of_op(st.op),
            start_s: if start.is_finite() { start.min(makespan) } else { 0.0 },
            end_s: end.min(makespan),
            binding: s.binding.clone(),
            runner_up: s.runner_up.clone(),
            macs_useful: st.useful_macs as u64,
            macs_issued: st.issued_macs as u64,
            bytes_by_level: st.level_bytes.iter().map(|(g, b)| LevelBytes { level: v.groups[*g].name.clone(), bytes: *b as u64 }).collect(),
            energy: EnergyBreakdown { compute_j: ec, padding_j: ep, ..Default::default() }.with_total(),
            target: Target::Host,
            host_vs_nmp_s: None,
            group: Some(st.group),
            floors,
        });
    }
    energy.static_j = match v.phys.m3() {
        // Board-level time-proportional power (leakage at V(f) and T_j, clock tree, DRAM background, board, VR loss).
        Some(_) => {
            let pe = phase_energy(v, &energy, makespan, a.clocks);
            let bd = v.phys.phase_power(&pe, a.clocks).unwrap_or_default();
            (bd.board_w * makespan - pe.core_dyn_j - pe.indep_j - pe.dram_j).max(0.0)
        }
        None => v.phys.static_power_w() * makespan,
    };
    let energy = energy.with_total();
    let groups = if a.trace_ops || a.scope == Scope::Op {
        a.graph
            .groups
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let s = &run.segs[seg_of_group[gi]];
                let n = (s.groups.1 - s.groups.0) as f64;
                GroupResult {
                    group: gi as u32,
                    kind: if g.fused { GroupKind::Fused } else { GroupKind::Single },
                    ops: g.ops.iter().map(|&o| id(&a.prog.ops[a.graph.ops[o as usize].op].id)).collect(),
                    start_s: s.start_s.min(makespan),
                    end_s: (s.start_s + s.time_est).min(makespan),
                    binding: s.binding.clone(),
                    bubble_s: 0.0,
                    exposed_overhead_s: (s.overhead + s.stack_overhead) / n,
                }
            })
            .collect()
    } else {
        vec![]
    };
    let clocks = v
        .hw
        .clocks
        .iter()
        .enumerate()
        .map(|(c, ci)| ClockSample { domain: id(&ci.path), t_s: 0.0, hz: a.clocks.hz[c] })
        .collect();
    let mut top: Vec<&ResourceResult> = resources.iter().collect();
    top.sort_by(|x, y| y.utilization.total_cmp(&x.utilization).then(x.resource.cmp(&y.resource)));
    let bottleneck = Bottleneck {
        time_by_binding: by,
        top_resources: top
            .iter()
            .take(5)
            .map(|r| kiln_trace::sim::TopResource { resource: r.resource.clone(), utilization: r.utilization, shadow_price: 0.0 })
            .collect(),
        slack: top.iter().take(5).map(|r| kiln_trace::sim::ResourceSlack { resource: r.resource.clone(), slack: 1.0 - r.utilization }).collect(),
        summary: String::new(),
    };
    let power = PowerSummary {
        avg_w: if makespan > 0.0 { energy.total_j / makespan } else { 0.0 },
        peak_windowed_w: if makespan > 0.0 { energy.total_j / makespan } else { 0.0 },
        cap_w: v.phys.power_cap_w(),
        throttled: a.clocks.throttled,
    };
    SimResult {
        schema: kiln_trace::SIM_SCHEMA.into(),
        provenance: a.provenance.clone(),
        phase: a.phase.clone(),
        tier: Tier::A,
        scope: a.scope,
        corner: a.corner,
        makespan_s: makespan,
        t_a0_s: t_a0,
        t_a2_s: t_a2,
        clocks,
        energy,
        power,
        ops,
        resources,
        groups,
        collectives: vec![],
        bottleneck,
        invariants: InvariantReport::default(),
        cost_model: CostModelSummary { cache_hits: a.graph.cost_hits, cache_misses: a.graph.cost_misses, truncated_searches: 0 },
        calibration: CalibrationContribution::default(),
        trace: None,
    }
}

/// Compute and padding energy of an op at `plan`: each slice at its unit's `(V/V_nom)^2`, only units inside
/// `nodes` when given.
pub fn op_energy(v: &HwView, st: &kiln_map::lower::OpStats, plan: &ClockPlan, nodes: Option<&[bool]>) -> (f64, f64) {
    st.e_by_unit.iter().filter(|x| nodes.is_none_or(|n| n[v.hw.units[v.units[x.0].unit].node])).fold((0.0, 0.0), |acc, &(u, e, pad)| {
        let k = v.phys.dyn_scale(v.units[u].clock, Some(v.hw.units[v.units[u].unit].node), plan);
        (acc.0 + e * k, acc.1 + pad * k)
    })
}

/// Arena node each resource's energy is charged to (a link's at its first channel's source; none for the sequencer).
pub fn resource_nodes(v: &HwView) -> Vec<Option<usize>> {
    let hw = &v.hw;
    let mut out = vec![None; v.resources.len()];
    for c in &hw.channels {
        out[v.res_of_shared[c.resource] as usize].get_or_insert(hw.node_of(c.src));
    }
    for (m, &id) in v.res_of_mem.iter().enumerate() {
        out[id as usize] = Some(hw.memories[m].node);
    }
    for u in &v.units {
        out[u.compute as usize] = Some(hw.units[u.unit].node);
    }
    out
}

type ByKey = BTreeMap<String, f64>;

/// Die-side PHY node of every DRAM resource: the PHY block its stack's channels attach to.
fn dram_phy_nodes(v: &HwView) -> Vec<Option<usize>> {
    let hw = &v.hw;
    let mut phy = vec![false; hw.nodes.len()];
    for b in hw.blocks.iter().filter(|b| matches!(b.spec.kind, BlockKind::Phy(_))) {
        phy[b.node] = true;
    }
    let mut out = vec![None; v.resources.len()];
    for ch in &hw.channels {
        let (s, d) = (hw.node_of(ch.src), hw.node_of(ch.dst));
        for (p, m) in [(s, ch.dst), (d, ch.src)] {
            if let (true, NodeIx::Mem(m)) = (phy[p], m) {
                out[v.res_of_mem[m] as usize].get_or_insert(p);
            }
        }
    }
    out
}

/// Memory energy by level and link energy by class of the busy resources, each key summed in ascending order of
/// its terms so declaration order never changes a bit (04 §17 rule 5). Inside `nodes` when given: DRAM energy
/// is then returned apart as `(core, phy)`, the core share charged to the stack and the PHY share to the die-side
/// PHY its channels attach to, each counted only where it lies inside.
fn resource_energy(a: &Assembly, run: &RunOut, nodes: Option<&[bool]>) -> (ByKey, ByKey, (f64, f64)) {
    let v = a.view;
    let w = Weights::new(a.scope, a.prog.window);
    let rnodes = resource_nodes(v);
    let at = nodes.map(|_| dram_phy_nodes(v));
    let phy = v.phys.dram_phy_fraction();
    let comb = |x: &[Vec<f64>; 3], r: usize| w.pe * x[PE][r] + w.mid * x[MID][r] + w.other * x[OTHER][r];
    let (mut mem, mut link): (BTreeMap<String, Vec<f64>>, BTreeMap<String, Vec<f64>>) = Default::default();
    let (mut core_j, mut phy_j) = (vec![], vec![]);
    for (r, res) in v.resources.iter().enumerate() {
        if comb(&run.busy, r) <= 0.0 {
            continue;
        }
        let e = || comb(&run.bytes, r) * res.energy_j_per_b * dyn_scale(v, res, rnodes[r], a.clocks);
        let (n, phy_at) = match (nodes, &at) {
            (Some(n), Some(phy_at)) => (n, phy_at),
            _ => {
                match res.class {
                    ResClass::Mem | ResClass::Dram => mem.entry(format!("l{}", res.level.unwrap_or(0))).or_default().push(e()),
                    ResClass::Link => link.entry(res.link_class.clone().unwrap_or_else(|| "link".into())).or_default().push(e()),
                    _ => {}
                }
                continue;
            }
        };
        let inside = |x: Option<usize>| x.is_some_and(|x| n[x]);
        let at = rnodes[r];
        match res.class {
            ResClass::Dram => {
                if inside(at) {
                    core_j.push(e() * (1.0 - phy));
                }
                if inside(phy_at[r].or(at)) {
                    phy_j.push(e() * phy);
                }
            }
            ResClass::Mem if inside(at) => mem.entry(format!("l{}", res.level.unwrap_or(0))).or_default().push(e()),
            ResClass::Link if inside(at) => link.entry(res.link_class.clone().unwrap_or_else(|| "link".into())).or_default().push(e()),
            _ => {}
        }
    }
    let sum = |mut xs: Vec<f64>| -> f64 {
        xs.sort_by(f64::total_cmp);
        xs.iter().sum()
    };
    let total = |m: BTreeMap<String, Vec<f64>>| -> ByKey { m.into_iter().map(|(k, xs)| (k, sum(xs))).collect() };
    (total(mem), total(link), (sum(core_j), sum(phy_j)))
}

/// Energies of one run (as [`assemble`] accounts them) spent inside a cap's members, split the way the power
/// model accounts them (see [`phase_energy_in`]); DRAM traffic energy splits between the stack (core) and its
/// die-side PHY before the scope applies.
pub fn phase_energy_within(a: &Assembly, run: &RunOut, makespan: f64, nodes: &[bool]) -> kiln_phys::PhaseEnergy {
    let v = a.view;
    let w = Weights::new(a.scope, a.prog.window);
    let mut energy = EnergyBreakdown::default();
    let (dram_core, dram_phy);
    (energy.memory_j, energy.link_j, (dram_core, dram_phy)) = resource_energy(a, run, Some(nodes));
    let mut seg_of_group = vec![0usize; a.graph.groups.len()];
    for (si, s) in run.segs.iter().enumerate() {
        seg_of_group[s.groups.0..s.groups.1].iter_mut().for_each(|x| *x = si);
    }
    for st in &a.graph.ops {
        let k = w.of(iteration_class(run.segs[seg_of_group[st.group as usize]].iteration, run.mid_iteration));
        let (ec, ep) = op_energy(v, st, a.clocks, Some(nodes));
        energy.compute_j += k * ec;
        energy.padding_j += k * ep;
    }
    let mut pe = phase_energy_in(v, &energy.with_total(), makespan, a.clocks, Some(nodes));
    pe.dram_j += dram_core;
    pe.indep_j += dram_phy;
    pe
}

/// Link classes whose energy does not scale with the core voltage (PHYs, package traces, host links, bonds).
pub const OFF_DIE: [&str; 5] = ["d2d", "serdes", "optical", "host", "vertical"];

/// `(V/V_nom)^2` for a resource's energy: on-chip memories and on-die links of the core domain; DRAM and off-die
/// links are clock independent (03 §4.5 split).
pub fn dyn_scale(v: &HwView, res: &kiln_map::hwview::Resource, node: Option<usize>, plan: &ClockPlan) -> f64 {
    match res.class {
        ResClass::Mem => v.phys.dyn_scale(res.clock, node, plan),
        ResClass::Link if !res.link_class.as_deref().is_some_and(|c| OFF_DIE.contains(&c)) => v.phys.dyn_scale(res.clock, node, plan),
        _ => 1.0,
    }
}

/// A result's energies split the way the power model accounts them (04 §8): core-domain dynamic, die-level
/// clock-independent (PHYs, off-die links) and DRAM core; MAC activity from compute power vs its peak at `plan`.
pub fn phase_energy(v: &HwView, e: &EnergyBreakdown, makespan: f64, plan: &ClockPlan) -> kiln_phys::PhaseEnergy {
    phase_energy_in(v, e, makespan, plan, None)
}

/// [`phase_energy`] of a cap's members (`nodes`): activity against their own MAC units' peak.
pub fn phase_energy_in(v: &HwView, e: &EnergyBreakdown, makespan: f64, plan: &ClockPlan, nodes: Option<&[bool]>) -> kiln_phys::PhaseEnergy {
    let dram_levels: Vec<String> = v.resources.iter().filter(|r| r.class == ResClass::Dram).map(|r| format!("l{}", r.level.unwrap_or(0))).collect();
    let phy = v.phys.dram_phy_fraction();
    let mut core = e.compute_j + e.padding_j + e.conversion_j + e.nmp_j;
    let (mut indep, mut dram) = (0.0, 0.0);
    for (k, x) in &e.memory_j {
        if dram_levels.contains(k) {
            dram += x * (1.0 - phy);
            indep += x * phy;
        } else {
            core += x;
        }
    }
    for (k, x) in &e.link_j {
        if OFF_DIE.contains(&k.as_str()) { indep += x } else { core += x }
    }
    let peak = v.phys.peak_compute_w(plan, nodes);
    let activity = if peak > 0.0 && makespan > 0.0 { (e.compute_j / makespan / peak).clamp(0.0, 1.0) } else { 0.0 };
    kiln_phys::PhaseEnergy { makespan_s: makespan, core_dyn_j: core, indep_j: indep, dram_j: dram, activity, busy: if makespan > 0.0 { 1.0 } else { 0.0 } }
}

/// The binding class and resource of the largest attributed share.
pub fn dominant(r: &SimResult) -> Option<(BindingClass, f64)> {
    r.bottleneck.dominant()
}

pub fn binding_label(b: &Binding) -> String {
    match b.resource() {
        Some(r) => format!("{:?} {}", b.class(), r.as_str()),
        None => format!("{:?}", b.class()),
    }
}
