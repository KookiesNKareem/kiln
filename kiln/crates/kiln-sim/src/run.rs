//! One phase end to end: map (heuristic, optional seeded beam), lower, Tier A at the parameter corners
//! (03 §9.1), attribution, shadow prices, invariants and explanation.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::hw::types::ExecModel;
use kiln_map::cost::{KilnCost, UnitCostModel};
use kiln_map::heuristic::{MapOptions, MapReport, heuristic_lowered};
use kiln_map::lower::{TaskGraph, lower};
use kiln_map::mapping::Mapping;
use kiln_map::program::Program;
use kiln_map::search::{BeamSearch, MappingSearch, Score};
use kiln_map::HwView;
use kiln_phys::{ClockMode, ClockPlan};
use kiln_trace::result::{IntervalDriver, IntervalInfo};
use kiln_trace::sim::{ParamContribution, Scope, SimResult};
use kiln_trace::{Corner, Interval, IntervalMethod, Provenance, TraceLevel};
use kiln_wl::stack::Stack;

use crate::engine::{Engine, RunOut};
use crate::explain::explain_run;
use crate::invariants::{Inputs, check_all};
use crate::params::{ParamSet, SimParams};
use crate::result::{Assembly, assemble};

#[derive(Clone)]
pub struct SimOptions {
    /// Whole-step repeat window `w` (03 §4.9).
    pub window: u32,
    pub interval: IntervalMethod,
    pub exec_model: Option<ExecModel>,
    pub seed: u64,
    pub search: Option<BeamSearch>,
    pub trace: TraceLevel,
    /// Explicit engine parameters; overrides `calibration`.
    pub params: Option<ParamSet>,
    /// Calibration set resolved against each design (06 §3.6); `None` with no `params`: the assumed priors
    /// (`assumed-v0`) for the design's execution model and DRAM kind.
    pub calibration: Option<Arc<crate::calib::CalibSet>>,
    pub clock: ClockMode,
    /// Corner evaluations run on up to this many threads; results are identical for any value.
    pub threads: usize,
    pub shadow_prices: bool,
    pub git_hash: String,
    /// `None`: a fresh (cold) [`KilnCost`] per call.
    pub cost_model: Option<Arc<dyn UnitCostModel>>,
    /// Diagnostic only: when the resident set does not fit, report the middle layer (`scope: layer`) instead
    /// of failing the phase as infeasible. Never used for tokens/s or scores.
    pub layer_scope_fallback: bool,
    /// Software-stack recipe (kernel decomposition, 08 §F); `None`: the default for the effective execution
    /// model ([`kiln_wl::stack::default_for`]). Candidate and baseline must be scored under the same one.
    pub stack: Option<Arc<Stack>>,
    /// Cooperative cancellation (06 §6.8): past it, the run stops at its next checkpoint with `E-TIMEOUT`.
    pub deadline: Option<Instant>,
}

pub const TIMEOUT_CODE: &str = "E-TIMEOUT";

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions {
            window: 3,
            interval: IntervalMethod::Corners,
            exec_model: None,
            seed: 0,
            search: None,
            trace: TraceLevel::Summary,
            params: None,
            calibration: None,
            clock: ClockMode::PowerCapped,
            threads: 1,
            shadow_prices: true,
            git_hash: option_env!("KILN_GIT_HASH").unwrap_or("unknown").into(),
            cost_model: None,
            layer_scope_fallback: false,
            stack: None,
            deadline: None,
        }
    }
}

impl std::fmt::Debug for SimOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimOptions").field("window", &self.window).field("interval", &self.interval).field("seed", &self.seed).finish()
    }
}

impl SimOptions {
    /// `E-TIMEOUT` once the deadline has passed.
    pub fn check_deadline(&self) -> Result<(), Diagnostic> {
        match self.deadline {
            Some(d) if Instant::now() >= d => Err(Diagnostic::error(TIMEOUT_CODE, "the evaluation deadline passed")
                .hint("simplify the design (fewer chips, tiles or memory levels) or raise options.timeout_s")),
            _ => Ok(()),
        }
    }

    pub fn cost(&self) -> Arc<dyn UnitCostModel> {
        self.cost_model.clone().unwrap_or_else(|| Arc::new(KilnCost::new()))
    }

    /// The recipe this run uses: the explicit one, else the default for the effective execution model.
    pub fn stack(&self, view: &HwView) -> Arc<Stack> {
        static BUILTIN: OnceLock<Vec<(&'static str, Arc<Stack>)>> = OnceLock::new();
        if let Some(s) = &self.stack {
            return s.clone();
        }
        let id = kiln_wl::stack::default_for(self.exec_model.unwrap_or(view.hw.exec_model));
        let all = BUILTIN.get_or_init(|| {
            kiln_wl::stack::builtin_ids().into_iter().map(|i| (i, Arc::new(Stack::load(i).expect("built-in stack parses")))).collect()
        });
        all.iter().find(|b| b.0 == id).map(|b| b.1.clone()).expect("default stack is built in")
    }

    pub fn param_set(&self, view: &HwView) -> ParamSet {
        if let (None, Some(c)) = (&self.params, &self.calibration) {
            return c.resolve(view, self.exec_model.unwrap_or(view.hw.exec_model));
        }
        self.params.clone().unwrap_or_else(|| {
            let dram = view
                .hw
                .memories
                .iter()
                .find_map(|m| m.dram_kind())
                .map_or("sram".to_string(), |k| serde_json::to_value(k).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default());
            ParamSet::assumed(self.exec_model.unwrap_or(view.hw.exec_model), &dram)
        })
    }
}

#[derive(Clone, Debug)]
pub struct PhaseRun {
    pub mapping: Mapping,
    pub report: MapReport,
    pub graph: TaskGraph,
    pub central: SimResult,
    pub low: Option<SimResult>,
    pub high: Option<SimResult>,
    pub time: Interval,
    pub energy: Interval,
    pub interval: IntervalInfo,
    pub params: ParamSet,
    pub clocks: ClockPlan,
    /// Power of every enforced cap's members (kiln-phys `caps` order) at the central, low and high corners, W.
    pub caps_w: Vec<(f64, f64, f64)>,
    /// Tier A thermal checks at the central, low and high corners (only the central one without corners); empty
    /// without kiln-phys M3.
    pub heat: Vec<Heat>,
}

/// Tier A thermal checks of one run (04 §9), each zone on its own power: the hottest package junction, thermal
/// runaway in any package, and the densest die's average power density at its central and smallest area.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Heat {
    pub t_j_c: f64,
    pub runaway: bool,
    pub q_w_mm2: f64,
    pub q_small_w_mm2: f64,
}

/// One corner's result, cap powers and [`Heat`].
type CornerOut = (SimResult, (Vec<f64>, Option<Heat>));

/// [`Heat`] of an assembled run; `None` without kiln-phys M3 zones.
pub fn heat(a: &Assembly, run: &RunOut, makespan: f64) -> Option<Heat> {
    let m = a.view.phys.m3().filter(|m| !m.zones.is_empty())?;
    let mut h = Heat { t_j_c: f64::NEG_INFINITY, ..Heat::default() };
    for z in &m.zones {
        let pe = crate::result::phase_energy_within(a, run, makespan, &z.nodes);
        let b = z.power.power(&pe, &a.clocks.hz);
        match z.die.and_then(|i| m.report.dies.get(i)) {
            Some(d) => {
                h.q_w_mm2 = h.q_w_mm2.max(b.chip_w / d.area_mm2.max(1e-9));
                h.q_small_w_mm2 = h.q_small_w_mm2.max(b.chip_w / d.area_low_mm2.max(1e-9));
            }
            None => {
                h.t_j_c = h.t_j_c.max(b.t_j_c);
                h.runaway |= b.runaway;
            }
        }
    }
    Some(h)
}

struct Ctx<'a> {
    view: &'a HwView,
    prog: &'a Program,
    graph: &'a TaskGraph,
    clocks: &'a ClockPlan,
    mid: Option<u32>,
}

impl Ctx<'_> {
    fn run(&self, p: &SimParams) -> (RunOut, ClockPlan) {
        let clocks = op_clocks(self.view, self.prog, self.graph, self.mid, self.clocks, p);
        (Engine { view: self.view, g: self.graph, params: p, clocks: &clocks }.run(self.mid), clocks)
    }

    fn makespan(&self, p: &SimParams, scope: Scope, provenance: &Provenance) -> f64 {
        let (run, clocks) = self.run(p);
        let a = Assembly {
            view: self.view,
            prog: self.prog,
            graph: self.graph,
            phase: Id::new("probe").expect("id"),
            scope,
            corner: Corner::Central,
            provenance: provenance.clone(),
            params: p,
            clocks: &clocks,
            trace_ops: false,
        };
        assemble(&a, &run).makespan_s
    }
}

/// Peak MAC rate of every MAC unit at `clocks`.
/// Dense bf16 MAC/s of the MAC units at `clocks`: the denominator of MAC-pipe activity, defined as kiln-calib's
/// telemetry fit defines it (bf16 FLOP/s over bf16 peak at the clock). The fastest mode (int8/int4) would put a
/// dense bf16 GEMM at a fraction of its activity and never select a throttled operating point.
fn peak_macs(view: &HwView, clocks: &ClockPlan) -> f64 {
    view.units
        .iter()
        .filter(|u| view.hw.units[u.unit].spec.kind.is_mac())
        .map(|u| {
            let s = &view.hw.units[u.unit].spec;
            let rate = |bf16: bool| s.precisions.iter().filter(|m| !bf16 || m.key().starts_with("bf16*bf16+")).map(|m| s.kind.ops_per_cycle(m)).fold(0.0, f64::max);
            let r = rate(true);
            (if r > 0.0 { r } else { rate(false) }) * clocks.hz(u.clock).unwrap_or(1.0e9)
        })
        .sum()
}

/// Scope-weighted makespan of one engine run (as [`assemble`] computes it) and the weighted useful MACs.
fn weighted(prog: &Program, g: &TaskGraph, run: &RunOut, scope: Scope, p: &SimParams) -> (f64, f64) {
    let w = crate::result::Weights::new(scope, prog.window);
    let class = |it: Option<u32>| w.of(crate::result::iteration_class(it, run.mid_iteration));
    let t: f64 = run.segs.iter().map(|s| class(s.iteration) * s.time_est).sum();
    let ovh = match g.groups.first().map(|x| x.launch) {
        Some(kiln_map::mapping::LaunchKind::HostLaunch) => p.t_launch,
        Some(kiln_map::mapping::LaunchKind::StaticProgram) => p.t_program,
        _ => 0.0,
    };
    let macs: f64 = g.ops.iter().map(|o| class(g.groups[o.group as usize].iteration) * o.useful_macs as f64).sum();
    (t + w.step_overhead * ovh, macs)
}

/// Clocks of a phase under the calibrated telemetry operating points (`f_cap_op`): MAC-pipe activity at
/// `base` clocks selects the sustained clock of the capped domains, never above `base`.
pub fn op_clocks(view: &HwView, prog: &Program, g: &TaskGraph, mid: Option<u32>, base: &ClockPlan, p: &SimParams) -> ClockPlan {
    let Some(op) = &p.clock_op else { return base.clone() };
    let run = Engine { view, g, params: p, clocks: base }.run(mid);
    let scope = if prog.kind == kiln_map::program::ProgramKind::Isolated { Scope::Op } else { Scope::Step };
    let (t, macs) = weighted(prog, g, &run, scope, p);
    let activity = if t > 0.0 { macs / (peak_macs(view, base) * t) } else { 0.0 };
    let mut out = base.clone();
    for &c in &op.clocks {
        if let Some(h) = out.hz.get_mut(c) {
            let f = op.hz(activity).min(*h);
            out.throttled |= f < *h;
            *h = f;
        }
    }
    out.solved = true;
    out
}

/// Makespan of an already mapped and lowered program at one parameter point (calibration fits and corner
/// re-costing reuse the mapping, 03 §9.1). `base` is the clock plan before telemetry operating points.
pub fn recost(view: &HwView, prog: &Program, g: &TaskGraph, scope: Scope, p: &SimParams, base: &ClockPlan) -> f64 {
    let mid = mid_iteration(prog);
    let clocks = op_clocks(view, prog, g, mid, base, p);
    let run = Engine { view, g, params: p, clocks: &clocks }.run(mid);
    weighted(prog, g, &run, scope, p).0
}

/// Power of every enforced cap's members at `plan`, at the cap's level (die, package or board): one engine run plus
/// 04 §8's power model; infinite without a stable junction temperature.
#[allow(clippy::too_many_arguments)]
pub fn phase_cap_power(view: &HwView, prog: &Program, g: &TaskGraph, mid: Option<u32>, plan: &ClockPlan, p: &SimParams, scope: Scope, prov: &Provenance) -> Vec<f64> {
    let run = Engine { view, g, params: p, clocks: plan }.run(mid);
    let a = Assembly {
        view,
        prog,
        graph: g,
        phase: Id::new("power").expect("id"),
        scope,
        corner: Corner::Central,
        provenance: prov.clone(),
        params: p,
        clocks: plan,
        trace_ops: false,
    };
    let makespan = assemble(&a, &run).makespan_s;
    cap_power(&a, &run, makespan)
}

/// Power of every enforced cap's members for an assembled run (infinite on thermal runaway).
pub fn cap_power(a: &Assembly, run: &RunOut, makespan: f64) -> Vec<f64> {
    let Some(m) = a.view.phys.m3() else { return vec![0.0; a.view.phys.caps().len()] };
    m.caps
        .iter()
        .map(|c| {
            let pe = crate::result::phase_energy_within(a, run, makespan, &c.nodes);
            let b = c.power.power(&pe, &a.clocks.hz);
            if b.runaway { f64::INFINITY } else { b.at(c.level) }
        })
        .collect()
}

fn mid_iteration(prog: &Program) -> Option<u32> {
    prog.window.map(|(w, _)| w / 2)
}

/// Seconds per op from a central run (move ranking for the beam).
fn op_times(prog: &Program, g: &TaskGraph, run: &RunOut) -> Vec<(String, f64)> {
    let mut seg_of = vec![0usize; g.groups.len()];
    for (si, s) in run.segs.iter().enumerate() {
        seg_of[s.groups.0..s.groups.1].iter_mut().for_each(|x| *x = si);
    }
    g.ops.iter().map(|o| (prog.ops[o.op].id.clone(), run.segs[seg_of[o.group as usize]].time_est)).collect()
}

/// The physical corner (kiln-phys parameters) of a simulation corner; none at the center.
fn phys_corner(c: Corner) -> Option<kiln_phys::PCorner> {
    match c {
        Corner::Central => None,
        Corner::Low => Some(kiln_phys::PCorner::Optimistic),
        Corner::High => Some(kiln_phys::PCorner::Pessimistic),
    }
}

/// `f` over the mapper's cost model under `opts`: the base model, padded to the stack's library tiles.
fn with_cost<R>(view: &HwView, prog: &Program, opts: &SimOptions, f: impl FnOnce(&dyn UnitCostModel) -> R) -> Result<R, Diagnostic> {
    let base = opts.cost();
    let rows = crate::stack::row_padding(prog, &opts.stack(view))?;
    let padded = kiln_map::cost::TilePadded { inner: base.as_ref(), rows };
    Ok(f(if padded.rows.iter().all(|&f| f <= 1.0) { base.as_ref() } else { &padded }))
}

/// Heuristic mapping, refined by the seeded beam when `opts.search` is set; with the matching task graph.
pub fn map_program(view: &HwView, prog: &Program, opts: &SimOptions, workload_hash: &str) -> Result<(Mapping, MapReport, TaskGraph), Diagnostic> {
    let base = opts.cost();
    let rows = crate::stack::row_padding(prog, &opts.stack(view))?;
    let padded = kiln_map::cost::TilePadded { inner: base.as_ref(), rows };
    let cost: &dyn UnitCostModel = if padded.rows.iter().all(|&f| f <= 1.0) { base.as_ref() } else { &padded };
    let mo = MapOptions {
        seed: opts.seed,
        exec_model: opts.exec_model,
        workload_hash: workload_hash.into(),
        fuse_elementwise: opts.stack(view).fuse_elementwise,
        ..MapOptions::default()
    };
    let (m, report, g) = heuristic_lowered(prog, view, cost, &mo)?;
    let Some(beam) = &opts.search else { return Ok((m, report, g)) };
    let params = opts.param_set(view).at(Corner::Central);
    let clocks = view.phys.clock_plan(&opts.clock);
    let eval = |m: &Mapping| -> Option<Score> {
        let g = lower(prog, view, m, cost).ok()?;
        let ctx = Ctx { view, prog, graph: &g, clocks: &clocks, mid: mid_iteration(prog) };
        let (run, _) = ctx.run(&params);
        Some(Score { time_s: run.segs.iter().map(|s| s.time_est).sum(), op_times: op_times(prog, &g, &run) })
    };
    let out = beam.search(prog, view, m, &eval, opts.seed);
    let g = lower(prog, view, &out.best, cost)?;
    Ok((out.best, report, g))
}

/// Simulates one phase program under `opts` and returns its corner results.
pub fn simulate(view: &HwView, prog: &Program, phase: Id, scope: Scope, opts: &SimOptions, mut provenance: Provenance) -> Result<PhaseRun, Diagnostic> {
    opts.check_deadline()?;
    let (mapping, report, graph) = map_program(view, prog, opts, &provenance.workload_hash)?;
    opts.check_deadline()?;
    if let Some(e) = mapping.validate(prog, view).into_iter().next() {
        return Err(e);
    }
    simulate_mapped(view, prog, phase, scope, opts, &mut provenance, mapping, report, graph)
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_mapped(
    view: &HwView,
    prog: &Program,
    phase: Id,
    scope: Scope,
    opts: &SimOptions,
    provenance: &mut Provenance,
    mapping: Mapping,
    report: MapReport,
    mut graph: TaskGraph,
) -> Result<PhaseRun, Diagnostic> {
    let stack = opts.stack(view);
    crate::stack::attach(view, prog, &mut graph, &stack)?;
    provenance.flags.insert("stack".into(), stack.label());
    let params = opts.param_set(view);
    let telemetry = params.params.iter().any(|p| p.name == "f_cap_op");
    // 03 §4.5: largest V/f point whose phase power (04 §8) fits the cap at corner `c`'s parameters; the platform
    // telemetry table (f_cap_op) overrides it when the calibration set carries one.
    let prov0 = provenance.clone();
    let solve = |v: &HwView, g: &TaskGraph, c: Corner| match opts.clock {
        ClockMode::PowerCapped if !telemetry && v.phys.m3().is_some() => {
            let p = params.at(c);
            let mid = mid_iteration(prog);
            let prov = prov0.clone();
            let f = |hz: &[f64]| phase_cap_power(v, prog, g, mid, &ClockPlan { hz: hz.to_vec(), solved: true, throttled: false }, &p, scope, &prov);
            v.phys.solve_clock(Some(&f)).0
        }
        ClockMode::PowerCapped => v.phys.solve_clock(None).0,
        ref m => v.phys.clock_plan(m),
    };
    let clocks = solve(view, &graph, Corner::Central);
    provenance.mapping_hash = Some(mapping.hash());
    provenance.calibration_hash = params.hash();
    provenance.calibration_id = Some(params.id.clone());
    provenance.window = prog.window.map(|w| w.0);
    if !params.extrapolated.is_empty() {
        provenance.flags.insert("extrapolated".into(), params.extrapolated.join(","));
    }
    provenance.flags.insert("cost_model".into(), opts.cost().name().into());
    provenance.flags.insert("phys".into(), view.phys.model.into());
    provenance.flags.insert(
        "clock".into(),
        match (telemetry, clocks.solved, &opts.clock) {
            (true, _, _) => "telemetry_operating_points".into(),
            (false, true, ClockMode::PowerCapped) => "power_cap_solved".into(),
            (false, true, _) => "fixed".into(),
            (false, false, _) => "power_cap_unsolved_nominal".into(),
        },
    );
    provenance.flags.insert("mapper".into(), if opts.search.is_some() { "heuristic+beam".into() } else { "heuristic".into() });
    let ctx = Ctx { view, prog, graph: &graph, clocks: &clocks, mid: mid_iteration(prog) };
    let corners: Vec<Corner> = match opts.interval {
        IntervalMethod::Corners | IntervalMethod::Sampled => vec![Corner::Central, Corner::Low, Corner::High],
        _ => vec![Corner::Central],
    };
    // 03 §9.1: a corner re-costs the fixed mapping on the physical model at that corner (energies, latencies) and
    // solves its own power cap.
    let corner_views: Vec<Option<(HwView, TaskGraph, ClockPlan)>> = corners
        .iter()
        .map(|&c| -> Result<_, Diagnostic> {
            let (Some(m), Some(pc)) = (view.phys.m3(), phys_corner(c)) else { return Ok(None) };
            let cv = HwView::with_phys(view.hw.clone(), kiln_phys::Phys::with(&view.hw, m.params.at(pc), m.fp.tier))?;
            let mut g = with_cost(&cv, prog, opts, |cost| lower(prog, &cv, &mapping, cost))??;
            crate::stack::attach(&cv, prog, &mut g, &stack)?;
            let k = solve(&cv, &g, c);
            Ok(Some((cv, g, k)))
        })
        .collect::<Result<_, _>>()?;
    let one = |(c, cv): (Corner, &Option<(HwView, TaskGraph, ClockPlan)>)| -> CornerOut {
        let (view, graph, clocks) = cv.as_ref().map_or((view, &graph, &clocks), |x| (&x.0, &x.1, &x.2));
        let ctx = Ctx { view, prog, graph, clocks, mid: mid_iteration(prog) };
        let p = params.at(c);
        let (run, clocks) = ctx.run(&p);
        let clocks = &clocks;
        let a = Assembly {
            view,
            prog,
            graph,
            phase: phase.clone(),
            scope,
            corner: c,
            provenance: provenance.clone(),
            params: &p,
            clocks,
            trace_ops: opts.trace >= TraceLevel::Ops || scope == Scope::Op,
        };
        let mut r = assemble(&a, &run);
        let caps = cap_power(&a, &run, r.makespan_s);
        let h = heat(&a, &run, r.makespan_s);
        if c == Corner::Central && opts.shadow_prices {
            let base = r.makespan_s;
            let index: std::collections::BTreeMap<&str, u32> = view.resource_ids().iter().enumerate().map(|(i, y)| (y.as_str(), i as u32)).collect();
            let busy: Vec<(&str, f64)> = r.resources.iter().map(|x| (x.resource.as_str(), x.busy_s)).collect();
            for tr in r.bottleneck.top_resources.iter_mut() {
                let Some(b) = busy.iter().find(|x| x.0 == tr.resource.as_str()).map(|x| x.1) else { continue };
                let twins: Vec<u32> = busy.iter().filter(|x| (x.1 - b).abs() <= 1e-9 * b).filter_map(|x| index.get(x.0).copied()).collect();
                let mut pp = p.clone();
                pp.cap_scale.extend(twins.into_iter().map(|i| (i, 1.1)));
                let t = ctx.makespan(&pp, scope, provenance);
                tr.shadow_price = ((base - t) / (base * (1.0 - 1.0 / 1.1))).clamp(0.0, 1.0);
            }
        }
        let resident_overflow = report.capacity_overflow.is_some();
        r.invariants = check_all(&r, &Inputs { view, prog, graph, run: &run, clocks, resident_overflow });
        (r, (caps, h))
    };
    opts.check_deadline()?;
    let (mut results, caps): (Vec<SimResult>, Vec<_>) = if opts.threads > 1 && corners.len() > 1 {
        std::thread::scope(|s| {
            let hs: Vec<_> = corners.iter().zip(&corner_views).map(|(&c, cv)| s.spawn(move || one((c, cv)))).collect();
            hs.into_iter().map(|h| h.join().expect("corner thread")).collect()
        })
    } else {
        corners.iter().zip(&corner_views).map(|(&c, cv)| one((c, cv))).collect()
    };
    opts.check_deadline()?;
    let (caps, heat): (Vec<Vec<f64>>, Vec<Option<Heat>>) = caps.into_iter().unzip();
    let heat = heat.into_iter().flatten().collect();
    let corner = |i: usize, k: usize| caps.get(i).unwrap_or(&caps[0])[k];
    let caps_w = (0..caps[0].len()).map(|k| (corner(0, k), corner(1, k), corner(2, k))).collect();
    let uncal = ctx.makespan(&SimParams::null(), scope, provenance);
    let mut contributions = vec![];
    let mut drivers = vec![];
    if params.params.iter().len() > 0 && matches!(opts.interval, IntervalMethod::Sensitivity | IntervalMethod::Corners) {
        let tc = results[0].makespan_s;
        let deltas: Vec<f64> = (0..params.params.len())
            .map(|i| opts.check_deadline().map(|()| ctx.makespan(&params.perturbed(i, Corner::Low), scope, provenance) - tc))
            .collect::<Result<_, _>>()?;
        let width: f64 = deltas.iter().map(|d| d.max(0.0)).sum();
        for (i, p) in params.params.iter().enumerate() {
            contributions.push(ParamContribution { name: p.name.clone(), key: p.key.clone(), delta_makespan_s: deltas[i], delta_energy_j: 0.0 });
            if width > 0.0 && deltas[i] > 0.0 {
                drivers.push(IntervalDriver { param: p.name.clone(), key: p.key.clone(), share: deltas[i] / width });
            }
        }
        drivers.sort_by(|a, b| b.share.total_cmp(&a.share).then(a.param.cmp(&b.param)));
    }
    for r in &mut results {
        r.calibration.set_hash = params.hash();
        r.calibration.uncalibrated_makespan_s = Some(uncal);
        r.calibration.params.clone_from(&contributions);
        r.bottleneck.summary = explain_run(r, None, 5);
    }
    let central = results.remove(0);
    let (low, high) = if results.len() == 2 { (Some(results.remove(0)), Some(results.remove(0))) } else { (None, None) };
    let (time, energy, method) = match (&low, &high) {
        (Some(l), Some(h)) => (
            Interval::from_corners(central.makespan_s, l.makespan_s, h.makespan_s),
            Interval::from_corners(central.energy.total_j, l.energy.total_j, h.energy.total_j),
            IntervalMethod::Corners,
        ),
        _ if opts.interval == IntervalMethod::Sensitivity => {
            let tc = central.makespan_s;
            let up: f64 = contributions.iter().map(|c| c.delta_makespan_s.max(0.0)).sum();
            (Interval::from_corners(tc, tc + up, tc), Interval::point(central.energy.total_j), IntervalMethod::Sensitivity)
        }
        _ => (Interval::point(central.makespan_s), Interval::point(central.energy.total_j), IntervalMethod::None),
    };
    let interval = IntervalInfo { method, corner_flips: vec![], drivers, low_remapped: None };
    Ok(PhaseRun { mapping, report, graph, central, low, high, time, energy, interval, params, clocks, caps_w, heat })
}
