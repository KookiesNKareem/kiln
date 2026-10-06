//! Tier A predictions for measurement records: each micro record becomes an isolated program (a bench
//! contraction, or a one-node streaming workload), mapped and lowered once, then re-costed at every parameter
//! point the fit tries (03 §9.1: corners and fits reuse the mapping). Whole steps run the full engine path.

use std::path::Path;
use std::sync::Arc;

use indexmap::IndexMap;
use kiln_ir::common::{Diagnostic, Id};
use kiln_ir::hw::Profile;
use kiln_ir::wl::{
    ActAttrs, ElemType, EntryPoints, FnSpec, Graph, MapAttrs, MapFn, Model, Node, Op, ReduceAttrs, ReduceKind,
    SeqBatch, SymbolDecl, TensorClass, TensorDecl, DimExpr, PhaseKind,
};
use kiln_map::lower::TaskGraph;
use kiln_map::program::Program;
use kiln_map::{HwView, UnitCostModel};
use kiln_phys::{ClockMode, ClockPlan};
use kiln_sim::{CalibSet, Prepared, SimOptions, SimParams};
use kiln_trace::sim::Scope;
use kiln_trace::{Interval, IntervalMethod};

use crate::records::{Kind, Record, StreamOp};

pub const PRED_CODE: &str = "E-CAL-PRED-001";

pub fn designs_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference")
}

/// A mapped isolated program of one record.
pub struct Case {
    pub prog: Program,
    pub graph: TaskGraph,
    /// Launch probes: kernels per chain (`None` for ordinary records).
    pub chain: Option<u64>,
    /// Kernels with no work at all (`empty` probe): only launch-path terms.
    pub empty: bool,
}

pub struct Bench {
    pub prepared: Prepared,
    pub cost: Arc<dyn UnitCostModel>,
    /// Clock domains of MAC units (set from telemetry when a record carries a measured clock).
    pub core_clocks: Vec<usize>,
}

fn id(s: &str) -> Id {
    Id::new(s).expect("valid id")
}

/// `out = f(in...)` over `elems` bf16 elements as a one-node workload.
pub fn stream_model(op: StreamOp, elems: u64) -> Option<Model> {
    let t = |class| TensorDecl::new([DimExpr::int(elems)], ElemType::BF16, class);
    let mut gt = IndexMap::new();
    let (node, ins): (Node, Vec<&str>) = match op {
        StreamOp::Add => (Node::new("op", Op::Map(MapAttrs { func: FnSpec::One(MapFn::Add) }), &["x", "y"], &["z"]), vec!["x", "y"]),
        StreamOp::Scale => (Node::new("op", Op::Map(MapAttrs { func: FnSpec::One(MapFn::Scale) }), &["x"], &["z"]), vec!["x"]),
        StreamOp::Copy => (Node::new("op", Op::Map(MapAttrs { func: FnSpec::One(MapFn::Cast) }), &["x"], &["z"]), vec!["x"]),
        StreamOp::Silu => (Node::new("op", Op::Act(ActAttrs { func: MapFn::Silu }), &["x"], &["z"]), vec!["x"]),
        StreamOp::Read => (Node::new("op", Op::Reduce(ReduceAttrs { axes: vec![0], combiner: ReduceKind::Sum }), &["x"], &["z"]), vec!["x"]),
        StreamOp::Write => return None,
    };
    for i in &ins {
        gt.insert(id(i), t(TensorClass::Input));
    }
    let out = if op == StreamOp::Read { TensorDecl::new([DimExpr::int(1)], ElemType::BF16, TensorClass::Output) } else { t(TensorClass::Output) };
    gt.insert(id("z"), out);
    let mut graphs = IndexMap::new();
    graphs.insert(
        id("main"),
        Graph { params: ins.iter().map(|i| id(i)).collect(), results: vec![id("z")], tensors: gt, nodes: vec![node], regions: IndexMap::new() },
    );
    let mut symbols = IndexMap::new();
    symbols.insert("seqs".to_string(), SymbolDecl::segments());
    Some(Model { symbols, graphs, entry: EntryPoints { forward: id("main") }, tensors: IndexMap::new(), routing: None })
}

fn isolated(model: &Model) -> Result<Program, Diagnostic> {
    let sc = kiln_wl::zoo::whole_step(PhaseKind::Custom, SeqBatch::uniform(1, 1, 1));
    let (_, lg, _) = kiln_wl::evaluate_snapshot(model, &sc).map_err(|mut e| e.remove(0))?;
    Program::isolated(&lg)
}

impl Bench {
    pub fn new(design: &str) -> Result<Bench, Diagnostic> {
        let prepared = Prepared::from_file(&designs_dir().join(design), Profile::Reference).map_err(|mut e| e.remove(0))?;
        let v = &prepared.view;
        let mut core_clocks: Vec<usize> =
            v.units.iter().filter(|u| v.hw.units[u.unit].spec.kind.is_mac()).filter_map(|u| u.clock).collect();
        core_clocks.sort();
        core_clocks.dedup();
        Ok(Bench { prepared, cost: Arc::new(kiln_map::cost::KilnCost::new()), core_clocks })
    }

    pub fn view(&self) -> &HwView {
        &self.prepared.view
    }

    /// Nominal clocks, with the MAC-unit domains at `core_hz` when a measured clock is given.
    pub fn clocks(&self, core_hz: Option<f64>) -> ClockPlan {
        let ph = &self.view().phys;
        let mut hz: Vec<f64> = (0..self.view().hw.clocks.len()).map(|c| ph.nominal_hz(c)).collect();
        if let Some(f) = core_hz {
            for &c in &self.core_clocks {
                hz[c] = f;
            }
        }
        ph.clock_plan(&ClockMode::Fixed(hz))
    }

    fn program(&self, r: &Record) -> Result<(Program, Option<u64>, bool), Diagnostic> {
        let unsupported = || Diagnostic::error(PRED_CODE, format!("{}: no kernel for {:?}", r.name, r.kind));
        Ok(match &r.kind {
            Kind::Contraction { op } => (Program::bench_op(op)?, None, false),
            Kind::Stream { op, elems } => (isolated(&stream_model(*op, *elems).ok_or_else(unsupported)?)?, None, false),
            Kind::Launch { kernel, side, chain } => {
                let prog = match kernel.as_str() {
                    "matmul" => Program::bench_op(&kiln_ir::bench::BenchOp::gemm(*side, *side, *side, false))?,
                    _ => isolated(&stream_model(StreamOp::Add, 1).expect("add"))?,
                };
                (prog, Some(*chain), kernel == "empty")
            }
            _ => return Err(unsupported()),
        })
    }

    /// Maps and lowers a record's isolated program once (the mapping does not depend on parameters).
    pub fn case(&self, r: &Record) -> Result<Case, Diagnostic> {
        let (prog, chain, empty) = self.program(r)?;
        let opts = SimOptions { cost_model: Some(self.cost.clone()), ..SimOptions::default() };
        let (_, _, graph) = kiln_sim::run::map_program(self.view(), &prog, &opts, "calib")?;
        Ok(Case { prog, graph, chain, empty })
    }

    /// Predicted seconds in the record's measured semantics. Launch probes report per kernel on GPUs
    /// (`t_launch` amortized over the chain) and per loop iteration on static-dataflow chips (one sync per
    /// iteration, kernels issued back to back inside the program).
    pub fn predict(&self, c: &Case, p: &SimParams, core_hz: Option<f64>) -> f64 {
        let clocks = self.clocks(core_hz);
        let one = if c.empty {
            p.t_min_kernel + p.t_gap + p.t_sync
        } else {
            kiln_sim::recost(self.view(), &c.prog, &c.graph, Scope::Op, p, &clocks)
        };
        match c.chain {
            None => one,
            Some(n) if self.view().hw.exec_model == kiln_ir::hw::types::ExecModel::StaticDataflow => {
                n as f64 * (one - p.t_sync).max(0.0) + p.t_sync
            }
            Some(n) => one + p.t_launch / n as f64,
        }
    }
}

/// Full-path predictions (official engine entry points with a calibration set), used for test records.
pub struct Official<'a> {
    pub bench: &'a Bench,
    pub opts: SimOptions,
}

impl<'a> Official<'a> {
    pub fn new(bench: &'a Bench, set: Option<Arc<CalibSet>>, params: Option<kiln_sim::ParamSet>, interval: IntervalMethod) -> Self {
        let opts = SimOptions {
            interval,
            cost_model: Some(bench.cost.clone()),
            calibration: set,
            params,
            shadow_prices: false,
            layer_scope_fallback: true,
            ..SimOptions::default()
        };
        Official { bench, opts }
    }

    pub fn op(&self, r: &Record) -> Result<f64, Diagnostic> {
        let Kind::Contraction { op } = &r.kind else { return Err(Diagnostic::error(PRED_CODE, "not a contraction")) };
        let o = SimOptions { interval: IntervalMethod::None, ..self.opts.clone() };
        Ok(kiln_sim::simulate_bench_op(&self.bench.prepared, op, &o)?.central.makespan_s)
    }

    /// Whole step of Llama-3-8B with `layers` layers (TPU v5e measures 16 because 32 do not fit).
    pub fn step(&self, phase: &str, layers: u64) -> Result<(Interval, f64), Diagnostic> {
        let mut cfg = kiln_wl::zoo::preset("llama3_8b")?;
        cfg.n_layers = layers as u32;
        let model = kiln_wl::zoo::build_model(&cfg)?;
        let sc = kiln_wl::zoo::scenario(phase)?;
        let (_, lg, st) = kiln_wl::evaluate_snapshot(&model, &sc).map_err(|mut e| e.remove(0))?;
        let r = st.resident;
        let prog = Program::whole_step(&model, &lg, self.opts.window)?.with_resident(r.weights + r.kv_cache + r.constants);
        let prov = kiln_sim::evaluate::base_provenance(&self.bench.prepared.design.hash, "", &self.opts);
        let run = kiln_sim::simulate(self.bench.view(), &prog, id("step"), Scope::Step, &self.opts, prov)?;
        if let Some((g, need, cap)) = &run.report.capacity_overflow {
            return Err(Diagnostic::error("E-MAP-CAP-001", format!("{phase} x{layers}: resident {need} B exceeds {g} capacity {cap} B")));
        }
        let uncal = run.central.calibration.uncalibrated_makespan_s.unwrap_or(f64::NAN);
        Ok((run.time, uncal))
    }
}
