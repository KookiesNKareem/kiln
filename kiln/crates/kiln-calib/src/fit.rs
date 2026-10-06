//! Staged calibration fits (03 §9, 06 §3.3) on calib-micro records only, per device, in the fixed order
//! P (clock operating points from telemetry), L (launch / sync), D (DRAM), U (unit efficiency). Each stage
//! fits its own records with earlier stages frozen; no joint refit. A parameter converging onto a bound is
//! not accepted (frozen at its prior, flagged, and the fit fails); one whose bootstrap CI spans more than half
//! its bound range is frozen as non-identifiable; a stage with fewer than 10 records per free parameter is
//! refused.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::types::ExecModel;
use kiln_sim::SimParams;
use kiln_sim::calib::{CalParam, FitDiag, ParamStatus, RangeSpec, registered, unit_templates};
use kiln_sim::params::{ParamSet, PessDir, exec_key};
use kiln_trace::Corner;

use crate::predict::{Bench, Case};
use crate::records::{Device, Kind, Record, Split, StreamOp};
use crate::solve::{Bounded, Rng, minimize, percentile};

pub const LEAK_CODE: &str = "E-CAL-LEAK";
pub const BOUND_CODE: &str = "E-CAL-BOUND";
pub const BOOTSTRAP_CODE: &str = "E-CAL-BOOTSTRAP";
pub const MIN_RECORDS_PER_PARAM: usize = 10;
pub const BOOTSTRAP: usize = 200;
pub const BOOTSTRAP_SEED: u64 = 0x6b69_6c6e_6d32;
const MIB: u64 = 1 << 20;

#[derive(Clone, Debug, PartialEq)]
pub struct Free {
    pub name: &'static str,
    pub key: BTreeMap<String, String>,
    pub b: Bounded,
}

impl Free {
    fn new(name: &'static str, key: (&str, &str)) -> Free {
        let r = registered(name).expect("registered");
        Free { name, key: BTreeMap::from([(key.0.to_string(), key.1.to_string())]), b: Bounded { lo: r.bounds.0, hi: r.bounds.1, prior: r.prior } }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageOut {
    pub stage: &'static str,
    pub records: Vec<usize>,
    pub free: Vec<Free>,
    pub value: Vec<f64>,
    pub ci: Vec<[f64; 2]>,
    pub at_bound: Vec<bool>,
    pub frozen: Vec<Option<String>>,
    pub loss: f64,
    pub iterations: usize,
    /// Log residuals of the stage's fit records at the accepted values.
    pub residuals: Vec<f64>,
    pub refused: Option<String>,
}

/// Telemetry operating points of a power-capped clock domain (stage P).
#[derive(Clone, Debug, PartialEq)]
pub struct ClockFit {
    pub domain: String,
    pub cap: String,
    pub clock_ix: usize,
    pub points: Vec<[f64; 2]>,
    pub range: (f64, f64),
    pub n: usize,
}

pub struct DeviceFit {
    pub device: &'static Device,
    pub records: Vec<Record>,
    pub stages: Vec<StageOut>,
    pub clock: Option<ClockFit>,
    pub params: SimParams,
    pub entries: Vec<CalParam>,
    /// `(record index, log(pred/meas))` for fit and diagnostic hold-out records at the final parameters.
    pub residuals: Vec<(usize, f64)>,
    pub failures: Vec<Diagnostic>,
    pub uncal_residuals: Vec<(usize, f64)>,
}

/// Sets one named engine parameter (the single place each enters, 03 §9).
pub fn apply(p: &mut SimParams, name: &str, v: f64, unit_paths: &[String]) {
    match name {
        "eta_res" => p.eta_dram = v,
        "t_dram_ramp" => p.t_dram_ramp = v,
        "t_launch" => p.t_launch = v,
        "t_min_kernel" => p.t_min_kernel = v,
        "t_gap" => p.t_gap = v,
        "t_dispatch" => p.t_dispatch = v,
        "t_program" => p.t_program = v,
        "t_sync" => p.t_sync = v,
        "unit_eff" => p.unit_eff = unit_paths.iter().map(|t| (t.clone(), v)).collect(),
        _ => {}
    }
}

fn is_ew(op: StreamOp) -> bool {
    matches!(op, StreamOp::Add | StreamOp::Scale | StreamOp::Silu)
}

/// Bytes of one input operand at or below which an elementwise kernel is a launch-path (L) record.
fn tiny_limit(exec: ExecModel) -> u64 {
    if exec == ExecModel::StaticDataflow { MIB } else { 4 * MIB }
}

/// Stage membership of a fit-suite record (by mechanism it isolates; group names from the runner).
pub fn stage_of(r: &Record, exec: ExecModel) -> Option<&'static str> {
    match &r.kind {
        Kind::Launch { kernel, .. } if kernel == "empty" => Some("L"),
        Kind::Stream { op, elems } if is_ew(*op) && 2 * elems <= tiny_limit(exec) => Some("L"),
        Kind::Stream { .. } => Some("D"),
        Kind::Contraction { op } if r.group == "gemv" || (op.kind == kiln_ir::bench::BenchKind::Bmm && op.dim("m").unwrap_or(1) <= 16) => Some("D"),
        Kind::Contraction { .. } => Some("U"),
        _ => None,
    }
}

pub struct Fitter<'a> {
    pub bench: &'a Bench,
    pub recs: &'a [Record],
    pub cases: Vec<Option<Case>>,
    pub unit_paths: Vec<String>,
    pub unit_key: Option<String>,
    pub exec: ExecModel,
    pub dram: String,
    pub threads: usize,
    pub bootstrap: usize,
    /// Stages to run, in the fixed order P, L, D, U; skipped stages keep their priors (frozen).
    pub stages: Vec<String>,
}

impl<'a> Fitter<'a> {
    pub fn new(bench: &'a Bench, recs: &'a [Record]) -> Fitter<'a> {
        let v = bench.view();
        let cases = recs
            .iter()
            .map(|r| match r.split {
                Split::Fit | Split::FitHoldout if !matches!(r.kind, Kind::PowerStep | Kind::OnChip | Kind::Step { .. }) => bench.case(r).ok(),
                _ => None,
            })
            .collect();
        let units = unit_templates(v);
        let exec = v.hw.exec_model;
        let dram = kiln_sim::calib::design_keys(v, exec).get("dram_kind").cloned().unwrap_or_else(|| "sram".into());
        let mut matrix: Vec<(String, String)> = units.into_iter().filter(|(_, k)| !k.ends_with(":mac")).collect();
        matrix.sort();
        Fitter {
            bench,
            recs,
            cases,
            unit_paths: matrix.iter().map(|x| x.0.clone()).collect(),
            unit_key: matrix.first().map(|x| x.1.clone()),
            exec,
            dram,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
            bootstrap: BOOTSTRAP,
            stages: ["P", "L", "D", "U"].map(String::from).to_vec(),
        }
    }

    /// Registry priors of the parameters this design's execution model and DRAM kind use.
    pub fn base(&self) -> SimParams {
        let set = ParamSet::assumed(self.exec, &self.dram);
        let mut p = set.at(Corner::Central);
        for name in set.params.iter().map(|x| x.name.as_str()).chain(["t_dram_ramp"]) {
            if let Some(r) = registered(name) {
                apply(&mut p, name, r.prior, &self.unit_paths);
            }
        }
        p
    }

    /// Log residual of record `i` at engine parameters `p` (telemetry clock when measured).
    pub fn residual(&self, i: usize, p: &SimParams) -> f64 {
        let r = &self.recs[i];
        let c = self.cases[i].as_ref().expect("fit records have cases");
        (self.bench.predict(c, p, r.clock_hz) / r.meas_s).ln()
    }

    fn with(&self, base: &SimParams, free: &[Free], x: &[f64]) -> SimParams {
        let mut p = base.clone();
        for (f, &v) in free.iter().zip(x) {
            apply(&mut p, f.name, v, &self.unit_paths);
        }
        p
    }

    /// Fits one stage on `idx` (must all be `Split::Fit`: the 06 leakage guard). A parameter that converges
    /// onto a bound is frozen at its prior and the stage is refit over the rest, so the set stays coherent while
    /// the failure is recorded.
    pub fn stage(&self, name: &'static str, idx: Vec<usize>, free: Vec<Free>, base: &SimParams) -> Result<StageOut, Diagnostic> {
        if let Some(&i) = idx.iter().find(|&&i| self.recs[i].split != Split::Fit) {
            return Err(Diagnostic::error(LEAK_CODE, format!("stage {name}: record {} ({:?}) is not in the fit split", self.recs[i].name, self.recs[i].split))
                .hint("only calib-micro fit records may enter a fit (06 §3.4)"));
        }
        if let Some(&i) = idx.iter().find(|&&i| self.cases[i].is_none()) {
            return Err(Diagnostic::error(LEAK_CODE, format!("stage {name}: record {} has no prediction", self.recs[i].name)));
        }
        let n = free.len();
        let priors: Vec<f64> = free.iter().map(|f| f.b.prior).collect();
        let mut out = StageOut {
            stage: name,
            records: idx.clone(),
            value: priors.clone(),
            ci: priors.iter().map(|&p| [p, p]).collect(),
            at_bound: vec![false; n],
            frozen: vec![None; n],
            loss: 0.0,
            iterations: 0,
            residuals: vec![],
            refused: None,
            free,
        };
        if n == 0 {
            return Ok(out);
        }
        if idx.len() < MIN_RECORDS_PER_PARAM * n {
            let why = format!("refused: {} fit records for {n} free parameters (< {MIN_RECORDS_PER_PARAM} per parameter, 06 §3.1 rule 5)", idx.len());
            out.frozen = vec![Some(why.clone()); n];
            out.refused = Some(why);
            out.residuals = idx.iter().map(|&i| self.residual(i, &self.with(base, &out.free, &priors))).collect();
            return Ok(out);
        }
        let (sol, active) = loop {
            let active: Vec<usize> = (0..n).filter(|&j| !out.at_bound[j]).collect();
            if active.is_empty() {
                break (None, active);
            }
            let fixed = self.with(base, &out.free, &out.value);
            let free: Vec<Free> = active.iter().map(|&j| out.free[j].clone()).collect();
            let bounded: Vec<Bounded> = free.iter().map(|f| f.b.clone()).collect();
            let sol = minimize(&bounded, None, 200, &|x: &[f64]| {
                let p = self.with(&fixed, &free, x);
                idx.iter().map(|&i| self.residual(i, &p)).collect()
            });
            let mut hit = false;
            for (a, &j) in active.iter().enumerate() {
                out.value[j] = sol.x[a];
                if sol.at_bound[a] {
                    hit = true;
                    out.at_bound[j] = true;
                    out.frozen[j] = Some(format!(
                        "fit converged onto its bound ({:.4e} in [{:.4e}, {:.4e}]): the structural model is wrong for this mechanism on this \
                         platform (03 §9 principle 2); frozen at its prior and the stage refit without it",
                        sol.x[a], out.free[j].b.lo, out.free[j].b.hi
                    ));
                    out.value[j] = out.free[j].b.prior;
                }
            }
            if !hit {
                break (Some(sol), active);
            }
        };
        if let Some(sol) = sol {
            out.loss = sol.loss;
            out.iterations = sol.iterations;
            let fixed = self.with(base, &out.free, &out.value);
            let free: Vec<Free> = active.iter().map(|&j| out.free[j].clone()).collect();
            let bounded: Vec<Bounded> = free.iter().map(|f| f.b.clone()).collect();
            let at = |sel: &[usize], x: &[f64]| -> Vec<f64> {
                let p = self.with(&fixed, &free, x);
                sel.iter().map(|&i| self.residual(i, &p)).collect()
            };
            // Bootstrap CI: resample records with replacement, warm start at the optimum.
            let bs = self.bootstrap;
            let chunks: Vec<Vec<usize>> = (0..self.threads).map(|t| (0..bs).filter(|b| b % self.threads == t).collect()).collect();
            let draws: Vec<(usize, Vec<f64>)> = std::thread::scope(|s| {
                let hs: Vec<_> = chunks
                    .iter()
                    .map(|ch| {
                        let (idx, bounded, x0, at) = (&idx, &bounded, &sol.x, &at);
                        s.spawn(move || {
                            ch.iter()
                                .map(|&b| {
                                    let mut rng = Rng::new(BOOTSTRAP_SEED ^ (b as u64).wrapping_mul(0x9e37_79b9));
                                    let sel: Vec<usize> = (0..idx.len()).map(|_| idx[rng.below(idx.len())]).collect();
                                    (b, minimize(bounded, Some(x0), 30, &|x: &[f64]| at(&sel, x)).x)
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                let mut all: Vec<(usize, Vec<f64>)> = hs.into_iter().flat_map(|h| h.join().expect("bootstrap thread")).collect();
                all.sort_by_key(|x| x.0);
                all
            });
            for (a, &j) in active.iter().enumerate() {
                let v: Vec<f64> = draws.iter().map(|d| d.1[a]).collect();
                out.ci[j] = [percentile(&v, 0.025), percentile(&v, 0.975)];
                let b = &out.free[j].b;
                if out.ci[j][1] - out.ci[j][0] > 0.5 * (b.hi - b.lo) {
                    out.frozen[j] = Some(format!("non-identifiable: bootstrap CI95 [{:.4e}, {:.4e}] spans more than half its bound range", out.ci[j][0], out.ci[j][1]));
                    out.value[j] = b.prior;
                }
            }
        }
        out.residuals = idx.iter().map(|&i| self.residual(i, &self.with(base, &out.free, &out.value))).collect();
        Ok(out)
    }

    /// Stage P: sustained clock vs MAC activity from NVML telemetry of the GEMM sweeps (isotonic, non-increasing).
    pub fn clock_stage(&self) -> Option<ClockFit> {
        let v = self.bench.view();
        let (ix, (domain, cap)) = kiln_sim::calib::capped_clocks(v).into_iter().next()?;
        let nom = v.phys.nominal_hz(ix);
        let peak = v.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None);
        let mut pts: Vec<(f64, f64)> = self
            .recs
            .iter()
            .filter(|r| r.split == Split::Fit && matches!(r.kind, Kind::Contraction { .. }))
            .filter_map(|r| {
                let f = r.clock_hz?;
                (r.flops > 0.0 && r.meas_s > 0.0).then(|| ((r.flops / r.meas_s) / (peak * f / nom), f))
            })
            .collect();
        for r in self.recs.iter().filter(|r| r.split == Split::Fit && matches!(r.kind, Kind::PowerStep) && r.name.contains("copy")) {
            pts.extend(r.clock_hz.map(|f| (0.0, f)));
        }
        if pts.len() < MIN_RECORDS_PER_PARAM {
            return None;
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        // Pool-adjacent-violators for a non-increasing fit.
        let mut blocks: Vec<(f64, f64, f64)> = vec![]; // (sum x, sum f, n)
        for &(x, f) in &pts {
            blocks.push((x, f, 1.0));
            while blocks.len() > 1 {
                let (a, b) = (blocks[blocks.len() - 2], blocks[blocks.len() - 1]);
                if a.1 / a.2 >= b.1 / b.2 {
                    break;
                }
                blocks.pop();
                let l = blocks.len() - 1;
                blocks[l] = (a.0 + b.0, a.1 + b.1, a.2 + b.2);
            }
        }
        let all: Vec<[f64; 2]> = blocks.iter().map(|b| [b.0 / b.2, (b.1 / b.2).min(nom)]).collect();
        // Plateaus keep their first and last knot (interpolation is unchanged).
        let points: Vec<[f64; 2]> = (0..all.len())
            .filter(|&i| i == 0 || i + 1 == all.len() || all[i][1] != all[i - 1][1] || all[i][1] != all[i + 1][1])
            .map(|i| all[i])
            .collect();
        let table = kiln_sim::params::ClockOp { clocks: vec![ix], points: points.iter().map(|p| (p[0], p[1])).collect(), scale: 1.0 };
        // Spread where the cap binds (below nominal); unthrottled points sit at nominal by construction.
        let ratios: Vec<f64> = pts.iter().filter(|&&(x, _)| table.hz(x) < nom).map(|&(x, f)| f / table.hz(x)).collect();
        let reg = registered("f_cap_op").expect("registered");
        // No point below nominal: the cap never bound in telemetry, so its spread is the assumed band.
        let (lo, hi) = if ratios.is_empty() {
            let (lo, hi, _) = assumed_band("f_cap_op", 1.0, reg.bounds);
            (lo, hi)
        } else {
            (percentile(&ratios, 0.1), percentile(&ratios, 0.9))
        };
        let (lo, hi) = (lo.clamp(reg.bounds.0, 1.0), hi.clamp(1.0, reg.bounds.1));
        Some(ClockFit { domain, cap, clock_ix: ix, points, range: (lo, hi), n: pts.len() })
    }

    pub fn run(&self) -> Result<DeviceFit, Diagnostic> {
        if self.bootstrap == 0 {
            return Err(Diagnostic::error(BOOTSTRAP_CODE, "bootstrap needs at least one resample: fitted parameters carry bootstrap CI95 ranges")
                .hint(format!("use --bootstrap {BOOTSTRAP} (default) or any positive count")));
        }
        let mut base = self.base();
        let pick = |stage: &str| -> Vec<usize> {
            (0..self.recs.len()).filter(|&i| self.recs[i].split == Split::Fit && self.cases[i].is_some() && stage_of(&self.recs[i], self.exec) == Some(stage)).collect()
        };
        let ek = exec_key(self.exec);
        let l_free = match self.exec {
            ExecModel::HostLaunched => vec![Free::new("t_gap", ("exec_model", ek)), Free::new("t_min_kernel", ("exec_model", ek))],
            ExecModel::DeviceQueued => vec![Free::new("t_dispatch", ("exec_model", ek))],
            ExecModel::StaticDataflow => vec![Free::new("t_sync", ("exec_model", ek))],
        };
        let mut stages = vec![];
        let on = |st: &str| self.stages.iter().any(|x| x == st);
        let run_stage = |name: &'static str, free: Vec<Free>, base: &SimParams| -> Result<StageOut, Diagnostic> {
            if on(name) {
                return self.stage(name, pick(name), free, base);
            }
            let mut out = self.stage(name, vec![], vec![], base)?;
            out.value = free.iter().map(|f| f.b.prior).collect();
            out.ci = out.value.iter().map(|&v| [v, v]).collect();
            out.at_bound = vec![false; free.len()];
            out.frozen = vec![Some(format!("stage {name} not run (--stages)")); free.len()];
            out.free = free;
            Ok(out)
        };
        let clock = if on("P") { self.clock_stage() } else { None };
        let l = run_stage("L", l_free, &base)?;
        for (f, &v) in l.free.iter().zip(&l.value) {
            apply(&mut base, f.name, v, &self.unit_paths);
        }
        stages.push(l);
        let dram = self.dram.as_str();
        let d = run_stage("D", vec![Free::new("eta_res", ("dram_kind", dram)), Free::new("t_dram_ramp", ("dram_kind", dram))], &base)?;
        for (f, &v) in d.free.iter().zip(&d.value) {
            apply(&mut base, f.name, v, &self.unit_paths);
        }
        stages.push(d);
        let u_free = self.unit_key.as_deref().map(|k| vec![Free::new("unit_eff", ("unit_template", k))]).unwrap_or_default();
        let u = run_stage("U", u_free, &base)?;
        for (f, &v) in u.free.iter().zip(&u.value) {
            apply(&mut base, f.name, v, &self.unit_paths);
        }
        stages.push(u);
        let mut failures = vec![];
        for s in &stages {
            for (j, f) in s.free.iter().enumerate() {
                if s.at_bound[j] {
                    failures.push(
                        Diagnostic::error(BOUND_CODE, format!("{} {}: {}", self.bench.prepared.design.hash, f.name, s.frozen[j].clone().unwrap_or_default()))
                            .at(format!("{}:{}", self.recs.first().map_or("", |r| r.device.as_str()), f.name))
                            .hint("fix the mechanism (structure before residual, 03 §9 principle 4); the value is not accepted"),
                    );
                }
            }
        }
        let resid_of = |p: &SimParams| -> Vec<(usize, f64)> {
            (0..self.recs.len())
                .filter(|&i| matches!(self.recs[i].split, Split::Fit | Split::FitHoldout) && self.cases[i].is_some() && stage_of(&self.recs[i], self.exec).is_some())
                .map(|i| (i, self.residual(i, p)))
                .collect()
        };
        let residuals = resid_of(&base);
        let uncal = ParamSet::assumed(self.exec, &self.dram).at(Corner::Central);
        let uncal_residuals = resid_of(&uncal);
        let entries = self.entries(&stages, clock.as_ref());
        Ok(DeviceFit { device: crate::records::device(&self.recs[0].device).expect("device"), records: self.recs.to_vec(), stages, clock, params: base, entries, residuals, failures, uncal_residuals })
    }

    /// Mechanism-keyed set entries: fitted values with ranges from bootstrap CI and residual dispersion, frozen
    /// priors with assumed bands, and telemetry operating points.
    fn entries(&self, stages: &[StageOut], clock: Option<&ClockFit>) -> Vec<CalParam> {
        let mut out = vec![];
        let source = |s: &StageOut| format!("stage {} on {} calib-micro fit records ({})", s.stage, s.records.len(), self.recs[0].session);
        for s in stages {
            let mad = {
                let med = percentile(&s.residuals, 0.5);
                let dev: Vec<f64> = s.residuals.iter().map(|r| (r - med).abs()).collect();
                1.4826 * percentile(&dev, 0.5)
            };
            let abs: Vec<f64> = s.residuals.iter().map(|r| r.exp_m1().abs()).collect();
            for (j, f) in s.free.iter().enumerate() {
                let reg = registered(f.name).expect("registered");
                let v = s.value[j];
                let frozen = s.frozen[j].is_some();
                let (lower, upper, basis) = if frozen {
                    assumed_band(f.name, v, reg.bounds)
                } else {
                    let half = (0.5 * (s.ci[j][1] - s.ci[j][0])).max(mad * v.abs());
                    ((v - half).max(reg.bounds.0), (v + half).min(reg.bounds.1), "single_device".to_string())
                };
                out.push(CalParam {
                    name: f.name.into(),
                    key: f.key.clone(),
                    value: v,
                    unit: reg.unit.into(),
                    bounds: [reg.bounds.0, reg.bounds.1],
                    prior: reg.prior,
                    ci95: (!frozen).then_some(s.ci[j]),
                    range: RangeSpec { lower: lower.min(v), upper: upper.max(v), basis },
                    pess_dir: reg.pess_dir,
                    status: if frozen { ParamStatus::Frozen } else { ParamStatus::Fit },
                    frozen_reason: s.frozen[j].clone(),
                    source: Some(source(s)),
                    table: None,
                    diagnostics: Some(FitDiag {
                        stage: s.stage.into(),
                        n_records: s.records.len(),
                        loss: s.loss,
                        abs_err_median: percentile(&abs, 0.5),
                        abs_err_p90: percentile(&abs, 0.9),
                        at_bound: s.at_bound[j],
                        iterations: s.iterations,
                        per_device: BTreeMap::from([(self.recs[0].device.clone(), s.value[j])]),
                    }),
                });
            }
        }
        let ek = exec_key(self.exec);
        let frozen_only: &[(&str, &str)] = match self.exec {
            ExecModel::HostLaunched => &[(
                "t_launch",
                "frozen at prior: only the empty-kernel chain intercept identifies it (3 records < 10 per parameter); charged once per captured step",
            )],
            ExecModel::StaticDataflow => &[("t_program", "frozen at prior: loop-mode records never start a program; host dispatch (`single`) is not a program start")],
            ExecModel::DeviceQueued => &[],
        };
        for &(name, why) in frozen_only {
            let reg = registered(name).expect("registered");
            let (lower, upper, basis) = assumed_band(name, reg.prior, reg.bounds);
            out.push(CalParam {
                name: name.into(),
                key: BTreeMap::from([("exec_model".to_string(), ek.to_string())]),
                value: reg.prior,
                unit: reg.unit.into(),
                bounds: [reg.bounds.0, reg.bounds.1],
                prior: reg.prior,
                ci95: None,
                range: RangeSpec { lower, upper, basis },
                pess_dir: reg.pess_dir,
                status: ParamStatus::Frozen,
                frozen_reason: Some(why.into()),
                source: None,
                table: None,
                diagnostics: None,
            });
        }
        if let Some(c) = clock {
            let reg = registered("f_cap_op").expect("registered");
            out.push(CalParam {
                name: "f_cap_op".into(),
                key: BTreeMap::from([("domain".to_string(), c.domain.clone()), ("power_cap".to_string(), c.cap.clone())]),
                value: 1.0,
                unit: reg.unit.into(),
                bounds: [reg.bounds.0, reg.bounds.1],
                prior: reg.prior,
                ci95: None,
                range: RangeSpec { lower: c.range.0, upper: c.range.1, basis: "telemetry".into() },
                pess_dir: PessDir::Lower,
                status: ParamStatus::Telemetry,
                frozen_reason: None,
                source: Some(format!(
                    "telemetry-derived: NVML sustained clock vs MAC activity of {} calib-micro GEMM/GEMV records (isotonic, non-increasing); no power model",
                    c.n
                )),
                table: Some(c.points.clone()),
                diagnostics: None,
            });
        }
        out
    }
}

/// Range of a parameter that is not fitted: the assumed-v0 band where M1 defines one, else the registry
/// plausible band, else +-50% of the prior within bounds.
pub fn assumed_band(name: &str, v: f64, bounds: (f64, f64)) -> (f64, f64, String) {
    for exec in [ExecModel::HostLaunched, ExecModel::StaticDataflow, ExecModel::DeviceQueued] {
        if let Some(p) = ParamSet::assumed(exec, "x").params.iter().find(|p| p.name == name) {
            return (p.range.lower.min(v), p.range.upper.max(v), "band".into());
        }
    }
    match registered(name).and_then(|r| r.band) {
        Some((lo, hi)) => (lo.min(v), hi.max(v), "band".into()),
        None => ((0.5 * v).max(bounds.0), (1.5 * v).min(bounds.1), "band".into()),
    }
}
