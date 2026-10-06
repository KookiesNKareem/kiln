//! `kiln calibrate report` (06 §3.3-3.5): test-split evaluation of a calibration set with the uncalibrated
//! (`assumed-v0`) prediction alongside: per-op error on the LLM suite, whole-step phase error against the
//! best-software step, interval coverage, residual structure, noise floor and acceptance verdicts.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use kiln_ir::common::Diagnostic;
use kiln_sim::CalibSet;
use kiln_trace::IntervalMethod;
use serde::Serialize;
use serde_json::{Value, json};

use crate::predict::{Bench, Official};
use crate::records::{self, Device, Kind, Record, Split};
use crate::solve::percentile;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The set was fitted on this device's micro records (same device).
    Fit,
    /// The device contributed nothing to the set.
    HeldOut,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpRow {
    pub name: String,
    pub phase: String,
    pub class: String,
    pub count: u64,
    pub meas_s: f64,
    pub cal_s: f64,
    pub uncal_s: f64,
    pub bytes: f64,
    pub flops: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepRow {
    pub phase: String,
    pub layers: u64,
    pub attn: String,
    pub meas_s: f64,
    pub central_s: f64,
    pub low_time_s: f64,
    pub high_time_s: f64,
    pub uncal_s: f64,
    pub covered: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub n: usize,
    pub median: f64,
    pub p90: f64,
    pub geomean: f64,
}

pub fn stats(ratios: &[f64]) -> Stats {
    let abs: Vec<f64> = ratios.iter().map(|r| (r - 1.0).abs()).collect();
    Stats {
        n: ratios.len(),
        median: percentile(&abs, 0.5),
        p90: percentile(&abs, 0.9),
        geomean: (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len().max(1) as f64).exp(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Slope {
    pub regressor: String,
    pub slope: f64,
    pub t: f64,
    /// Two-sided p-value of `t` under Student's t with `n - 2` degrees of freedom.
    pub p: f64,
    /// |slope| times the regressor's observed range (log units): the effect across the data.
    pub effect: f64,
    pub significant: bool,
}

/// Univariate OLS of `y` on `x` (06 §3.3.6: significance at p < 0.01 two-sided and effect > 5%).
pub fn slope(name: &str, x: &[f64], y: &[f64]) -> Slope {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let resid: f64 = x.iter().zip(y).map(|(a, c)| (c - my - b * (a - mx)).powi(2)).sum();
    let se = if n > 2.0 && sxx > 0.0 { (resid / (n - 2.0) / sxx).sqrt() } else { f64::INFINITY };
    let t = if se > 0.0 {
        b / se
    } else if b != 0.0 {
        f64::INFINITY.copysign(b)
    } else {
        0.0
    };
    let range = x.iter().copied().fold(f64::NEG_INFINITY, f64::max) - x.iter().copied().fold(f64::INFINITY, f64::min);
    let effect = (b * range).abs();
    let p = if n > 2.0 { student_t_two_sided(t, n - 2.0) } else { 1.0 };
    Slope { regressor: name.into(), slope: b, t, p, effect, significant: p < 0.01 && effect > 0.05_f64.ln_1p() }
}

/// `P(|T| >= |t|)` for Student's t with `df` degrees of freedom: `I_{df/(df+t^2)}(df/2, 1/2)`.
fn student_t_two_sided(t: f64, df: f64) -> f64 {
    if t.is_infinite() {
        return 0.0;
    }
    reg_inc_beta(df / (df + t * t), 0.5 * df, 0.5)
}

/// Regularized incomplete beta `I_x(a, b)` (continued fraction, modified Lentz).
fn reg_inc_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - reg_inc_beta(1.0 - x, b, a);
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp() / a;
    let tiny = 1e-300;
    let (mut c, mut d) = (1.0, 1.0 - (a + b) * x / (a + 1.0));
    d = 1.0 / if d.abs() < tiny { tiny } else { d };
    let mut f = d;
    for m in 1..500 {
        let m = f64::from(m);
        for num in [m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)), -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))] {
            d = 1.0 + num * d;
            d = 1.0 / if d.abs() < tiny { tiny } else { d };
            c = 1.0 + num / c;
            if c.abs() < tiny {
                c = tiny;
            }
            f *= c * d;
        }
        if (c * d - 1.0).abs() < 1e-15 {
            break;
        }
    }
    front * f
}

/// Lanczos approximation (g = 7, n = 9), accurate to ~1e-15 for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = G[1..].iter().enumerate().fold(G[0], |s, (i, g)| s + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub device: String,
    pub role: Role,
    pub ops: Vec<OpRow>,
    pub steps: Vec<StepRow>,
    pub op_cal: Stats,
    pub op_uncal: Stats,
    pub class_cal: BTreeMap<String, Stats>,
    pub class_uncal: BTreeMap<String, Stats>,
    /// Count-weighted per-phase sums of the contraction ops (06 §3.5 `sum` form; contractions only).
    pub phase_sum: BTreeMap<String, (f64, f64)>,
    pub slopes: Vec<Slope>,
    pub noise: Vec<(String, f64, usize)>,
    /// Fraction of the expected whole steps whose measurement lies inside the predicted interval; a step whose
    /// prediction failed counts as not covered.
    pub coverage: f64,
    /// Whole-step measurements on the test split (predicted or not).
    pub expected_steps: usize,
    /// Chip-state gate per TPU session (08 §F): `(file, summary, compute-bound records gated)`.
    pub chip_state: Vec<(String, String, usize)>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    pub device: String,
    pub metric: String,
    pub value: f64,
    pub target: f64,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub set_id: String,
    pub set_hash: String,
    pub devices: Vec<DeviceReport>,
    pub verdicts: Vec<Verdict>,
}

pub fn op_class(op: &kiln_ir::bench::BenchOp) -> &'static str {
    let d = |k: &str| op.dim(k).unwrap_or(1);
    if op.kind == kiln_ir::bench::BenchKind::Bmm {
        "attn_bmm"
    } else if d("n") == 128_256 {
        "lm_head"
    } else if d("m") >= 128 {
        "gemm_compute"
    } else {
        "gemv_memory"
    }
}

/// Count-weighted `(predicted, measured)` sums per phase, one weight layout per op (the `_linear` twin where both
/// were measured).
pub fn phase_sums(ops: &[OpRow]) -> BTreeMap<String, (f64, f64)> {
    let mut out: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for o in ops {
        let twin = format!("{}_linear", o.name);
        if ops.iter().any(|x| x.name == twin) {
            continue;
        }
        let e = out.entry(o.phase.clone()).or_default();
        e.0 += o.count as f64 * o.cal_s;
        e.1 += o.count as f64 * o.meas_s;
    }
    out
}

/// Test-split evaluation of `set` on one device.
pub fn device_report(set: &Arc<CalibSet>, dev: &Device, role: Role) -> Result<DeviceReport, Diagnostic> {
    let bench = Bench::new(dev.design)?;
    let cal = Official::new(&bench, Some(set.clone()), None, IntervalMethod::Corners);
    let uncal = Official::new(&bench, None, None, IntervalMethod::None);
    let all = records::load_device(dev)?;
    let chip_state = records::chip_states(dev)?
        .into_iter()
        .map(|(f, st)| {
            let gated = all.iter().filter(|r| r.session == f && r.chip_state.is_some()).count();
            (f, st.summary(), gated)
        })
        .collect();
    let recs: Vec<Record> = all.into_iter().filter(|r| r.split == Split::Test).collect();
    let mut errors = vec![];
    let mut ops = vec![];
    for r in recs.iter().filter(|r| r.group == "llm_op") {
        let Kind::Contraction { op } = &r.kind else { continue };
        match (cal.op(r), uncal.op(r)) {
            (Ok(c), Ok(u)) => ops.push(OpRow {
                name: r.name.clone(),
                phase: r.phase.clone().unwrap_or_default(),
                class: op_class(op).into(),
                count: r.count,
                meas_s: r.meas_s,
                cal_s: c,
                uncal_s: u,
                bytes: r.bytes,
                flops: r.flops,
            }),
            (Err(e), _) | (_, Err(e)) => errors.push(format!("{}: {} {}", r.name, e.code, e.message)),
        }
    }
    let mut steps = vec![];
    let mut expected_steps = 0;
    for r in recs.iter().filter(|r| matches!(r.kind, Kind::Step { .. })) {
        let Kind::Step { phase, layers, attn } = &r.kind else { continue };
        expected_steps += 1;
        match (cal.step(phase, *layers), uncal.step(phase, *layers)) {
            (Ok((t, _)), Ok((u, _))) => steps.push(StepRow {
                phase: phase.clone(),
                layers: *layers,
                attn: attn.clone(),
                meas_s: r.meas_s,
                central_s: t.central,
                low_time_s: t.low.min(t.high),
                high_time_s: t.high.max(t.low),
                uncal_s: u.central,
                covered: t.low.min(t.high) <= r.meas_s && r.meas_s <= t.high.max(t.low),
            }),
            (Err(e), _) | (_, Err(e)) => errors.push(format!("step {phase} x{layers}: {} {}", e.code, e.message)),
        }
    }
    let ratio = |f: fn(&OpRow) -> f64| -> Vec<f64> { ops.iter().map(|o| f(o) / o.meas_s).collect() };
    let by_class = |f: fn(&OpRow) -> f64| -> BTreeMap<String, Stats> {
        let mut m: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for o in &ops {
            m.entry(o.class.clone()).or_default().push(f(o) / o.meas_s);
        }
        m.into_iter().map(|(k, v)| (k, stats(&v))).collect()
    };
    let phase_sum = phase_sums(&ops);
    let y: Vec<f64> = ops.iter().map(|o| (o.cal_s / o.meas_s).ln()).collect();
    let lx = |f: fn(&OpRow) -> f64| -> Vec<f64> { ops.iter().map(|o| f(o).max(1.0).ln()).collect() };
    let slopes = if ops.len() > 3 {
        vec![
            slope("log bytes", &lx(|o| o.bytes), &y),
            slope("log flops", &lx(|o| o.flops), &y),
            slope("log arithmetic intensity", &ops.iter().map(|o| (o.flops / o.bytes.max(1.0)).max(1e-3).ln()).collect::<Vec<_>>(), &y),
        ]
    } else {
        vec![]
    };
    let noise = dev.noise_pairs.iter().filter_map(|(a, b)| records::session_noise(dev, a, b).map(|(m, n)| (format!("{a} vs {b}"), m, n))).collect();
    let coverage = if expected_steps == 0 { f64::NAN } else { steps.iter().filter(|s| s.covered).count() as f64 / expected_steps as f64 };
    Ok(DeviceReport {
        device: dev.id.into(),
        role,
        op_cal: stats(&ratio(|o| o.cal_s)),
        op_uncal: stats(&ratio(|o| o.uncal_s)),
        class_cal: by_class(|o| o.cal_s),
        class_uncal: by_class(|o| o.uncal_s),
        phase_sum,
        slopes,
        noise,
        coverage,
        expected_steps,
        chip_state,
        errors,
        ops,
        steps,
    })
}

/// Acceptance targets (06 §3.5 as given for M2): same device per-op median 8%, p90 20%, phase sum and whole-step
/// phase 5%; held-out per-op median 15%, phase sum and whole step 12%; interval coverage 80%; no target tighter
/// than twice the measured cross-session noise.
pub fn verdicts(d: &DeviceReport) -> Vec<Verdict> {
    let noise = d.noise.iter().map(|n| n.1).fold(0.0, f64::max);
    let t = |target: f64| target.max(2.0 * noise);
    let mut v = vec![];
    let mut push = |metric: String, value: f64, target: f64| v.push(Verdict { device: d.device.clone(), metric, value, target, pass: value <= target });
    match d.role {
        Role::Fit => {
            push("per-op |err| median (LLM suite)".into(), d.op_cal.median, t(0.08));
            push("per-op |err| p90 (LLM suite)".into(), d.op_cal.p90, t(0.20));
            for s in &d.steps {
                push(format!("whole-step |err| {}", s.phase), (s.central_s / s.meas_s - 1.0).abs(), t(0.05));
            }
            for (p, (pred, meas)) in &d.phase_sum {
                push(format!("phase-sum |err| {p}"), (pred / meas - 1.0).abs(), t(0.05));
            }
        }
        Role::HeldOut => {
            push("held-out per-op |err| median (LLM suite)".into(), d.op_cal.median, t(0.15));
            for s in &d.steps {
                push(format!("held-out whole-step |err| {}", s.phase), (s.central_s / s.meas_s - 1.0).abs(), t(0.12));
            }
            for (p, (pred, meas)) in &d.phase_sum {
                push(format!("held-out phase-sum |err| {p}"), (pred / meas - 1.0).abs(), t(0.12));
            }
        }
    }
    // 06 §3.3.6: a significant residual slope or an op-class geomean bias outside [0.95, 1.05] fails.
    for (k, c) in &d.class_cal {
        push(format!("op-class geomean bias |pred/meas-1|, {k}"), (c.geomean - 1.0).abs(), t(0.05));
    }
    for e in &d.errors {
        push(format!("prediction error: {e}"), 1.0, 0.0);
    }
    for s in &d.slopes {
        v.push(Verdict { device: d.device.clone(), metric: format!("residual slope effect (significant fails), {}", s.regressor), value: s.effect.exp_m1(), target: 0.05, pass: !s.significant });
    }
    if d.expected_steps > 0 {
        v.push(Verdict { device: d.device.clone(), metric: "whole-step interval coverage (>=)".into(), value: d.coverage, target: 0.8, pass: d.coverage >= 0.8 });
    }
    v
}

pub fn report(set: &Arc<CalibSet>, devices: &[(&Device, Role)]) -> Result<Report, Diagnostic> {
    let mut out = Report { set_id: set.id.clone(), set_hash: set.compute_hash(), devices: vec![], verdicts: vec![] };
    for (dev, role) in devices {
        let d = device_report(set, dev, *role)?;
        out.verdicts.extend(verdicts(&d));
        out.devices.push(d);
    }
    Ok(out)
}

fn pct(x: f64) -> String {
    format!("{:+.1}%", 100.0 * x)
}

pub fn render(r: &Report) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "calibration report: {} ({})", r.set_id, r.set_hash);
    for d in &r.devices {
        let _ = writeln!(s, "\n== {} ({:?}) ==", d.device, d.role);
        let _ = writeln!(s, "per-op LLM suite (test, n={}): |err| median / p90 / geomean pred/meas", d.op_cal.n);
        let _ = writeln!(s, "  calibrated    {:>6.1}% / {:>6.1}% / {:.3}", 100.0 * d.op_cal.median, 100.0 * d.op_cal.p90, d.op_cal.geomean);
        let _ = writeln!(s, "  uncalibrated  {:>6.1}% / {:>6.1}% / {:.3}", 100.0 * d.op_uncal.median, 100.0 * d.op_uncal.p90, d.op_uncal.geomean);
        let _ = writeln!(s, "  by class (calibrated geomean, |err| median | uncalibrated geomean):");
        for (k, c) in &d.class_cal {
            let u = &d.class_uncal[k];
            let _ = writeln!(s, "    {k:<13} n={:<3} {:.3}  {:>5.1}%  | {:.3}", c.n, c.geomean, 100.0 * c.median, u.geomean);
        }
        let _ = writeln!(s, "  phase sums (contractions only, count-weighted): pred/meas");
        for (p, (a, b)) in &d.phase_sum {
            let _ = writeln!(s, "    {p:<11} {:.3}", a / b);
        }
        let _ = writeln!(s, "whole steps vs best software (test):");
        let _ = writeln!(s, "  {:<11} {:>3} {:<22} {:>9} {:>9} {:>17} {:>8} {:>9} {:>8} cov", "phase", "L", "impl", "meas_ms", "cal_ms", "[low, high] ms", "err", "uncal_ms", "err");
        for st in &d.steps {
            let _ = writeln!(
                s,
                "  {:<11} {:>3} {:<22} {:>9.3} {:>9.3} [{:>7.3}, {:>7.3}] {:>8} {:>9.3} {:>8} {}",
                st.phase,
                st.layers,
                st.attn,
                st.meas_s * 1e3,
                st.central_s * 1e3,
                st.low_time_s * 1e3,
                st.high_time_s * 1e3,
                pct(st.central_s / st.meas_s - 1.0),
                st.uncal_s * 1e3,
                pct(st.uncal_s / st.meas_s - 1.0),
                if st.covered { "yes" } else { "NO" }
            );
        }
        let _ = writeln!(s, "  interval coverage {:.0}%", 100.0 * d.coverage);
        let _ = writeln!(s, "residual structure (test ops, log(pred/meas) on regressor):");
        for sl in &d.slopes {
            let _ = writeln!(s, "  {:<26} slope {:+.4}  t {:+6.2}  p {:.2e}  effect {:.1}%  {}", sl.regressor, sl.slope, sl.t, sl.p, 100.0 * sl.effect.exp_m1(), if sl.significant { "SIGNIFICANT" } else { "ok" });
        }
        for (k, c) in &d.class_cal {
            if !(0.95..=1.05).contains(&c.geomean) {
                let _ = writeln!(s, "  class bias {k}: geomean {:.3} outside [0.95, 1.05]", c.geomean);
            }
        }
        if !d.chip_state.is_empty() {
            let _ = writeln!(s, "chip-state gate (08 §F; compute-bound records of changed sessions never enter a fit):");
            for (f, summary, gated) in &d.chip_state {
                let _ = writeln!(s, "  {f}: {summary}; {gated} compute-bound records gated");
            }
        }
        let _ = writeln!(s, "noise floor (median |a/b-1| across sessions):");
        for (k, m, n) in &d.noise {
            let _ = writeln!(s, "  {k}: {:.2}% (n={n})", 100.0 * m);
        }
        for e in &d.errors {
            let _ = writeln!(s, "  error: {e}");
        }
        let _ = writeln!(s, "per-op detail (pred/meas):");
        for o in &d.ops {
            let _ = writeln!(s, "  {:<40} {:<13} meas {:>9.2}us  cal {:.3}  uncal {:.3}", o.name, o.class, o.meas_s * 1e6, o.cal_s / o.meas_s, o.uncal_s / o.meas_s);
        }
    }
    let _ = writeln!(s, "\nacceptance:");
    for v in &r.verdicts {
        let _ = writeln!(s, "  {:<4} {:<10} {:<44} {:>7.1}% vs {:>5.1}%", if v.pass { "PASS" } else { "FAIL" }, v.device, v.metric, 100.0 * v.value, 100.0 * v.target);
    }
    s
}

/// Snapshot stored as the set's `acceptance` (not hashed).
pub fn snapshot(r: &Report) -> Value {
    json!({
        "set_hash": r.set_hash,
        "verdicts": r.verdicts,
        "devices": r.devices.iter().map(|d| json!({
            "device": d.device, "role": d.role, "op_cal": d.op_cal, "op_uncal": d.op_uncal,
            "steps": d.steps, "coverage": d.coverage, "slopes": d.slopes,
        })).collect::<Vec<_>>(),
    })
}
