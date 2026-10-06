//! M1 end-to-end report: Llama-3-8B whole steps and isolated per-op predictions vs the 2026-10-05 measurements.
//! `cargo run --release -p kiln-sim --example m1_report [-- --roofline] [--only=<design>]`
//!
//! Predictions run at nominal clocks (no calibrated power model in M1); for the A100 the report also shows the
//! effect of the clock observed under the 400 W cap on compute-bound work (decode runs at 1410 MHz), as a
//! what-if, never as the prediction.

use std::collections::BTreeMap;
use std::path::PathBuf;

use kiln_ir::hw::Profile;
use kiln_phys::ClockMode;
use kiln_sim::{Prepared, SimOptions, simulate_bench_op, simulate_member};
use kiln_trace::IntervalMethod;
use kiln_trace::meas::runner::{GPU_OP_MODES, OpRecord, StepRecord, TPU_OP_MODES, op_records, step_records};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(root().join("calibration/measurements").join(name)).unwrap_or_default()
}

struct Meas {
    suite: &'static str,
    /// Per-op timing modes counted as "best software" (empty = all).
    op_modes: &'static [&'static str],
    seqs: &'static [&'static str],
    /// Whole-step attention implementations and modes counted (empty = all that passed their check).
    attn: &'static [&'static str],
    step_modes: &'static [&'static str],
    /// Observed sustained clock of the first clock domain under load (what-if only).
    observed_hz: Option<f64>,
}

/// Best-software whole-step medians by phase: `(seconds, layers measured, implementation)`.
fn steps(m: &Meas) -> BTreeMap<String, (f64, u64, String)> {
    let mut out: BTreeMap<String, (f64, u64, String)> = BTreeMap::new();
    for f in m.seqs {
        for s in step_records(&read(f), f).unwrap_or_default() {
            if s.checked == Some(false) || !(m.attn.is_empty() || m.attn.contains(&s.attn.as_str())) {
                continue;
            }
            let Some(t) = best(&s, m.step_modes) else { continue };
            let e = out.entry(s.phase.clone()).or_insert((f64::INFINITY, s.n_layers, String::new()));
            if t < e.0 {
                *e = (t, s.n_layers, s.attn.clone());
            }
        }
    }
    out
}

fn best(s: &StepRecord, modes: &[&str]) -> Option<f64> {
    s.median_s.iter().filter(|(k, _)| modes.is_empty() || modes.contains(&k.as_str())).map(|x| *x.1).reduce(f64::min)
}

/// Whole-step prediction for Llama-3-8B with `layers` layers (TPU v5e measures 16 because 32 do not fit).
fn predict_layers(p: &Prepared, phase: &str, layers: u32, opts: &SimOptions) -> Result<f64, String> {
    let mut cfg = kiln_wl::zoo::preset("llama3_8b").unwrap();
    cfg.n_layers = layers;
    let model = kiln_wl::zoo::build_model(&cfg).unwrap();
    let sc = kiln_wl::zoo::scenario(phase).unwrap();
    let (_, lg, st) = kiln_wl::evaluate_snapshot(&model, &sc).unwrap();
    let r = st.resident;
    let prog = kiln_map::Program::whole_step(&model, &lg, opts.window).unwrap().with_resident(r.weights + r.kv_cache + r.constants);
    let prov = kiln_sim::evaluate::base_provenance("", "", opts);
    let run = kiln_sim::simulate(&p.view, &prog, kiln_ir::common::Id::new("x").unwrap(), kiln_trace::sim::Scope::Step, opts, prov).map_err(|e| e.code.clone())?;
    match run.report.capacity_overflow {
        Some(_) => Err("E-MAP-CAP-001".into()),
        None => Ok(run.central.makespan_s),
    }
}

fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len().max(1) as f64).exp()
}

fn at_clock(p: &Prepared, hz: f64) -> ClockMode {
    let mut v: Vec<f64> = (0..p.view.hw.clocks.len()).map(|c| p.view.phys.nominal_hz(c)).collect();
    v[0] = hz;
    ClockMode::Fixed(v)
}

fn main() {
    let only: Option<String> = std::env::args().find_map(|a| a.strip_prefix("--only=").map(String::from));
    let a100 = Meas {
        suite: "a100_2026-10-05_suite_r2.json",
        op_modes: GPU_OP_MODES,
        seqs: &["a100_2026-10-05_seq.json", "a100_2026-10-05_seq_r2.json"],
        attn: &["sdpa"],
        step_modes: &["graph"],
        observed_hz: Some(1.29e9),
    };
    // Canonical TPU per-op mode: in-program `loop`, as whole steps run (08 §F); `pipelined_rot` is diagnostic only.
    let tpu = |suite, seq| Meas { suite, op_modes: TPU_OP_MODES, seqs: seq, attn: &[], step_modes: &[], observed_hz: None };
    let designs = [
        ("a100_sxm4_40gb.json5", Some(a100)),
        ("tpu_v5e.json5", Some(tpu("tpuv5e_2026-10-05_suite.json", &["tpuv5e_2026-10-05_seq_fused.json"]))),
        ("tpu_v6e.json5", Some(tpu("tpuv6e_2026-10-05_suite.json", &["tpuv6e_2026-10-05_seq_fused.json"]))),
        ("v100_sxm2_32gb.json5", None),
        ("h100_sxm5_80gb.json5", None),
        ("tpu_v4.json5", None),
    ];
    for (file, meas) in designs {
        if only.as_ref().is_some_and(|o| !file.contains(o.as_str())) {
            continue;
        }
        let p = Prepared::from_file(&root().join("kiln/designs/reference").join(file), Profile::Reference).expect("design loads");
        let cost: Option<std::sync::Arc<dyn kiln_map::UnitCostModel>> =
            if std::env::args().any(|a| a == "--roofline") { Some(std::sync::Arc::new(kiln_map::RooflineCost)) } else { None };
        let opts = SimOptions { interval: IntervalMethod::Corners, cost_model: cost, layer_scope_fallback: true, ..SimOptions::default() };
        let no_bf16 = p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None) == 0.0;
        println!("\n=== {file} (exec {:?}, cost model {})", p.view.hw.exec_model, opts.cost().name());
        let seq = meas.as_ref().map(steps).unwrap_or_default();
        println!(
            "{:<11} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9} {:>6} {:>7}  {:<22} binding",
            "phase", "step_ms", "low_ms", "high_ms", "A2_ms", "pred_ms", "meas_ms", "L", "p/m", "measured impl"
        );
        for member in kiln_wl::zoo::suite("standard").expect("suite") {
            let ph = member.scenario.to_string();
            let member = if no_bf16 { fp16(member) } else { member };
            let (run, _) = simulate_member(&p, &member, &opts).expect("simulates");
            let (meas_s, layers, imp) = seq.get(&ph).cloned().unwrap_or((f64::NAN, 32, String::new()));
            let pred = if run.central.scope == kiln_trace::sim::Scope::Step && layers == 32 {
                Ok(run.time.central)
            } else {
                predict_layers(&p, &ph, layers as u32, &opts)
            };
            let dom = run.central.bottleneck.dominant().map(|(c, x)| format!("{c:?} {:.0}%", 100.0 * x / run.central.makespan_s)).unwrap_or_default();
            let scope = if run.central.scope == kiln_trace::sim::Scope::Step { String::new() } else { " [32 layers do not fit: layer diagnostic]".into() };
            println!(
                "{ph:<11} {:>9.3} {:>9.3} {:>9.3} {:>8.3} {:>9} {:>9.3} {:>6} {:>7}  {:<22} {dom}{scope}",
                run.time.central * 1e3,
                run.time.high * 1e3,
                run.time.low * 1e3,
                run.central.t_a2_s * 1e3,
                pred.as_ref().map_or_else(|e| e.clone(), |t| format!("{:.3}", t * 1e3)),
                meas_s * 1e3,
                layers,
                pred.as_ref().map_or_else(|_| "-".into(), |t| format!("{:.3}", t / meas_s)),
                imp,
            );
            let fails: Vec<String> = run.central.invariants.failures().map(|c| format!("{:?}: {}", c.id, c.message)).collect();
            if !fails.is_empty() {
                println!("  INVARIANT FAILURES: {fails:?}");
            }
            if let Some(hz) = meas.as_ref().and_then(|m| m.observed_hz).filter(|_| ph.starts_with("prefill")) {
                let o = SimOptions { clock: at_clock(&p, hz), interval: IntervalMethod::None, shadow_prices: false, ..opts.clone() };
                let (r, _) = simulate_member(&p, &member, &o).expect("simulates");
                println!("  what-if at the observed {:.0} MHz: {:.3} ms ({:.3} of measured)", hz / 1e6, r.time.central * 1e3, r.time.central / meas_s);
            }
        }
        let Some(m) = meas else { continue };
        let records: Vec<OpRecord> = op_records(&read(m.suite), m.suite).unwrap_or_default();
        println!("\n{:<28} {:>8} {:>10} {:>10} {:>7} {:>9}  bound", "record (per-op, isolated)", "AI F/B", "meas_us", "pred_us", "p/m", "TF meas");
        let ridge = p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None) / p.view.hw.offchip_bandwidth(None).0;
        let (mut mem_r, mut cmp_r) = (vec![], vec![]);
        let mut whatif = vec![];
        for r in &records {
            let Some(meas) = r.best(m.op_modes) else { continue };
            let o = SimOptions { interval: IntervalMethod::None, shadow_prices: false, ..opts.clone() };
            let Ok(b) = simulate_bench_op(&p, &r.op, &o) else { continue };
            let flops = r.op.flops().unwrap_or(0) as f64;
            let ai = flops / r.op.min_bytes().unwrap_or(1).max(1) as f64;
            let ratio = b.central.makespan_s / meas;
            if ai < ridge { mem_r.push(ratio) } else { cmp_r.push(ratio) }
            if let (Some(hz), true) = (m.observed_hz, ai >= ridge) {
                let o = SimOptions { clock: at_clock(&p, hz), ..o.clone() };
                whatif.push(simulate_bench_op(&p, &r.op, &o).map_or(f64::NAN, |x| x.central.makespan_s / meas));
            }
            let bound = b.central.bottleneck.dominant().map(|(c, _)| format!("{c:?}")).unwrap_or_default();
            let tag = r.phase.as_deref().map_or_else(|| r.group.clone(), String::from);
            println!(
                "{:<28} {:>8.1} {:>10.2} {:>10.2} {:>7.3} {:>9.1}  {bound}",
                format!("{tag}/{}", r.name),
                ai,
                meas * 1e6,
                b.central.makespan_s * 1e6,
                ratio,
                flops / meas / 1e12
            );
        }
        println!(
            "per-op geomean pred/meas: memory-bound (AI < {ridge:.0}) {:.3} (n={}), compute-bound {:.3} (n={}), all {:.3}",
            geomean(&mem_r),
            mem_r.len(),
            geomean(&cmp_r),
            cmp_r.len(),
            geomean(&[mem_r.clone(), cmp_r.clone()].concat())
        );
        if let Some(hz) = m.observed_hz {
            println!("  what-if: compute-bound geomean at the observed {:.0} MHz: {:.3}", hz / 1e6, geomean(&whatif));
        }
    }
}

/// Volta has no bf16 MACs (pub:wp): the same model stored in fp16.
fn fp16(mut m: kiln_wl::zoo::SuiteMember) -> kiln_wl::zoo::SuiteMember {
    use kiln_ir::wl::{ElemType, ModelSrc};
    if let ModelSrc::Full(model) = &mut m.doc.model {
        let model = model.as_mut();
        let tensors = model.tensors.values_mut().chain(model.graphs.values_mut().flat_map(|g| g.tensors.values_mut()));
        for t in tensors.filter(|t| t.dtype == ElemType::BF16) {
            t.dtype = ElemType::from(kiln_ir::precision::Precision::Fp16);
        }
    }
    m
}
