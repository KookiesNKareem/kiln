//! Ideal vs realistic scores of the reference designs against the simulated A100 baseline (08 §F software stack
//! in scoring), `corners`, standard suite: `score` runs both sides under `kiln_ideal`, `score_realistic` runs
//! each under its own execution model's default stack. Reference designs are compared `baseline_relative` (each
//! at its own envelope): most have more off-chip bandwidth than the A100-40GB and fail its matched envelope
//! (E-ENV-0007) by design.
//!
//! [KILN_ONLY=a,b] cargo run --release -p kiln-py --example stack_scores [-- <calibration set>...]

use kiln_py::inputs::{DesignInput, WorkloadInput};
use kiln_py::{Options, Session, SessionConfig};
use kiln_trace::Interval;
use serde_json::json;

const DESIGNS: &[&str] = &[
    "a100_sxm4_40gb",
    "h100_sxm5_80gb",
    "h100_pcie_80gb",
    "v100_sxm2_32gb",
    "tpu_v4",
    "tpu_v5e",
    "tpu_v5e_2x2",
    "tpu_v6e",
    "ember",
];

fn cell(i: Option<Interval>) -> String {
    i.map_or_else(
        || format!("{:>23}", "-"),
        |i| format!("{:>7.3} {:>7.3} {:>7.3}", i.low, i.central, i.high),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sets: Vec<Option<String>> = if args.is_empty() {
        vec![None]
    } else {
        args.into_iter().map(Some).collect()
    };
    let wl = WorkloadInput::Str("standard".into());
    let o = Options::from_value(&json!({"profile": "full", "interval": "corners",
        "fitness": {"kind": "baseline_relative", "baseline": "a100_40gb"}, "timeout_s": {"A": 120}}))
    .expect("options");
    for set in sets {
        let s = Session::new(SessionConfig {
            calibration: set,
            no_cache: true,
            ..Default::default()
        })
        .expect("session");
        let c = s.calibration();
        println!(
            "== {} ({}): vs simulated a100_40gb, corners, standard suite",
            c.id, c.hash
        );
        println!(
            "  {:<16} {:>23}  {:>23}  {:<24} status",
            "design", "ideal low/central/high", "realistic low/cen/high", "realistic stacks"
        );
        let only = std::env::var("KILN_ONLY").ok();
        for d in DESIGNS.iter().filter(|d| {
            only.as_deref()
                .is_none_or(|o| o.split(',').any(|x| x == **d))
        }) {
            let r = s.evaluate(&DesignInput::Str((*d).into()), &wl, &o);
            let stacks = r.score_realistic.as_ref().map_or_else(
                || "-".into(),
                |rs| {
                    let id = |l: &str| l.split('@').next().unwrap_or(l).to_string();
                    format!("{} / {}", id(&rs.candidate_stack), id(&rs.baseline_stack))
                },
            );
            let why = r
                .errors
                .iter()
                .chain(&r.violations)
                .chain(
                    r.warnings
                        .iter()
                        .filter(|w| w.diag.code.starts_with("W-FIT")),
                )
                .map(|e| e.diag.code.clone())
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "  {d:<16} {}  {}  {stacks:<24} {:?} {why}",
                cell(r.score_interval),
                cell(r.score_realistic.as_ref().map(|x| x.interval)),
                r.status
            );
            for w in r
                .warnings
                .iter()
                .filter(|w| w.diag.code.starts_with("W-FIT"))
            {
                println!("    {}", w.diag.message);
            }
        }
    }
}
