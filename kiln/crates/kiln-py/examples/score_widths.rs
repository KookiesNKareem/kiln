//! Score intervals of the reference designs against the simulated A100 baseline under each generic set
//! (`corners`, standard suite): `(high - low) / central` per design and the median over designs that score.
//! `a100_novel` is the A100 design without its `family`, i.e. the same silicon treated as a novel design.
//!
//! cargo run --release -p kiln-py --example score_widths [-- generic-v1 generic-v2]

use kiln_py::inputs::{DesignInput, WorkloadInput, default_designs_dir};
use kiln_py::{Options, Session, SessionConfig};
use kiln_trace::result::Status;
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sets: Vec<String> = if args.is_empty() {
        vec!["generic-v1".into(), "generic-v2".into()]
    } else {
        args
    };
    let dir = default_designs_dir().join("reference");
    let a100 = std::fs::read_to_string(dir.join("a100_sxm4_40gb.json5")).expect("a100 design");
    let novel: String = a100
        .lines()
        .filter(|l| !l.trim_start().starts_with("family:"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut designs: Vec<(String, DesignInput)> = DESIGNS
        .iter()
        .map(|d| {
            let text = std::fs::read_to_string(dir.join(format!("{d}.json5"))).expect("design");
            (d.to_string(), DesignInput::Str(text))
        })
        .collect();
    designs.push(("a100_novel".into(), DesignInput::Str(novel)));
    let wl = WorkloadInput::Str("standard".into());
    for set in sets {
        let s = Session::new(SessionConfig {
            calibration: Some(set.clone()),
            no_cache: true,
            ..Default::default()
        })
        .expect("session");
        let o = Options::from_value(&json!({"profile": "full", "interval": "corners",
            "fitness": {"baseline": "a100_40gb"}}))
        .expect("options");
        println!(
            "== {set} ({}): score vs simulated a100_40gb, corners, standard suite",
            s.calibration().hash
        );
        println!(
            "  {:<16} {:>8} {:>8} {:>8} {:>9}  status",
            "design", "low", "central", "high", "rel_width"
        );
        let mut widths = vec![];
        for (name, d) in &designs {
            let r = s.evaluate(d, &wl, &o);
            match (r.status, r.score_interval) {
                (Status::Ok, Some(si)) => {
                    let w = (si.high - si.low) / si.central;
                    widths.push(w);
                    println!(
                        "  {name:<16} {:>8.3} {:>8.3} {:>8.3} {:>8.1}%  ok",
                        si.low,
                        si.central,
                        si.high,
                        100.0 * w
                    );
                }
                (st, _) => println!(
                    "  {name:<16} {:>8} {:>8} {:>8} {:>9}  {st:?}: {}",
                    "-",
                    "-",
                    "-",
                    "-",
                    r.errors.first().map(|e| e.diag.code.as_str()).unwrap_or("")
                ),
            }
        }
        widths.sort_by(f64::total_cmp);
        if !widths.is_empty() {
            let n = widths.len();
            let med = if n % 2 == 1 {
                widths[n / 2]
            } else {
                0.5 * (widths[n / 2 - 1] + widths[n / 2])
            };
            println!("  median rel width {:.1}% over {n} designs", 100.0 * med);
        }
    }
}
