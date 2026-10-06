//! Power and clock under the cap from kiln-phys (04 §8, 03 §4.5) on reference designs: GEMM, copy and Llama steps,
//! at the solved clock and at fixed clocks. `cargo run -p kiln-sim --release --example phys_power [design] [set]`.

use std::sync::Arc;

use kiln_ir::bench::{BenchKind, BenchOp};
use kiln_phys::{ClockMode, ClockPlan};
use kiln_sim::result::phase_energy;
use kiln_sim::run::PhaseRun;
use kiln_sim::{CalibSet, Prepared, SimOptions, simulate_bench_op, simulate_member};
use kiln_trace::IntervalMethod;

fn line(p: &Prepared, name: &str, run: &PhaseRun) {
    let c = &run.central;
    let plan = ClockPlan { hz: c.clocks.iter().map(|x| x.hz).collect(), solved: true, throttled: false };
    let pe = phase_energy(&p.view, &c.energy, &c.resources, c.makespan_s, &plan);
    let b = p.view.phys.phase_power(&pe, &plan).unwrap_or_default();
    let f = c.clocks.iter().map(|x| x.hz).fold(0.0, f64::max);
    println!(
        "{name:<28} {:>7.1} MHz  board {:>6.1} W (dyn {:>5.1} indep {:>5.1} static {:>5.1} clk {:>5.1} dram {:>5.1} vr {:>5.1} board {:>4.1})  pkg {:>6.1} chip {:>6.1}  Tj {:>4.0} C  act {:.2}  t {:.3e} s",
        f / 1e6, b.board_w, b.dyn_core_w, b.indep_w, b.static_w, b.clock_w, b.dram_w, b.vr_loss_w, b.board_fixed_w, b.package_w, b.chip_w, b.t_j_c, pe.activity, c.makespan_s
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let design = args.first().cloned().unwrap_or_else(|| "a100_sxm4_40gb".into());
    let set = args.get(1).cloned().unwrap_or_else(|| "generic-v2".into());
    let path = format!("{}/../../designs/reference/{design}.json5", env!("CARGO_MANIFEST_DIR"));
    let p = Prepared::from_file(std::path::Path::new(&path), kiln_ir::hw::Profile::Reference).expect("design");
    let cal = if set == "none" { None } else { Some(Arc::new(CalibSet::by_id(&set).expect("set"))) };
    let base = SimOptions { interval: IntervalMethod::None, shadow_prices: false, calibration: cal, layer_scope_fallback: true, ..SimOptions::default() };
    let capped = SimOptions { clock: ClockMode::PowerCapped, ..base.clone() };
    let m = p.view.phys.m3().expect("m3");
    let idle = m.power.power(&kiln_phys::PhaseEnergy { makespan_s: 1.0, ..Default::default() }, &p.view.phys.clock_plan(&ClockMode::Nominal).hz);
    println!("{design} [{set}] idle at nominal: board {:.1} W (static {:.1} clk {:.1} dram bg {:.1} board {:.1} vr {:.1}) cap {:?}", idle.board_w, idle.static_w, idle.clock_w, idle.dram_w, idle.board_fixed_w, idle.vr_loss_w, p.view.phys.power_cap_w());
    let ops = [("gemm 8192^3", BenchOp::gemm(8192, 8192, 8192, false)), ("gemm 16384^3", BenchOp::gemm(16384, 16384, 16384, false)), ("copy 1 GiB", BenchOp::stream(BenchKind::Copy, 1 << 30, &[("x", "uint8")]))];
    for (n, op) in &ops {
        for (tag, o) in [("cap", &capped), ("nominal", &base)] {
            match simulate_bench_op(&p, op, o) {
                Ok(r) => line(&p, &format!("{n} [{tag}]"), &r),
                Err(e) => println!("{n}: {}", e.message),
            }
        }
    }
    for w in ["llama3_8b:decode_b1", "llama3_8b:decode_b8", "llama3_8b:decode_b32", "llama3_8b:prefill_b1"] {
        let mem = kiln_wl::zoo::workload(w).unwrap();
        for (tag, o) in [("cap", &capped), ("nominal", &base)] {
            match simulate_member(&p, &mem, o) {
                Ok((r, _)) => line(&p, &format!("{w} [{tag}]"), &r),
                Err(e) => println!("{w}: {:?}", e.iter().map(|d| &d.message).collect::<Vec<_>>()),
            }
        }
    }
}
