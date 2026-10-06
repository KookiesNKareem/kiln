//! Per-segment breakdown of the middle window iteration of a whole step (diagnostic): which groups form each
//! barrier segment, its occupancy bound A1, critical path L, overheads and stack-kernel terms.
//! `cargo run --release -p kiln-sim --example step_breakdown -- <design> <phase> [layers] [calib]`
use kiln_ir::hw::Profile;
use kiln_sim::engine::Engine;
use kiln_sim::{Prepared, SimOptions};
use kiln_trace::Corner;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = Prepared::from_file(std::path::Path::new(&a[1]), Profile::Reference).unwrap_or_else(|e| panic!("{e:?}"));
    let phase = a.get(2).map_or("prefill_b1", String::as_str);
    let layers: u32 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(32);
    let calibration = a.get(4).map(|c| std::sync::Arc::new(kiln_sim::CalibSet::load(std::path::Path::new(c)).expect("calib")));
    let opts = SimOptions { calibration, trace: kiln_trace::TraceLevel::Ops, ..SimOptions::default() };
    let mut cfg = kiln_wl::zoo::preset("llama3_8b").unwrap();
    cfg.n_layers = layers;
    let model = kiln_wl::zoo::build_model(&cfg).unwrap();
    let sc = kiln_wl::zoo::scenario(phase).unwrap();
    let (_, lg, st) = kiln_wl::evaluate_snapshot(&model, &sc).unwrap();
    let r = st.resident;
    let prog = kiln_map::Program::whole_step(&model, &lg, opts.window).unwrap().with_resident(r.weights + r.kv_cache + r.constants);
    let run = kiln_sim::simulate(&p.view, &prog, kiln_ir::common::Id::new("x").unwrap(), kiln_trace::sim::Scope::Step, &opts, kiln_sim::evaluate::base_provenance("", "", &opts)).unwrap();
    println!("makespan central {:.3} ms  (low {:?} high {:?})", run.central.makespan_s * 1e3, run.low.as_ref().map(|x| x.makespan_s * 1e3), run.high.as_ref().map(|x| x.makespan_s * 1e3));
    for o in run.central.ops.iter().filter(|o| o.layer == Some(1)) {
        println!(
            "  op {:<36} {:>9.2} us  MAC use {:>5.3}  {:?}",
            o.op.as_str(),
            o.time_s() * 1e6,
            if o.macs_issued > 0 { o.macs_useful as f64 / o.macs_issued as f64 } else { f64::NAN },
            o.binding
        );
    }
    {
        let g = &run.graph;
        let filt = std::env::var("OPS").unwrap_or_else(|_| "attn.i1".into());
        for (gi, grp) in g.groups.iter().enumerate().filter(|(_, x)| x.label.contains(&filt)) {
            let mut per: std::collections::BTreeMap<(u32, String), f64> = Default::default();
            for t in &g.tasks[grp.tasks.0 as usize..grp.tasks.1 as usize] {
                for a in g.demands_of(t) {
                    let (r, x) = match *a {
                        kiln_map::lower::Amount::Res(r, x) => (r as usize, x),
                        kiln_map::lower::Amount::Prof(pi, x) => {
                            for &(r, f) in &g.profiles[pi as usize].entries {
                                let res = &p.view.resources[r as usize];
                                *per.entry((t.op, res.path.clone())).or_default() += x * f / res.capacity;
                            }
                            continue;
                        }
                    };
                    let res = &p.view.resources[r];
                    let s = if res.class == kiln_map::hwview::ResClass::Compute { x / run.clocks.hz(res.clock).unwrap_or(1e9) } else { x / res.capacity };
                    *per.entry((t.op, res.path.clone())).or_default() += s;
                }
            }
            println!("  group {gi} {}", grp.label);
            let mut v: Vec<_> = per.into_iter().collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            for ((op, r), s) in v.iter().take(14) {
                println!("    op {op} {r:<40} {:>9.2} us", s * 1e6);
            }
        }
    }
    for (c, name) in [(Corner::Central, "central"), (Corner::High, "high")] {
        let sp = run.params.at(c);
        println!("[{name}] eta_dram {:.3} t_sync {:.2e} t_program {:.2e} unit_eff {:?} ramp {:.2e} contention {}", sp.eta_dram, sp.t_sync, sp.t_program, sp.unit_eff, sp.t_dram_ramp, sp.contention_scale);
        let g = &run.graph;
        let out = Engine { view: &p.view, g, params: &sp, clocks: &run.clocks }.run(prog.window.map(|w| w.0 / 2));
        let mid = out.mid_iteration;
        let mut tot = [0.0f64; 8];
        for s in out.segs.iter().filter(|s| s.iteration == mid) {
            let labels: Vec<String> = g.groups[s.groups.0..s.groups.1].iter().map(|x| x.label.rsplit('/').next().unwrap_or("").to_string()).collect();
            let row = [s.a1_floor, s.a1_est, s.l_est, s.contention, s.overhead, s.stack_dram + s.stack_onchip, s.stack_overhead, s.time_est];
            for (t, x) in tot.iter_mut().zip(row) {
                *t += x;
            }
            if name == "central" {
                println!(
                    "  {:<40} A1f {:>8.2} A1e {:>8.2} L {:>8.2} C {:>6.2} ovh {:>6.2} stk {:>6.2}+{:>5.2} T {:>8.2} us  {:?}",
                    labels.join(","),
                    row[0] * 1e6, row[1] * 1e6, row[2] * 1e6, row[3] * 1e6, row[4] * 1e6, row[5] * 1e6, row[6] * 1e6, row[7] * 1e6,
                    s.binding
                );
            }
        }
        println!(
            "  mid-iteration totals: A1f {:.1} A1e {:.1} L {:.1} C {:.1} ovh {:.1} stk {:.1}+{:.1} T {:.1} us",
            tot[0] * 1e6, tot[1] * 1e6, tot[2] * 1e6, tot[3] * 1e6, tot[4] * 1e6, tot[5] * 1e6, tot[6] * 1e6, tot[7] * 1e6
        );
    }
}
