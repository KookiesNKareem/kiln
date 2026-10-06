//! Single-GEMM diagnostic: achieved throughput, mapping split and per-resource demand of each group.
//! `cargo run --release -p kiln-sim --example gemm_probe -- <design> <m> [n] [k]`
use kiln_ir::bench::BenchOp;
use kiln_ir::hw::Profile;
use kiln_sim::{Prepared, SimOptions, simulate_bench_op};
use kiln_trace::IntervalMethod;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = Prepared::from_file(std::path::Path::new(&a[1]), Profile::Reference).unwrap_or_else(|e| panic!("{e:?}"));
    let m: u64 = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(8192);
    let n: u64 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(m);
    let k: u64 = a.get(4).and_then(|x| x.parse().ok()).unwrap_or(m);
    let opts = SimOptions { interval: IntervalMethod::None, shadow_prices: false, trace: kiln_trace::TraceLevel::Ops, ..SimOptions::default() };
    let run = simulate_bench_op(&p, &BenchOp::gemm(m, n, k, false), &opts).unwrap_or_else(|e| panic!("{}", e.message));
    let c = &run.central;
    let flops = 2.0 * (m * n * k) as f64;
    println!("gemm {m}x{n}x{k}: {:.4} ms  {:.1} TF  clocks {:?}", c.makespan_s * 1e3, flops / c.makespan_s / 1e12, c.clocks.iter().map(|x| x.hz / 1e6).collect::<Vec<_>>());
    for o in &c.ops {
        println!("  op {:<30} {:>9.2} us  issued {:.3e} useful {:.3e}  {:?}", o.op.as_str(), o.time_s() * 1e6, o.macs_issued as f64, o.macs_useful as f64, o.binding);
    }
    for (name, pl) in &run.mapping.ops {
        let parts: Vec<String> = pl.split.iter().map(|s| format!("{}:{:?}", s.dim, s.parts)).collect();
        println!("  place {name} {:?} split [{}] slices {}", pl.target, parts.join(" "), pl.slice_to_unit.len());
    }
    let g = &run.graph;
    for grp in &g.groups {
        let mut per: std::collections::BTreeMap<String, f64> = Default::default();
        let mut nt = 0usize;
        for t in &g.tasks[grp.tasks.0 as usize..grp.tasks.1 as usize] {
            nt += 1;
            for d in g.demands_of(t) {
                match *d {
                    kiln_map::lower::Amount::Res(r, x) => {
                        let res = &p.view.resources[r as usize];
                        let s = if res.class == kiln_map::hwview::ResClass::Compute { x / run.clocks.hz(res.clock).unwrap_or(1e9) } else { x / res.capacity };
                        *per.entry(res.path.clone()).or_default() += s;
                    }
                    kiln_map::lower::Amount::Prof(pi, x) => {
                        for &(r, f) in &g.profiles[pi as usize].entries {
                            let res = &p.view.resources[r as usize];
                            *per.entry(res.path.clone()).or_default() += x * f / res.capacity;
                        }
                    }
                }
            }
        }
        // Collapse per-instance resources by stripping indices: report max instance and sum.
        let mut agg: std::collections::BTreeMap<String, (f64, f64, usize)> = Default::default();
        for (path, s) in per {
            let key: String = path.chars().map(|ch| if ch.is_ascii_digit() { '#' } else { ch }).collect();
            let e = agg.entry(key).or_default();
            e.0 = e.0.max(s);
            e.1 += s;
            e.2 += 1;
        }
        let mut v: Vec<_> = agg.into_iter().collect();
        v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        println!("  group {} tasks {nt}", grp.label);
        for (r, (mx, sum, cnt)) in v.iter().take(16) {
            println!("    {r:<48} max {:>9.2} us  sum {:>11.2} us  n {cnt}", mx * 1e6, sum * 1e6);
        }
    }
}
