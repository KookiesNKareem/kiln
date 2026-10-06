//! Costs every Llama-3-8B standard-suite contraction on the A100 SM (4 tensor cores) and the TPU v5e
//! TensorCore (4 MXUs), cold cache, and prints latency, limiter and query time.
//! `cargo run --release -p kiln-cost --example llama_gemms`

use std::path::PathBuf;
use std::time::Instant;

use kiln_cost::*;
use kiln_ir::hw::{Profile, check_file};
use kiln_ir::wl::KernelClass;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference");
    let a100 = check_file(root.join("a100_sxm4_40gb.json5"), Profile::Reference).model.expect("a100");
    let tpu = check_file(root.join("tpu_v5e.json5"), Profile::Reference).model.expect("tpu");
    let find = |hw: &kiln_ir::hw::HwModel, s: &str| hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with(s)).expect("unit");
    let sm = UnitTemplate::from_hw(
        &a100,
        find(&a100, "gpc0_0.tpc0.sm0.smsp0.tc"),
        &TemplateOptions { gang: 4, bw_assumed: Some(1555e9 / 108.0), ..Default::default() },
    )
    .expect("sm");
    let tc = UnitTemplate::from_hw(&tpu, find(&tpu, "tc.mxu0"), &TemplateOptions { gang: 4, ..Default::default() }).expect("tc");
    let cache = CostCache::new();
    let mut cold = vec![];
    let mut warm = vec![];
    for m in kiln_wl::zoo::suite("standard").expect("suite") {
        let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
        let mut seen: Vec<Vec<u64>> = vec![];
        for n in &lg.nodes {
            for (ki, k) in n.lowered.kernels.iter().enumerate() {
                if k.class != KernelClass::Contraction {
                    continue;
                }
                let nest = OpNest::from_lowered(n, ki).expect("nest");
                let sizes: Vec<u64> = nest.dims.iter().map(|d| d.size).collect();
                if seen.contains(&sizes) {
                    continue;
                }
                seen.push(sizes.clone());
                for (name, unit, slices) in [("a100-sm", &sm, 108u64), ("v5e-tc", &tc, 1)] {
                    // One SM's share: the largest parallel dim split evenly over the units.
                    let mut tile = sizes.clone();
                    let d = (0..tile.len()).filter(|&d| nest.dims[d].kind == LoopKind::Parallel).max_by_key(|&d| tile[d]).expect("par");
                    tile[d] = tile[d].div_ceil(slices);
                    let t = nest.tile(&tile, None);
                    let q = CostQuery { unit, nest: &t, objective: Objective::Latency, options: CostOptions::default() };
                    let t0 = Instant::now();
                    let r = cost(&q);
                    let dt = t0.elapsed();
                    cold.push(dt.as_secs_f64() * 1e3);
                    let h = unit.hash();
                    let _ = cache.query_hashed(&q, &h);
                    let t1 = Instant::now();
                    for _ in 0..100 {
                        let _ = cache.query_hashed(&q, &h);
                    }
                    warm.push(t1.elapsed().as_secs_f64() * 1e6 / 100.0);
                    match r {
                        Ok(e) => println!(
                            "{:<22} {:<18} {name:<8} tile {:?}: {:>10.3} us util {:.3} {:?} evals {} pruned {} trunc {} [{:.2} ms]",
                            m.name,
                            n.path,
                            tile,
                            e.latency_s * 1e6,
                            e.utilization,
                            e.limiter,
                            e.search.evaluated,
                            e.search.pruned,
                            e.search.truncated,
                            dt.as_secs_f64() * 1e3
                        ),
                        Err(e) => println!("{:<22} {:<18} {name:<8} tile {tile:?}: ERROR {e}", m.name, n.path),
                    }
                }
            }
        }
    }
    cold.sort_by(f64::total_cmp);
    warm.sort_by(f64::total_cmp);
    let pct = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p) as usize];
    println!(
        "cold query: p50 {:.2} ms p90 {:.2} ms max {:.2} ms; warm (cache hit): p50 {:.1} us max {:.1} us; {} queries",
        pct(&cold, 0.5),
        pct(&cold, 0.9),
        pct(&cold, 1.0),
        pct(&warm, 0.5),
        pct(&warm, 1.0),
        cold.len()
    );
}
