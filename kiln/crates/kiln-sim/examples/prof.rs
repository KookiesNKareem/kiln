//! Tier A wall time (06 §8): per design-layer (w = 1 window with prologue/epilogue) and design-step (w = 3),
//! `interval = none`, no shadow prices; best of `REPS` runs per phase (shared machines), then p50/p95 over the
//! standard suite's phases. Cold: a fresh kiln-cost cache per run. Warm: kiln-cost answers from a persistent
//! cache file written by an earlier run (`--warm <file>`; the file is filled first if it lacks entries).
//! `cargo run --release -p kiln-sim --example prof [-- [--warm <file>] <design.json5>...]`

use std::sync::Arc;
use std::time::Instant;

use kiln_ir::hw::Profile;
use kiln_map::cost::{KilnCost, UnitCostModel};
use kiln_sim::{Prepared, SimOptions};
use kiln_trace::IntervalMethod;

const DESIGNS: [&str; 6] = ["a100_sxm4_40gb.json5", "v100_sxm2_32gb.json5", "h100_sxm5_80gb.json5", "tpu_v4.json5", "tpu_v5e.json5", "tpu_v6e.json5"];
const REPS: usize = 3;
const BYTES: u64 = 256 << 20;

/// Volta has no bf16 MACs (pub:wp): run the same model stored in fp16.
pub fn fp16(mut m: kiln_wl::zoo::SuiteMember) -> kiln_wl::zoo::SuiteMember {
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

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * p).ceil() as usize).clamp(1, v.len()) - 1]
}

fn main() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference");
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let warm = args.iter().position(|a| a == "--warm").map(|i| {
        let f = args.remove(i + 1);
        args.remove(i);
        std::path::PathBuf::from(f)
    });
    let designs: Vec<String> = if args.is_empty() { DESIGNS.iter().map(|s| s.to_string()).collect() } else { args };
    let tag = if warm.is_some() { "warm" } else { "cold" };
    println!("{:<22} {:>10} {:>10} {:>10} {:>10}   ({tag} kiln-cost cache)", "design", "layer p50", "layer p95", "step p50", "step p95");
    for f in designs {
        let p = Prepared::from_file(&root.join(&f), Profile::Reference).unwrap();
        let no_bf16 = p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None) == 0.0;
        let members: Vec<_> = kiln_wl::zoo::suite("standard").unwrap().into_iter().map(|m| if no_bf16 { fp16(m) } else { m }).collect();
        let opts = |w: u32, cost: Option<Arc<dyn UnitCostModel>>| SimOptions {
            window: w,
            interval: IntervalMethod::None,
            shadow_prices: false,
            layer_scope_fallback: true,
            cost_model: cost,
            ..SimOptions::default()
        };
        let shared: Option<Arc<dyn UnitCostModel>> = warm.as_ref().map(|file| {
            let fill = Arc::new(KilnCost::with_cache_file(file, BYTES));
            for m in &members {
                for w in [1u32, 3] {
                    let _ = kiln_sim::simulate_member(&p, m, &opts(w, Some(fill.clone())));
                }
            }
            fill.save_cache().expect("save cost cache");
            Arc::new(KilnCost::with_cache_file(file, BYTES)) as Arc<dyn UnitCostModel>
        });
        let mut ms = [vec![], vec![]];
        for m in &members {
            for (i, w) in [1u32, 3].into_iter().enumerate() {
                let o = opts(w, shared.clone());
                let best = (0..REPS)
                    .map(|_| {
                        let t = Instant::now();
                        let _ = kiln_sim::simulate_member(&p, m, &o).unwrap();
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .fold(f64::INFINITY, f64::min);
                ms[i].push(best);
            }
        }
        let [l, s] = &mut ms;
        println!("{f:<22} {:>8.1}ms {:>8.1}ms {:>8.1}ms {:>8.1}ms", pct(l, 0.5), pct(l, 0.95), pct(s, 0.5), pct(s, 0.95));
    }
}
