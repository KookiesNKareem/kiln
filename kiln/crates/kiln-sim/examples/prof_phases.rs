//! Where Tier A time goes per design-layer (w = 1): mapping (kiln-cost misses and hits by quick / full search,
//! roofline-costed vector kernels, the rest is lowering) and simulation (engine runs, assembly, invariants).
//! Cold kiln-cost cache, or warm from a persistent cache file (`WARM=<file>`, as written by `prof --warm`).
//! `REPS=<n>` repeats each phase (sampling profilers), `MEMBER=<prefix>` selects phases.
//! `cargo run --release -p kiln-sim --example prof_phases [-- <design.json5>...]`

use std::sync::{Arc, Mutex};
use std::time::Instant;

use kiln_ir::hw::Profile;
use kiln_map::cost::{KilnCost, NestCost, NestQuery, UnitCostModel};
use kiln_map::program::Program;
use kiln_sim::{Prepared, SimOptions};
use kiln_trace::IntervalMethod;
use kiln_trace::sim::Scope;

#[derive(Default, Clone, Copy)]
struct Acc {
    n: u64,
    s: f64,
}

struct Timed {
    inner: Arc<KilnCost>,
    /// [quick miss, quick hit, full miss, full hit, roofline]
    acc: Mutex<[Acc; 5]>,
}

impl UnitCostModel for Timed {
    fn name(&self) -> &str {
        "kiln-cost"
    }

    fn cost_floor(&self, q: &NestQuery) -> Option<Result<NestCost, kiln_ir::common::Diagnostic>> {
        self.inner.cost_floor(q)
    }

    fn cost(&self, q: &NestQuery) -> Result<NestCost, kiln_ir::common::Diagnostic> {
        let m0 = self.inner.cache_stats().misses;
        let t = Instant::now();
        let r = self.inner.cost(q);
        let dt = t.elapsed().as_secs_f64();
        let miss = self.inner.cache_stats().misses > m0;
        let i = if q.op.class() != kiln_ir::wl::KernelClass::Contraction { 4 } else { usize::from(!q.quick) * 2 + usize::from(!miss) };
        let mut a = self.acc.lock().unwrap();
        a[i].n += 1;
        a[i].s += dt;
        r
    }
}

fn main() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let designs: Vec<String> = if args.is_empty() {
        ["a100_sxm4_40gb.json5", "v100_sxm2_32gb.json5", "h100_sxm5_80gb.json5", "tpu_v4.json5"].iter().map(|s| s.to_string()).collect()
    } else {
        args
    };
    for f in designs {
        let p = Prepared::from_file(&root.join(&f), Profile::Reference).unwrap();
        let no_bf16 = p.view.hw.peak_ops_for(kiln_ir::precision::Precision::Bf16, None) == 0.0;
        println!("== {f}");
        let reps: usize = std::env::var("REPS").ok().and_then(|x| x.parse().ok()).unwrap_or(1);
        let warm = std::env::var_os("WARM").map(|f| Arc::new(KilnCost::with_cache_file(f, 1 << 30)));
        let only = std::env::var("MEMBER").unwrap_or_default();
        for m in kiln_wl::zoo::suite("standard").unwrap().into_iter().filter(|m| m.scenario.to_string().starts_with(&only)).flat_map(|m| std::iter::repeat_n(m, reps)) {
            let m = if no_bf16 { prof_fp16(m) } else { m };
            let (model, sc) = (m.model(), m.scenario());
            let (_, lg, st) = kiln_wl::evaluate_snapshot(model, sc).unwrap();
            let prog = Program::whole_step(model, &lg, 1).unwrap();
            let r = st.resident;
            let prog = prog.with_resident(r.weights + r.kv_cache + r.constants);
            let inner = warm.clone().unwrap_or_else(|| Arc::new(KilnCost::new()));
            let timed = Arc::new(Timed { inner, acc: Mutex::default() });
            let opts = SimOptions {
                window: 1,
                interval: IntervalMethod::None,
                shadow_prices: false,
                cost_model: Some(timed.clone()),
                ..SimOptions::default()
            };
            let t0 = Instant::now();
            let (mapping, report, graph) = kiln_sim::run::map_program(&p.view, &prog, &opts, "w").unwrap();
            let t_map = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let prov = kiln_sim::evaluate::base_provenance("d", "w", &opts);
            let scope = if report.capacity_overflow.is_some() { Scope::Layer } else { Scope::Step };
            let ntasks = graph.tasks.len();
            let _ = kiln_sim::run::simulate_mapped(&p.view, &prog, kiln_ir::common::Id::new("x").unwrap(), scope, &opts, &mut prov.clone(), mapping, report, graph)
                .unwrap();
            let t_sim = t1.elapsed().as_secs_f64();
            let a = *timed.acc.lock().unwrap();
            let ms = |x: f64| x * 1e3;
            println!(
                "{:<14} total {:6.1}ms  map {:6.1} (cost: qmiss {:3}x {:5.1}  qhit {:4}x {:5.1}  fmiss {:3}x {:5.1}  fhit {:4}x {:5.1}  roof {:4}x {:4.1})  sim {:5.1}  tasks {}",
                m.scenario,
                ms(t_map + t_sim),
                ms(t_map),
                a[0].n,
                ms(a[0].s),
                a[1].n,
                ms(a[1].s),
                a[2].n,
                ms(a[2].s),
                a[3].n,
                ms(a[3].s),
                a[4].n,
                ms(a[4].s),
                ms(t_sim),
                ntasks
            );
        }
    }
}

fn prof_fp16(mut m: kiln_wl::zoo::SuiteMember) -> kiln_wl::zoo::SuiteMember {
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
