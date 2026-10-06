//! Golden results on the reference designs (01 §20): Llama-3-8B standard-suite contractions on one A100 SM
//! (4 tensor cores) and the TPU v5e TensorCore (4 MXUs). Exact regression values live in
//! `tests/data/golden_llama.json`; regenerate with `KILN_BLESS=1 cargo test -p kiln-cost --test golden`.
//! Physical sanity is asserted independently of the stored values.

use std::path::PathBuf;

use kiln_cost::*;
use kiln_ir::hw::{HwModel, Profile, check_file};
use kiln_ir::wl::KernelClass;
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model(name: &str) -> HwModel {
    check_file(root().join("../../designs/reference").join(name), Profile::Reference).model.expect("reference design")
}

fn unit_ix(hw: &HwModel, suffix: &str) -> usize {
    hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with(suffix)).expect("unit")
}

fn level_ix(t: &UnitTemplate, needle: &str) -> usize {
    t.levels.iter().position(|l| l.name.contains(needle)).expect("level")
}

struct Target {
    name: &'static str,
    unit: UnitTemplate,
    /// Level activations and outputs stay in between ops.
    act_level: usize,
    /// Per-dim tile caps for (A-only parallel, B-only parallel) dims; None = whole op on the unit.
    tile: Option<(u64, u64)>,
}

fn targets() -> Vec<Target> {
    let a100 = model("a100_sxm4_40gb.json5");
    let sm = UnitTemplate::from_hw(
        &a100,
        unit_ix(&a100, "gpc0_0.tpc0.sm0.smsp0.tc"),
        &TemplateOptions { gang: 4, bw_assumed: Some(1555e9 / 108.0), ..Default::default() },
    )
    .expect("sm");
    let tpu = model("tpu_v5e.json5");
    let tc = UnitTemplate::from_hw(&tpu, unit_ix(&tpu, "tc.mxu0"), &TemplateOptions { gang: 4, ..Default::default() }).expect("tc");
    vec![
        Target { name: "a100-sm", act_level: level_ix(&sm, "l2p"), unit: sm, tile: Some((256, 128)) },
        Target { name: "v5e-tc", act_level: level_ix(&tc, "vmem"), unit: tc, tile: None },
    ]
}

fn llama_nests() -> Vec<(String, OpNest)> {
    let mut out: Vec<(String, OpNest)> = vec![];
    for w in ["llama3_8b:decode_b8", "llama3_8b:prefill_b1"] {
        let m = kiln_wl::zoo::workload(w).expect("workload");
        let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
        for n in &lg.nodes {
            for (ki, k) in n.lowered.kernels.iter().enumerate() {
                // Weight GEMMs only; attention BMMs have no weight operand.
                if k.class != KernelClass::Contraction || !n.path.ends_with("proj") && !n.path.ends_with("qkv") && !n.path.ends_with("gate_up") && !n.path.ends_with("down") {
                    continue;
                }
                out.push((format!("{w}/{}", n.path), OpNest::from_lowered(n, ki).expect("nest")));
            }
        }
    }
    out
}

fn tiled(t: &Target, n: &OpNest) -> OpNest {
    let mut sizes: Vec<u64> = n.dims.iter().map(|d| d.size).collect();
    let uses = |role, d: usize| n.operands.iter().any(|o| o.role == role && o.axes.iter().any(|a| a.terms.iter().any(|&(x, _)| x == d)));
    if let Some((cap_m, cap_n)) = t.tile {
        for (d, s) in sizes.iter_mut().enumerate() {
            if n.dims[d].kind != LoopKind::Parallel {
                continue;
            }
            let (a, b) = (uses(kiln_ir::hw::compute::OperandRole::A, d), uses(kiln_ir::hw::compute::OperandRole::B, d));
            if a && !b {
                *s = (*s).min(cap_m);
            } else if b && !a {
                *s = (*s).min(cap_n);
            }
        }
    }
    let mut x = n.tile(&sizes, None);
    let room = t.unit.levels[t.act_level].capacity_bytes / 4;
    let bytes = |o: &NestOperand| o.axes.iter().map(|a| sizes[a.terms[0].0]).product::<u64>() * 2;
    let fits = x.operands.iter().filter(|o| o.role != kiln_ir::hw::compute::OperandRole::B).map(bytes).sum::<u64>() <= room;
    for o in &mut x.operands {
        if fits && o.role != kiln_ir::hw::compute::OperandRole::B {
            if o.is_output {
                o.sink = Some(t.act_level);
            } else {
                o.source = Some(t.act_level);
            }
        }
    }
    x
}

fn summarize(e: &CostEntry) -> Value {
    let mut levels: Vec<(usize, u64, u64)> = vec![];
    for a in &e.accesses {
        match levels.iter_mut().find(|x| x.0 == a.level) {
            Some(x) => {
                x.1 += a.read_bytes;
                x.2 += a.write_bytes;
            }
            None => levels.push((a.level, a.read_bytes, a.write_bytes)),
        }
    }
    levels.sort();
    json!({
        "cycles": e.cycles, "issue": e.issue_cycles, "stall": e.stall_cycles, "fill_drain": e.fill_drain_cycles,
        "useful_macs": e.useful_macs, "issued_macs": e.issued_macs, "mode": e.mode, "limiter": e.limiter,
        "latency_s": e.latency_s, "energy_j": e.energy.total_j, "level_rw_bytes": levels, "truncated": e.search.truncated,
    })
}

#[test]
fn golden_llama_gemms_on_reference_units() {
    let mut got = serde_json::Map::new();
    for t in targets() {
        let clock = t.unit.clock_hz;
        for (name, n) in llama_nests() {
            let nest = tiled(&t, &n);
            let q = CostQuery { unit: &t.unit, nest: &nest, objective: Objective::Latency, options: CostOptions::default() };
            let e = cost(&q).unwrap_or_else(|d| panic!("{} {name}: {d}", t.name));
            assert!(!e.search.truncated, "{} {name}", t.name);
            assert!(e.cycles >= e.floors.compute_cycles);
            for &bw in &e.floors.bandwidth_cycles {
                assert!(e.cycles as f64 >= bw.floor());
            }
            // Weights stream from HBM exactly once per tile: the HBM level's traffic floor binds decode.
            let hbm = level_ix(&t.unit, "hbm");
            let w_bytes = 2 * nest.dims.iter().filter(|d| d.kind == LoopKind::Reduction).map(|d| d.size).product::<u64>()
                * nest.dims.iter().enumerate().filter(|&(d, _)| nest.operands[1].axes.iter().any(|a| a.terms[0].0 == d) && nest.dims[d].kind == LoopKind::Parallel).map(|(_, x)| x.size).product::<u64>();
            let hbm_bw: f64 = t.unit.levels[hbm].ports.iter().map(|p| p.bytes_per_cycle).sum();
            let floor = w_bytes as f64 / hbm_bw;
            assert!(e.cycles as f64 >= floor, "{} {name}: {} < weight floor {floor}", t.name, e.cycles);
            if name.contains("decode") {
                assert!((e.cycles as f64) < 1.6 * floor, "{} {name}: decode {} not near weight-streaming floor {floor}", t.name, e.cycles);
            }
            assert!(e.latency_s > 0.0 && (e.latency_s * clock - e.cycles as f64).abs() <= 1.0);
            got.insert(format!("{}/{name}", t.name), summarize(&e));
        }
    }
    let path = root().join("tests/data/golden_llama.json");
    let text = serde_json::to_string_pretty(&Value::Object(got)).expect("json") + "\n";
    // Through text so both sides take the same float parse (lossy unless serde_json's `float_roundtrip` is on).
    let got: Value = serde_json::from_str(&text).expect("json");
    if std::env::var_os("KILN_BLESS").is_some() {
        std::fs::write(&path, text).expect("write golden");
        return;
    }
    let want: Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("golden file; run with KILN_BLESS=1")).expect("json");
    let (g, w) = (got.as_object().expect("obj"), want.as_object().expect("obj"));
    assert_eq!(g.keys().collect::<Vec<_>>(), w.keys().collect::<Vec<_>>());
    for (k, v) in g {
        assert_eq!(v, &w[k], "golden mismatch for {k}");
    }
}

/// One tensor core and one MXU reach their peak on a large GEMM tile: the A100 TC with an SM's L1 shared four
/// ways (pass-through levels inherit the buffering below them; buffering is enabled wherever tiles fit), and
/// the v5e MXU whose 4 read / 2 write vreg ports serve the input row and the partial sums in the same cycle.
#[test]
fn single_units_reach_peak_on_large_gemm_tiles() {
    let a100 = model("a100_sxm4_40gb.json5");
    let mut tc = UnitTemplate::from_hw(&a100, unit_ix(&a100, "gpc0_0.tpc0.sm0.smsp0.tc"), &TemplateOptions::default()).expect("tc");
    let l1 = level_ix(&tc, ".l1");
    tc.levels[l1].capacity_bytes /= 4;
    let tpu = model("tpu_v5e.json5");
    let mxu = UnitTemplate::from_hw(&tpu, unit_ix(&tpu, "tc.mxu0"), &TemplateOptions::default()).expect("mxu");
    let (_, qkv) = llama_nests().into_iter().find(|(n, _)| n.ends_with("prefill_b1/layers.qkv")).expect("qkv");
    for (unit, sizes) in [(&tc, [512, 80, 4096]), (&mxu, [512, 6144, 4096])] {
        let nest = qkv.tile(&sizes, None);
        let e = cost(&CostQuery { unit, nest: &nest, objective: Objective::Latency, options: CostOptions::default() }).expect("cost");
        assert!(e.utilization > 0.99, "{}: utilization {} (stall {})", unit.name, e.utilization, e.stall_cycles);
    }
}

/// `latency_floor` and `access_floors` bound what the search returns (the mapper prunes candidates with them).
#[test]
fn floors_bound_search_results() {
    let zz: Value = serde_json::from_str(&std::fs::read_to_string(root().join("tests/data/zigzag_corpus.json")).expect("corpus")).expect("json");
    let mut cases: Vec<(String, UnitTemplate, OpNest)> = vec![];
    for t in targets() {
        for (name, n) in llama_nests() {
            cases.push((format!("{}/{name}", t.name), t.unit.clone(), tiled(&t, &n)));
        }
    }
    for c in zz["cases"].as_array().expect("cases") {
        cases.push((c["name"].as_str().expect("name").into(), serde_json::from_value(c["unit"].clone()).expect("unit"), serde_json::from_value(c["nest"].clone()).expect("nest")));
    }
    for (name, unit, nest) in &cases {
        let quick = CostOptions { budget: SearchBudget { top_k_spatial: 2, max_evals_per_spatial: 300, stop_at_floor: true }, ..Default::default() };
        for options in [CostOptions::default(), quick] {
            let Ok(e) = cost(&CostQuery { unit, nest, objective: Objective::Latency, options }) else { continue };
            let floor = latency_floor(unit, nest).expect("floor");
            assert!(e.cycles >= floor, "{name}: {} cycles below floor {floor}", e.cycles);
            let acc = access_floors(unit, nest).expect("access floors");
            for (oi, levels) in acc.iter().enumerate() {
                for (l, &(r, w)) in levels.iter().enumerate() {
                    let got = e.accesses.iter().filter(|a| a.level == l && a.operand == oi).fold((0, 0), |s, a| (s.0 + a.read_bytes, s.1 + a.write_bytes));
                    assert!(got.0 >= r && got.1 >= w, "{name}: operand {oi} level {l}: {got:?} below {:?}", (r, w));
                }
            }
        }
    }
}
