//! Hand-worked mappings with exact counts, cycles and energy; quantization edges; precision and MX handling;
//! structured errors; search behavior (03 §2, 06 §2.1 kiln-cost obligations).

use kiln_cost::*;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{Precision, PrecisionSpec};

const PJ: f64 = 1e-12;

fn ps(p: Precision) -> PrecisionSpec {
    PrecisionSpec::new(p)
}

fn port(dir: PortDir, bytes_per_cycle: f64) -> MemPort {
    MemPort { dir, bytes_per_cycle, serves: vec![], lanes: 1 }
}

fn level(name: &str, capacity: u64, ports: Vec<MemPort>, e_r: f64, e_w: f64) -> MemLevel {
    MemLevel {
        name: name.into(),
        mem: None,
        capacity_bytes: capacity,
        ports,
        double_buffer: true,
        instance_axes: vec![],
        e_read_j_per_b: e_r,
        e_write_j_per_b: e_w,
        latency_cycles: 0,
        external: false,
    }
}

/// 2x2 array (rows bind k, cols bind n), one buffer (4 B/cycle read and write) over DRAM (2 B/cycle).
fn unit(rows: u32, cols: u32) -> UnitTemplate {
    let mode = |a: Precision, acc: Precision, rate: f64, e: f64| MacMode {
        a: ps(a),
        b: ps(a),
        acc: ps(acc),
        out: None,
        macs_per_cycle: f64::from(rows * cols) * rate,
        e_mac_j: e,
        mx_native: false,
        k_pack: 1,
    };
    UnitTemplate {
        name: "toy".into(),
        clock_hz: 1e9,
        axes: vec![
            SpatialAxis { name: "rows".into(), size: rows, allowed: vec!["k".into()] },
            SpatialAxis { name: "cols".into(), size: cols, allowed: vec!["n".into()] },
        ],
        modes: vec![mode(Precision::Bf16, Precision::Fp32, 1.0, PJ), mode(Precision::Int8, Precision::Int32, 2.0, 0.25 * PJ)],
        levels: vec![
            level("buf", 1 << 20, vec![port(PortDir::Read, 4.0), port(PortDir::Write, 4.0)], 0.1 * PJ, 0.2 * PJ),
            MemLevel { external: true, ..level("dram", 1 << 34, vec![port(PortDir::ReadWrite, 2.0)], 10.0 * PJ, 11.0 * PJ) },
        ],
        chains: [OperandRole::A, OperandRole::B, OperandRole::O]
            .into_iter()
            .map(|role| OperandChain { role, levels: vec![0, 1] })
            .collect(),
        pipeline: PipelineCycles::default(),
        psum_precision: None,
        fused_down_conversion: true,
        e_mac_idle_ratio: 0.0,
        e_vector_op_j: 0.0,
        energy_source: EnergySource::Supplied,
        operand_run: vec![],
    }
}

fn axis(d: usize) -> AxisExpr {
    AxisExpr { terms: vec![(d, 1)], div: 1, offset: 0 }
}

/// `O[m, n] += A[m, k] * B[k, n]` with dims (m, n, k).
fn gemm(m: u64, n: u64, k: u64, a: Precision, out: Precision) -> OpNest {
    let op = |t: &str, role, axes: Vec<AxisExpr>, dtype, is_output| NestOperand {
        tensor: t.into(),
        role,
        axes,
        dtype: ps(dtype),
        is_output,
        source: None,
        sink: None,
        block_axis: None,
    };
    OpNest {
        kind: NestKind::Contraction,
        dims: vec![
            NestDim { name: "m".into(), size: m, kind: LoopKind::Parallel },
            NestDim { name: "n".into(), size: n, kind: LoopKind::Parallel },
            NestDim { name: "k".into(), size: k, kind: LoopKind::Reduction },
        ],
        operands: vec![
            op("x", OperandRole::A, vec![axis(0), axis(2)], a, false),
            op("w", OperandRole::B, vec![axis(2), axis(1)], a, false),
            op("y", OperandRole::O, vec![axis(0), axis(1)], out, true),
        ],
        macs_per_point: 1,
        vector_ops_per_point: 0,
        points: None,
        accum: None,
    }
}

fn tl(dim: usize, factor: u64) -> TemporalLoop {
    TemporalLoop { dim, factor }
}

/// Spatial k:2 on rows, n:2 on cols; loops (innermost first) m4, k4, n2; the buffer holds loop m4 for every
/// operand.
fn worked_mapping(db: bool) -> Mapping {
    Mapping {
        classes: vec![TileClass {
            sizes: vec![4, 4, 8],
            count: 1,
            spatial: SpatialMapping { axes: vec![vec![(2, 2)], vec![(1, 2)]] },
            temporal: TemporalMapping {
                loops: vec![tl(0, 4), tl(2, 4), tl(1, 2)],
                alloc: vec![vec![1, 3]; 3],
                double_buffer: vec![vec![db, false]; 3],
            },
        }],
    }
}

fn access(e: &CostEntry, level: usize, operand: usize) -> &LevelAccess {
    e.accesses.iter().find(|a| a.level == level && a.operand == operand).expect("access row")
}

#[test]
fn worked_example_counts_bytes_cycles_energy() {
    let u = unit(2, 2);
    let n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    let e = evaluate(&u, &n, &worked_mapping(false), &CostOptions::default()).unwrap();
    // A refetched once per n2 iteration (irrelevant loop above the buffer): 2 x 32 elements.
    let a = access(&e, 0, 0);
    assert_eq!((a.to_low, a.from_high, a.to_high, a.from_low), (64, 64, 0, 0));
    assert_eq!(access(&e, 1, 0).to_low, 64);
    // B stays on the array inputs across m4 (stationary), fetched exactly once.
    let b = access(&e, 0, 1);
    assert_eq!((b.to_low, b.from_high), (32, 32));
    assert_eq!(access(&e, 1, 1).to_low, 32);
    // O: k4 merges down into the buffer (output tile stationary across it); 48 partial-sum readbacks at fp32,
    // the 16 final writes at bf16.
    let o = access(&e, 0, 2);
    assert_eq!((o.to_low, o.from_high, o.to_high, o.from_low), (48, 0, 16, 64));
    assert_eq!((o.read_bytes, o.write_bytes), (48 * 4 + 16 * 2, 48 * 4 + 16 * 2));
    assert_eq!(access(&e, 1, 2).from_low, 16);
    assert_eq!(e.useful_macs, 128);
    assert_eq!(e.issue_cycles, 32);
    // DRAM port: unbuffered A 8x8 + B 4x8 + O 8x2 = 112 stall; startup 8 and drain 8.
    assert_eq!(e.stall_cycles, 112);
    assert_eq!(e.fill_drain_cycles, 16);
    assert_eq!(e.cycles, 160);
    assert_eq!(e.limiter, Limiter::Port { level: 1, port: 0 });
    let want = (128.0 + 0.1 * 416.0 + 0.2 * 416.0 + 10.0 * 192.0 + 11.0 * 32.0) * PJ;
    assert!((e.energy.total_j - want).abs() < 1e-18, "{} vs {want}", e.energy.total_j);
    assert_eq!(e.energy_source, EnergySource::Supplied);
}

#[test]
fn double_buffering_overlaps_refills() {
    let u = unit(2, 2);
    let n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    let single = evaluate(&u, &n, &worked_mapping(false), &CostOptions::default()).unwrap();
    let double = evaluate(&u, &n, &worked_mapping(true), &CostOptions::default()).unwrap();
    // Buffer read port: buffered 120 busy, 32 rigid excess, over 32 compute cycles -> 88.
    assert_eq!(double.stall_cycles, 88);
    assert_eq!(double.cycles, 136);
    assert_eq!(single.accesses, double.accesses);
}

#[test]
fn compulsory_and_floors() {
    let u = unit(2, 2);
    let n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    let f = floors(&u, &n).unwrap();
    assert_eq!(f.compute_cycles, 32);
    // A 64 B + B 64 B + O 32 B cross both levels once.
    assert_eq!(f.compulsory_bytes, vec![160, 160]);
    assert_eq!(f.bandwidth_cycles[1], 80.0);
}

#[test]
fn reduction_above_output_level_moves_partial_sums_at_accumulator_width() {
    let u = unit(2, 2);
    let n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    // k4 outermost and above every level: the output tile goes to DRAM and back as fp32 partial sums.
    let m = Mapping {
        classes: vec![TileClass {
            sizes: vec![4, 4, 8],
            count: 1,
            spatial: SpatialMapping { axes: vec![vec![(2, 2)], vec![(1, 2)]] },
            temporal: TemporalMapping {
                loops: vec![tl(0, 2), tl(1, 2), tl(2, 4), tl(0, 2)],
                alloc: vec![vec![1, 4]; 3],
                double_buffer: vec![vec![false, false]; 3],
            },
        }],
    };
    let e = evaluate(&u, &n, &m, &CostOptions::default()).unwrap();
    let dram_o = access(&e, 1, 2);
    assert_eq!((dram_o.from_low, dram_o.to_low), (64, 48));
    // 48 fp32 partial sums, then the 16 results at their bf16 output precision (03 §2.4).
    assert_eq!((dram_o.write_bytes, dram_o.read_bytes), (48 * 4 + 16 * 2, 48 * 4));
    let buf_o = access(&e, 0, 2);
    assert_eq!((buf_o.to_high, buf_o.from_high), (64, 48));
    assert_eq!(buf_o.dir_bytes[2], 48 * 4 + 16 * 2);
}

#[test]
fn array_shape_quantization_and_ragged_classes() {
    let u = unit(4, 4);
    // GEMV: m = 1, k = 3 < rows, n = 7 prime against 4 columns.
    let n = gemm(1, 7, 3, Precision::Bf16, Precision::Bf16);
    let q = CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() };
    let e = cost(&q).unwrap();
    assert_eq!(e.useful_macs, 21);
    // Every temporal iteration issues the whole 4x4 array: ceil(3/3) * ceil(7/4) = 2 iterations.
    assert_eq!(e.issue_cycles, 2);
    assert_eq!(e.issued_macs, 32);
    assert!((e.spatial_util - 21.0 / 32.0).abs() < 1e-12);
    // n = 7 splits into a full class (4) and a remainder class (3), each with its own temporal mapping.
    let mut sizes: Vec<Vec<u64>> = e.mapping.classes.iter().map(|c| c.sizes.clone()).collect();
    sizes.sort();
    assert_eq!(sizes, vec![vec![1, 3, 3], vec![1, 4, 3]]);
    // Padding instead moves bytes for lanes that do no work.
    let pad = cost(&CostQuery { options: CostOptions { ragged: RaggedPolicy::Pad, ..Default::default() }, ..q }).unwrap();
    let bytes = |e: &CostEntry| e.accesses.iter().filter(|a| a.level == 1).map(|a| a.read_bytes + a.write_bytes).sum::<u64>();
    assert!(bytes(&e) < bytes(&pad), "{} vs {}", bytes(&e), bytes(&pad));
    assert_eq!(pad.issue_cycles, e.issue_cycles);
}

#[test]
fn precision_mode_sets_throughput() {
    let u = unit(4, 4);
    let bf = cost(&CostQuery {
        unit: &u,
        nest: &gemm(64, 64, 64, Precision::Bf16, Precision::Bf16),
        objective: Objective::Latency,
        options: CostOptions::default(),
    })
    .unwrap();
    let i8 = cost(&CostQuery {
        unit: &u,
        nest: &gemm(64, 64, 64, Precision::Int8, Precision::Int32),
        objective: Objective::Latency,
        options: CostOptions::default(),
    })
    .unwrap();
    assert_eq!((bf.mode, i8.mode), (0, 1));
    assert_eq!(bf.floors.compute_cycles, 64 * 64 * 64 / 16);
    assert_eq!(i8.floors.compute_cycles, 64 * 64 * 64 / 32);
    assert!(i8.issue_cycles * 2 <= bf.issue_cycles + 1);
}

#[test]
fn required_accumulator_precision_constrains_the_mode() {
    let mut u = unit(2, 2);
    let mut fast = u.modes[0].clone();
    fast.acc = ps(Precision::Bf16);
    fast.macs_per_cycle *= 2.0;
    u.modes = vec![u.modes[0].clone(), fast];
    let mut n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    let q = |u: &UnitTemplate, n: &OpNest| cost(&CostQuery { unit: u, nest: n, objective: Objective::Latency, options: CostOptions::default() });
    assert_eq!(q(&u, &n).unwrap().mode, 1, "no requirement: the faster mode");
    n.accum = Some(Precision::Fp32);
    assert_eq!(q(&u, &n).unwrap().mode, 0, "fp32 accumulation needs the fp32 mode");
    let m = Mapping {
        classes: vec![TileClass {
            sizes: vec![4, 4, 8],
            count: 1,
            spatial: SpatialMapping { axes: vec![vec![(2, 2)], vec![(1, 2)]] },
            temporal: TemporalMapping {
                loops: vec![tl(0, 2), tl(1, 2), tl(2, 4), tl(0, 2)],
                alloc: vec![vec![1, 4]; 3],
                double_buffer: vec![vec![false, false]; 3],
            },
        }],
    };
    let e = evaluate(&u, &n, &m, &CostOptions::default()).unwrap();
    assert_eq!(access(&e, 1, 2).read_bytes, 48 * 4, "partial sums at the required 32 bits");
    u.modes.remove(0);
    let d = q(&u, &n).unwrap_err();
    assert_eq!(d.code, "E-COST-UNMAPPABLE");
}

#[test]
fn affine_footprints_count_distinct_coordinates() {
    let u = unit(2, 2);
    let mut n = gemm(1, 1, 4, Precision::Bf16, Precision::Bf16);
    // A[m, 2k] over k = 0..4 touches 4 elements, not the 7 of its bounding span.
    n.operands[0].axes[1] = AxisExpr { terms: vec![(2, 2)], ..axis(2) };
    let dram = |n: &OpNest| access_floors(&u, n).unwrap()[0][1].0;
    assert_eq!(dram(&n), 4 * 2);
    // A[floor((k + 31) / 32)] over k = 0..2 touches A[0] and A[1].
    let mut n = gemm(1, 1, 2, Precision::Bf16, Precision::Bf16);
    n.operands[0].axes[1] = AxisExpr { terms: vec![(2, 1)], div: 32, offset: 31 };
    assert_eq!(dram(&n), 2 * 2);
}

#[test]
fn unmappable_precision_is_a_structured_error() {
    let u = unit(2, 2);
    let n = gemm(4, 4, 4, Precision::Fp8E4m3, Precision::Bf16);
    let q = CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() };
    let d = cost(&q).unwrap_err();
    assert_eq!(d.code, "E-COST-UNMAPPABLE");
    assert!(d.hint.is_some());
}

#[test]
fn infeasible_tile_names_level_capacity_and_hint() {
    let mut u = unit(4, 4);
    u.levels[0].capacity_bytes = 16;
    let n = gemm(8, 8, 8, Precision::Bf16, Precision::Bf16);
    let d = cost(&CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() }).unwrap_err();
    assert_eq!(d.code, "E-COST-INFEASIBLE");
    assert!(d.message.contains("buf"), "{}", d.message);
    assert!(d.hint.is_some());
}

fn mx_unit(native: bool) -> UnitTemplate {
    let mut u = unit(4, 4);
    let p = if native { Precision::Mxfp8E4m3 } else { Precision::Fp8E4m3 };
    u.modes = vec![MacMode {
        a: ps(p),
        b: ps(p),
        acc: ps(Precision::Fp32),
        out: None,
        macs_per_cycle: 16.0,
        e_mac_j: PJ,
        mx_native: native,
        k_pack: 1,
    }];
    u.e_vector_op_j = 0.5 * PJ;
    u
}

#[test]
fn mx_scale_streams_and_emulation_pass() {
    let n = gemm(32, 32, 128, Precision::Mxfp8E4m3, Precision::Bf16);
    let q = |u| CostQuery { unit: u, nest: &n, objective: Objective::Latency, options: CostOptions::default() };
    let un = mx_unit(true);
    let native = cost(&q(&un)).unwrap();
    assert_eq!(native.vector_ops, 0);
    // Streams: x, w, y, then the scale streams of x and w (E8M0, one per 32 elements along k).
    let dram = |e: &CostEntry, s: usize| access(e, 1, s).to_low;
    assert_eq!(dram(&native, 3) * 32, dram(&native, 0));
    assert_eq!(dram(&native, 4) * 32, dram(&native, 1));
    let ue = mx_unit(false);
    let emulated = cost(&q(&ue)).unwrap();
    // Non-native: one scale multiply-add per (MAC block of 32) per MX operand.
    assert_eq!(emulated.vector_ops, 2 * 32 * 32 * 128 / 32);
    assert!(emulated.energy.vector_j > 0.0);
}

#[test]
fn mx_emulation_counts_partial_k_blocks() {
    let ue = mx_unit(false);
    for (m, n, k, ops) in [(1, 1, 1, 2), (2, 3, 33, 2 * 2 * 3 * 2)] {
        let nest = gemm(m, n, k, Precision::Mxfp8E4m3, Precision::Fp32);
        let e = cost(&CostQuery { unit: &ue, nest: &nest, objective: Objective::Latency, options: CostOptions::default() }).unwrap();
        assert_eq!(e.vector_ops, ops, "{m}x{n}x{k}: M*N*ceil(K/32) per emulated MX operand");
    }
}

#[test]
fn search_reports_truncation() {
    let u = unit(4, 4);
    let n = gemm(96, 96, 96, Precision::Bf16, Precision::Bf16);
    let budget = SearchBudget { top_k_spatial: 2, max_evals_per_spatial: 1, stop_at_floor: false };
    let e = cost(&CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions { budget, ..Default::default() } })
        .unwrap();
    assert!(e.search.truncated);
    let full = cost(&CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() }).unwrap();
    assert!(!full.search.truncated);
    assert!(full.cycles <= e.cycles);
}

#[test]
fn searched_optimum_beats_greedy_mapping_and_is_reproducible() {
    // 144 B buffer, even share 48 B per stream. Greedy double-buffered fill of the prime-factor order
    // m2 m2 k2 k2 n2 keeps 2, 3 and 1 loops of A, B and O in the buffer; the search must do at least as well.
    let mut u = unit(2, 2);
    u.levels[0].capacity_bytes = 144;
    let n = gemm(4, 4, 8, Precision::Bf16, Precision::Bf16);
    let greedy = Mapping {
        classes: vec![TileClass {
            temporal: TemporalMapping {
                loops: vec![tl(0, 2), tl(0, 2), tl(2, 2), tl(2, 2), tl(1, 2)],
                alloc: vec![vec![2, 5], vec![3, 5], vec![1, 5]],
                double_buffer: vec![vec![true, false]; 3],
            },
            ..worked_mapping(true).classes[0].clone()
        }],
    };
    let g = evaluate(&u, &n, &greedy, &CostOptions::default()).unwrap();
    let q = CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() };
    let e = cost(&q).unwrap();
    assert!(e.cycles <= g.cycles, "{} vs {}", e.cycles, g.cycles);
    assert_eq!(evaluate(&u, &n, &e.mapping, &CostOptions::default()).unwrap().cycles, e.cycles);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (u, n) = (u.clone(), n.clone());
            std::thread::spawn(move || cost(&CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() }).unwrap())
        })
        .collect();
    for t in threads {
        assert_eq!(t.join().unwrap(), e);
    }
    let en = cost(&CostQuery { objective: Objective::Energy, ..q }).unwrap();
    assert!(en.energy.total_j <= e.energy.total_j);
}

#[test]
fn cache_hits_on_renamed_nests() {
    let u = unit(4, 4);
    let n = gemm(16, 32, 64, Precision::Bf16, Precision::Bf16);
    let mut renamed = n.clone();
    renamed.dims[0].name = "tokens".into();
    renamed.operands[0].tensor = "act".into();
    let cache = CostCache::new();
    let q = |nest| CostQuery { unit: &u, nest, objective: Objective::Latency, options: CostOptions::default() };
    let a = cache.query(&q(&n)).unwrap();
    let b = cache.query(&q(&renamed)).unwrap();
    assert_eq!(a, b);
    assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1, truncated: 0 });
    assert_eq!(cache.len(), 1);
    assert_eq!(CostKey::new(&q(&n)), CostKey::new(&q(&renamed)));
    let other = gemm(16, 32, 65, Precision::Bf16, Precision::Bf16);
    assert_ne!(CostKey::new(&q(&n)), CostKey::new(&q(&other)));
}

/// A file-backed cache answers a fresh process bit-identically without searching, ignores files of another
/// model version, merges with entries saved meanwhile and keeps the most recently used within its size bound.
#[test]
fn persistent_cache_round_trips() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("costs.jsonl");
    let unit = unit(4, 4);
    let g = |m, n, k| gemm(m, n, k, Precision::Bf16, Precision::Bf16);
    let nests: Vec<OpNest> = [(64, 128, 256), (8, 512, 4096), (33, 70, 96)].iter().map(|&(m, n, k)| g(m, n, k)).collect();
    fn query<'a>(unit: &'a UnitTemplate, nest: &'a OpNest) -> CostQuery<'a> {
        CostQuery { unit, nest, objective: Objective::Latency, options: CostOptions::default() }
    }
    let q = |n| query(&unit, n);
    let h = unit.hash();
    let a = CostCache::open(&path, CostCache::DEFAULT_FILE_BYTES);
    let fresh: Vec<CostEntry> = nests.iter().map(|n| (*a.query_hashed(&q(n), &h).expect("cost")).clone()).collect();
    a.save().expect("save");
    let b = CostCache::open(&path, CostCache::DEFAULT_FILE_BYTES);
    for (n, f) in nests.iter().zip(&fresh) {
        assert_eq!(*b.query_hashed(&q(n), &h).expect("cost"), *f, "bit-identical after a round trip");
    }
    assert_eq!(b.stats().misses, 0);
    // Another process saved a different entry meanwhile: both survive the next save.
    let extra = g(16, 16, 16);
    let c = CostCache::open(&path, CostCache::DEFAULT_FILE_BYTES);
    c.query_hashed(&q(&extra), &h).expect("cost");
    c.save().expect("save");
    b.save().expect("save");
    assert_eq!(CostCache::open(&path, CostCache::DEFAULT_FILE_BYTES).len(), 4);
    // Size bound: only the most recently used entry fits.
    let one = std::fs::read_to_string(&path).expect("file").lines().nth(1).expect("entry").len() as u64;
    let d = CostCache::open(&path, 1 << 30);
    d.query_hashed(&q(&nests[1]), &h).expect("cost");
    let small = dir.path().join("small.jsonl");
    std::fs::copy(&path, &small).expect("copy");
    let e = CostCache::open(&small, 200 + one * 2);
    e.query_hashed(&q(&nests[1]), &h).expect("cost");
    e.save().expect("save");
    let kept = CostCache::open(&small, 1 << 30);
    assert!(!kept.is_empty() && kept.len() < 4, "{}", kept.len());
    assert_eq!(kept.stats().misses, 0);
    kept.query_hashed(&q(&nests[1]), &h).expect("cost");
    assert_eq!(kept.stats().misses, 0, "the most recently used entry is kept");
    // A file of another model version is ignored.
    let text = std::fs::read_to_string(&path).expect("file").replacen(MODEL_HASH, "0000000000000000", 1);
    std::fs::write(&path, text).expect("write");
    assert!(CostCache::open(&path, CostCache::DEFAULT_FILE_BYTES).is_empty());
}

#[test]
fn lowered_nest_keeps_mx_scaling() {
    let m = kiln_wl::zoo::workload("llama3_8b:decode_b8").expect("workload");
    let (_, lg, _) = kiln_wl::evaluate_snapshot(m.model(), m.scenario()).expect("lower");
    let mut node = lg.nodes.iter().find(|n| n.path.ends_with("down")).expect("down proj").clone();
    for t in node.inputs.iter_mut() {
        t.dtype = kiln_ir::wl::ElemType::from(Precision::Mxfp8E4m3);
    }
    let nest = OpNest::from_lowered(&node, 0).expect("nest");
    let inputs: Vec<_> = nest.operands.iter().filter(|o| !o.is_output).map(|o| o.dtype.precision).collect();
    assert_eq!(inputs, [Precision::Mxfp8E4m3; 2]);
    let tile = nest.tile(&nest.dims.iter().map(|d| d.size.min(64)).collect::<Vec<_>>(), None);
    let ue = mx_unit(true);
    let e = cost(&CostQuery { unit: &ue, nest: &tile, objective: Objective::Latency, options: CostOptions::default() });
    assert!(e.is_ok(), "a native MXFP8 mode maps the lowered MX operands: {:?}", e.err());
}

#[test]
fn ragged_reductions_finalize_each_output_once() {
    let mut u = unit(4, 4);
    u.fused_down_conversion = false;
    u.e_vector_op_j = PJ;
    // k = 5 on four rows: both reduction classes accumulate into the one output element.
    let n = gemm(1, 1, 5, Precision::Bf16, Precision::Bf16);
    let e = cost(&CostQuery { unit: &u, nest: &n, objective: Objective::Latency, options: CostOptions::default() }).unwrap();
    assert_eq!(e.useful_macs, 5);
    assert_eq!(e.conversion_ops, 1, "one fp32 -> bf16 conversion of the single output");
    assert_eq!(access(&e, 1, 2).write_bytes, 2, "the single bf16 result written once");
}
