//! Property tests (06 §2.2) over random GEMMs and small units: physical floors, compulsory traffic,
//! monotonicity at fixed mapping, renaming invariance, determinism, energy conservation.

use kiln_cost::*;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{Precision, PrecisionSpec};
use proptest::prelude::*;

const PJ: f64 = 1e-12;

fn ps(p: Precision) -> PrecisionSpec {
    PrecisionSpec::new(p)
}

fn unit(rows: u32, cols: u32, buf: u64, rf: u64, dram_bw: f64) -> UnitTemplate {
    let level = |name: &str, cap: u64, ports: Vec<MemPort>, e: f64, inst: Vec<usize>| MemLevel {
        name: name.into(),
        mem: None,
        capacity_bytes: cap,
        ports,
        double_buffer: true,
        instance_axes: inst,
        e_read_j_per_b: e,
        e_write_j_per_b: 1.1 * e,
        latency_cycles: 1,
        external: false,
        partitions: vec![],
    };
    let p = |dir, b| MemPort { dir, bytes_per_cycle: b, serves: vec![], lanes: 1 };
    UnitTemplate {
        name: "prop".into(),
        clock_hz: 1e9,
        axes: vec![
            SpatialAxis { name: "rows".into(), size: rows, allowed: vec!["k".into()] },
            SpatialAxis { name: "cols".into(), size: cols, allowed: vec!["n".into()] },
        ],
        modes: vec![MacMode {
            a: ps(Precision::Bf16),
            b: ps(Precision::Bf16),
            acc: ps(Precision::Fp32),
            out: None,
            macs_per_cycle: f64::from(rows * cols),
            e_mac_j: PJ,
            mx_native: false,
            k_pack: 1,
        }],
        levels: vec![
            // Per-column weight registers, a shared buffer, DRAM.
            level("rf", rf * u64::from(cols), vec![p(PortDir::ReadWrite, 8.0)], 0.05 * PJ, vec![1]),
            level("buf", buf, vec![p(PortDir::Read, 32.0), p(PortDir::Write, 16.0)], 0.5 * PJ, vec![]),
            MemLevel { external: true, ..level("dram", 1 << 36, vec![p(PortDir::ReadWrite, dram_bw)], 20.0 * PJ, vec![]) },
        ],
        chains: vec![
            OperandChain { role: OperandRole::A, levels: vec![1, 2] },
            OperandChain { role: OperandRole::B, levels: vec![0, 1, 2] },
            OperandChain { role: OperandRole::O, levels: vec![1, 2] },
        ],
        pipeline: PipelineCycles { fill: u64::from(rows + cols - 1), drain: 0, issue_overhead: 0.0 },
        psum_precision: None,
        fused_down_conversion: false,
        e_mac_idle_ratio: 0.1,
        e_vector_op_j: 0.5 * PJ,
        energy_source: EnergySource::Supplied,
        operand_run: vec![],
    }
}

fn gemm(m: u64, n: u64, k: u64) -> OpNest {
    let ax = |d: usize| AxisExpr { terms: vec![(d, 1)], div: 1, offset: 0 };
    let op = |t: &str, role, axes: Vec<AxisExpr>, is_output| NestOperand {
        tensor: t.into(),
        role,
        axes,
        dtype: ps(Precision::Bf16),
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
            op("x", OperandRole::A, vec![ax(0), ax(2)], false),
            op("w", OperandRole::B, vec![ax(2), ax(1)], false),
            op("y", OperandRole::O, vec![ax(0), ax(1)], true),
        ],
        macs_per_point: 1,
        vector_ops_per_point: 0,
        points: None,
        accum: None,
    }
}

prop_compose! {
    fn case()(m in 1u64..160, n in 1u64..160, k in 1u64..200, rows in 0u32..5, cols in 0u32..5,
              buf in 10u32..20, rf in 4u32..10, bw in 1u32..64)
        -> (OpNest, UnitTemplate) {
        (gemm(m, n, k), unit(1 << rows, 1 << cols, 1 << buf, 1 << rf, f64::from(bw)))
    }
}

fn run(u: &UnitTemplate, n: &OpNest) -> CostEntry {
    cost(&CostQuery { unit: u, nest: n, objective: Objective::Latency, options: CostOptions::default() }).expect("feasible")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(debug_assertions) { 24 } else { 128 }, ..ProptestConfig::default() })]

    #[test]
    fn floors_and_compulsory_traffic((n, u) in case()) {
        let e = run(&u, &n);
        let points = n.box_points();
        prop_assert_eq!(e.useful_macs, points);
        prop_assert!(e.issued_macs >= e.useful_macs);
        prop_assert!(e.cycles >= e.floors.compute_cycles);
        for (l, &bw) in e.floors.bandwidth_cycles.iter().enumerate() {
            prop_assert!(e.cycles as f64 >= bw.floor(), "level {} floor {} > {}", l, bw, e.cycles);
        }
        prop_assert!(e.utilization <= 1.0 + 1e-12);
        // Every operand crosses each of its levels at least once per distinct element.
        let size = |s: usize| match s { 0 => n.dims[0].size * n.dims[2].size, 1 => n.dims[2].size * n.dims[1].size, _ => n.dims[0].size * n.dims[1].size };
        for s in 0..3 {
            for &l in &u.chains[s].levels {
                let a = e.accesses.iter().find(|a| a.level == l && a.operand == s).expect("row");
                let moved = if s == 2 { a.from_low } else { a.to_low };
                prop_assert!(moved >= size(s), "stream {} level {}: {} < {}", s, l, moved, size(s));
            }
        }
        for (l, &c) in e.floors.compulsory_bytes.iter().enumerate() {
            let moved: u64 = e.accesses.iter().filter(|a| a.level == l).map(|a| a.read_bytes + a.write_bytes).sum();
            prop_assert!(moved >= c, "level {}: {} < {}", l, moved, c);
        }
        let en = &e.energy;
        let parts = en.mac_j + en.idle_mac_j + en.vector_j + en.conversion_j + en.levels_j.iter().sum::<f64>();
        prop_assert!((parts - en.total_j).abs() <= 1e-9 * en.total_j);
        prop_assert!(en.total_j >= e.useful_macs as f64 * PJ);
    }

    #[test]
    fn monotone_at_fixed_mapping((n, u) in case()) {
        let e = run(&u, &n);
        let opts = CostOptions::default();
        // More PEs (the mapping still fits) and more DRAM bandwidth never make a fixed mapping slower.
        let mut bigger = u.clone();
        bigger.axes[1].size *= 2;
        bigger.modes[0].macs_per_cycle *= 2.0;
        bigger.levels[0].capacity_bytes *= 2;
        bigger.pipeline.fill = u.pipeline.fill;
        prop_assert!(evaluate(&bigger, &n, &e.mapping, &opts).unwrap().cycles <= e.cycles);
        let mut faster = u.clone();
        faster.levels[2].ports[0].bytes_per_cycle *= 2.0;
        prop_assert!(evaluate(&faster, &n, &e.mapping, &opts).unwrap().cycles <= e.cycles);
        prop_assert_eq!(evaluate(&u, &n, &e.mapping, &opts).unwrap().cycles, e.cycles);
    }

    #[test]
    fn renaming_invariance_and_determinism((n, u) in case()) {
        let e = run(&u, &n);
        let mut r = n.clone();
        r.dims[0].name = "rows_of_a".into();
        r.operands.iter_mut().for_each(|o| o.tensor = format!("{}_renamed", o.tensor));
        // Names a template axis restricts on keep their meaning through the contraction class (k, n).
        r.dims[2].name = "reduce".into();
        r.dims[1].name = "cols_of_b".into();
        prop_assert_eq!(&run(&u, &r), &e);
        prop_assert_eq!(&run(&u, &n), &e);
    }
}
