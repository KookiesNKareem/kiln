//! Regressions from the cost/map review (r6).

use kiln_cost::*;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::precision::{Precision, PrecisionSpec};

fn level(name: &str, ports: Vec<MemPort>, external: bool) -> MemLevel {
    MemLevel {
        name: name.into(),
        mem: None,
        capacity_bytes: 1 << 20,
        ports,
        double_buffer: false,
        instance_axes: vec![],
        e_read_j_per_b: 0.0,
        e_write_j_per_b: 0.0,
        latency_cycles: 0,
        external,
        partitions: vec![],
    }
}

/// A 1-MAC fp32 unit; the bottom level gives each role its own port (O's array side: 0.5 B/cycle read-write).
fn unit() -> UnitTemplate {
    let ps = PrecisionSpec::new(Precision::Fp32);
    let port = |role, dirs: &[Direction], dir, bpc| MemPort { dir, bytes_per_cycle: bpc, serves: dirs.iter().map(|&d| (role, d)).collect(), lanes: 1 };
    let rd = [Direction::ToLow, Direction::FromHigh, Direction::ToHigh, Direction::FromLow];
    UnitTemplate {
        name: "psum".into(),
        clock_hz: 1e9,
        axes: vec![],
        modes: vec![MacMode { a: ps, b: ps, acc: ps, out: None, macs_per_cycle: 1.0, e_mac_j: 0.0, mx_native: false, k_pack: 1 }],
        levels: vec![
            level(
                "buf",
                vec![
                    port(OperandRole::A, &rd, PortDir::ReadWrite, 1024.0),
                    port(OperandRole::B, &rd, PortDir::ReadWrite, 1024.0),
                    port(OperandRole::O, &[Direction::ToLow, Direction::FromLow], PortDir::ReadWrite, 0.5),
                    port(OperandRole::O, &[Direction::ToHigh, Direction::FromHigh], PortDir::ReadWrite, 1024.0),
                ],
                false,
            ),
            level("dram", vec![MemPort { dir: PortDir::ReadWrite, bytes_per_cycle: 1024.0, serves: vec![], lanes: 1 }], true),
        ],
        chains: [OperandRole::A, OperandRole::B, OperandRole::O].into_iter().map(|role| OperandChain { role, levels: vec![0, 1] }).collect(),
        pipeline: PipelineCycles::default(),
        psum_precision: None,
        fused_down_conversion: true,
        e_mac_idle_ratio: 0.0,
        e_vector_op_j: 0.0,
        energy_source: EnergySource::Supplied,
        operand_run: vec![],
    }
}

/// `O[m] += A[m, k] * B[k]`, fp32, dims (m, k).
fn gemv(m: u64, k: u64) -> OpNest {
    let ax = |d: usize| AxisExpr { terms: vec![(d, 1)], div: 1, offset: 0 };
    let op = |t: &str, role, axes: Vec<AxisExpr>, is_output| NestOperand {
        tensor: t.into(),
        role,
        axes,
        dtype: PrecisionSpec::new(Precision::Fp32),
        is_output,
        source: None,
        sink: None,
        block_axis: None,
    };
    OpNest {
        kind: NestKind::Contraction,
        dims: vec![NestDim { name: "m".into(), size: m, kind: LoopKind::Parallel }, NestDim { name: "k".into(), size: k, kind: LoopKind::Reduction }],
        operands: vec![op("a", OperandRole::A, vec![ax(0), ax(1)], false), op("b", OperandRole::B, vec![ax(1)], false), op("o", OperandRole::O, vec![ax(0)], true)],
        macs_per_point: 1,
        vector_ops_per_point: 0,
        points: None,
        accum: Some(Precision::Fp32),
    }
}

#[test]
fn partial_sum_latency_charges_only_real_readbacks() {
    let mapping = Mapping {
        classes: vec![TileClass {
            sizes: vec![2, 2],
            count: 1,
            spatial: SpatialMapping { axes: vec![] },
            temporal: TemporalMapping {
                loops: vec![TemporalLoop { dim: 0, factor: 2 }, TemporalLoop { dim: 1, factor: 2 }],
                alloc: vec![vec![0, 2]; 3],
                double_buffer: vec![vec![false, false]; 3],
            },
        }],
    };
    let e = evaluate(&unit(), &gemv(2, 2), &mapping, &CostOptions::default()).unwrap();
    let o = e.accesses.iter().find(|a| a.level == 0 && a.operand == 2).unwrap();
    assert_eq!((o.to_low, o.from_low), (2, 4));
    // Two 4 B readbacks and four 4 B writes on O's 0.5 B/cycle port: 48 busy cycles (not 64), overlapping compute.
    assert_eq!(e.stall_cycles + e.issue_cycles, 48, "{e:#?}");
}

#[test]
fn fractional_issue_overhead_is_charged_per_tile() {
    let mut u = unit();
    u.pipeline = PipelineCycles { fill: 0, drain: 0, issue_overhead: 0.49 };
    let mapping = Mapping {
        classes: vec![TileClass {
            sizes: vec![100, 1],
            count: 1,
            spatial: SpatialMapping { axes: vec![] },
            temporal: TemporalMapping {
                loops: vec![TemporalLoop { dim: 0, factor: 100 }],
                alloc: vec![vec![0, 1]; 3],
                double_buffer: vec![vec![false, false]; 3],
            },
        }],
    };
    let e = evaluate(&u, &gemv(100, 1), &mapping, &CostOptions::default()).unwrap();
    let free = evaluate(&unit(), &gemv(100, 1), &mapping, &CostOptions::default()).unwrap();
    assert_eq!(e.fill_drain_cycles - free.fill_drain_cycles, 49, "100 output tiles x 0.49 cycles");
}

#[test]
fn template_keeps_fractional_pipeline_cycles() {
    use kiln_ir::hw::{Design, ExpandOptions, MemLoader, Profile, check};
    let doc = serde_json::json!({
        "schema": "kiln.hw/1.0", "name": "pipe", "tech": "tsmc_n5",
        "clocks": [ { "id": "u", "freq": 1e9 } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "u",
                "clusters": [ { "id": "tile",
                    "units": [ { "id": "op", "kind": "matrix", "geometry": { "outer_product": { "rows": 8, "cols": 8 } },
                                 "precisions": ["bf16*bf16+fp32"], "feeds": { "a": "feed", "b": "feed", "o": "feed" },
                                 "pipeline": { "fill": 2.2, "drain": 0.3, "issue_overhead": 0.49 } } ],
                    "memories": [ { "id": "feed", "kind": "scratchpad", "capacity": 1024, "ports": [ { "dir": "rw", "width_bits": 256 } ] } ] } ],
                "networks": [ { "id": "noc", "topology": "crossbar", "endpoints": ["tile.feed"], "link": "128b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "2Gbps", "attach": "die.noc" } ] } }
    });
    let r = check(Design::from_source(&MemLoader::default(), None, &doc.to_string()).expect("parses"), Profile::Full, &ExpandOptions::default());
    let hw = r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics));
    let unit = hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with("op")).unwrap();
    let t = UnitTemplate::from_hw(&hw, unit, &TemplateOptions::default()).unwrap();
    assert_eq!(t.pipeline, PipelineCycles { fill: 3, drain: 1, issue_overhead: 0.49 });
}

/// `O[m] += A[m, floor((k + p) / 32)] * B[k]` with `k = 32`.
fn param_kernel() -> kiln_ir::wl::Kernel {
    use kiln_ir::wl::*;
    let t = |coeff, dim: Option<&str>, param: Option<&str>| Term { coeff, dim: dim.map(Into::into), param: param.map(Into::into) };
    let idx = IndexExpr::FloorDiv { inner: Box::new(IndexExpr::Affine { terms: vec![t(1, Some("k"), None), t(1, None, Some("p"))], offset: 0 }), by: 32 };
    Kernel {
        id: "pk".into(),
        dims: vec![LoopDim { name: "m".into(), extent: 1, kind: DimKind::Parallel }, LoopDim { name: "k".into(), extent: 32, kind: DimKind::Reduction }],
        domain: Domain::Box,
        operands: vec![
            Operand::new(&kiln_ir::common::Id::new("a").unwrap(), Access::Read, vec![IndexExpr::dim("m"), idx]),
            Operand::new(&kiln_ir::common::Id::new("b").unwrap(), Access::Read, vec![IndexExpr::dim("k")]),
            Operand::new(&kiln_ir::common::Id::new("o").unwrap(), Access::ReadWrite, vec![IndexExpr::dim("m")]),
        ],
        body: ScalarBody { mac: 1, ..Default::default() },
        combine: None,
        accum: Some(Precision::Fp32),
        class: KernelClass::Contraction,
        opaque_cost: None,
    }
}

#[test]
fn unbound_index_params_are_rejected_not_dropped() {
    let ps = [PrecisionSpec::new(Precision::Fp32); 3];
    let e = OpNest::from_kernel(&param_kernel(), &ps, Some(&[OperandRole::A, OperandRole::B, OperandRole::O])).expect_err("p is unbound");
    assert_eq!(e.code, "E-COST-NEST", "{e:?}");
}
