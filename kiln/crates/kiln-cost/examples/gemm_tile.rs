//! Costs one GEMM tile (one SM's slice) on a GPU SM gang and prints the chosen mapping and per-level traffic.
//! `cargo run --release -p kiln-cost --example gemm_tile -- <design> <unit suffix> <gang> <m> <n> <k> [bw_bytes_per_s]`
//! `DT=fp8` costs fp8 operands; `HAND="n32,m2,k2,k256,n3,m6;3,5,6;3,5,6;4,4,5,6"` (loops innermost first, then
//! per-stream allocs) also evaluates that temporal mapping under spatial m16 n8 k16 and gang over m.
use kiln_cost::*;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::hw::{Profile, check_file};
use kiln_ir::precision::{Precision, PrecisionSpec};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let hw = check_file(&a[1], Profile::Reference).model.expect("model");
    let unit = hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with(&a[2])).expect("unit");
    let gang: u32 = a[3].parse().unwrap();
    let (m, n, k): (u64, u64, u64) = (a[4].parse().unwrap(), a[5].parse().unwrap(), a[6].parse().unwrap());
    let bw = a.get(7).map(|x| x.parse::<f64>().unwrap());
    let t = UnitTemplate::from_hw(&hw, unit, &TemplateOptions { gang, bw_assumed: bw, ..Default::default() }).unwrap();
    for (i, l) in t.levels.iter().enumerate() {
        let p: Vec<String> = l.ports.iter().map(|p| format!("{:?} {:.0}B/c x{}", p.dir, p.bytes_per_cycle, p.lanes)).collect();
        println!("level {i} {} cap {} inst_axes {:?} ports [{}]", l.name, l.capacity_bytes, l.instance_axes, p.join(", "));
    }
    println!("axes {:?}", t.axes.iter().map(|x| (x.name.clone(), x.size)).collect::<Vec<_>>());
    println!("chains {:?}", t.chains.iter().map(|c| (c.role, c.levels.clone())).collect::<Vec<_>>());
    let fp8 = std::env::var("DT").is_ok_and(|d| d == "fp8");
    let ps = |p: Precision| PrecisionSpec::from(if fp8 && p == Precision::Bf16 { Precision::Fp8E4m3 } else { p });
    let ax = |d: usize| AxisExpr { terms: vec![(d, 1)], div: 1, offset: 0 };
    let op = |tn: &str, role, axes, dtype, is_output| NestOperand { tensor: tn.into(), role, axes, dtype, is_output, source: None, sink: None, block_axis: None };
    let nest = OpNest {
        kind: NestKind::Contraction,
        dims: vec![
            NestDim { name: "m".into(), size: m, kind: LoopKind::Parallel },
            NestDim { name: "n".into(), size: n, kind: LoopKind::Parallel },
            NestDim { name: "k".into(), size: k, kind: LoopKind::Reduction },
        ],
        operands: vec![
            op("x", OperandRole::A, vec![ax(0), ax(2)], ps(Precision::Bf16), false),
            op("w", OperandRole::B, vec![ax(2), ax(1)], ps(Precision::Bf16), false),
            op("y", OperandRole::O, vec![ax(0), ax(1)], ps(Precision::Bf16), true),
        ],
        macs_per_point: 1,
        vector_ops_per_point: 0,
        points: None,
        accum: None,
    };
    if let Ok(h) = std::env::var("HAND") {
        // HAND="n32,m2,k2,k256,n3,m6;3,6,6;3,6,6;4,4,6,6" spatial fixed to m16 n8 k16 gang-m4
        let parts: Vec<&str> = h.split(';').collect();
        let loops: Vec<TemporalLoop> = parts[0].split(',').map(|x| TemporalLoop { dim: "mnk".find(&x[..1]).unwrap(), factor: x[1..].parse().unwrap() }).collect();
        let alloc: Vec<Vec<usize>> = parts[1..].iter().map(|p| p.split(',').map(|x| x.parse().unwrap()).collect()).collect();
        let db: Vec<Vec<bool>> = alloc.iter().map(|a| a.iter().enumerate().map(|(j, _)| j + 1 < a.len() && j < 2).collect()).collect();
        let mp = Mapping { classes: vec![TileClass { sizes: vec![m, n, k], count: 1, spatial: SpatialMapping { axes: vec![vec![(0, 16)], vec![(1, 8)], vec![(2, 16)], vec![(if std::env::var("SP").is_ok_and(|x| x == "n") { 1 } else { 0 }, 4)]] }, temporal: TemporalMapping { loops, alloc, double_buffer: db } }] };
        match evaluate(&t, &nest, &mp, &CostOptions::default()) {
            Ok(e) => {
                println!("HAND cycles {} issue {} stall {} util {:.3} limiter {:?}", e.cycles, e.issue_cycles, e.stall_cycles, e.utilization, e.limiter);
                for x in &e.accesses {
                    println!("  L{} op {} rd/MAC {:.4} wr/MAC {:.4}", x.level, x.operand, x.read_bytes as f64 / (m * n * k) as f64, x.write_bytes as f64 / (m * n * k) as f64);
                }
            }
            Err(d) => println!("HAND err {d:?}"),
        }
    }
    let opts = CostOptions { budget: SearchBudget { top_k_spatial: 8, max_evals_per_spatial: 20_000, stop_at_floor: std::env::var_os("NOSTOP").is_none() }, ..Default::default() };
    let e = cost(&CostQuery { unit: &t, nest: &nest, objective: Objective::Latency, options: opts }).unwrap();
    let macs = (m * n * k) as f64;
    println!(
        "cycles {} issue {} stall {} fill {} util {:.3} limiter {:?} floor compute {} bw {:?}",
        e.cycles, e.issue_cycles, e.stall_cycles, e.fill_drain_cycles, e.utilization, e.limiter, e.floors.compute_cycles, e.floors.bandwidth_cycles
    );
    println!("search {:?}", e.search);
    for c in &e.mapping.classes {
        println!("class sizes {:?} x{} spatial {:?}", c.sizes, c.count, c.spatial.axes);
        println!("  loops {:?}", c.temporal.loops.iter().map(|l| (["m", "n", "k"][l.dim], l.factor)).collect::<Vec<_>>());
        println!("  alloc {:?} db {:?}", c.temporal.alloc, c.temporal.double_buffer);
    }
    for x in &e.accesses {
        println!(
            "  L{} {:<40} op {} rd {:>14} wr {:>14}  rd/MAC {:.4} wr/MAC {:.4}",
            x.level, t.levels[x.level].name.rsplit('.').next().unwrap_or(""), x.operand, x.read_bytes, x.write_bytes, x.read_bytes as f64 / macs, x.write_bytes as f64 / macs
        );
    }
}
