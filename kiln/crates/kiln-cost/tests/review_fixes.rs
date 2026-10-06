//! Regressions from the cost/map review (r5): templates keep what the hardware declares.

use kiln_cost::*;
use kiln_ir::hw::compute::OperandRole;
use kiln_ir::hw::{Design, ExpandOptions, HwModel, MemLoader, Profile, check};
use kiln_ir::precision::{Precision, PrecisionSpec};
use serde_json::{Value, json};

/// One 8x8 outer-product unit fed from `feed` (bf16 x bf16 + fp32), backed by one HBM stack; `unit_clk` and
/// `mem_clk` in Hz.
fn design(feed: Value, unit_clk: f64, mem_clk: f64) -> HwModel {
    let doc = json!({
        "schema": "kiln.hw/1.0", "name": "review", "tech": "tsmc_n5",
        "clocks": [ { "id": "u", "freq": unit_clk }, { "id": "m", "freq": mem_clk } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "u",
                "clusters": [ { "id": "tile",
                    "units": [ { "id": "op", "kind": "matrix", "geometry": { "outer_product": { "rows": 8, "cols": 8 } },
                                 "precisions": ["bf16*bf16+fp32"], "feeds": { "a": "feed", "b": "feed", "o": "feed" } } ],
                    "memories": [ feed ] } ],
                "networks": [ { "id": "noc", "topology": "crossbar", "endpoints": ["tile.feed"], "link": "128b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "2Gbps", "attach": "die.noc" } ] } }
    });
    let r = check(Design::from_source(&MemLoader::default(), None, &doc.to_string()).expect("design parses"), Profile::Full, &ExpandOptions::default());
    r.model.unwrap_or_else(|| panic!("design expands: {:?}", r.diagnostics))
}

fn feed(extra: Value) -> Value {
    let mut m = json!({ "id": "feed", "kind": "scratchpad", "capacity": 1024, "ports": [ { "dir": "rw", "width_bits": 256 } ] });
    m.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    m
}

fn template(hw: &HwModel, home: bool) -> UnitTemplate {
    let unit = hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with("op")).expect("unit");
    let feed = hw.memories.iter().position(|m| hw.nodes[m.node].path.ends_with("feed")).expect("feed");
    UnitTemplate::from_hw(hw, unit, &TemplateOptions { home: home.then_some(feed), ..Default::default() }).expect("template")
}

/// `O[m, n] += A[m, k] * B[k, n]`, bf16 in and out, dims (m, n, k).
fn gemm(m: u64, n: u64, k: u64) -> OpNest {
    let ax = |d: usize| AxisExpr { terms: vec![(d, 1)], div: 1, offset: 0 };
    let op = |t: &str, role, axes: Vec<AxisExpr>, is_output| NestOperand {
        tensor: t.into(),
        role,
        axes,
        dtype: PrecisionSpec::new(Precision::Bf16),
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

fn cost_of(unit: &UnitTemplate, nest: &OpNest) -> Result<CostEntry, kiln_ir::common::Diagnostic> {
    cost(&CostQuery { unit, nest, objective: Objective::Latency, options: CostOptions::default() })
}

#[test]
fn fixed_operand_partitions_bound_each_role() {
    let nest = gemm(8, 8, 8);
    let unified = template(&design(feed(json!({})), 1e9, 1e9), true);
    cost_of(&unified, &nest).expect("384 B of operands fit 1 KiB");
    let parts = json!({ "operands": { "policy": "partitioned", "parts": { "a": 64, "b": 448, "o": 512 } } });
    let split = template(&design(feed(parts.clone()), 1e9, 1e9), true);
    let l = split.chains[0].levels[0];
    assert_eq!(split.levels[l].share(OperandRole::A), (0, 64));
    let e = cost_of(&split, &nest).expect_err("A's 128 B exceed its 64 B partition");
    assert_eq!(e.code, "E-COST-INFEASIBLE", "{e:?}");
    // Staging through the feed (operands homed in HBM), A's tiles must fit 64 B: k at most 4 per m row of 8.
    let staged = template(&design(feed(parts), 1e9, 1e9), false);
    let r = cost_of(&staged, &nest).expect("A streams through its partition");
    assert!(r.accesses.iter().filter(|a| a.level == l && a.operand == 0).all(|a| a.write_bytes > 0));
}

#[test]
fn memory_latency_is_converted_to_unit_cycles() {
    let lat = json!({ "clock": "m", "overrides": { "latency": 10 } });
    let t = template(&design(feed(lat), 2e9, 1e9), false);
    let l = t.chains[0].levels[0];
    assert_eq!(t.levels[l].latency_cycles, 20, "10 cycles at 1 GHz are 20 cycles of the 2 GHz unit");
    let t = template(&design(feed(json!({ "clock": "m", "overrides": { "latency": 3 } })), 1e9, 3e9), false);
    assert_eq!(t.levels[t.chains[0].levels[0]].latency_cycles, 1, "rounded up to whole unit cycles");
}
