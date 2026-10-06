//! T1 backtest designs (06 §12.1): validate clean under `reference` and reproduce published peaks and bandwidths.

use kiln_ir::hw::compute::MemKind;
use kiln_ir::hw::model::{MemSpec, NodeIx};
use kiln_ir::hw::{HwModel, Profile, check_file};
use kiln_ir::op_class::OpClass;
use kiln_ir::precision::Precision;
use kiln_trust::{BACKTEST, design_path, unit_peak};

fn clean(name: &str) -> HwModel {
    let r = check_file(design_path(name), Profile::Reference);
    assert!(r.diagnostics.is_empty(), "{name}: {:#?}", r.diagnostics);
    r.model.expect("model")
}

fn within(got: f64, published: f64, tol: f64, what: &str) {
    assert!((got / published - 1.0).abs() <= tol, "{what}: derived {got:.4e} vs published {published:.4e}");
}

fn enabled(m: &HwModel, entity: &str) -> usize {
    m.nodes.iter().filter(|n| n.enabled && n.entity == entity).count()
}

fn tops(m: &HwModel, p: Precision) -> f64 {
    m.peak_ops_for(p, None)
}

/// Vector FMA peak as published (2 FLOP per lane-op).
fn vector_fma(m: &HwModel, mode: &str) -> f64 {
    2.0 * unit_peak(m, "vector", mode, OpClass::Elementwise)
}

#[test]
fn all_backtest_designs_validate_clean_and_carry_no_platform_key() {
    for name in BACKTEST {
        let m = clean(name);
        assert_eq!(m.family, None, "{name}: backtest designs are evaluated with the generic set only (06 §12.1)");
        let d = kiln_trust::load(name).unwrap();
        assert!(d.doc.meta.claims.len() >= 3, "{name}");
        for c in &d.doc.meta.claims {
            assert!(d.doc.meta.citations.contains_key(&c.source), "{name}: claim {} cites unknown {}", c.metric, c.source);
        }
        let s = check_file(design_path(name), Profile::Search);
        assert!(s.diagnostics.is_empty(), "{name} under search: {:#?}", s.diagnostics);
    }
}

#[test]
fn v100_reproduces_datasheet() {
    let m = clean("v100_sxm2_32gb");
    within(tops(&m, Precision::Fp16), 125e12, 0.01, "fp16 tensor");
    within(vector_fma(&m, "fp32"), 15.7e12, 0.01, "fp32");
    within(vector_fma(&m, "fp64"), 7.8e12, 0.01, "fp64");
    within(m.offchip_bandwidth(None).0, 900e9, 0.01, "HBM2");
    assert_eq!(m.offchip_capacity(None).0, 32 << 30);
    assert_eq!(tops(&m, Precision::Bf16), 0.0, "Volta has no bf16 tensor path");
    assert_eq!(enabled(&m, "board.gpu.gv100.gpc.tpc.sm"), 80);
    assert_eq!(enabled(&m, "board.gpu.gv100.gpc.tpc.sm.smsp.tc"), 640);
    assert_eq!(m.onchip_capacity(None, Some(MemKind::Cache)).0, 6 << 20);
    let mcs = m.nodes.iter().filter(|n| n.enabled && n.entity == "board.gpu.gv100.hbm_if.mc").count();
    assert_eq!(mcs, 8, "8 x 512-bit controllers");
}

#[test]
fn h100_sxm_reproduces_datasheet() {
    let m = clean("h100_sxm5_80gb");
    within(tops(&m, Precision::Bf16), 989.4e12, 0.001, "bf16 dense");
    within(tops(&m, Precision::Fp8E4m3), 1978.9e12, 0.001, "fp8 dense");
    within(tops(&m, Precision::Int8), 1978.9e12, 0.001, "int8 dense");
    within(tops(&m, Precision::Tf32), 494.7e12, 0.001, "tf32 dense");
    within(m.summary().chips[0].peak_ops["bf16*bf16+fp32:sparse"], 1979e12, 0.001, "bf16 sparse (datasheet headline)");
    within(m.offchip_bandwidth(None).0, 3.35e12, 0.002, "HBM3");
    assert_eq!(m.offchip_capacity(None).0, 80 << 30);
    assert_eq!(m.onchip_capacity(None, Some(MemKind::Cache)).0, 50 << 20);
    assert_eq!(enabled(&m, "board.gpu.gh100.gpc.tpc.sm"), 132);
    assert_eq!(enabled(&m, "board.gpu.hbm"), 5);
    // One modelled clock (1.83 GHz, the tensor-peak clock): FP32 lands at 1830/1980 of the 67 TFLOPS datasheet value.
    within(vector_fma(&m, "fp32"), 66.9e12 * 1830.0 / 1980.0, 0.001, "fp32 at the tensor clock");
    let ports: Vec<_> = m.ports.iter().filter(|p| m.nodes[p.node].entity_id == "nvlink").collect();
    let bw = ports[0].spec.link.as_ref().unwrap().derived_bandwidth(None).unwrap().0;
    assert_eq!(ports.len() as f64 * 2.0 * bw / 1e9, 900.0, "18 links x 2 dirs x 25 GB/s");
}

#[test]
fn h100_pcie_reproduces_datasheet() {
    let m = clean("h100_pcie_80gb");
    within(tops(&m, Precision::Bf16), 756.5e12, 0.001, "bf16 dense");
    within(tops(&m, Precision::Fp8E4m3), 1513e12, 0.001, "fp8 dense");
    within(m.offchip_bandwidth(None).0, 2.0e12, 0.001, "HBM2e");
    assert_eq!(enabled(&m, "board.gpu.gh100.gpc.tpc.sm"), 114);
    let sxm = clean("h100_sxm5_80gb");
    let r = m.offchip_bandwidth(None).0 / sxm.offchip_bandwidth(None).0;
    within(r, 2.0 / 3.35, 0.002, "PCIe/SXM bandwidth ratio");
}

#[test]
fn tpu_v4_reproduces_published() {
    let m = clean("tpu_v4");
    within(tops(&m, Precision::Bf16), 275e12, 0.002, "bf16");
    within(tops(&m, Precision::Int8), 275e12, 0.002, "int8 at the bf16 rate");
    within(m.offchip_bandwidth(None).0, 1200e9, 1e-9, "HBM2");
    assert_eq!(m.offchip_capacity(None).0, 32 << 30);
    assert_eq!(enabled(&m, "board.chip.die.tc.mxu"), 8);
    let mem = |p: &str| match m.index[p] {
        NodeIx::Mem(i) => i,
        other => panic!("{p} is {other:?}"),
    };
    let (vmem, cmem, hbm) = (mem("board.chip.die.tc1.vmem"), mem("board.chip.die.cmem"), mem("board.chip.hbm3"));
    assert_eq!(m.memories[vmem].capacity.0, 16 << 20);
    assert_eq!(m.memories[cmem].capacity.0, 128 << 20);
    assert!(matches!(m.memories[cmem].spec, MemSpec::OnChip(_)));
    assert!(m.route(NodeIx::Mem(hbm), NodeIx::Mem(cmem)).is_some());
    assert!(m.route(NodeIx::Mem(cmem), NodeIx::Mem(vmem)).is_some());
    let ici: Vec<_> = m.ports.iter().filter(|p| m.nodes[p.node].entity_id == "ici").collect();
    let bw = ici[0].spec.link.as_ref().unwrap().derived_bandwidth(None).unwrap().0;
    assert_eq!((ici.len(), bw / 1e9), (6, 50.0), "6 links @ 50 GB/s (ISCA 2023 Table 4)");
}

#[test]
fn peak_ratios_between_generations() {
    let v100 = clean("v100_sxm2_32gb");
    let a100 = kiln_ir::hw::check_file(design_path("a100_sxm4_40gb"), Profile::Reference).model.unwrap();
    let h100 = clean("h100_sxm5_80gb");
    let v4 = clean("tpu_v4");
    let v5e = kiln_ir::hw::check_file(design_path("tpu_v5e"), Profile::Reference).model.unwrap();
    let bw = |m: &HwModel| m.offchip_bandwidth(None).0;
    within(bw(&h100) / bw(&a100), 3.35 / 1.555, 0.005, "H100/A100 HBM");
    within(bw(&a100) / bw(&v100), 1.555 / 0.9, 0.005, "A100/V100 HBM");
    within(bw(&v5e) / bw(&v4), 819.0 / 1200.0, 0.005, "v5e/v4 HBM");
    within(tops(&h100, Precision::Bf16) / tops(&a100, Precision::Bf16), 989.4 / 312.0, 0.005, "H100/A100 bf16");
    within(tops(&a100, Precision::Fp16) / tops(&v100, Precision::Fp16), 312.0 / 125.0, 0.005, "A100/V100 fp16");
    within(tops(&v5e, Precision::Bf16) / tops(&v4, Precision::Bf16), 197.0 / 275.0, 0.005, "v5e/v4 bf16");
}
