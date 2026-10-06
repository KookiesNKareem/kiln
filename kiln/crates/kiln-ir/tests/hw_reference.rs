//! Reference designs (01 §20): parse, expand, validate clean under `reference`, reproduce 01's derived numbers.

use std::path::PathBuf;

use kiln_ir::hw::model::{ChannelKind, ContainerKind, MemSpec, NodeIx};
use kiln_ir::hw::{Design, HwModel, Profile, Report, check_file, load_file};
use kiln_ir::precision::Precision;

fn design(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference").join(name)
}

fn clean(name: &str) -> HwModel {
    let r: Report = check_file(design(name), Profile::Reference);
    assert!(r.diagnostics.is_empty(), "{name}: {:#?}", r.diagnostics);
    r.model.expect("model")
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn tflops(m: &HwModel, p: Precision) -> f64 {
    round1(m.peak_ops_for(p, None) / 1e12)
}

fn gbps(m: &HwModel) -> f64 {
    round1(m.offchip_bandwidth(None).0 / 1e9)
}

fn enabled<'a>(m: &'a HwModel, entity: &'a str) -> impl Iterator<Item = &'a kiln_ir::hw::model::InstNode> + 'a {
    m.nodes.iter().filter(move |n| n.enabled && n.entity == entity)
}

fn mem(m: &HwModel, path: &str) -> usize {
    match m.index[path] {
        NodeIx::Mem(i) => i,
        other => panic!("{path} is {other:?}"),
    }
}

#[test]
fn a100_matches_01() {
    let m = clean("a100_sxm4_40gb.json5");
    assert_eq!(tflops(&m, Precision::Bf16), 311.9);
    assert_eq!(tflops(&m, Precision::Int8), 623.7, "01 rounds 2 x 311.87 to 623.8");
    assert_eq!(tflops(&m, Precision::Tf32), 155.9);
    assert_eq!(tflops(&m, Precision::Fp64), 19.5);
    assert_eq!(gbps(&m), 1555.2);
    assert_eq!(enabled(&m, "board.gpu.ga100.gpc.tpc.sm").count(), 108);
    assert_eq!(enabled(&m, "board.gpu.ga100.gpc.tpc.sm.smsp.tc").count(), 432);
    assert_eq!(enabled(&m, "board.gpu.ga100.l2p.slice").count(), 80);
    assert_eq!(m.onchip_capacity(None, Some(kiln_ir::hw::compute::MemKind::Cache)).0, 40 << 20);
    // 01 §7's formula sums every port (read + write = 128 B/clk/slice); 01 §20.1's 7.22 TB/s counts reads only.
    let read_bw: f64 = enabled(&m, "board.gpu.ga100.l2p.slice")
        .map(|n| match &m.memories[mem(&m, &n.path)].spec {
            MemSpec::OnChip(s) => s.ports.iter().filter(|p| p.dir == kiln_ir::hw::compute::PortDir::Read).map(|p| f64::from(p.width_bits) / 8.0).sum::<f64>() * 1.41e9,
            _ => 0.0,
        })
        .sum();
    assert_eq!(round1(read_bw / 1e12), 7.2, "80 slices x 64 B/clk at 1410 MHz");
    let rf = mem(&m, "board.gpu.ga100.gpc0_0.tpc0.sm0.smsp0.rf");
    let l1 = mem(&m, "board.gpu.ga100.gpc0_0.tpc0.sm0.l1");
    let slice = mem(&m, "board.gpu.ga100.l2p0_1.slice7");
    let hbm = mem(&m, "board.gpu.hbm0");
    assert_eq!([m.level(rf), m.level(l1), m.level(slice), m.level(hbm)], [1, 2, 3, 4]);
    assert!(!m.node(m.index["board.gpu.hbm5"]).enabled);
    assert!(m.route(NodeIx::Mem(hbm), NodeIx::Mem(l1)).is_some());
    assert_eq!(m.memories[l1].backing.len(), 80);
    let s = m.summary();
    assert_eq!(s.chip_count, 1);
    assert_eq!(round1(s.chips[0].peak_ops["bf16*bf16+fp32:sparse"] / 1e12), 623.7);
    assert_eq!(m.power_domains.len(), 1);
    assert_eq!(m.exec_model, kiln_ir::hw::types::ExecModel::HostLaunched);
    let mc = match m.index["board.gpu.ga100.hbm_if0.mc0"] {
        NodeIx::Block(b) => b,
        other => panic!("{other:?}"),
    };
    let mc_bw: Vec<f64> = m.channels.iter().filter(|c| c.src == NodeIx::Block(mc) && c.kind == ChannelKind::NocHop).map(|c| c.bandwidth.unwrap().0).collect();
    assert_eq!(mc_bw.len(), 1);
    assert_eq!(round1(mc_bw[0] / 1e9), 155.5, "MC port: 1024 b per hbm_clk (link clock), not the fabric's gpc_clk");
}

#[test]
fn tpu_v5e_matches_01() {
    let m = clean("tpu_v5e.json5");
    assert_eq!(tflops(&m, Precision::Bf16), 196.6);
    assert_eq!(tflops(&m, Precision::Int8), 393.2);
    assert_eq!(gbps(&m), 819.2);
    for synth in ["board.chip.die.hbm0_mc", "board.chip.die.hbm1_phy"] {
        let NodeIx::Block(b) = m.index[synth] else { panic!("{synth}") };
        assert!(m.blocks[b].synthesized);
    }
    let (vreg, vmem) = (mem(&m, "board.chip.die.tc.vreg"), mem(&m, "board.chip.die.tc.vmem"));
    assert_eq!((m.level(vreg), m.level(vmem), m.level(mem(&m, "board.chip.hbm1"))), (1, 2, 3));
    let route = m.route(NodeIx::Mem(mem(&m, "board.chip.hbm0")), NodeIx::Mem(vmem)).unwrap();
    assert_eq!(route.len(), 4, "hbm -> phy -> mc -> xbar -> vmem");
    let ici: Vec<_> = m.ports.iter().filter(|p| m.nodes[p.node].entity_id == "ici").collect();
    let bw = ici[0].spec.link.as_ref().unwrap().derived_bandwidth(None).unwrap().0;
    assert_eq!(ici.len() as f64 * 2.0 * bw / 1e9, 400.0, "4 ports x 2 dirs x 50 GB/s");
}

#[test]
fn tpu_v6e_extends_v5e() {
    let m = clean("tpu_v6e.json5");
    assert_eq!(tflops(&m, Precision::Bf16), 917.5);
    assert_eq!(gbps(&m), 1638.4);
    assert_eq!(m.family.as_deref(), Some("tpu_v6e"));
    assert_eq!(m.exec_model, kiln_ir::hw::types::ExecModel::StaticDataflow, "inherited from v5e");
    assert_eq!(enabled(&m, "board.chip.die.sc.tile").count(), 32);
    let dma = m.networks.iter().find(|n| m.nodes[n.node].path == "board.chip.die.dma").unwrap();
    assert!(dma.endpoints.contains(&m.index["board.chip.die.sc1.spmem"]));
    let v5e = load_file(design("tpu_v5e.json5")).unwrap();
    let v6e = load_file(design("tpu_v6e.json5")).unwrap();
    assert_eq!(v6e.doc.meta.claims.len(), 3);
    assert!(v6e.doc.meta.citations.contains_key("v5e") && v6e.doc.meta.citations.contains_key("v6e"));
    let d = v5e.diff(&v6e);
    assert!(d.added.contains(&"board.chip.die.sc".to_string()), "{d:#?}");
    assert!(d.changed.iter().any(|c| c.field == "geometry.systolic.rows" && c.new == 256));
    assert!(v5e.diff(&v5e).is_empty());
}

#[test]
fn tpu_v5e_2x2_torus() {
    let m = clean("tpu_v5e_2x2.json5");
    assert_eq!(round1(m.peak_ops_for(Precision::Bf16, None) / 1e12), 786.4);
    assert_eq!(gbps(&m), 3276.8);
    let ici = m.networks.iter().position(|n| m.nodes[n.node].path == "tray.ici").unwrap();
    let chans: Vec<_> = m.channels.iter().filter(|c| c.network == Some(ici)).collect();
    assert_eq!(chans.len(), 16, "8 links (size-2 dims doubled) x 2 directions");
    assert!(chans.iter().all(|c| c.kind == ChannelKind::Serdes && c.bandwidth.unwrap().0 == 50e9));
    let used: std::collections::BTreeSet<_> = chans.iter().map(|c| c.src).collect();
    assert_eq!(used.len(), 16, "every chip uses all 4 ICI ports");
    let bisection: f64 = chans
        .iter()
        .filter(|c| m.path(c.src).starts_with("tray.chip0_") && m.path(c.dst).starts_with("tray.chip1_"))
        .map(|c| c.bandwidth.unwrap().0)
        .sum();
    assert_eq!(bisection / 1e9, 200.0, "01 §20.4: 2 pairs x 2 links x 50 GB/s per direction");
    let a = mem(&m, "tray.chip0_0.die.tc.vmem");
    let b = mem(&m, "tray.chip1_1.die.tc.vmem");
    assert!(m.route(NodeIx::Mem(a), NodeIx::Mem(b)).is_some());
    assert_eq!(m.channels.iter().filter(|c| c.kind == ChannelKind::Host).count(), 8);
    let s = m.summary();
    assert_eq!((s.chip_count, s.inter_chip.len()), (4, 1));
    assert_eq!(m.tree.iter().filter(|c| c.kind == ContainerKind::Host).count(), 1);
}

#[test]
fn ember_exercises_every_feature() {
    let m = clean("ember.json5");
    let pim = enabled(&m, "board.pkg.hbm.pim").count();
    assert_eq!(pim, 4 * 64, "HBM4 32 channels x 2 pseudo-channels per stack");
    assert_eq!(enabled(&m, "board.pkg.cc.c.cim").count(), 4 * 8 * 16);
    assert_eq!(enabled(&m, "board.pkg.cc.t.mx").count(), 4 * 64 - 2);
    assert_eq!(gbps(&m), 8192.0);
    let pim0 = m.units.iter().find(|u| m.nodes[u.node].path == "board.pkg.hbm2.pim5").unwrap();
    let near = pim0.near.as_ref().unwrap();
    assert_eq!((m.nodes[m.memories[near.mem].node].path.as_str(), near.slice), ("board.pkg.hbm2", 5));
    for kind in [ChannelKind::D2d, ChannelKind::Vertical, ChannelKind::Near] {
        assert!(m.channels.iter().any(|c| c.kind == kind), "{kind:?}");
    }
    let d2d = m.channels.iter().filter(|c| c.kind == ChannelKind::D2d).count();
    assert_eq!(d2d, 4 * 2, "2x2 mesh: 4 links x 2 directions");
    let l3 = mem(&m, "board.pkg.sram1_0.l3");
    assert!(m.route(NodeIx::Mem(l3), NodeIx::Mem(mem(&m, "board.pkg.cc1_0.t7_7.sram"))).is_some());
    assert!(matches!(m.memories[l3].spec, MemSpec::OnChip(_)));
    let noc = m.networks.iter().find(|n| m.nodes[n.node].path == "board.pkg.cc0_1.noc").unwrap();
    assert_eq!(noc.routers.len(), 72);
}

#[test]
fn ember_cim_peak_is_bit_serial() {
    let m = clean("ember.json5");
    let s = m.summary();
    let macros = 4.0 * 8.0 * 16.0;
    assert_eq!(s.peak_ops["int8*int4+int32"], 2.0 * macros * (256.0 * 64.0 / 8.0) * 1.6e9, "3355.4 TOPS, not 107,374");
    let cim: Vec<usize> = (0..m.units.len()).filter(|&u| m.units[u].spec.kind.name() == "cim").collect();
    let int8 = m.peak_ops(&cim, "int8*int8+", kiln_ir::op_class::OpClass::Matmul);
    assert_eq!(int8, 2.0 * macros * (256.0 * 32.0 / 8.0) * 1.6e9);
}

#[test]
fn reference_designs_under_search() {
    for (name, expected) in [
        ("a100_sxm4_40gb.json5", &["E-IR-1102"][..]),
        ("tpu_v5e.json5", &["E-IR-1102", "E-IR-1106"]),
        ("tpu_v6e.json5", &["E-IR-1102", "E-IR-1106"]),
        ("tpu_v5e_2x2.json5", &["E-IR-1106"; 4]),
        ("ember.json5", &[]),
    ] {
        let r = check_file(design(name), Profile::Search);
        let codes: Vec<&str> = r.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert_eq!(codes, expected, "{name}: only `family` (calibration inheritance) and unpublished caps are rejected");
    }
}

#[test]
fn golden_expansion_counts_and_hashes() {
    let golden = [
        ("a100_sxm4_40gb.json5", 1536, 726, 5854),
        ("tpu_v5e.json5", 7, 9, 50),
        ("tpu_v6e.json5", 37, 9, 110),
        ("tpu_v5e_2x2.json5", 28, 36, 224),
        ("ember.json5", 1536, 808, 5292),
    ];
    for (name, units, mems, chans) in golden {
        let d = load_file(design(name)).unwrap();
        let (m, _) = d.expand(&Default::default()).unwrap();
        assert_eq!((m.units.len(), m.memories.len(), m.channels.len()), (units, mems, chans), "{name}");
        let (m2, _) = d.expand(&Default::default()).unwrap();
        assert_eq!(serde_json::to_string(&m).unwrap(), serde_json::to_string(&m2).unwrap(), "{name}: deterministic");
        let again = Design::from_value(d.canonical.clone()).unwrap();
        assert_eq!(again.hash, d.hash, "{name}: canonical form reloads to the same hash");
        assert!(d.hash.starts_with("hw1-") && d.hash.len() == 36);
    }
}
