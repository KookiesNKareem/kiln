//! `kiln.hw/0` round trip: every harness design imports, validates under `stream_compat`, and keeps its
//! derived peak FLOP/s, off-chip bandwidth and on-chip capacity (harness/design.py formulas).

use std::path::PathBuf;

use kiln_ir::hw::{ExpandOptions, Profile, check, import_harness};
use kiln_ir::op_class::OpClass;
use serde_json::Value;

const DESIGNS: [&str; 5] = ["a100", "a100_40gb", "tpuv4", "tpuv5e", "tpuv6e"];

fn harness(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../harness/designs").join(format!("{name}.json"));
    std::fs::read_to_string(p).expect("harness design present")
}

struct Legacy {
    peak_flops: f64,
    offchip_bps: f64,
    onchip_bytes: f64,
}

fn legacy_metrics(v: &Value) -> Legacy {
    let clk = v["clock_mhz"].as_f64().unwrap() * 1e6;
    let f = |x: &Value, k: &str, d: f64| x.get(k).and_then(Value::as_f64).unwrap_or(d);
    let units = v["compute"].as_array().unwrap();
    let peak_flops = units
        .iter()
        .filter(|c| c["kind"] == "matrix")
        .map(|c| 2.0 * f(c, "rows", 0.0) * f(c, "cols", 0.0) * f(c, "count", 1.0))
        .sum::<f64>()
        * clk;
    let mem: f64 = v["memory"].as_array().unwrap().iter().map(|m| f(m, "size_mib", 0.0) * 1048576.0 * f(m, "count", 1.0)).sum();
    let bufs: f64 = units
        .iter()
        .map(|c| (f(c, "buffer_kib", 2048.0) + f(c, "regfile_kib", 0.0)) * 1024.0 * f(c, "count", 1.0))
        .sum();
    Legacy { peak_flops, offchip_bps: v["offchip"]["bandwidth_gbps"].as_f64().unwrap() * 1e9, onchip_bytes: mem + bufs }
}

#[test]
fn harness_designs_round_trip() {
    for name in DESIGNS {
        let text = harness(name);
        let legacy = legacy_metrics(&serde_json::from_str(&text).unwrap());
        let design = import_harness(&text).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let r = check(design, Profile::StreamCompat, &ExpandOptions::default());
        let errs: Vec<_> = r.errors().collect();
        assert!(errs.is_empty(), "{name}: {errs:?}");
        let m = r.model.as_ref().unwrap();
        let mats: Vec<usize> = (0..m.units.len()).filter(|&u| m.units[u].spec.kind.is_mac()).collect();
        let peak = m.peak_ops(&mats, "", OpClass::Matmul);
        let rel = |a: f64, b: f64| (a - b).abs() / b;
        assert!(rel(peak, legacy.peak_flops) < 1e-12, "{name}: peak {peak} vs {}", legacy.peak_flops);
        let bw = m.offchip_bandwidth(None).0;
        assert!(rel(bw, legacy.offchip_bps) < 1e-12, "{name}: bw {bw} vs {}", legacy.offchip_bps);
        let cap = m.onchip_capacity(None, None).as_f64();
        assert_eq!(cap, legacy.onchip_bytes, "{name}: on-chip bytes");
    }
}

#[test]
fn harness_import_is_deterministic_and_loads_via_schema_0() {
    for name in DESIGNS {
        let text = harness(name);
        let a = import_harness(&text).unwrap();
        let b = import_harness(&text).unwrap();
        assert_eq!(a.hash, b.hash);
        let mut v: Value = serde_json::from_str(&text).unwrap();
        v["schema"] = "kiln.hw/0".into();
        let via_schema = kiln_ir::hw::Design::from_source(&kiln_ir::hw::MemLoader::default(), None, &v.to_string()).unwrap();
        assert_eq!(via_schema.hash, a.hash, "{name}");
        assert!(via_schema.warnings.iter().any(|w| w.code == "W-IR-1902"));
    }
}

#[test]
fn harness_designs_import_without_warnings_and_match_golden() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/legacy");
    for name in DESIGNS {
        let d = import_harness(&harness(name)).unwrap();
        let golden = dir.join(format!("{name}.json"));
        let text = serde_json::to_string_pretty(&d.canonical).unwrap() + "\n";
        if std::env::var_os("KILN_BLESS").is_some() {
            std::fs::write(&golden, &text).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&golden).expect("golden present (KILN_BLESS=1 to write)"), text, "{name}");
        let r = check(d, Profile::StreamCompat, &ExpandOptions::default());
        assert!(r.diagnostics.is_empty(), "{name}: {:#?}", r.diagnostics);
    }
}

#[test]
fn harness_overrides_equal_derived_and_pass_search() {
    for name in DESIGNS {
        let r = check(import_harness(&harness(name)).unwrap(), Profile::Search, &ExpandOptions::default());
        assert!(r.diagnostics.is_empty(), "{name}: overrides equal derived values: {:#?}", r.diagnostics);
    }
    let legacy = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/legacy/a100.json");
    let text = std::fs::read_to_string(legacy).unwrap().replace("\"bandwidth\": 407800000000", "\"bandwidth\": 815600000000");
    let r = kiln_ir::hw::check_str(&kiln_ir::hw::MemLoader::default(), None, &text, Profile::Search);
    let over: Vec<_> = r.diagnostics.iter().filter(|d| d.code == "E-IR-1101").collect();
    assert_eq!(over.len(), 1, "{:#?}", r.diagnostics);
    assert_eq!(over[0].path.as_deref(), Some("board.chip.hbm0"));
    assert!(over[0].message.contains("ratio 2.00"), "{}", over[0].message);
}
