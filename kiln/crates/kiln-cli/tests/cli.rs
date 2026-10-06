use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use kiln_trace::meas::MeasSession;

fn kiln(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kiln"))
        .args(args)
        .output()
        .unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn measurement(name: &str) -> String {
    root()
        .join("../calibration/measurements")
        .join(name)
        .display()
        .to_string()
}

fn golden_result() -> String {
    root()
        .join("crates/kiln-trace/tests/golden/result_min.json")
        .display()
        .to_string()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn legacy_import_writes_sealed_session() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a100.json");
    let o = kiln(&[
        "bench",
        "import",
        "--legacy",
        &measurement("a100_2026-10-04.json"),
        "-o",
        s(&out),
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let text = String::from_utf8(o.stdout).unwrap();
    assert!(
        text.contains("decode_b1") && text.contains("12.732 ms"),
        "{text}"
    );
    let session = MeasSession::parse(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert!(session.validate().is_empty());
    assert_eq!(
        code(&kiln(&["validate", "--kind", "measurement", s(&out)])),
        0
    );
    assert_eq!(code(&kiln(&["trace", "validate", s(&out)])), 0);
}

#[test]
fn legacy_import_json_and_store() {
    let dir = tempfile::tempdir().unwrap();
    let o = kiln(&[
        "bench",
        "import",
        "--legacy",
        &measurement("tpuv5e_2026-10-04.json"),
        "--format",
        "json",
        "--store",
        s(dir.path()),
    ]);
    assert_eq!(code(&o), 0);
    let session = MeasSession::parse(&String::from_utf8(o.stdout).unwrap()).unwrap();
    let stored = dir
        .path()
        .join("google/tpu-v5-lite/2026-10-05_legacy-tpuv5e_2026-10-04.json");
    assert_eq!(
        MeasSession::parse(&std::fs::read_to_string(stored).unwrap()).unwrap(),
        session
    );
    assert!(dir.path().join("index.json").is_file());
}

#[test]
fn superseded_file_is_flagged() {
    let o = kiln(&[
        "bench",
        "import",
        "--legacy",
        &measurement("a100_2026-10-04_v0writeflush.json"),
    ]);
    assert_eq!(code(&o), 1);
    assert!(String::from_utf8_lossy(&o.stderr).contains("E-CAL-0303"));
}

#[test]
fn input_errors_exit_3() {
    assert_eq!(
        code(&kiln(&["bench", "import", "--legacy", "/nonexistent.json"])),
        3
    );
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.json");
    std::fs::write(&junk, r#"{"hello": 1}"#).unwrap();
    assert_eq!(code(&kiln(&["bench", "import", "--legacy", s(&junk)])), 3);
    assert_eq!(code(&kiln(&["trace", "info", s(&junk)])), 3);
}

#[test]
fn trace_info_and_validate_on_result() {
    let g = golden_result();
    let o = kiln(&["trace", "info", &g, "--format", "json"]);
    assert_eq!(code(&o), 0);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["kind"], "result");
    assert_eq!(code(&kiln(&["trace", "validate", &g])), 0);

    let mut bad: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&g).unwrap()).unwrap();
    bad["status"] = "invalid".into();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("bad.json");
    std::fs::write(&p, bad.to_string()).unwrap();
    let o = kiln(&["trace", "validate", s(&p)]);
    assert_eq!(code(&o), 4);
    assert!(String::from_utf8_lossy(&o.stderr).contains("E-TRACE-SCORE"));
}

#[test]
fn unimplemented_commands_exit_69() {
    for args in [
        &["bench", "export", "--suite", "legacy", "--sequence"][..],
        &["bench", "import", "x.json", "--runner", "cuda"],
        &["diff-test", "--oracle", "stream"],
        &["agree"],
        &["trust", "run"],
        &["trust", "report"],
        &["corpus", "run", "--update"],
        &["perf"],
        &["trace", "pack"],
        &["trace", "unpack"],
        &["trace", "upgrade"],
        &["trace", "recover"],
        &["fmt", "d.json", "--canonical"],
        &["expand", "d.json"],
        &["hash", "d.json"],
        &["migrate", "d.json", "--write"],
        &["phys", "export-def"],
        &["phys", "import-def"],
        &["schema", "result"],
    ] {
        let o = kiln(args);
        assert_eq!(
            code(&o),
            69,
            "{args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("E-NOT-IMPLEMENTED"),
            "{args:?}"
        );
    }
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(code(&kiln(&["--bogus"])), 2);
    assert_eq!(code(&kiln(&["eval", "--tier", "Z"])), 2);
    assert_eq!(code(&kiln(&["bench", "export", "--target", "gpu"])), 2);
    assert_eq!(code(&kiln(&["--help"])), 0);
}

fn design(dir: &str, name: &str) -> String {
    root()
        .join("designs")
        .join(dir)
        .join(name)
        .display()
        .to_string()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn validate_reference_designs() {
    for name in [
        "a100_sxm4_40gb.json5",
        "ember.json5",
        "tpu_v5e.json5",
        "tpu_v5e_2x2.json5",
        "tpu_v6e.json5",
    ] {
        let path = design("reference", name);
        let o = kiln(&["validate", &path, "--profile", "reference"]);
        assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));
        let o = kiln(&[
            "validate",
            &path,
            "--profile",
            "reference",
            "--format",
            "json",
        ]);
        let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        let hash = kiln_ir::hw::load_file(&path).unwrap().hash;
        assert_eq!(v["summary"]["hash"], hash.as_str(), "{name}");
        assert_eq!(
            (v["kind"].as_str(), v["ok"].as_bool()),
            (Some("hardware"), Some(true))
        );
        assert!(v["summary"]["peak_ops"]["bf16*bf16+fp32"].as_f64().unwrap() > 1e14);
    }
    let o = kiln(&["validate", &design("reference", "a100_sxm4_40gb.json5")]);
    let text = stdout(&o);
    for want in [
        "311.9 TFLOPS  bf16*bf16+fp32",
        "623.7 TFLOPS  int8*int8+int32",
        "1555.2 GB/s",
        "40.0 GiB",
    ] {
        assert!(text.contains(want), "{want}: {text}");
    }
}

#[test]
fn validate_legacy_designs_and_profiles() {
    for name in ["a100", "a100_40gb", "tpuv4", "tpuv5e", "tpuv6e"] {
        let path = design("legacy", &format!("{name}.json"));
        assert_eq!(
            code(&kiln(&["validate", &path, "--profile", "stream_compat"])),
            0,
            "{name}"
        );
        let o = kiln(&["validate", "--kind", "hw", &path, "--profile", "reference"]);
        assert_eq!(code(&o), 1, "{name}");
        assert!(stderr(&o).contains("E-IR-1103"), "{name}");
    }
    let o = kiln(&[
        "validate",
        &design("reference", "ember.json5"),
        "--profile",
        "stream_compat",
    ]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("E-IR-1105"));
}

#[test]
fn validate_input_errors() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.json5");
    std::fs::write(&bad, "{schema: \"kiln.hw/1.0\", name: ").unwrap();
    let o = kiln(&["validate", s(&bad)]);
    assert_eq!(code(&o), 3);
    assert!(stderr(&o).contains("E-IR-0100"));
    let unknown = dir.path().join("x.json");
    std::fs::write(&unknown, r#"{"hello": 1}"#).unwrap();
    let o = kiln(&["validate", s(&unknown), "--format", "json"]);
    assert_eq!(code(&o), 3);
    assert!(stderr(&o).contains("E-CLI-0001"));
    assert_eq!(code(&kiln(&["validate", "/nonexistent.json5"])), 3);
    assert_eq!(
        code(&kiln(&["validate", "--kind", "workload", s(&unknown)])),
        3
    );
}

fn fixture_workload() -> serde_json::Value {
    let p = root().join("crates/kiln-wl/tests/fixtures/llama3_8b_decode_b8.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn validate_workloads() {
    let dir = tempfile::tempdir().unwrap();
    let mut doc = fixture_workload();
    doc["scenarios"]["static_b2"] =
        serde_json::json!({"mode": {"static": {"batch": 2, "prompt_len": 128, "gen_len": 16}}});
    let ok = dir.path().join("wl.json");
    std::fs::write(&ok, doc.to_string()).unwrap();
    let o = kiln(&["validate", s(&ok), "--format", "json"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let wl: kiln_ir::wl::WorkloadDoc = serde_json::from_value(fixture_workload()).unwrap();
    let model = kiln_wl::zoo::expand_doc(&wl).unwrap();
    assert_eq!(
        v["summary"]["model_hash"],
        kiln_ir::wl::model_hash(model.expanded().unwrap())
    );
    let scen = v["summary"]["scenarios"].as_array().unwrap();
    let decode = &scen[0]["phases"][0];
    assert_eq!(decode["params"], 8_030_261_248u64);
    assert_eq!(decode["tokens"], 8);
    assert!(decode["flops_mm"].as_f64().unwrap() > 2.0 * 8.0 * 7.5e9);
    assert!(
        scen[1]["phases"].as_array().unwrap().len() >= 2,
        "static expands to prefill + decode points"
    );
    let text = stdout(&kiln(&["validate", s(&ok)]));
    assert!(
        text.contains("decode_b8_kv2048") && text.contains("params 8.030e9"),
        "{text}"
    );

    doc["scenarios"]["decode_b8_kv2048"]["mode"]["snapshot"]["seqs"]["segments"][0]["q_len"] =
        0.into();
    let bad = dir.path().join("bad.json");
    std::fs::write(&bad, doc.to_string()).unwrap();
    let o = kiln(&["validate", s(&bad)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("E-WL-SCN"));
}

#[test]
fn validate_auto_detects_measurements() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a100.json");
    let m = measurement("a100_2026-10-04.json");
    assert_eq!(
        code(&kiln(&["bench", "import", "--legacy", &m, "-o", s(&out)])),
        0
    );
    assert_eq!(code(&kiln(&["validate", s(&out)])), 0);
}

#[test]
fn import_harness_designs_match_legacy_goldens() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["a100", "a100_40gb", "tpuv4", "tpuv5e", "tpuv6e"] {
        let src = root()
            .join("../harness/designs")
            .join(format!("{name}.json"));
        let out = dir.path().join(format!("{name}.json"));
        let o = kiln(&["import", "harness-design", s(&src), "-o", s(&out)]);
        assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));
        assert!(stdout(&o).contains("hw1-"));
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            std::fs::read_to_string(design("legacy", &format!("{name}.json"))).unwrap(),
            "{name}"
        );
        assert_eq!(
            code(&kiln(&["validate", s(&out), "--profile", "stream_compat"])),
            0
        );
    }
    let src = root().join("../harness/designs/tpuv5e.json");
    let o = kiln(&["import", "harness-design", s(&src)]);
    assert_eq!(
        stdout(&o),
        std::fs::read_to_string(design("legacy", "tpuv5e.json")).unwrap()
    );
    assert_eq!(
        code(&kiln(&["import", "harness-design", "/nonexistent.json"])),
        3
    );
    let junk = dir.path().join("junk.json");
    std::fs::write(&junk, "[1, 2]").unwrap();
    assert_eq!(code(&kiln(&["import", "harness-design", s(&junk)])), 3);
}

#[test]
fn bench_export_manifests() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("legacy.json");
    let o = kiln(&["bench", "export", "--suite", "legacy", "-o", s(&out)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stdout(&o).contains("gemm_2048_6144_4096_linear"));
    let m: kiln_wl::bench::BenchManifest =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(m, kiln_wl::bench::manifest("legacy").unwrap());
    assert_eq!(m.ops.len(), 52);
    assert!(m.ops.iter().all(|o| o.key == o.op.key()));

    let a = kiln(&[
        "bench",
        "export",
        "--suite",
        "llama3_8b:decode_b1",
        "--format",
        "json",
    ]);
    let b = kiln(&[
        "bench",
        "export",
        "--workload",
        "llama3_8b:decode_b1",
        "--format",
        "json",
    ]);
    assert_eq!(code(&a), 0);
    assert_eq!(a.stdout, b.stdout);
    let m: kiln_wl::bench::BenchManifest = serde_json::from_slice(&a.stdout).unwrap();
    assert_eq!(m.schema, kiln_wl::bench::MANIFEST_SCHEMA);
    assert!(
        m.ops
            .iter()
            .any(|o| o.legacy_name.as_deref() == Some("gemm_1_6144_4096_linear"))
    );

    let smoke = kiln(&["bench", "export", "--suite", "smoke", "--format", "json"]);
    assert_eq!(code(&smoke), 0);
    let o = kiln(&["bench", "export", "--suite", "nope"]);
    assert_eq!(code(&o), 3);
    assert!(stderr(&o).contains("E-WL-SCN-001"));
    assert_eq!(code(&kiln(&["bench", "export", "--suite", "evolve"])), 3);
}

#[test]
fn calibration_sets_validate_and_report_without_logging() {
    let set = root().join("calibration/sets/generic-v1.json");
    let before = std::fs::read(&set).unwrap();
    let o = kiln(&["validate", "--kind", "calibration", s(&set)]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let o = kiln(&["calibrate", "report", "--set", "generic-v1", "--devices", "tpu_v5e", "--no-log", "--format", "json"]);
    assert!(matches!(code(&o), 0 | 4), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v[0]["set_id"], "generic-v1");
    assert!(v[0]["verdicts"].as_array().is_some_and(|a| !a.is_empty()));
    assert_eq!(std::fs::read(&set).unwrap(), before, "--no-log must not touch the set file");
    let dir = tempfile::tempdir().unwrap();
    let mut bad: serde_json::Value = serde_json::from_slice(&before).unwrap();
    bad["parameters"][0]["value"] = 0.5.into();
    let p = dir.path().join("bad.json");
    std::fs::write(&p, bad.to_string()).unwrap();
    assert_eq!(code(&kiln(&["validate", "--kind", "calibration", s(&p)])), 4);
}
