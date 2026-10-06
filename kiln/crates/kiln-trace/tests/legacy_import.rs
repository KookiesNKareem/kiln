use std::path::PathBuf;

use kiln_trace::meas::legacy::{LegacyImport, import_legacy};
use kiln_trace::meas::store::{MeasStore, SessionStatus};
use kiln_trace::meas::{MeasSession, QualityStatus, Suite, codes};

fn calibration() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../calibration")
}

fn import(name: &str, with_oplist: bool) -> LegacyImport {
    let dir = calibration();
    let raw = std::fs::read(dir.join("measurements").join(name))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let oplist = with_oplist.then(|| std::fs::read(dir.join("oplist.json")).unwrap());
    import_legacy(&raw, name, oplist.as_deref()).unwrap()
}

fn round_trip(s: &MeasSession) {
    let json = s.canonical_json();
    let back = MeasSession::parse(&json).unwrap();
    assert_eq!(&back, s);
    assert_eq!(back.canonical_json(), json);
    assert_eq!(back.compute_hash(), s.hash.clone().unwrap());
    assert_eq!(back.validate(), vec![]);
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a / b - 1.0).abs() <= rel
}

#[test]
fn a100_import_preserves_values() {
    let imp = import("a100_2026-10-04.json", true);
    assert!(!imp.superseded);
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
    let s = &imp.session;
    round_trip(s);
    assert_eq!(s.session_id, "legacy-a100_2026-10-04");
    assert_eq!(s.device.sku, "A100-SXM4-40GB");
    assert_eq!(s.device.l2_bytes, Some(40 * (1 << 20)));
    assert_eq!(s.clocks_power.power_limit_w, Some(400.0));
    assert_eq!(s.method.flush_bytes, Some(256 << 20));
    assert_eq!(s.quality.status, QualityStatus::Accepted);
    let count = |suite| s.records.iter().filter(|r| r.suite == suite).count();
    assert_eq!(
        (
            count(Suite::LlmOps),
            count(Suite::Sweep),
            count(Suite::Peak)
        ),
        (52, 33, 4)
    );

    let g = s.record("sweep/gemm_4096_4096_4096").unwrap();
    let t = g.modes["graph_cold"].median_s;
    assert!(close(t, 546.1196899414062e-6, 1e-15), "{t}");
    assert_eq!(g.modes["graph_cold"].details["copies"], 6);
    assert_eq!(
        g.bench_key,
        s.record("ops/gemm_4096_4096_4096").unwrap().bench_key
    );

    let decode = s.phase_sum_s("decode_b1", "graph_cold").unwrap();
    assert!(close(decode, 12.732e-3, 1e-4), "{decode}");
    assert!(close(
        s.phase_sum_s("prefill_b1", "graph_cold").unwrap(),
        138.62e-3,
        1e-4
    ));
    assert!(close(
        s.phase_sum_s("decode_b1", "flushed").unwrap(),
        13.593e-3,
        1e-4
    ));
    assert_eq!(s.uses.iter().filter(|u| u.suite == "all").count(), 28);

    let peak = s.record("peak/matmul_16384^3").unwrap();
    assert_eq!(peak.clock.as_ref().unwrap().sm_hz_median, 1275e6);
    assert_eq!(peak.quality.status, QualityStatus::Accepted);
    let tiny = s.record("sweep/gemm_256_256_256").unwrap();
    assert_eq!(tiny.quality.status, QualityStatus::Flagged);
    assert!(
        tiny.quality
            .reasons
            .iter()
            .any(|r| r.contains("L2-resident"))
    );
}

#[test]
fn import_is_deterministic() {
    let a = import("a100_2026-10-04.json", true).session;
    let b = import("a100_2026-10-04.json", true).session;
    assert_eq!(a.hash, b.hash);
    // Golden: a change here means the imported content changed (importer version is part of the hash).
    assert_eq!(
        a.hash.as_deref(),
        Some("meas1-b4676e6c09bb38abaffdb7c5a7ecfb6b")
    );
    // Re-blessed for 08 §F: canonical TPU mode `loop`, `best` and `pipelined_rot` marked diagnostic.
    let t = import("tpuv5e_2026-10-04.json", false).session;
    assert_eq!(
        t.hash.as_deref(),
        Some("meas1-5b4237557c0551c7a540a7675f5ea015")
    );
    assert_ne!(a.hash, import("a100_2026-10-04.json", false).session.hash);
}

#[test]
fn overflowing_phase_sum_is_an_error() {
    let mut s = import("a100_2026-10-04.json", true).session;
    let u = s.uses.iter().position(|u| u.phase == "decode_b1").unwrap();
    s.uses[u].count = 2;
    let rec = s.uses[u].record.clone();
    let r = s.records.iter_mut().find(|r| r.id == rec).unwrap();
    r.modes.get_mut("graph_cold").unwrap().median_s = 1e308;
    let s = s.seal();
    assert_eq!(s.validate(), vec![]);
    assert_eq!(
        s.phase_sum_s("decode_b1", "graph_cold").unwrap_err().code,
        codes::FIELD
    );
}

#[test]
fn a100_without_oplist_warns() {
    let imp = import("a100_2026-10-04.json", false);
    assert!(imp.session.uses.is_empty());
    assert_eq!(imp.warnings[0].code, codes::OPLIST);
    assert_eq!(
        imp.session
            .phase_sum_s("decode_b1", "graph_cold")
            .unwrap_err()
            .code,
        codes::LOOKUP
    );
}

#[test]
fn superseded_a100_is_rejected() {
    let imp = import("a100_2026-10-04_v0writeflush.json", true);
    assert!(imp.superseded);
    assert!(imp.warnings.iter().any(|w| w.code == codes::SUPERSEDED));
    let s = &imp.session;
    round_trip(s);
    assert_eq!(s.quality.status, QualityStatus::Rejected);
    assert!(
        s.records
            .iter()
            .all(|r| r.quality.status == QualityStatus::Rejected)
    );
    assert!(
        s.records
            .iter()
            .all(|r| !r.modes.contains_key("graph_cold"))
    );
}

#[test]
fn tpu_v5e_import_preserves_values() {
    let imp = import("tpuv5e_2026-10-04.json", false);
    assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
    let s = &imp.session;
    round_trip(s);
    assert_eq!(s.device.sku, "TPU v5 lite");
    // 08 §F: canonical TPU per-op mode is `loop`; `best` and `pipelined_rot` are kept as diagnostics.
    assert_eq!(s.canonical_mode, "loop");
    assert!(s.timing_modes["pipelined_rot"].diagnostic && s.timing_modes["best"].diagnostic);
    assert!(!s.timing_modes["loop"].diagnostic);
    assert!(
        s.method.settings["canonical_op_mode"]
            .as_str()
            .unwrap()
            .starts_with("loop")
    );
    let count = |suite| s.records.iter().filter(|r| r.suite == suite).count();
    assert_eq!(
        (
            count(Suite::LlmOps),
            count(Suite::Sweep),
            count(Suite::Peak),
            count(Suite::Hbm)
        ),
        (33, 27, 3, 8)
    );
    let decode = s.phase_sum_s("decode_b1", &s.canonical_mode).unwrap();
    assert!(close(decode, 21.004328229885075e-3, 1e-12), "{decode}");
    let best = s.phase_sum_s("decode_b1", "best").unwrap();
    assert!(close(best, 21.00244194098609e-3, 1e-12), "{best}");
    let qkv = s.record("ops/prefill_b1/qkv").unwrap();
    assert_eq!(qkv.modes["best"].median_s, 0.0005812025116309214);
    assert_eq!(qkv.modes["best"].details["from"], "loop");
    assert_eq!(qkv.modes["loop"].details["R"], 6);
    let big = s.record("peak/gemm_16384_16384_16384").unwrap();
    assert!(!big.modes.contains_key("best") && big.quality.status == QualityStatus::Flagged);
}

#[test]
fn a100_and_tpu_keys_join() {
    let a = import("a100_2026-10-04.json", true).session;
    let t = import("tpuv5e_2026-10-04.json", false).session;
    let a_key = &a.record("ops/bmm_1_8192_2048_128").unwrap().bench_key;
    let t_key = &t.record("ops/prefill_b1/attn_score").unwrap().bench_key;
    assert_eq!(a_key, t_key);
    let shared = t
        .records
        .iter()
        .filter(|r| a.by_key(&r.bench_key).next().is_some())
        .count();
    assert!(shared >= 60, "{shared}");
}

#[test]
fn tpu_v6e_if_present() {
    let name = "tpuv6e_2026-10-04.json";
    if !calibration().join("measurements").join(name).exists() {
        return;
    }
    round_trip(&import(name, false).session);
}

#[test]
fn store_is_append_only() {
    let dir = tempfile::tempdir().unwrap();
    let store = MeasStore::new(dir.path());
    let s = import("tpuv5e_2026-10-04.json", false).session;
    let p = store.put(&s, SessionStatus::Active).unwrap();
    assert!(p.ends_with("google/tpu-v5-lite/2026-10-05_legacy-tpuv5e_2026-10-04.json"));
    assert_eq!(store.put(&s, SessionStatus::Active).unwrap(), p);
    let hash = s.hash.clone().unwrap();
    assert_eq!(store.load(&hash).unwrap(), s);

    let mut tampered = s.clone();
    tampered.records.pop();
    let tampered = tampered.seal();
    assert_eq!(
        store
            .put(&tampered, SessionStatus::Active)
            .unwrap_err()
            .code,
        codes::APPEND_ONLY
    );
    let mut unsealed = s.clone();
    unsealed.session_id = "other".into();
    assert_eq!(
        store
            .put(&unsealed, SessionStatus::Active)
            .unwrap_err()
            .code,
        codes::HASH
    );

    store
        .set_status(
            &hash,
            SessionStatus::Superseded {
                by: "meas1-x".into(),
            },
        )
        .unwrap();
    let index = store.index().unwrap();
    assert_eq!(
        index.sessions[&hash].status,
        SessionStatus::Superseded {
            by: "meas1-x".into()
        }
    );
    assert_eq!(store.load(&hash).unwrap(), s);
}

#[test]
fn canonical_mode_is_trusted_and_not_diagnostic() {
    let s = import("tpuv5e_2026-10-04.json", false).session;
    for mode in ["pipelined_rot", "pipelined", "best"] {
        let m = &s.timing_modes[mode];
        assert!(m.diagnostic || !m.trusted, "{mode}");
        let mut bad = s.clone();
        bad.canonical_mode = mode.into();
        let bad = bad.seal();
        assert!(
            bad.validate().iter().any(|d| d.code == codes::SCHEMA),
            "{mode}: {:?}",
            bad.validate()
        );
    }
}

#[test]
fn unrepresentable_descriptors_fail_validation() {
    let mut s = import("a100_2026-10-04.json", true).session;
    let r = s
        .records
        .iter_mut()
        .find(|r| r.op.kind.is_contraction())
        .unwrap();
    r.op.dims.insert("m".into(), 1 << 43);
    r.op.dims.insert("n".into(), 1 << 43);
    r.op.dims.insert("k".into(), 1 << 43);
    r.bench_key = r.op.key();
    let s = s.seal();
    assert!(
        s.validate().iter().any(|d| d.code == codes::FIELD),
        "{:?}",
        s.validate()
    );
}

#[test]
fn rotation_footprints_do_not_wrap() {
    let reimport = |name: &str, edit: &dyn Fn(&mut serde_json::Value)| {
        let mut v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(calibration().join("measurements").join(name)).unwrap(),
        )
        .unwrap();
        edit(&mut v["ops"][0]);
        let s = import_legacy(&serde_json::to_vec(&v).unwrap(), name, None)
            .unwrap()
            .session;
        s.records
            .iter()
            .find(|r| r.legacy_name.as_deref() == v["ops"][0]["name"].as_str())
            .unwrap()
            .quality
            .reasons
            .clone()
    };
    let a100 = reimport("a100_2026-10-04.json", &|r| {
        r["graph_cold"]["copies"] = 64.into();
        r["min_bytes"] = 288230376151711745u64.into();
    });
    assert!(!a100.iter().any(|q| q.contains("< 512 MiB")), "{a100:?}");
    let tpu = reimport("tpuv5e_2026-10-04.json", &|r| {
        r["loop_info"]["R"] = 64.into();
        r["bytes"] = 288230376151711745u64.into();
    });
    assert!(!tpu.iter().any(|q| q.contains("< 512 MiB")), "{tpu:?}");
}

#[test]
fn unrepresentable_flush_sizes_are_rejected() {
    let raw = std::fs::read_to_string(calibration().join("measurements/a100_2026-10-04.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["flush_bytes"] = serde_json::json!(1u64 << 62);
    let e = import_legacy(v.to_string().as_bytes(), "a100_2026-10-04.json", None).unwrap_err();
    assert_eq!(e.code, codes::FIELD);
}
