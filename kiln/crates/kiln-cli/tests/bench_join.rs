//! Measurements (kiln-trace import) and bench exports (kiln-wl) join on identical descriptor keys.

use std::path::PathBuf;

use kiln_trace::meas::legacy::import_legacy;
use kiln_trace::meas::{MeasSession, Suite};
use kiln_wl::bench::{BenchManifest, manifest};

fn session(name: &str, with_oplist: bool) -> MeasSession {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../calibration");
    let raw = std::fs::read(dir.join("measurements").join(name)).unwrap();
    let oplist = with_oplist.then(|| std::fs::read(dir.join("oplist.json")).unwrap());
    import_legacy(&raw, name, oplist.as_deref())
        .unwrap()
        .session
}

fn assert_joins_one_to_one(s: &MeasSession, m: &BenchManifest) -> usize {
    let llm: Vec<_> = s
        .records
        .iter()
        .filter(|r| r.suite == Suite::LlmOps)
        .collect();
    for r in &llm {
        let hits: Vec<_> = m.ops.iter().filter(|o| o.key == r.bench_key).collect();
        assert_eq!(hits.len(), 1, "{} joins {} exported ops", r.id, hits.len());
        assert_eq!(hits[0].op, r.op, "{}", r.id);
    }
    llm.len()
}

#[test]
fn legacy_manifest_and_a100_session_join_one_to_one() {
    let a100 = session("a100_2026-10-04.json", true);
    let m = manifest("legacy").unwrap();
    for o in &m.ops {
        let name = o.legacy_name.as_deref().unwrap();
        let r = a100
            .record(&format!("ops/{name}"))
            .unwrap_or_else(|| panic!("no A100 record for {name}"));
        assert_eq!(r.bench_key, o.key, "{name}");
        assert_eq!(r.legacy_name.as_deref(), Some(name));
    }
    assert_eq!(assert_joins_one_to_one(&a100, &m), m.ops.len());
    assert_eq!(m.ops.len(), 52);
    for u in &a100.uses {
        assert!(m.get(&u.bench_key).is_some(), "{}/{}", u.phase, u.op);
    }
}

#[test]
fn oplist_rows_join_by_legacy_key() {
    let a100 = session("a100_2026-10-04.json", true);
    let m = manifest("legacy").unwrap();
    for row in kiln_wl::legacy::oplist().unwrap() {
        let variants: Vec<_> = m
            .ops
            .iter()
            .filter(|o| o.op.legacy_key().as_deref() == Some(row.key.as_str()))
            .collect();
        assert_eq!(
            variants.len(),
            if row.weight { 2 } else { 1 },
            "{}",
            row.key
        );
        for o in variants {
            let name = o.op.legacy_name().unwrap();
            assert_eq!(
                a100.record(&format!("ops/{name}")).unwrap().bench_key,
                o.key
            );
        }
    }
}

#[test]
fn llama_suite_exports_join_a100_records() {
    let a100 = session("a100_2026-10-04.json", true);
    let mut joined = 0;
    for suite in ["legacy", "standard"] {
        for member in kiln_wl::zoo::suite(suite).unwrap() {
            let ops = kiln_wl::bench::export_scenario(member.model(), member.scenario()).unwrap();
            for e in ops.iter().filter(|e| e.op.kind.is_contraction()) {
                let name = e.op.legacy_name().unwrap();
                let Some(r) = a100.record(&format!("ops/{name}")) else {
                    continue;
                };
                let same_dtypes = e.op.operands.values().all(|o| o.dtype == "bf16");
                assert_eq!(
                    r.bench_key == e.key(),
                    same_dtypes,
                    "{} {name}",
                    member.name
                );
                joined += usize::from(same_dtypes);
            }
        }
    }
    assert!(joined >= 30, "{joined}");
}

#[test]
fn tpu_session_joins_legacy_manifest() {
    let tpu = session("tpuv5e_2026-10-04.json", false);
    let m = manifest("legacy").unwrap();
    for r in tpu.records.iter().filter(|r| r.suite == Suite::LlmOps) {
        assert!(m.get(&r.bench_key).is_some(), "{}", r.id);
    }
}
