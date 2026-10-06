//! Calibration-set format (06 §3.6): round trip, hashing, validation against the registry, resolution.

use std::collections::BTreeMap;

use kiln_calib::predict::Bench;
use kiln_sim::calib::{CALIB_SCHEMA, CalParam, CalibSet, ParamStatus, RangeSpec, SetKind};
use kiln_sim::params::PessDir;

fn param(name: &str, key: (&str, &str), value: f64, range: (f64, f64)) -> CalParam {
    CalParam {
        name: name.into(),
        key: BTreeMap::from([(key.0.to_string(), key.1.to_string())]),
        value,
        unit: "1".into(),
        bounds: [0.0, 1.0],
        prior: value,
        ci95: Some([range.0, range.1]),
        range: RangeSpec { lower: range.0, upper: range.1, basis: "single_device".into() },
        pess_dir: PessDir::Lower,
        status: ParamStatus::Fit,
        frozen_reason: None,
        source: None,
        table: None,
        diagnostics: None,
    }
}

fn set() -> CalibSet {
    CalibSet {
        schema: CALIB_SCHEMA.into(),
        id: "generic-test".into(),
        version: 1,
        kind: SetKind::Generic,
        platform: None,
        parameters: vec![
            param("eta_res", ("dram_kind", "hbm2e"), 0.92, (0.9, 0.94)),
            param("eta_res", ("dram_kind", "*"), 0.91, (0.85, 0.97)),
            CalParam { unit: "s".into(), bounds: [1e-7, 5e-5], pess_dir: PessDir::Upper, ..param("t_sync", ("exec_model", "static_dataflow"), 1.5e-6, (1e-6, 2e-6)) },
        ],
        fit: None,
        range_policy: None,
        acceptance: None,
        test_access_log: vec![],
        created: "2026-10-05".into(),
        hash: None,
    }
}

#[test]
fn round_trip_and_hash() {
    let s = set();
    let text = s.to_json();
    let back = CalibSet::from_value(serde_json::from_str(&text).unwrap(), "t").unwrap();
    assert_eq!(back.hash.as_deref(), Some(s.compute_hash().as_str()));
    assert_eq!(back.compute_hash(), s.compute_hash());
    assert!(s.compute_hash().starts_with("cal1-"));
    // Report fields are not hashed: a report never changes what results carry.
    let mut logged = back.clone();
    logged.test_access_log.push(serde_json::json!({"command": "report"}));
    logged.acceptance = Some(serde_json::json!({"pass": false}));
    assert_eq!(logged.compute_hash(), s.compute_hash());
    assert!(logged.validate().is_empty());
    // Content changes do.
    let mut tampered = back;
    tampered.parameters[0].value = 0.93;
    assert_ne!(tampered.compute_hash(), s.compute_hash());
    assert!(tampered.validate().iter().any(|d| d.message.contains("does not match its content")));
}

#[test]
fn registry_rejects_per_op_and_unregistered_parameters() {
    let codes = |s: &CalibSet| s.validate().into_iter().map(|d| d.message).collect::<Vec<_>>();
    let mut s = set();
    s.parameters.push(param("gemm_fudge", ("dram_kind", "hbm2"), 0.9, (0.8, 1.0)));
    assert!(codes(&s).iter().any(|m| m.contains("unregistered parameter \"gemm_fudge\"")));
    let mut s = set();
    s.parameters.push(param("eta_res", ("op", "decode_b1/o_proj"), 0.9, (0.8, 1.0)));
    assert!(codes(&s).iter().any(|m| m.contains("per-op")));
    let mut s = set();
    s.parameters.push(param("eta_res", ("exec_model", "host_launched"), 0.9, (0.8, 1.0)));
    assert!(codes(&s).iter().any(|m| m.contains("not its mechanism key")));
    let mut s = set();
    s.parameters[0].range.lower = 0.95;
    assert!(codes(&s).iter().any(|m| m.contains("does not bracket")));
    let mut s = set();
    s.platform = Some("a100_40gb".into());
    assert!(codes(&s).iter().any(|m| m.contains("names a platform")));
    let mut s = set();
    s.parameters.push(s.parameters[0].clone());
    assert!(codes(&s).iter().any(|m| m.contains("appears twice")));
    // Unknown fields are rejected by the schema itself.
    let mut v = serde_json::to_value(set()).unwrap();
    v["parameters"][0]["per_op_scale"] = 1.1.into();
    assert!(CalibSet::from_value(v, "t").is_err());
}

#[test]
fn resolution_uses_exact_keys_then_wildcards_then_priors() {
    let b = Bench::new("tpu_v5e.json5").unwrap();
    let exec = b.view().hw.exec_model;
    let ps = set().resolve(b.view(), exec);
    let get = |n: &str| ps.params.iter().find(|p| p.name == n).unwrap();
    assert_eq!(get("eta_res").range.central, 0.92); // hbm2e exact key
    assert_eq!(get("t_sync").range.central, 1.5e-6);
    assert_eq!(get("t_program").status, "assumed"); // not in the set: assumed prior
    assert!(ps.extrapolated.contains(&"t_program".to_string()) && !ps.extrapolated.contains(&"eta_res".to_string()));
    assert_eq!(ps.hash(), set().compute_hash());
    let v6 = Bench::new("tpu_v6e.json5").unwrap();
    let ps6 = set().resolve(v6.view(), exec);
    assert_eq!(ps6.params.iter().find(|p| p.name == "eta_res").unwrap().range.central, 0.91); // hbm3 -> `*`
    assert!(ps6.extrapolated.contains(&"eta_res".to_string()));
}

#[test]
fn checked_in_sets_validate() {
    let dir = kiln_sim::calib::sets_dir();
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "json") {
            let s = CalibSet::load(&p).unwrap_or_else(|d| panic!("{}: {}", p.display(), d.message));
            assert_eq!(s.hash.as_deref(), Some(s.compute_hash().as_str()), "{}", p.display());
            if s.kind == SetKind::Generic {
                assert!(s.parameters.iter().all(|p| !p.key.contains_key("unit_template") || p.key["unit_template"] == "*"));
                assert!(s.parameters.iter().all(|p| p.name != "f_cap_op"));
            }
            n += 1;
        }
    }
    assert!(n >= 3, "expected platform:a100_40gb, generic-v1 and generic-v2 in {}", dir.display());
}
