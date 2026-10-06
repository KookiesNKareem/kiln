//! 08 §F rulings: canonical TPU per-op mode `loop`, the chip-state gate, the `eta_res` bound and the v6e
//! platform set.

use kiln_calib::records::{self, Device, Split};
use kiln_calib::split;
use kiln_sim::calib::{CalibSet, SetKind, registered};

fn mode_median(file: &str, name: &str, mode: &str) -> f64 {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(records::measurements_dir().join(file)).unwrap()).unwrap();
    let r = v["records"].as_array().unwrap().iter().find(|r| r["name"] == name).unwrap();
    r["modes"][mode]["median_s"].as_f64().unwrap()
}

#[test]
fn tpu_records_use_loop_never_pipelined_rot() {
    for id in ["tpu_v5e", "tpu_v6e"] {
        let dev = records::device(id).unwrap();
        assert_eq!(dev.op_modes, ["loop"]);
        let recs = records::load_device(dev).unwrap();
        assert!(recs.iter().all(|r| r.mode != "pipelined_rot"), "{id}");
        assert!(recs.iter().filter(|r| r.split == Split::Test && r.group == "llm_op").all(|r| r.mode == "loop"));
    }
    // F.4's example: pipelined_rot 86.7 us beat loop 106.5 us; the canonical value is now the loop median.
    let dev = records::device("tpu_v6e").unwrap();
    let file = dev.suite[0];
    let recs = records::suite(dev, file).unwrap();
    let r = recs.iter().find(|r| r.name.ends_with("/gemm_1_14336_4096")).unwrap();
    assert_eq!(r.meas_s, mode_median(file, "gemm_1_14336_4096", "loop"));
    assert!(r.meas_s > 1.15 * mode_median(file, "gemm_1_14336_4096", "pipelined_rot"));
    assert_eq!(records::device("a100_40gb").unwrap().op_modes, ["graph_cold"]);
}

#[test]
fn chip_state_gate_excludes_compute_bound_records_of_a_changed_session() {
    // diag3 switched from ~586 to ~733 TF mid-run (08 §F.4); read it as if it were a fit session.
    let dev = Device { micro: &["tpuv6e_2026-10-05_diag3_suite.json"], ..records::device("tpu_v6e").unwrap().clone() };
    let mut recs = records::micro(&dev, dev.micro[0]).unwrap();
    let gated: Vec<_> = recs.iter().filter(|r| r.chip_state.is_some()).collect();
    assert!(gated.len() >= 5, "{}", gated.len());
    assert!(gated.iter().all(|r| r.flops / 918e12 >= r.bytes / 1.64e12));
    // Memory-bound GEMVs (HBM unaffected by the state) stay.
    assert!(recs.iter().any(|r| r.chip_state.is_none() && r.name.starts_with("gemm_1_")));
    split::assign(&mut recs, &Default::default(), split::DEFAULT_SALT);
    for r in recs.iter().filter(|r| r.chip_state.is_some()) {
        assert!(matches!(&r.split, Split::Excluded(why) if why.starts_with("chip state")), "{}: {:?}", r.name, r.split);
    }
    // The reference fit sessions are steady: nothing gated.
    for id in ["tpu_v5e", "tpu_v6e"] {
        let dev = records::device(id).unwrap();
        let states = records::chip_states(dev).unwrap();
        assert!(states.iter().all(|(_, s)| !s.changed() && s.recorded()), "{id}: {states:?}");
        assert!(records::load_device(dev).unwrap().iter().all(|r| r.chip_state.is_none()));
    }
    assert!(records::chip_states(records::device("a100_40gb").unwrap()).unwrap().is_empty());
}

#[test]
fn eta_res_bound_admits_single_program_v6e_hbm() {
    let r = registered("eta_res").unwrap();
    assert_eq!(r.bounds, (0.5, 1.0));
    assert!(r.bounds.0 < 0.665 && r.band == Some((0.85, 0.97)));
}

#[test]
fn v6e_platform_set_round_trips_and_stays_out_of_generic_sets() {
    let s = kiln_calib::campaign::load_set("platform:tpu_v6e").unwrap();
    assert_eq!((s.kind, s.platform.as_deref()), (SetKind::Platform, Some("tpu_v6e")));
    assert!(s.validate().is_empty(), "{:?}", s.validate());
    let back = CalibSet::from_value(serde_json::from_str(&s.to_json()).unwrap(), "t").unwrap();
    assert_eq!(back.compute_hash(), s.compute_hash());
    assert_eq!(s.hash.as_deref(), Some(s.compute_hash().as_str()));
    let fit = s.fit.as_ref().unwrap();
    assert_eq!(fit.fit_devices, ["tpu_v6e"]);
    assert!(fit.measurement_sessions.iter().all(|m| m.starts_with("tpuv6e_2026-10-05_micro.json:")));
    assert!(fit.notes.iter().any(|n| n.contains("canonical per-op timing mode loop")));
    assert!(fit.notes.iter().any(|n| n.contains("chip state") && n.contains("steady")));
    let eta = s.parameters.iter().find(|p| p.name == "eta_res" && p.key.get("dram_kind").map(String::as_str) == Some("hbm3")).unwrap();
    assert!((0.5..0.85).contains(&eta.value), "{}", eta.value);
    for id in ["generic-v1", "generic-v2"] {
        let g = kiln_calib::campaign::load_set(id).unwrap();
        assert!(g.fit.as_ref().unwrap().fit_devices.iter().all(|d| d != "tpu_v6e"), "{id}");
        assert!(g.parameters.iter().all(|p| p.key.get("dram_kind").map(String::as_str) != Some("hbm3")), "{id}");
    }
}
