//! Fit determinism, recovery of known parameters from synthetic measurements, and the leakage guard.

use kiln_calib::campaign::Campaign;
use kiln_calib::fit::{Fitter, LEAK_CODE, apply};
use kiln_calib::predict::Bench;
use kiln_calib::records::{Record, Split};
use kiln_calib::split::{DEFAULT_SALT, test_families};

fn device_records(c: &Campaign, id: &str) -> Vec<Record> {
    c.records[id].clone()
}

#[test]
fn no_test_record_reaches_a_fit() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let recs = device_records(&c, "tpu_v5e");
    let b = Bench::new("tpu_v5e.json5").unwrap();
    let mut f = Fitter::new(&b, &recs);
    f.bootstrap = 4;
    // Direct attempt: a test record in a stage is refused.
    let test_ix = recs.iter().position(|r| r.split == Split::Test).unwrap();
    let fit_ix = recs.iter().position(|r| r.split == Split::Fit && matches!(r.kind, kiln_calib::records::Kind::Contraction { .. })).unwrap();
    let base = f.base();
    let err = f.stage("U", vec![fit_ix, test_ix], vec![], &base).unwrap_err();
    assert_eq!(err.code, LEAK_CODE);
    let hold = recs.iter().position(|r| r.split == Split::FitHoldout).unwrap();
    assert_eq!(f.stage("U", vec![hold], vec![], &base).unwrap_err().code, LEAK_CODE);
    // Full fit: every staged record is a calib-micro fit record whose family no test op shares.
    let out = f.run().unwrap();
    let mut all = vec![];
    for d in kiln_calib::records::DEVICES {
        all.extend(c.records[d.id].clone());
    }
    let tests = test_families(&all);
    let test_hashes: std::collections::BTreeSet<&str> = all.iter().filter(|r| r.split == Split::Test).map(|r| r.hash.as_str()).collect();
    let mut n = 0;
    for s in &out.stages {
        for &i in &s.records {
            let r = &recs[i];
            assert_eq!(r.split, Split::Fit, "{}", r.name);
            assert!(r.session.contains("_micro"), "{} from {}", r.name, r.session);
            assert!(!test_hashes.contains(r.hash.as_str()));
            assert!(r.family().is_none_or(|fam| !tests.contains(&fam)), "{} shares a test family", r.name);
            n += 1;
        }
    }
    assert!(n > 100);
    // LLM-suite and whole-step records are all test.
    assert!(recs.iter().filter(|r| r.group == "llm_op" || r.group == "sequence").all(|r| r.split == Split::Test));
}

#[test]
fn fit_is_deterministic_across_thread_counts() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let recs = device_records(&c, "tpu_v5e");
    let b = Bench::new("tpu_v5e.json5").unwrap();
    let run = |threads| {
        let mut f = Fitter::new(&b, &recs);
        f.bootstrap = 12;
        f.threads = threads;
        f.run().unwrap().entries
    };
    let (a, b2) = (run(1), run(5));
    assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b2).unwrap());
}

/// Synthetic measurements generated from known parameters through the same engine are recovered. Stages run
/// in a fixed order without a joint refit (06 §3.3), so each scenario moves the parameters of the stages it
/// tests and keeps later-stage DRAM terms at their priors (L-stage elementwise records still carry a little
/// DRAM time, so an offset of `t_dram_ramp` leaks into the launch terms: scenarios move eta, not the ramp).
/// (device, design file, true parameter values).
type Scenario<'a> = (&'a str, &'a str, &'a [(&'a str, f64)]);

#[test]
fn recovers_known_parameters_from_synthetic_measurements() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let cases: &[Scenario] = &[
        ("tpu_v5e", "tpu_v5e.json5", &[("t_sync", 1.7e-6), ("unit_eff", 0.9)]),
        ("tpu_v5e", "tpu_v5e.json5", &[("eta_res", 0.935), ("unit_eff", 0.97)]),
        ("a100_40gb", "a100_sxm4_40gb.json5", &[("t_gap", 0.6e-6), ("t_min_kernel", 1.1e-6), ("unit_eff", 0.88)]),
        ("a100_40gb", "a100_sxm4_40gb.json5", &[("eta_res", 0.93)]),
    ];
    for &(id, design, truth) in cases {
        let b = Bench::new(design).unwrap();
        let mut recs = device_records(&c, id);
        let f0 = Fitter::new(&b, &recs);
        let mut p = f0.base();
        for &(n, v) in truth {
            apply(&mut p, n, v, &f0.unit_paths);
        }
        let synth: Vec<Option<f64>> = (0..recs.len())
            .map(|i| f0.cases[i].as_ref().map(|c| b.predict(c, &p, recs[i].clock_hz)))
            .collect();
        drop(f0);
        for (r, s) in recs.iter_mut().zip(synth) {
            if let Some(s) = s {
                r.meas_s = s;
            }
        }
        let mut f = Fitter::new(&b, &recs);
        f.bootstrap = 4;
        let out = f.run().unwrap();
        for st in &out.stages {
            for (j, x) in st.free.iter().enumerate() {
                // Parameters of stages before the moved ones absorb the offset (the documented cross-talk).
                let Some(v) = truth.iter().find(|t| t.0 == x.name).map(|t| t.1).or((truth[0].0.starts_with("t_")).then_some(x.b.prior)) else {
                    continue;
                };
                assert!(st.frozen[j].is_none(), "{id}: {} frozen: {:?}", x.name, st.frozen[j]);
                let rel = (st.value[j] / v - 1.0).abs();
                assert!(rel < 0.02, "{id}: {} recovered {:.4e}, truth {v:.4e}", x.name, st.value[j]);
            }
        }
        assert!(out.failures.is_empty());
    }
}

#[test]
fn clock_fit_uses_fit_records_only() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let recs = device_records(&c, "a100_40gb");
    let b = Bench::new("a100_sxm4_40gb.json5").unwrap();
    let base = Fitter::new(&b, &recs).clock_stage().unwrap();
    let mut moved = recs.clone();
    let mut n = 0;
    for r in moved.iter_mut().filter(|r| r.split != Split::Fit) {
        if let Some(f) = r.clock_hz.as_mut() {
            *f *= 0.5;
            n += 1;
        }
    }
    assert!(n > 0);
    assert_eq!(Fitter::new(&b, &moved).clock_stage().unwrap(), base);
    for r in moved.iter_mut().filter(|r| r.split == Split::Fit) {
        r.split = Split::FitHoldout;
    }
    assert_eq!(Fitter::new(&b, &moved).clock_stage(), None);
}

#[test]
fn zero_bootstrap_samples_are_rejected() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let recs = device_records(&c, "tpu_v5e");
    let b = Bench::new("tpu_v5e.json5").unwrap();
    let mut f = Fitter::new(&b, &recs);
    f.bootstrap = 0;
    let out = f.run();
    assert!(out.is_err(), "{:?}", out.map(|o| o.entries.iter().map(|e| e.ci95).collect::<Vec<_>>()));
}

#[test]
fn unthrottled_telemetry_keeps_a_finite_clock_range() {
    let c = Campaign::load(DEFAULT_SALT).unwrap();
    let mut recs = device_records(&c, "a100_40gb");
    for r in recs.iter_mut().filter(|r| r.split == Split::Fit) {
        if let Some(f) = r.clock_hz.as_mut() {
            *f = 1e12;
        }
    }
    let b = Bench::new("a100_sxm4_40gb.json5").unwrap();
    let fit = Fitter::new(&b, &recs).clock_stage().unwrap();
    assert!(fit.range.0.is_finite() && fit.range.1.is_finite(), "{:?}", fit.range);
    assert!(fit.range.0 <= 1.0 && fit.range.1 >= 1.0, "{:?}", fit.range);
}
