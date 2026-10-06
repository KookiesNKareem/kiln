//! Calibration-set assembly (06 §3.1 rule 2, §3.2 ranges, §3.6 file): platform sets bind one chip's fits;
//! generic sets pool technology keys over the fit devices and carry `*` fallbacks (registry prior, plausible
//! band or the span of measured values) for every key a held-out or novel design may need.

use std::collections::BTreeMap;

use kiln_sim::calib::{CALIB_SCHEMA, CalParam, CalibSet, FitInfo, ParamStatus, RangeSpec, SetKind, registered};

use crate::fit::DeviceFit;
use crate::records::{Record, Split, measurements_dir};
use crate::split::SplitFile;

pub const CREATED: &str = "2026-10-05";
pub const LAUNCH_TERMS: &[&str] = &["t_launch", "t_min_kernel", "t_gap", "t_dispatch", "t_program", "t_sync"];

/// File-name form of a set id (`platform:a100_40gb` -> `platform-a100_40gb`).
pub fn file_id(id: &str) -> String {
    id.replace(':', "-")
}

/// sha256 of a measurement file's bytes (`meas-sha256-` + 32 hex).
pub fn session_hash(file: &str) -> String {
    use sha2::Digest;
    let raw = std::fs::read(measurements_dir().join(file)).unwrap_or_default();
    format!("meas-sha256-{}", &hex::encode(sha2::Sha256::digest(&raw))[..32])
}

fn fit_info(fits: &[&DeviceFit], split: &SplitFile, test: &[&Record], bootstrap: usize) -> FitInfo {
    let mut fit_records = vec![];
    let mut sessions = vec![];
    let mut by_class: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut notes = vec![];
    for f in fits {
        for s in &f.stages {
            fit_records.extend(s.records.iter().map(|&i| f.records[i].hash.clone()));
            for (k, &i) in s.records.iter().enumerate() {
                by_class.entry(format!("{}/{}/{}", f.device.id, s.stage, f.records[i].group)).or_default().push(s.residuals[k]);
            }
        }
        sessions.extend(f.device.micro.iter().map(|m| format!("{m}:{}", session_hash(m))));
        notes.push(format!("{}: canonical per-op timing mode {} (06 §4.3, 08 §F)", f.device.id, f.device.op_modes.join(", ")));
        for m in f.device.micro {
            if let Ok(Some(st)) = crate::records::session_state(f.device, m) {
                let gated = f.records.iter().filter(|r| r.session == *m && r.chip_state.is_some()).count();
                notes.push(format!("{}: chip state {m}: {}; {gated} compute-bound records excluded", f.device.id, st.summary()));
            }
        }
        notes.extend(f.failures.iter().map(|d| format!("{}: {}", d.code, d.message)));
    }
    fit_records.sort();
    fit_records.dedup();
    let mut test_records: Vec<String> = test.iter().map(|r| r.hash.clone()).collect();
    test_records.sort();
    test_records.dedup();
    FitInfo {
        kiln_version: kiln_trace::KILN_VERSION.into(),
        git_hash: option_env!("KILN_GIT_HASH").unwrap_or("unknown").into(),
        method: format!(
            "staged P,L,D,U (03 §9); bounded Levenberg-Marquardt on log(pred/meas), Tier A re-cost at fixed mapping; bootstrap {bootstrap}"
        ),
        loss: "huber(delta=0.05)/delta^2 + gaussian prior (sigma = bound width / 4)".into(),
        split_id: split.id.clone(),
        split_hash: split.hash.clone(),
        fit_records,
        test_records,
        measurement_sessions: sessions,
        residuals_by_class: by_class.into_iter().map(|(k, v)| (k, (v.iter().sum::<f64>() / v.len().max(1) as f64).exp())).collect(),
        bootstrap_seed: crate::fit::BOOTSTRAP_SEED,
        fit_devices: fits.iter().map(|f| f.device.id.to_string()).collect(),
        notes,
    }
}

fn finish(mut set: CalibSet) -> CalibSet {
    set.parameters.sort_by(|a, b| a.name.cmp(&b.name).then(a.key.cmp(&b.key)));
    set.hash = Some(set.compute_hash());
    set
}

pub fn platform_set(fit: &DeviceFit, split: &SplitFile, bootstrap: usize) -> CalibSet {
    let test: Vec<&Record> = fit.records.iter().filter(|r| r.split == Split::Test).collect();
    finish(CalibSet {
        schema: CALIB_SCHEMA.into(),
        id: format!("platform:{}", fit.device.id),
        version: 1,
        kind: SetKind::Platform,
        platform: Some(fit.device.id.into()),
        parameters: fit.entries.clone(),
        fit: Some(fit_info(&[fit], split, &test, bootstrap)),
        range_policy: None,
        acceptance: None,
        test_access_log: vec![],
        created: CREATED.into(),
        hash: None,
    })
}

/// Pools technology-keyed entries of the fit devices (platform keys: unit templates and telemetry clock
/// tables stay out) and adds `*` fallbacks.
pub fn generic_set(id: &str, fits: &[&DeviceFit], all_test: &[&Record], split: &SplitFile, bootstrap: usize) -> CalibSet {
    type Key = (String, BTreeMap<String, String>);
    let mut by_key: BTreeMap<Key, Vec<(&str, &CalParam)>> = BTreeMap::new();
    for f in fits {
        for p in &f.entries {
            if p.key.contains_key("unit_template") || p.name == "f_cap_op" {
                continue;
            }
            by_key.entry((p.name.clone(), p.key.clone())).or_default().push((f.device.id, p));
        }
    }
    let mut params = vec![];
    for ((_, _), v) in &by_key {
        if v.len() == 1 {
            params.push(v[0].1.clone());
            continue;
        }
        let mut p = v[0].1.clone();
        let fitted: Vec<&CalParam> = v.iter().map(|x| x.1).filter(|x| x.status == ParamStatus::Fit).collect();
        if !fitted.is_empty() {
            p.value = fitted.iter().map(|x| x.value).sum::<f64>() / fitted.len() as f64;
            p.range = RangeSpec {
                lower: fitted.iter().map(|x| x.range.lower).fold(f64::INFINITY, f64::min).min(p.value),
                upper: fitted.iter().map(|x| x.range.upper).fold(f64::NEG_INFINITY, f64::max).max(p.value),
                basis: "device_spread".into(),
            };
            p.status = ParamStatus::Fit;
            p.frozen_reason = None;
            if let Some(d) = p.diagnostics.as_mut() {
                d.per_device = v.iter().map(|(dev, x)| (dev.to_string(), x.value)).collect();
            }
        }
        params.push(p);
    }
    // Wildcards: prior centre; band where registered, else the span of measured values of the family.
    let measured = |names: &[&str]| -> Vec<&CalParam> {
        params.iter().filter(|p| names.contains(&p.name.as_str()) && p.status == ParamStatus::Fit).collect()
    };
    let mut wild = vec![];
    for (name, keyk) in [
        ("eta_res", "dram_kind"),
        ("t_dram_ramp", "dram_kind"),
        ("unit_eff", "unit_template"),
        ("t_launch", "exec_model"),
        ("t_min_kernel", "exec_model"),
        ("t_gap", "exec_model"),
        ("t_dispatch", "exec_model"),
        ("t_program", "exec_model"),
        ("t_sync", "exec_model"),
    ] {
        let reg = registered(name).expect("registered");
        let family: Vec<&CalParam> = if LAUNCH_TERMS.contains(&name) { measured(LAUNCH_TERMS) } else { measured(&[name]) };
        let (lower, upper, basis) = match reg.band {
            Some((lo, hi)) => (lo.min(reg.prior), hi.max(reg.prior), "band".to_string()),
            None if !family.is_empty() => (
                family.iter().map(|p| p.value).fold(reg.prior, f64::min),
                family.iter().map(|p| p.value).fold(reg.prior, f64::max),
                format!("band: span of measured {}", if LAUNCH_TERMS.contains(&name) { "launch-path terms" } else { name }),
            ),
            None => crate::fit::assumed_band(name, reg.prior, reg.bounds),
        };
        wild.push(CalParam {
            name: name.into(),
            key: BTreeMap::from([(keyk.to_string(), "*".to_string())]),
            value: reg.prior,
            unit: reg.unit.into(),
            bounds: [reg.bounds.0, reg.bounds.1],
            prior: reg.prior,
            ci95: None,
            range: RangeSpec { lower, upper, basis },
            pess_dir: reg.pess_dir,
            status: ParamStatus::Assumed,
            frozen_reason: Some("fallback for keys no fit device measured (extrapolated)".into()),
            source: None,
            table: None,
            diagnostics: None,
        });
    }
    params.extend(wild);
    finish(CalibSet {
        schema: CALIB_SCHEMA.into(),
        id: id.into(),
        version: 1,
        kind: SetKind::Generic,
        platform: None,
        parameters: params,
        fit: Some(fit_info(fits, split, all_test, bootstrap)),
        range_policy: None,
        acceptance: None,
        test_access_log: vec![],
        created: CREATED.into(),
        hash: None,
    })
}
