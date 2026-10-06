//! The M2 calibration campaign: which sets are fitted on which devices and evaluated on which (06 §3.4), and
//! the file layout (`kiln/calibration/sets/<id>.json`, `kiln/calibration/splits/<id>.json`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kiln_ir::common::Diagnostic;
use kiln_sim::CalibSet;
use serde_json::json;

use crate::fit::{DeviceFit, Fitter};
use crate::predict::Bench;
use crate::records::{self, DEVICES, Device, Record, Split};
use crate::report::Role;
use crate::split::{self, SplitFile};

pub const SPLIT_ID: &str = "m2-2026-10-05";

/// `(set id, fit devices, evaluated devices with their role)`.
pub type PlanRow = (&'static str, &'static [&'static str], &'static [(&'static str, Role)]);

pub const PLAN: &[PlanRow] = &[
    ("platform:a100_40gb", &["a100_40gb"], &[("a100_40gb", Role::Fit)]),
    // v6e has its own platform set (loop mode, chip-state gated) and stays out of the generic sets (08 §F).
    ("platform:tpu_v6e", &["tpu_v6e"], &[("tpu_v6e", Role::Fit)]),
    ("generic-v1", &["a100_40gb"], &[("tpu_v5e", Role::HeldOut), ("tpu_v6e", Role::HeldOut)]),
    ("generic-v2", &["a100_40gb", "tpu_v5e"], &[("tpu_v5e", Role::Fit), ("tpu_v6e", Role::HeldOut)]),
];

pub fn calib_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../calibration")
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub bootstrap: usize,
    pub stages: Vec<String>,
    pub threads: Option<usize>,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions { bootstrap: crate::fit::BOOTSTRAP, stages: ["P", "L", "D", "U"].map(String::from).to_vec(), threads: None }
    }
}

pub struct Campaign {
    pub records: BTreeMap<String, Vec<Record>>,
    pub split: SplitFile,
}

impl Campaign {
    /// Every device's records with splits assigned against the union of all test families.
    pub fn load(salt: &str) -> Result<Campaign, Diagnostic> {
        let mut all = vec![];
        for d in DEVICES {
            all.extend(records::load_device(d)?);
        }
        let tests = split::test_families(&all);
        split::assign(&mut all, &tests, salt);
        let sp = split::split_file(SPLIT_ID, salt, &all);
        let mut records: BTreeMap<String, Vec<Record>> = BTreeMap::new();
        for r in all {
            records.entry(r.device.clone()).or_default().push(r);
        }
        Ok(Campaign { records, split: sp })
    }

    pub fn device_fit(&self, dev: &Device, opts: &FitOptions) -> Result<DeviceFit, Diagnostic> {
        let bench = Bench::new(dev.design)?;
        let recs = &self.records[dev.id];
        let mut f = Fitter::new(&bench, recs);
        f.bootstrap = opts.bootstrap;
        f.stages.clone_from(&opts.stages);
        if let Some(t) = opts.threads {
            f.threads = t.max(1);
        }
        f.run()
    }

    pub fn test_records(&self, devs: &[&str]) -> Vec<&Record> {
        devs.iter().flat_map(|d| self.records[*d].iter().filter(|r| r.split == Split::Test)).collect()
    }
}

/// Fits every device a set in `ids` needs and assembles the sets.
pub fn fit_sets(c: &Campaign, ids: &[&str], opts: &FitOptions) -> Result<(Vec<CalibSet>, BTreeMap<String, DeviceFit>), Diagnostic> {
    let bootstrap = opts.bootstrap;
    let mut fits: BTreeMap<String, DeviceFit> = BTreeMap::new();
    let mut sets = vec![];
    for (id, fit_devs, eval) in PLAN.iter().filter(|p| ids.contains(&p.0)) {
        for d in *fit_devs {
            if !fits.contains_key(*d) {
                fits.insert(d.to_string(), c.device_fit(records::device(d).expect("device"), opts)?);
            }
        }
        let fs: Vec<&DeviceFit> = fit_devs.iter().map(|d| &fits[*d]).collect();
        let set = if id.starts_with("platform:") {
            crate::sets::platform_set(fs[0], &c.split, bootstrap)
        } else {
            let devs: Vec<&str> = fit_devs.iter().copied().chain(eval.iter().map(|e| e.0)).collect();
            crate::policy::with_policy(crate::sets::generic_set(id, &fs, &c.test_records(&devs), &c.split, bootstrap))?
        };
        sets.push(set);
    }
    Ok((sets, fits))
}

pub fn set_path(id: &str) -> PathBuf {
    kiln_sim::calib::sets_dir().join(format!("{}.json", crate::sets::file_id(id)))
}

pub fn split_path(id: &str) -> PathBuf {
    calib_root().join("splits").join(format!("{id}.json"))
}

/// Writes sets (keeping an existing file's report fields when its hash is unchanged) and the split.
pub fn write(sets: &[CalibSet], sp: &SplitFile) -> Result<Vec<PathBuf>, Diagnostic> {
    let io = |e: std::io::Error| Diagnostic::error(kiln_sim::calib::CAL_CODE, e.to_string());
    let mut out = vec![];
    for s in sets {
        let p = set_path(&s.id);
        std::fs::create_dir_all(p.parent().expect("dir")).map_err(io)?;
        let mut s = s.clone();
        if let Ok(old) = CalibSet::load(&p) {
            // A refit after test data was read is a new version; the access history carries over so
            // iteration on test data stays visible (06 §3.1 rule 4). Content is compared at the old version
            // number: a fresh set starts at version 1, which alone would change the hash.
            let same = {
                let mut t = s.clone();
                t.version = old.version;
                t.compute_hash() == old.compute_hash()
            };
            s.version = if !same && !old.test_access_log.is_empty() { s.version.max(old.version + 1) } else { s.version.max(old.version) };
            if same {
                s.acceptance = old.acceptance;
            }
            s.test_access_log = old.test_access_log;
        }
        std::fs::write(&p, s.to_json()).map_err(io)?;
        out.push(p);
    }
    let p = split_path(&sp.id);
    std::fs::create_dir_all(p.parent().expect("dir")).map_err(io)?;
    std::fs::write(&p, serde_json::to_string_pretty(sp).expect("split serializes") + "\n").map_err(io)?;
    out.push(p);
    Ok(out)
}

/// Re-derives the range policy of written generic sets (all planned ones by default) without refitting: only
/// `range_policy` changes, so central values and fitted ranges stay as they are.
pub fn attach_policies(ids: &[&str]) -> Result<Vec<(CalibSet, PathBuf)>, Diagnostic> {
    let mut out = vec![];
    for id in ids {
        let path = set_path(id);
        let set = CalibSet::load(&path)?;
        if set.kind != kiln_sim::calib::SetKind::Generic {
            continue;
        }
        let new = crate::policy::with_policy(set.clone())?;
        if new.compute_hash() == set.compute_hash() {
            out.push((new, path));
            continue;
        }
        let mut s = new;
        // New content after test reads is a new version; the access history carries over (06 §3.1 rule 4).
        if !s.test_access_log.is_empty() {
            s.version += 1;
        }
        s.acceptance = None;
        std::fs::write(&path, s.to_json()).map_err(|e| Diagnostic::error(kiln_sim::calib::CAL_CODE, e.to_string()))?;
        out.push((s, path));
    }
    Ok(out)
}

/// Loads a set by id or path (`platform:a100_40gb`, `generic-v1`, or a file).
pub fn load_set(spec: &str) -> Result<CalibSet, Diagnostic> {
    let p = Path::new(spec);
    if p.is_file() { CalibSet::load(p) } else { CalibSet::by_id(&crate::sets::file_id(spec)) }
}

/// Evaluates a set on its planned devices (or `devices`), and appends a `test_access_log` entry to its file.
pub fn report(spec: &str, devices: Option<&[String]>, log: bool) -> Result<(crate::report::Report, usize), Diagnostic> {
    let set = Arc::new(load_set(spec)?);
    let plan: Vec<(&Device, Role)> = match devices {
        Some(ds) => ds
            .iter()
            .map(|d| {
                let dev = records::device(d).ok_or_else(|| Diagnostic::error(records::MEAS_CODE, format!("unknown device {d:?}")))?;
                let fit = set.fit.as_ref().is_some_and(|f| f.fit_devices.iter().any(|x| x == d)) || set.platform.as_deref() == Some(d.as_str());
                Ok((dev, if fit { Role::Fit } else { Role::HeldOut }))
            })
            .collect::<Result<_, Diagnostic>>()?,
        None => PLAN
            .iter()
            .find(|p| p.0 == set.id)
            .map(|p| p.2.iter().map(|(d, r)| (records::device(d).expect("device"), *r)).collect())
            .unwrap_or_default(),
    };
    let r = crate::report::report(&set, &plan)?;
    let current = set.compute_hash();
    let count = |log: &[serde_json::Value]| log.iter().filter(|e| e["set_hash"] == current.as_str()).count();
    let mut accesses = count(&set.test_access_log);
    if log {
        let path = if Path::new(spec).is_file() { PathBuf::from(spec) } else { set_path(&set.id) };
        let mut s = (*set).clone();
        s.test_access_log.push(json!({
            "command": "kiln calibrate report",
            "devices": plan.iter().map(|p| p.0.id).collect::<Vec<_>>(),
            "set_hash": s.compute_hash(),
            "kiln_version": kiln_trace::KILN_VERSION,
        }));
        s.acceptance = Some(crate::report::snapshot(&r));
        accesses = count(&s.test_access_log);
        std::fs::write(&path, s.to_json()).map_err(|e| Diagnostic::error(kiln_sim::calib::CAL_CODE, e.to_string()))?;
    }
    Ok((r, accesses))
}
