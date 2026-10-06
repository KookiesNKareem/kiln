//! Generic range policy (06 §3.2, 08 §F cross-chip intervals): the ranges a generic set gives designs outside its
//! fit devices, each the span of measured evidence across every measured device. Evidence per parameter: the
//! set's own fitted ranges where a device was fitted, and direct micro-benchmark measurements of devices the set
//! did not fit. Only ranges come from here; central values stay the fitted or prior values.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use kiln_ir::hw::Profile;
use kiln_ir::precision::Precision;
use kiln_sim::calib::{CalParam, CalibSet, ParamStatus, PolicyRange, RangeEvidence, RangePolicy};

use crate::predict::designs_dir;
use crate::records::{self, DEVICES, Device, Kind, Record};

/// Weight-streaming GEMVs count as DRAM evidence once the weights exceed every on-chip level of the devices.
const GEMV_MIN_WEIGHT_BYTES: f64 = 64.0 * 1024.0 * 1024.0;
/// GEMMs this large in every dimension are pipeline-bound (fill, wave quantization negligible).
const GEMM_MIN_DIM: u64 = 2048;

pub const APPLIES_TO: &str = "designs whose family is not in fit.fit_devices (novel designs, held-out chips)";

/// Launch-path terms charged per kernel (or per barrier) vs per program launch.
const PER_KERNEL: &[&str] = &["t_min_kernel", "t_gap", "t_dispatch", "t_sync"];
const PER_PROGRAM: &[&str] = &["t_launch", "t_program"];

fn span(v: &[f64]) -> (f64, f64) {
    (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

fn evidence(device: &str, quantity: &str, values: &[f64], source: &str) -> Option<RangeEvidence> {
    (!values.is_empty()).then(|| {
        let (lower, upper) = span(values);
        RangeEvidence { device: device.into(), quantity: quantity.into(), lower, upper, n: values.len(), source: source.into() }
    })
}

/// Direct measurements of one device from its calib-micro records.
pub struct DeviceEvidence {
    pub dram: Option<RangeEvidence>,
    pub unit: Option<RangeEvidence>,
    pub per_kernel: Option<RangeEvidence>,
    pub per_program: Option<RangeEvidence>,
}

pub fn device_evidence(dev: &Device) -> Result<DeviceEvidence, Diagnostic> {
    let p = kiln_sim::Prepared::from_file(&designs_dir().join(dev.design), Profile::Reference)
        .map_err(|e| e.into_iter().next().unwrap_or_else(|| Diagnostic::error(records::MEAS_CODE, "design does not load")))?;
    let bw = p.view.hw.offchip_bandwidth(None).0;
    let peak = p.view.hw.peak_ops_for(Precision::Bf16, None);
    let mut recs: Vec<Record> = vec![];
    for f in dev.micro {
        recs.extend(records::micro(dev, f)?.into_iter().filter(|r| r.quality_ok && r.meas_s > 0.0));
    }
    let src = dev.micro.join(", ");
    let mut gemv = vec![];
    let mut gemm: Vec<f64> = vec![];
    // (kernel, side) -> [(chain, total seconds per chain)]
    let mut chains: BTreeMap<(String, u64), Vec<(u64, f64)>> = BTreeMap::new();
    let gpu = dev.id.starts_with("a100");
    for r in &recs {
        match &r.kind {
            Kind::Contraction { op } => {
                let d = |k: &str| op.dim(k).unwrap_or(1);
                if r.group == "gemv" && d("m") <= 16 && (2 * d("n") * d("k")) as f64 >= GEMV_MIN_WEIGHT_BYTES {
                    gemv.push(r.bytes / r.meas_s / bw);
                } else if r.group == "gemm" && d("m").min(d("n")).min(d("k")) >= GEMM_MIN_DIM {
                    gemm.push(r.flops / r.meas_s / peak);
                }
            }
            Kind::Launch { kernel, side, chain } => {
                // GPU probes report per-kernel time in the chain, TPU probes the whole chain (records.rs).
                let total = if gpu { r.meas_s * *chain as f64 } else { r.meas_s };
                chains.entry((kernel.clone(), *side)).or_default().push((*chain, total));
            }
            _ => {}
        }
    }
    let mut per_kernel = vec![];
    let mut per_program = vec![];
    for pts in chains.values_mut() {
        pts.sort_by_key(|x| x.0);
        let (first, last) = (pts[0], pts[pts.len() - 1]);
        if first.0 == 1 {
            per_program.push(first.1);
        }
        if last.0 > first.0 {
            per_kernel.push((last.1 - first.1) / (last.0 - first.0) as f64);
        }
    }
    let best = |v: &[f64]| if v.is_empty() { vec![] } else { vec![span(v).1] };
    let fastest = |v: &[f64]| if v.is_empty() { vec![] } else { vec![span(v).0] };
    Ok(DeviceEvidence {
        dram: evidence(
            dev.id,
            "achieved/peak DRAM bandwidth, weight-streaming GEMVs (m <= 16, weights >= 64 MiB)",
            &gemv,
            &src,
        ),
        unit: evidence(dev.id, "best achieved/peak bf16 FLOP/s at nominal clock, GEMMs with m, n, k >= 2048", &best(&gemm), &src),
        per_kernel: evidence(dev.id, "marginal cost per kernel in a launch chain, fastest probe kernel", &fastest(&per_kernel), &src),
        per_program: evidence(dev.id, "single-kernel program/graph launch, fastest probe kernel", &fastest(&per_program), &src),
    })
}

fn fitted(set: &CalibSet, name: &str) -> Vec<RangeEvidence> {
    set.parameters
        .iter()
        .filter(|p: &&CalParam| p.name == name && p.status == ParamStatus::Fit)
        .map(|p| {
            let devices = p.diagnostics.as_ref().map(|d| d.per_device.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
            RangeEvidence {
                device: if devices.is_empty() { "fit".into() } else { devices },
                quantity: format!("{name} {:?} fit, bootstrap range", p.key),
                lower: p.range.lower,
                upper: p.range.upper,
                n: p.diagnostics.as_ref().map_or(0, |d| d.n_records),
                source: p.source.clone().unwrap_or_default(),
            }
        })
        .collect()
}

/// The policy for `set` from its fits and every measured device's direct evidence (the DRAM direct evidence
/// only for devices the set did not fit, whose fitted `eta_res` already describes them).
pub fn range_policy(set: &CalibSet) -> Result<RangePolicy, Diagnostic> {
    let fit_devs: Vec<String> = set.fit.as_ref().map(|f| f.fit_devices.clone()).unwrap_or_default();
    let mut dev_ev = vec![];
    for d in DEVICES {
        dev_ev.push((d.id, device_evidence(d)?));
    }
    let mut ranges = vec![];
    let mut push = |name: &str, evidence: Vec<RangeEvidence>| {
        if evidence.is_empty() {
            return;
        }
        let lower = evidence.iter().map(|e| e.lower).fold(f64::INFINITY, f64::min);
        let upper = evidence.iter().map(|e| e.upper).fold(f64::NEG_INFINITY, f64::max);
        ranges.push(PolicyRange { name: name.into(), lower, upper, evidence });
    };
    let mut eta = fitted(set, "eta_res");
    eta.extend(dev_ev.iter().filter(|(id, _)| !fit_devs.iter().any(|f| f == id)).filter_map(|(_, e)| e.dram.clone()));
    push("eta_res", eta);
    let mut unit = fitted(set, "unit_eff");
    unit.extend(dev_ev.iter().filter_map(|(_, e)| e.unit.clone()));
    push("unit_eff", unit);
    for (terms, pick) in [(PER_KERNEL, 0), (PER_PROGRAM, 1)] {
        for name in terms {
            let mut ev = fitted(set, name);
            ev.extend(dev_ev.iter().filter_map(|(_, e)| if pick == 0 { e.per_kernel.clone() } else { e.per_program.clone() }));
            push(name, ev);
        }
    }
    Ok(RangePolicy { applies_to: APPLIES_TO.into(), ranges })
}

/// `set` with its range policy (re)derived and the hash updated; central values untouched.
pub fn with_policy(mut set: CalibSet) -> Result<CalibSet, Diagnostic> {
    set.range_policy = Some(range_policy(&set)?);
    set.hash = Some(set.compute_hash());
    Ok(set)
}
