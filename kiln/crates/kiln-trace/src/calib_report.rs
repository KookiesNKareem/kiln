//! Calibration report table (05 §3.9): predicted vs measured per benchmark op and device, joined on 06's
//! BenchOp key. Read by the calibration view; also convertible from `kiln calibrate report --format json`.

use std::path::Path;

use arrow_array::RecordBatch;
use kiln_ir::common::Diagnostic;
use serde_json::Value;

use crate::arrowx::{self, Table};

pub const TABLE: &str = "calibration_report";
pub const VERSION: &str = "1.0";

#[derive(Clone, Debug, PartialEq)]
pub struct CalibRow {
    pub device: String,
    /// 06 BenchOp key (`bop1-...`); the op name when the source does not carry keys.
    pub bench_key: String,
    pub name: String,
    pub phase: Option<String>,
    pub op_kind: String,
    pub shape: Option<String>,
    pub precision: Option<String>,
    pub timing_mode: Option<String>,
    pub measured_s: f64,
    /// Cross-session coefficient of variation (noise floor).
    pub measured_cv: Option<f64>,
    pub n_reps: Option<u32>,
    pub predicted_s_tier_a: f64,
    pub predicted_s_tier_b: Option<f64>,
    pub uncalibrated_s: Option<f64>,
    /// `fit` or `test` (06 §3.4).
    pub split: String,
    pub evidence_grade: Option<String>,
    pub session_hash: Option<String>,
    pub source: Option<String>,
    pub calibration_set_hash: String,
    pub count: Option<u64>,
    pub flops: Option<f64>,
    pub bytes: Option<f64>,
}

impl CalibRow {
    /// `log2(predicted / measured)` at Tier A.
    pub fn log2_err(&self) -> f64 {
        (self.predicted_s_tier_a / self.measured_s).log2()
    }
}

pub fn to_batch(rows: &[CalibRow]) -> RecordBatch {
    use arrowx::*;
    let s = |f: fn(&CalibRow) -> &str| utf8s(rows.iter().map(f));
    let o = |f: fn(&CalibRow) -> Option<&str>| utf8(rows.iter().map(f));
    let mut c = Cols::new();
    c.push("device", false, s(|r| &r.device))
        .push("bench_key", false, s(|r| &r.bench_key))
        .push("name", false, s(|r| &r.name))
        .push("phase", true, o(|r| r.phase.as_deref()))
        .push("op_kind", false, s(|r| &r.op_kind))
        .push("shape", true, o(|r| r.shape.as_deref()))
        .push("precision", true, o(|r| r.precision.as_deref()))
        .push("timing_mode", true, o(|r| r.timing_mode.as_deref()))
        .push("measured_s", false, f64v(rows.iter().map(|r| r.measured_s)))
        .push(
            "measured_cv",
            true,
            f64s(rows.iter().map(|r| r.measured_cv)),
        )
        .push("n_reps", true, u32s(rows.iter().map(|r| r.n_reps)))
        .push(
            "predicted_s_tier_a",
            false,
            f64v(rows.iter().map(|r| r.predicted_s_tier_a)),
        )
        .push(
            "predicted_s_tier_b",
            true,
            f64s(rows.iter().map(|r| r.predicted_s_tier_b)),
        )
        .push(
            "uncalibrated_s",
            true,
            f64s(rows.iter().map(|r| r.uncalibrated_s)),
        )
        .push("split", false, s(|r| &r.split))
        .push("evidence_grade", true, o(|r| r.evidence_grade.as_deref()))
        .push("session_hash", true, o(|r| r.session_hash.as_deref()))
        .push("source", true, o(|r| r.source.as_deref()))
        .push(
            "calibration_set_hash",
            false,
            s(|r| &r.calibration_set_hash),
        )
        .push("count", true, u64s(rows.iter().map(|r| r.count)))
        .push("flops", true, f64s(rows.iter().map(|r| r.flops)))
        .push("bytes", true, f64s(rows.iter().map(|r| r.bytes)));
    c.batch(TABLE, VERSION)
}

pub fn from_batches(b: &[RecordBatch]) -> Result<Vec<CalibRow>, Diagnostic> {
    let t = Table::new(TABLE, b);
    let device = t.str("device")?;
    let key = t.str("bench_key")?;
    let name = t.opt_str("name")?;
    let phase = t.opt_str("phase")?;
    let kind = t.opt_str("op_kind")?;
    let shape = t.opt_str("shape")?;
    let prec = t.opt_str("precision")?;
    let mode = t.opt_str("timing_mode")?;
    let meas = t.f64("measured_s")?;
    let cv = t.opt_f64("measured_cv")?;
    let reps = t.opt_u32("n_reps")?;
    let pa = t.f64("predicted_s_tier_a")?;
    let pb = t.opt_f64("predicted_s_tier_b")?;
    let un = t.opt_f64("uncalibrated_s")?;
    let split = t.opt_str("split")?;
    let grade = t.opt_str("evidence_grade")?;
    let sess = t.opt_str("session_hash")?;
    let src = t.opt_str("source")?;
    let set = t.opt_str("calibration_set_hash")?;
    let count = t.opt_u64("count")?;
    let flops = t.opt_f64("flops")?;
    let bytes = t.opt_f64("bytes")?;
    Ok((0..t.rows())
        .map(|i| CalibRow {
            device: device[i].clone(),
            name: name[i].clone().unwrap_or_else(|| key[i].clone()),
            bench_key: key[i].clone(),
            phase: phase[i].clone(),
            op_kind: kind[i].clone().unwrap_or_default(),
            shape: shape[i].clone(),
            precision: prec[i].clone(),
            timing_mode: mode[i].clone(),
            measured_s: meas[i],
            measured_cv: cv[i],
            n_reps: reps[i],
            predicted_s_tier_a: pa[i],
            predicted_s_tier_b: pb[i],
            uncalibrated_s: un[i],
            split: split[i].clone().unwrap_or_else(|| "fit".into()),
            evidence_grade: grade[i].clone(),
            session_hash: sess[i].clone(),
            source: src[i].clone(),
            calibration_set_hash: set[i].clone().unwrap_or_default(),
            count: count[i],
            flops: flops[i],
            bytes: bytes[i],
        })
        .collect())
}

/// Rows from the JSON of `kiln calibrate report --format json` (an array of reports, or one report):
/// each device's per-op rows; `role: fit` devices give `split = fit`, held-out devices `test`.
pub fn from_calibrate_json(v: &Value) -> Vec<CalibRow> {
    let reports: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        o => vec![o],
    };
    let mut out = vec![];
    for r in reports {
        let set = r["set_hash"].as_str().unwrap_or_default().to_string();
        for d in r["devices"].as_array().into_iter().flatten() {
            let device = d["device"].as_str().unwrap_or_default().to_string();
            let split = if d["role"].as_str() == Some("fit") {
                "fit"
            } else {
                "test"
            };
            for o in d["ops"].as_array().into_iter().flatten() {
                let name = o["name"].as_str().unwrap_or_default().to_string();
                let (Some(meas), Some(pred)) = (o["meas_s"].as_f64(), o["cal_s"].as_f64()) else {
                    continue;
                };
                out.push(CalibRow {
                    device: device.clone(),
                    bench_key: name.clone(),
                    shape: name.rsplit('/').next().map(String::from),
                    name,
                    phase: o["phase"].as_str().map(String::from),
                    op_kind: o["class"].as_str().unwrap_or_default().to_string(),
                    precision: None,
                    timing_mode: None,
                    measured_s: meas,
                    measured_cv: None,
                    n_reps: None,
                    predicted_s_tier_a: pred,
                    predicted_s_tier_b: None,
                    uncalibrated_s: o["uncal_s"].as_f64(),
                    split: split.into(),
                    evidence_grade: None,
                    session_hash: None,
                    source: Some("kiln calibrate report".into()),
                    calibration_set_hash: set.clone(),
                    count: o["count"].as_u64(),
                    flops: o["flops"].as_f64(),
                    bytes: o["bytes"].as_f64(),
                });
            }
        }
    }
    out
}

/// Reads an Arrow calibration report or `kiln calibrate report` JSON; a directory reads
/// `calibration_report.arrow` inside it.
pub fn load(path: &Path) -> Result<Vec<CalibRow>, Diagnostic> {
    let p = if path.is_dir() {
        path.join(format!("{TABLE}.arrow"))
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(&p).map_err(|e| {
        Diagnostic::error(
            "E-CALIB-REPORT",
            format!("cannot read {}: {e}", p.display()),
        )
        .hint(
            "pass a calibration_report.arrow or the JSON of `kiln calibrate report --format json`",
        )
    })?;
    if bytes.first().is_some_and(|b| *b == b'[' || *b == b'{') {
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| {
            Diagnostic::error(
                "E-CALIB-REPORT",
                format!("{} is not JSON: {e}", p.display()),
            )
        })?;
        return Ok(from_calibrate_json(&v));
    }
    let (b, _) = arrowx::read_ipc(&bytes)?;
    from_batches(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_to_rows_to_arrow_and_back() {
        let v: Value = serde_json::json!([{"set_hash": "cal1-x", "devices": [
            {"device": "a100_40gb", "role": "fit", "ops": [
                {"name": "prefill_b1/gemm_1x2x3", "phase": "prefill_b1", "class": "gemm_compute", "count": 32,
                 "meas_s": 2e-4, "cal_s": 1.8e-4, "uncal_s": 1.5e-4, "bytes": 1e6, "flops": 1e9}]},
            {"device": "tpu_v6e", "role": "held_out", "ops": [
                {"name": "decode_b1/gemv", "class": "gemv", "meas_s": 1e-5, "cal_s": 1.2e-5}]}]}]);
        let rows = from_calibrate_json(&v);
        assert_eq!(rows.len(), 2);
        assert_eq!(
            (rows[0].split.as_str(), rows[1].split.as_str()),
            ("fit", "test")
        );
        assert!((rows[0].log2_err() - (0.9f64).log2()).abs() < 1e-12);
        let (b, _) = arrowx::read_ipc(&arrowx::to_ipc_file(&to_batch(&rows))).unwrap();
        assert_eq!(from_batches(&b).unwrap(), rows);
    }
}
