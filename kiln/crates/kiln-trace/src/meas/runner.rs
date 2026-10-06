//! Reader for the calibration runner's own JSON (`calibration/gpu_bench.py`, `tpu_bench.py`, 2026-10-05 on):
//! per-op suite records with descriptors and per-mode statistics, and whole-step `sequence` records. The legacy
//! importer (`meas::legacy`) stays for the 2026-10-04 files.

use std::collections::BTreeMap;

use kiln_ir::bench::BenchOp;
use kiln_ir::common::Diagnostic;
use serde_json::Value;

/// Canonical per-op timing mode of the CUDA runner (06 §4.3).
pub const GPU_OP_MODES: &[&str] = &["graph_cold"];
/// Canonical per-op timing mode of the TPU runner: in-program `loop`, as scored whole steps run (08 §F, from
/// §F.4). The former canonical `best = min(loop, pipelined_rot)` is no longer used.
pub const TPU_OP_MODES: &[&str] = &["loop"];
/// TPU modes kept as diagnostics only: `pipelined_rot` runs separate programs and on v6e is 20-27% faster than
/// any single program through cross-program effects (08 §F.4).
pub const TPU_DIAGNOSTIC_OP_MODES: &[&str] = &["pipelined_rot"];

/// Relative change of the session sanity GEMM above which a session's chip state counts as changed: the 3%
/// per-record noise gate of 06 §4.3 (v6e's two TensorCore states differ by ~20%, 08 §F.4).
pub const CHIP_STATE_GATE: f64 = 0.03;

/// One sanity measurement (8192^3 bf16 GEMM and 1 GiB copy, `loop` mode) of a TPU session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SanityPoint {
    pub gemm_s: f64,
    pub gemm_tflops: Option<f64>,
    pub copy_s: Option<f64>,
}

/// Chip state of a TPU session from its sanity GEMM at start and end (08 §F chip-state gate).
#[derive(Clone, Debug, PartialEq)]
pub struct ChipState {
    pub start: Option<SanityPoint>,
    pub end: Option<SanityPoint>,
}

impl ChipState {
    /// `end / start - 1` of the sanity GEMM time, when both were recorded.
    pub fn gemm_drift(&self) -> Option<f64> {
        Some(self.end?.gemm_s / self.start?.gemm_s - 1.0)
    }

    pub fn recorded(&self) -> bool {
        self.start.is_some() && self.end.is_some()
    }

    /// The state changed during the session: the two sanity GEMMs differ by more than [`CHIP_STATE_GATE`].
    /// Sessions without both points are not flagged (nothing to compare); callers report them as unrecorded.
    pub fn changed(&self) -> bool {
        self.gemm_drift().is_some_and(|d| d.abs() > CHIP_STATE_GATE)
    }

    pub fn summary(&self) -> String {
        let tf = |p: &Option<SanityPoint>| {
            p.and_then(|p| p.gemm_tflops)
                .map_or("-".into(), |x| format!("{x:.1} TF"))
        };
        match self.gemm_drift() {
            Some(d) => format!(
                "sanity GEMM 8192^3 start {} end {} ({:+.1}%, gate {:.0}%): {}",
                tf(&self.start),
                tf(&self.end),
                100.0 * d,
                100.0 * CHIP_STATE_GATE,
                if self.changed() {
                    "state changed"
                } else {
                    "steady"
                }
            ),
            None => "sanity GEMM start/end not recorded".into(),
        }
    }
}

/// The session's sanity block (`sanity.start` / `sanity.end`, written by `calibration/tpu_bench.py`).
pub fn chip_state(raw: &[u8], file: &str) -> Result<ChipState, Diagnostic> {
    let v = parse(raw, file)?;
    let time = |tag: &str, key: &str| -> Result<Option<f64>, Diagnostic> {
        match &v["sanity"][tag][key]["loop_s"] {
            Value::Null => Ok(None),
            x => match x.as_f64() {
                Some(t) if t.is_finite() && t > 0.0 => Ok(Some(t)),
                _ => Err(Diagnostic::error(
                    "E-TRACE-MEAS-001",
                    format!("{file}: sanity.{tag}.{key}.loop_s {x} is not a positive finite time"),
                )),
            },
        }
    };
    let point = |tag: &str| -> Result<Option<SanityPoint>, Diagnostic> {
        let copy_s = time(tag, "copy_1GiB")?;
        Ok(time(tag, "gemm_8192^3")?.map(|gemm_s| SanityPoint {
            gemm_s,
            gemm_tflops: v["sanity"][tag]["gemm_8192^3"]["tflops"].as_f64(),
            copy_s,
        }))
    };
    Ok(ChipState {
        start: point("start")?,
        end: point("end")?,
    })
}

/// One contraction record of a per-op suite.
#[derive(Clone, Debug, PartialEq)]
pub struct OpRecord {
    pub name: String,
    pub op: BenchOp,
    /// Runner group: `llm_op` (Llama-3-8B ops, with `phase` and `count`) or `legacy_sweep`.
    pub group: String,
    pub phase: Option<String>,
    pub count: u64,
    /// Median seconds per timing mode.
    pub median_s: BTreeMap<String, f64>,
}

impl OpRecord {
    /// Fastest median over `modes` (all modes when empty).
    pub fn best(&self, modes: &[&str]) -> Option<f64> {
        self.median_s
            .iter()
            .filter(|(k, _)| modes.is_empty() || modes.contains(&k.as_str()))
            .map(|x| *x.1)
            .reduce(f64::min)
    }
}

/// One whole-step record (`kind: sequence`, `scope: step`).
#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    pub name: String,
    pub phase: String,
    /// Attention implementation (`sdpa`, `bmm`, `einsum_f32`, `pallas_flash_b512`, ...).
    pub attn: String,
    pub n_layers: u64,
    /// The runner's numerical check of the attention implementation passed (absent = not checked).
    pub checked: Option<bool>,
    /// The runner accepted the timing (`quality` absent or `ok`); rejected records carry `quality_flags`.
    pub quality_ok: bool,
    pub quality_flags: Vec<String>,
    pub median_s: BTreeMap<String, f64>,
}

fn medians(r: &Value, file: &str) -> Result<BTreeMap<String, f64>, Diagnostic> {
    let mut out = BTreeMap::new();
    for (k, m) in r["modes"].as_object().into_iter().flatten() {
        match &m["median_s"] {
            Value::Null => {}
            x => match x.as_f64() {
                Some(t) if t.is_finite() && t > 0.0 => {
                    out.insert(k.clone(), t);
                }
                _ => {
                    return Err(Diagnostic::error(
                        "E-TRACE-MEAS-001",
                        format!(
                            "{file}: record {} mode {k} median_s {x} is not a positive finite time",
                            r["name"]
                        ),
                    ));
                }
            },
        }
    }
    Ok(out)
}

fn parse(raw: &[u8], file: &str) -> Result<Value, Diagnostic> {
    serde_json::from_slice(raw)
        .map_err(|e| Diagnostic::error("E-TRACE-MEAS-001", format!("{file}: {e}")))
}

/// GEMM / linear / BMM records of a per-op suite file, as bf16 bench descriptors.
pub fn op_records(raw: &[u8], file: &str) -> Result<Vec<OpRecord>, Diagnostic> {
    let v = parse(raw, file)?;
    let mut out = vec![];
    for r in v["records"].as_array().into_iter().flatten() {
        let d = &r["dims"];
        let dim = |k: &str| d[k].as_u64();
        let op = match (r["kind"].as_str(), dim("m"), dim("n"), dim("k")) {
            (Some("gemm"), Some(m), Some(n), Some(k)) => BenchOp::gemm(m, n, k, false),
            (Some("linear"), Some(m), Some(n), Some(k)) => BenchOp::gemm(m, n, k, true),
            (Some("bmm"), Some(m), Some(n), Some(k)) => {
                BenchOp::bmm(dim("batch").unwrap_or(1), m, n, k)
            }
            _ => continue,
        };
        if op.flops().is_none() || op.min_bytes().is_none() {
            return Err(Diagnostic::error(
                "E-TRACE-MEAS-001",
                format!(
                    "{file}: record {} dims {d} overflow its FLOP or byte count",
                    r["name"]
                ),
            ));
        }
        out.push(OpRecord {
            name: r["name"].as_str().unwrap_or_default().to_string(),
            op,
            group: r["group"].as_str().unwrap_or_default().to_string(),
            phase: r["phase"].as_str().map(String::from),
            count: r["count"].as_u64().unwrap_or(1),
            median_s: medians(r, file)?,
        });
    }
    Ok(out)
}

/// Whole-step records of a sequence file.
pub fn step_records(raw: &[u8], file: &str) -> Result<Vec<StepRecord>, Diagnostic> {
    let v = parse(raw, file)?;
    v["records"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| {
            r["kind"] == "sequence"
                && r["scope"] == "step"
                && r["n_layers"].as_u64().unwrap_or(0) > 0
        })
        .map(|r| {
            Ok(StepRecord {
                name: r["name"].as_str().unwrap_or_default().to_string(),
                phase: r["phase"].as_str().unwrap_or_default().to_string(),
                attn: r["attn_impl"]
                    .as_str()
                    .or(r["attn"].as_str())
                    .unwrap_or_default()
                    .to_string(),
                n_layers: r["n_layers"].as_u64().unwrap_or(0),
                checked: r["attn_check"]["ok"].as_bool(),
                quality_ok: r["quality"].as_str().is_none_or(|q| q == "ok"),
                quality_flags: r["quality_flags"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|f| f.as_str().map(String::from))
                    .collect(),
                median_s: medians(r, file)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/../../../calibration/measurements/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("measurement file")
    }

    #[test]
    fn invalid_sanity_timings_are_rejected() {
        let raw = |x: &str| {
            format!(
                r#"{{"sanity": {{"start": {{"gemm_8192^3": {{"loop_s": {x}}}}}, "end": {{"gemm_8192^3": {{"loop_s": {x}}}}}}}}}"#
            )
        };
        for x in ["0", "-1.0", "0.0"] {
            assert_eq!(
                chip_state(raw(x).as_bytes(), "f").unwrap_err().code,
                "E-TRACE-MEAS-001",
                "{x}"
            );
        }
        let ok = chip_state(raw("0.5").as_bytes(), "f").unwrap();
        assert!(ok.recorded() && !ok.changed());
    }

    #[test]
    fn nonpositive_medians_are_rejected() {
        let raw = |x: &str| {
            format!(
                r#"{{"records": [{{"kind": "sequence", "scope": "step", "n_layers": 32, "phase": "decode_b1", "modes": {{"loop": {{"median_s": {x}}}}}}}, {{"kind": "gemm", "dims": {{"m": 8, "n": 8, "k": 8}}, "modes": {{"loop": {{"median_s": {x}}}}}}}]}}"#
            )
        };
        for x in ["0", "-1.0", "1e999"] {
            assert_eq!(
                step_records(raw(x).as_bytes(), "f").unwrap_err().code,
                "E-TRACE-MEAS-001",
                "{x}"
            );
            assert_eq!(
                op_records(raw(x).as_bytes(), "f").unwrap_err().code,
                "E-TRACE-MEAS-001",
                "{x}"
            );
        }
        let ok = step_records(raw("0.5").as_bytes(), "f").unwrap();
        assert_eq!(ok[0].median_s["loop"], 0.5);
        assert!(ok[0].quality_ok && ok[0].quality_flags.is_empty());
        assert_eq!(
            op_records(raw("0.5").as_bytes(), "f").unwrap()[0].median_s["loop"],
            0.5
        );
    }

    #[test]
    fn unrepresentable_descriptors_are_rejected() {
        let raw = br#"{"records": [{"kind": "gemm", "dims": {"m": 8796093022208, "n": 8796093022208, "k": 8796093022208}, "modes": {"loop": {"median_s": 0.5}}}]}"#;
        assert_eq!(op_records(raw, "f").unwrap_err().code, "E-TRACE-MEAS-001");
    }

    #[test]
    fn step_records_keep_rejection_status() {
        let raw = br#"{"records": [{"kind": "sequence", "scope": "step", "n_layers": 32, "phase": "decode_b1", "quality": "rejected", "quality_flags": ["loop cv 0.5 > 0.03"], "modes": {"loop": {"median_s": 0.001}}}]}"#;
        let r = &step_records(raw, "f").unwrap()[0];
        assert!(!r.quality_ok);
        assert_eq!(r.quality_flags, ["loop cv 0.5 > 0.03"]);
    }

    #[test]
    fn reads_the_2026_10_05_runner_files() {
        let ops = op_records(&file("a100_2026-10-05_suite_r2.json"), "a100").unwrap();
        assert_eq!(ops.iter().filter(|r| r.group == "llm_op").count(), 52);
        let qkv = ops
            .iter()
            .find(|r| r.name == "gemm_2048_6144_4096")
            .unwrap();
        assert_eq!((qkv.phase.as_deref(), qkv.count), (Some("prefill_b1"), 32));
        assert!(qkv.best(&["graph_cold"]).unwrap() > 3e-4);
        let steps = step_records(&file("tpuv6e_2026-10-05_seq_fused.json"), "v6e").unwrap();
        let pre: Vec<&StepRecord> = steps.iter().filter(|s| s.phase == "prefill_b1").collect();
        assert!(
            pre.len() > 5
                && pre
                    .iter()
                    .all(|s| s.n_layers == 32 && s.checked == Some(true))
        );
        let a100 = step_records(&file("a100_2026-10-05_seq_r2.json"), "a100").unwrap();
        assert!(
            a100.iter()
                .any(|s| s.attn == "sdpa" && s.median_s.contains_key("graph"))
        );
    }

    #[test]
    fn chip_state_gate_flags_the_session_that_switched_state() {
        let st = |f: &str| chip_state(&file(f), f).unwrap();
        let diag3 = st("tpuv6e_2026-10-05_diag3_suite.json");
        assert!(
            diag3.changed() && diag3.gemm_drift().unwrap() < -0.15,
            "{}",
            diag3.summary()
        );
        for f in [
            "tpuv6e_2026-10-05_micro.json",
            "tpuv6e_2026-10-05_suite.json",
            "tpuv5e_2026-10-05_micro.json",
        ] {
            let s = st(f);
            assert!(s.recorded() && !s.changed(), "{f}: {}", s.summary());
        }
        let legacy = st("tpuv6e_2026-10-04.json");
        assert!(!legacy.recorded() && !legacy.changed());
        let synth = |a: f64, b: f64| ChipState {
            start: Some(SanityPoint {
                gemm_s: a,
                gemm_tflops: None,
                copy_s: None,
            }),
            end: Some(SanityPoint {
                gemm_s: b,
                gemm_tflops: None,
                copy_s: None,
            }),
        };
        assert!(!synth(1.0, 1.0 + 0.9 * CHIP_STATE_GATE).changed());
        assert!(synth(1.0, 1.0 + 1.1 * CHIP_STATE_GATE).changed());
        assert!(synth(1.0, 1.0 - 1.1 * CHIP_STATE_GATE).changed());
    }
}
