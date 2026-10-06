//! Measurement records of the 2026-10-05 runner sessions (calib-micro, per-op suite, whole steps), each with
//! its device, timing mode, telemetry clock, content hash and split (06 §3.4).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kiln_ir::bench::BenchOp;
use kiln_ir::common::{Diagnostic, content_hash};
use kiln_trace::meas::runner::{self, ChipState, GPU_OP_MODES, TPU_OP_MODES};
use serde::Serialize;
use serde_json::{Value, json};

pub const MEAS_CODE: &str = "E-CAL-MEAS-001";

/// A device of the calibration campaign and the files and modes that define its canonical numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    /// Platform key (`a100_40gb`, `tpu_v5e`, `tpu_v6e`).
    pub id: &'static str,
    pub design: &'static str,
    pub micro: &'static [&'static str],
    pub suite: &'static [&'static str],
    pub seq: &'static [&'static str],
    /// Canonical per-op timing modes (minimum over them), 06 §4.3; TPU: `loop` only (08 §F).
    pub op_modes: &'static [&'static str],
    pub step_modes: &'static [&'static str],
    /// Whole-step attention implementations counted as best software (empty = every checked one).
    pub attn: &'static [&'static str],
    /// Session pairs measuring the same records, for the cross-session noise floor.
    pub noise_pairs: &'static [(&'static str, &'static str)],
}

pub const DEVICES: &[Device] = &[
    Device {
        id: "a100_40gb",
        design: "a100_sxm4_40gb.json5",
        micro: &["a100_2026-10-05_micro_r2.json"],
        suite: &["a100_2026-10-05_suite_r2.json"],
        seq: &["a100_2026-10-05_seq.json", "a100_2026-10-05_seq_r2.json"],
        op_modes: GPU_OP_MODES,
        step_modes: &["graph"],
        attn: &["sdpa"],
        noise_pairs: &[
            ("a100_2026-10-05_suite.json", "a100_2026-10-05_suite_r2.json"),
            ("a100_2026-10-05_seq.json", "a100_2026-10-05_seq_r2.json"),
        ],
    },
    Device {
        id: "tpu_v5e",
        design: "tpu_v5e.json5",
        micro: &["tpuv5e_2026-10-05_micro.json"],
        suite: &["tpuv5e_2026-10-05_suite.json"],
        seq: &["tpuv5e_2026-10-05_seq_fused.json"],
        op_modes: TPU_OP_MODES,
        step_modes: &[],
        attn: &[],
        noise_pairs: &[("tpuv5e_2026-10-05_seq.json", "tpuv5e_2026-10-05_seq_fused.json")],
    },
    Device {
        id: "tpu_v6e",
        design: "tpu_v6e.json5",
        micro: &["tpuv6e_2026-10-05_micro.json"],
        suite: &["tpuv6e_2026-10-05_suite.json"],
        seq: &["tpuv6e_2026-10-05_seq_fused.json"],
        op_modes: TPU_OP_MODES,
        step_modes: &[],
        attn: &[],
        noise_pairs: &[("tpuv6e_2026-10-05_seq.json", "tpuv6e_2026-10-05_seq_fused.json")],
    },
];

pub fn device(id: &str) -> Option<&'static Device> {
    DEVICES.iter().find(|d| d.id == id)
}

/// `calibration/measurements` in the source tree (or `$KILN_MEAS_DIR`).
pub fn measurements_dir() -> PathBuf {
    std::env::var_os("KILN_MEAS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../calibration/measurements"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamOp {
    Copy,
    Read,
    Write,
    Add,
    Scale,
    Silu,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Kind {
    Contraction { op: BenchOp },
    /// `elems` bf16 elements per input operand.
    Stream { op: StreamOp, elems: u64 },
    /// `chain` back-to-back copies of a tiny kernel (`empty`, `add1`, or a square bf16 matmul of side `side`).
    Launch { kernel: String, side: u64, chain: u64 },
    Step { phase: String, layers: u64, attn: String },
    /// Sustained clock and power under a step load (telemetry only).
    PowerStep,
    /// Operands resident on chip (L2 / VMEM sweeps): no registered parameter describes them.
    OnChip,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Fit,
    /// Fit-suite record held out by the diagnostic family split (06 §3.4).
    FitHoldout,
    Test,
    Excluded(String),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Record {
    pub device: String,
    pub session: String,
    pub name: String,
    pub group: String,
    pub kind: Kind,
    pub split: Split,
    /// Canonical seconds (minimum over the device's canonical modes) and the mode it came from.
    pub meas_s: f64,
    pub mode: String,
    /// Mean core clock over the timed window (NVML), when measured.
    pub clock_hz: Option<f64>,
    pub flops: f64,
    pub bytes: f64,
    pub phase: Option<String>,
    pub count: u64,
    pub quality_ok: bool,
    /// Chip-state gate (08 §F): set on compute-bound records of a TPU session whose sanity GEMM changed state
    /// between start and end; such records never enter a fit.
    pub chip_state: Option<String>,
    /// `rec2-` content hash of (session, the complete runner record, mode, seconds): any change to what the
    /// record measured or describes (dims, layers, attention, modes, quality) changes it.
    pub hash: String,
}

impl Record {
    /// 06 §3.4 family key: (op kind, m bucket, n, k, batch bucket, dtype); `None` for non-contractions.
    pub fn family(&self) -> Option<String> {
        match &self.kind {
            Kind::Contraction { op } => Some(family_of(op)),
            _ => None,
        }
    }

    /// Family key used by the diagnostic hold-out (any kind).
    pub fn diag_family(&self) -> String {
        match &self.kind {
            Kind::Contraction { op } => family_of(op),
            Kind::Stream { op, elems } => format!("stream/{op:?}/{elems}"),
            Kind::Launch { kernel, side, .. } => format!("launch/{kernel}/{side}"),
            other => format!("{other:?}"),
        }
    }
}

pub fn bucket(x: u64) -> &'static str {
    match x {
        0 | 1 => "1",
        2..=16 => "2-16",
        17..=128 => "17-128",
        _ => ">128",
    }
}

pub fn family_of(op: &BenchOp) -> String {
    let d = |k: &str| op.dim(k).unwrap_or(1);
    let kind = if op.kind == kiln_ir::bench::BenchKind::Bmm { "bmm" } else { "gemm" };
    format!("{kind}|{}|{}|{}|{}|bf16", bucket(d("m")), d("n"), d("k"), bucket(d("batch")))
}

fn rec_hash(session: &str, record: &Value, mode: &str, s: f64) -> String {
    content_hash("rec2-", &json!({"session": session, "record": record, "mode": mode, "s": s}))
}

/// Chip state of a TPU session file (`None` for GPU devices: the gate is a TPU ruling).
pub fn session_state(dev: &Device, file: &str) -> Result<Option<ChipState>, Diagnostic> {
    if dev.id.starts_with("a100") {
        return Ok(None);
    }
    let p = measurements_dir().join(file);
    let raw = std::fs::read(&p).map_err(|e| Diagnostic::error(MEAS_CODE, format!("cannot read {}: {e}", p.display())))?;
    runner::chip_state(&raw, file).map(Some)
}

/// Chip state of every TPU session file of a device (calib-micro, suite, sequence), in that order.
pub fn chip_states(dev: &Device) -> Result<Vec<(String, ChipState)>, Diagnostic> {
    let mut out = vec![];
    for f in dev.micro.iter().chain(dev.suite).chain(dev.seq) {
        if let Some(st) = session_state(dev, f)? {
            out.push((f.to_string(), st));
        }
    }
    Ok(out)
}

/// The gate's verdict for one record: compute-bound (roofline at the session's spec peaks: FLOP time >= byte
/// time) in a session whose chip state changed.
fn gate(state: Option<&ChipState>, header: &Value, flops: f64, bytes: f64) -> Option<String> {
    let st = state.filter(|s| s.changed())?;
    let (peak, bw) = (header["device"]["spec_bf16_flops"].as_f64()?, header["device"]["spec_hbm_bps"].as_f64()?);
    (flops > 0.0 && flops / peak >= bytes / bw).then(|| st.summary())
}

fn read_json(file: &str) -> Result<Value, Diagnostic> {
    let p = measurements_dir().join(file);
    let raw = std::fs::read(&p).map_err(|e| Diagnostic::error(MEAS_CODE, format!("cannot read {}: {e}", p.display())))?;
    serde_json::from_slice(&raw).map_err(|e| Diagnostic::error(MEAS_CODE, format!("{file}: {e}")))
}

/// Canonical value: minimum median over `modes` present, with that mode's clock (MHz -> Hz).
fn canonical(r: &Value, modes: &[&str]) -> Option<(f64, String, Option<f64>)> {
    let mut best: Option<(f64, String, Option<f64>)> = None;
    for (k, m) in r["modes"].as_object().into_iter().flatten() {
        if !modes.contains(&k.as_str()) {
            continue;
        }
        let Some(t) = m["median_s"].as_f64() else { continue };
        if best.as_ref().is_none_or(|b| t < b.0) {
            best = Some((t, k.clone(), m["clock"]["sm_mhz"].as_f64().map(|x| x * 1e6)));
        }
    }
    best
}

fn stream_op(kind: &str, op: Option<&str>) -> Option<StreamOp> {
    Some(match (kind, op) {
        ("copy", _) => StreamOp::Copy,
        ("read_reduce", _) => StreamOp::Read,
        ("write", _) => StreamOp::Write,
        ("elementwise", Some("add")) => StreamOp::Add,
        ("elementwise", Some("scale")) => StreamOp::Scale,
        ("elementwise", Some("silu")) => StreamOp::Silu,
        _ => return None,
    })
}

/// calib-micro records of one session: fit candidates (splits assigned later by [`crate::split`]).
pub fn micro(dev: &Device, file: &str) -> Result<Vec<Record>, Diagnostic> {
    let v = read_json(file)?;
    let gpu = dev.id.starts_with("a100");
    let state = session_state(dev, file)?;
    let mut out = vec![];
    for r in v["records"].as_array().into_iter().flatten() {
        let name = r["name"].as_str().unwrap_or_default().to_string();
        let group = r["group"].as_str().unwrap_or_else(|| name.split('/').next().unwrap_or_default()).to_string();
        let kind_s = r["kind"].as_str().unwrap_or_default();
        let ok = r["quality"].as_str() == Some("ok");
        let d = &r["dims"];
        let dim = |k: &str| d[k].as_u64();
        let kind = match kind_s {
            "gemm" | "linear" | "bmm" => {
                let (Some(m), Some(n), Some(k)) = (dim("m"), dim("n"), dim("k")) else { continue };
                let op = match kind_s {
                    "bmm" => BenchOp::bmm(dim("batch").unwrap_or(1), m, n, k),
                    k2 => BenchOp::gemm(m, n, k, k2 == "linear"),
                };
                Kind::Contraction { op }
            }
            "launch_probe" => {
                let kernel = d["kernel"].as_str().unwrap_or("empty_jit").to_string();
                let side = kernel.rsplit('_').next().and_then(|x| x.parse().ok()).unwrap_or(0);
                let (kernel, side) = match kernel.as_str() {
                    "sleep0" | "empty_jit" => ("empty".to_string(), 0),
                    "add_1elem" => ("add1".into(), 0),
                    _ => ("matmul".into(), side),
                };
                Kind::Launch { kernel, side, chain: dim("chain").unwrap_or(1) }
            }
            "power_step" => Kind::PowerStep,
            _ if matches!(group.as_str(), "l2" | "vmem" | "vmem_resident") => Kind::OnChip,
            _ => match stream_op(kind_s, d["op"].as_str()) {
                Some(op) => {
                    let bytes = dim("bytes").or_else(|| dim("numel").map(|n| 2 * n)).unwrap_or(0);
                    Kind::Stream { op, elems: bytes / 2 }
                }
                None => continue,
            },
        };
        // GPU launch probes: per-kernel graph-chain time; TPU: per-loop-iteration chain time.
        let modes: &[&str] = match (&kind, gpu) {
            (Kind::Launch { .. }, true) => &["graph_chain"],
            (Kind::Launch { .. }, false) => &["loop"],
            (Kind::OnChip, true) => &["graph_unflushed"],
            (Kind::OnChip, false) => &["loop_r1"],
            _ => dev.op_modes,
        };
        let (meas_s, mode, clock_hz) = match (&kind, canonical(r, modes)) {
            (Kind::PowerStep, _) => (0.0, "steady".into(), r["steady"]["sm_mhz"].as_f64().map(|x| x * 1e6)),
            (_, Some(c)) => c,
            (_, None) => continue,
        };
        let (flops, bytes) = (r["flops"].as_f64().unwrap_or(0.0), r["bytes"].as_f64().unwrap_or(0.0));
        out.push(Record {
            device: dev.id.into(),
            session: file.into(),
            hash: rec_hash(file, r, &mode, meas_s),
            name,
            group,
            kind,
            split: Split::Fit,
            meas_s,
            mode,
            clock_hz,
            flops,
            bytes,
            phase: None,
            count: 1,
            quality_ok: ok,
            chip_state: gate(state.as_ref(), &v, flops, bytes),
        });
    }
    Ok(out)
}

/// Per-op suite records (contractions): test-only, always.
pub fn suite(dev: &Device, file: &str) -> Result<Vec<Record>, Diagnostic> {
    let raw = std::fs::read(measurements_dir().join(file)).map_err(|e| Diagnostic::error(MEAS_CODE, format!("{file}: {e}")))?;
    let v: Value = serde_json::from_slice(&raw).map_err(|e| Diagnostic::error(MEAS_CODE, format!("{file}: {e}")))?;
    let extra: BTreeMap<String, &Value> =
        v["records"].as_array().into_iter().flatten().filter_map(|r| r["name"].as_str().map(|n| (n.to_string(), r))).collect();
    let state = session_state(dev, file)?;
    let mut out = vec![];
    for r in runner::op_records(&raw, file)? {
        let Some(src) = extra.get(&r.name) else { continue };
        let Some((meas_s, mode, clock_hz)) = canonical(src, dev.op_modes) else { continue };
        let (flops, bytes) = (r.op.flops().unwrap_or(0) as f64, r.op.min_bytes().unwrap_or(0) as f64);
        out.push(Record {
            device: dev.id.into(),
            session: file.into(),
            hash: rec_hash(file, src, &mode, meas_s),
            name: format!("{}/{}", r.phase.as_deref().unwrap_or(&r.group), r.name),
            group: r.group.clone(),
            flops,
            bytes,
            chip_state: gate(state.as_ref(), &v, flops, bytes),
            kind: Kind::Contraction { op: r.op },
            split: Split::Test,
            meas_s,
            mode,
            clock_hz,
            phase: r.phase,
            count: r.count,
            quality_ok: src["quality"].as_str().is_none_or(|q| q == "ok"),
        });
    }
    Ok(out)
}

/// Best-software whole steps per phase (fastest checked implementation): test-only, always.
pub fn steps(dev: &Device) -> Result<Vec<Record>, Diagnostic> {
    steps_in(dev, &measurements_dir())
}

fn steps_in(dev: &Device, dir: &Path) -> Result<Vec<Record>, Diagnostic> {
    let mut best: BTreeMap<String, Record> = BTreeMap::new();
    for file in dev.seq {
        let raw = std::fs::read(dir.join(file)).map_err(|e| Diagnostic::error(MEAS_CODE, format!("{file}: {e}")))?;
        let v: Value = serde_json::from_slice(&raw).map_err(|e| Diagnostic::error(MEAS_CODE, format!("{file}: {e}")))?;
        let src: BTreeMap<&str, &Value> =
            v["records"].as_array().into_iter().flatten().filter_map(|r| r["name"].as_str().map(|n| (n, r))).collect();
        for s in runner::step_records(&raw, file)? {
            if s.checked == Some(false) || !s.quality_ok || !(dev.attn.is_empty() || dev.attn.contains(&s.attn.as_str())) {
                continue;
            }
            let Some((mode, t)) = s
                .median_s
                .iter()
                .filter(|(k, _)| dev.step_modes.is_empty() || dev.step_modes.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), *v))
                .min_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            if best.get(&s.phase).is_some_and(|b| b.meas_s <= t) {
                continue;
            }
            best.insert(
                s.phase.clone(),
                Record {
                    device: dev.id.into(),
                    session: (*file).into(),
                    hash: rec_hash(file, src.get(s.name.as_str()).copied().unwrap_or(&Value::Null), &mode, t),
                    name: s.name.clone(),
                    group: "sequence".into(),
                    kind: Kind::Step { phase: s.phase.clone(), layers: s.n_layers, attn: s.attn.clone() },
                    split: Split::Test,
                    meas_s: t,
                    mode,
                    clock_hz: None,
                    flops: 0.0,
                    bytes: 0.0,
                    phase: Some(s.phase.clone()),
                    count: 1,
                    quality_ok: true,
                    chip_state: None,
                },
            );
        }
    }
    Ok(best.into_values().collect())
}

/// Every record of a device: micro (fit candidates), suite and steps (test).
pub fn load_device(dev: &Device) -> Result<Vec<Record>, Diagnostic> {
    let mut out = vec![];
    for f in dev.micro {
        out.extend(micro(dev, f)?);
    }
    for f in dev.suite {
        out.extend(suite(dev, f)?);
    }
    out.extend(steps(dev)?);
    Ok(out)
}

/// Cross-session noise: median |a/b - 1| over records present in both sessions (per-op files), or over
/// whole-step phases (sequence files), by matching name and canonical mode.
pub fn session_noise(dev: &Device, a: &str, b: &str) -> Option<(f64, usize)> {
    let load = |f: &str| -> Option<BTreeMap<String, f64>> {
        let v = read_json(f).ok()?;
        let mut m = BTreeMap::new();
        for r in v["records"].as_array()? {
            let name = r["name"].as_str()?.to_string();
            let modes: Vec<&str> = if r["kind"] == "sequence" { dev.step_modes.to_vec() } else { dev.op_modes.to_vec() };
            let t = if modes.is_empty() {
                r["modes"].as_object()?.values().filter_map(|m| m["median_s"].as_f64()).reduce(f64::min)
            } else {
                canonical(r, &modes).map(|c| c.0)
            };
            if let Some(t) = t {
                m.insert(name, t);
            }
        }
        Some(m)
    };
    let (x, y) = (load(a)?, load(b)?);
    let mut d: Vec<f64> = x.iter().filter_map(|(k, va)| y.get(k).map(|vb| (va / vb - 1.0).abs())).collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    Some((d[d.len() / 2], d.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_dir(tag: &str, records: Value) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kiln-calib-steps-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("seq.json"), json!({ "records": records }).to_string()).unwrap();
        dir
    }

    const DEV: Device = Device {
        id: "test",
        design: "",
        micro: &[],
        suite: &[],
        seq: &["seq.json"],
        op_modes: &[],
        step_modes: &[],
        attn: &[],
        noise_pairs: &[],
    };

    fn step(name: &str, median_s: f64, quality: &str, n_layers: u64) -> Value {
        json!({"name": name, "kind": "sequence", "scope": "step", "phase": "decode_b1", "n_layers": n_layers,
               "attn_impl": "sdpa", "quality": quality, "modes": {"loop": {"median_s": median_s}}})
    }

    #[test]
    fn rejected_steps_are_not_best_software() {
        let dir = seq_dir("q", json!([step("ok", 0.010, "ok", 32), step("fast", 0.001, "rejected", 32)]));
        let best = steps_in(&DEV, &dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(best.len(), 1);
        assert_eq!((best[0].name.as_str(), best[0].meas_s, best[0].quality_ok), ("ok", 0.010, true));
    }

    #[test]
    fn record_hash_covers_workload_semantics() {
        let hash = |n_layers| {
            let dir = seq_dir(&format!("h{n_layers}"), json!([step("s", 0.01, "ok", n_layers)]));
            let h = steps_in(&DEV, &dir).unwrap()[0].hash.clone();
            std::fs::remove_dir_all(&dir).unwrap();
            h
        };
        assert_ne!(hash(32), hash(16));
    }
}
