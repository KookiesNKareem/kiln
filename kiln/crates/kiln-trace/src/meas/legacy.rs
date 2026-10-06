//! `kiln bench import --legacy`: converts `calibration/gpu_bench.py` and `tpu_bench.py` output into `kiln.meas/1`
//! (06 §4.5). Values are converted to SI units (us -> s, MHz -> Hz) and nothing is fitted or filled in.

use std::collections::BTreeMap;

use kiln_ir::common::Diagnostic;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::runner::{TPU_DIAGNOSTIC_OP_MODES, TPU_OP_MODES};
use super::*;

const MIB: u64 = 1 << 20;
/// `COLD_BYTES` / `ROTATE_BYTES` of both runners; not recorded in the files.
const ROTATE_BYTES: u64 = 512 * MIB;
const CUDA_MODES: [&str; 4] = ["flushed", "unflushed", "graph_unflushed", "graph_cold"];

#[derive(Clone, Debug)]
pub struct LegacyImport {
    pub session: MeasSession,
    pub warnings: Vec<Diagnostic>,
    /// The file used a superseded methodology; the session is imported with `quality: rejected`.
    pub superseded: bool,
}

/// One row of `calibration/oplist.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct OplistEntry {
    pub suite: String,
    pub phase: String,
    pub op: String,
    pub key: String,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub batch: u64,
    pub weight: bool,
    pub count: u64,
    pub tokens: u64,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn import_legacy(
    raw: &[u8],
    file_name: &str,
    oplist: Option<&[u8]>,
) -> Result<LegacyImport, Diagnostic> {
    let root: Value = serde_json::from_slice(raw)
        .map_err(|e| Diagnostic::error(codes::FORMAT, format!("{file_name}: not JSON: {e}")))?;
    let oplist = oplist
        .map(|b| {
            serde_json::from_slice::<Vec<OplistEntry>>(b)
                .map(|o| (o, sha256_hex(b)))
                .map_err(|e| {
                    Diagnostic::error(
                        codes::OPLIST,
                        format!("oplist is not a list of op rows: {e}"),
                    )
                })
        })
        .transpose()?;
    let stem = file_name
        .rsplit('/')
        .next()
        .unwrap_or(file_name)
        .trim_end_matches(".json");
    let mut cx = Cx {
        warnings: Vec::new(),
        source: LegacySource {
            file_name: stem.to_string() + ".json",
            file_sha256: sha256_hex(raw),
            format: String::new(),
            importer: format!("kiln-trace {}", crate::KILN_VERSION),
            oplist_sha256: oplist.as_ref().map(|(_, h)| h.clone()),
            notes: Vec::new(),
        },
    };
    let session_id = format!("legacy-{}", slug(stem));
    let (session, superseded) = if root.get("device_kind").is_some() {
        if cx.source.oplist_sha256.take().is_some() {
            cx.warn(Diagnostic::warning(
                codes::OPLIST,
                "oplist ignored: tpu_bench records carry their own counts",
            ));
        }
        (cx.jax(&root, session_id)?, false)
    } else if root.get("device").is_some_and(Value::is_object) && root.get("ops").is_some() {
        cx.cuda(&root, session_id, stem, oplist.map(|(o, _)| o))?
    } else {
        return Err(Diagnostic::error(
            codes::FORMAT,
            format!("{file_name}: neither gpu_bench nor tpu_bench output"),
        )
        .hint(
            "expected a top-level `device` object (gpu_bench.py) or `device_kind` (tpu_bench.py)",
        ));
    };
    Ok(LegacyImport {
        session: session.seal(),
        warnings: cx.warnings,
        superseded,
    })
}

/// Lowercase `[a-z0-9_.-]` form used for ids and store paths.
pub fn slug(s: &str) -> String {
    s.trim()
        .chars()
        .map(|c| match c.to_ascii_lowercase() {
            c @ ('a'..='z' | '0'..='9' | '_' | '.' | '-') => c,
            _ => '-',
        })
        .collect()
}

fn get<'a>(v: &'a Value, path: &str, key: &str) -> Result<&'a Value, Diagnostic> {
    v.get(key)
        .ok_or_else(|| Diagnostic::error(codes::FIELD, format!("missing field `{key}`")).at(path))
}

fn num(v: &Value, path: &str, key: &str) -> Result<f64, Diagnostic> {
    get(v, path, key)?
        .as_f64()
        .ok_or_else(|| Diagnostic::error(codes::FIELD, format!("`{key}` is not a number")).at(path))
}

fn uint(v: &Value, path: &str, key: &str) -> Result<u64, Diagnostic> {
    get(v, path, key)?.as_u64().ok_or_else(|| {
        Diagnostic::error(codes::FIELD, format!("`{key}` is not an unsigned integer")).at(path)
    })
}

fn opt_num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn opt_str(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn us(x: f64) -> f64 {
    x / 1e6
}

/// First number of an nvidia-smi value such as `"1410 MHz"` or `"400.00 W"`.
fn smi_num(smi: &Value, key: &str) -> Option<f64> {
    smi.get(key)?
        .as_str()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn mib(bytes: f64) -> String {
    format!("{:.1} MiB", bytes / MIB as f64)
}

struct Cx {
    warnings: Vec<Diagnostic>,
    source: LegacySource,
}

impl Cx {
    fn warn(&mut self, d: Diagnostic) {
        self.warnings.push(Diagnostic {
            severity: kiln_ir::common::Severity::Warning,
            ..d
        });
    }

    fn note(&mut self, s: impl Into<String>) {
        self.source.notes.push(s.into());
    }

    fn check_name(&mut self, path: &str, name: &str, op: &BenchOp) {
        if op.legacy_name().is_some_and(|n| n != name) {
            self.warn(
                Diagnostic::warning(
                    codes::FIELD,
                    format!("name {name:?} disagrees with its shape fields"),
                )
                .at(path),
            );
        }
    }

    fn cuda(
        &mut self,
        root: &Value,
        session_id: String,
        stem: &str,
        oplist: Option<Vec<OplistEntry>>,
    ) -> Result<(MeasSession, bool), Diagnostic> {
        self.source.format = "gpu_bench/v0".into();
        let dev = get(root, "$", "device")?;
        let smi = dev.get("smi").cloned().unwrap_or(Value::Null);
        let name = opt_str(&smi, "name")
            .or_else(|| opt_str(dev, "torch_name"))
            .unwrap_or_default();
        let sku = name.trim_start_matches("NVIDIA ").to_string();
        let mhz = |key| smi_num(&smi, key).map(|x| x * 1e6);
        let l2_bytes = dev.get("l2_bytes").and_then(Value::as_u64);
        let device = Device {
            vendor: Vendor::Nvidia,
            sku: sku.clone(),
            device_kind: None,
            pci_device_id: opt_str(&smi, "pci.device_id"),
            vbios: opt_str(&smi, "vbios_version"),
            memory_bytes: smi_num(&smi, "memory.total").map(|x| (x * MIB as f64) as u64),
            core_count: dev
                .get("sm_count")
                .and_then(Value::as_u64)
                .map(|x| x as u32),
            l2_bytes,
            mem_bus_width_bits: dev
                .get("mem_bus_width")
                .and_then(Value::as_u64)
                .map(|x| x as u32),
            compute_capability: opt_str(dev, "cc"),
            chip_count: 1,
            topology: None,
            spec_mem_bw_bps: opt_num(dev, "spec_hbm_gbps").map(|x| x * 1e9),
            spec_bf16_flops: opt_num(dev, "spec_bf16_dense_tflops").map(|x| x * 1e12),
        };
        let clocks_power = ClocksPower {
            sm_max_hz: mhz("clocks.max.sm"),
            mem_max_hz: mhz("clocks.max.mem"),
            sm_observed_hz: mhz("clocks.sm"),
            mem_observed_hz: mhz("clocks.mem"),
            power_limit_w: smi_num(&smi, "power.limit"),
            power_default_limit_w: smi_num(&smi, "power.default_limit"),
            power_max_limit_w: smi_num(&smi, "power.max_limit"),
            ..Default::default()
        };
        let mut software = BTreeMap::new();
        for (k, v) in [
            ("driver", smi.get("driver_version")),
            ("cuda", dev.get("cuda")),
            ("cudnn", dev.get("cudnn")),
            ("cublas_lt", dev.get("cublas_lt")),
            ("torch", dev.get("torch")),
            ("python", dev.get("python")),
            ("nvcc", dev.get("nvcc")),
        ] {
            match v {
                Some(Value::String(s)) if !s.trim().is_empty() => {
                    software.insert(k.to_string(), s.trim().replace('\n', " "));
                }
                Some(Value::Number(n)) => {
                    software.insert(k.to_string(), n.to_string());
                }
                _ => {}
            }
        }
        let flush_elems = root.get("flush_bytes").and_then(Value::as_u64);
        self.note(
            "flush_bytes in the file is FLUSH.numel() of a float32 tensor; converted to bytes (x4)",
        );
        self.note("host.provider, method.rotate_bytes and method.settings are taken from the runner source, not the file");
        let method = Method {
            runner: "calibration/gpu_bench.py".into(),
            warmup: root.get("warmup").and_then(Value::as_u64).map(|x| x as u32),
            iters: root.get("iters").and_then(Value::as_u64).map(|x| x as u32),
            rotate_bytes: Some(ROTATE_BYTES),
            flush_bytes: flush_elems.map(|x| x * 4),
            started: opt_str(root, "started"),
            finished: opt_str(root, "finished"),
            settings: BTreeMap::from([(
                "allow_bf16_reduced_precision_reduction".into(),
                json!(true),
            )]),
            ..Default::default()
        };
        let mode = |description: &str, launch, oh: &[Overhead], operands| TimingMode {
            description: description.into(),
            launch,
            overheads_included: oh.iter().copied().collect(),
            operands,
            trusted: true,
            diagnostic: false,
            derived_from: vec![],
        };
        use Overhead::*;
        let timing_modes = BTreeMap::from([
            (
                "flushed".into(),
                mode(
                    "eager stream, CUDA events around each call, L2 flushed before each call, stream pre-filled with \
                     torch.cuda._sleep so host dispatch is hidden; median of 50",
                    LaunchPath::EagerStream,
                    &[KernelLaunch, InterKernelGap],
                    Residency::ColdDram,
                ),
            ),
            (
                "unflushed".into(),
                mode(
                    "as flushed without the L2 flush; same operands every call",
                    LaunchPath::EagerStream,
                    &[KernelLaunch, InterKernelGap, L2Warm],
                    Residency::L2Warm,
                ),
            ),
            (
                "graph_unflushed".into(),
                mode(
                    "one CUDA graph of 50 calls on the same operands; per-call mean, median over 5 replays",
                    LaunchPath::GraphReplay,
                    &[KernelLaunch, InterKernelGap, L2Warm],
                    Residency::L2Warm,
                ),
            ),
            (
                "graph_cold".into(),
                mode(
                    "one CUDA graph of max(copies, 50) calls rotating over `copies` operand sets (copies = \
                     ceil(512 MiB / op bytes), capped at 64); per-call mean, median over 5 replays",
                    LaunchPath::GraphReplay,
                    &[KernelLaunch, InterKernelGap],
                    Residency::ColdDram,
                ),
            ),
        ]);
        let has_graph_cold = ["ops", "sweep"]
            .iter()
            .filter_map(|k| root.get(*k).and_then(Value::as_array))
            .flatten()
            .any(|r| r.get("graph_cold").is_some());
        let superseded = stem.contains("v0writeflush") || !has_graph_cold;
        let mut quality = Quality::accepted();
        quality.unevaluated = vec![
            "cv <= 3% (per-sample times not recorded)".into(),
            "session start/end sanity drift".into(),
            "repeat on >= 2 instances".into(),
        ];
        if superseded {
            quality
                .reject("superseded methodology: v0 write-flush run without the graph_cold mode");
            self.warn(
                Diagnostic::warning(codes::SUPERSEDED, format!("{stem}: superseded methodology, imported as rejected"))
                    .hint("use the session without the _v0writeflush suffix; rejected records never enter a fit"),
            );
        }
        let mut records = Vec::new();
        for (suite, prefix) in [(Suite::LlmOps, "ops"), (Suite::Sweep, "sweep")] {
            for (i, r) in root
                .get(prefix)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                records.push(self.cuda_matmul(
                    r,
                    &format!("{prefix}[{i}]"),
                    suite,
                    prefix,
                    l2_bytes,
                )?);
            }
        }
        if let Some(peaks) = root.get("peaks").and_then(Value::as_object) {
            for (name, p) in peaks {
                if let Some(rec) = self.cuda_peak(name, p)? {
                    records.push(rec);
                }
            }
        }
        if superseded {
            for r in &mut records {
                r.quality.reject("session superseded");
            }
        }
        let uses = match oplist {
            Some(o) => self.uses_from_oplist(&o, &records),
            None => {
                self.warn(
                    Diagnostic::warning(
                        codes::OPLIST,
                        "no oplist given: phase membership and counts not imported",
                    )
                    .hint("pass the calibration/oplist.json the session ran with (`--oplist`)"),
                );
                vec![]
            }
        };
        let session = MeasSession {
            schema: MEAS_SCHEMA.into(),
            session_id,
            device,
            clocks_power,
            software,
            host: Host {
                provider: "colab".into(),
                hostname: opt_str(dev, "hostname"),
                ..Default::default()
            },
            method,
            timing_modes,
            canonical_mode: "graph_cold".into(),
            evidence_grade: EvidenceGrade::G2,
            quality,
            legacy: Some(self.source.clone()),
            records,
            uses,
            hash: None,
        };
        Ok((session, superseded))
    }

    fn cuda_matmul(
        &mut self,
        r: &Value,
        path: &str,
        suite: Suite,
        prefix: &str,
        l2_bytes: Option<u64>,
    ) -> Result<MeasRecord, Diagnostic> {
        let name = get(r, path, "name")?
            .as_str()
            .unwrap_or_default()
            .to_string();
        let (m, n, k) = (
            uint(r, path, "m")?,
            uint(r, path, "n")?,
            uint(r, path, "k")?,
        );
        let batch = r.get("batch").and_then(Value::as_u64).unwrap_or(1);
        let weight = r.get("weight").and_then(Value::as_bool).unwrap_or(true);
        let linear = r.get("linear").and_then(Value::as_bool).unwrap_or(false);
        let op = if weight {
            BenchOp::gemm(m, n, k, linear)
        } else {
            BenchOp::bmm(batch, m, n, k)
        };
        self.check_name(path, &name, &op);
        let implementation = match (weight, linear) {
            (false, _) => "torch.bmm",
            (true, true) => "torch.nn.functional.linear",
            (true, false) => "torch.matmul",
        };
        let mut modes = BTreeMap::new();
        let mut mode_errors = BTreeMap::new();
        if let Some(e) = opt_str(r, "error") {
            mode_errors.insert("*".to_string(), e);
        }
        for mode in CUDA_MODES {
            let Some(mv) = r.get(mode) else { continue };
            if let Some(e) = opt_str(mv, "error") {
                mode_errors.insert(mode.to_string(), e);
                continue;
            }
            let p = format!("{path}.{mode}");
            let mut s = ModeStats {
                median_s: us(num(mv, &p, "median_us")?),
                min_s: opt_num(mv, "min_us").map(us),
                p90_s: opt_num(mv, "p90_us").map(us),
                n: mv.get("n").and_then(Value::as_u64).map(|x| x as u32),
                cv: None,
                details: BTreeMap::new(),
            };
            for key in ["copies", "reps"] {
                if let Some(x) = mv.get(key) {
                    s.details.insert(key.to_string(), x.clone());
                }
            }
            modes.insert(mode.to_string(), s);
        }
        let (clock, power) = clock_power(r);
        let mut quality = Quality::accepted();
        gate_clock(&mut quality, clock.as_ref(), power.as_ref());
        if !modes.contains_key("graph_cold") {
            quality.flag("canonical mode graph_cold missing");
        }
        if !mode_errors.is_empty() {
            quality.flag(format!(
                "modes failed: {}",
                mode_errors.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        let copies = modes
            .get("graph_cold")
            .and_then(|s| s.details.get("copies"))
            .and_then(Value::as_u64);
        if let (Some(c), Some(b)) = (copies, r.get("min_bytes").and_then(Value::as_u64)) {
            let footprint = c * b;
            if footprint < ROTATE_BYTES {
                let l2 = l2_bytes
                    .filter(|&l2| footprint <= l2)
                    .map_or(String::new(), |l2| {
                        format!(
                            "; fits in the {} L2, so operands may be L2-resident rather than cold",
                            mib(l2 as f64)
                        )
                    });
                quality.flag(format!(
                    "graph_cold rotates {c} copies = {} < 512 MiB (64-copy cap){l2}",
                    mib(footprint as f64)
                ));
            }
        }
        let aliases = match (opt_str(r, "phase"), opt_str(r, "op")) {
            (Some(phase), Some(op)) => vec![PhaseOp { phase, op }],
            _ => vec![],
        };
        Ok(MeasRecord {
            id: format!("{prefix}/{name}"),
            suite,
            bench_key: op.key(),
            op,
            implementation: implementation.into(),
            legacy_name: Some(name),
            modes,
            mode_errors,
            clock,
            power,
            quality,
            aliases,
        })
    }

    fn cuda_peak(&mut self, name: &str, p: &Value) -> Result<Option<MeasRecord>, Diagnostic> {
        let path = format!("peaks.{name}");
        let (op, implementation) = if name == "copy_1GiB" {
            (
                BenchOp::stream(BenchKind::Copy, 1 << 30, &[("x", "uint8")]),
                "Tensor.copy_",
            )
        } else if name == "read_sum_1GiB" {
            (
                BenchOp::stream(
                    BenchKind::ReadReduce,
                    1 << 30,
                    &[("x", "fp32"), ("acc", "fp32")],
                ),
                "Tensor.sum",
            )
        } else if let Some(n) = name
            .strip_prefix("matmul_")
            .and_then(|s| s.strip_suffix("^3")?.parse().ok())
        {
            (BenchOp::gemm(n, n, n, false), "torch.matmul")
        } else {
            self.warn(
                Diagnostic::warning(codes::FIELD, format!("unknown peak {name:?} skipped"))
                    .at(path),
            );
            return Ok(None);
        };
        let stats = ModeStats {
            median_s: us(num(p, &path, "median_us")?),
            min_s: opt_num(p, "min_us").map(us),
            p90_s: opt_num(p, "p90_us").map(us),
            n: p.get("n").and_then(Value::as_u64).map(|x| x as u32),
            cv: None,
            details: BTreeMap::new(),
        };
        let (clock, power) = clock_power(p);
        let mut quality = Quality::accepted();
        gate_clock(&mut quality, clock.as_ref(), power.as_ref());
        Ok(Some(MeasRecord {
            id: format!("peak/{name}"),
            suite: Suite::Peak,
            bench_key: op.key(),
            op,
            implementation: implementation.into(),
            legacy_name: Some(name.to_string()),
            modes: BTreeMap::from([("unflushed".to_string(), stats)]),
            mode_errors: BTreeMap::new(),
            clock,
            power,
            quality,
            aliases: vec![],
        }))
    }

    fn uses_from_oplist(&mut self, oplist: &[OplistEntry], records: &[MeasRecord]) -> Vec<OpUse> {
        let mut uses = Vec::new();
        for o in oplist {
            let op = if o.weight {
                BenchOp::gemm(o.m, o.n, o.k, false)
            } else {
                BenchOp::bmm(o.batch, o.m, o.n, o.k)
            };
            let id = format!("ops/{}", o.key);
            match records.iter().find(|r| r.id == id) {
                Some(r) if r.bench_key == op.key() => uses.push(OpUse {
                    suite: o.suite.clone(),
                    phase: o.phase.clone(),
                    op: o.op.clone(),
                    bench_key: r.bench_key.clone(),
                    record: id,
                    count: o.count,
                    tokens: o.tokens,
                }),
                Some(_) => self.warn(Diagnostic::warning(
                    codes::OPLIST,
                    format!("{id}: shape differs from oplist row"),
                )),
                None => self.warn(Diagnostic::warning(
                    codes::OPLIST,
                    format!("oplist key {} has no record", o.key),
                )),
            }
        }
        uses
    }

    fn jax(&mut self, root: &Value, session_id: String) -> Result<MeasSession, Diagnostic> {
        self.source.format = "tpu_bench/loop_v3".into();
        let kind = opt_str(root, "device_kind").unwrap_or_default();
        let n_runs = root.get("n_runs").and_then(Value::as_u64).map(|x| x as u32);
        let device = Device {
            vendor: Vendor::Google,
            sku: kind.clone(),
            device_kind: Some(kind),
            pci_device_id: None,
            vbios: None,
            memory_bytes: None,
            core_count: None,
            l2_bytes: None,
            mem_bus_width_bits: None,
            compute_capability: None,
            chip_count: root
                .get("devices")
                .and_then(Value::as_array)
                .map_or(1, |d| d.len() as u32),
            topology: None,
            spec_mem_bw_bps: None,
            spec_bf16_flops: None,
        };
        let software: BTreeMap<_, _> = ["jax", "jaxlib", "libtpu", "platform"]
            .iter()
            .filter_map(|k| opt_str(root, k).map(|v| (k.to_string(), v)))
            .collect();
        self.note("host.provider, method.rotate_bytes and method.settings are taken from the runner source, not the file");
        let method = Method {
            runner: "calibration/tpu_bench.py".into(),
            iters: n_runs,
            rotate_bytes: Some(ROTATE_BYTES),
            started: opt_str(root, "date"),
            settings: BTreeMap::from([
                ("loop_target_s".into(), json!(0.03)),
                ("loop_version".into(), json!(3)),
                (
                    "canonical_op_mode".into(),
                    json!(
                        "loop (08 §F TPU per-op timing mode; was best = min(loop, pipelined_rot), 08 E.3)"
                    ),
                ),
            ]),
            ..Default::default()
        };
        let mode =
            |description: &str, launch, oh: &[Overhead], operands, trusted, derived: &[&str]| {
                TimingMode {
                    description: description.into(),
                    launch,
                    overheads_included: oh.iter().copied().collect(),
                    operands,
                    trusted,
                    diagnostic: false,
                    derived_from: derived.iter().map(|s| s.to_string()).collect(),
                }
            };
        use Overhead::*;
        let mut timing_modes: BTreeMap<String, TimingMode> = BTreeMap::from([
            (
                "single".into(),
                mode(
                    "one jitted call with block_until_ready per call; median of n_runs; includes host dispatch",
                    LaunchPath::HostBlocking,
                    &[HostDispatch, KernelLaunch],
                    Residency::L2Warm,
                    true,
                    &[],
                ),
            ),
            (
                "pipelined".into(),
                mode(
                    "n_runs back-to-back calls on the same buffers, one block; min over 3 batches of the per-call mean; \
                     untrusted: libtpu keeps VMEM-sized operands resident",
                    LaunchPath::HostPipelined,
                    &[KernelLaunch, InterKernelGap],
                    Residency::Resident,
                    false,
                    &[],
                ),
            ),
            (
                "pipelined_rot".into(),
                mode(
                    "as pipelined, cycling over R operand copies (>= 512 MiB when small, R <= 64)",
                    LaunchPath::HostPipelined,
                    &[KernelLaunch, InterKernelGap],
                    Residency::ColdDram,
                    true,
                    &[],
                ),
            ),
            (
                "loop".into(),
                mode(
                    "slope of a jitted fori_loop between K1 and K2 iterations over R rotated operand copies and output \
                     slots, with a scalar perturbation carry; median of 7 per K",
                    LaunchPath::DeviceLoop,
                    &[],
                    Residency::ColdDram,
                    true,
                    &[],
                ),
            ),
            (
                "best".into(),
                mode(
                    "min(loop, pipelined_rot) as computed by the runner (`best_s`); details.from names the winner; \
                     canonical until 08 §F made `loop` canonical, kept as a diagnostic",
                    LaunchPath::Derived,
                    &[],
                    Residency::ColdDram,
                    true,
                    &["loop", "pipelined_rot"],
                ),
            ),
        ]);
        for m in TPU_DIAGNOSTIC_OP_MODES.iter().chain(&["best"]) {
            if let Some(t) = timing_modes.get_mut(*m) {
                t.diagnostic = true;
            }
        }
        let mut quality = Quality::accepted();
        quality.unevaluated = vec![
            "cv <= 3% (per-sample times not recorded)".into(),
            "clock window (no TPU clock/power telemetry)".into(),
            "session start/end sanity drift".into(),
            "repeat on >= 2 instances".into(),
        ];
        if root.get("done").and_then(Value::as_bool) != Some(true) {
            quality.flag("run incomplete: `done` flag missing");
        }
        let mut records = Vec::new();
        let mut uses = Vec::new();
        for (suite, prefix) in [
            (Suite::LlmOps, "ops"),
            (Suite::Sweep, "sweep"),
            (Suite::Peak, "peak"),
        ] {
            for (i, r) in root
                .get(prefix)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let rec = self.jax_matmul(r, &format!("{prefix}[{i}]"), suite, prefix, n_runs)?;
                if let (Some(phase), Some(op)) = (opt_str(r, "phase"), opt_str(r, "op")) {
                    uses.push(OpUse {
                        suite: match phase.as_str() {
                            "smoke" => "smoke",
                            "gemm" => "make_gemm",
                            _ => "all",
                        }
                        .into(),
                        phase,
                        op,
                        bench_key: rec.bench_key.clone(),
                        record: rec.id.clone(),
                        count: uint(r, prefix, "count")?,
                        tokens: uint(r, prefix, "tokens")?,
                    });
                }
                records.push(rec);
            }
        }
        for (i, r) in root
            .get("hbm")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            records.push(self.jax_hbm(r, &format!("hbm[{i}]"), n_runs)?);
        }
        Ok(MeasSession {
            schema: MEAS_SCHEMA.into(),
            session_id,
            device,
            clocks_power: ClocksPower::default(),
            software,
            host: Host {
                provider: "colab".into(),
                ..Default::default()
            },
            method,
            timing_modes,
            canonical_mode: TPU_OP_MODES[0].into(),
            evidence_grade: EvidenceGrade::G2,
            quality,
            legacy: Some(self.source.clone()),
            records,
            uses,
            hash: None,
        })
    }

    fn jax_matmul(
        &mut self,
        r: &Value,
        path: &str,
        suite: Suite,
        prefix: &str,
        n_runs: Option<u32>,
    ) -> Result<MeasRecord, Diagnostic> {
        let (m, n, k) = (
            uint(r, path, "m")?,
            uint(r, path, "n")?,
            uint(r, path, "k")?,
        );
        let batch = r.get("batch").and_then(Value::as_u64).unwrap_or(1);
        let weight = r.get("weight").and_then(Value::as_bool).unwrap_or(true);
        let op = if weight {
            BenchOp::gemm(m, n, k, false)
        } else {
            BenchOp::bmm(batch, m, n, k)
        };
        let shape_name = op.legacy_name().expect("matmul has a legacy name");
        let name = opt_str(r, "name").unwrap_or_else(|| shape_name.clone());
        let implementation = if batch > 1 && !weight {
            "jnp.einsum(bmk,bkn->bmn)"
        } else {
            "jnp.matmul"
        };
        let mut modes = BTreeMap::new();
        modes.insert(
            "single".to_string(),
            ModeStats {
                min_s: opt_num(r, "single_min_s"),
                n: n_runs,
                ..ModeStats::median(num(r, path, "single_median_s")?)
            },
        );
        for (mode, key) in [
            ("pipelined", "pipelined_s"),
            ("pipelined_rot", "pipelined_rot_s"),
            ("loop", "loop_s"),
        ] {
            if let Some(x) = opt_num(r, key) {
                modes.insert(mode.to_string(), ModeStats::median(x));
            }
        }
        if let (Some(info), Some(l)) = (
            r.get("loop_info").and_then(Value::as_object),
            modes.get_mut("loop"),
        ) {
            l.details = info.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        }
        let mut mode_errors = BTreeMap::new();
        if let Some(e) = opt_str(r, "loop_error") {
            mode_errors.insert("loop".to_string(), e);
        }
        let mut quality = Quality::accepted();
        let candidates: Vec<(&str, f64)> = ["loop", "pipelined_rot"]
            .into_iter()
            .filter_map(|m| modes.get(m).map(|s| (m, s.median_s)))
            .collect();
        match (
            opt_num(r, "best_s"),
            candidates.iter().min_by(|a, b| a.1.total_cmp(&b.1)),
        ) {
            (Some(best), Some(&(from, min))) => {
                if best != min {
                    self.warn(
                        Diagnostic::warning(
                            codes::FIELD,
                            format!("best_s {best} != min(loop, pipelined_rot) {min}"),
                        )
                        .at(path),
                    );
                }
                let mut s = ModeStats::median(best);
                s.details.insert("from".into(), json!(from));
                modes.insert("best".to_string(), s);
            }
            (Some(_), None) => quality
                .flag("best_s present without loop or pipelined_rot; not imported as canonical"),
            (None, _) => {}
        }
        if !modes.contains_key("loop") {
            quality
                .flag("canonical mode loop missing (record predates loop v3 or loop was skipped)");
        }
        if let (Some(rot), Some(bytes)) = (
            r.get("loop_info")
                .and_then(|i| i.get("R"))
                .and_then(Value::as_u64),
            r.get("bytes").and_then(Value::as_u64),
        ) && rot * bytes < ROTATE_BYTES
        {
            quality.flag(format!(
                "loop rotates R={rot} copies = {} < 512 MiB (64-copy cap)",
                mib((rot * bytes) as f64)
            ));
        }
        if !mode_errors.is_empty() {
            quality.flag("loop mode failed");
        }
        let aliases = match (opt_str(r, "phase"), opt_str(r, "op")) {
            (Some(phase), Some(op)) => vec![PhaseOp { phase, op }],
            _ => vec![],
        };
        let id = if suite == Suite::LlmOps {
            format!("{prefix}/{name}")
        } else {
            format!("{prefix}/{shape_name}")
        };
        Ok(MeasRecord {
            id,
            suite,
            bench_key: op.key(),
            op,
            implementation: implementation.into(),
            legacy_name: Some(name),
            modes,
            mode_errors,
            clock: None,
            power: None,
            quality,
            aliases,
        })
    }

    fn jax_hbm(
        &mut self,
        r: &Value,
        path: &str,
        n_runs: Option<u32>,
    ) -> Result<MeasRecord, Diagnostic> {
        let test = get(r, path, "test")?
            .as_str()
            .unwrap_or_default()
            .to_string();
        let gib = num(r, path, "gib")?;
        let input_bytes = (gib * (1u64 << 30) as f64) as u64;
        let (op, implementation) = match test.as_str() {
            "read_reduce" => (
                BenchOp::stream(
                    BenchKind::ReadReduce,
                    input_bytes,
                    &[("x", "bf16"), ("acc", "fp32")],
                ),
                "jnp.sum(dtype=float32)",
            ),
            "copy_scale" => (
                BenchOp::stream(
                    BenchKind::Scale,
                    input_bytes,
                    &[("x", "bf16"), ("out", "bf16")],
                ),
                "jnp.multiply(v, 2)",
            ),
            other => {
                return Err(
                    Diagnostic::error(codes::FIELD, format!("unknown hbm test {other:?}")).at(path),
                );
            }
        };
        let stats = ModeStats {
            min_s: opt_num(r, "min_s"),
            n: n_runs,
            ..ModeStats::median(num(r, path, "median_s")?)
        };
        Ok(MeasRecord {
            id: format!("hbm/{test}_{gib}gib"),
            suite: Suite::Hbm,
            bench_key: op.key(),
            op,
            implementation: implementation.into(),
            legacy_name: Some(format!("{test}_{gib}gib")),
            modes: BTreeMap::from([("single".to_string(), stats)]),
            mode_errors: BTreeMap::new(),
            clock: None,
            power: None,
            quality: Quality::accepted(),
            aliases: vec![],
        })
    }
}

fn clock_power(r: &Value) -> (Option<ClockWindow>, Option<PowerWindow>) {
    let Some(c) = r.get("clock").filter(|c| c.is_object()) else {
        return (None, None);
    };
    let clock = (|| {
        Some(ClockWindow {
            samples: c.get("n")?.as_u64()? as u32,
            sm_hz_median: c.get("sm_mhz")?.as_f64()? * 1e6,
            sm_hz_min: c.get("sm_mhz_min")?.as_f64()? * 1e6,
            mem_hz_median: c.get("mem_mhz")?.as_f64()? * 1e6,
            temp_c: opt_num(c, "temp_c"),
        })
    })();
    let power = (|| {
        Some(PowerWindow {
            median_w: opt_num(c, "power_w")?,
            max_w: opt_num(c, "power_w_max")?,
        })
    })();
    (clock, power)
}

/// 06 §4.3 clock gate plus a check that the power window is not dominated by idle samples.
fn gate_clock(q: &mut Quality, clock: Option<&ClockWindow>, power: Option<&PowerWindow>) {
    match clock {
        None => q.flag("no clock window"),
        Some(c) if c.sm_hz_min < 0.9 * c.sm_hz_median => q.flag(format!(
            "SM clock min {:.0} MHz < 0.9 x median {:.0} MHz",
            c.sm_hz_min / 1e6,
            c.sm_hz_median / 1e6
        )),
        _ => {}
    }
    if let Some(p) = power.filter(|p| p.max_w > 2.0 * p.median_w) {
        q.flag(format!(
            "power window median {:.0} W vs max {:.0} W: window mostly idle, median power unusable",
            p.median_w, p.max_w
        ));
    }
}
