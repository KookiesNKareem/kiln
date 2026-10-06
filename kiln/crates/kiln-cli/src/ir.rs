//! `kiln validate` (hardware, workload), `kiln import harness-design` and `kiln bench export`.

use std::fmt::Write as _;
use std::path::Path;

use kiln_ir::common::{Diagnostic, Severity, canonical_json};
use kiln_ir::hw::{self, Profile};
use kiln_ir::wl::{self, PhaseKind, WorkloadDoc};
use serde_json::{Value, json};

use crate::cli::{Format, Global, Kind};
use crate::commands::{read, write_out};
use crate::{Failure, emit, exit};

const KIND_CODE: &str = "E-CLI-0001";
const WL_DOC_CODE: &str = "E-WL-DOC-001";

fn input_error(code: &str, path: &Path, msg: impl Into<String>) -> Failure {
    Failure::new(
        exit::INPUT,
        Diagnostic::error(code, msg).at(path.display().to_string()),
    )
}

/// `--kind auto`: `.json5` is hardware; JSON is classified by its `schema` or `kiln_workload` field.
pub fn detect_kind(path: &Path, bytes: &[u8]) -> Result<Kind, Failure> {
    if path.extension().is_some_and(|e| e == "json5") {
        return Ok(Kind::Hardware);
    }
    let v: Value = serde_json::from_slice(bytes)
        .map_err(|e| input_error(KIND_CODE, path, format!("not JSON: {e}")))?;
    let schema = v.get("schema").and_then(Value::as_str).unwrap_or_default();
    match schema.split('/').next().unwrap_or_default() {
        "kiln.hw" => Ok(Kind::Hardware),
        "kiln.meas" | "kiln.result" | "kiln.sim" => Ok(Kind::Measurement),
        _ if v.get("kiln_workload").is_some() => Ok(Kind::Workload),
        _ => Err(Failure::new(
            exit::INPUT,
            Diagnostic::error(KIND_CODE, "cannot tell what kind of document this is")
                .at(path.display().to_string())
                .hint("expected `schema: kiln.hw/1.x | kiln.meas/1` or a `kiln_workload` field; or pass --kind hw|workload|measurement"),
        )),
    }
}

fn profile(name: &str) -> Profile {
    match name {
        "reference" => Profile::Reference,
        "search" => Profile::Search,
        "stream_compat" => Profile::StreamCompat,
        _ => Profile::Full,
    }
}

/// Prints diagnostics and the report; exit 1 when errors (or warnings under `--deny-warnings`) remain.
fn finish(
    g: &Global,
    file: &Path,
    kind: &str,
    diags: &[Diagnostic],
    deny_warnings: bool,
    summary: Value,
    text: String,
) -> Result<u8, Failure> {
    emit(g.format, diags);
    let errors = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .count();
    let warnings = diags.len() - errors;
    let ok = errors == 0 && !(deny_warnings && warnings > 0);
    match g.format {
        Format::Json | Format::Jsonl => {
            let doc = json!({"file": file.display().to_string(), "kind": kind, "ok": ok, "errors": errors,
                "warnings": warnings, "summary": summary, "diagnostics": diags});
            match &g.out {
                Some(p) => write_out(p, &(canonical_json(&doc) + "\n"))?,
                None => println!("{}", canonical_json(&doc)),
            }
        }
        Format::Text | Format::Llm => print!(
            "{:<9} {} ({kind}, {errors} errors, {warnings} warnings)\n{text}",
            if ok { "ok" } else { "invalid" },
            file.display(),
        ),
    }
    Ok(if ok {
        exit::OK
    } else {
        exit::COMPLETED_WITH_INVALID
    })
}

pub fn validate_hw(
    g: &Global,
    file: &Path,
    profile_name: Option<&str>,
    deny_warnings: bool,
) -> Result<u8, Failure> {
    let profile_name = profile_name.unwrap_or("full");
    let design = hw::load_file(file).map_err(|diags| Failure {
        code: exit::INPUT,
        diags,
    })?;
    let r = kiln_sim::check_priced(design, profile(profile_name));
    let d = r.design.as_ref().expect("check keeps the design");
    let mut text = format!(
        "design    {}  {}\nprofile   {profile_name}\n",
        d.doc.name, d.hash
    );
    let mut summary = json!({"name": d.doc.name, "hash": d.hash, "profile": profile_name});
    if let Some(m) = &r.model {
        let s = m.summary();
        writeln!(text, "chips     {}", s.chip_count).unwrap();
        for (mode, ops) in &s.peak_ops {
            writeln!(text, "peak      {:>10.1} TFLOPS  {mode}", ops / 1e12).unwrap();
        }
        writeln!(
            text,
            "offchip   {:>10.1} GB/s    {:.1} GiB\nonchip    {:>10.2} MiB",
            s.offchip_bandwidth.0 / 1e9,
            s.offchip_capacity.0 as f64 / f64::from(1 << 30),
            s.onchip_capacity.0 as f64 / f64::from(1 << 20),
        )
        .unwrap();
        summary["chip_count"] = json!(s.chip_count);
        summary["peak_ops"] = json!(s.peak_ops);
        summary["offchip_bandwidth_bps"] = json!(s.offchip_bandwidth.0);
        summary["offchip_bytes"] = json!(s.offchip_capacity.0);
        summary["onchip_bytes"] = json!(s.onchip_capacity.0);
    }
    finish(
        g,
        file,
        "hardware",
        &r.diagnostics,
        deny_warnings,
        summary,
        text,
    )
}

fn phase_label(scenario: &str, kind: PhaseKind, index: Option<u64>, n: usize) -> String {
    let kind = serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default();
    match (n, index) {
        (1, _) => scenario.to_string(),
        (_, Some(i)) => format!("{scenario}/{kind}@{i}"),
        (_, None) => format!("{scenario}/{kind}"),
    }
}

pub fn validate_workload(g: &Global, file: &Path, deny_warnings: bool) -> Result<u8, Failure> {
    let bytes = read(file)?;
    let doc: WorkloadDoc = serde_json::from_slice(&bytes).map_err(|e| {
        input_error(
            WL_DOC_CODE,
            file,
            format!("not a kiln workload document: {e}"),
        )
    })?;
    let doc = kiln_wl::zoo::expand_doc(&doc).map_err(|d| Failure::new(exit::INPUT, d))?;
    let mut diags = wl::validate_doc(&doc);
    let model = doc.expanded().expect("expand_doc yields a full model");
    let model_hash = wl::model_hash(model);
    let mut text = format!("workload  {}  model {model_hash}\n", doc.id);
    let mut scenarios = Vec::new();
    if !diags.iter().any(|d| d.severity == Severity::Error) {
        for (sid, s) in &doc.scenarios {
            let insts = match kiln_wl::expand::expand(s, model) {
                Ok(i) => i,
                Err(d) => {
                    diags.push(d.at(format!("scenarios.{sid}")));
                    continue;
                }
            };
            let hash = wl::workload_hash(model, s, None);
            writeln!(text, "scenario  {sid}  {hash}").unwrap();
            let mut phases = Vec::new();
            for inst in &insts {
                let label = phase_label(sid.as_str(), inst.kind, inst.index, insts.len());
                let (b, _, st) = match kiln_wl::evaluate_instance(model, s, inst) {
                    Ok(x) => x,
                    Err(ds) => {
                        diags.extend(ds.into_iter().map(|d| match d.path {
                            Some(_) => d,
                            None => d.at(format!("scenarios.{sid}")),
                        }));
                        continue;
                    }
                };
                let params =
                    kiln_wl::param_count(model, &b).map_err(|d| Failure::new(exit::INPUT, d))?;
                let (mm, vec) = (st.useful.flops_mm, st.useful.vec_ops);
                let bytes = st.compulsory_bytes();
                writeln!(
                    text,
                    "  {label:<22} x{:<4} params {:.3e}  flops {:.4e} (+{:.3e} vec)  bytes {:.4e}  tokens {}",
                    inst.multiplicity,
                    params as f64,
                    mm as f64,
                    vec as f64,
                    bytes as f64,
                    b.seqs.tokens(),
                )
                .unwrap();
                phases.push(
                    json!({"phase": label, "multiplicity": inst.multiplicity.to_string(),
                    "tokens": b.seqs.tokens(), "params": params, "flops_mm": mm, "vec_ops": vec,
                    "compulsory_bytes": bytes, "weight_bytes": st.resident.weights}),
                );
            }
            scenarios.push(json!({"id": sid, "workload_hash": hash, "phases": phases}));
        }
    }
    let summary = json!({"id": doc.id, "model_hash": model_hash, "scenarios": scenarios});
    finish(g, file, "workload", &diags, deny_warnings, summary, text)
}

/// `kiln import harness-design`: `kiln.hw/0` harness JSON -> `kiln.hw/1.0` canonical form (pretty, field order).
pub fn import_harness_design(g: &Global, file: &Path) -> Result<u8, Failure> {
    let text = String::from_utf8(read(file)?)
        .map_err(|e| input_error("E-IR-0100", file, format!("not UTF-8: {e}")))?;
    let d = hw::import_harness(&text).map_err(|diags| Failure {
        code: exit::INPUT,
        diags,
    })?;
    emit(g.format, &d.warnings);
    let doc = serde_json::to_string_pretty(&d.canonical).expect("value serializes") + "\n";
    match &g.out {
        Some(p) => {
            write_out(p, &doc)?;
            match g.format {
                Format::Json | Format::Jsonl => println!(
                    "{}",
                    canonical_json(
                        &json!({"name": d.doc.name, "hash": d.hash, "out": p.display().to_string()})
                    )
                ),
                _ => println!("imported  {}  {} -> {}", d.doc.name, d.hash, p.display()),
            }
        }
        None => print!("{doc}"),
    }
    Ok(exit::OK)
}

/// `kiln bench export`: the manifest of a suite or one `<preset>:<scenario>` workload.
pub fn bench_export(g: &Global, name: &str) -> Result<u8, Failure> {
    let m = kiln_wl::bench::manifest(name).map_err(|diags| Failure {
        code: exit::INPUT,
        diags,
    })?;
    let v = serde_json::to_value(&m).expect("manifest serializes");
    let doc = canonical_json(&v) + "\n";
    if let Some(p) = &g.out {
        write_out(p, &doc)?;
    }
    match (g.format, &g.out) {
        (Format::Json | Format::Jsonl, None) => print!("{doc}"),
        (Format::Json | Format::Jsonl, Some(_)) => {}
        _ => {
            let mut t = format!("{}  {}  {} ops\n", m.schema, m.suite, m.ops.len());
            for o in &m.ops {
                let count: u64 = o.uses.iter().map(|u| u.count).sum();
                let label = o.legacy_name.clone().unwrap_or_else(|| {
                    let kind = serde_json::to_value(o.op.kind).unwrap();
                    format!(
                        "{}:{}",
                        kind.as_str().unwrap_or_default(),
                        o.op.op.as_deref().unwrap_or("-")
                    )
                });
                writeln!(
                    t,
                    "{}  {label:<30} uses {:<3} count {count}",
                    o.key,
                    o.uses.len()
                )
                .unwrap();
            }
            if let Some(p) = &g.out {
                writeln!(t, "written   {}", p.display()).unwrap();
            }
            print!("{t}");
        }
    }
    Ok(exit::OK)
}
