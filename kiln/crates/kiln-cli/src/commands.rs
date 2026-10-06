use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use kiln_ir::common::{Diagnostic, Severity, canonical_json};
use kiln_trace::check::{check_result, check_sim};
use kiln_trace::container::{self, Manifest};
use kiln_trace::meas::legacy::import_legacy;
use kiln_trace::meas::store::{MeasStore, SessionStatus};
use kiln_trace::meas::{MeasSession, QualityStatus, codes};
use kiln_trace::{EvalResult, Interval, RESULT_SCHEMA, SIM_SCHEMA, SimResult};
use serde_json::{Value, json};

use crate::cli::{
    BenchCmd, Cli, Cmd, Format, Global, ImportArgs, ImportCmd, Kind, TraceCmd, VizCmd,
};
use crate::{Failure, emit, eval, exit, ir};

pub fn run(cli: &Cli) -> Result<u8, Failure> {
    let g = &cli.global;
    match &cli.cmd {
        Cmd::Bench(BenchCmd::Import(a)) if a.legacy => bench_import_legacy(g, a),
        Cmd::Bench(BenchCmd::Export {
            workload,
            suite,
            sequence,
            scope,
            ..
        }) => {
            if *sequence || scope.is_some() {
                return Err(unimplemented("bench export --sequence/--scope", "M2"));
            }
            ir::bench_export(
                g,
                suite.as_deref().or(workload.as_deref()).unwrap_or_default(),
            )
        }
        Cmd::Import(ImportCmd::HarnessDesign { file }) => ir::import_harness_design(g, file),
        Cmd::Calibrate(c) => crate::calibrate::run(g, c),
        Cmd::Eval(a) => eval::eval(g, a),
        Cmd::Compare(a) => eval::compare(g, a),
        Cmd::Explain { result, max_items } => eval::explain_cmd(g, result, *max_items),
        Cmd::Trace(TraceCmd::Info { file, json }) => trace_info(g, file, *json),
        Cmd::Trace(TraceCmd::Validate { file }) => trace_validate(g, file, exit::CHECK_FAILED),
        Cmd::Trace(TraceCmd::Export(a)) => crate::viz::export_cmd(g, a),
        Cmd::Viz(v) => match &v.sub {
            Some(VizCmd::Render(a)) => crate::viz::render_cmd(g, a),
            None => crate::viz::viz_cmd(g, v),
        },
        Cmd::Validate {
            file,
            kind,
            profile,
            deny_warnings,
            ..
        } => {
            let kind = match kind {
                Kind::Auto => ir::detect_kind(file, &read(file)?)?,
                k => *k,
            };
            match kind {
                Kind::Hardware => ir::validate_hw(g, file, profile.as_deref(), *deny_warnings),
                Kind::Workload => ir::validate_workload(g, file, *deny_warnings),
                Kind::Measurement => trace_validate(g, file, exit::COMPLETED_WITH_INVALID),
                Kind::Calibration => crate::calibrate::validate(g, file),
                Kind::Auto => Err(unimplemented("validate --kind auto (calibration)", "M2")),
            }
        }
        other => Err(not_implemented(other)),
    }
}

fn not_implemented(cmd: &Cmd) -> Failure {
    unimplemented(cmd.name(), cmd.milestone())
}

fn unimplemented(what: &str, milestone: &str) -> Failure {
    Failure::new(
        exit::NOT_IMPLEMENTED,
        Diagnostic::error(
            "E-NOT-IMPLEMENTED",
            format!("`kiln {what}` is not implemented in this build"),
        )
        .hint(format!("planned for milestone {milestone} (06 §10)")),
    )
}

pub fn read(path: &Path) -> Result<Vec<u8>, Failure> {
    fs::read(path).map_err(|e| {
        Failure::new(
            exit::INPUT,
            Diagnostic::error(codes::IO, format!("cannot read {}: {e}", path.display())),
        )
    })
}

pub fn write_out(path: &Path, text: &str) -> Result<(), Failure> {
    fs::write(path, text).map_err(|e| {
        Failure::new(
            exit::INPUT,
            Diagnostic::error(codes::IO, format!("cannot write {}: {e}", path.display())),
        )
    })
}

/// JSON to `--out` or stdout; text to stdout.
fn output(g: &Global, json: &Value, text: impl FnOnce() -> String) -> Result<(), Failure> {
    match (g.format, &g.out) {
        (Format::Json | Format::Jsonl, None) => println!("{}", canonical_json(json)),
        (Format::Json | Format::Jsonl, Some(p)) => write_out(p, &(canonical_json(json) + "\n"))?,
        (_, _) => print!("{}", text()),
    }
    Ok(())
}

fn bench_import_legacy(g: &Global, a: &ImportArgs) -> Result<u8, Failure> {
    let raw = read(&a.file)?;
    let oplist_path = a.oplist.clone().or_else(|| default_oplist(&a.file, &raw));
    let oplist = oplist_path.as_deref().map(read).transpose()?;
    let name = a
        .file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("legacy.json");
    let imp =
        import_legacy(&raw, name, oplist.as_deref()).map_err(|d| Failure::new(exit::INPUT, d))?;
    emit(g.format, &imp.warnings);
    let s = &imp.session;
    let doc = s.canonical_json() + "\n";
    if let Some(out) = &g.out {
        write_out(out, &doc)?;
    }
    let stored = match &a.store {
        Some(root) => {
            let status = if imp.superseded {
                SessionStatus::Rejected {
                    reason: "superseded methodology".into(),
                }
            } else {
                SessionStatus::Active
            };
            Some(
                MeasStore::new(root)
                    .put(s, status)
                    .map_err(|d| Failure::new(exit::INPUT, d))?,
            )
        }
        None => None,
    };
    match g.format {
        Format::Json | Format::Jsonl if g.out.is_none() => print!("{doc}"),
        Format::Json | Format::Jsonl => {}
        _ => {
            let mut t = session_summary(s);
            if let Some(p) = &oplist_path {
                writeln!(t, "oplist    {}", p.display()).unwrap();
            }
            for p in g.out.iter().chain(stored.iter()) {
                writeln!(t, "written   {}", p.display()).unwrap();
            }
            print!("{t}");
        }
    }
    Ok(if imp.superseded {
        exit::COMPLETED_WITH_INVALID
    } else {
        exit::OK
    })
}

/// `calibration/oplist.json` next to the `measurements/` directory, for gpu_bench output only.
fn default_oplist(file: &Path, raw: &[u8]) -> Option<PathBuf> {
    let is_cuda = serde_json::from_slice::<Value>(raw)
        .ok()?
        .get("device")?
        .is_object();
    let p = file.parent()?.parent()?.join("oplist.json");
    (is_cuda && p.is_file()).then_some(p)
}

fn fmt_interval(i: &Interval) -> String {
    format!("{:.4e} [{:.4e}, {:.4e}]", i.central, i.low, i.high)
}

pub fn session_summary(s: &MeasSession) -> String {
    let mut t = String::new();
    let count = |st| s.records.iter().filter(|r| r.quality.status == st).count();
    let mut suites = std::collections::BTreeMap::new();
    for r in &s.records {
        *suites
            .entry(
                serde_json::to_value(r.suite)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
            )
            .or_insert(0) += 1;
    }
    let suites: Vec<_> = suites.iter().map(|(k, v)| format!("{k} {v}")).collect();
    let modes: Vec<_> = s
        .timing_modes
        .keys()
        .map(|m| {
            if *m == s.canonical_mode {
                format!("{m}*")
            } else {
                m.clone()
            }
        })
        .collect();
    writeln!(
        t,
        "session   {}  {}",
        s.session_id,
        s.hash.as_deref().unwrap_or("(unsealed)")
    )
    .unwrap();
    writeln!(
        t,
        "device    {:?} {} ({} chip)",
        s.device.vendor, s.device.sku, s.device.chip_count
    )
    .unwrap();
    writeln!(
        t,
        "quality   {:?}, evidence {:?}  {}",
        s.quality.status,
        s.evidence_grade,
        s.quality.reasons.join("; ")
    )
    .unwrap();
    writeln!(
        t,
        "records   {} ({}); flagged {}, rejected {}",
        s.records.len(),
        suites.join(", "),
        count(QualityStatus::Flagged),
        count(QualityStatus::Rejected)
    )
    .unwrap();
    writeln!(t, "modes     {} (* canonical)", modes.join(", ")).unwrap();
    for phase in s.phases() {
        match s.phase_sum_s(phase, &s.canonical_mode) {
            Ok(x) => writeln!(
                t,
                "phase     {phase:<12} {:>10.3} ms  (count-weighted sum)",
                x * 1e3
            )
            .unwrap(),
            Err(e) => writeln!(t, "phase     {phase:<12} n/a ({})", e.message).unwrap(),
        }
    }
    t
}

enum Doc {
    Result(Box<EvalResult>),
    Sim(Box<SimResult>),
    Meas(Box<MeasSession>),
    Trace(Box<Manifest>),
}

fn load(path: &Path) -> Result<Doc, Failure> {
    let bytes = read(path)?;
    if bytes.starts_with(&container::MAGIC) {
        return container::read_manifest(&bytes)
            .map(|m| Doc::Trace(Box::new(m)))
            .map_err(|d| Failure::new(exit::INPUT, d));
    }
    let bad = |msg: String| {
        Failure::new(
            exit::INPUT,
            Diagnostic::error("E-TRACE-SCHEMA", msg).at(path.display().to_string()),
        )
    };
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| bad(format!("not JSON: {e}")))?;
    let schema = v
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let parse_err = |e: serde_json::Error| bad(format!("does not match {schema}: {e}"));
    match schema.as_str() {
        RESULT_SCHEMA => serde_json::from_value(v)
            .map(|r| Doc::Result(Box::new(r)))
            .map_err(parse_err),
        SIM_SCHEMA => serde_json::from_value(v)
            .map(|r| Doc::Sim(Box::new(r)))
            .map_err(parse_err),
        kiln_trace::meas::MEAS_SCHEMA => serde_json::from_value(v)
            .map(|r| Doc::Meas(Box::new(r)))
            .map_err(parse_err),
        _ => Err(bad(format!(
            "unknown schema {schema:?}; expected {RESULT_SCHEMA}, {SIM_SCHEMA}, {} or a .kiln container",
            kiln_trace::meas::MEAS_SCHEMA
        ))),
    }
}

fn trace_info(g: &Global, file: &Path, json_flag: bool) -> Result<u8, Failure> {
    let doc = load(file)?;
    let (json, text) = match &doc {
        Doc::Result(r) => {
            let phases: Vec<_> = r
                .phases
                .iter()
                .map(|p| json!({"phase": p.phase, "scope": p.scope, "time_s": p.time_s, "tokens_per_s": p.tokens_per_s}))
                .collect();
            let mut t = format!("{RESULT_SCHEMA}  status {:?}  score {}", r.status, r.score);
            if let Some(i) = &r.score_interval {
                write!(t, " [{}, {}]", i.low, i.high).unwrap();
            }
            writeln!(t, "  stage {:?}  tier {:?}", r.stage_reached, r.tier).unwrap();
            provenance_line(&mut t, &r.provenance);
            for p in &r.phases {
                writeln!(
                    t,
                    "phase     {:<12} {:?}  time_s {}  tok/s {}",
                    p.phase,
                    p.scope,
                    fmt_interval(&p.time_s),
                    fmt_interval(&p.tokens_per_s)
                )
                .unwrap();
            }
            let j = json!({"kind": "result", "schema": RESULT_SCHEMA, "status": r.status, "score": r.score,
                "score_interval": r.score_interval, "provenance": r.provenance, "phases": phases,
                "deterministic_hash": r.deterministic_hash()});
            (j, t)
        }
        Doc::Sim(s) => {
            let mut t = format!(
                "{SIM_SCHEMA}  phase {}  tier {:?}  scope {:?}  corner {:?}\nmakespan  {:.6e} s (A0 {:.6e}, A2 {:.6e})\nenergy    {:.6e} J, avg power {:.1} W\n",
                s.phase,
                s.tier,
                s.scope,
                s.corner,
                s.makespan_s,
                s.t_a0_s,
                s.t_a2_s,
                s.energy.total_j,
                s.power.avg_w
            );
            provenance_line(&mut t, &s.provenance);
            if let Some((c, x)) = s.bottleneck.dominant() {
                writeln!(
                    t,
                    "bound     {c:?} {:.0}% of makespan  {}",
                    100.0 * x / s.makespan_s,
                    s.bottleneck.summary
                )
                .unwrap();
            }
            let j = json!({"kind": "sim", "schema": SIM_SCHEMA, "phase": s.phase, "tier": s.tier, "corner": s.corner,
                "makespan_s": s.makespan_s, "t_a0_s": s.t_a0_s, "t_a2_s": s.t_a2_s, "energy_j": s.energy.total_j,
                "time_by_binding": s.bottleneck.time_by_binding, "provenance": s.provenance});
            (j, t)
        }
        Doc::Meas(s) => {
            let phases: serde_json::Map<String, Value> = s
                .phases()
                .into_iter()
                .filter_map(|p| {
                    Some((
                        p.to_string(),
                        json!(s.phase_sum_s(p, &s.canonical_mode).ok()?),
                    ))
                })
                .collect();
            let j = json!({"kind": "measurement", "schema": s.schema, "session_id": s.session_id, "hash": s.hash,
                "device": s.device, "quality": s.quality, "records": s.records.len(),
                "canonical_mode": s.canonical_mode, "phase_sum_s": phases});
            (j, session_summary(s))
        }
        Doc::Trace(m) => {
            let mut t = format!(
                "{} {}  level {:?}  tier {:?}  tables {}\n",
                m.format,
                m.schema_version,
                m.level,
                m.tier,
                m.tables.len()
            );
            provenance_line(&mut t, &m.provenance);
            if let Some(d) = &m.design_name {
                writeln!(
                    t,
                    "design    {d}  workload {}  floorplan {}",
                    m.workload_name.as_deref().unwrap_or("-"),
                    m.floorplan_source.as_deref().unwrap_or("-")
                )
                .unwrap();
            }
            for e in &m.tables {
                writeln!(
                    t,
                    "table     {:<24} {:>8} rows {:>10} B",
                    e.name, e.rows, e.length
                )
                .unwrap();
            }
            for n in &m.notes {
                writeln!(t, "note      {n}").unwrap();
            }
            for (name, i) in [
                ("latency_s", &m.headline.latency_s),
                ("tokens/s", &m.headline.tokens_per_s),
                ("energy_j", &m.headline.energy_j),
                ("power_w", &m.headline.power_w),
                ("area_mm2", &m.headline.area_mm2),
                ("score", &m.headline.score),
            ] {
                if let Some(i) = i {
                    writeln!(t, "{name:<9} {}", fmt_interval(i)).unwrap();
                }
            }
            let j = json!({"kind": "trace", "format": m.format, "schema_version": m.schema_version, "level": m.level,
                "tier": m.tier, "provenance": m.provenance, "headline": m.headline, "tables": m.tables});
            (j, t)
        }
    };
    let g = Global {
        format: if json_flag { Format::Json } else { g.format },
        ..g.clone()
    };
    output(&g, &json, || text)?;
    Ok(exit::OK)
}

fn provenance_line(t: &mut String, p: &kiln_trace::Provenance) {
    writeln!(
        t,
        "prov      kiln {} (git {})  design {}  workload {}  calib {}  tier {:?}  trust {:?}",
        p.kiln_version,
        p.git_hash,
        p.design_hash,
        p.workload_hash,
        p.calibration_hash,
        p.tier,
        p.trust_level
    )
    .unwrap();
}

fn trace_validate(g: &Global, file: &Path, fail_code: u8) -> Result<u8, Failure> {
    let (kind, diags) = match load(file)? {
        Doc::Result(r) => ("result", check_result(&r)),
        Doc::Sim(s) => ("sim", check_sim(&s)),
        Doc::Meas(s) => ("measurement", s.validate()),
        Doc::Trace(m) => {
            let bytes = read(file)?;
            let mut d = container::verify_members(&bytes, &m);
            match container::read_kiln(&bytes) {
                Ok(t) => d.extend(kiln_trace::check::check_trace(&t)),
                Err(e) => d.push(e),
            }
            d.extend(
                [
                    m.headline.latency_s,
                    m.headline.tokens_per_s,
                    m.headline.energy_j,
                    m.headline.power_w,
                    m.headline.area_mm2,
                    m.headline.score,
                ]
                .into_iter()
                .flatten()
                .filter(|i| !i.is_valid())
                .map(|i| {
                    Diagnostic::error(
                        "E-TRACE-INTERVAL",
                        format!("invalid headline interval {i:?}"),
                    )
                }),
            );
            d.extend(
                m.tables
                    .iter()
                    .filter(|t| container::table_schema(&t.name).is_none())
                    .map(|t| {
                        Diagnostic::warning(
                            "W-TRACE-TABLE",
                            format!("unknown table {:?} ignored", t.name),
                        )
                    }),
            );
            ("trace", d)
        }
    };
    let errors = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .count();
    emit(g.format, &diags);
    let json = json!({"file": file.display().to_string(), "kind": kind, "ok": errors == 0, "diagnostics": diags});
    output(g, &json, || {
        if errors == 0 {
            format!(
                "ok        {} ({kind}, {} warnings)\n",
                file.display(),
                diags.len()
            )
        } else {
            format!("invalid   {} ({kind}, {errors} errors)\n", file.display())
        }
    })?;
    Ok(if errors == 0 { exit::OK } else { fail_code })
}
