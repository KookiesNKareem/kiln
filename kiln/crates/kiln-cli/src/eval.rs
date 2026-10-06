//! `kiln eval`, `kiln compare` and `kiln explain` over `kiln_py::Session` (06 §6-7).

use std::fmt::Write as _;
use std::path::Path;

use kiln_ir::common::{Diagnostic, canonical_json};
use kiln_py::engine::NOT_IMPLEMENTED;
use kiln_py::explain::{DEFAULT_MAX_CHARS, explain, sig};
use kiln_py::inputs::{self, DesignInput, WorkloadInput, WorkloadSet};
use kiln_py::{Options, Session, SessionConfig};
use kiln_trace::Interval;
use kiln_trace::result::{EvalResult, PhaseResult, Status};
use serde_json::{Value, json};

use crate::cli::{EvalArgs, Format, Global};
use crate::commands::{read, write_out};
use crate::{Failure, exit};

const OPTS_CODE: &str = "E-CLI-0002";
const DEFAULT_WORKLOAD: &str = "standard";

fn usage(msg: impl Into<String>, hint: &str) -> Failure {
    Failure::new(exit::USAGE, Diagnostic::error(OPTS_CODE, msg).hint(hint))
}

/// `--fitness` is inline JSON or a path to a JSON file.
fn json_arg(s: &str) -> Result<Value, Failure> {
    let text = if s.trim_start().starts_with('{') {
        s.to_string()
    } else {
        String::from_utf8_lossy(&read(Path::new(s))?).into_owned()
    };
    serde_json::from_str(&text).map_err(|e| {
        usage(
            format!("--fitness is not JSON: {e}"),
            "pass a JSON object or a path",
        )
    })
}

pub fn options(a: &EvalArgs) -> Result<Options, Failure> {
    let mut o = json!({});
    if let Some(t) = &a.tier {
        o["tier"] = json!(match t.as_str() {
            "a" | "A" => "A",
            "b" | "B" => "B",
            other => other,
        });
    }
    if let Some(f) = &a.fitness {
        o["fitness"] = json_arg(f)?;
    }
    if let Some(b) = &a.baseline {
        if o.get("fitness").is_none() {
            o["fitness"] = json!({});
        }
        o["fitness"]["baseline"] = json!(b);
    }
    if let Some(s) = &a.seeds {
        let seeds: Result<Vec<u64>, _> = s.split(',').map(|x| x.trim().parse::<u64>()).collect();
        o["seeds"] = json!(seeds.map_err(|_| usage(
            format!("--seeds {s:?} is not a comma-separated list of integers"),
            "e.g. --seeds 0,1,2"
        ))?);
    }
    if let Some(t) = a.timeout {
        o["timeout_s"] = json!({"A": t, "B": t});
    }
    if let Some(t) = &a.trace {
        o["trace"] = json!(t);
    }
    if let Some(st) = &a.stack {
        o["stack"] = json!(st);
    }
    o["profile"] = json!(a.profile);
    Options::from_value(&o).map_err(|d| Failure::new(exit::USAGE, d))
}

fn session(g: &Global) -> Result<Session, Failure> {
    Session::new(SessionConfig {
        calibration: g.calib.clone(),
        cache_dir: g.cache_dir.clone(),
        no_cache: g.no_cache,
        threads: g.threads,
        designs_dir: None,
    })
    .map_err(|d| Failure::new(exit::INPUT, d))
}

/// `--workload` (default `standard`) narrowed by `--scenario`; unknown names and unreadable files exit 3.
pub fn workload(a: &EvalArgs) -> Result<WorkloadSet, Failure> {
    let name = a.workload.as_deref().unwrap_or(DEFAULT_WORKLOAD);
    let is_file = Path::new(name).is_file();
    let input = match &a.scenario {
        Some(sc) if !is_file && !name.contains(':') => format!("{name}:{sc}"),
        _ => name.to_string(),
    };
    let mut set = inputs::resolve_workload(&WorkloadInput::Str(input))
        .map_err(|d| Failure::new(exit::INPUT, d))?;
    if let Some(sc) = &a.scenario {
        set.members.retain(|m| m.scenario.as_str() == sc);
        if set.members.is_empty() {
            return Err(Failure::new(
                exit::INPUT,
                Diagnostic::error(
                    inputs::WORKLOAD_CODE,
                    format!("workload {name} has no scenario {sc:?}"),
                )
                .at("--scenario"),
            ));
        }
        set.name = format!("{}/{sc}", set.name);
    }
    Ok(set)
}

/// Parse-level failures (unreadable file, JSON5 syntax, schema) exit 3 before evaluation (06 §7).
fn design_input(s: &str) -> Result<DesignInput, Failure> {
    let d = DesignInput::Str(s.to_string());
    inputs::load_design(&d, &inputs::default_designs_dir()).map_err(|diags| Failure {
        code: exit::INPUT,
        diags,
    })?;
    Ok(d)
}

pub fn result_exit(r: &EvalResult) -> u8 {
    if r.errors.iter().any(|e| e.diag.code == NOT_IMPLEMENTED) {
        exit::NOT_IMPLEMENTED
    } else {
        r.status.exit_code()
    }
}

fn status_word(s: Status) -> String {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}

fn iv(i: &Interval) -> String {
    if i.low == i.high {
        sig(i.central)
    } else {
        format!("{} [{}, {}]", sig(i.central), sig(i.low), sig(i.high))
    }
}

fn dominant(p: &PhaseResult) -> String {
    p.bound_breakdown
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map_or_else(|| "-".into(), |(k, x)| format!("{k} {:.0}%", 100.0 * x))
}

fn name_of(arg: &str) -> String {
    Path::new(arg)
        .file_stem()
        .map_or_else(|| arg.to_string(), |s| s.to_string_lossy().into_owned())
}

fn stack_id(label: &str) -> &str {
    label.split('@').next().unwrap_or(label)
}

fn score_line(r: &EvalResult) -> String {
    let c = r.score_components.as_ref();
    let base = c
        .map(|c| format!(" vs {}", c.baseline_id))
        .unwrap_or_default();
    let stack = c
        .and_then(|c| {
            c.candidate_stack
                .as_deref()
                .zip(c.baseline_stack.as_deref())
        })
        .map(|(a, b)| {
            if a == b {
                format!(" ({} both)", stack_id(a))
            } else {
                format!(" ({} vs {})", stack_id(a), stack_id(b))
            }
        })
        .unwrap_or_default();
    match &r.score_interval {
        Some(i) => format!("{}{base}{stack}", iv(i)),
        None => sig(r.score),
    }
}

/// The realistic-stack score (08 §F): each design under its own execution model's default stack.
fn realistic_line(r: &EvalResult) -> String {
    r.score_realistic.as_ref().map_or_else(
        || "-".into(),
        |rs| {
            format!(
                "{} ({} vs {})",
                iv(&rs.interval),
                stack_id(&rs.candidate_stack),
                stack_id(&rs.baseline_stack)
            )
        },
    )
}

fn diag_lines(t: &mut String, r: &EvalResult) {
    for e in r.violations.iter().chain(&r.errors).chain(&r.warnings) {
        let d = &e.diag;
        let sev = if d.severity == kiln_ir::common::Severity::Error {
            "error"
        } else {
            "warning"
        };
        write!(t, "{sev}[{}]: {}", d.code, d.message).unwrap();
        if let Some(p) = &d.path {
            write!(t, "  at {p}").unwrap();
        }
        t.push('\n');
        if let Some(h) = &d.hint {
            writeln!(t, "  hint: {h}").unwrap();
        }
    }
}

pub fn eval_text(r: &EvalResult, design: &str, wl: &str, verbose: bool) -> String {
    let p = &r.provenance;
    let mut t = String::new();
    writeln!(
        t,
        "design    {design}  {}\nworkload  {wl}  {}\ncalib     {}  {}",
        p.design_hash,
        p.workload_hash,
        p.calibration_id.as_deref().unwrap_or("-"),
        p.calibration_hash
    )
    .unwrap();
    writeln!(
        t,
        "status    {}  stage {:?}  tier {:?}  audit {}\nscore     {}\nrealistic {}",
        status_word(r.status),
        r.stage_reached,
        p.tier,
        serde_json::to_value(r.audit.status)
            .unwrap_or_default()
            .as_str()
            .unwrap_or("-"),
        score_line(r),
        realistic_line(r)
    )
    .unwrap();
    if !r.phases.is_empty() {
        let rows: Vec<[String; 7]> = r
            .phases
            .iter()
            .map(|ph| {
                [
                    ph.phase.to_string(),
                    format!("{:?}", ph.scope).to_lowercase(),
                    iv(&ph.time_s),
                    iv(&ph.tokens_per_s),
                    iv(&ph.tokens_per_j),
                    sig(ph.avg_power_w.central),
                    dominant(ph),
                ]
            })
            .collect();
        table(
            &mut t,
            &[
                "phase", "scope", "time_s", "tokens/s", "tokens/J", "power_W", "bound",
            ],
            &rows,
        );
    }
    if let Some(ph) = &r.physical {
        let die: f64 = ph.die_mm2.values().map(|d| d.central).sum();
        writeln!(
            t,
            "physical  die {} mm^2  tdp {} W  node {}",
            sig(die),
            sig(ph.tdp_w),
            ph.node
        )
        .unwrap();
    }
    if verbose && !r.ops.is_empty() {
        let rows: Vec<[String; 5]> = r
            .ops
            .iter()
            .map(|o| {
                [
                    o.phase.to_string(),
                    o.op.to_string(),
                    sig(o.time_s),
                    sig(o.energy_j),
                    format!(
                        "{:?}{}",
                        o.bound,
                        o.bound_resource
                            .as_ref()
                            .map(|x| format!(" {x}"))
                            .unwrap_or_default()
                    )
                    .to_lowercase(),
                ]
            })
            .collect();
        table(
            &mut t,
            &["phase", "op", "time_s", "energy_J", "bound"],
            &rows,
        );
    }
    diag_lines(&mut t, r);
    t
}

fn table<const N: usize>(t: &mut String, head: &[&str; N], rows: &[[String; N]]) {
    let mut w = head.map(str::len);
    for r in rows {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(c.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let mut s = String::new();
        for (i, c) in cells.iter().enumerate() {
            if i + 1 == N {
                s.push_str(c);
            } else {
                write!(s, "{c:<width$}  ", width = w[i]).unwrap();
            }
        }
        s.trim_end().to_string() + "\n"
    };
    t.push_str(&line(head.to_vec()));
    for r in rows {
        t.push_str(&line(r.iter().map(String::as_str).collect()));
    }
}

/// Per-phase tokens/s with ratio intervals vs the first design (candidate corners over the first's central).
pub fn compare_text(rs: &[EvalResult], names: &[String], wl: &str) -> String {
    let mut t = format!("workload  {wl}  (ratios vs {})\n", names[0]);
    let rows: Vec<[String; 5]> = rs
        .iter()
        .zip(names)
        .map(|(r, n)| {
            [
                n.clone(),
                status_word(r.status),
                score_line(r),
                realistic_line(r),
                r.provenance.design_hash.clone(),
            ]
        })
        .collect();
    table(
        &mut t,
        &["design", "status", "score", "realistic", "hash"],
        &rows,
    );
    let first = &rs[0];
    let mut phases: Vec<&str> = Vec::new();
    for r in rs {
        for p in &r.phases {
            if !phases.contains(&p.phase.as_str()) {
                phases.push(p.phase.as_str());
            }
        }
    }
    for ph in phases {
        writeln!(t, "\nphase {ph}").unwrap();
        let base = first.phase(ph).map(|p| p.tokens_per_s.central);
        let rows: Vec<[String; 5]> = rs
            .iter()
            .zip(names)
            .map(|(r, n)| match r.phase(ph) {
                None => [n.clone(), "-".into(), "-".into(), "-".into(), "-".into()],
                Some(p) => {
                    let ratio = base.filter(|b| *b > 0.0).map_or("-".into(), |b| {
                        let i = &p.tokens_per_s;
                        iv(&Interval::from_corners(
                            i.central / b,
                            i.low / b,
                            i.high / b,
                        ))
                    });
                    [
                        n.clone(),
                        iv(&p.tokens_per_s),
                        ratio,
                        iv(&p.time_s),
                        dominant(p),
                    ]
                }
            })
            .collect();
        table(
            &mut t,
            &["design", "tokens/s", "ratio", "time_s", "bound"],
            &rows,
        );
    }
    let mut diags = String::new();
    for (r, n) in rs.iter().zip(names) {
        let mut d = String::new();
        diag_lines(&mut d, r);
        if !d.is_empty() {
            write!(diags, "\n{n}:\n{d}").unwrap();
        }
    }
    t + &diags
}

fn emit_doc(g: &Global, doc: &Value, text: impl FnOnce() -> String) -> Result<(), Failure> {
    let body = match g.format {
        Format::Json => canonical_json(doc) + "\n",
        Format::Jsonl => match doc {
            Value::Array(a) => a.iter().map(|x| canonical_json(x) + "\n").collect(),
            v => canonical_json(v) + "\n",
        },
        Format::Text | Format::Llm => text(),
    };
    match &g.out {
        Some(p) => write_out(p, &body),
        None => {
            print!("{body}");
            Ok(())
        }
    }
}

pub fn eval(g: &Global, a: &EvalArgs) -> Result<u8, Failure> {
    let [design] = a.designs.as_slice() else {
        return Err(usage(
            format!("kiln eval takes one design, got {}", a.designs.len()),
            "use `kiln compare <design>...` for several",
        ));
    };
    let mut opts = options(a)?;
    let wl = workload(a)?;
    let d = design_input(design)?;
    let s = session(g)?;
    // `-o run.kiln` writes the trace (05 §3.3); the engine records ops so the summary tables are complete.
    let kiln_out = g
        .out
        .as_ref()
        .filter(|p| p.extension().is_some_and(|e| e == "kiln"))
        .cloned();
    let level = opts.trace;
    if kiln_out.is_some() {
        if level == kiln_trace::TraceLevel::None {
            return Err(usage(
                "--trace none writes no trace",
                "drop --trace none or write JSON with -o result.json",
            ));
        }
        opts.trace = opts.trace.max(kiln_trace::TraceLevel::Ops);
    }
    let r = s.evaluate_set(&d, &wl, &opts);
    let name = name_of(design);
    if let Some(p) = kiln_out {
        let t = crate::viz::trace_for_eval(
            &r,
            design,
            &wl.name,
            level.min(kiln_trace::TraceLevel::Ops),
        )?;
        crate::viz::write_trace(&p, &t)?;
        let g = Global {
            out: None,
            format: if g.format == Format::Json {
                Format::Text
            } else {
                g.format
            },
            ..g.clone()
        };
        emit_doc(&g, &r.to_value(), || match g.format {
            Format::Llm => explain(&r, 8, DEFAULT_MAX_CHARS),
            _ => {
                eval_text(&r, &name, &wl.name, a.verbose) + &format!("trace     {}\n", p.display())
            }
        })?;
        return Ok(result_exit(&r));
    }
    emit_doc(g, &r.to_value(), || match g.format {
        Format::Llm => explain(&r, 8, DEFAULT_MAX_CHARS),
        _ => eval_text(&r, &name, &wl.name, a.verbose),
    })?;
    Ok(result_exit(&r))
}

pub fn compare(g: &Global, a: &EvalArgs) -> Result<u8, Failure> {
    if a.designs.len() < 2 {
        return Err(usage(
            "kiln compare needs at least two designs",
            "kiln compare <A> <B> ... --workload <name>",
        ));
    }
    let opts = options(a)?;
    let wl = workload(a)?;
    let ds = a
        .designs
        .iter()
        .map(|d| design_input(d))
        .collect::<Result<Vec<_>, _>>()?;
    let s = session(g)?;
    let items: Vec<_> = ds.into_iter().map(|d| (d, wl.clone())).collect();
    let rs: Vec<EvalResult> = std::thread::scope(|sc| {
        let hs: Vec<_> = items
            .iter()
            .map(|(d, w)| sc.spawn(|| s.evaluate_set(d, w, &opts)))
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("evaluation threads catch panics"))
            .collect()
    });
    let names: Vec<String> = a.designs.iter().map(|d| name_of(d)).collect();
    let doc = Value::Array(rs.iter().map(EvalResult::to_value).collect());
    emit_doc(g, &doc, || match g.format {
        Format::Llm => rs
            .iter()
            .zip(&names)
            .map(|(r, n)| format!("## {n}\n{}", explain(r, 8, DEFAULT_MAX_CHARS)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => compare_text(&rs, &names, &wl.name),
    })?;
    Ok(rs.iter().map(result_exit).max().unwrap_or(exit::OK))
}

pub fn explain_cmd(g: &Global, file: &Path, max_items: Option<usize>) -> Result<u8, Failure> {
    let bytes = read(file)?;
    let r: EvalResult = serde_json::from_slice(&bytes).map_err(|e| {
        Failure::new(
            exit::INPUT,
            Diagnostic::error(
                "E-TRACE-SCHEMA",
                format!("not a kiln.result/1 document: {e}"),
            )
            .at(file.display().to_string()),
        )
    })?;
    let text = explain(&r, max_items.unwrap_or(8), DEFAULT_MAX_CHARS);
    emit_doc(
        g,
        &json!({"file": file.display().to_string(), "explain": text}),
        || text.clone(),
    )?;
    Ok(exit::OK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_ir::common::Id;

    fn res(phases: &[(&str, f64)], score: f64) -> EvalResult {
        let mut r: EvalResult = serde_json::from_str(include_str!(
            "../../kiln-trace/tests/golden/result_min.json"
        ))
        .unwrap();
        let tmpl = r.phases[0].clone();
        r.phases = phases
            .iter()
            .map(|(id, tps)| PhaseResult {
                phase: Id::new(*id).unwrap(),
                tokens_per_s: Interval {
                    low: 0.9 * tps,
                    central: *tps,
                    high: 1.1 * tps,
                },
                ..tmpl.clone()
            })
            .collect();
        r.score = score;
        r
    }

    #[test]
    fn eval_text_has_phase_table_and_score() {
        let r = res(&[("decode_b1", 80.0), ("prefill_b1", 1234.5)], 1.05);
        let t = eval_text(&r, "a100", "standard", false);
        assert!(
            t.contains("status    ok  stage S3  tier A  audit not_run"),
            "{t}"
        );
        assert!(t.contains("score     1.050 [0.9500, 1.100]"), "{t}");
        assert!(t.contains("phase       scope  time_s"), "{t}");
        assert!(
            t.contains("decode_b1   step   0.01250 [0.01200, 0.01400]  80.00 [72.00, 88.00]"),
            "{t}"
        );
        assert!(t.contains("1234 [1111, 1358]"), "{t}");
        assert!(t.contains("mem:chip0.hbm 75%"), "{t}");
    }

    #[test]
    fn compare_ratios_vs_first() {
        let a = res(&[("decode_b1", 80.0)], 1.0);
        let b = res(&[("decode_b1", 160.0), ("decode_b8", 10.0)], 1.0);
        let t = compare_text(&[a, b], &["a".into(), "b".into()], "standard");
        assert!(t.contains("ratios vs a"), "{t}");
        assert!(t.contains("1.000 [0.9000, 1.100]"), "{t}");
        assert!(t.contains("2.000 [1.800, 2.200]"), "{t}");
        assert!(t.contains("phase decode_b8"), "{t}");
        assert!(
            t.lines().any(|l| l.starts_with("a ") && l.contains(" -")),
            "{t}"
        );
    }

    #[test]
    fn exit_codes() {
        let mut r = res(&[("decode_b1", 80.0)], 1.0);
        assert_eq!(result_exit(&r), 0);
        r.status = Status::FloorViolation;
        assert_eq!(result_exit(&r), 7);
        r.status = Status::InternalError;
        r.errors
            .push(Diagnostic::error(NOT_IMPLEMENTED, "x").into());
        assert_eq!(result_exit(&r), exit::NOT_IMPLEMENTED);
    }
}
