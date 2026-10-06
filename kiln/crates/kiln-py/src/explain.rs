//! 06 §6.5 `Result.explain()`: status, score vs baseline, top errors with hints, dominant bound per phase,
//! tightest envelope margins. Bound text comes from 03's `explain_run` via `SimResult.bottleneck.summary`.

use std::fmt::Write as _;

use kiln_trace::Corner;
use kiln_trace::result::{EvalResult, ResultError, Status};

pub const DEFAULT_MAX_CHARS: usize = 1500;

fn status_word(s: Status) -> &'static str {
    match s {
        Status::Ok => "ok",
        Status::Invalid => "invalid",
        Status::Envelope => "envelope",
        Status::Infeasible => "infeasible",
        Status::FloorViolation => "floor_violation",
        Status::Pruned => "pruned",
        Status::Timeout => "timeout",
        Status::InternalError => "internal_error",
    }
}

fn error_line(e: &ResultError) -> String {
    let d = &e.diag;
    let mut s = format!("- {}", d.code);
    if let Some(p) = &d.path {
        write!(s, " at {p}").unwrap();
    }
    write!(s, ": {}", d.message).unwrap();
    if let Some(h) = &d.hint {
        write!(s, " -> {h}").unwrap();
    }
    s
}

/// Four significant digits; scientific outside `[1e-3, 1e5)`.
pub fn sig(x: f64) -> String {
    let a = x.abs();
    if x == 0.0 || !x.is_finite() {
        format!("{x}")
    } else if (1e-3..1e5).contains(&a) {
        let decimals = (3 - a.log10().floor() as i32).max(0) as usize;
        format!("{x:.decimals$}")
    } else {
        format!("{x:.3e}")
    }
}

fn pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

pub fn explain(r: &EvalResult, max_items: usize, max_chars: usize) -> String {
    let mut lines = vec![format!(
        "status: {} (stage {:?}{})",
        status_word(r.status),
        r.stage_reached,
        r.tier.map(|t| format!(", tier {t:?}")).unwrap_or_default()
    )];
    if r.status == Status::FloorViolation {
        lines.push("simulator bug; this design is quarantined".into());
    }
    match (&r.score_interval, &r.score_components) {
        (Some(i), Some(c)) => lines.push(format!(
            "score: {:.3} x {} [low {:.3}, high {:.3}]",
            r.score, c.baseline_id, i.low, i.high
        )),
        _ => lines.push(format!("score: {}", r.score)),
    }
    if let Some(rs) = &r.score_realistic {
        lines.push(format!(
            "realistic score (each design under its own software stack, {} vs {}): {:.3} [low {:.3}, high {:.3}]",
            rs.candidate_stack.split('@').next().unwrap_or_default(),
            rs.baseline_stack.split('@').next().unwrap_or_default(),
            rs.score,
            rs.interval.low,
            rs.interval.high
        ));
    }
    let errs: Vec<_> = r.violations.iter().chain(&r.errors).collect();
    if !errs.is_empty() {
        lines.push(format!("errors ({}):", errs.len()));
        lines.extend(errs.iter().take(max_items).map(|e| error_line(e)));
    }
    for p in r.phases.iter().take(max_items) {
        let summary = r
            .sim
            .iter()
            .find(|s| s.phase == p.phase && s.corner == Corner::Central)
            .map(|s| s.bottleneck.summary.trim())
            .filter(|s| !s.is_empty());
        let line = match summary {
            Some(s) => s.to_string(),
            None => {
                let top = p
                    .bound_breakdown
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)));
                match top {
                    Some((k, x)) => {
                        let what = k
                            .strip_prefix("mem:")
                            .map_or(k.clone(), |m| format!("{m} bandwidth"));
                        format!("{}: {} of time bound by {what}", p.phase, pct(*x))
                    }
                    None => format!("{}: no bound breakdown", p.phase),
                }
            }
        };
        lines.push(format!(
            "{line} ({} tok/s, roofline {})",
            sig(p.tokens_per_s.central),
            pct(p.roofline_frac)
        ));
    }
    if let Some(ph) = &r.physical {
        let mut m: Vec<_> = ph.margins_central.iter().collect();
        m.sort_by(|a, b| a.1.total_cmp(b.1).then_with(|| a.0.cmp(b.0)));
        if !m.is_empty() {
            let items: Vec<_> = m
                .iter()
                .take(3)
                .map(|(k, v)| format!("{k} {v:+.3}"))
                .collect();
            lines.push(format!("tightest envelope margins: {}", items.join(", ")));
        }
    }
    let warns = r.warnings.len();
    if warns > 0 {
        lines.push(format!(
            "warnings: {warns} (first: {})",
            r.warnings[0].diag.code
        ));
    }
    if !r.audit.reasons.is_empty() {
        lines.push(format!(
            "audit {:?}: {}",
            r.audit.status,
            r.audit.reasons.join("; ")
        ));
    }
    let mut out = String::new();
    for l in lines {
        if out.len() + l.len() + 1 > max_chars {
            let room = max_chars.saturating_sub(out.len() + 4);
            let cut = l
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|i| *i <= room)
                .last()
                .unwrap_or(0);
            out.push_str(&l[..cut]);
            out.push_str("...\n");
            break;
        }
        out.push_str(&l);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::result_with;
    use kiln_ir::common::Diagnostic;

    #[test]
    fn bound_lines_and_errors() {
        let mut r = result_with(&[("decode_b1", 80.0)], 800.0, 400.0);
        let t = explain(&r, 8, DEFAULT_MAX_CHARS);
        assert!(t.starts_with("status: ok"), "{t}");
        assert!(
            t.contains("decode_b1: 80% of time bound by chip0.hbm bandwidth"),
            "{t}"
        );
        assert!(t.contains("tightest envelope margins: tdp_w +0.000"), "{t}");
        r.status = Status::Invalid;
        r.errors.push(
            Diagnostic::error("E-IR-0300", "bad")
                .at("chip0.x")
                .hint("fix x")
                .into(),
        );
        let t = explain(&r, 8, DEFAULT_MAX_CHARS);
        assert!(t.contains("- E-IR-0300 at chip0.x: bad -> fix x"), "{t}");
    }

    #[test]
    fn length_capped() {
        let mut r = result_with(&[("decode_b1", 80.0)], 800.0, 400.0);
        for i in 0..200 {
            r.errors
                .push(Diagnostic::error("E-IR-0300", format!("error number {i} ü")).into());
        }
        let t = explain(&r, 500, 300);
        assert!(t.len() <= 300, "{}", t.len());
        assert!(t.ends_with("...\n"));
    }
}
