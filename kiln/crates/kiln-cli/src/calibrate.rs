//! `kiln calibrate split | fit | report` and `kiln validate --kind calibration` (06 §3, §7).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use kiln_calib::campaign::{self, Campaign, FitOptions, PLAN};
use kiln_calib::split::{DEFAULT_SALT, split_name};
use kiln_ir::common::Diagnostic;
use kiln_sim::CalibSet;

use crate::cli::{CalibrateCmd, Format, Global};
use crate::{Failure, emit, exit};

fn fail(code: u8) -> impl Fn(Diagnostic) -> Failure {
    move |d| Failure::new(code, d)
}

pub fn run(g: &Global, cmd: &CalibrateCmd) -> Result<u8, Failure> {
    match cmd {
        CalibrateCmd::Split { salt, .. } => split(g, salt.as_deref().unwrap_or(DEFAULT_SALT)),
        CalibrateCmd::Fit { scope, stages, bootstrap, .. } => fit(g, scope.as_deref(), stages.as_deref(), *bootstrap),
        CalibrateCmd::Policy { set } => policy(g, set.as_deref()),
        CalibrateCmd::Report { set, devices, tier, no_log, .. } => {
            if tier.as_deref() == Some("B") {
                return Err(Failure::new(exit::NOT_IMPLEMENTED, Diagnostic::error("E-NOT-IMPLEMENTED", "tier B residuals are M4")));
            }
            report(g, set.as_deref(), devices.as_deref(), !*no_log)
        }
    }
}

fn split(g: &Global, salt: &str) -> Result<u8, Failure> {
    let c = Campaign::load(salt).map_err(fail(exit::INPUT))?;
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (d, rs) in &c.records {
        for r in rs {
            let side = split_name(&r.split);
            let side = side.split(':').next().unwrap_or_default().to_string();
            *counts.entry((d.clone(), side)).or_default() += 1;
        }
    }
    campaign::write(&[], &c.split).map_err(fail(exit::INPUT))?;
    if g.format == Format::Text {
        println!("split {} ({}) -> {}", c.split.id, c.split.hash, campaign::split_path(&c.split.id).display());
        for ((d, s), n) in counts {
            println!("  {d:<10} {s:<12} {n}");
        }
    } else {
        println!("{}", serde_json::to_string(&c.split).expect("split serializes"));
    }
    Ok(exit::OK)
}

fn fit(g: &Global, scope: Option<&str>, stages: Option<&str>, bootstrap: Option<usize>) -> Result<u8, Failure> {
    let ids: Vec<&str> = match scope {
        None | Some("all") => PLAN.iter().map(|p| p.0).collect(),
        Some(s) => match PLAN.iter().find(|p| p.0 == s || kiln_calib::sets::file_id(p.0) == s) {
            Some(p) => vec![p.0],
            None => {
                return Err(Failure::new(
                    exit::USAGE,
                    Diagnostic::error("E-CLI-0002", format!("unknown --scope {s:?}")).hint("platform:a100_40gb, platform:tpu_v6e, generic-v1, generic-v2 or all"),
                ));
            }
        },
    };
    let mut opts = FitOptions { threads: g.threads, ..FitOptions::default() };
    if let Some(b) = bootstrap {
        opts.bootstrap = b;
    }
    if let Some(s) = stages {
        let want: Vec<String> = s.split(',').map(|x| x.trim().to_uppercase()).filter(|x| !x.is_empty()).collect();
        if let Some(bad) = want.iter().find(|x| !["P", "L", "D", "U"].contains(&x.as_str())) {
            return Err(Failure::new(exit::USAGE, Diagnostic::error("E-CLI-0002", format!("unknown stage {bad:?}")).hint("stages are P, L, D, U (run in that order)")));
        }
        opts.stages = want;
    }
    let c = Campaign::load(DEFAULT_SALT).map_err(fail(exit::INPUT))?;
    let (sets, fits) = campaign::fit_sets(&c, &ids, &opts).map_err(fail(exit::INTERNAL))?;
    let written = campaign::write(&sets, &c.split).map_err(fail(exit::INPUT))?;
    let mut text = String::new();
    for (dev, f) in &fits {
        let _ = writeln!(text, "== fit {dev} ({} micro records in stages)", f.stages.iter().map(|s| s.records.len()).sum::<usize>());
        for m in f.device.micro {
            if let Ok(Some(st)) = kiln_calib::records::session_state(f.device, m) {
                let gated = f.records.iter().filter(|r| r.session == *m && r.chip_state.is_some()).count();
                let _ = writeln!(text, "  chip state {m}: {}; {gated} compute-bound records excluded", st.summary());
            }
        }
        if let Some(cl) = &f.clock {
            let _ = writeln!(
                text,
                "  P  f_cap_op {} @ {}: {} telemetry points -> {} knots, {:.0}-{:.0} MHz, scale range [{:.4}, {:.4}]",
                cl.domain,
                cl.cap,
                cl.n,
                cl.points.len(),
                cl.points.last().map_or(0.0, |p| p[1] / 1e6),
                cl.points.first().map_or(0.0, |p| p[1] / 1e6),
                cl.range.0,
                cl.range.1
            );
        }
        for s in &f.stages {
            let mut abs: Vec<f64> = s.residuals.iter().map(|r| r.exp_m1().abs()).collect();
            abs.sort_by(f64::total_cmp);
            let med = abs.get(abs.len() / 2).copied().unwrap_or(f64::NAN);
            let p90 = abs.get(abs.len() * 9 / 10).copied().unwrap_or(f64::NAN);
            let _ = writeln!(text, "  {}  n={:<4} |err| median {:.1}% p90 {:.1}%{}", s.stage, s.records.len(), 100.0 * med, 100.0 * p90, s.refused.as_deref().map(|r| format!("  {r}")).unwrap_or_default());
            for (j, p) in s.free.iter().enumerate() {
                let _ = writeln!(
                    text,
                    "     {:<13} {:<40} {:>12.5e}  CI95 [{:.4e}, {:.4e}]{}",
                    p.name,
                    format!("{:?}", p.key),
                    s.value[j],
                    s.ci[j][0],
                    s.ci[j][1],
                    s.frozen[j].as_deref().map(|r| format!("\n       FROZEN: {r}")).unwrap_or_default()
                );
            }
        }
    }
    for (s, p) in sets.iter().zip(&written) {
        let _ = writeln!(text, "set {} {} -> {}", s.id, s.compute_hash(), p.display());
    }
    let failures: Vec<Diagnostic> = fits.values().flat_map(|f| f.failures.clone()).collect();
    match g.format {
        Format::Text => print!("{text}"),
        _ => println!("{}", serde_json::to_string(&sets).expect("sets serialize")),
    }
    if failures.is_empty() {
        Ok(exit::OK)
    } else {
        emit(g.format, &failures);
        Ok(exit::CHECK_FAILED)
    }
}

fn policy(g: &Global, set: Option<&str>) -> Result<u8, Failure> {
    let ids: Vec<&str> = match set {
        None | Some("all") => PLAN.iter().map(|p| p.0).filter(|id| !id.starts_with("platform:")).collect(),
        Some(s) => vec![s],
    };
    let sets = campaign::attach_policies(&ids).map_err(fail(exit::INPUT))?;
    match g.format {
        Format::Text => {
            for (s, p) in &sets {
                println!("set {} v{} {} -> {}", s.id, s.version, s.compute_hash(), p.display());
                for r in s.range_policy.iter().flat_map(|p| &p.ranges) {
                    println!("  {:<13} [{:.4e}, {:.4e}]", r.name, r.lower, r.upper);
                    for e in &r.evidence {
                        println!("    {:<22} [{:.4e}, {:.4e}] n={:<3} {} ({})", e.device, e.lower, e.upper, e.n, e.quantity, e.source);
                    }
                }
            }
        }
        _ => println!("{}", serde_json::to_string(&sets.iter().map(|s| &s.0).collect::<Vec<_>>()).expect("sets serialize")),
    }
    Ok(exit::OK)
}

fn report(g: &Global, set: Option<&str>, devices: Option<&str>, log: bool) -> Result<u8, Failure> {
    let ids: Vec<String> = match set {
        Some(s) => vec![s.to_string()],
        None => PLAN.iter().map(|p| p.0.to_string()).collect(),
    };
    let devs: Option<Vec<String>> = devices.map(|d| d.split(',').map(|x| x.trim().to_string()).collect());
    let mut any_fail = false;
    let mut json = vec![];
    for id in &ids {
        let (r, accesses) = campaign::report(id, devs.as_deref(), log).map_err(fail(exit::INPUT))?;
        any_fail |= r.verdicts.iter().any(|v| !v.pass);
        match g.format {
            Format::Text => {
                print!("{}", kiln_calib::report::render(&r));
                if accesses > 3 {
                    println!("  WARNING: {accesses} test accesses on this set version (> 3 fails CI, 06 §3.1 rule 4)");
                }
            }
            _ => json.push(serde_json::to_value(&r).expect("report serializes")),
        }
    }
    if g.format != Format::Text {
        println!("{}", serde_json::Value::from(json));
    }
    Ok(if any_fail { exit::CHECK_FAILED } else { exit::OK })
}

/// `kiln validate --kind calibration <file>`.
pub fn validate(g: &Global, file: &Path) -> Result<u8, Failure> {
    let set = CalibSet::load(file).map_err(fail(exit::CHECK_FAILED))?;
    if g.format == Format::Text {
        println!("ok: {} v{} ({:?}, {} parameters) {}", set.id, set.version, set.kind, set.parameters.len(), set.compute_hash());
    }
    Ok(exit::OK)
}
