mod calibrate;
mod cli;
mod commands;
mod eval;
mod ir;
mod viz;

use std::process::ExitCode;

use clap::Parser;
use kiln_ir::common::{Diagnostic, Severity};

use cli::{Cli, Format};

/// 06 §7 exit codes. `NOT_IMPLEMENTED` is not in the spec table (sysexits `EX_UNAVAILABLE`).
pub mod exit {
    pub const OK: u8 = 0;
    pub const COMPLETED_WITH_INVALID: u8 = 1;
    pub const USAGE: u8 = 2;
    pub const INPUT: u8 = 3;
    pub const CHECK_FAILED: u8 = 4;
    pub const NOT_IMPLEMENTED: u8 = 69;
    pub const INTERNAL: u8 = 70;
}

/// A failed command: diagnostics to print and the exit code.
pub struct Failure {
    pub code: u8,
    pub diags: Vec<Diagnostic>,
}

impl Failure {
    pub fn new(code: u8, diag: Diagnostic) -> Self {
        Self {
            code,
            diags: vec![diag],
        }
    }
}

pub fn emit(format: Format, diags: &[Diagnostic]) {
    for d in diags {
        if matches!(format, Format::Json | Format::Jsonl) {
            eprintln!(
                "{}",
                serde_json::to_string(d).expect("diagnostic serializes")
            );
        } else {
            let sev = if d.severity == Severity::Error {
                "error"
            } else {
                "warning"
            };
            eprintln!("{sev}[{}]: {}", d.code, d.message);
            if let Some(p) = &d.path {
                eprintln!("  at: {p}");
            }
            if let Some(h) = &d.hint {
                eprintln!("  hint: {h}");
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(if e.use_stderr() {
                exit::USAGE
            } else {
                exit::OK
            });
        }
    };
    let format = cli.global.format;
    let code = match std::panic::catch_unwind(|| commands::run(&cli)) {
        Ok(Ok(code)) => code,
        Ok(Err(f)) => {
            emit(format, &f.diags);
            f.code
        }
        Err(_) => {
            emit(
                format,
                &[Diagnostic::error("E-INTERNAL", "kiln panicked")
                    .hint("this is a kiln bug; please report it")],
            );
            exit::INTERNAL
        }
    };
    ExitCode::from(code)
}
