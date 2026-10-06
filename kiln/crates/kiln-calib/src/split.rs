//! Fit / test splits and leakage rules (06 §3.4): LLM-suite ops and whole steps are test-only, always; a
//! calib-micro contraction whose shape family equals a test op's family is dropped; 30% of the remaining
//! contraction families are held out by a salted hash as a diagnostic split. Splits are hashed files.

use std::collections::{BTreeMap, BTreeSet};

use kiln_ir::common::content_hash;
use serde::Serialize;
use serde_json::json;

use crate::records::{Kind, Record, Split, StreamOp};

pub const SPLIT_SCHEMA: &str = "kiln.split/1";
pub const DEFAULT_SALT: &str = "m2-2026-10-05";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SplitFile {
    pub schema: String,
    pub id: String,
    pub salt: String,
    pub rule: String,
    pub families: BTreeMap<String, Vec<String>>,
    /// Record hash -> split name.
    pub records: BTreeMap<String, String>,
    pub hash: String,
}

/// Families of every contraction in the test files (LLM suite ops and the suite files' other sweeps).
pub fn test_families(recs: &[Record]) -> BTreeSet<String> {
    recs.iter().filter(|r| r.split == Split::Test).filter_map(Record::family).collect()
}

/// Diagnostic hold-out: `sha256(family + salt) mod 10 < 3`.
pub fn diag_holdout(family: &str, salt: &str) -> bool {
    let h = content_hash("", &json!(format!("{family}{salt}")));
    u64::from_str_radix(&h[h.len() - 8..], 16).unwrap_or(0) % 10 < 3
}

/// Assigns the split of every fit candidate in place. Test records are never touched.
pub fn assign(recs: &mut [Record], tests: &BTreeSet<String>, salt: &str) {
    for r in recs.iter_mut() {
        if r.split == Split::Test {
            continue;
        }
        let static_df = !r.device.starts_with("a100");
        r.split = match &r.kind {
            _ if !r.quality_ok => Split::Excluded("quality: rejected by the runner (cv above its gate)".into()),
            _ if r.chip_state.is_some() => {
                Split::Excluded(format!("chip state: compute-bound record of a session whose {}", r.chip_state.as_deref().unwrap_or_default()))
            }
            Kind::OnChip => Split::Excluded("on-chip resident sweep: no registered parameter describes it".into()),
            Kind::Stream { op: StreamOp::Write, .. } => Split::Excluded("fill (write without read) has no IR op".into()),
            Kind::Launch { kernel, .. } if kernel != "empty" => {
                Split::Excluded("chain reuses resident operands; isolated programs are cold (E-CAL-MODE)".into())
            }
            Kind::Launch { chain, .. } if static_df && *chain > 1 => {
                Split::Excluded("in-program issue rate: no registered static_dataflow parameter".into())
            }
            Kind::Contraction { .. } => match r.family() {
                Some(f) if tests.contains(&f) => Split::Excluded(format!("leakage: shape family {f} of a test op")),
                Some(f) if diag_holdout(&f, salt) => Split::FitHoldout,
                _ => Split::Fit,
            },
            _ => Split::Fit,
        };
    }
}

pub fn split_name(s: &Split) -> String {
    match s {
        Split::Fit => "fit".into(),
        Split::FitHoldout => "fit_holdout".into(),
        Split::Test => "test".into(),
        Split::Excluded(why) => format!("excluded: {why}"),
    }
}

pub fn split_file(id: &str, salt: &str, recs: &[Record]) -> SplitFile {
    let mut families: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut records = BTreeMap::new();
    for r in recs {
        let side = match &r.split {
            Split::Excluded(_) => "excluded".to_string(),
            s => split_name(s),
        };
        if let Some(f) = r.family() {
            families.entry(side).or_default().insert(f);
        }
        records.insert(r.hash.clone(), split_name(&r.split));
    }
    let families: BTreeMap<String, Vec<String>> = families.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
    let rule = "test: suite + sequence files; leakage: contraction family (kind, m bucket {1,2-16,17-128,>128}, n, k, batch bucket, \
                dtype) of any test op; diagnostic: sha256(family+salt) mod 10 < 3"
        .to_string();
    let body = json!({"schema": SPLIT_SCHEMA, "id": id, "salt": salt, "rule": rule, "families": families, "records": records});
    SplitFile { schema: SPLIT_SCHEMA.into(), id: id.into(), salt: salt.into(), rule, families, records, hash: content_hash("split1-", &body) }
}
