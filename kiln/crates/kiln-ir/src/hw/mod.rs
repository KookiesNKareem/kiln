//! Hardware IR (spec 01): authoring, canonical form and hashing, expansion to [`HwModel`], validation.
//!
//! Pipeline (01 §15): [`author`] (steps 1-5) -> typed [`HwDoc`] (6) -> canonical form + `design_hash` (7) ->
//! [`expand`] (8-11) -> [`validate`] (12). [`check_file`] / [`check_str`] run all of it.

pub mod author;
pub mod canon;
pub mod compute;
pub mod diag;
pub mod expand;
pub mod expr;
pub mod legacy;
pub mod model;
pub mod net;
pub mod phys;
pub mod quantity;
pub mod select;
pub mod types;
pub mod validate;

use std::path::Path as FsPath;

use indexmap::IndexMap;
use serde_json::Value;

pub use author::{FsLoader, Loader, MemLoader};
pub use canon::{HwDiff, diff};
pub use compute::ComputeKind;
pub use expand::ExpandOptions;
pub use model::{HwModel, HwSummary, NodeIx};
pub use types::HwDoc;
pub use validate::{PricedField, Pricer, Profile};

use crate::common::{Diagnostic, Severity};
use compute::{ComputeUnit, DataflowChoice};
use types::Contents;

/// A loaded design in canonical compact form (pipeline step 7).
#[derive(Clone, Debug)]
pub struct Design {
    pub doc: HwDoc,
    pub canonical: Value,
    /// `hw1-` + 32 hex chars of sha256 over the canonical form without `meta` (01 §17).
    pub hash: String,
    /// Per-entity `notes` strings from the authoring source; not hashed.
    pub notes: IndexMap<String, String>,
    pub warnings: Vec<Diagnostic>,
}

impl Design {
    /// Steps 1-7 from authoring text. `key` names the document for relative `imports`/`extends`.
    pub fn from_source(loader: &dyn Loader, key: Option<&str>, text: &str) -> Result<Self, Vec<Diagnostic>> {
        let a = author::author(loader, key, text)?;
        let mut d = Self::typed(a.value, &a.patched_fields)?;
        d.notes = a.notes;
        d.warnings = a.warnings;
        Ok(d)
    }

    /// Steps 6-7 from an already-authored value (canonical JSON, archive entries).
    pub fn from_value(v: Value) -> Result<Self, Vec<Diagnostic>> {
        Self::typed(v, &[])
    }

    fn typed(mut v: Value, patched: &[String]) -> Result<Self, Vec<Diagnostic>> {
        author::shorthands(&mut v).map_err(|d| vec![d])?;
        let mut doc: HwDoc = diag::from_value(v, "").map_err(|mut d| {
            if d.code == "E-IR-0101" && patched.iter().any(|f| d.message.contains(&format!("`{f}`"))) {
                d.code = "E-IR-0212".into();
                d.hint = Some("a `set` patch targets a field the entity does not have".into());
            }
            vec![d]
        })?;
        if doc.schema != types::SCHEMA_CURRENT && doc.schema != "kiln.hw/1" {
            return Err(vec![
                Diagnostic::error("E-IR-0107", format!("unknown schema version {:?}", doc.schema))
                    .hint("supported: kiln.hw/1.0; legacy kiln.hw/0 harness JSON is migrated on load"),
            ]);
        }
        doc.schema = types::SCHEMA_CURRENT.into();
        materialize_defaults(&mut doc);
        let canonical = canon::to_canonical_value(&doc);
        let hash = canon::design_hash(&canonical);
        Ok(Self { doc, canonical, hash, notes: IndexMap::new(), warnings: vec![] })
    }

    pub fn canonical_json(&self) -> String {
        crate::common::canonical_json(&self.canonical)
    }

    pub fn expand(&self, opts: &ExpandOptions) -> Result<(HwModel, Vec<Diagnostic>), Vec<Diagnostic>> {
        expand::expand(&self.doc, &self.hash, opts)
    }

    pub fn diff(&self, other: &Design) -> HwDiff {
        canon::diff(&self.canonical, &other.canonical)
    }
}

/// Everything `kiln validate` reports for one design.
#[derive(Debug)]
pub struct Report {
    pub design: Option<Design>,
    pub model: Option<HwModel>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Report {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.severity == Severity::Error)
    }

    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(|d| d.severity == Severity::Error)
    }

    pub fn codes(&self) -> Vec<&str> {
        self.diagnostics.iter().map(|d| d.code.as_str()).collect()
    }
}

/// Full pipeline on a loaded design: expand, then structural validation under `profile`.
pub fn check(design: Design, profile: Profile, opts: &ExpandOptions) -> Report {
    check_priced(design, profile, opts, None)
}

/// Builds a [`Pricer`] for an expanded model (kiln-phys derives energies, areas and latencies from structure).
pub type PricerFactory<'a> = &'a dyn Fn(&HwModel) -> Box<dyn Pricer>;

/// [`check`] where `search` compares "less is better" overrides with the values `pricer` derives (01 §18.3).
pub fn check_priced(design: Design, profile: Profile, opts: &ExpandOptions, pricer: Option<PricerFactory>) -> Report {
    let mut diagnostics = design.warnings.clone();
    match design.expand(opts) {
        Ok((model, warnings)) => {
            diagnostics.extend(warnings);
            let p = pricer.filter(|_| profile == Profile::Search).map(|f| f(&model));
            diagnostics.extend(validate::validate_priced(&design.doc, &model, profile, p.as_deref()));
            Report { design: Some(design), model: Some(model), diagnostics }
        }
        Err(errs) => {
            diagnostics.extend(errs);
            Report { design: Some(design), model: None, diagnostics }
        }
    }
}

pub fn check_str(loader: &dyn Loader, key: Option<&str>, text: &str, profile: Profile) -> Report {
    match Design::from_source(loader, key, text) {
        Ok(d) => check(d, profile, &ExpandOptions::default()),
        Err(diagnostics) => Report { design: None, model: None, diagnostics },
    }
}

pub fn load_file(path: impl AsRef<FsPath>) -> Result<Design, Vec<Diagnostic>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        vec![Diagnostic::error("E-IR-0209", format!("cannot read {}: {e}", path.display()))]
    })?;
    let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Design::from_source(&FsLoader, Some(&key.to_string_lossy()), &text)
}

pub fn check_file(path: impl AsRef<FsPath>, profile: Profile) -> Report {
    match load_file(path) {
        Ok(d) => check(d, profile, &ExpandOptions::default()),
        Err(diagnostics) => Report { design: None, model: None, diagnostics },
    }
}

/// `kiln import harness-design`: a `kiln.hw/0` harness JSON migrated to a loaded `kiln.hw/1.0` design.
pub fn import_harness(json: &str) -> Result<Design, Vec<Diagnostic>> {
    let v: Value = serde_json::from_str(json)
        .map_err(|e| vec![Diagnostic::error("E-IR-0100", format!("harness design is not JSON: {e}"))])?;
    let migrated = legacy::migrate_v0(&v).map_err(|d| vec![d])?;
    let text = serde_json::to_string(&migrated).expect("value serializes");
    Design::from_source(&MemLoader::default(), None, &text)
}

/// Defaults that depend on other fields are written explicitly so equal designs hash equally (01 §17 rule 4).
fn materialize_defaults(doc: &mut HwDoc) {
    fn contents(c: &mut Contents) {
        c.units.iter_mut().for_each(unit);
        c.clusters.iter_mut().for_each(|cl| contents(&mut cl.contents));
    }
    fn unit(u: &mut ComputeUnit) {
        if let ComputeKind::Matrix(m) = &mut u.kind
            && m.dataflow.is_none()
        {
            m.dataflow = Some(DataflowChoice::One(m.geometry.default_dataflow()));
        }
    }
    for b in &mut doc.system.boards {
        for p in &mut b.packages {
            p.dies.iter_mut().for_each(|d| contents(&mut d.contents));
            for s in &mut p.mem_stacks {
                if let Some(ld) = &mut s.logic_die {
                    contents(&mut ld.contents);
                }
            }
        }
    }
}
