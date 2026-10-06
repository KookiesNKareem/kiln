//! Diagnostic plumbing: carrying structured codes through serde errors, and per-entity deduplication (01 §18.1).

use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::common::{Diagnostic, Severity};

const SEP: char = '\u{1f}';

/// Encodes a diagnostic into a serde custom error so its code survives typed deserialization.
pub fn de_error<E: serde::de::Error>(d: Diagnostic) -> E {
    E::custom(format!("{}{SEP}{}{SEP}{}", d.code, d.message, d.hint.unwrap_or_default()))
}

fn decode(msg: &str) -> Option<Diagnostic> {
    let mut it = msg.splitn(3, SEP);
    let (code, message, hint) = (it.next()?, it.next()?, it.next()?);
    let d = Diagnostic::error(code, message);
    Some(if hint.is_empty() { d } else { d.hint(hint) })
}

/// Typed deserialization (pipeline step 6) with serde errors mapped to E-IR-0101..0103 and embedded codes.
pub fn from_value<T: DeserializeOwned>(v: Value, root: &str) -> Result<T, Diagnostic> {
    serde_path_to_error::deserialize(v).map_err(|e| {
        let path = e.path().to_string();
        let path = if path == "." { root.to_owned() } else { format!("{root}.{path}") };
        let msg = e.inner().to_string();
        let raw = msg.split(" at line ").next().unwrap_or(&msg);
        let embedded = raw.find("E-IR-").and_then(|i| decode(&raw[i..]));
        let d = if let Some(d) = embedded {
            d
        } else if raw.starts_with("unknown field") {
            Diagnostic::error("E-IR-0101", raw.to_owned()).hint("remove the field or fix its spelling")
        } else if raw.starts_with("missing field") {
            Diagnostic::error("E-IR-0102", raw.to_owned()).hint("add the required field")
        } else if raw.starts_with("unknown variant") {
            Diagnostic::error("E-IR-0103", raw.to_owned()).hint("use one of the listed values")
        } else if raw.contains("E-ID-0001") || raw.contains("invalid id") {
            Diagnostic::error("E-IR-0104", raw.to_owned())
                .hint("ids match ^[a-z][a-z0-9_-]*$ (lowercase, no dots)")
        } else {
            Diagnostic::error("E-IR-0103", raw.to_owned())
        };
        d.at(path)
    })
}

/// Collects diagnostics, folding repeats of the same code on instances of one template-level entity.
#[derive(Default, Debug)]
pub struct Diags {
    items: IndexMap<(String, String), (Diagnostic, usize)>,
    plain: Vec<Diagnostic>,
}

impl Diags {
    pub fn push(&mut self, d: Diagnostic) {
        self.plain.push(d);
    }

    /// `entity` is the template-level entity key; only the first instance's diagnostic is kept.
    pub fn push_inst(&mut self, entity: &str, d: Diagnostic) {
        self.items.entry((d.code.clone(), entity.to_owned())).and_modify(|(_, n)| *n += 1).or_insert((d, 0));
    }

    pub fn extend(&mut self, ds: impl IntoIterator<Item = Diagnostic>) {
        self.plain.extend(ds);
    }

    pub fn has_errors(&self) -> bool {
        self.plain.iter().chain(self.items.values().map(|(d, _)| d)).any(|d| d.severity == Severity::Error)
    }

    pub fn into_vec(self) -> Vec<Diagnostic> {
        let mut out = self.plain;
        out.extend(self.items.into_values().map(|(mut d, n)| {
            if n > 0 {
                d.message = format!("{} (and {n} other instances)", d.message);
            }
            d
        }));
        out
    }
}
