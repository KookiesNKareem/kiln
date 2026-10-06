//! Canonical compact form, design hash and structural diff (01 §16.2, §17).

use serde::Serialize;
use serde_json::Value;

use super::types::HwDoc;
use crate::common::{canonical_json, content_hash, jcs_number};

/// Arrays whose order is not semantic (01 §17 rule 2).
const UNORDERED: &[&str] =
    &["precisions", "ops", "functions", "in_network_reduce", "disabled", "coherent_with", "sparsity"];

/// Canonical compact form of a typed document: nulls (`None`) dropped, unordered arrays sorted, numbers in JCS
/// form ([`jcs_number`]).
pub fn to_canonical_value(doc: &HwDoc) -> Value {
    let mut v = serde_json::to_value(doc).expect("HwDoc serializes");
    normalize(&mut v);
    v
}

pub fn to_canonical_json(doc: &HwDoc) -> String {
    canonical_json(&to_canonical_value(doc))
}

fn normalize(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.retain(|_, x| !x.is_null());
            for (k, x) in m.iter_mut() {
                normalize(x);
                if UNORDERED.contains(&k.as_str())
                    && let Value::Array(a) = x
                {
                    a.sort_by_cached_key(canonical_json);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(normalize),
        Value::Number(n) => *v = jcs_number(n),
        _ => {}
    }
}

/// The hashed view: `meta` removed and `schema` reduced to its major version (01 §17 rule 5).
pub fn hash_view(canonical: &Value) -> Value {
    let mut v = canonical.clone();
    if let Value::Object(m) = &mut v {
        m.remove("meta");
        if let Some(Value::String(s)) = m.get("schema") {
            let major = s.split('.').next().unwrap_or(s).to_owned();
            m.insert("schema".into(), Value::String(major));
        }
    }
    v
}

pub fn design_hash(canonical: &Value) -> String {
    content_hash("hw1-", &hash_view(canonical))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct HwDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<FieldChange>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FieldChange {
    /// Template-level entity path (`board.gpu.ga100.gpc`), or a top-level key.
    pub entity: String,
    pub field: String,
    pub old: Value,
    pub new: Value,
}

impl HwDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Structural diff of two canonical forms; arrays of entities are matched by `id`, `meta` is ignored.
pub fn diff(a: &Value, b: &Value) -> HwDiff {
    let mut d = HwDiff::default();
    diff_obj(&hash_view(a), &hash_view(b), "", "", &mut d);
    d
}

fn is_entity_list(a: &[Value]) -> bool {
    !a.is_empty() && a.iter().all(|x| x.get("id").is_some_and(Value::is_string))
}

fn join(p: &str, s: &str) -> String {
    if p.is_empty() { s.to_owned() } else { format!("{p}.{s}") }
}

fn diff_obj(a: &Value, b: &Value, entity: &str, field: &str, d: &mut HwDiff) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                let (va, vb) = (x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null));
                if va == vb || k == "id" {
                    continue;
                }
                diff_obj(va, vb, entity, &join(field, k), d);
            }
        }
        (Value::Array(x), Value::Array(y)) if is_entity_list(x) || is_entity_list(y) => {
            let id = |v: &Value| v["id"].as_str().unwrap_or_default().to_owned();
            for ea in x {
                let path = join(entity, &id(ea));
                match y.iter().find(|eb| id(eb) == id(ea)) {
                    Some(eb) => diff_obj(ea, eb, &path, "", d),
                    None => d.removed.push(path),
                }
            }
            for eb in y.iter().filter(|eb| !x.iter().any(|ea| id(ea) == id(eb))) {
                d.added.push(join(entity, &id(eb)));
            }
        }
        _ => {
            if a != b {
                d.changed.push(FieldChange {
                    entity: entity.to_owned(),
                    field: field.to_owned(),
                    old: a.clone(),
                    new: b.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_follows_jcs_numbers_and_sorts_unordered_arrays() {
        let mut v = json!({"a": 2.0, "b": -0.0, "c": null, "precisions": ["z", "a"], "dims": [3, 1], "f": 1.5});
        normalize(&mut v);
        assert_eq!(v, json!({"a": 2, "b": 0, "precisions": ["a", "z"], "dims": [3, 1], "f": 1.5}));
    }

    #[test]
    fn hash_view_drops_meta_and_minor_version() {
        let a = json!({"schema": "kiln.hw/1.0", "meta": {"description": "x"}, "name": "n"});
        let b = json!({"schema": "kiln.hw/1.3", "name": "n"});
        assert_eq!(design_hash(&a), design_hash(&b));
    }

    #[test]
    fn diff_matches_entities_by_id() {
        let a = json!({"system": {"units": [{"id": "u", "lanes": 8}, {"id": "gone"}]}});
        let b = json!({"system": {"units": [{"id": "u", "lanes": 16}, {"id": "new"}]}});
        let d = diff(&a, &b);
        assert_eq!((d.added, d.removed), (vec!["new".to_string()], vec!["gone".to_string()]));
        assert_eq!(d.changed[0].entity, "u");
        assert_eq!(d.changed[0].field, "lanes");
    }
}
