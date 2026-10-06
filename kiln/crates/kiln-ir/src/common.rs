//! Shared conventions from spec 00: ids, structured diagnostics, canonical JSON and hashing.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Id(String);

impl Id {
    pub fn new(s: impl Into<String>) -> Result<Self, Diagnostic> {
        let s = s.into();
        let valid = !s.is_empty()
            && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-'));
        if valid {
            Ok(Self(s))
        } else {
            Err(Diagnostic::error("E-ID-0001", format!("invalid id {s:?}"))
                .hint("ids use only [a-z0-9_.-] and must be non-empty"))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Id {
    type Error = Diagnostic;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<Id> for String {
    fn from(id: Id) -> Self {
        id.0
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

/// Structured, LLM-actionable diagnostic (code + message + entity path + hint).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{code}: {message}{}", path.as_deref().map(|p| format!(" at {p}")).unwrap_or_default())]
pub struct Diagnostic {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// Other entity paths involved (01 §18.1), e.g. the other end of a conflict.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Box<Span>>,
}

/// Location in the authoring source (01 §18.1): 1-based line and column, optional byte range and file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    pub line: u32,
    pub col: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            path: None,
            hint: None,
            related: vec![],
            span: None,
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, ..Self::error(code, message) }
    }

    pub fn at(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub fn related(mut self, path: impl Into<String>) -> Self {
        self.related.push(path.into());
        self
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = Some(Box::new(span));
        self
    }
}

/// Canonical JSON per spec 00 decision 2 (RFC 8785 / JCS): object keys sorted bytewise, no whitespace, numbers in
/// ECMAScript form, so floats that are exact integers print as integers and `-0.0` as `0`. Other finite floats use
/// serde_json's shortest round-trip form; non-finite floats cannot appear in serde_json values.
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&canonical_value(value)).expect("serializing a JSON value cannot fail")
}

/// The value `canonical_json` serializes: keys sorted recursively, numbers as in [`jcs_number`].
pub fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: BTreeMap<&String, Value> = map.iter().map(|(k, v)| (k, canonical_value(v))).collect();
            Value::Object(ordered.into_iter().map(|(k, v)| (k.clone(), v)).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        Value::Number(n) => jcs_number(n),
        other => other.clone(),
    }
}

/// JCS number form: a float that is an exact integer becomes that integer (`2.0` -> `2`, `-0.0` -> `0`).
pub fn jcs_number(n: &serde_json::Number) -> Value {
    match n.as_f64().filter(|f| n.is_f64() && f.fract() == 0.0) {
        Some(f) if f.abs() < 9.223_372_036_854_775e18 => Value::from(f as i64),
        Some(f) if (0.0..1.844_674_407_370_955e19).contains(&f) => Value::from(f as u64),
        _ => Value::Number(n.clone()),
    }
}

/// sha256 over canonical JSON (00 decision 9), rendered as `prefix + 32 hex chars` (01 §16).
pub fn content_hash(prefix: &str, value: &Value) -> String {
    let digest = Sha256::digest(canonical_json(value).as_bytes());
    format!("{prefix}{}", &hex::encode(digest)[..32])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        let v = json!({"b": 1, "a": {"d": [3, {"z": 0, "y": 1}], "c": 2.5}});
        assert_eq!(canonical_json(&v), r#"{"a":{"c":2.5,"d":[3,{"y":1,"z":0}]},"b":1}"#);
    }

    #[test]
    fn canonical_json_follows_jcs_numbers() {
        let v = json!({"a": 2.0, "b": -0.0, "c": 1.5, "d": 1.41e9, "e": [3.0, -7.0], "g": 5});
        assert_eq!(canonical_json(&v), r#"{"a":2,"b":0,"c":1.5,"d":1410000000,"e":[3,-7],"g":5}"#);
        assert_eq!(content_hash("x-", &json!({"a": 2.0})), content_hash("x-", &json!({"a": 2})));
    }

    #[test]
    fn hash_ignores_key_order() {
        assert_eq!(content_hash("hw1-", &json!({"a": 1, "b": 2})), content_hash("hw1-", &json!({"b": 2, "a": 1})));
    }

    #[test]
    fn diagnostic_optional_fields_are_skipped() {
        let d = Diagnostic::error("E-X", "m");
        assert_eq!(serde_json::to_value(&d).unwrap(), json!({"code": "E-X", "severity": "error", "message": "m"}));
        let d = d.related("a.b").span(Span { line: 3, col: 7, ..Span::default() });
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["related"], json!(["a.b"]));
        assert_eq!(v["span"], json!({"line": 3, "col": 7}));
        assert_eq!(serde_json::from_value::<Diagnostic>(v).unwrap(), d);
    }

    #[test]
    fn id_validation() {
        assert!(Id::new("chip0.tile3.sram").is_ok());
        assert_eq!(Id::new("Bad Id").unwrap_err().code, "E-ID-0001");
    }
}
