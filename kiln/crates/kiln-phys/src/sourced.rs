//! `Sourced` table values (04 §3): value, quality, confidence, citation, bounds and the derived range.

use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    Pub,
    Der,
    Asm,
    Fit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Conf {
    H,
    M,
    L,
}

impl Conf {
    /// Log-space one-sigma band (04 §3; the §12.3 prior sigmas).
    pub fn sigma(self) -> f64 {
        match self {
            Conf::H => 0.1,
            Conf::M => 0.3,
            Conf::L => 0.7,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sourced {
    pub v: f64,
    pub q: Quality,
    pub c: Conf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Sourced {
    pub fn asm(v: f64) -> Sourced {
        Sourced { v, q: Quality::Asm, c: Conf::L, src: None, bounds: None, range: None, fit: false, note: None }
    }

    /// `(lower, upper)` per 04 §3: explicit range, else `central * exp(-+sigma_c)`, clipped to bounds.
    pub fn band(&self) -> (f64, f64) {
        let (lo, hi) = match self.range {
            Some([a, b]) => (a, b),
            None => {
                let s = self.c.sigma();
                (self.v * (-s).exp(), self.v * s.exp())
            }
        };
        match self.bounds {
            Some([a, b]) => (lo.max(a), hi.min(b)),
            None => (lo, hi),
        }
    }
}

/// Every numeric leaf of a data file must sit inside a `Sourced` object (`E-PHYS-UNSOURCED`), and every `src`
/// must resolve in `sources.json`.
pub fn check_sourced(file: &str, v: &Value, sources: &serde_json::Map<String, Value>, out: &mut Vec<Diagnostic>) {
    fn walk(file: &str, path: &str, v: &Value, sources: &serde_json::Map<String, Value>, out: &mut Vec<Diagnostic>) {
        match v {
            Value::Number(_) => out.push(
                Diagnostic::error("E-PHYS-UNSOURCED", format!("{file}: bare number at {path}"))
                    .at(format!("{file}:{path}"))
                    .hint("write {\"v\": .., \"q\": pub|der|asm|fit, \"c\": H|M|L, \"src\": <sources.json key>}"),
            ),
            Value::Object(o) if o.contains_key("v") && o.contains_key("q") => {
                if let Err(e) = serde_json::from_value::<Sourced>(v.clone()) {
                    out.push(Diagnostic::error("E-PHYS-UNSOURCED", format!("{file}: malformed sourced value at {path}: {e}")).at(format!("{file}:{path}")));
                }
                if let Some(Value::String(s)) = o.get("src")
                    && !sources.contains_key(s)
                {
                    out.push(Diagnostic::error("E-PHYS-UNSOURCED", format!("{file}: source {s:?} at {path} is not in sources.json")).at(format!("{file}:{path}")));
                }
            }
            Value::Object(o) => {
                for (k, x) in o {
                    walk(file, &format!("{path}.{k}"), x, sources, out);
                }
            }
            Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    walk(file, &format!("{path}[{i}]"), x, sources, out);
                }
            }
            _ => {}
        }
    }
    walk(file, "", v, sources, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_follow_confidence_and_bounds() {
        let s = Sourced { v: 1.0, q: Quality::Asm, c: Conf::L, src: None, bounds: Some([0.8, 1.5]), range: None, fit: false, note: None };
        let (lo, hi) = s.band();
        assert_eq!((lo, hi), (0.8, 1.5));
        let h = Sourced { c: Conf::H, bounds: None, ..s.clone() };
        let (lo, hi) = h.band();
        assert!((lo - (-0.1f64).exp()).abs() < 1e-12 && (hi - 0.1f64.exp()).abs() < 1e-12);
        let r = Sourced { range: Some([500.0, 600.0]), bounds: None, ..s };
        assert_eq!(r.band(), (500.0, 600.0));
    }

    #[test]
    fn bare_numbers_are_unsourced() {
        let src = serde_json::json!({"a": {}}).as_object().unwrap().clone();
        let mut out = vec![];
        check_sourced("t.json", &serde_json::json!({"x": 3, "y": {"v": 1.0, "q": "asm", "c": "L", "src": "b"}}), &src, &mut out);
        let codes: Vec<&str> = out.iter().map(|d| d.code.as_str()).collect();
        assert_eq!(codes, ["E-PHYS-UNSOURCED", "E-PHYS-UNSOURCED"]);
    }
}
