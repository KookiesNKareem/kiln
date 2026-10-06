//! Design and workload inputs: JSON/JSON5 text, dicts, paths, reference names, built-in workload names (06 §6.1).

use std::path::{Path, PathBuf};

use kiln_ir::common::{Diagnostic, content_hash};
use kiln_ir::hw::{self, Design, MemLoader};
use kiln_ir::wl::{self, WorkloadDoc};
use kiln_wl::zoo::{self, SuiteMember};
use serde_json::Value;

pub const DESIGN_CODE: &str = "E-API-0001";
pub const WORKLOAD_CODE: &str = "E-API-0002";

#[derive(Clone, Debug)]
pub enum DesignInput {
    /// JSON or JSON5 source text, a file path, or a reference design name.
    Str(String),
    Value(Value),
}

#[derive(Clone, Debug)]
pub enum WorkloadInput {
    /// `<preset>:<scenario>`, a suite name, JSON text, or a file path.
    Str(String),
    Value(Value),
}

/// Reference designs shipped in `designs/reference`, plus aliases used as baselines.
const ALIASES: [(&str, &str); 3] = [
    ("a100", "a100_sxm4_40gb"),
    ("a100_40gb", "a100_sxm4_40gb"),
    ("tpuv5e", "tpu_v5e"),
];

pub fn default_designs_dir() -> PathBuf {
    std::env::var_os("KILN_DESIGNS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs"))
}

pub fn reference_path(designs_dir: &Path, name: &str) -> Option<PathBuf> {
    let name = ALIASES
        .iter()
        .find(|(a, _)| *a == name)
        .map_or(name, |(_, n)| n);
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if !valid {
        return None;
    }
    ["json5", "json"]
        .iter()
        .map(|ext| designs_dir.join("reference").join(format!("{name}.{ext}")))
        .find(|p| p.is_file())
}

fn looks_like_text(s: &str) -> bool {
    let t = s.trim_start();
    t.starts_with('{') || ((t.starts_with("//") || t.starts_with("/*")) && t.contains('{'))
}

fn design_error(msg: impl Into<String>) -> Vec<Diagnostic> {
    vec![Diagnostic::error(DESIGN_CODE, msg).at("design").hint(
        "pass a kiln.hw/1.0 design as a dict or JSON/JSON5 text, a path to a .json5/.json file, \
         or a reference name (a100_40gb, tpu_v5e, tpu_v6e, tpu_v5e_2x2, ember)",
    )]
}

/// Loads a design (01 pipeline steps 1-7). `kiln.hw/0` harness JSON is migrated (01 §19).
pub fn load_design(input: &DesignInput, designs_dir: &Path) -> Result<Design, Vec<Diagnostic>> {
    match input {
        DesignInput::Value(v) => from_value(v),
        DesignInput::Str(s) if looks_like_text(s) => {
            if is_legacy_text(s) {
                hw::import_harness(s)
            } else {
                Design::from_source(&MemLoader::default(), None, s)
            }
        }
        DesignInput::Str(s) => {
            let p = Path::new(s);
            let path = if p.is_file() {
                p.to_path_buf()
            } else {
                reference_path(designs_dir, s).ok_or_else(|| {
                    design_error(format!(
                        "design {s:?} is not a file, JSON text or reference design name"
                    ))
                })?
            };
            let text = std::fs::read_to_string(&path)
                .map_err(|e| design_error(format!("cannot read {}: {e}", path.display())))?;
            if is_legacy_text(&text) {
                hw::import_harness(&text)
            } else {
                hw::load_file(&path)
            }
        }
    }
}

fn from_value(v: &Value) -> Result<Design, Vec<Diagnostic>> {
    if !v.is_object() {
        return Err(design_error("design must be an object"));
    }
    let schema = v.get("schema").and_then(Value::as_str);
    if schema == Some("kiln.hw/0") {
        return hw::import_harness(&v.to_string());
    }
    Design::from_source(&MemLoader::default(), None, &v.to_string())
}

fn is_legacy_text(s: &str) -> bool {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("schema")?.as_str().map(|x| x == "kiln.hw/0"))
        .unwrap_or(false)
}

/// A resolved workload set: one or more `(doc, scenario)` members, scored together (02 §11.4).
#[derive(Clone, Debug)]
pub struct WorkloadSet {
    pub name: String,
    pub members: Vec<SuiteMember>,
}

impl WorkloadSet {
    pub fn member_hash(m: &SuiteMember) -> String {
        wl::workload_hash(m.model(), m.scenario(), None)
    }

    pub fn hash(&self) -> String {
        match self.members.as_slice() {
            [m] => Self::member_hash(m),
            ms => content_hash(
                "wl1-",
                &Value::from(ms.iter().map(Self::member_hash).collect::<Vec<_>>()),
            ),
        }
    }
}

fn wl_error(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(WORKLOAD_CODE, msg).at("workload").hint(
        "use a built-in name such as \"llama3_8b:decode_b8\" (optionally +weights=, +kv=, +acts=<dtype>) or a \
         suite (standard, legacy, smoke; standard+weights=fp8_e4m3), or a kiln workload document (02) as a \
         dict, JSON text or path",
    )
}

pub fn resolve_workload(input: &WorkloadInput) -> Result<WorkloadSet, Diagnostic> {
    match input {
        WorkloadInput::Value(v) => from_doc_value(v.clone(), "workload"),
        WorkloadInput::Str(s) if s.trim_start().starts_with('{') => {
            let v = serde_json::from_str(s)
                .map_err(|e| wl_error(format!("workload is not JSON: {e}")))?;
            from_doc_value(v, "workload")
        }
        WorkloadInput::Str(s) if s.contains(':') && !Path::new(s).is_file() => Ok(WorkloadSet {
            name: s.clone(),
            members: vec![zoo::workload(s).map_err(|d| d.at("workload"))?],
        }),
        WorkloadInput::Str(s) if Path::new(s).is_file() => {
            let text = std::fs::read_to_string(s)
                .map_err(|e| wl_error(format!("cannot read {s}: {e}")))?;
            let v = serde_json::from_str(&text)
                .map_err(|e| wl_error(format!("{s} is not JSON: {e}")))?;
            from_doc_value(v, s)
        }
        WorkloadInput::Str(s) => Ok(WorkloadSet {
            name: s.clone(),
            members: zoo::suite(s).map_err(|d| d.at("workload"))?,
        }),
    }
}

fn from_doc_value(v: Value, name: &str) -> Result<WorkloadSet, Diagnostic> {
    let doc: WorkloadDoc = serde_path_to_error::deserialize(&v).map_err(|e| {
        Diagnostic::error(
            "E-WL-DOC-001",
            format!("not a kiln workload document: {}", e.inner()),
        )
        .at(format!("workload.{}", e.path()))
    })?;
    let doc = zoo::expand_doc(&doc)?;
    if let Some(d) = wl::validate_doc(&doc)
        .into_iter()
        .find(|d| d.severity == kiln_ir::common::Severity::Error)
    {
        return Err(d);
    }
    if doc.scenarios.is_empty() {
        return Err(wl_error("workload document has no scenarios"));
    }
    let members = doc
        .scenarios
        .keys()
        .map(|sid| SuiteMember {
            name: format!("{}:{sid}", doc.id),
            doc: doc.clone(),
            scenario: sid.clone(),
        })
        .collect();
    Ok(WorkloadSet {
        name: if name == "workload" {
            doc.id.to_string()
        } else {
            name.into()
        },
        members,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_names_and_aliases() {
        let dir = default_designs_dir();
        assert!(reference_path(&dir, "a100_40gb").is_some());
        assert!(reference_path(&dir, "tpu_v6e").is_some());
        assert!(reference_path(&dir, "../etc/passwd").is_none());
        let d = load_design(&DesignInput::Str("a100".into()), &dir).unwrap();
        assert!(d.hash.starts_with("hw1-"));
    }

    #[test]
    fn workloads_resolve() {
        let w = resolve_workload(&WorkloadInput::Str("llama3_8b:decode_b8".into())).unwrap();
        assert_eq!(w.members.len(), 1);
        let s = resolve_workload(&WorkloadInput::Str("standard".into())).unwrap();
        assert_eq!(s.members.len(), 4);
        assert_ne!(s.hash(), w.hash());
        assert!(resolve_workload(&WorkloadInput::Str("nope:decode_b8".into())).is_err());
        assert!(resolve_workload(&WorkloadInput::Str("evolve".into())).is_err());
        // Storage dtypes per workload or per suite (kiln-wl `+weights=` / `+kv=`).
        let w = resolve_workload(&WorkloadInput::Str("llama3_8b:decode_b8+weights=fp8_e4m3+kv=fp8_e4m3".into())).unwrap();
        let m = w.members[0].model();
        let fp8 = kiln_ir::wl::ElemType::from(kiln_ir::precision::Precision::Fp8E4m3);
        assert!(m.tensors.values().filter(|t| matches!(t.class, kiln_ir::wl::TensorClass::Weight | kiln_ir::wl::TensorClass::KvCache)).all(|t| t.dtype == fp8));
        let s = resolve_workload(&WorkloadInput::Str("standard+weights=mxfp4".into())).unwrap();
        assert_eq!(s.members.len(), 4);
        assert!(s.members.iter().all(|x| x.name.ends_with("+weights=mxfp4")));
        assert!(resolve_workload(&WorkloadInput::Str("smoke+weights=fp8_e4m3".into())).is_err());
    }

    #[test]
    fn bad_design_inputs_are_structured() {
        let dir = default_designs_dir();
        let e = load_design(&DesignInput::Str("no_such_design".into()), &dir).unwrap_err();
        assert_eq!(e[0].code, DESIGN_CODE);
        let e = load_design(&DesignInput::Value(Value::from(3)), &dir).unwrap_err();
        assert_eq!(e[0].code, DESIGN_CODE);
        assert!(load_design(&DesignInput::Str("{schema: 'kiln.hw/1.0'".into()), &dir).is_err());
    }
}
