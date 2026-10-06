//! Append-only session storage (06 §4.5): `<root>/<vendor>/<sku>/<YYYY-MM-DD>_<session_id>.json` in canonical
//! JSON, plus `<root>/index.json` mapping session hash -> status. Session files are never rewritten.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use kiln_ir::common::Diagnostic;
use serde::{Deserialize, Serialize};

use super::legacy::slug;
use super::{MeasSession, codes};

pub const INDEX_SCHEMA: &str = "kiln.meas-index/1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Superseded { by: String },
    Rejected { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    pub path: String,
    #[serde(flatten)]
    pub status: SessionStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasIndex {
    pub schema: String,
    pub sessions: BTreeMap<String, IndexEntry>,
}

impl Default for MeasIndex {
    fn default() -> Self {
        Self {
            schema: INDEX_SCHEMA.into(),
            sessions: BTreeMap::new(),
        }
    }
}

pub struct MeasStore {
    root: PathBuf,
}

fn io(path: &Path, e: std::io::Error) -> Diagnostic {
    Diagnostic::error(codes::IO, format!("{}: {e}", path.display()))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Diagnostic> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes).map_err(|e| io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| io(path, e))
}

impl MeasStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn relative_path(session: &MeasSession) -> PathBuf {
        let date = session
            .method
            .started
            .as_deref()
            .and_then(|s| s.get(..10))
            .unwrap_or("undated");
        let vendor = serde_json::to_value(session.device.vendor)
            .ok()
            .and_then(|v| v.as_str().map(String::from));
        PathBuf::from(vendor.unwrap_or_else(|| "other".into()))
            .join(slug(&session.device.sku))
            .join(format!("{date}_{}.json", session.session_id))
    }

    pub fn index(&self) -> Result<MeasIndex, Diagnostic> {
        let p = self.root.join("index.json");
        match fs::read(&p) {
            Ok(b) => serde_json::from_slice(&b)
                .map_err(|e| Diagnostic::error(codes::SCHEMA, format!("{}: {e}", p.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(MeasIndex::default()),
            Err(e) => Err(io(&p, e)),
        }
    }

    fn write_index(&self, index: &MeasIndex) -> Result<(), Diagnostic> {
        let v = serde_json::to_value(index).expect("index serializes");
        write_atomic(
            &self.root.join("index.json"),
            (kiln_ir::common::canonical_json(&v) + "\n").as_bytes(),
        )
    }

    /// Stores a sealed session. Re-putting identical content is a no-op; different content at the same path is
    /// refused (append-only).
    pub fn put(&self, session: &MeasSession, status: SessionStatus) -> Result<PathBuf, Diagnostic> {
        let hash = session
            .hash
            .clone()
            .filter(|h| *h == session.compute_hash())
            .ok_or_else(|| {
                Diagnostic::error(
                    codes::HASH,
                    "session is unsealed or its hash does not match its content",
                )
            })?;
        let rel = Self::relative_path(session);
        let path = self.root.join(&rel);
        let bytes = session.canonical_json() + "\n";
        match fs::read(&path) {
            Ok(existing) if existing == bytes.as_bytes() => {}
            Ok(_) => {
                return Err(Diagnostic::error(codes::APPEND_ONLY, format!("{} exists with different content", rel.display()))
                    .at(path.display().to_string())
                    .hint("sessions are immutable; record a correction as a new session that supersedes this one"));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                write_atomic(&path, bytes.as_bytes())?
            }
            Err(e) => return Err(io(&path, e)),
        }
        let mut index = self.index()?;
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        index.sessions.entry(hash).or_insert(IndexEntry {
            path: rel_str,
            status,
        });
        self.write_index(&index)?;
        Ok(path)
    }

    /// Changes a session's status in the index (the session file itself never changes).
    pub fn set_status(&self, hash: &str, status: SessionStatus) -> Result<(), Diagnostic> {
        let mut index = self.index()?;
        let entry = index.sessions.get_mut(hash).ok_or_else(|| {
            Diagnostic::error(codes::LOOKUP, format!("session {hash} not in index"))
        })?;
        entry.status = status;
        self.write_index(&index)
    }

    pub fn load(&self, hash: &str) -> Result<MeasSession, Diagnostic> {
        let index = self.index()?;
        let entry = index.sessions.get(hash).ok_or_else(|| {
            Diagnostic::error(codes::LOOKUP, format!("session {hash} not in index"))
        })?;
        let p = self.root.join(&entry.path);
        let s = MeasSession::parse(&fs::read_to_string(&p).map_err(|e| io(&p, e))?)?;
        if s.hash.as_deref() != Some(hash) || s.compute_hash() != hash {
            return Err(Diagnostic::error(
                codes::HASH,
                format!("{} does not hash to {hash}", entry.path),
            ));
        }
        Ok(s)
    }
}
