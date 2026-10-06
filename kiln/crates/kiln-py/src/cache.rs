//! 06 §6.7 content-addressed result cache: `<dir>/v1/<ab>/<key>.json.zst`, atomic write-rename.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use kiln_ir::common::canonical_json;
use kiln_trace::result::{EvalResult, Status};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub struct Cache {
    dir: PathBuf,
}

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn key(parts: &Value) -> String {
    hex::encode(Sha256::digest(canonical_json(parts).as_bytes()))
}

/// `timeout` and `internal_error` are never cached (06 §6.7).
pub fn cacheable(s: Status) -> bool {
    matches!(
        s,
        Status::Ok | Status::Invalid | Status::Envelope | Status::Infeasible
    )
}

impl Cache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir
            .join("v1")
            .join(&key[..2])
            .join(format!("{key}.json.zst"))
    }

    pub fn get(&self, key: &str) -> Option<EvalResult> {
        let bytes = std::fs::read(self.path(key)).ok()?;
        let json = zstd::decode_all(bytes.as_slice()).ok()?;
        serde_json::from_slice(&json).ok()
    }

    /// Best effort: an unwritable cache never fails an evaluation.
    pub fn put(&self, key: &str, r: &EvalResult) {
        if !cacheable(r.status) {
            return;
        }
        let path = self.path(key);
        let Some(parent) = path.parent() else { return };
        let Ok(z) = zstd::encode_all(r.canonical_json().as_bytes(), 3) else {
            return;
        };
        let tmp = parent.join(format!(
            ".{key}.{}.{}.tmp",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let ok = std::fs::create_dir_all(parent).is_ok()
            && std::fs::write(&tmp, z).is_ok()
            && std::fs::rename(&tmp, &path).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::result_with;
    use serde_json::json;

    #[test]
    fn round_trip_and_policy() {
        let dir = tempfile::tempdir().unwrap();
        let c = Cache::new(dir.path());
        let k = key(&json!({"a": 1}));
        assert_eq!(k, key(&json!({"a": 1.0})));
        assert!(c.get(&k).is_none());
        let r = result_with(&[("decode_b1", 80.0)], 800.0, 400.0);
        c.put(&k, &r);
        assert_eq!(c.get(&k).unwrap(), r);
        assert!(c.path(&k).starts_with(dir.path().join("v1").join(&k[..2])));
        let mut t = r.clone();
        t.status = Status::Timeout;
        let k2 = key(&json!({"a": 2}));
        c.put(&k2, &t);
        assert!(c.get(&k2).is_none());
    }
}
