#[path = "../fingerprint.rs"]
mod fingerprint;

use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .status()
        .unwrap()
        .success();
    assert!(ok, "git {args:?}");
}

#[test]
fn untracked_engine_sources_change_the_fingerprint() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::create_dir_all(dir.join("eng/src")).unwrap();
    std::fs::write(dir.join("eng/src/lib.rs"), "mod new_model;\n").unwrap();
    git(dir, &["init", "-q"]);
    git(dir, &["add", "."]);
    git(dir, &["commit", "-qm", "c"]);
    let fp = || fingerprint::engine_fingerprint(dir, &["eng/src"]).unwrap();
    let clean = fp();
    assert!(!clean.contains("dirty"), "{clean}");
    std::fs::write(dir.join("eng/src/new_model.rs"), "pub fn speedup() -> f64 { 1.0 }\n").unwrap();
    let a = fp();
    std::fs::write(dir.join("eng/src/new_model.rs"), "pub fn speedup() -> f64 { 9.0 }\n").unwrap();
    let b = fp();
    assert!(a.starts_with(&format!("{clean}-dirty")), "{a}");
    assert_ne!(a, b);
    std::fs::write(dir.join("outside.rs"), "x").unwrap();
    assert_eq!(fp(), b);
}
