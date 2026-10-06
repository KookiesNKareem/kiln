//! `KILN_COST_MODEL_HASH`: a hash of every source that decides search results (kiln-cost and kiln-ir), so a
//! persistent cost cache written by another build of the model is never read back.

use std::path::{Path, PathBuf};

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let mut files = vec![];
    for d in [root.join("src"), root.join("../kiln-ir/src")] {
        println!("cargo:rerun-if-changed={}", d.display());
        sources(&d, &mut files);
    }
    files.sort();
    // FNV-1a over (relative path, contents): stable across machines and checkouts.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= u64::from(x);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for f in &files {
        eat(f.strip_prefix(&root).unwrap_or(f).to_string_lossy().as_bytes());
        eat(&std::fs::read(f).unwrap_or_default());
    }
    println!("cargo:rustc-env=KILN_COST_MODEL_HASH={h:016x}");
}
