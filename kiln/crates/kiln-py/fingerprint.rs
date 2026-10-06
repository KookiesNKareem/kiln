use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

/// The last commit touching `paths` (relative to `dir`), suffixed `-dirty<hash>` when the working tree differs from
/// it: the hash covers the tracked diff and the path and contents of every untracked, non-ignored file, since
/// cargo compiles an untracked module just like a tracked one.
pub fn engine_fingerprint(dir: &Path, paths: &[&str]) -> Option<String> {
    let mut args = vec!["log", "-1", "--format=%h", "--"];
    args.extend(paths);
    let head = String::from_utf8(git(dir, &args)?).ok()?.trim().to_string();
    let mut args = vec!["diff", "HEAD", "--"];
    args.extend(paths);
    let mut dirty = git(dir, &args)?;
    let mut args = vec!["ls-files", "-z", "--others", "--exclude-standard", "--"];
    args.extend(paths);
    let untracked = git(dir, &args)?;
    let mut files: Vec<&[u8]> = untracked.split(|&b| b == 0).filter(|f| !f.is_empty()).collect();
    files.sort_unstable();
    for f in files {
        let name = std::str::from_utf8(f).ok()?;
        dirty.extend_from_slice(b"\0untracked\0");
        dirty.extend_from_slice(f);
        dirty.push(0);
        dirty.extend(std::fs::read(dir.join(name)).ok()?);
    }
    Some(if dirty.is_empty() {
        head
    } else {
        format!("{head}-dirty{:08x}", fnv(&dirty))
    })
}

fn fnv(b: &[u8]) -> u32 {
    b.iter().fold(0x811c_9dc5u32, |h, &x| {
        (h ^ u32::from(x)).wrapping_mul(0x0100_0193)
    })
}
