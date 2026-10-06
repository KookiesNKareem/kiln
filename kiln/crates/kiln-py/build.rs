use std::process::Command;

const ENGINE_CRATES: [&str; 8] = [
    "kiln-ir",
    "kiln-wl",
    "kiln-cost",
    "kiln-map",
    "kiln-sim",
    "kiln-phys",
    "kiln-trace",
    "kiln-py",
];

fn git(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git")
        .args(args)
        .current_dir("..")
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

/// Data compiled into the engine with `include_str!` (paths relative to `crates/`).
const ENGINE_DATA: [&str; 2] = ["kiln-phys/data", "../stacks"];

fn main() {
    let paths: Vec<String> = ENGINE_CRATES
        .iter()
        .map(|c| format!("{c}/src"))
        .chain(ENGINE_DATA.iter().map(|d| d.to_string()))
        .collect();
    for p in &paths {
        println!("cargo:rerun-if-changed=../{p}");
    }
    println!("cargo:rerun-if-env-changed=KILN_GIT_HASH");
    let hash = std::env::var("KILN_GIT_HASH").ok().or_else(|| {
        let mut args = vec!["log", "-1", "--format=%h", "--"];
        args.extend(paths.iter().map(String::as_str));
        let head = String::from_utf8(git(&args)?).ok()?.trim().to_string();
        let mut args = vec!["diff", "HEAD", "--"];
        args.extend(paths.iter().map(String::as_str));
        let diff = git(&args)?;
        Some(if diff.is_empty() {
            head
        } else {
            format!("{head}-dirty{:08x}", fnv(&diff))
        })
    });
    println!(
        "cargo:rustc-env=KILN_GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".into())
    );
}

fn fnv(b: &[u8]) -> u32 {
    b.iter().fold(0x811c_9dc5u32, |h, &x| {
        (h ^ u32::from(x)).wrapping_mul(0x0100_0193)
    })
}
