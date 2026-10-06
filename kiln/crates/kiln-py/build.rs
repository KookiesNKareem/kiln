mod fingerprint;

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
        let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
        fingerprint::engine_fingerprint(std::path::Path::new(".."), &paths)
    });
    println!(
        "cargo:rustc-env=KILN_GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".into())
    );
}
