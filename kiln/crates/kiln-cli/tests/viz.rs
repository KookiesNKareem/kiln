//! `.kiln` writing from `kiln eval`, `kiln trace info|validate|export` and `kiln viz render` (05 §8).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn kiln(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kiln")).args(args).output().expect("kiln runs")
}

fn ok(o: &Output) {
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn design() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/a100_sxm4_40gb.json5")
}

#[test]
fn eval_writes_trace_that_validates_renders_and_exports() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("a100.kiln");
    ok(&kiln(&["eval", s(&design()), "--workload", "llama3_8b:decode_b1", "--trace", "ops", "-o", s(&run)]));
    let info = kiln(&["trace", "info", s(&run)]);
    ok(&info);
    let text = String::from_utf8_lossy(&info.stdout);
    assert!(text.contains("table     spans") && text.contains("floorplan kiln-phys/m3") && text.contains("table     wires"), "{text}");
    ok(&kiln(&["trace", "validate", s(&run)]));

    let png = dir.path().join("fig.png");
    ok(&kiln(&["viz", "render", s(&run), "--view", "floorplan", "--view", "roofline", "--view", "bottleneck", "--view", "timeline", "--size", "800x500", "-o", s(&png)]));
    for v in ["floorplan", "roofline", "bottleneck", "timeline"] {
        let b = std::fs::read(dir.path().join(format!("fig-{v}.png"))).unwrap();
        assert!(b.starts_with(b"\x89PNG"), "{v}");
    }
    let svg = dir.path().join("fp.svg");
    ok(&kiln(&["viz", "render", s(&run), "--view", "floorplan", "--color", "kind", "-o", s(&svg)]));
    let text = std::fs::read_to_string(&svg).unwrap();
    assert!(text.starts_with("<svg") && text.contains("compute") && !text.contains("UNPLACED"));
    let svg2 = dir.path().join("fp2.svg");
    ok(&kiln(&["viz", "render", s(&run), "--view", "floorplan", "--color", "kind", "-o", s(&svg2)]));
    assert_eq!(text, std::fs::read_to_string(&svg2).unwrap(), "headless output is deterministic");

    let cmp = dir.path().join("cmp.png");
    ok(&kiln(&["viz", "render", "--compare", s(&run), s(&run), "--view", "compare", "--size", "900x600", "-o", s(&cmp)]));

    let pf = dir.path().join("a100.pftrace");
    ok(&kiln(&["trace", "export", s(&run), "--perfetto", "-o", s(&pf)]));
    assert!(std::fs::metadata(&pf).unwrap().len() > 100);
    let cj = dir.path().join("a100.json");
    ok(&kiln(&["trace", "export", s(&run), "--chrome-json", "-o", s(&cj)]));
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&cj).unwrap()).unwrap();
    assert!(v["traceEvents"].as_array().unwrap().len() > 79);
    let csv = kiln(&["trace", "export", s(&run), "--csv", "phases"]);
    ok(&csv);
    assert!(String::from_utf8_lossy(&csv.stdout).starts_with("phase,id,scope"));
}

#[test]
fn design_only_floorplan_and_missing_gui() {
    let dir = tempfile::tempdir().unwrap();
    let png = dir.path().join("structure.png");
    ok(&kiln(&["viz", "render", "tpu_v5e", "--view", "floorplan", "--color", "kind", "--size", "640x400", "-o", s(&png)]));
    // A design file: placed floorplan, design sheet and wires without a simulation.
    let svg = dir.path().join("d.svg");
    ok(&kiln(&["viz", "render", s(&design()), "--view", "floorplan", "--view", "design", "--view", "wires", "--size", "900x560", "-o", s(&svg)]));
    let fp = std::fs::read_to_string(dir.path().join("d-floorplan.svg")).unwrap();
    assert!(fp.contains("kiln-phys/m3") && fp.contains("color: kind") && !fp.contains("UNPLACED"), "falls back to kind without a run");
    let sheet = std::fs::read_to_string(dir.path().join("d-design.svg")).unwrap();
    assert!(sheet.contains("Memory hierarchy") && sheet.contains("reference profile") && sheet.contains("TDP"));
    assert!(std::fs::read_to_string(dir.path().join("d-wires.svg")).unwrap().contains("pJ/bit"));
    for (flag, val) in [("--color", "density"), ("--wire-color", "length"), ("--layer", "0")] {
        ok(&kiln(&["viz", "render", s(&design()), "--view", "floorplan", flag, val, "--size", "640x400", "-o", s(&png)]));
    }
    let csv = kiln(&["trace", "export", s(&design()), "--csv", "wires"]);
    ok(&csv);
    assert!(String::from_utf8_lossy(&csv.stdout).lines().count() > 1000);
    if !cfg!(feature = "gui") {
        assert_eq!(kiln(&["viz", s(&design())]).status.code(), Some(69));
    }
}
