use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

fn kiln(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kiln"))
        .args(args)
        .arg("--no-cache")
        .output()
        .unwrap()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn design(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../designs/reference")
        .join(name)
        .display()
        .to_string()
}

fn engine_outcome(r: &Value, o: &Output) {
    assert_eq!(code(o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(r["status"], "ok");
    assert_eq!(r["stage_reached"], "S3");
}

#[test]
fn eval_json_is_a_result_document() {
    let o = kiln(&[
        "eval",
        &design("a100_sxm4_40gb.json5"),
        "--workload",
        "llama3_8b:decode_b1",
        "--format",
        "json",
    ]);
    let r: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(r["schema"], "kiln.result/1");
    assert!(
        r["provenance"]["design_hash"]
            .as_str()
            .unwrap()
            .starts_with("hw1-")
    );
    assert_eq!(r["provenance"]["calibration_id"], "generic-v1");
    assert_eq!(r["features"]["n_chips"], 1.0);
    engine_outcome(&r, &o);
    assert!(
        (r["score"].as_f64().unwrap() - 1.0).abs() < 1e-9,
        "A100 vs itself"
    );
    assert!(r["phases"][0]["time_s"]["central"].as_f64().unwrap() > 0.0);
}

#[test]
fn eval_text_llm_and_scenario() {
    let o = kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "llama3_8b",
        "--scenario",
        "decode_b8",
        "--tier",
        "a",
    ]);
    let t = out(&o);
    assert!(t.starts_with("design    a100_40gb  hw1-"), "{t}");
    assert!(t.contains("workload  llama3_8b:decode_b8"), "{t}");
    assert!(t.contains("tier A"), "{t}");
    assert_eq!(code(&o), 0, "{t}");
    assert!(t.contains("status    ok  stage S3"), "{t}");
    assert!(t.lines().any(|l| l.starts_with("decode_b8 ")), "{t}");
    let o = kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "smoke",
        "--format",
        "llm",
    ]);
    assert!(out(&o).starts_with("status: "), "{}", out(&o));
}

#[test]
fn eval_validate_tier_and_profile() {
    let o = kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "smoke",
        "--tier",
        "validate",
    ]);
    assert_eq!(code(&o), 0, "{}", out(&o));
    assert!(out(&o).contains("status    ok  stage S0"));
    let o = kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "smoke",
        "--tier",
        "validate",
        "--profile",
        "search",
        "--format",
        "json",
    ]);
    assert_eq!(code(&o), 1);
    let r: Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(r["status"], "invalid");
    assert_eq!(r["errors"][0]["code"], "E-IR-1102");
}

#[test]
fn eval_exit_codes() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.json5");
    std::fs::write(&bad, "{ schema: 'kiln.hw/1.0', name: 'x', ").unwrap();
    let bad = bad.to_str().unwrap();
    assert_eq!(code(&kiln(&["eval", bad, "--workload", "smoke"])), 3);
    assert_eq!(
        code(&kiln(&[
            "eval",
            "/no/such/design.json5",
            "--workload",
            "smoke"
        ])),
        3
    );
    let o = kiln(&["eval", "a100_40gb", "--workload", "nope:decode_b1"]);
    assert_eq!(code(&o), 3);
    assert!(String::from_utf8_lossy(&o.stderr).contains("E-WL"));
    assert_eq!(
        code(&kiln(&[
            "eval",
            "a100_40gb",
            "--workload",
            "llama3_8b",
            "--scenario",
            "nope"
        ])),
        3
    );
    assert_eq!(
        code(&kiln(&[
            "eval",
            "a100_40gb",
            "--fitness",
            "{\"kind\": \"nope\"}"
        ])),
        2
    );
    assert_eq!(code(&kiln(&["eval", "a100_40gb", "--seeds", "1,x"])), 2);
    assert_eq!(code(&kiln(&["eval", "a100_40gb", "tpu_v5e"])), 2);
    assert_eq!(code(&kiln(&["compare", "a100_40gb"])), 2);
}

#[test]
fn invalid_design_exits_1_with_hint() {
    let o = kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "smoke",
        "--profile",
        "search",
    ]);
    assert_eq!(code(&o), 1);
    let t = out(&o);
    assert!(t.contains("status    invalid  stage S0"), "{t}");
    assert!(
        t.contains("error[E-IR-1102]") && t.contains("hint: remove `family`"),
        "{t}"
    );
}

#[test]
fn compare_lists_designs_in_order() {
    let args = [
        "compare",
        "a100_40gb",
        "tpu_v5e",
        "tpu_v6e",
        "--workload",
        "smoke",
        "--format",
        "json",
    ];
    // v6e has more off-chip bandwidth than the A100-40GB: outside its matched envelope (06 §6.4).
    let o = kiln(&args);
    let rs: Vec<Value> = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(rs[2]["status"], "envelope");
    assert_eq!(rs[2]["violations"][0]["code"], "E-ENV-0007");
    let o = kiln(
        &[
            &args[..],
            &["--fitness", r#"{"kind": "baseline_relative"}"#],
        ]
        .concat(),
    );
    let rs: Vec<Value> = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(rs.len(), 3);
    for r in &rs {
        engine_outcome(r, &o);
    }
    let o = kiln(&[
        "compare",
        "a100_40gb",
        "tpu_v5e",
        "--workload",
        "smoke",
        "--format",
        "jsonl",
    ]);
    assert_eq!(out(&o).lines().count(), 2);
    let o = kiln(&["compare", "a100_40gb", "tpu_v5e", "--workload", "smoke"]);
    let t = out(&o);
    assert!(
        t.starts_with("workload  smoke  (ratios vs a100_40gb)"),
        "{t}"
    );
    assert!(t.lines().any(|l| l.starts_with("tpu_v5e ")), "{t}");
}

#[test]
fn eval_out_then_explain_and_trace_validate() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("r.json");
    let p = p.to_str().unwrap();
    kiln(&[
        "eval",
        "a100_40gb",
        "--workload",
        "smoke",
        "--format",
        "json",
        "-o",
        p,
    ]);
    let o = kiln(&["explain", p]);
    assert_eq!(code(&o), 0);
    assert!(out(&o).starts_with("status: "));
    let o = kiln(&["trace", "info", p]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn stack_flag_sets_the_score_stack_and_realistic_is_reported() {
    let a100 = design("a100_sxm4_40gb.json5");
    let run = |stack: &str| {
        let o = kiln(&[
            "eval",
            &a100,
            "--workload",
            "llama3_8b:decode_b1",
            "--stack",
            stack,
            "--format",
            "json",
        ]);
        let r: Value = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
        (o, r)
    };
    let (o, r) = run("kiln_ideal");
    engine_outcome(&r, &o);
    let sc = &r["score_components"];
    assert!(
        sc["candidate_stack"]
            .as_str()
            .unwrap()
            .starts_with("kiln_ideal@")
    );
    assert_eq!(sc["candidate_stack"], sc["baseline_stack"]);
    let rs = &r["score_realistic"];
    assert!(
        rs["candidate_stack"]
            .as_str()
            .unwrap()
            .starts_with("pytorch_cuda_graph_sdpa@")
    );
    assert!((rs["score"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    let (o, r) = run("own");
    engine_outcome(&r, &o);
    assert!(
        r["score_components"]["candidate_stack"]
            .as_str()
            .unwrap()
            .starts_with("pytorch_cuda_graph_sdpa@")
    );
    let (o, _) = run("no_such_stack");
    assert_eq!(code(&o), 2, "{}", String::from_utf8_lossy(&o.stderr));
    let o = kiln(&["eval", &a100, "--workload", "llama3_8b:decode_b1"]);
    assert!(out(&o).contains("realistic 1.000"), "{}", out(&o));
}
