//! Design-hash golden tests (01 §17): the hash ignores formatting, key order, literal forms, templates, comments,
//! `meta` and notes, and changes with any semantic edit.

use std::path::PathBuf;

use kiln_ir::hw::canon::hash_view;
use kiln_ir::hw::{Design, MemLoader, load_file};
use serde_json::{Map, Value};

fn load(text: &str) -> Design {
    Design::from_source(&MemLoader::default(), None, text).unwrap_or_else(|e| panic!("{e:#?}"))
}

fn reversed(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().rev().map(|(k, x)| (k.clone(), reversed(x))).collect::<Map<_, _>>()),
        Value::Array(a) => Value::Array(a.iter().map(reversed).collect()),
        other => other.clone(),
    }
}

const TEMPLATED: &str = r#"
// comments, unquoted keys and trailing commas are JSON5
{
  schema: "kiln.hw/1.0", name: "h", tech: "tsmc_n5",
  meta: { description: "first" },
  params: { edge: 32, cap: "512KiB" },
  clocks: [ { id: "clk", freq: "1.5GHz" } ],
  templates: { core: { kind: "cluster", body: {
    units: [ { id: "mx", kind: "matrix", geometry: { systolic: { rows: "=edge", cols: "=edge" } },
               precisions: ["int8*int8+int32@2", "bf16*bf16+fp32"],
               local: [ { id: "w", holds: "b", capacity: "= edge * edge * 2" } ], feeds: { any: "m" } } ],
    memories: [ { id: "m", kind: "scratchpad", capacity: "=cap", ports: [ { dir: "rw", width_bits: 512 } ], notes: "assumed" } ],
  } } },
  system: { die: { id: "d", default_clock: "clk", clusters: [ { id: "c", count: 2, use: "core" } ] } },
}"#;

const INLINE: &str = r#"{
  "schema": "kiln.hw/1.0", "name": "h", "tech": "tsmc_n5",
  "meta": { "description": "second", "authors": ["someone"] },
  "clocks": [ { "freq": 1500000000, "id": "clk" } ],
  "system": { "boards": [ { "id": "board", "packages": [ { "id": "chip", "dies": [ { "id": "d", "default_clock": "clk",
    "clusters": [ { "count": 2, "id": "c",
      "memories": [ { "id": "m", "kind": "scratchpad", "capacity": 524288, "word_bits": 32, "banks": 1,
                      "ports": [ { "dir": "rw", "width_bits": 512, "count": 1 } ] } ],
      "units": [ { "id": "mx", "kind": "matrix", "geometry": { "systolic": { "cols": 32, "rows": 32 } },
                   "dataflow": "weight_stationary",
                   "precisions": ["bf16*bf16+fp32@1", { "a": "int8", "b": "int8", "acc": "int32", "rate": 2 }],
                   "local": [ { "id": "w", "holds": "b", "capacity": "2KiB" } ], "feeds": { "any": { "from": "m" } } } ] } ] } ] } ] } ] }
}"#;

#[test]
fn equivalent_authorings_hash_identically() {
    let a = load(TEMPLATED);
    let b = load(INLINE);
    assert_eq!(hash_view(&a.canonical), hash_view(&b.canonical));
    assert_eq!(a.hash, b.hash);
    assert_eq!(a.notes.get("board.chip.d.c.m").map(String::as_str), Some("assumed"));
}

#[test]
fn hash_is_stable_under_key_order_and_formatting() {
    let a = load(TEMPLATED);
    let pretty = serde_json::to_string_pretty(&reversed(&a.canonical)).unwrap();
    assert_eq!(load(&pretty).hash, a.hash);
    let v5e = load_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference/tpu_v5e.json5")).unwrap();
    let pretty = serde_json::to_string_pretty(&reversed(&v5e.canonical)).unwrap();
    assert_eq!(load(&pretty).hash, v5e.hash);
}

#[test]
fn semantic_edits_change_the_hash() {
    let a = load(TEMPLATED);
    for (from, to) in [("count: 2", "count: 3"), ("cap: \"512KiB\"", "cap: \"1MiB\""), ("\"1.5GHz\"", "\"1.6GHz\""), ("id: \"mx\"", "id: \"mxu\"")] {
        assert_ne!(load(&TEMPLATED.replace(from, to)).hash, a.hash, "{from} -> {to}");
    }
    assert_eq!(load(&TEMPLATED.replace("\"first\"", "\"other\"")).hash, a.hash, "meta is not hashed");
}

#[test]
fn golden_hash() {
    assert_eq!(load(TEMPLATED).hash, GOLDEN);
}

/// Changing this means the canonical form changed: that is a schema-major event (01 §19).
const GOLDEN: &str = "hw1-da1975bc149f42ff0091ab41e25f75a6";
