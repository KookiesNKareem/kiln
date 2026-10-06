//! Property tests: expansion is deterministic, independent of key order and literal form, and structurally
//! invariant under consistent renaming of ids (01 §15, §17).

use kiln_ir::hw::{Design, ExpandOptions, MemLoader, Profile, check};
use kiln_ir::precision::Precision;
use proptest::prelude::*;
use serde_json::{Map, Value, json};

#[derive(Clone, Copy)]
struct Names {
    tile: &'static str,
    sram: &'static str,
    mxu: &'static str,
    noc: &'static str,
}

const A: Names = Names { tile: "tile", sram: "sram", mxu: "mxu", noc: "noc" };
const B: Names = Names { tile: "blk", sram: "buf", mxu: "mac", noc: "fabric" };

#[derive(Clone, Copy, Debug)]
struct Params {
    gr: u32,
    gc: u32,
    rows: u32,
    cols: u32,
    lanes: u32,
    sram_kib: u32,
    banks: u32,
    literal: bool,
}

fn build(p: Params, n: Names) -> Value {
    let cap = if p.literal { json!(format!("{}KiB", p.sram_kib)) } else { json!(p.sram_kib * 1024) };
    json!({
        "schema": "kiln.hw/1.0", "name": "prop", "tech": "tsmc_n5",
        "clocks": [ { "id": "clk", "freq": if p.literal { json!("1.25GHz") } else { json!(1.25e9) } } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "clk",
                "clusters": [ { "id": n.tile, "layout": { "grid": [p.gr, p.gc] },
                    "units": [
                        { "id": n.mxu, "kind": "matrix", "geometry": { "systolic": { "rows": p.rows, "cols": p.cols } },
                          "precisions": ["bf16*bf16+fp32", "int8*int8+int32@2"],
                          "local": [ { "id": "w", "holds": "b", "capacity": p.rows * p.cols * 2 } ],
                          "feeds": { "a": n.sram, "b": n.sram, "o": n.sram } },
                        { "id": "vpu", "kind": "vector", "lanes": p.lanes, "precisions": ["fp32@1"], "feeds": { "any": n.sram } } ],
                    "memories": [ { "id": n.sram, "kind": "scratchpad", "capacity": cap, "banks": p.banks,
                                    "ports": [ { "dir": "rw", "width_bits": 256 } ] } ] } ],
                "networks": [ { "id": n.noc, "topology": { "type": "mesh", "dims": [p.gr, p.gc] },
                                "endpoints": [ { "select": format!("{}*.{}", n.tile, n.sram), "at": "layout" } ], "link": "128b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": "6.4Gbps", "attach": format!("die.{}", n.noc) } ] } }
    })
}

fn shuffled(v: &Value, seed: &mut u64) -> Value {
    match v {
        Value::Object(m) => {
            let mut entries: Vec<(&String, &Value)> = m.iter().collect();
            for i in (1..entries.len()).rev() {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                entries.swap(i, (*seed >> 33) as usize % (i + 1));
            }
            Value::Object(entries.into_iter().map(|(k, x)| (k.clone(), shuffled(x, seed))).collect::<Map<_, _>>())
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| shuffled(x, seed)).collect()),
        other => other.clone(),
    }
}

fn load(v: &Value) -> Design {
    Design::from_source(&MemLoader::default(), None, &serde_json::to_string_pretty(v).unwrap()).unwrap()
}

fn params() -> impl Strategy<Value = Params> {
    (1u32..4, 1u32..4, 1u32..33, 1u32..33, 1u32..65, 1u32..65, prop::sample::select(vec![1u32, 2, 4]), any::<bool>()).prop_map(
        |(gr, gc, rows, cols, lanes, sram_kib, banks, literal)| Params { gr, gc, rows, cols, lanes, sram_kib, banks, literal },
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn expansion_is_deterministic_and_key_order_invariant(p in params(), seed in any::<u64>()) {
        let doc = build(p, A);
        let a = load(&doc);
        let mut s = seed;
        let b = load(&shuffled(&doc, &mut s));
        let c = load(&build(Params { literal: !p.literal, ..p }, A));
        prop_assert_eq!(&a.hash, &b.hash);
        prop_assert_eq!(&a.hash, &c.hash);
        let opts = ExpandOptions::default();
        let ma = serde_json::to_string(&a.expand(&opts).unwrap().0).unwrap();
        prop_assert_eq!(&ma, &serde_json::to_string(&b.expand(&opts).unwrap().0).unwrap());
        prop_assert_eq!(&ma, &serde_json::to_string(&a.expand(&opts).unwrap().0).unwrap());
        let r = check(a, Profile::Full, &opts);
        prop_assert!(!r.has_errors(), "{:?}", r.diagnostics);
    }

    #[test]
    fn expansion_structure_is_renaming_invariant(p in params()) {
        let opts = ExpandOptions::default();
        let (a, b) = (load(&build(p, A)), load(&build(p, B)));
        prop_assert_ne!(&a.hash, &b.hash);
        let (ma, mb) = (a.expand(&opts).unwrap().0, b.expand(&opts).unwrap().0);
        prop_assert_eq!(ma.units.len(), mb.units.len());
        prop_assert_eq!(ma.memories.len(), mb.memories.len());
        prop_assert_eq!(ma.routers.len(), mb.routers.len());
        prop_assert_eq!(ma.channels.len(), mb.channels.len());
        prop_assert_eq!(&ma.levels, &mb.levels);
        let kinds = |m: &kiln_ir::hw::HwModel| m.channels.iter().map(|c| (c.src, c.dst, c.kind)).collect::<Vec<_>>();
        prop_assert_eq!(kinds(&ma), kinds(&mb));
        prop_assert_eq!(ma.peak_ops_for(Precision::Bf16, None), mb.peak_ops_for(Precision::Bf16, None));
        let expected = 2.0 * f64::from(p.gr * p.gc * p.rows * p.cols) * 1.25e9;
        prop_assert_eq!(ma.peak_ops_for(Precision::Bf16, None), expected);
    }
}
