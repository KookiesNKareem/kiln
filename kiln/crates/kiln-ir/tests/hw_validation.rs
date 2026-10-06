//! One broken design per major diagnostic family: each mutation of a clean base must raise its code.

use kiln_ir::hw::{MemLoader, Profile, check_str};

const BASE: &str = r#"{
  schema: "kiln.hw/1.0", name: "mini", tech: "tsmc_n7",
  meta: {
    citations: { x: "test fixture" },
    claims: [ { metric: "peak_ops.bf16", value: 2.048e12, source: "x" }, { metric: "offchip_bw", value: 819.2e9, source: "x" } ],
  },
  clocks: [ { id: "clk", freq: "1GHz" } ],
  system: { package: {
    id: "chip",
    dies: [ {
      id: "die", default_clock: "clk",
      clusters: [ { id: "tile", count: 4, layout: { grid: [2, 2] },
        units: [
          { id: "mxu", kind: "matrix", geometry: { systolic: { rows: 16, cols: 16 } },
            precisions: ["bf16*bf16+fp32"], local: [ { id: "w", holds: "b", capacity: "1KiB" } ],
            feeds: { a: "sram", b: "sram", o: "sram" } },
          { id: "vpu", kind: "vector", lanes: 16, precisions: ["fp32@1"], feeds: { any: "sram" } },
        ],
        memories: [ { id: "sram", kind: "scratchpad", capacity: "256KiB", banks: 4, word_bits: 128,
                      ports: [ { dir: "rw", width_bits: 512 } ] } ],
      } ],
      networks: [ { id: "noc", topology: { type: "mesh", dims: [2, 2] },
                    endpoints: [ { select: "tile*.sram", at: "layout" } ], link: "256b" } ],
    } ],
    mem_stacks: [ { id: "hbm", kind: "hbm3", capacity: "8GiB", io_width_bits: 1024, pin_rate_bits_per_s: "6.4Gbps",
                    attach: { network: "die.noc" } } ],
  } },
}"#;

fn codes_with(src: &str, profile: Profile) -> Vec<String> {
    let r = check_str(&MemLoader::default(), None, src, profile);
    r.diagnostics.iter().map(|d| d.code.clone()).collect()
}

fn codes(src: &str) -> Vec<String> {
    codes_with(src, Profile::Full)
}

fn mutate(from: &str, to: &str) -> String {
    assert!(BASE.contains(from), "fixture lacks {from:?}");
    BASE.replacen(from, to, 1)
}

#[track_caller]
fn expect(src: &str, code: &str) {
    let c = codes(src);
    assert!(c.iter().any(|x| x == code), "expected {code}, got {c:?}");
}

#[test]
fn base_is_clean_in_every_profile() {
    for p in [Profile::Full, Profile::Reference, Profile::Search] {
        assert_eq!(codes_with(BASE, p), Vec::<String>::new(), "{p:?}");
    }
}

#[test]
fn e01_parse_and_schema() {
    expect(&mutate("banks: 4,", "bankz: 4,"), "E-IR-0101");
    expect(&mutate("kind: \"scratchpad\", capacity: \"256KiB\",", "kind: \"scratchpad\","), "E-IR-0102");
    expect(&mutate("count: 4, layout", "count: \"four\", layout"), "E-IR-0103");
    expect(&mutate("id: \"vpu\"", "id: \"VPU\""), "E-IR-0104");
    expect(&mutate("id: \"tile\"", "id: \"tile2\""), "E-IR-0105");
    expect(&mutate("id: \"vpu\"", "id: \"mxu\""), "E-IR-0106");
    expect(&mutate("schema: \"kiln.hw/1.0\"", "schema: \"kiln.hw/7.0\""), "E-IR-0107");
    expect(&mutate("capacity: \"256KiB\"", "capacity: \"1GHz\""), "E-IR-0108");
    expect(&mutate("freq: \"1GHz\"", "freq: \"141GHz\""), "E-IR-0109");
    expect(&mutate("capacity: \"1KiB\"", "capacity: \"0.5B\""), "E-IR-0110");
}

#[test]
fn e02_templates_and_expansion() {
    expect(&mutate("count: 4, layout: { grid: [2, 2] }", "count: 3, layout: { grid: [2, 2] }"), "E-IR-0213");
    expect(&mutate("select: \"tile*.sram\"", "select: \"tyle*.sram\""), "E-IR-0205");
    expect(&mutate("feeds: { any: \"sram\" }", "feeds: { any: \"sramm\" }"), "E-IR-0206");
    expect(&mutate("count: 4, layout: { grid: [2, 2] },", "count: 4, layout: { grid: [2, 2] }, disabled: [\"tile[9]\"],"), "E-IR-0211");
    expect(&mutate("count: 4, layout: { grid: [2, 2] },", "count: 4, layout: { grid: [2, 2] }, vary: [ { select: \"tile[7]\", set: { } } ],"), "E-IR-0211");
    let patched = BASE.replacen("system:", r#"set: [ { "board.chip.die.tile.sram#nonexistent": 1 } ], system:"#, 1);
    expect(&patched, "E-IR-0212");
    let tight = check_str(&MemLoader::default(), None, BASE, Profile::Full);
    assert!(!tight.has_errors());
    let d = kiln_ir::hw::Design::from_source(&MemLoader::default(), None, BASE).unwrap();
    let small = kiln_ir::hw::ExpandOptions { max_instances: 5 };
    let errs = d.expand(&small).unwrap_err();
    assert!(errs.iter().any(|e| e.code == "E-IR-0210"), "{errs:?}");
}

#[test]
fn vary_patches_individual_instances() {
    let src = mutate("{ id: \"vpu\", kind: \"vector\", lanes: 16,", "{ id: \"vpu\", count: 2, vary: [ { select: \"vpu[1]\", set: { lanes: 64 } } ], kind: \"vector\", lanes: 16,");
    let r = check_str(&MemLoader::default(), None, &src, Profile::Full);
    assert!(!r.has_errors(), "{:?}", r.diagnostics);
    let m = r.model.unwrap();
    let lanes: Vec<u64> = m.units.iter().filter(|u| u.spec.id.as_str() == "vpu").map(|u| u.spec.kind.base_ops_per_cycle()).take(2).collect();
    assert_eq!(lanes, [16, 64]);
}

#[test]
fn e03_compute_units() {
    expect(&mutate("precisions: [\"fp32@1\"]", "precisions: []"), "E-IR-0301");
    expect(&mutate("precisions: [\"fp32@1\"]", "precisions: [\"fp7@1\"]"), "E-IR-0302");
    expect(&mutate("precisions: [\"fp32@1\"]", "precisions: [\"int8_pc@1\"]"), "E-IR-0302");
    expect(&mutate("\"bf16*bf16+fp32\"", "\"int8*int8+fp16\""), "E-IR-0303");
    expect(&mutate("feeds: { a: \"sram\", b: \"sram\", o: \"sram\" }", "feeds: { a: \"sram\", o: \"sram\" }"), "E-IR-0304");
    expect(&mutate("lanes: 16, precisions", "lanes: 16, ops: [\"matmul\"], precisions"), "E-IR-0306");
    expect(&mutate("lanes: 16", "lanes: 0"), "E-IR-0307");
    expect(&mutate("geometry: { systolic: { rows: 16, cols: 16 } },", "geometry: { outer_product: { rows: 16, cols: 16 } }, dataflow: \"row_stationary\", local: [], accumulate_in: \"feed\","), "E-IR-0308");
    expect(&mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\"], sparsity: [ { pattern: \"4:2\", operand: \"a\", speedup: 2 } ],"), "E-IR-0309");
    expect(&mutate("feeds: { any: \"sram\" }", "feeds: { any: { from: \"sram\", width_bits: 4096 } }"), "E-IR-0310");
    expect(&mutate("capacity: \"1KiB\"", "capacity: \"256B\""), "E-IR-0311");
    expect(&mutate("[\"fp32@1\"]", "[\"fp32@1\", \"fp32@2\"]"), "W-IR-0313");
}

#[test]
fn e04_onchip_memory() {
    expect(&mutate("capacity: \"256KiB\"", "capacity: \"256100B\""), "E-IR-0401");
    expect(&mutate("ports: [ { dir: \"rw\", width_bits: 512 } ]", "ports: []"), "E-IR-0402");
    expect(&mutate("kind: \"scratchpad\"", "kind: \"cache\""), "E-IR-0403");
    expect(
        &mutate("word_bits: 128,", "word_bits: 128, operands: { policy: \"carveout\", options: [ { scratch: \"200KiB\", cache: \"100KiB\" } ] }, cache: { line: \"128B\", ways: 4 },"),
        "E-IR-0405",
    );
    expect(&mutate("word_bits: 128", "word_bits: 96"), "E-IR-0408");
    expect(&mutate("word_bits: 128,", "word_bits: 128, overrides: { bandwidth: \"1TB/s\" },"), "E-IR-0409");
    expect(&mutate("memories: [ { id: \"sram\"", "memories: [ { id: \"spare\", kind: \"fifo\", capacity: 64, ports: [ { dir: \"rw\", width_bits: 64 } ] }, { id: \"sram\""), "W-IR-0407");
}

#[test]
fn e05_offchip() {
    expect(&mutate("attach: { network: \"die.noc\" }", "dies_high: 8"), "E-IR-0501");
    expect(&mutate("pin_rate_bits_per_s: \"6.4Gbps\",", "pin_rate_bits_per_s: \"6.4Gbps\", overrides: { bandwidth: \"2TB/s\" },"), "E-IR-0506");
    expect(&mutate("pin_rate_bits_per_s: \"6.4Gbps\"", "pin_rate_bits_per_s: \"12Gbps\""), "W-IR-0503");
}

#[test]
fn e06_near_memory() {
    let near = "{ id: \"pim\", kind: \"matrix\", geometry: { mma: { m: 1, n: 16, k: 16 } }, precisions: [\"bf16*bf16+fp32\"], near: { memory: \"sram\", granularity: \"per_bank\" } },";
    let src = mutate("units: [", &format!("units: [ {near}"));
    assert!(!codes(&src).iter().any(|c| c.starts_with("E-")), "{:?}", codes(&src));
    expect(&src.replace("granularity: \"per_bank\" }", "granularity: \"per_pseudo_channel\" }"), "E-IR-0602");
    expect(&src.replace("granularity: \"per_bank\" } },", "granularity: \"per_bank\" }, feeds: { a: \"w2\" } },").replacen("memories: [ {", "memories: [ { id: \"w2\", kind: \"scratchpad\", capacity: 64, ports: [ { dir: \"rw\", width_bits: 64 } ] }, {", 1), "E-IR-0603");
    expect(&src.replace("kind: \"matrix\", geometry: { mma: { m: 1, n: 16, k: 16 } }, precisions", "count: 3, kind: \"matrix\", geometry: { mma: { m: 1, n: 16, k: 16 } }, precisions"), "E-IR-0607");
    expect(&src.replace("memory: \"sram\"", "memory: \"die\""), "E-IR-0601");
    let cim = "{ id: \"cim\", kind: \"cim\", rows: 64, cols: 64, weight_capacity: \"1KiB\", precisions: [\"int8*int8+int32\"], near: { memory: \"sram\", granularity: \"per_instance\" } },";
    expect(&mutate("units: [", &format!("units: [ {cim}")), "E-IR-0606");
}

#[test]
fn e06_cim_rate_is_derived() {
    let cim = |extra: &str, modes: &str| {
        mutate("units: [", &format!("units: [ {{ id: \"cim\", kind: \"cim\", rows: 64, cols: 64, {extra} precisions: [{modes}], near: {{ memory: \"sram\", granularity: \"per_instance\" }} }},"))
    };
    let ok = cim("weight_capacity: \"512B\",", "\"int8*int4+int32\", \"int8*int8+int32@0.5\"");
    assert!(!codes(&ok).iter().any(|c| c.starts_with("E-")), "{:?}", codes(&ok));
    for p in [Profile::Full, Profile::Search] {
        assert!(codes_with(&cim("", "\"int8*int4+int32@2\""), p).contains(&"E-IR-0608".into()), "{p:?}: over-claimed CIM rate");
    }
    expect(&cim("parallel_rows: 128,", "\"int8*int4+int32\""), "E-IR-0609");
    expect(&cim("style: \"analog\",", "\"int8*int4+int32\""), "E-IR-0609");
    expect(&cim("style: \"analog\", adc_bits: 4,", "\"int8*int4+int32\""), "W-IR-0610");
    assert!(codes_with(&cim("weight_write: \"10GB/s\",", "\"int8*int4+int32\""), Profile::Search).contains(&"E-IR-1101".into()));
}

#[test]
fn e07_interconnect() {
    expect(&mutate("select: \"tile*.sram\", at: \"layout\"", "select: \"tile*\", at: \"layout\""), "E-IR-0701");
    expect(&mutate("endpoints: [ { select: \"tile*.sram\", at: \"layout\" } ]", "endpoints: [ { select: \"tile*.sram\", at: \"layout\" }, { select: \"tile[0].sram\", at: \"layout\" } ]"), "E-IR-0702");
    expect(&mutate("dims: [2, 2]", "dims: [1, 2]"), "E-IR-0703");
    expect(&mutate("link: \"256b\" }", "link: \"256b\", router: { radix: 2 } }"), "E-IR-0704");
    let custom = mutate("topology: { type: \"mesh\", dims: [2, 2] },", "topology: { type: \"custom\", routers: 3, edges: [ { a: 0, b: 5 } ] },")
        .replacen("at: \"layout\" } ]", "at: { router: [0] } } ]", 1);
    expect(&custom, "E-IR-0705");
    expect(&mutate("link: \"256b\"", "link: { width_bits: 0 }"), "E-IR-0706");
    expect(&mutate("type: \"mesh\", dims: [2, 2] }", "type: \"torus\", dims: [2, 2] }, router: { vcs: 1 }"), "E-IR-0707");
    expect(&mutate("link: \"256b\"", "link: { width_bits: 256, bandwidth: \"1TB/s\" }"), "E-IR-0712");
    expect(&mutate("link: \"256b\"", "link: { width_bits: 64, phys: { type: \"serdes\", protocol: \"custom\", lanes: 4, lane_rate_bits_per_s: \"100Gbps\" } }"), "E-IR-0713");
    expect(&mutate("feeds: { any: \"sram\" }", "feeds: { any: { from: \"sram\", via: \"noc\" } }"), "E-IR-0714");
    expect(&mutate("attach: { network: \"die.noc\" }", "attach: { controllers: [] }"), "E-IR-0720");
}

#[test]
fn e07_direct_networks_need_ports() {
    let multi = r#"{ schema: "kiln.hw/1.0", name: "multi", tech: "tsmc_n5", clocks: [ { id: "c", freq: "1GHz" } ],
      system: { boards: [ { id: "b",
        packages: [ { id: "p", count: 4, layout: { grid: [2, 2] }, dies: [ { id: "d", default_clock: "c",
          units: [ { id: "v", kind: "vector", lanes: 8, precisions: ["fp32@1"], feeds: { any: "m" } } ],
          memories: [ { id: "m", kind: "scratchpad", capacity: "1MiB", ports: [ { dir: "rw", width_bits: 256 } ] } ],
          networks: [ { id: "x", topology: "crossbar", endpoints: ["m"], link: "256b" } ],
          ports: [ { id: "ici", count: 3, kind: "serdes", internal: "x" } ] } ],
          mem_stacks: [ { id: "hbm", kind: "hbm3", capacity: "16GiB", io_width_bits: 1024, pin_rate_bits_per_s: "6.4Gbps", attach: "d.x" } ] } ],
        networks: [ { id: "ici", topology: { type: "torus", dims: [2, 2] }, endpoints: [ { select: "p*", at: "layout", ports: "d.ici*" } ],
                      link: { phys: { type: "serdes", protocol: "ici_like", lanes: 4, lane_rate_bits_per_s: "100Gbps" } } } ] } ] } }"#;
    expect(multi, "E-IR-0710");
    let ok = multi.replace("count: 3, kind: \"serdes\"", "count: 4, kind: \"serdes\"");
    assert!(!codes(&ok).iter().any(|c| c.starts_with("E-")), "{:?}", codes(&ok));
    expect(&ok.replace("count: 4, kind: \"serdes\"", "count: 5, kind: \"serdes\""), "W-IR-0726");
    let no_ici = ok.replace("networks: [ { id: \"ici\"", "unused: [ { id: \"ici\"");
    assert!(codes(&no_ici).contains(&"E-IR-0101".to_string()));
}

#[test]
fn e08_floorplan_and_e09_clocks() {
    expect(&mutate("count: 4, layout: { grid: [2, 2] },", "count: 4, layout: { grid: [2, 2] }, placement: { mode: \"pinned\", x: -5, y: 0 },"), "E-IR-0811");
    expect(&mutate("id: \"vpu\", kind", "id: \"vpu\", count: 2, placement: { mode: \"array\" }, kind"), "E-IR-0812");
    expect(&mutate("id: \"vpu\", kind", "id: \"vpu\", clock: \"nope\", kind"), "E-IR-0901");
    expect(&mutate("freq: \"1GHz\" }", "freq: \"1GHz\", vf: [ { freq: \"1GHz\", voltage: \"0.8V\" }, { freq: \"0.5GHz\", voltage: \"0.7V\" } ] }"), "E-IR-0902");
    expect(&mutate("freq: \"1GHz\" }", "freq: \"1GHz\", vf: [ { freq: \"0.5GHz\", voltage: \"0.7V\" }, { freq: \"0.8GHz\", voltage: \"0.8V\" } ] }"), "E-IR-0903");
    expect(
        &mutate("clocks: [ { id: \"clk\", freq: \"1GHz\" } ],", "clocks: [ { id: \"clk\", freq: \"1GHz\" } ], power: [ { id: \"a\", members: \"board.chip\", cap: \"100W\" }, { id: \"b\", members: \"board.chip.die\", cap: \"90W\" } ],"),
        "E-IR-0904",
    );
    expect(&mutate("tech: \"tsmc_n7\"", "tech: \"tsmc_n1\""), "E-IR-1001");
    let layered = mutate("id: \"chip\",", "id: \"chip\", layers: [ { id: \"top\", index: 1, bond: \"hybrid\", pitch_um: 9 } ],")
        .replacen("id: \"die\", default_clock", "id: \"die\", layer: \"top\", default_clock", 1);
    expect(&layered, "E-IR-0809");
}

#[test]
fn e18_claims_and_profiles() {
    expect(&mutate("value: 2.048e12", "value: 3e12"), "W-IR-1801");
    let sram_override = mutate("word_bits: 128,", "word_bits: 128, overrides: { latency: 3 },");
    assert!(codes_with(&sram_override, Profile::Reference).contains(&"E-IR-1103".into()));
    assert!(!codes_with(&sram_override.replace("latency: 3 }", "latency: 3, source: \"x\" }"), Profile::Reference).contains(&"E-IR-1103".into()));
    assert!(codes_with(&mutate("claims: [", "claims_: [").replace("claims_", "notes: {}, claims").replace("{ metric: \"offchip_bw\", value: 819.2e9, source: \"x\" }", ""), Profile::Reference).contains(&"E-IR-1103".into()));
    let derate = mutate("word_bits: 128,", "word_bits: 128, overrides: { bandwidth: \"10GB/s\" },");
    assert!(!codes_with(&derate, Profile::Search).iter().any(|c| c.starts_with("E-")), "bandwidth de-rates are cost-neutral (00 decision 3)");
    let hbm = |bw: &str| mutate("pin_rate_bits_per_s: \"6.4Gbps\",", &format!("pin_rate_bits_per_s: \"6.4Gbps\", overrides: {{ bandwidth: \"{bw}\" }},"));
    let r = check_str(&MemLoader::default(), None, &hbm("1638.4GB/s"), Profile::Search);
    let over = r.diagnostics.iter().find(|d| d.code == "E-IR-1101").expect("2x HBM override is unpriced");
    assert!(over.message.contains("819.2 GB/s") && over.message.contains("1638.4 GB/s") && over.message.contains("ratio 2.00"), "{}", over.message);
    assert_eq!(over.path.as_deref(), Some("board.chip.hbm"));
    assert!(!codes_with(&hbm("737.28GB/s"), Profile::Search).iter().any(|c| c.starts_with("E-")), "0.9x de-rate");
    let sram_lat = codes_with(&sram_override, Profile::Search);
    assert!(sram_lat.contains(&"E-IR-1101".into()), "memory latency is not derivable yet: unverifiable");
    let serdes = mutate("link: \"256b\"", "link: { width_bits: 256, phys: { type: \"on_die\" }, latency: \"1ns\" }");
    assert!(codes_with(&serdes, Profile::Search).contains(&"E-IR-1101".into()));
    let resplit = mutate("capacity: \"256KiB\", banks: 4,", "capacity: \"256KiB\", banks: 8,")
        .replacen("count: 4, layout: { grid: [2, 2] },", "count: 4, layout: { grid: [2, 2] }, placement: { mode: \"array\" },", 1);
    assert_eq!(codes_with(&resplit, Profile::Search), Vec::<String>::new(), "re-placement and bank split are cost-neutral (00 decision 3)");
    let fam = mutate("name: \"mini\",", "name: \"mini\", family: \"tpu_v5e\",");
    assert!(codes_with(&fam, Profile::Search).contains(&"E-IR-1102".into()));
    assert!(!codes_with(&fam, Profile::Full).contains(&"E-IR-1102".into()));
    let fill = mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\"], pipeline: { fill: 4 },");
    let r = check_str(&MemLoader::default(), None, &fill, Profile::Search);
    assert!(r.diagnostics.iter().any(|d| d.code == "E-IR-1101" && d.message.contains("below derived 31")), "{:#?}", r.diagnostics);
    let slow_fill = fill.replace("fill: 4", "fill: 64");
    assert!(!codes_with(&slow_fill, Profile::Search).contains(&"E-IR-1101".into()));
    let custom = mutate("kind: \"hbm3\"", "kind: \"custom\"");
    assert!(codes_with(&custom, Profile::Search).contains(&"E-IR-1104".into()));
    assert!(codes_with(BASE, Profile::StreamCompat).contains(&"E-IR-1105".into()), "mesh NoC is outside stream_compat");
}

#[test]
fn search_bounds_class_and_fn_rate_multipliers() {
    let vec_rates = |r: &str| mutate("lanes: 16, precisions", &format!("lanes: 16, class_rates: {{ {r} }}, precisions"));
    assert_eq!(codes_with(&vec_rates("elementwise: 1, transcendental: 0.0625"), Profile::Search), Vec::<String>::new(), "de-rates are cost-neutral");
    for r in ["elementwise: 1000000", "transcendental: 0.25"] {
        assert!(codes_with(&vec_rates(r), Profile::Search).contains(&"E-IR-1101".into()), "{r}: above the table 6.3 default is unpriced");
        assert!(!codes(&vec_rates(r)).iter().any(|c| c.starts_with("E-")), "{r}: legal outside search");
    }
    expect(&vec_rates("elementwise: 0"), "E-IR-0303");
    expect(&vec_rates("elementwise: -1"), "E-IR-0303");
    let sfu = |r: &str| {
        mutate("units: [", &format!("units: [ {{ id: \"sfu\", kind: \"special\", lanes: 4, functions: [\"exp\"], fn_rates: {{ {r} }}, precisions: [\"fp32@1\"], feeds: {{ any: \"sram\" }} }},"))
    };
    assert_eq!(codes_with(&sfu("exp: 0.5"), Profile::Search), Vec::<String>::new());
    assert!(codes_with(&sfu("exp: 1000"), Profile::Search).contains(&"E-IR-1101".into()));
    expect(&sfu("exp: 0"), "E-IR-0303");
}

#[test]
fn custom_edge_links_are_checked_like_network_links() {
    let edge = |link: &str| {
        mutate("topology: { type: \"mesh\", dims: [2, 2] },", &format!("topology: {{ type: \"custom\", routers: 2, edges: [ {{ a: 0, b: 1{link} }} ] }},"))
            .replacen("at: \"layout\" } ]", "at: { router: [0] } } ]", 1)
    };
    assert_eq!(codes_with(&edge(""), Profile::Search), Vec::<String>::new());
    let cheap = edge(", link: { width_bits: 256, energy: 0, latency: 0 }");
    assert!(codes_with(&cheap, Profile::Search).contains(&"E-IR-1101".into()), "{:?}", codes_with(&cheap, Profile::Search));
    let fec = edge(", link: { phys: { type: \"serdes\", protocol: \"custom\", lanes: 4, lane_rate_bits_per_s: \"100Gbps\", encoding_efficiency: 100 } }");
    assert!(codes_with(&fec, Profile::Search).contains(&"E-IR-1101".into()), "{:?}", codes_with(&fec, Profile::Search));
    expect(&edge(", link: { width_bits: 0 }"), "E-IR-0706");
    expect(&edge(", link: { width_bits: 256, bandwidth: \"1TB/s\" }"), "E-IR-0712");
}

#[test]
fn near_unit_grid_must_match_granules() {
    let near = |rep: &str| {
        mutate("units: [", &format!("units: [ {{ id: \"pim\", {rep} kind: \"matrix\", geometry: {{ mma: {{ m: 1, n: 16, k: 16 }} }}, precisions: [\"bf16*bf16+fp32\"], near: {{ memory: \"sram\", granularity: \"per_bank\" }} }},"))
    };
    assert!(!codes(&near("layout: { grid: [4] },")).iter().any(|c| c.starts_with("E-")), "{:?}", codes(&near("layout: { grid: [4] },")));
    expect(&near("layout: { grid: [100] },"), "E-IR-0607");
    expect(&near("layout: { ring: 8 },"), "E-IR-0607");
}

#[test]
fn vary_patches_resolve_per_instance_clocks() {
    let src = mutate("clocks: [ { id: \"clk\", freq: \"1GHz\" } ]", "clocks: [ { id: \"clk\", freq: \"1GHz\" }, { id: \"slow\", freq: \"500MHz\" } ]")
        .replacen("{ id: \"vpu\", kind: \"vector\",", "{ id: \"vpu\", count: 2, vary: [ { select: \"vpu[1]\", set: { clock: \"slow\" } } ], kind: \"vector\",", 1)
        .replacen("memories: [ { id: \"sram\",", "memories: [ { id: \"spad\", count: 2, vary: [ { select: \"spad[1]\", set: { clock: \"slow\" } } ], kind: \"scratchpad\", capacity: \"64KiB\", ports: [ { dir: \"rw\", width_bits: 512 } ] }, { id: \"sram\",", 1);
    let r = check_str(&MemLoader::default(), None, &src, Profile::Full);
    let m = r.model.expect("expands");
    let hz = |c| m.clock_hz(c).map(|h| h.0);
    let units: Vec<_> = m.units.iter().filter(|u| u.spec.id.as_str() == "vpu").take(2).map(|u| hz(u.clock)).collect();
    assert_eq!(units, [Some(1e9), Some(5e8)]);
    let mems: Vec<_> = m.memories.iter().filter(|x| m.nodes[x.node].entity_id == "spad").take(2).map(|x| (hz(x.clock), x.bandwidth_derived.map(|b| b.0))).collect();
    assert_eq!(mems, [(Some(1e9), Some(64e9)), (Some(5e8), Some(32e9))]);
}

#[test]
fn search_rejects_lossy_analog_cim() {
    let cim = |adc: u32| {
        mutate("units: [", &format!("units: [ {{ id: \"cim\", kind: \"cim\", rows: 256, cols: 64, input_bits_per_cycle: 4, style: \"analog\", adc_bits: {adc}, precisions: [\"int8*int8+int32\"], near: {{ memory: \"sram\", granularity: \"per_instance\" }} }},"))
    };
    for adc in [0, 11] {
        assert!(codes(&cim(adc)).contains(&"W-IR-0610".into()));
        assert!(!codes(&cim(adc)).iter().any(|c| c.starts_with("E-")), "{adc}: lossy analog CIM stays legal outside search");
        assert!(codes_with(&cim(adc), Profile::Search).contains(&"E-IR-1107".into()), "{adc}: accuracy loss is unmodeled");
    }
    assert!(!codes_with(&cim(12), Profile::Search).iter().any(|c| c.starts_with("E-")), "{:?}", codes_with(&cim(12), Profile::Search));
}

#[test]
fn overflowing_capacity_sums_are_rejected() {
    let parts = mutate("word_bits: 128,", "word_bits: 128, operands: { policy: \"partitioned\", parts: { a: 9223372036854775808, b: 9223372036854775808 } },");
    expect(&parts, "E-IR-0406");
    let carve = mutate(
        "word_bits: 128,",
        "word_bits: 128, operands: { policy: \"carveout\", options: [ { scratch: 9223372036854775808, cache: 9223372036854775808 } ] }, cache: { line: \"128B\", ways: 4 },",
    );
    expect(&carve, "E-IR-0405");
    let ways = mutate("kind: \"scratchpad\", capacity: \"256KiB\",", "kind: \"cache\", capacity: \"256KiB\", cache: { line: 4611686018427387904, ways: 4 },");
    expect(&ways, "E-IR-0411");
}

#[test]
fn geometry_products_beyond_range_are_rejected() {
    let geo = |g: &str| mutate("geometry: { systolic: { rows: 16, cols: 16 } },", &format!("geometry: {g}, local: [], accumulate_in: \"feed\","));
    expect(&geo("{ mma: { m: 4194304, n: 4194304, k: 1048576 } }"), "E-IR-0109");
    expect(&geo("{ spatial: { dims: { m: 65536, n: 65536, k: 65536 } } }"), "E-IR-0109");
    let ok = geo("{ mma: { m: 64, n: 64, k: 16 } }");
    assert!(!codes(&ok).iter().any(|c| c.starts_with("E-")), "{:?}", codes(&ok));
}

#[test]
fn summary_takes_alternative_modes_once_per_unit() {
    let peak = |src: &str| {
        let m = check_str(&MemLoader::default(), None, src, Profile::Full).model.expect("expands");
        let s = m.summary();
        (s.peak_ops.get("bf16*bf16+fp32").copied(), s.peak_ops.get("bf16*bf16+fp32:sparse").copied(), s.chips[0].elem_ops.get("fp32:elementwise").copied())
    };
    let (dense, sparse, elem) = peak(BASE);
    assert_eq!((dense, sparse), (Some(2.048e12), None));
    let dup = mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\", \"bf16*bf16+fp32\"], sparsity: [ { pattern: \"2:4\", operand: \"b\", speedup: 2 }, { pattern: \"1:4\", operand: \"b\", speedup: 2 } ],")
        .replacen("precisions: [\"fp32@1\"]", "ops: [\"elementwise\", \"elementwise\"], precisions: [\"fp32@1\", \"fp32@1\"]", 1);
    assert_eq!(peak(&dup), (Some(2.048e12), Some(4.096e12), elem));
}

#[test]
fn near_count_is_checked_against_the_resolved_memory() {
    let src = |count: u32| {
        mutate("units: [", &format!("units: [ {{ id: \"pim\", count: {count}, kind: \"vector\", lanes: 16, precisions: [\"fp32@1\"], near: {{ memory: \"sram0\", granularity: \"per_bank\" }} }},"))
            .replacen("{ id: \"sram\", kind: \"scratchpad\"", "{ id: \"sram\", count: 2, kind: \"scratchpad\"", 1)
            .replace("select: \"tile*.sram\"", "select: \"tile*.sram0\"")
            .replace("feeds: { a: \"sram\", b: \"sram\", o: \"sram\" }", "feeds: { a: \"sram0\", b: \"sram0\", o: \"sram0\" }")
            .replace("feeds: { any: \"sram\" }", "feeds: { any: \"sram0\" }")
    };
    let ok = codes_with(&src(4), Profile::Search);
    assert!(!ok.iter().any(|c| c.starts_with("E-")), "{ok:?}");
    let c = codes_with(&src(100), Profile::Search);
    assert!(c.iter().any(|x| x == "E-IR-0607"), "{c:?}");
}

#[test]
fn sparsity_speedup_is_bounded_by_its_structure() {
    let sp = |s: &str| mutate("precisions: [\"bf16*bf16+fp32\"],", &format!("precisions: [\"bf16*bf16+fp32\"], sparsity: [ {s} ],"));
    let ok = sp("{ pattern: \"2:4\", operand: \"b\", speedup: 2, metadata_bits_per_nz: 2 }");
    assert_eq!(codes_with(&ok, Profile::Search), Vec::<String>::new());
    for bad in ["1000000", "2.5"] {
        expect(&sp(&format!("{{ pattern: \"2:4\", operand: \"b\", speedup: {bad}, metadata_bits_per_nz: 2 }}")), "E-IR-0309");
    }
    let search = |s: &str| codes_with(&sp(s), Profile::Search);
    assert!(search("{ pattern: \"2:4\", operand: \"b\", speedup: 2 }").iter().any(|c| c == "E-IR-1104"));
    assert!(search("{ pattern: \"unstructured\", operand: \"b\", speedup: 10, metadata_bits_per_nz: 16 }").iter().any(|c| c == "E-IR-1104"));
    assert!(search("{ pattern: \"block:4\", operand: \"b\", speedup: 10, metadata_bits_per_nz: 16 }").iter().any(|c| c == "E-IR-1104"));
    assert!(!codes(&sp("{ pattern: \"unstructured\", operand: \"b\", speedup: 10 }")).iter().any(|c| c.starts_with("E-")));
}

#[test]
fn cim_capacity_beyond_u64_is_rejected() {
    let cim = "{ id: \"cim\", kind: \"cim\", rows: 256, cols: 2147483648, parallel_rows: 1, cell_bits: 1, weight_sets: 2147483648, weight_capacity: 0, precisions: [\"int8*int8+int32\"], feeds: { a: \"sram\", o: \"sram\" } },";
    let c = codes_with(&mutate("units: [", &format!("units: [ {cim}")), Profile::Search);
    assert!(c.iter().any(|x| x.starts_with("E-")), "{c:?}");
}

#[test]
fn port_bandwidth_does_not_wrap() {
    let src = mutate(
        "capacity: \"256KiB\", banks: 4, word_bits: 128,\n                      ports: [ { dir: \"rw\", width_bits: 512 } ]",
        "capacity: \"4MiB\", banks: 4194304, word_bits: 8,\n                      ports: [ { dir: \"rw\", count: 2097152, width_bits: 2097152, per_bank: true } ]",
    );
    let r = check_str(&MemLoader::default(), None, &src, Profile::Full);
    let m = r.model.expect("expands");
    let bw = m.memories.iter().find(|x| x.bandwidth_derived.is_some() && !x.is_stack()).and_then(|x| x.bandwidth_derived).unwrap();
    assert!(bw.0 > 1e24, "{bw:?}");
}

#[test]
fn overflowing_grids_are_rejected() {
    let src = mutate("units: [", "units: [ { id: \"big\", kind: \"vector\", lanes: 16, precisions: [\"fp32@1\"], feeds: { any: \"sram\" }, layout: { grid: [4194304, 4194304, 1048576] } },");
    let c = codes_with(&src, Profile::Search);
    assert!(c.iter().any(|x| x.starts_with("E-")), "{c:?}");
}

#[test]
fn peak_query_matches_block_precisions() {
    let src = mutate("\"bf16*bf16+fp32\"", "\"mxfp4/16*mxfp4/16+fp32\"");
    let m = check_str(&MemLoader::default(), None, &src, Profile::Full).model.expect("expands");
    assert_eq!(m.peak_ops_for(kiln_ir::precision::Precision::Mxfp4, None), 2.048e12);
}

#[test]
fn unidirectional_rings_are_directed() {
    let hops = |bidir: bool| {
        let src = mutate("topology: { type: \"mesh\", dims: [2, 2] }", &format!("topology: {{ type: \"ring\", bidirectional: {bidir} }}"))
            .replace("at: \"layout\" }", "}");
        let r = check_str(&MemLoader::default(), None, &src, Profile::Full);
        assert!(!r.has_errors(), "{:?}", r.diagnostics);
        let m = r.model.unwrap();
        let rs: Vec<_> = (0..m.routers.len()).map(kiln_ir::hw::NodeIx::Router).collect();
        let n = rs.len();
        let h = |a: usize, b: usize| m.route(rs[a], rs[b]).expect("routable").len();
        (n, h(0, n - 1), h(n - 1, 0))
    };
    let (n, ..) = hops(true);
    assert!(n >= 3);
    assert_eq!(hops(true), (n, 1, 1));
    assert_eq!(hops(false), (n, n - 1, 1));
}

#[test]
fn near_memories_need_a_staging_path() {
    let src = mutate("units: [", "units: [ { id: \"pim\", kind: \"vector\", lanes: 16, precisions: [\"fp32@1\"], near: { memory: \"iso\", granularity: \"per_instance\" } },")
        .replacen("memories: [ {", "memories: [ { id: \"iso\", kind: \"scratchpad\", capacity: \"4KiB\", ports: [ { dir: \"rw\", width_bits: 64 } ] }, {", 1);
    let c = codes_with(&src, Profile::Search);
    assert!(c.iter().any(|x| x == "E-IR-0720"), "{c:?}");
}

#[test]
fn inherited_imports_resolve_against_the_base() {
    let dir = std::env::temp_dir().join(format!("kiln-ir-extends-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("variants")).unwrap();
    let lib = |lanes: u32| format!("{{ schema: \"kiln.hw/1.0\", templates: {{ vpu: {{ body: {{ id: \"vpu\", kind: \"vector\", lanes: {lanes}, precisions: [\"fp32@1\"], feeds: {{ any: \"sram\" }} }} }} }} }}");
    std::fs::write(dir.join("lib.json5"), lib(16)).unwrap();
    std::fs::write(dir.join("variants/lib.json5"), lib(64)).unwrap();
    let base = BASE.replacen("schema: \"kiln.hw/1.0\",", "schema: \"kiln.hw/1.0\", imports: { l: \"lib.json5\" },", 1).replacen(
        "{ id: \"vpu\", kind: \"vector\", lanes: 16, precisions: [\"fp32@1\"], feeds: { any: \"sram\" } },",
        "{ use: \"l.vpu\" },",
        1,
    );
    std::fs::write(dir.join("base.json5"), base).unwrap();
    std::fs::write(dir.join("variants/v.json5"), "{ schema: \"kiln.hw/1.0\", extends: \"../base.json5\", name: \"v\" }").unwrap();
    let r = kiln_ir::hw::check_file(dir.join("variants/v.json5"), Profile::Full);
    std::fs::remove_dir_all(&dir).unwrap();
    let m = r.model.unwrap_or_else(|| panic!("{:?}", r.diagnostics));
    let lanes: Vec<u64> = m.units.iter().filter(|u| u.spec.id.as_str() == "vpu").map(|u| u.spec.kind.base_ops_per_cycle()).collect();
    assert_eq!(lanes[0], 16);
}

#[test]
fn search_rejects_unpriced_scalar_and_special_mode_rates() {
    let unit = |u: &str| mutate("units: [", &format!("units: [ {u},"));
    let sfu = |p: &str| unit(&format!("{{ id: \"sfu\", kind: \"special\", lanes: 1, functions: [\"exp\"], precisions: [\"{p}\"], feeds: {{ any: \"sram\" }} }}"));
    let scalar = |p: &str| unit(&format!("{{ id: \"sc\", kind: \"scalar\", issue_width: 1, precisions: [\"{p}\"], feeds: {{ any: \"sram\" }} }}"));
    for src in [sfu("fp32@1000"), scalar("fp32@1000")] {
        assert!(codes_with(&src, Profile::Search).contains(&"E-IR-1101".into()), "{:?}", codes_with(&src, Profile::Search));
        assert!(!codes(&src).iter().any(|c| c.starts_with("E-")), "legal outside search: {:?}", codes(&src));
    }
    for src in [sfu("fp32@1"), sfu("fp32@0.5"), scalar("fp32@1")] {
        assert_eq!(codes_with(&src, Profile::Search), Vec::<String>::new());
    }
}

#[test]
fn precision_mode_form_matches_unit_kind() {
    expect(&mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\", \"int8\"],"), "E-IR-0302");
    expect(&mutate("precisions: [\"fp32@1\"]", "precisions: [\"bf16*bf16+fp32\"]"), "E-IR-0302");
    let cim = mutate("units: [", "units: [ { id: \"cim\", kind: \"cim\", rows: 64, cols: 64, precisions: [\"int8\"], feeds: { a: \"sram\", o: \"sram\" } },");
    assert!(codes_with(&cim, Profile::Search).contains(&"E-IR-0302".into()), "{:?}", codes_with(&cim, Profile::Search));
}

#[test]
fn summary_peaks_respect_declared_ops() {
    let peak = |src: &str| check_str(&MemLoader::default(), None, src, Profile::Full).model.expect("expands").summary().peak_ops.get("bf16*bf16+fp32").copied();
    assert_eq!(peak(&mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\"], ops: [],")), None);
    assert_eq!(peak(&mutate("precisions: [\"bf16*bf16+fp32\"],", "precisions: [\"bf16*bf16+fp32\"], ops: [\"conv\"],")), None);
    assert_eq!(peak(BASE), Some(2.048e12));
}

#[test]
fn aggregate_memory_capacity_beyond_u64_is_rejected() {
    let src = mutate("capacity: \"256KiB\", banks: 4,", "capacity: 9223372036854775808, banks: 4,");
    let r = check_str(&MemLoader::default(), None, &src, Profile::Full);
    assert!(r.diagnostics.iter().any(|d| d.code == "E-IR-0109"), "{:?}", r.diagnostics);
    if let Some(m) = r.model {
        assert_eq!(m.summary().onchip_capacity.0, u64::MAX);
    }
}

fn with_net(net: &str) -> String {
    mutate("networks: [", &format!("networks: [ {net},"))
}

#[test]
fn crossbar_radix_counts_every_endpoint_port() {
    let xb = |radix: u32| {
        with_net(&format!(
            "{{ id: \"xb\", topology: {{ type: \"crossbar\" }}, router: {{ radix: {radix} }}, endpoints: [ {{ select: \"tile[0].sram\" }}, {{ select: \"tile[1].sram\", multiplicity: 16 }} ], link: \"256b\" }}"
        ))
    };
    expect(&xb(2), "E-IR-0704");
    expect(&xb(16), "E-IR-0704");
    assert!(!codes(&xb(17)).contains(&"E-IR-0704".into()), "{:?}", codes(&xb(17)));
}

#[test]
fn router_radix_counts_replicated_links() {
    let mesh = |radix: u32, count: u32| {
        mutate("endpoints: [ { select: \"tile*.sram\", at: \"layout\" } ], link: \"256b\" }", &format!("endpoints: [ {{ select: \"tile*.sram\", at: \"layout\" }} ], link: {{ width_bits: 256, count: {count} }}, router: {{ radix: {radix} }} }}"))
    };
    // Corner router: 2 mesh neighbours, its tile's sram and the attached HBM stack, each over `count` links.
    assert!(!codes(&mesh(4, 1)).contains(&"E-IR-0704".into()), "{:?}", codes(&mesh(4, 1)));
    expect(&mesh(4, 2), "E-IR-0704");
    expect(&mesh(7, 2), "E-IR-0704");
    assert!(!codes(&mesh(8, 2)).contains(&"E-IR-0704".into()), "{:?}", codes(&mesh(8, 2)));
}

#[test]
fn synthesized_routers_are_charged_to_the_expansion_budget() {
    use kiln_ir::hw::{Design, ExpandOptions};
    let src = |dims: &str| mutate("dims: [2, 2] }", &format!("dims: {dims} }}")).replace("at: \"layout\" }", "}");
    let expand = |dims: &str, max: u64| Design::from_source(&MemLoader::default(), None, &src(dims)).unwrap().expand(&ExpandOptions { max_instances: max });
    let err = expand("[100, 100]", 1000).expect_err("10,000 routers exceed a 1000-instance budget");
    assert!(err.iter().any(|d| d.code == "E-IR-0210"), "{err:?}");
    let err = expand("[4294967295, 4294967295, 4294967295]", 1000).expect_err("dimension product overflows");
    assert!(err.iter().any(|d| d.code == "E-IR-0210"), "{err:?}");
    assert!(expand("[2, 2]", 1000).is_ok());
}
