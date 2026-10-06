#![allow(dead_code)]

use std::path::PathBuf;

use kiln_ir::common::Id;
use kiln_ir::hw::{Design, MemLoader, Profile};
use kiln_ir::wl::{Model, PhaseKind, Scenario, SeqBatch};
use kiln_map::Program;
use kiln_sim::{PhaseRun, Prepared, SimOptions, simulate};
use kiln_trace::IntervalMethod;
use kiln_trace::sim::Scope;
use serde_json::json;

pub fn reference(name: &str) -> Prepared {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../designs/reference").join(name);
    Prepared::from_file(&p, Profile::Reference).expect("reference design loads")
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub gr: u32,
    pub gc: u32,
    pub rows: u32,
    pub cols: u32,
    pub lanes: u32,
    pub sram_kib: u32,
    pub pin_gbps: u32,
}

#[derive(Clone, Copy)]
pub struct Names {
    pub tile: &'static str,
    pub sram: &'static str,
    pub mxu: &'static str,
    pub noc: &'static str,
}

pub const A: Names = Names { tile: "tile", sram: "sram", mxu: "mxu", noc: "noc" };
pub const B: Names = Names { tile: "blk", sram: "buf", mxu: "mac", noc: "fabric" };

/// Extra hardware for P6-style checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extra {
    None,
    /// An SRAM no unit and no network reaches.
    DeadMemory,
    /// An int8-only matrix unit: no bf16 op may use it.
    Int8Unit,
}

/// A small tiled mesh accelerator with per-tile SRAM and one HBM stack (cf. kiln-ir `hw_prop`).
pub fn mesh(p: Params, n: Names, extra: Extra) -> Prepared {
    let mut units = vec![
        json!({ "id": n.mxu, "kind": "matrix", "geometry": { "systolic": { "rows": p.rows, "cols": p.cols } },
                "precisions": ["bf16*bf16+fp32"],
                "local": [ { "id": "w", "holds": "b", "capacity": p.rows * p.cols * 2 } ],
                "feeds": { "a": n.sram, "b": n.sram, "o": n.sram } }),
        json!({ "id": "vpu", "kind": "vector", "lanes": p.lanes, "precisions": ["fp32@1"], "feeds": { "any": n.sram } }),
    ];
    match extra {
        Extra::None => {}
        Extra::DeadMemory => {}
        Extra::Int8Unit => units.push(json!({ "id": "i8", "kind": "matrix", "geometry": { "systolic": { "rows": 64, "cols": 64 } },
                                               "precisions": ["int8*int8+int32"], "local": [ { "id": "w", "holds": "b", "capacity": 8192 } ],
                                               "feeds": { "a": n.sram, "b": n.sram, "o": n.sram } })),
    }
    let mut memories = vec![json!({ "id": n.sram, "kind": "scratchpad", "capacity": p.sram_kib * 1024, "banks": 4,
                                     "ports": [ { "dir": "rw", "width_bits": 256 } ] })];
    if extra == Extra::DeadMemory {
        memories.push(json!({ "id": "orphan", "kind": "scratchpad", "capacity": 4096, "ports": [ { "dir": "rw", "width_bits": 64 } ] }));
    }
    let doc = json!({
        "schema": "kiln.hw/1.0", "name": "prop", "tech": "tsmc_n5",
        "clocks": [ { "id": "clk", "freq": 1.25e9 } ],
        "system": { "package": { "id": "chip",
            "dies": [ { "id": "die", "default_clock": "clk",
                "clusters": [ { "id": n.tile, "layout": { "grid": [p.gr, p.gc] },
                    "units": units,
                    "memories": memories } ],
                "networks": [ { "id": n.noc, "topology": { "type": "mesh", "dims": [p.gr, p.gc] },
                                "endpoints": [ { "select": format!("{}*.{}", n.tile, n.sram), "at": "layout" } ], "link": "128b" } ] } ],
            "mem_stacks": [ { "id": "hbm", "kind": "hbm3", "capacity": "16GiB", "io_width_bits": 1024,
                              "pin_rate_bits_per_s": format!("{}Gbps", p.pin_gbps), "attach": format!("die.{}", n.noc) } ] } }
    });
    let d = Design::from_source(&MemLoader::default(), None, &serde_json::to_string_pretty(&doc).unwrap()).expect("design parses");
    Prepared::load(d, Profile::Full).expect("design validates")
}

/// A 4-layer Llama-shaped model small enough for property tests; `decode` uses a 256-token KV cache, prefill
/// a 64-token prompt.
pub fn tiny(decode: bool, batch: u64) -> (Model, Scenario) {
    let mut cfg = kiln_wl::zoo::preset("llama3_8b").unwrap();
    cfg.d_model = 256;
    cfg.n_layers = 4;
    cfg.vocab = 1024;
    cfg.attn = kiln_wl::zoo::AttnConfig::Gqa { heads: 4, kv_heads: 2, head_dim: 64 };
    cfg.mlp = kiln_wl::zoo::MlpConfig::Dense { d_ff: 512, act: kiln_ir::wl::MapFn::Silu };
    let model = kiln_wl::zoo::build_model(&cfg).unwrap();
    let (kind, q, kv) = if decode { (PhaseKind::Decode, 1, 256) } else { (PhaseKind::Prefill, 64, 64) };
    (model, kiln_wl::zoo::whole_step(kind, SeqBatch::uniform(batch, q, kv)))
}

/// Corners, no shadow prices; phases that do not fit run their layer-scope diagnostic instead of failing.
pub fn quick() -> SimOptions {
    SimOptions { interval: IntervalMethod::Corners, shadow_prices: false, layer_scope_fallback: true, clock: kiln_phys::ClockMode::Nominal, ..SimOptions::default() }
}

pub fn program(model: &Model, sc: &Scenario, w: u32) -> Program {
    let (_, lg, _) = kiln_wl::evaluate_snapshot(model, sc).expect("lowers");
    Program::whole_step(model, &lg, w).expect("program")
}

pub fn run(p: &Prepared, prog: &Program, opts: &SimOptions) -> PhaseRun {
    let prov = kiln_sim::evaluate::base_provenance(&p.design.hash, "wl-test", opts);
    simulate(&p.view, prog, Id::new("test").unwrap(), Scope::Step, opts, prov).expect("simulates")
}
