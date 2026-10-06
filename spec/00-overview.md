# kiln: accelerator system simulator (spec v0)

Working name `kiln`. Rust from day 1. This file is the shared contract every other spec section must follow.

## Purpose

Evaluate structurally open AI-accelerator designs fast enough for LLM-driven evolutionary search, and accurately
enough that a design beating A100/TPU-class silicon in kiln would beat it on silicon at the same physical envelope.

## Hard requirements (from the user)

1. Multi-chip systems (chips, packages, inter-chip links, collectives, parallelism strategies).
2. Near-memory / in-memory compute as a first-class unit type.
3. No structural caps: any number of memories, memory levels, compute tiles, NoC nodes, chips.
4. Precisions: bf16, fp16, fp8 (e4m3/e5m2), MX formats (MXFP8/6/4, MXINT8), int8/int4, fp32 accumulate; per-tensor.
5. Physical realism: floorplan placement, wire distance drives link latency/bandwidth/energy; area, power,
   power density, HBM shoreline, process node.
6. A native visualizer built into the tool.
7. Rust core. Python only at the edges (workload import from PyTorch, evolution-loop bindings).
8. Full spec on day 1: every section below is specified now, even if implemented in phases.

## Non-goals (v0)

RTL generation; cycle-exact microarchitecture of individual PEs; training (out of scope by owner ruling, 08 §C;
the workload IR stays extensible to it); OS/runtime/driver effects beyond a per-launch overhead term.

## Spec sections and owners

| File | Section |
|---|---|
| 00-overview.md | This contract (main session) |
| 01-hardware-ir.md | Hardware description: components, physical attributes, design language, validation |
| 02-workload-ir.md | Workload description: operator graph, LLM ops, precision, parallelism, frontends |
| 03-mapping-engine.md | Mapper, intra-core cost model, analytical tier, event-driven tier, multi-chip, near-memory execution |
| 04-physical-model.md | Floorplan/placer, wire model, area/power/thermal, process nodes, calibration constants |
| 05-visualizer.md | Native visualizer: architecture, views, trace formats, interaction |
| 06-validation-api.md | Validation and calibration, test strategy, evolution-loop API, CLI, performance targets |
| 07-prior-art.md | What exists, what we reuse, what we deliberately do not |

## Crate layout (Cargo workspace `kiln/`)

| Crate | Role |
|---|---|
| `kiln-ir` | Hardware + workload IR types, serde (JSON canonical, also RON/YAML accepted), validation |
| `kiln-phys` | Floorplan, placer, wire model, area/power/thermal, technology node tables |
| `kiln-cost` | Intra-tile cost model (loop-nest mapping + energy/latency per unit), ZigZag-equivalent |
| `kiln-map` | Mapper: partitioning across tiles/chips, tiling into memory hierarchy, scheduling, search |
| `kiln-sim` | Engines: analytical tier (fast) and event-driven tier (contention, timelines) |
| `kiln-trace` | Result + trace data model shared by sim and viz; Perfetto export |
| `kiln-wl` | Workload graph passes: lowering to affine kernels, `partition` (inserts collectives from a `ParallelPlan`), model zoo (02) |
| `kiln-viz-render` | Headless view rendering to PNG/SVG; no winit/wgpu, so `kiln-py` can depend on it (05) |
| `kiln-viz` | Native visualizer (egui/wgpu, also built to WASM) |
| `kiln-py` | PyO3 bindings: evaluate(design, workload) for the evolution loop; workload import glue |
| `kiln-cli` | `kiln eval`, `kiln compare`, `kiln viz`, `kiln validate`, `kiln calibrate` |

Sections may propose splitting/merging crates but must say why.

## Shared conventions (all sections must use these)

- Units in IR and APIs: bytes (B), seconds (s), hertz (Hz), joules (J), watts (W). Floorplan lengths in micrometres
  (um), areas in mm^2. Bandwidth in bytes/s. Never bits except where a field name says `_bits`.
  Display layers may convert (GB/s, TFLOPS, us, pJ).
- Every hardware and workload entity has a stable string id (`[a-z0-9_.-]+`, unique within its scope) and a
  typed index assigned at load. Hierarchy uses dotted paths: `chip0.tile3.sram`.
- Arrays of identical components are first-class (`count` + layout rule) and expand to instances at load; IR stays
  compact for LLM authoring, the engine works on expanded instances.
- Determinism: same inputs give bit-identical outputs. No HashMap iteration order may affect results
  (use IndexMap/BTreeMap or sorted ids). Seeded RNG only.
- Every result carries provenance: kiln version + git hash, design hash, workload hash, calibration-set hash, tier.
- Every simulated result is checked against physical floors (compute, memory bandwidth per level, link bandwidth,
  energy conservation); violations are errors, not warnings.
- Errors are structured (code + message + entity path + hint), written to be read and acted on by an LLM.
- Calibration constants live in versioned data files (not code), each with a source citation or "assumed".

## Cross-section decisions (main session, binding)

1. Power cap / DVFS is solved per steady-state phase (03 §4.5), not per op. 01 and 04 follow 03.
2. Canonical form for hashing is JSON (JCS). Authoring may use JSON5 (hardware), TOML (data tables); both are
   converted to canonical JSON before hashing.
3. `E-IR-UNPRICED` is a schema rule: every performance-bearing field must have a cost computed by 04. It never
   rejects a design for being cost-neutral or cost-reducing (re-placement, topology, memory splits, dataflow).
4. One precision registry in `kiln-ir`, owned by 01, including 02's scale-carrying storage variants.
5. Sharding propagation and collective insertion live in `kiln-wl` (02); 03 chooses the `ParallelPlan`, maps it,
   picks collective algorithms, and may only apply volume-preserving rewrites.
6. 03's `C_eff` calibration parameter is realised as 04's named energy factors (`kappa_E_*`).
7. The evolution archive schema is owned by 05 and consumed by 06. A single `explain_run` (owned by 03, in
   `kiln-sim`) produces the bottleneck text used by 05's views and 06's `Result.explain()`.
8. Tier A may emit the `ops` trace level when requested; default for evolution is `summary`.
9. One hash function everywhere: sha256 over canonical JSON (01 design hash, 02 workload hashes, 06 calibration
   sets, measurements and results).

## Fidelity tiers

- **Tier A (analytical):** target <= 50 ms per (design, LLM layer) on one core; used inside evolution.
- **Tier B (event-driven):** contention on links/ports/banks, full timelines; target <= 10 s per layer; used for
  validation, audit of elites, and visualization.
- Tier A must be within a stated tolerance of Tier B on a regression corpus; disagreement beyond it is a bug.

## Ground truth available now

- Measured: NVIDIA A100-SXM4-40GB (Colab), per-op Llama-3-8B suite + GEMM sweeps (`calibration/measurements/`).
  TPU v5e / v6e measurements in progress. H100 not available on Colab.
- Reference oracle: our fixed Stream fork (`third_party/stream`, branch `fix/sim-correctness`) on the subset it can
  express; prior harness designs in `harness/designs/` (A100, TPU v4/v5e/v6e).
