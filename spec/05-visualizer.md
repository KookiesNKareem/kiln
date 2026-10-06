# 05: Visualizer and trace format (`kiln-trace`, `kiln-viz`, `kiln-viz-render`)

Status: spec v0. Follows `00-overview.md`. Covers the result/trace data model every engine writes, its file format and
exports, the native visualizer, headless rendering for reports and the evolution loop, CLI entry points, and testing.

Crate versions below were checked on crates.io / GitHub on 2026-10-04 and are the versions to pin at project start.

---

## 1. Goals and non-goals

Goals

1. One data model (`kiln-trace`) written by both fidelity tiers and read by the visualizer, the CLI, `kiln-py` and
   exporters. Tier A writes a summary subset; Tier B writes full timelines. The visualizer degrades gracefully on Tier A.
2. A native, single-binary visualizer (`kiln viz`) that opens a 1 GB trace in < 2 s and renders 10k floorplan blocks
   at 60 fps.
3. The same views render headlessly to PNG and SVG, deterministically, so the evolution loop can attach images to LLM
   prompts and reports can embed figures.
4. Shareability: send a single `.kiln` file, or a link to a static web build of the same viewer, or a Perfetto trace.
5. Linked views: one selection model and one time cursor shared by all views.

Non-goals (v0)

- Editing designs in the visualizer (the IR is authored as text by humans and LLMs; viz is read-only). A "copy IR
  path" action exists so a human can jump to the entity in the design file.
- 3D rendering of stacked dies. Stacks are shown as switchable layers (section 6.1).
- Replacing Perfetto for generic trace analysis. We export to it; we do not reimplement its SQL engine.

---

## 2. Architecture decision

### 2.1 Options evaluated

| Criterion | (a) egui/eframe + wgpu, native, same code compiled to WASM | (b) axum server + TypeScript/WebGL frontend | (c) Tauri 2 (Rust core + webview UI) |
|---|---|---|---|
| "Native" feel | Native window, GPU-drawn, instant startup. Widgets are egui-styled, not OS-styled (acceptable for a tool; Rerun, the main existence proof, ships this way). | Runs in a browser tab. Not native. | Native window chrome, but UI is a webview: WKWebView (macOS), WebView2 (Windows), WebKitGTK (Linux), each with different WebGL/WebGPU performance and bugs. |
| Single binary | Yes. `kiln` contains the viewer; no runtime deps beyond a GPU driver (falls back to GL via wgpu). | Binary + bundled static assets (can be embedded with `include_bytes!`), but needs a browser. | One app bundle per OS; still a binary, but heavier build (Node toolchain + Rust) and OS webview dependency on Linux. |
| Large traces (millions of events, 1000s of tiles) | Best. Trace is mmapped in-process; renderer reads Arrow buffers directly, no serialization boundary. Custom wgpu pipelines via `egui_wgpu` paint callbacks for instanced rects. Rerun demonstrates egui + wgpu + Arrow at tens of millions of rows. | Data must cross HTTP/WebSocket and be decoded in JS; browser tab memory limits; requires a second implementation of LOD/indexing logic in TS or a WASM module anyway. | Same IPC/serialization boundary as (b) between Rust core and webview, plus webview variance. |
| Shareability | Send the `.kiln` file; or a URL to the static WASM build of the same viewer (opens files by drag-drop or `?url=`); plus Perfetto export. | Best for links: any browser, nothing to install. | Worst: collaborator must install the app. |
| Development speed | One language, one codebase, immediate-mode UI is fast to iterate. Weaker: fewer off-the-shelf charting widgets (we write our own plots on a small scene layer; `egui_plot` 0.37.0 is usable for simple charts). | Richest ecosystem (d3, deck.gl, Plotly), but two languages, two build systems, a typed API boundary to maintain. | Two stacks like (b) plus Tauri packaging. |
| Embedding in evolution workflow | Views are Rust functions over `kiln-trace`; the same code renders headless PNG/SVG from `kiln-py` with no browser and no GPU. | Headless images need a headless browser (Puppeteer/Playwright) in the evolution loop: heavy, slow, flaky. | Same problem as (b). |

### 2.2 Decision

**Option (a): pure Rust, `eframe` 0.36.2 / `egui` 0.36.2 on `wgpu` 30.0.1, with a WASM build of the same crate.**
(Implemented on `eframe`/`egui` 0.35.0, the newest release supporting the workspace's rustc 1.92: 0.36 requires
rustc 1.95. Move to 0.36 with the next toolchain bump; see section 12.)

Justification, in order of weight:

1. Headless rendering for the evolution loop is a hard requirement of the workflow (images attached to LLM prompts,
   thousands per run). Only (a) lets the exact view code produce PNG/SVG without a browser or GPU.
2. Large traces: zero-copy mmap of Arrow buffers into the renderer. No serialization boundary.
3. The user asked for a *native* visualizer and the overview mandates a Rust core; (a) is the only option with no
   second language.
4. Shareability, the main weakness of native apps, is recovered by (i) the single-file `.kiln` container, (ii) the
   WASM build of the same viewer hosted as a static site, (iii) `kiln viz web` which serves that WASM build from the
   binary itself (useful on headless GPU pods over an SSH tunnel), and (iv) Perfetto export.

Ecosystem pieces adopted (versions as of 2026-10-04):

| Need | Crate | Version | Note |
|---|---|---|---|
| App shell | `eframe` | 0.36.2 | native (winit) and web (wasm-bindgen) backends |
| Immediate-mode UI | `egui` | 0.36.2 | |
| GPU | `wgpu` / `egui-wgpu` | 30.0.1 / 0.36.2 | Metal, Vulkan, DX12, GL fallback; WebGPU and WebGL2 on web |
| Dockable panes | `egui_tiles` | 0.17.1 | same tiling model Rerun uses; `egui_dock` 0.21.1 is the alternative |
| Simple charts | `egui_plot` | 0.37.0 | only for inspector sparklines; main plots go through our scene layer (2.3) |
| Columnar data | `arrow` / `arrow-ipc` | 60.0.0 | IPC file format, mmap |
| Parquet export | `parquet` | 60.0.0 | export only |
| mmap | `memmap2` | 0.9.11 | |
| Compression | `lz4_flex` 0.14.0, `zstd` 0.14.0 | | Arrow IPC body compression and pack compression |
| Perfetto protobuf | `prost` | 0.14.4 | vendored Perfetto `.proto` files from Perfetto v58.2; the `perfetto-protos` crate (0.51.1, last updated 2025-07) is stale and not used |
| Headless raster | `tiny-skia` | 0.12.0 | CPU rasterizer, deterministic |
| SVG write / test parse | own writer; `usvg`/`resvg` 0.48.1 for tests | | |
| UI tests + snapshots | `egui_kittest` 0.36.2, `insta` 1.49.0 | | |
| Image diff | `dify` 0.8.0 (or own) | | |
| File watch | `notify` | 8.2.0 | live archive view |
| WASM build | `trunk` 0.21.14, `wasm-bindgen` 0.2.129 | | |

Rejected but noted: `vello` 0.11.0 (GPU 2D vector renderer). Attractive for anti-aliased vector plots, but egui's own
tessellator plus instanced rect pipelines cover our needs, and vello would add a second GPU renderer. Revisit if
plot quality becomes a complaint.

### 2.3 Internal layering

```
            engines (kiln-sim tiers A/B), kiln-phys, kiln-map
                              | TraceWriter
                              v
  kiln-trace    : schema, writer, mmap reader, indices, LOD pyramids, derived analysis, exporters
                  (no GUI deps; used by kiln-sim, kiln-py, kiln-cli, kiln-viz-render, kiln-viz)
                              |
                              v
  kiln-viz-render: view models -> Scene (retained 2D primitive list) -> {tiny-skia PNG, SVG}
                  (no winit/wgpu deps; used by kiln-py for headless images and by kiln-viz)
                              |
                              v
  kiln-viz      : eframe app: panes, interaction, selection, Scene -> egui shapes + wgpu instanced pipelines
```

**Crate split proposal.** The overview lists `kiln-trace` and `kiln-viz`. This section adds `kiln-viz-render`.
Reason: `kiln-py` (shipped into evolution workers, often headless Linux boxes) must render PNGs without pulling in
winit, wgpu and platform windowing; Cargo features on one crate would work but make it easy to accidentally leak GUI
deps into `kiln-py`. A separate crate makes the dependency boundary a compile error.

**Scene.** Every *canvas* view (floorplan, NoC heatmap, timeline, roofline, bottleneck charts, diff, evolution grid,
lineage, calibration) is a pure function `ViewModel x ViewState -> Scene`. A `Scene` is an ordered list of primitives
in a view coordinate space: rect (optionally instanced), rounded rect, polyline, polygon, circle, text (font id, size,
anchor), image (raster tile for thermal maps), clip push/pop, plus a hit-test id on each primitive. Interactive mode
converts Scene to egui shapes, except bulk rect batches (> 2k rects) which go to a dedicated wgpu instanced pipeline.
Headless mode rasterizes the same Scene with tiny-skia or writes it as SVG. Panels (tables, inspectors, menus) are
plain egui and are not part of headless output.

---

## 3. Data model (`kiln-trace`)

### 3.1 Concepts

- **Run**: one evaluation of (design, workload, mapping, calibration set, tier). Result of `kiln eval`. One `.kiln` file.
- **Entity**: anything with a stable id from the IRs: hardware instances (chips, dies, tiles, units, memories, NoC
  routers, links, PHYs), workload ops, collectives. Each gets a dense `u32` index in this trace (assigned in sorted id
  order for determinism) and keeps its dotted path string.
- **Span**: a time interval on a resource lane doing something attributable to an op.
- **Transfer**: bytes moved from a source endpoint to a destination endpoint over a route of links.
- **Counter**: a piecewise-constant time series on a resource (utilization, bandwidth, occupancy, power, temperature).
- **Aggregate**: per-op, per-resource and per-(op, resource) totals (time, energy, bytes, flops).
- **Limiter record**: per op, the cost terms the engine evaluated and which bound it (drives "what limits this op").
- **SimResult** (defined in 03 section 10, owned here): the in-memory result struct. A `.kiln` file is SimResult
  persisted (headline fields in the manifest, vectors as tables) plus, for Tier B, the `Trace` of 03 section 5.5.
  Field names below follow 03 where 03 names them.
- **Archive** (separate file family): the MAP-Elites archive written by the evolution loop (section 3.9).

### 3.2 Time representation

The overview fixes seconds as the API unit. On disk, times are stored as `i64` ticks with `tick_s` declared in the
manifest (default `1e-12`, i.e. 1 ps; range +-106 days). Reason: integer times are exact, sort and delta-encode well,
and make determinism checks bit-exact; f64 seconds lose ns resolution late in long multi-step runs. All public
accessors (`Span::start_s()`, Python bindings, exports) return seconds as f64. Renderers compute positions as
`(t - view_origin)` in i64 before converting to f64, so zooming to ns at t = 10 s stays precise.

### 3.3 Trace levels

| Level | Written by | Contents | Typical size (one Llama-3-8B layer, 1 chip) |
|---|---|---|---|
| `summary` | Tier A default, evolution loop | manifest, entities, ops, aggregates, limiters, floorplan, roofline inputs | 10 KB to 2 MB |
| `ops` | Tier A with `--trace ops` | + analytical schedule spans (one per op per resource class, flagged `estimated`) | + few MB |
| `full` | Tier B default | + all spans, transfers, counters, thermal frames, critical path | 10 MB to 10s of GB |

06's `options.trace` and `kiln eval --trace` expose `none | summary | ops | full`; `none` writes no file (in-memory
SimResult only). Tier A emits `ops` only when requested; the evolution default is `summary` (00 decision 8).
The visualizer opens any level; views whose data is missing show a banner ("timeline requires `ops` or `full`;
re-run with `kiln eval --tier b`") instead of an empty pane.

### 3.4 Tables

All tables are Arrow schemas. Index columns are `u32` into the entity tables. Enumerations are `u8`/`u16` dictionary
codes whose string tables live in the manifest (so new enum values never break old readers). Every table's Arrow
schema metadata carries `kiln.table`, `kiln.schema_version`.

**`resources`** (hardware entity instances, expanded from arrays)

| Column | Type | Meaning |
|---|---|---|
| `idx` | u32 | dense index |
| `path` | utf8 (dict) | 01's instance id, e.g. `board0.chip0.die0.gpc3.tpc1.sm0.smsp2.tc` |
| `kind` | u16 enum | mirrors 01 section 4.1: host, board, package (= "chip" in 00's wording), die (chiplets are dies), cluster (recursive; a tile is a cluster), unit_matrix, unit_vector, unit_scalar, unit_special, unit_cim, nmp_unit (a unit with an 01 `near` binding), memory, mem_stack, network, router, channel (expanded link/port pair), switch, port, block |
| `class` | u8 enum | compute, memory, interconnect, io, container |
| `parent` | u32 (nullable) | container hierarchy |
| `chip` | u32 | owning chip index (for grouping) |
| `array_id`, `array_pos` | u32, u32 (nullable) | array origin from IR `count` expansion; lets views collapse arrays |
| `mem_level` | u8 (nullable) | memory level index for memories (0 = closest to compute) |
| `capacity_b`, `peak_bw_bps` | f64 (nullable) | for memories / links |
| `peak_flops` | map<precision, f64> | per precision, compute units only |
| `lanes` | u16 | number of non-overlapping span lanes on this resource |

**`ops`** (workload op instances after mapping; one row per op instance, e.g. per layer, per micro-batch)

| Column | Type | Meaning |
|---|---|---|
| `idx` | u32 | |
| `path` | utf8 | stable op id from 02 (dotted node path), e.g. `layers.block.qkv`; repeat iteration in `layer` |
| `kind` | u16 enum | 02's `Op` tag (einsum, attention, rms_norm, map, collective, ...) |
| `phase` | u8 enum | 02's `PhaseKind` (prefill, decode, mixed, custom) |
| `layer` | i32 (nullable) | |
| `flops` | f64 | algorithmic FLOPs (02's definition) |
| `precision` | u8 enum | compute precision; operand precisions in `op_tensors` |
| `bytes_by_level` | list<f64> | bytes moved at each memory level boundary (index = `mem_level`) |
| `link_bytes` | f64 | bytes over inter-chip links |
| `t_start`, `t_end` | i64 ticks | op envelope (first span start, last span end) |
| `chips` | list<u32> | chips it runs on |
| `mapping` | u32 | row in `mapping` table |
| `macs_useful`, `macs_issued` | u64, u64 | padding/quantization waste = issued - useful (03) |
| `target` | u8 enum | host units or NMP sites; `host_vs_nmp_s` (f64, f64) nullable when both were costed |
| `group` | u32 | execution group (fusion / pipelining) |

**`spans`** (Tier B, and Tier A `ops` level). Sorted by `(resource, lane, t_start)`; 48 bytes per row uncompressed. 03 emits `start_ps, end_ps`; the writer
converts to `t_start, dur` in ticks (identical at the default 1 ps tick).

| Column | Type |
|---|---|
| `resource` | u32 |
| `lane` | u16 |
| `kind` | u8 enum: compute, load, store, transfer_send, transfer_recv, collective_step, nmp_compute, nmp_command, mode_switch, stall_contention, stall_dependency, launch_overhead, idle_powergated, estimated |
| `flags` | u8 bitset: on_critical_path, estimated (Tier A), clipped |
| `op` | u32 (nullable = 0xFFFFFFFF) |
| `task` | u32 (03's lowered task index; joins to critical path) |
| `slice` | u32 (op slice / chunk index from 03; 0 if unsliced) |
| `t_start`, `dur` | i64, i64 |
| `bytes` | f64 (0 for compute) |
| `energy_j` | f32 |

**`transfers`**: `id u64`, `op u32`, `src u32`, `dst u32`, `route u32` (row in `routes`), `bytes f64`, `t_start i64`,
`dur i64`, `collective u32 nullable`, `step u16 nullable`. Each transfer emits `transfer_send`/`transfer_recv` spans on
endpoints and one span per traversed link; `id` doubles as the Perfetto flow id.

**`routes`**: `idx u32`, `links list<u32>` (ordered link resources), `wire_len_um f64`. Deduplicated.

**`counters`**: one Arrow table per metric family, columns `resource u32`, `t i64`, `value f64`, sorted by
`(resource, t)`, piecewise-constant from `t` until the next sample. Metric families (enum, extensible): `util`
(0..1), `bw_read_bps`, `bw_write_bps`, `occupancy_b`, `queue_depth`, `credits`, `power_w` (per resource or per power
domain), `clock_hz` (per clock domain), `temp_k`. Tier B samples at event boundaries (step functions, 03 section 5.5);
Tier A writes one sample per phase.

**`thermal_frames`**: per die, a raster grid: `die u32`, `t i64`, `nx u16`, `ny u16`, `origin_um (f64,f64)`,
`cell_um f64`, `temp_k list<f32>`. Steady-state only = one frame at `t = 0` with a flag (from 04).

**`aggregates_resource`**: `resource`, `busy_s`, `stall_s`, `bytes`, `flops`, `energy_dyn_j`, `energy_leak_j`,
`avg_power_w`, `peak_power_w`, `power_density_w_mm2`, `peak_temp_k`.

**`aggregates_op_resource`**: `op`, `resource`, `time_s`, `energy_j` broken into `energy_component` enum
(compute, sram_read, sram_write, dram, noc, link, leakage, other).

**`limiters`**: per op (or per execution group in Tier A), one row per evaluated term: `op u32`, `group u32`,
`binding u8 enum` mirroring 03's `Binding` (Compute, Link, MemPort, Dram, Nmp, Dependency, Overhead(kind),
Contention, PipelineBubble), `resource u32 nullable`, `time_s f64` (the term's own time if it were the only
constraint, i.e. the per-resource busy time `B_r` of 03 section 4.2), `attained_frac f64` (achieved / ceiling),
`rank u8` (0 = binding, 1 = runner-up, ...), `share f64` (fraction of the op's attributed time; sums to 1 per op).
Tier A: shares follow 03's attribution rule (argmax term of A2 gets the group time; runner-up and its ratio
recorded). Tier B: shares come from the critical-path decomposition. Plus run-level `bottleneck` table mirroring
03's `Bottleneck`: `time_by_binding` (sums exactly to makespan), `top_resources` (utilization, shadow price),
`slack` per near-binding resource.

**`critical_path`**: ordered rows `(task u32, span row u64 nullable, transfer id u64 nullable, reason u8 enum)` with
03's reasons (Compute, Link, Port, Dram, Dependency, Overhead, Queue), covering `[0, makespan]` without gaps (03
invariant I8); plus `slack_s` per op (0 on the path).

**`groups`**: 03's execution groups: `group u32`, `ops list<u32>`, `kind` (fused, pipelined, layer_by_layer),
`t_start`, `t_end`, `bubble_s`, `exposed_overhead_s`. Shown as a track in the timeline.

**`collectives`**: 03's `CollectiveResult`: `collective u32`, `op u32`, `algorithm utf8`, `group_chips list<u32>`,
`steps u16`, `bytes f64`, `t_start`, `t_end`, `link_bytes_by_tier list<f64>`.

**`run_scalars`**: one-row tables for SimResult's `power` summary (avg, peak windowed, cap, throttled), `energy`
breakdown, `invariants` report (each check with measured margin), `cost_model` summary and `calibration`
contribution (per-parameter delta to makespan/energy), plus `intervals` (low / central / high of time, tokens/s,
energy, power, area and score, with `interval_method` and top width drivers; 03 §9.1, 06 §6.3) when the run
evaluated them. Shown in the Run info panel.

**`mapping`**: per op: `op u32`, `tiles list<u32>`, `chips list<u32>`, `parallelism utf8` (e.g. `tp=8,pp=2`),
`loop_nest utf8` (JSON, opaque to viz, pretty-printed in inspector), `tiling_by_level list<utf8>`.

**`floorplan`**: per placed block: `resource u32`, `die u32`, `layer u8` (stack layer, 0 = base), `x_um, y_um, w_um,
h_um f64`, `poly list<(f64,f64)>` (nullable; rectilinear outlines), `rotation u8`; since schema 1.1 also `source u8`
(enum: unplaced, placed, layout, filled, site), `block u8` (enum: misc, compute, sram, noc, phy, hbm, control) and
`area_um2`, `leak_w` (f64, nullable: the subtree's silicon area and leakage). Plus `package_geometry` (`kind u8`
enum package, shoreline, harvested; `resource`, `layer`, `x_um, y_um, w_um, h_um`, `label`, `value`, `value2`:
package outlines with their technology, shoreline per die edge with used and HBM um, harvested dies and stacks) and
`wires` (`link u32`, `src u32`, `dst u32`, `kind u8`, `source u8`, `layer u8`, `polyline` as a flat
`x0, y0, x1, y1, ...` list, `length_um`, `width_bits`, `bw_bps`, `latency_s`, `e_j_per_bit`, `class u8`,
`pipeline_stages u32`). Produced from 04's placer and link derivation.

**`diagnostics`**: structured errors/warnings carried through from engines and physical checks (code, message,
entity path, hint, as in the overview). Viz shows them as an issue list with click-to-entity.

### 3.5 Manifest (JSON header)

Fields: `format: "kiln-trace"`, `schema_version: "MAJOR.MINOR"`, `tick_s`, `level`, `tier`, provenance block
(kiln version, git hash, design hash, workload hash, calibration-set hash, tier, mapping hash, seed, wall-clock run
time, host), embedded or referenced design IR and workload IR (canonical JSON, so a `.kiln` is self-contained and
re-runnable), enum string tables, table directory (name, member offset, length, row count, compression, sha256),
precomputed headline numbers (latency, throughput, energy, area, power as low / central / high when present,
physical floor check results) so `kiln trace
info` and the archive view never touch big tables, and a `thumbnails` list (member names of PNGs, section 7.3).

### 3.6 File format: `.kiln` container

Single file, designed for mmap and for streaming writes:

```
offset 0      : 64-byte fixed header: magic "KILNTRC\0", container version u32, flags, offset+len of manifest
offset 64     : members, each 64-byte aligned:
                  Arrow IPC *file* format tables (one member per table, or per metric family)
                  PNG thumbnails
end           : manifest JSON (written last), then a 64-byte trailer repeating magic + manifest offset/len
```

- **Streaming write.** `TraceWriter` appends Arrow record batches (64k rows) to per-table temp members while the
  engine runs, then on `finish()` it sorts where needed (k-way merge of sorted runs, bounded memory), builds indices
  and LOD pyramids, and writes the manifest + trailer. A crash leaves members whose Arrow IPC streams are recoverable
  with `kiln trace recover`.
- **mmap reads.** Native members are uncompressed so Arrow buffers are used zero-copy via `memmap2`.
- **Compression for sharing.** `kiln trace pack --compress zstd|lz4` rewrites members with Arrow IPC body compression
  (LZ4 frame or zstd, per batch). The viewer decompresses batches lazily on access; the < 2 s open target applies to
  uncompressed files and to compressed files of up to 200 MB.
- **Directory form.** `kiln trace unpack run.kiln run.kiln.d/` writes each member as a plain `.arrow` file plus
  `manifest.json`, readable by pyarrow/polars directly. `kiln trace pack` reverses it.
- **Parquet export.** `kiln trace export --parquet out_dir/` for analysis in pandas/polars/DuckDB. Parquet is not the
  native format because it is not zero-copy mmappable and random access to a time window needs page decoding.
- **Why not a custom binary format:** Arrow gives zero-copy, Python interop for free (`kiln-py` returns pyarrow tables
  without copies), schema evolution by column, and an existing ecosystem; the 64-byte container is only a bundler.

### 3.7 Indices and LOD pyramids (built at `finish()`)

- **Track offset index**: for each `(resource, lane)`, the row range in `spans`. Lookup of visible spans is a binary
  search on `t_start` within that range, then a scan backward bounded by the lane's `max_dur` (stored per lane).
- **Span pyramid**: per `(resource, lane)`, levels k = 0..K where level k buckets time in `base_bucket * 4^k`
  (base_bucket = run_duration / 2^20, rounded to ticks). Each bucket stores busy fraction, dominant span kind, dominant
  op, span count. The timeline draws from the coarsest level whose bucket width <= 1 px. Overhead ~ 1/3 of level 0.
- **Array summary pyramid**: per resource array (`array_id`), the same buckets aggregated over all members (mean,
  min, max busy fraction) so a 4096-tile array renders as one heat strip.
- **Counter pyramids**: min/max/mean per bucket per counter, same level scheme.
- **Link-time matrix**: per link, per level-L bucket utilization (L chosen so the matrix fits 4096 time columns),
  stored as a dense f16 matrix. Drives the NoC heatmap at any time without touching `spans`.
- **Op index**: op -> span row ids (CSR layout), op -> transfer ids.

### 3.8 Derived analysis (`kiln_trace::analysis`)

Pure, deterministic functions over a trace, shared by viz, CLI and `kiln-py`: time breakdown by op kind / phase /
resource class, energy breakdown tree, roofline points per op per memory level, diff alignment of two traces (section 6.6), and consistency checks (`kiln trace validate`): spans within op
envelopes, no overlap within a lane, sum of per-op energy equals total within 1e-9 relative, counters consistent with
spans (util integral equals busy time), bandwidth never above `peak_bw_bps`, physical floors from the overview. A
validate failure is an error, matching the overview's "violations are errors".

### 3.9 Archive and calibration inputs

These are written by 06's evolution and calibration tooling; the schema is owned here and 06 consumes it (00
decision 7).

**Archive directory** (`kiln viz --archive <dir>`):

```
<dir>/archive.json     : descriptor axes (name, unit, bins or edges), fitness name and direction, run config
<dir>/designs.arrow    : design_id, generation, parent_ids list<utf8>, mutation_summary utf8, operator utf8,
                         descriptor values list<f64>, cell list<u32>, fitness f64, fitness components struct,
                         status enum (elite, displaced, invalid, failed), run path utf8, thumbnail path utf8,
                         design hash, wall time; nullable (required by 06 §6.9): trust_level, audit_status,
                         heldout_score, calibration_set_hash, kiln_git_hash, stage_reached, extrapolated list<utf8>,
                         fitness_low f64, fitness_high f64, interval_method utf8
<dir>/generations.arrow: generation, best, median, qd_score, coverage, evaluations, invalid_count
<dir>/runs/<design_id>.kiln  (summary level by default; elites re-run at Tier B on demand)
```

Append-only during a run (new Arrow IPC batches appended; a `.lock`-free reader tolerates a partial last batch).

**Exact schema (implemented in `kiln_trace::archive`, schema `kiln.archive/1`).** `archive.json`:
`{"schema": "kiln.archive/1", "axes": [{"name", "unit", "log"?, "range"?: [lo, hi], "bins"?, "edges"?}],
"fitness": {"name", "direction": "maximize"|"minimize"}, "run_config"?: any}`; other keys (e.g. a driver's
provenance block) are ignored. An axis is binned by `edges` (`bins + 1` values) when given, else uniformly over
`range` in `bins` bins (log-spaced when `log`). The first draft's spellings (`format`, `descriptors`) are accepted.

| `designs.arrow` column | Arrow type | Null |
|---|---|---|
| `design_id` | utf8 | no |
| `generation` | uint32 (any integer accepted) | no |
| `parent_ids` | list<utf8> (empty for seeds; 2 for crossover) | no |
| `mutation_summary`, `operator` | utf8 | no |
| `descriptor_values` | list<f64>, archive.json axis order (`descriptors` accepted) | no |
| `cell` | list<uint32>, bin per axis | no |
| `fitness` | f64 | no |
| `fitness_components` | map<utf8, f64>, or utf8 holding a JSON object (e.g. per-phase ratios) | no (may be empty) |
| `status` | utf8 or dictionary<utf8>: `elite`, `displaced`, `invalid`, `failed` | no |
| `run_path`, `thumbnail_path` | utf8, relative to the archive directory | yes |
| `design_hash` | utf8 | no |
| `wall_time` | f64 seconds (`wall_time_s` accepted) | no |
| `trust_level`, `audit_status`, `calibration_set_hash`, `kiln_git_hash`, `stage_reached`, `interval_method` | utf8 | yes |
| `heldout_score`, `fitness_low`, `fitness_high` | f64 | yes |
| `extrapolated` | list<utf8> | yes |

`generations.arrow`: `generation` uint32, `best`, `median`, `qd_score`, `coverage` f64, `evaluations`,
`invalid_count` uint64, `best_low`, `best_high` f64 nullable.

Every evaluated design gets a row (not only elites); a later row with the same `design_id` supersedes earlier ones
(e.g. an elite displaced later is appended again with `status = displaced`). Writers either append (each append is
one complete Arrow IPC **stream**, schema + batches + end-of-stream marker, written at the end of the file:
`with open(p, "ab") as f, pa.ipc.new_stream(f, schema) as w: w.write_batch(b)`) or rewrite the file atomically
(temp file + rename) as an IPC stream or IPC file. Readers accept all three, read concatenated streams to the end,
and tolerate a truncated last stream. Columns match by name: extra columns are ignored, missing nullable columns
read as null, and any integer width, utf8 / large_utf8 / utf8_view / dictionary strings and list / large_list are
accepted, so pyarrow and polars output needs no casts. `kiln-trace/tests/golden/archive_pyarrow/` holds a
pyarrow-written fixture; `kiln_evo` (06) writes this schema.


**Calibration inputs** (`kiln viz --calibration <dir|file>`): 06 owns the measurement schema (`kiln.meas/1`,
`calibration/measurements/<vendor>/<sku>/<date>_<session>.json`) and calibration sets (`kiln.calib/1`,
`calibration/sets/<set_id>.json`). The viewer consumes a **calibration report** table that `kiln calibrate`
(06) writes next to the set, joined on 06's BenchOp key (sha256 of the canonical op descriptor, so joins never
rely on names): `device utf8`, `bench_key utf8`, `op_kind`, `shape struct`, `precision`, `timing_mode utf8`,
`measured_s f64`, `measured_cv f64` (cross-session noise floor), `n_reps u32`, `predicted_s_tier_a f64`,
`predicted_s_tier_b f64 nullable`, `split enum` (fit, test; 06 §3.4 split names), `evidence_grade utf8` (06's G1/G2...),
`session_hash`, `source utf8`, `calibration_set_hash`.

### 3.10 Versioning

- `schema_version` is `MAJOR.MINOR`. MINOR: new tables, new nullable columns, new enum values (enum strings live in
  the manifest, so old readers display them by name). MAJOR: anything else.
- Readers support the current and previous MAJOR via a migration layer (`kiln trace upgrade` rewrites in place to
  current). Unknown tables and columns are ignored with an info-level diagnostic.
- Container version is independent of schema version and changes only if the 64-byte layout changes.
- Golden traces for every released schema version live in `tests/golden/trace/` and are read in CI forever.

### 3.11 Exports

**Perfetto (native protobuf, primary).** `kiln trace export --perfetto run.kiln -o run.pftrace`. Encoded with `prost`
from vendored Perfetto v58.2 protos (`TracePacket`, `TrackDescriptor`, `TrackEvent`, `InternedData`). Mapping:

| kiln | Perfetto |
|---|---|
| system / chip / die / tile hierarchy | nested `TrackDescriptor`s via `parent_uuid`, `child_ordering = EXPLICIT` with `sibling_order_rank` = resource index (deterministic order) |
| resource lane | track; spans as `TYPE_SLICE_BEGIN/END` |
| span op | slice name = op path (interned), category = op kind, debug annotations: bytes, energy, flags |
| transfer | slices on link tracks connected with `flow_ids` = transfer id |
| counters | counter tracks with `unit_name`, related counters share `y_axis_share_key` (e.g. all HBM channels) |
| critical path | a separate "critical path" track duplicating those slices |

Interning is mandatory (op names repeat millions of times); the exporter streams packets with bounded memory.
Arrays larger than `--max-tracks` (default 512 per chip) are exported as summary counter tracks unless
`--all-tracks` is passed, because Perfetto UI degrades with tens of thousands of tracks.

**Chrome JSON** (`--chrome-json`): for small traces and tools that only speak the legacy format; refuses above 2M
events unless `--force`.

**Opening in Perfetto.** `kiln viz --perfetto run.kiln` exports to a temp file, serves it on `127.0.0.1:9001` with
CORS, and opens `https://ui.perfetto.dev/#!/?url=http://127.0.0.1:9001/<name>`, mirroring Perfetto's own
`open_trace_in_ui` script. Per Perfetto's docs, traces opened this way or via `postMessage` stay in the browser and
are not uploaded. The web build of kiln-viz offers "Open in Perfetto" using `postMessage` with a `ReadableStream`.

**Other**: `--parquet` (3.6), `--csv <table>` for small tables, `kiln trace info --json` (manifest + headline).

### 3.12 Writer API (contract with engines)

`TraceWriter::new(path_or_memory, level, manifest_seed)`, `add_resources`, `add_ops`, `add_floorplan`,
`push_span`, `push_transfer`, `push_counter(metric, resource, t, value)`, `push_thermal_frame`, `set_limiters`,
`set_critical_path`, `add_diagnostic`, `finish() -> TraceHandle`. In-memory mode yields 03's `SimResult` (with `trace: Option<TraceRef>`), which is what
the evolution loop gets from `kiln-py` (no file I/O at `summary` level); `SimResult::save(path)` writes a `.kiln`.
03 section 5.6 allows span decimation for very long traces; decimated traces set `flags.clipped` and a manifest
note, and the timeline shows a "decimated" banner. Engines may push
spans in any order; `finish()` sorts. Writer throughput target: >= 20M spans/s/core into memory, disk-bound
otherwise.

---

## 4. Application structure (`kiln-viz`)

### 4.1 Window layout

```
+----------------------------------------------------------------------------------------------+
| kiln viz  [run: evo_g42_d17.kiln  tier B  design 3f2a..  wl llama3-8b/L0-31]  [Cmd-K search] |
| Views: [1 Floorplan][2 NoC][3 Timeline][4 Roofline][5 Bottleneck][6 Diff][7 Evolution][8 Calib]
+---------------------------------------------------------------+------------------------------+
|                                                               | Inspector                    |
|   main pane(s): any view; panes split/dock (egui_tiles)       |  selection: chip0.tile3      |
|                                                               |  kind tile, 0.41 mm^2        |
|                                                               |  util 0.72  P 1.9 W          |
|                                                               |  ops mapped (14) ...         |
+---------------------------------------------------------------+------------------------------+
| time: |----[=====window=====]-------------------| cursor 1.204 us  [play] 1x  run 3.88 ms  |
| issues: 0 errors, 2 warnings                                   status: 23.1M spans, 60 fps  |
+----------------------------------------------------------------------------------------------+
```

- Global **time bar** at the bottom: full-run overview (busy density sparkline from the system pyramid), a draggable
  window `[t0, t1]` and a cursor `t`. Every time-aware view reads it.
- **Inspector** on the right shows the current selection; contents depend on entity type.
- **Issue list** (diagnostics) collapsible at bottom.
- Layout persisted per user in `~/.config/kiln/viz.ron` (native) or localStorage (web, best effort).

### 4.2 Selection and linking model

- `Selection` = ordered set of `EntityRef` (resource, op, span, transfer, link, design, archive cell) + optional time
  range. Single source of truth in app state; every view highlights members and dims others when "focus" is on (`F`).
- **Hover** is a separate, transient `EntityRef`; all views cross-highlight on hover (e.g. hover an op in the
  roofline, its tiles glow on the floorplan and its spans outline in the timeline).
- **View state** (active view, camera per view, color mode, time window, cursor, selection) serializes to a compact
  string. Native: "Copy view link" copies `kiln://open?file=<path>#<state>`; web: the state lives in the URL fragment
  (`#v=timeline&w=1.2e-6..1.5e-6&sel=chip0.tile3`). `kiln viz run.kiln --state '<string>'` restores it. This is how a
  human says "look here" to a collaborator.

### 4.3 Global interaction and shortcuts

| Key | Action |
|---|---|
| `1`..`8` | switch main pane to view N |
| `Cmd/Ctrl-K` | command palette: fuzzy search entities by path, ops by name, commands |
| `F` | toggle focus (dim non-selected) |
| `Z` | zoom-to-fit selection in current view |
| `Esc` | clear selection |
| `Space` | play/pause time animation |
| `,` / `.` | step cursor by one frame bucket (or by one span on the selected track in timeline) |
| `[` / `]` | previous / next op on critical path |
| `C` | cycle color mode (floorplan, NoC) |
| `L` | lock color scale (keeps min/max fixed across time/compare) |
| `Cmd/Ctrl-S` | save screenshot of current canvas (PNG via the headless path, identical pixels) |
| `Cmd/Ctrl-Shift-C` | copy view link |
| `?` | shortcut overlay |
| Timeline only: `W`/`S` zoom in/out, `A`/`D` pan, `M` mark range, `G` go to time (Perfetto conventions) |
| Floorplan only: `U` go up one hierarchy level, `Enter` drill into selected container, `T` toggle wires, `H` heat overlay |

Mouse: wheel zoom about pointer, drag pan (middle or space-drag), click select, shift-click add, box-drag (with
`Shift`) multi-select / time-range select, double-click drill in, right-click context menu (copy path, open in
timeline, show ops mapped, compare with...).

### 4.4 Color and units

- Sequential: cividis (default, colorblind-safe) and viridis; diverging (blue-white-red, centered at 0) for diffs;
  fixed 12-color categorical palette for unit kinds and op kinds, consistent across views and headless images.
- Idle / no data is never encoded by color alone: idle uses a diagonal hatch, missing data a dotted outline.
- Every colored view draws its legend with units; legends are part of the Scene, so they appear in PNG/SVG.
- Display units auto-scale (ps/ns/us/ms, B/KB/MB/GB with 1000 base, GB/s, TFLOP/s, pJ/nJ/mJ, W/mm^2, degC from K).
  Tooltips show the exact value in SI base units on hover with `Alt` held.

---

## 5. Performance budget

| Scenario | Target | Measured on |
|---|---|---|
| Open 1 GB uncompressed `.kiln` to first interactive frame | < 2 s (manifest parse + mmap + small tables + summary pyramids only; no full-table scan) | M-series laptop SSD and Linux pod NVMe |
| Floorplan, 10k blocks, pan/zoom with heat coloring and wires | 60 fps (frame <= 16.6 ms; CPU <= 6 ms, GPU <= 8 ms) | Apple M1 integrated GPU |
| Floorplan, 100k blocks | >= 30 fps | same |
| Timeline, 25M spans, 5k tracks (arrays collapsed), any zoom from full run to 10 ns | 60 fps; <= 200k primitives submitted per frame | same |
| Time-scrub NoC heatmap, 50k links | 60 fps while dragging the cursor (reads link-time matrix only) | same |
| Archive view, 100k designs | 60 fps grid; lineage tree layout <= 500 ms (cached) | same |
| Viewer RSS beyond mmapped pages | <= 400 MB for a 1 GB trace | |
| Headless PNG, 1600x1000 floorplan of 10k blocks | <= 150 ms on one core (tiny-skia) | evolution worker |
| Headless render memory | <= 100 MB | |
| Compare two 1 GB traces | < 4 s open | |

Techniques (normative): mmap + lazy decode; pyramids chosen by pixel width; instanced rect pipeline for > 2k rects;
text labels culled by on-screen size (no label below 40 px block width) and capped at 2k glyph runs per frame; view
models cached and rebuilt only when inputs (camera bucket, time window, color mode, selection) change; background
thread pool (`std::thread` + channels; `rayon` on native, single-thread fallback on wasm) for aggregations over
selections; no allocation per primitive in the hot path (reuse Scene buffers).

Web build limits: wasm32 memory caps the in-memory working set at ~4 GB; web reads files via `File.slice` / HTTP
range requests instead of mmap. Target: web opens a 1 GB trace in < 5 s; larger traces print "use native `kiln viz`
or `kiln viz web` on the machine holding the file".

---

## 6. Views

Each view lists: purpose (questions it answers), data used, layout (ASCII), interactions, headless output.

### 6.1 Floorplan

**Answers:** Where is the silicon area going? Which blocks are hot (power density, temperature) or idle? Are links
long because of placement? Is the HBM shoreline used? Which ops ran on this tile? Does a chiplet split make sense?

**Data:** `floorplan`, `package_geometry`, `wires`, `aggregates_resource`, counter pyramids (for time-window modes),
`thermal_frames`, `mapping`, `diagnostics`.

**Hierarchy and zoom.** One continuous zoomable canvas in micrometres with semantic zoom following 01's hierarchy
(section 4.1), which has arbitrary depth:

1. **System**: hosts, boards, switches and packages drawn as boxes (not to physical scale across boards; to scale
   within a package), board and inter-package networks as arcs with width proportional to capacity and color to
   utilization.
2. **Package**: interposer/substrate outline, dies (chiplets), memory stacks (HBM etc., with logic-die NMP units),
   shoreline segments (thick edge marks), die-to-die and die-to-stack links. To physical scale.
3. **Die**: top-level clusters, units, memories, routers, ports (PHYs), blocks. To scale.
4. **Cluster, recursively** (GPC > TPC > SM > sub-partition, or tile > PE array / SRAM banks): children placed by
   04 inside the parent's outline. Depth is unbounded; the breadcrumb bar shows the current path
   (`board0 > chip0 > die0 > gpc3 > tpc1`).

A container's children appear when its on-screen size exceeds 24 px; below that the container is drawn as one block
colored by the aggregate of its children. Stacked dies (3D, hybrid-bonded near-memory) show a layer selector
(`layer 0 base / layer 1 DRAM`); `Alt` shows layers side by side.

```
 Floorplan  [color: power density W/mm^2 v] [window: whole run v] [wires on] [thermal off]   scale |--1 mm--|
 +-------------------------------- package pkg0 -------------------------------------------+   legend
 | [HBM0]  [HBM1]                                                 [HBM2]  [HBM3]           |   0.0 ##
 | ======  ======   +----------------- die0 -----------------+     ======  ======          |   0.4 ##
 |                  | t00 t01 t02 t03 | L2 slice | t04 t05   |                             |   0.8 ##
 |                  | ### ### ##. ... | ======== | ### ##.   |    +-------- die1 --------+  |   1.2 ##
 |                  | t08 t09 ...     |  router mesh -----+   |====| ...                  |  |   !! limit
 |                  | ##. ### ...     |  heat on links    |   | d2d | ...                 |  |
 |                  +------------------------------------------+    +---------------------+  |
 +-----------------------------------------------------------------------------------------+
  ### busy/hot   ##. medium   ... idle (hatched)     ==== shoreline / d2d link (width = capacity)
```

**Color modes** (`C` cycles): utilization (busy fraction), power density (W/mm^2, with the node's limit from 04 drawn
as a red tick on the legend and over-limit blocks outlined), temperature (from `thermal_frames`, either block mean or
raster overlay `H`), idle fraction, energy, bytes moved, number of ops mapped, unit kind (categorical), slack
(critical path involvement), and any counter family. **Window** modes: whole run, current time window, or "at
cursor" (animates with `Space`).

**Wires.** Links drawn along their routed polylines from `wires`; bidirectional links as two offset lanes; width
proportional to capacity, color to utilization in the window. Toggle `T`. Hover shows length, latency, energy/bit
derived by 04.

**Interactions.** Click block: inspector shows path, kind, area, utilization, power/power density, peak temperature,
energy breakdown, sparkline of util over time, and the list of ops mapped to it (sorted by start time, each with time
share; click op selects it globally). Double-click container: drill in. Right-click: "show in timeline",
"select ops mapped here", "compare this block across runs". Box-select sums area and power of the selection. Overlays:
critical-path route (blocks and wires used by critical spans), physical diagnostics (overlaps, power-density
violations, shoreline overflow), mapping of the selected op (tiles it occupies highlighted, data-movement routes
drawn).

**Headless:** `--view floorplan --level package|die|system --color <mode> --window <t0..t1|all>` renders title,
scale bar, legend, optional labels.

### 6.2 NoC / link heatmap over time

**Answers:** Which links saturate, when, and because of which ops? Is traffic balanced across the mesh? Do
collectives create hot spots on the inter-chip fabric? Does congestion move over time (phases)?

**Data:** link-time matrix (3.7), `routes`, `transfers` (on demand for drill-down), `resources` (link endpoints), floorplan
for spatial layout.

Two linked panes:

```
 NoC heatmap  [layout: physical | logical mesh v] [metric: util v] [dir: both v]   t = 1.204 us   [|<][>][>>] 4x
 Spatial (at cursor, window = 1 bucket):        Link x time matrix (rows sorted by peak util v):
  o==o==o--o--o                                  link                  |time ---------------------------->|
  #  #  |  |  |       == util > 0.8              chip0.r(2,3)->r(2,4)  |..::##########::....::######::...|
  o==o==o==o--o       -- util 0.3..0.8           chip0.r(2,4)->r(3,4)  |....::########::......::####::...|
  |  |  #  |  |       .. util < 0.3              ici.chip0->chip1      |::::::::::::::::######::::::::::|
  o--o--o==o--o       #  vertical > 0.8          ...                    |                ^cursor           |
```

- **Spatial pane**: routers as nodes, links as edges at physical positions (from floorplan) or on a logical grid (from
  router coordinates in 01, for readability on irregular placements). Color = metric at the cursor bucket.
- **Matrix pane**: rows = links (sortable by peak, mean, id, or grouped by chip / NoC), columns = time buckets
  at the coarsest pyramid level fitting the pane width. Supports 50k rows via virtual scrolling with a minimap.
- **Animation**: `Space` plays the cursor through the run at a chosen speed (simulated ns per wall second, or
  "fit run in 10 s"); scrubbing drags the global cursor. Color scale locked by default across time.
- **Drill-down**: click a link (either pane) at a time: inspector lists transfers crossing it in that bucket, top-5
  ops by bytes, the collective and step if any, and queue depth if Tier B recorded it. "Show transfers" draws their
  full routes on the spatial pane.
- **Metrics**: util, bytes, queue depth, energy; direction filter.

**Headless:** a static spatial frame at time t, or the matrix; `--frames N` emits an image sequence (and optionally an
animated PNG) for reports.

### 6.3 Timeline

**Answers:** What runs when and where? Where are the bubbles? Which transfers and collectives overlap compute? What
is the critical path and how much slack does each op have? What happened at a specific ns?

**Data:** `spans` + pyramids, `transfers`, `critical_path`, counters, `ops`.

```
 Timeline   [group: hierarchy v] [collapse arrays: on] [critical path: on]   window 1.180 .. 1.260 us
            |1.18us       |1.20us       |1.22us        |1.24us       |1.26us
 v chip0
   v compute (array tile[0..127], 128)   [heat strip: busy fraction per bucket ##########::::#######]
     > tile3.mac                       [=q_proj=========][==k_proj======]  [=attn====]
     > tile3.vec                                          [rms]               [silu]
   v memory
     hbm0  bw_read  ___/~~~~~~~\________/~~~~~~\___   (counter)
     l2    occupancy ___/~~~~~~~~~~~~~~~\____
     tile3.sram  [ld][ld]   [st]  [ld][ld]
   v noc
     r(2,3)->r(2,4)        [xfer 4KB]-----> (flow arrow to consumer)
 v chip1
   ...
 v collectives
   allreduce#7 (tp group 0..7)   [rs step0][rs step1]...[ag step0]...   (spans across chips' link tracks)
 critical path  ****[q_proj]****[xfer]****[attn]****      slack of selected op: 0 ns
```

- **Tracks** follow the resource hierarchy: system > chip > die > compute arrays / memories / NoC / IO > resource >
  lanes. Arrays of identical resources (from IR `count`) collapse into one **array summary track** (heat strip from
  the array pyramid) with expand-on-click; expanding a 4096-tile array shows tiles virtualized (only visible rows
  materialized). Track grouping alternatives: by op (each op one row, spans across resources), by chip, by resource
  class.
- **Spans** colored by op kind (categorical) or by phase; stalls hatched (contention red-hatched, dependency wait
  grey-hatched); Tier A estimated spans drawn with dashed outlines and a banner "analytical schedule, no contention".
- **Transfers and flows**: transfer spans on link tracks; flow arrows from producer span to transfer to consumer span
  when a span is selected (not all at once).
- **Execution groups**: one track showing 03's groups (fused / pipelined / layer-by-layer) with pipeline bubbles
  hatched and exposed overheads marked.
- **Collectives**: a dedicated group with one track per collective instance, showing steps; selecting a step
  highlights the link spans across all participating chips.
- **Critical path**: spans flagged `on_critical_path` get a bold outline; the "critical path" track shows the chain;
  `[`/`]` jump along it; the inspector shows per-op slack.
- **Counters**: drawn as step lines with min/max envelope from pyramids when zoomed out.
- **Zoom**: from full run to 10 ns per screen (ns resolution guaranteed by i64 tick math, 3.2). Time axis labels adapt.
- **Area select** (`Shift`-drag across tracks and time): inspector shows aggregates for the box (busy time per
  resource, top ops, bytes, energy), like Perfetto's area selection.
- **Search**: Cmd-K "op:attn layer:3" highlights matching spans and adds next/prev navigation.

**Headless:** time window + track filter (glob on paths) -> PNG/SVG; default renders the chip-level summary.

### 6.4 Roofline

**Answers:** Is each op compute- or bandwidth-bound, and at which memory level? How far below its ceiling is it? Do
measured points agree with predicted ones? Which phase (prefill vs decode) lives where?

**Data:** `ops` (flops, bytes_by_level, time), `resources` (peak_flops per precision, peak_bw per memory level,
aggregated across the op's mapped resources), calibration set (optional).

```
 Roofline  [reference level: HBM v] [precision: bf16 v] [hierarchical: on] [color: phase v] [size: time share]
  attained
  FLOP/s  |                                   ___________________ peak bf16 1.2 PFLOP/s
   1e15   |                        _________/      o (q_proj, prefill)
          |              ______/ L2 4.8 TB/s     o  O
   1e14   |       ______/                    x->o            o = predicted   x = measured (A100 calib)
          |  ___/  HBM 2.0 TB/s     . . (decode ops, bandwidth bound)
   1e13   | /         .:.
          +------------------------------------------------------------------ arithmetic intensity
             0.1        1         10        100       1000   FLOP/B (w.r.t. reference level)
```

- **Ceilings**: one horizontal line per compute precision the design supports (from 01, at the op's mapped resource
  set; multi-tile ops scale by tiles used), one diagonal per memory level (bandwidth to that level, summed across the
  mapped resources), plus inter-chip link bandwidth for collectives. Ceilings labelled with value and unit.
- **Points**: one per op instance (or aggregated per op path across layers, toggle), x = flops / bytes at the
  reference level, y = flops / time. **Hierarchical mode** plots each op once per memory level with points connected,
  so the binding level is visible (as in hierarchical roofline analysis).
- **Per-phase**: aggregate points per phase (prefill, decode, mixed) drawn as large diamonds.
- **Intervals**: when the run carries intervals, each point gets a vertical error bar from its low/high corners
  (attained FLOP/s; x moves only if bytes change) and ceilings that depend on ranged parameters (sustained DRAM
  and link bandwidth) are drawn as bands.
- **Measured**: when a calibration set matches the design (device and op keys), measured points are drawn hollow with
  a thin line to the predicted point; only for designs that correspond to a measured device (A100 now, TPU v5e/v6e
  when available).
- **Interactions**: click point selects op globally; lasso multi-select; hover shows the limiter explanation (6.5);
  log-log axes; ceilings toggle per level.

**Headless:** yes; default reference level = outermost DRAM.

### 6.5 Bottleneck attribution

**Answers:** Where do time and energy go (by op, op kind, phase, resource class, chip)? What limits each op and what
would it gain if that limit were lifted? Which few changes would help most?

**Data:** `limiters`, `aggregates_op_resource`, `aggregates_resource`, `ops`, `critical_path`.

```
 Bottleneck   [group by: op kind v] [metric: time v] [critical path only: on]
 Time (3.88 ms on critical path)
  matmul      |##############################--------| 61%   compute 38% | HBM bw 19% | link 4%
  attention   |###########---------------------------| 22%   HBM bw 15% | SRAM cap 5% | dep wait 2%
  collective  |#####---------------------------------|  9%   link bw 9%
  elementwise |###-----------------------------------|  6%   HBM bw 6%
  other       |#-------------------------------------|  2%
 Energy (412 mJ)                treemap: chip0 > compute 48% | HBM 31% | NoC 9% | leakage 12% ...
 Top limiters (time recoverable if lifted, upper bound):
  1. HBM read bandwidth on chip0.hbm*        -0.71 ms (18%)   ops: decode attention, lm_head
  2. bf16 compute on chip0.tile[*].mac        -0.52 ms (13%)
  3. ici link chip0<->chip1 bandwidth         -0.20 ms  (5%)   ops: allreduce#3..#31
 Selected op: layer3.attn.q_proj (prefill)
  "61% of its 42.1 us is bound by HBM read bandwidth (attained 1.88 of 2.04 TB/s, 92%). Compute alone would take
   16.4 us. Next limiter if HBM were unbounded: bf16 compute at 16.4 us. SRAM capacity at tile level forces K to be
   re-read 3 times (bytes at L1 3.0x minimum). On critical path, slack 0."
```

- **Breakdowns**: stacked bars by group (op kind, op path, phase, layer, chip, resource class), each bar split by
  limiter term using `limiters.share`; energy as treemap (chip > component class > resource kind > op) with sunburst
  alternative.
- **Recoverable time**: for each limiter term, the upper bound on time saved if that term's time dropped to the
  runner-up term (`limiters.rank` 0 vs 1). Labelled as an upper bound. Where 03 computed a shadow price (Tier A, top 5
  resources re-evaluated at +10% bandwidth), it is shown next to the bound as the measured sensitivity. Tier B
  contention terms are included only as measured stall time.
- **Explanation text** is produced by 03's `explain_run` (kiln-sim, 03 §10; 00 decision 7) and persisted in the
  trace (`bottleneck.summary` and per-op explanation strings); this view displays it verbatim, so what a human reads
  here is exactly what the evolution LLM is told.
- **Interactions**: click a bar segment selects the ops in it (cross-highlight in timeline/floorplan); click a top
  limiter selects its resources on the floorplan.

**Headless:** bars + top-limiter table + explanation of the top-N critical ops (PNG/SVG, and `--format json` for the
text).

### 6.6 Design diff / compare

**Answers:** How does an evolved design differ from A100 (or from its parent) in area, power, latency, energy, and
where? Which ops got faster or slower and why (limiter change)? For the same design, how do two mappings, tiers or
calibration sets differ?

**Data:** two traces A and B (`kiln viz --compare a.kiln b.kiln`); `kiln_trace::analysis::align`.

**Alignment rules** (in order):

1. Ops align by op path when the workload hashes match; otherwise by op path with a warning banner and an "unmatched
   ops" list.
2. Resources align by path only when design hashes match or the structural diff (from 01's canonical form) maps paths
   one-to-one; otherwise resource-level diff is off and comparison happens at aggregate levels (resource class,
   chip, memory level) and spatially (same physical scale).

```
 Compare  A: a100_ref.kiln (A100-SXM4 envelope)   B: evo_g42_d17.kiln          [swap] [lock scales: on]
 +------------------ A ------------------+------------------ B ------------------+  Headline   A      B     delta
 | (floorplan, same mm-per-px as B)      | (floorplan, same mm-per-px as A)      |  area    826    812  -1.7%
 |                                       |                                       |  power   400    388  -3.0%
 |  color: power density (shared scale)  |                                       |  latency 3.88  2.91  -25%
 +---------------------------------------+---------------------------------------+  energy  412    351  -15%
 Per-op delta (waterfall, sorted by |delta time|):
  layer*.attn.core    -0.41 ms  ####################    limiter A: HBM bw -> B: compute
  layer*.mlp.up       -0.22 ms  ###########
  allreduce*          +0.07 ms  ###  (more chips' link traffic)
 Roofline overlay: arrows from A point to B point per op.     Timeline: A and B stacked, cursor synced (abs | normalized)
```

- **Panes**: side-by-side floorplans at the same physical scale (`mm per px` locked) and same color scale; headline
  table with deltas, each value shown as central with `[low, high]` and each delta as an interval; per-op waterfall; limiter transitions (A's binding term -> B's); roofline overlay with arrows;
  stacked timelines with synced cursor (absolute time, or normalized to each run's duration).
- **Same-design mode** (resource alignment available): single floorplan colored by delta (diverging map), e.g.
  "utilization B minus A", plus per-resource delta table.
- **Structural diff panel**: summary of IR differences (counts of units per kind, memory capacities per level,
  NoC topology, link counts, process node), sourced from 01's canonical IR diff; click an item highlights the
  corresponding region on both floorplans.
- **More than two**: `--compare a b c ...` gives a headline table across N runs plus small-multiple floorplans; deep
  diff stays pairwise (choose pair).

**Headless:** `--view compare` renders side-by-side floorplans + headline table + top-10 waterfall; this is the
default image attached when the evolution loop reports a new elite against its parent and against the reference.

### 6.7 Evolution view

**Answers:** What regions of the descriptor space have been filled, and how good are they? Where did a good design
come from (lineage, which mutations helped)? Is search still improving (score over generations, coverage)?

**Data:** archive directory (3.9); run files for opened designs; thumbnails.

```
 Evolution  archive: runs/evo_2026-10-04   [x: area mm^2 v] [y: power W v] [slice: chips=1 v] [color: fitness]  live: on
  power W
   500 | .  .  [] [] [] .  .        [] = elite (color = fitness)   . = empty (hatched)   X = invalid-only
   400 | .  [] [] ## [] [] .        ## = selected cell
   300 | [] [] [] [] [] [] []
   200 | .  [] [] [] [] .  .        hover: thumbnail floorplan + headline
   100 | .  .  [] .  .  .  .
       +-----------------------  area mm^2
         200 400 600 800 ...
 Lineage of selected elite (d17, gen 42):          Score over generations:
   ref_a100 (g0)                                     best  ____/~~~~~~~----~~~~~
     +- d3 (g5)  "double L2, halve tiles"            QD    __/~~~~~~~~~~~~~~~~
         +- d9 (g17) "add NMC units to HBM base die" coverage _/~~~~~~~~~~~~~
             +- d17 (g42) x d12 (crossover)          invalid rate \___________
```

- **Archive grid**: choose 2 descriptor axes for x/y from the archive's descriptor list (defaults to 06 section 6.3's
  standard descriptors with their fixed ranges and log flags; vector descriptors such as `energy_split` are offered
  per component); remaining axes become slice
  selectors or a facet grid (small multiples). Cell color = fitness (or any fitness component, `fitness_low`, or
  interval width); empty cells hatched; cells with only invalid attempts marked. Hover: thumbnail floorplan PNG
  (pre-rendered, 7.3) and headline numbers with `[low, high]`.
  Click: select design; double-click or `Enter`: open its run in a new tab (if the run is `summary` level, offer
  "re-run at Tier B" which calls `kiln eval` in the background and opens the result).
- **Lineage tree**: DAG (crossover has 2 parents) laid out top-down by generation; nodes colored by fitness, displaced
  ancestors greyed; edge label = `mutation_summary` (the LLM's one-line description of its change). Clicking an edge
  opens the compare view (parent vs child). "Path to root" highlights the ancestry of the selected elite.
- **Score over generations**: best (with its low/high band), median, QD-score, coverage, invalid rate,
  evaluations per generation.
- **Live mode**: `--watch` uses `notify` to pick up appended batches; grid and plots update without losing selection.
- **Scatter mode**: any two descriptors or fitness components as a scatter over all evaluated designs (not just elites),
  with the Pareto front highlighted and low/high error bars per point.

**Headless:** archive grid image and score plot, used in evolution progress reports.

### 6.8 Calibration view

**Answers:** How accurate is the model per device, per op family, per tier? Where are the outliers? Is error
systematic (e.g. small GEMMs under-predicted)? Did a calibration-constant change help or hurt?

**Data:** calibration set (3.9); optionally two calibration sets (before/after) for diff.

```
 Calibration  device: [A100-SXM4-40GB v]  tier: [A v]  op family: [all v]   set 9c1e..  n = 1,284
 Predicted vs measured (log-log)               Error histogram  log2(pred/meas)
  pred |            . ./                         |      ___
       |         .:./:  +-25%                    |    _|   |_
       |       .:/:.                             |  _|       |__
       |     ./:.                                | |            |_
       |   ./                                    +----------------------
       +---------------- measured                 -1   -0.5   0   0.5   1
 Worst outliers                                  Error vs size (GEMM sweep):  heat over (M, N) at fixed K
  gemm 4096x128x4096 bf16  pred 41 us  meas 63 us  -35%
  rmsnorm 1x4096 bf16      pred 2.1 us meas 4.9 us -57% (launch overhead?)
```

- Fit vs test points distinguished by marker shape; the noise floor (`measured_cv`) drawn as an error bar so
  "within noise" is visible.
- Scatter with y = x and +-10% / +-25% bands, color by op kind, facet by device; error histogram of
  `log2(pred/meas)` with median, MAPE and p90 annotated; error vs shape heatmaps for sweeps (choose two shape dims);
  outlier table sorted by |error| linking to the measurement source file and to the predicted op's limiter
  explanation; Tier A vs Tier B toggle (and the A-vs-B disagreement histogram, which must stay within 06's tolerance).
- **Set diff**: two calibration sets -> per-point arrows and a delta-MAPE table per op family.

**Headless:** scatter + histogram + table, embedded in calibration reports produced by `kiln calibrate`.

### 6.9 Auxiliary panels

- **Ops table**: virtualized table of all ops (path, kind, phase, time, energy, flops, limiter, slack), sortable,
  filterable; row click selects.
- **Resources table**: same for resources.
- **Mapping inspector**: for a selected op, the mapping record (tiles, parallelism, loop nest pretty-printed, tiling
  per level) from `mapping`.
- **Run info**: manifest provenance, headline numbers, physical floor checks, diagnostics.

---

## 7. Headless rendering and evolution-loop integration

### 7.1 Determinism

Headless output must be byte-identical for the same inputs across machines: fonts are bundled (one sans, one mono,
embedded in `kiln-viz-render`, no system font lookup), tiny-skia rasterization (CPU, no GPU variance), fixed
antialiasing settings, sorted iteration, no time-dependent content (render timestamp excluded unless
`--stamp`). PNGs are written without timestamps in metadata. SVG output uses fixed float formatting (4 significant
decimals in view units).

### 7.2 Interfaces

- CLI: `kiln viz render <run.kiln> --view <name> [view options] --out fig.png|fig.svg [--size WxH] [--scale 2]
  [--theme light|dark] [--state '<view-state string>']`. Multiple `--view` flags produce multiple files.
- Python (`kiln-py`): `kiln.render(result_or_path, view="floorplan", fmt="png", **opts) -> bytes`; works directly on
  an in-memory `SimResult` without writing a `.kiln`. Also `kiln.explain(result, op=None) -> str` (06; wraps 03's `explain_run`).
- Rust: `kiln_viz_render::render(&TraceHandle, &ViewSpec) -> Scene`, then `to_png`/`to_svg`.

### 7.3 Evolution usage (contract with 06)

- For each new elite, the loop may request a small **thumbnail** (floorplan, package level, power density, 320x200)
  stored in the archive and in the `.kiln` (`thumbnails` member) for hover previews.
- For LLM prompts, the default image bundle per candidate is: floorplan (power density), bottleneck summary, and
  compare-vs-parent; size 1024x640 each; total render budget <= 0.5 s per candidate on one core. 06 decides whether
  multimodal input is used; text (`explain`) is always available.

---

## 8. CLI entry points (in `kiln-cli`)

| Command | Behavior |
|---|---|
| `kiln viz <run.kiln>` | open the viewer on a run (any trace level) |
| `kiln viz <design.json>` | no simulation: runs 01 expansion + 04 placement only and opens the floorplan (structure, area, kinds); "Evaluate" button runs `kiln eval` and reloads (06's `kiln viz <design|result|trace>`) |
| `kiln viz <result.json>` | follows 06's `Result.trace` handle `{id, tier, path, viewer_url?}` to the `.kiln` |
| `kiln viz <run.kiln> --view timeline --state '<s>'` | open at a specific view/state |
| `kiln viz --compare a.kiln b.kiln [c.kiln ...]` | compare view |
| `kiln viz --archive <dir> [--watch]` | evolution view, live if `--watch` |
| `kiln viz --calibration <dir|file> [--vs <other>]` | calibration view, optional set diff |
| `kiln viz --perfetto <run.kiln>` | export and open in ui.perfetto.dev via localhost (3.11) |
| `kiln viz web [<run|dir> ...] [--port 8080] [--bind 127.0.0.1]` | serve the embedded WASM build plus the given files over HTTP (range requests); for remote pods via `ssh -L` |
| `kiln viz render ...` | headless PNG/SVG (7.2) |
| `kiln trace info <run.kiln> [--json]` | manifest and headline |
| `kiln trace validate <run.kiln>` | consistency checks (3.8); non-zero exit on violation |
| `kiln trace export <run.kiln> --perfetto|--chrome-json|--parquet|--csv <table>` | exports |
| `kiln trace pack|unpack|upgrade|recover` | container ops (3.6, 3.10) |

`kiln viz` with no display (no `DISPLAY`/`WAYLAND_DISPLAY` on Linux) prints a hint to use `kiln viz web` or
`kiln viz render` instead of failing with a windowing error. All errors use the overview's structured error format.

`Result.trace.viewer_url` (06) is filled when a viewer base URL is configured (`KILN_VIEWER_URL`, e.g. the static web
build or a running `kiln viz web`) and the trace is reachable from it; otherwise null. Exit codes follow 06 section 7.

**Web build distribution.** The WASM build (`trunk`) is (1) embedded in the `kiln` binary for `kiln viz web`, and (2)
publishable as a static site (any static host) that opens files by drag-and-drop or `?url=<https url with CORS>`, with
view state in the fragment. Nothing is uploaded anywhere by the viewer itself.

---

## 9. Testing strategy

1. **Format round-trip** (`kiln-trace`): property tests (proptest) generate random traces (resources, ops, spans,
   transfers, counters) -> write -> read -> equal; writer determinism: two writes of the same inputs are
   byte-identical; sorted-order and index invariants checked.
2. **Golden traces**: small hand-built and engine-produced traces per schema version in `tests/golden/trace/`;
   readers must open all of them; `upgrade` output compared to the next version's golden.
3. **Fuzzing**: `cargo fuzz` targets on container header, manifest and Arrow member parsing (malformed or truncated
   files must yield structured errors, never panics or UB).
4. **Analysis correctness**: unit tests for pyramids (each level equals brute-force bucketing), roofline points,
   displayed explanation text (snapshot with `insta`), alignment in compare, validate checks (each check has a trace
   that violates it).
5. **Rendered-view snapshots**: for each view, a fixed set of `ViewSpec`s over golden traces rendered headless:
   - SVG snapshots as text via `insta` (diffable in review).
   - PNG snapshots compared with a perceptual diff (`dify`-style, threshold 0 changed pixels by default since the
     CPU path is deterministic; tolerance knob only for font-hinting changes on dependency upgrades).
   - Snapshot update via `cargo insta review` / `KILN_UPDATE_SNAPSHOTS=1`.
6. **Interactive tests**: `egui_kittest` 0.36.2 harness drives the real app headless: click a floorplan block ->
   assert selection and inspector contents; scrub cursor -> assert NoC frame; keyboard shortcuts; view-state string
   round-trip; kittest wgpu snapshots for a few full-window layouts (run only on a CI runner with a fixed GPU/software
   adapter, tolerance > 0).
7. **Perfetto export**: CI loads exported traces with Perfetto's `trace_processor` (pinned v58.2) and asserts slice
   counts, track hierarchy and flow counts via SQL; Chrome JSON validated by schema.
8. **Performance gates**: criterion benches on synthetic traces (1 GB, 25M spans, 10k and 100k blocks): open time,
   view-model build time, Scene size per frame, headless render time. Run on a designated perf machine (not shared CI)
   with regression thresholds of 10%; frame-time measured with a scripted pan/zoom replay in the native app.
9. **Web build**: CI builds the WASM target and runs a smoke test (load golden trace, render each view) in a headless
   browser; this is the only place a browser enters the test suite.

---

## 10. Acceptance criteria (06's milestone M5 refers to these)

1. Opens every golden trace of every supported schema version; `kiln trace validate` passes on all engine-produced
   goldens.
2. Performance budget (section 5) met on the perf machine for the 1 GB / 25M-span synthetic trace and the
   10k-block floorplan.
3. All eight views render headless for every golden trace, and snapshot tests are green.
4. Perfetto export of the Tier B goldens loads in `trace_processor` v58.2 with matching slice and flow counts.
5. Clicking any block in the floorplan reaches its spans in the timeline and its ops in the bottleneck view in one
   action each (linked selection works across all views).
6. Elites from an archive open from the evolution grid in < 1 s (summary level).

## 11. Implementation phases

| Phase | Contents | Exit criterion | 06 milestone |
|---|---|---|---|
| V0 | `kiln-trace` schema, writer, reader, container, validate, Perfetto export; `kiln trace` CLI | Tier B golden traces open in ui.perfetto.dev | M0 (schema, writer, reader); Perfetto exit at M4 |
| V1 | `kiln-viz-render` Scene + PNG/SVG; floorplan, roofline, bottleneck views headless; `kiln.render` in kiln-py | evolution loop attaches floorplan + bottleneck PNGs | M3 (needs 04 floorplan) |
| V2 | eframe app: shell, selection model, floorplan, timeline (pyramids), inspector, time bar | performance budget met on 1 GB synthetic trace | M5 |
| V3 | NoC heatmap, compare, calibration views; `kiln viz web` + static WASM build | share-a-link workflow works | M5 |
| V4 | Evolution view (grid, lineage, live mode), thumbnails | browse a 10k-design archive live | M5 |

---

## 12. Implementation notes (M5 v0, 2026-10-05)

What the first implementation does where this spec left room, or deviates from it.

- **Crates and pins.** `eframe`/`egui`/`egui-wgpu` 0.35.0 (rustc 1.92; 0.36 needs 1.95), `arrow-array`/`arrow-ipc`
  60.0.0 without compression features, `tiny-skia` 0.12.0, `ab_glyph` 0.2.32 with egui's own font files
  (`epaint_default_fonts`, Ubuntu-Light and Hack) so PNG, SVG and the app use the same faces, `prost` 0.14.4 with
  hand-declared messages carrying Perfetto's field numbers (no `protoc`, no vendored `.proto` files), `notify`
  8.2.0, `insta` 1.49.0. Not used yet: `egui_tiles` (fixed tabs plus inspector instead of docking), `egui_plot`,
  `memmap2` (the workspace forbids `unsafe_code`; files are read whole), `parquet`, `lz4_flex`/`zstd`, `dify`.
- **Lean default builds.** The eframe app is behind `kiln-viz`'s non-default `gui` feature, enabled through
  `kiln-cli --features gui`; without it `kiln-viz` holds only the GUI-free app state and `kiln viz <run>` exits 69
  with a hint. `cargo test --workspace` therefore never builds winit, wgpu or the windowing stack.
  `kiln-viz-render` (tiny-skia, fonts) is a normal dependency of `kiln-cli`. `kiln-trace`, `kiln-viz-render` and
  `kiln-viz` (with `gui`, `start_web` entry) compile for `wasm32-unknown-unknown`; the trunk page, file loading in
  the browser and `kiln viz web` are not built yet.
- **Tables.** Implemented: `resources`, `ops`, `phases` (new: one row per simulated phase with makespan, corner
  makespans, floors, energy, power, tokens/s, `explain_run` text, and the phase's offset on the trace time axis;
  phases are laid out back to back), `aggregates_resource` (+ `phase`, `utilization`), `aggregates_op_resource`,
  `limiters`, `bottleneck` (+ `phase`), `groups` (+ `phase`, `binding`), `collectives` (+ `name`, `phase`),
  `floorplan` (`poly` as a flat `x0, y0, x1, y1, ...` list), `ceilings` (new: per-chip roofline ceilings: dense
  peak per MAC mode, bandwidth per memory level, inter-chip links), `diagnostics`, `run_scalars` (`name`, canonical
  `json`: energy, power, invariants and calibration per phase, and the result without `sim`/`timing`), `spans`.
  `ops` gains `family` (path without `.i<N>`/`.k<N>`), `energy_j`, `time_low_s`, `time_high_s`. Schema 1.1
  (2026-10-06) adds `wires`, `package_geometry`, the floorplan columns of §3.4 and a `design_summary` run scalar
  (the design sheet). Not yet written: `mapping`, `routes`, `transfers`, `counters`, `thermal_frames`,
  `critical_path` (Tier B, M4) and the
  LOD pyramids. Enum codes index the manifest lists; `ops.kind` is a coarse class derived from the op path
  (matmul, attention, norm, memory, ...) because `SimResult` does not carry 02's op tag.
- **Writing a trace.** `kiln eval <design> --trace summary|ops -o run.kiln`; the engine always records ops for a
  `.kiln` output so the summary tables are complete, and `ops` adds one estimated span per op on its binding
  resource (lanes assigned so spans in a lane never overlap). Resources are the expanded model's instances plus
  03's engine resources (links and ports as `channel`, the sequencer); memory levels also resolve 03's
  multi-instance group names (the template entity path).
- **Floorplan (2026-10-06).** With the design, `kiln-trace` builds the floorplan from kiln-phys's model
  (`Phys::new`, Tier A): die outlines, die-level macros (PHYs on their shoreline edges), memory stacks and package
  outlines are its placement (`source = placed`), blocks inside a macro its hierarchical layout (`layout`).
  kiln-phys arranges dies without compute-unit area (04 §6.1, P6), so units and their local buffers have no
  rectangle there: they are laid out by area inside their placed parent (`filled`), as are their siblings.
  Routers and footprint-less ports are position-only `site` rows. One frame per package, y down; packages are laid
  out side by side (on their 2-D array coordinates when declared), containers above them get the box around their
  contents. Every enabled channel is a `channel` resource and a `wires` row: a Manhattan L between its endpoints'
  positions with kiln-phys's link length, latency and energy (cluster-internal lengths are kiln-phys's 0.5 sqrt of
  the live cluster area, not the drawn L). Blocks kiln-phys leaves out of geometry (no channels, e.g. A100
  `uncore`) are listed, not drawn. `manifest.floorplan_source = "kiln-phys/m3 tier A"`; if kiln-phys panics or
  places no die, the unplaced treemap remains, labelled UNPLACED with the reason. The view draws to scale (scale
  bar in mm), opens containers above `drill_px`, colors by run metrics, block kind, area, static power or power
  density, and draws wires aggregated onto the blocks drawn at the current depth (bandwidth and bytes summed,
  utilization, length, energy and latency maximal), colored by utilization (runs) or bandwidth (designs) or any
  of traffic, length, energy, latency, width ~ log bandwidth. Stacked dies are drawn one panel per layer (or one
  layer with `--layer`, the others outlined); vertical links show as rings at both ends.
- **Design-only (2026-10-06).** `kiln viz render <design> --view floorplan|design|wires` and `kiln viz <design>`
  build a structure trace (kiln-ir expansion, reference and search profile validation, kiln-phys) without kiln-sim.
  `design` is a sheet of peaks per precision, memory levels, off-chip memory, networks, area per die with its
  parts, a power estimate (all MAC units and DRAM at peak, nominal clocks, against the TDP), clocks and V/f,
  validation and E-PHYS findings; `wires` is a length vs energy-per-bit scatter and a table of link classes.
  Run-only color modes fall back to `kind` on a design.
- **Op times.** Tier A op envelopes do not span their execution group's traffic (a fused GEMV's envelope ends long
  before its weights finish streaming), while group times tile the scheduled window. Views therefore (a) plot the
  roofline per execution group (FLOPs and boundary bytes summed, labelled by the largest op) and (b) attribute
  group time to ops by the group's binding term (FLOPs for compute-bound groups, boundary bytes for memory-bound
  ones) for bottleneck bars, recoverable time and compare. 03 records bytes *delivered into* each level, so the
  traffic across a level's boundary is what was delivered into it plus into the next level inward.
- **Views.** All ten render headless (the eight of §6 plus `design` and `wires`). NoC at Tier A is a ranked list of link utilizations (no time axis); the
  timeline at Tier A shows phases, execution groups, op envelopes per family and estimated span lanes; compare
  aligns by `phase/op family`; the calibration view reads the 05 §3.9 table or `kiln calibrate report --format
  json` output (`kiln viz --calibration`), and measured points also appear on the roofline (`--device`).
- **View state** is the JSON of the view spec (`--state`, "copy view state"), not yet the compact fragment form.
- **Tests.** Golden traces in `kiln-trace/tests/golden/trace/` (open, validate, rewrite byte-identically, export),
  insta SVG snapshots of every view in `kiln-viz-render/tests/snapshots/`, PNG determinism, CLI end-to-end
  (`eval -o .kiln`, `trace info|validate|export`, `viz render`), app-state tests (hit-test, select, drill, linked
  selection) on a golden. The Perfetto export was checked once by hand with `trace_processor` v57.2 (the
  `perfetto` Python package): slice count = spans + phases + groups, nested tracks, interned names, debug args.
  Not yet: proptest round-trips of random traces, fuzzing, a CI `trace_processor` test, egui_kittest,
  performance gates.

## Cross-section dependencies

| From | What this section needs | Used in |
|---|---|---|
| 01-hardware-ir (partially read: sections 4.1 and naming) | Instance naming and expansion order from its section 15 (`array_id`, `array_pos` derived from `count` + layout); expanded-model indices from section 16; floorplan data from section 11 (die outline, shoreline sites, keepouts); unit/resource kind taxonomy; memory level numbering; peak FLOP/s per precision per unit; peak bandwidth per memory level and per link; router coordinates for logical NoC layout; a canonical structural diff of two designs | 3.4 `resources`, 6.2, 6.4, 6.6 |
| 02-workload-ir | Stable op paths; op kind taxonomy; phase enum; layer index; FLOP and per-level byte definitions; collective op representation (group, algorithm, steps) | 3.4 `ops`, 6.3 collectives, 6.4 |
| 03-mapping-engine (read; aligned to its sections 5.5 and 10) | `SimResult`, `Binding`, `Bottleneck`, critical-path reasons, groups and collectives as defined there; still needed from 03: per-op runner-up term with `B_r` for every evaluated term (not only binding), lane assignment for multi-issue resources; mapping record per op; Tier B emission of spans/transfers/counters through `TraceWriter` with lanes; Tier A analytical schedule (`ops` level, flagged estimated); the `limiters` table (terms of the cost expression, binding term, shares); critical path and per-op slack; contention vs dependency stall attribution; collective decomposition into steps and routes | 3.4, 3.12, 6.3, 6.5 |
| 04-physical-model | Floorplan rects/polygons per block with die and stack layer; package geometry (interposer, HBM stacks, shoreline segments); routed wire polylines and lengths per link; power per block (aggregate and time series); thermal grid frames; power-density limits per process node; physical diagnostics | 3.4 `floorplan`, `wires`, `thermal_frames`, 6.1 |
| 06-validation-api (read; aligned to its 6.1 to 6.5, calibration schemas and CLI table) | Archive directory schema (3.9) adoption or amendment (06 leaves binning to the evolution loop and names no archive file format); a calibration report table (3.9) written by `kiln calibrate`; `trace: "ops"` option; Tier A/B tolerance (for calibration view); `kiln-py` API surface for `render`/`explain`; CLI conventions and structured errors; whether evolution prompts use images | 3.9, 6.7, 6.8, 7, 8 |
| 07-prior-art | Confirm reuse decisions: Perfetto (export target, UI conventions), Rerun (architecture precedent: egui + wgpu + Arrow, web build), Chakra / ASTRA-sim trace formats (possible additional export), Timeloop/Accelergy and ZigZag/Stream visual outputs (what users of those tools expect) | 2, 3.11 |

What other sections can rely on from this one: the `TraceWriter` API (3.12), the `.kiln` container and table schemas
(3.4 to 3.6), `kiln_trace::analysis::{align, validate}` (3.8), headless `render` (7.2), and the CLI commands
in section 8.

---

## Open questions

1. **Archive ownership.** Resolved by 00 decision 7: owned here, consumed by 06.
2. **Images in LLM prompts.** Do floorplan/bottleneck PNGs measurably help the evolution LLM over the `explain`
   text? This is a cheap ablation once V1 lands; if images don't help, only thumbnails for humans get rendered and
   the per-candidate render budget disappears.
3. **Tick size.** 1 ps ticks cover +-106 days. Is anything below 1 ps needed (e.g. on-die wire delays summed per hop)?
   If so, make `tick_s` per trace (already supported) and pick 0.1 ps for Tier B.
4. **Tier A timelines.** Resolved by 00 decision 8: `ops` on request, `summary` by default for evolution.
5. **Thermal transients.** Will 04 produce time-varying thermal frames or only steady state? The floorplan's
   temperature-at-cursor mode depends on it.
6. **Perfetto track cap.** Default 512 tracks per chip in Perfetto export is a guess; needs measurement against the
   current Perfetto UI with a 4096-tile design.
7. **Perfetto localhost opening.** Section 3.11 mirrors Perfetto's `open_trace_in_ui` (port 9001, `?url=` to
   localhost). Perfetto's deep-linking doc says `?url=` expects HTTPS with CORS; confirm the localhost exception still
   holds in v58 before relying on it, otherwise fall back to writing the file and printing instructions.
8. **Web memory ceiling.** wasm32 caps at 4 GB; Memory64 support in browsers and in wasm-bindgen would lift it. Decide
   whether to target wasm64 later or keep "large traces use native or `kiln viz web`".
9. **Diff across structurally different designs** stays at aggregate and spatial level. Is a heuristic block matching
   (by kind + relative position) worth building for evolved-vs-parent diffs, where most structure is shared? 01's
   canonical diff may make it unnecessary.
10. **One explanation function.** Resolved by 00 decision 7: 03's `explain_run` in kiln-sim.
11. **Crate split.** Resolved: `kiln-viz-render` is in 00's crate table.
