# kiln design language (compact; full spec: spec/01-hardware-ir.md)

A design is a Python program whose `build()` returns one `kiln.hw/1.0` document as a dict (JSON data only; no
I/O, no network, finishes in seconds). `PARAMS` holds the knobs `build()` reads; `PARAM_SPACE` bounds them.

## Document
`{schema: "kiln.hw/1.0", name, tech: "tsmc_n7", exec_model, clocks, power, params, templates, system, set}`
- `exec_model`: `host_launched` (GPU, per-kernel launch), `device_queued`, `static_dataflow` (one compiled program).
- `clocks: [{id, freq: "1410MHz", vf: [{freq, voltage}]}]`; `power: [{id, members, cap: "400W", policy: "dvfs"}]`.
- Quantities are strings with units: `"192KiB"`, `"1.41GHz"`, `"2.43Gbps"`, `"400W"`, `"25.6mm"`.
- `params` + `"=expr"` strings: `"= n_mxu * 4"`; `if(c,a,b) min max ceil floor round sqrt log2 pow abs`.
- `templates: {name: {kind: "cluster", body: {...}}}`, used with `{id, use: "name", with: {param: v}}`.
- Replication on any entity: `count`, `layout: {grid: [r, c]}` (instances named `id{r}_{c}`), `disabled: [...]`
  (harvested instances), `vary`.
- Selectors: `sm*`, `sm[3]`, `sm[0..54]`, `tile[1;2]`, `**`, `^.` (parent). A selector matching nothing is an error.

## Hierarchy
`system: {package: {id, substrate, dies: [...], mem_stacks: [...], power}}` (shorthand for board/package).
A die: `{id, default_clock, floorplan: {outline: {type: "fixed", w, h}, shoreline: [{id, count, edge, kind}]},
clusters, units, memories, networks, blocks, ports}`. Clusters nest recursively (`clusters`, `units`, `memories`,
`networks`).

## Compute units (`units: [...]`)
- `kind: "matrix"`, `geometry: {systolic: {rows, cols}}` | `{mma: {m, n, k}}` | `{outer_product: {rows, cols}}`,
  `dataflow`: weight_stationary | output_stationary | input_stationary | row_stationary | any,
  `precisions: ["bf16*bf16+fp32@0.125", "int8*int8+int32"]` (`@rate` = ops per cycle multiplier),
  `feeds: {a: "rf", b: "rf", o: "rf"}` (memories the operands come from), optional `local` buffers,
  `sparsity`.
- `kind: "vector"` (`lanes`, `sublanes`), `"special"` (`functions: [exp, rsqrt, ...]`), `"scalar"`.
- Near-memory: a unit with `near: {memory, granularity}` inside a stack's `logic_die` or bound to an SRAM.

## Memories (`memories: [...]`)
`{id, kind: register_file|scratchpad|cache|fifo, capacity, banks, word_bits, ports: [{dir: read|write|rw, count,
width_bits}], cache: {line, ways, ...}, backing}`. Bandwidth is derived from port widths x clock.

## Networks
`{id, topology: bus|crossbar|p2p|{type: ring}|{type: mesh, dims}|{type: torus, dims}|{type: custom, routers,
edges}, endpoints: [{select, at: {router: [i]}, port}], link: {width_bits}}`. Bandwidth = width x clock.

## Off-chip memory (`mem_stacks`)
`{id, count, kind: hbm2|hbm2e|hbm3|hbm3e|hbm4|lpddr5x|gddr7|ddr5|stacked_dram..., capacity, io_width_bits,
pin_rate_bits_per_s, attach: {phys: [...]} | {network: "die.dma"}, logic_die}`. Stack bandwidth =
io_width_bits x pin rate / 8. HBM stacks need die shoreline sites. Off-chip memory is bought, not designed: total
active stack bandwidth and capacity may not exceed the baseline's (E-ENV-0007/0008; more stacks or a higher pin
rate score 0). Near-memory units' internal bandwidth is inside the stack and is not counted as off-chip.

## Rules (profile `search`)
- No `family` key. No performance asserted without its cost: any bandwidth/latency/energy/area override must not
  beat what kiln derives from structure (E-IR-1101). Cost-neutral changes (re-placement, topology, memory splits,
  dataflow, precisions within a unit's modes) are exactly what search should find.
- Every result is checked against physical floors; a floor violation is a simulator bug and the design is
  quarantined, never rewarded.
- Area, power and HBM shoreline are priced by kiln-phys and must fit the baseline's envelope (matched envelope).
  If the prompt says the envelope is not yet checked, still design within it: designs are re-scored at matched
  envelope later and those that only win by adding silicon or power will be discarded. Off-chip memory bandwidth
  and capacity are checked now.

## Scoring
Whole decode/prefill steps of the listed workloads; score = tokens/s ratio vs the baseline, both simulated by kiln
under the same software stack (`kiln_ideal`); interval `[low, high]` from calibration-parameter corners. Feedback
lists, per workload, which resources bind (e.g. "off-chip memory bandwidth 92%").
