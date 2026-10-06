# 01: Hardware IR (`kiln-ir::hw`)

Status: spec v0, implementation-ready. Owner of: hardware entity types, the authoring language (templates,
params, expressions, selectors, imports), expansion to instances, the expanded hardware graph that engines consume,
canonical form and design hashing, structural validation, schema versioning. Formulas for area, energy, latency,
wire delay and placement live in 04-physical-model; this section defines only the data those formulas read and the
fields through which their outputs can be overridden.

Conventions from 00-overview.md apply throughout (SI base units in IR, string ids `[a-z0-9_.-]+`, dotted paths,
deterministic ordering, structured errors).

## Contents

1. Design principles
2. Quantities, units and literals
3. Document structure
4. System hierarchy
5. Compute units
6. Precisions and op classes
7. On-chip memories
8. Off-chip memory
9. Near-memory and in-memory compute
10. Interconnect
11. Floorplan data
12. Clocks, voltage, power caps, harvesting
13. Technology reference
14. Compact authoring: templates, params, expressions, selectors, imports
15. Expansion pipeline and instance naming
16. Expanded hardware model (engine-facing)
17. Canonical form and design hashing
18. Validation (error codes)
19. Schema versioning and migration
20. Worked examples
21. Cross-section dependencies
22. Open questions

---

## 1. Design principles

| # | Principle | Consequence |
|---|---|---|
| P1 | No structural caps | Every collection is a `Vec`/`IndexMap`; counts are `u32`/`u64`; nesting of clusters is recursive. The only limit is a configurable expansion budget (resource guard, default 50 M instances), never a schema limit. |
| P2 | Declare structure, derive physics | Bandwidth, latency, energy, area are derived by kiln-phys from structure (port widths, clocks, wire lengths, node tables). Every derived quantity has an `override` field so published numbers can be pinned. Overrides that exceed what the structure can physically deliver are errors (physical floors, 00-overview). |
| P3 | Real structure, compact text | The IR describes real hierarchy (108 SMs, 4 tensor cores each), but `count`, `layout`, templates and selectors keep the text short. Engines only see expanded instances. |
| P4 | "Mapper decides" is explicit | Every choice the mapper or placer may make is an explicit enum variant (`dataflow: "any"`, carveout options, `placement: "auto"`, `binding: "auto"`), never an absent field with implicit meaning. |
| P5 | Errors are for LLMs | Every rejection carries a code, the entity path, the offending value, what is allowed, and a concrete fix. |
| P6 | One graph | Everything movable-through (memory ports, buses, NoCs, die-to-die, chip-to-chip, switches) becomes channels in one expanded graph with shared-resource annotations. Engines never special-case a link type for routing; only kiln-phys distinguishes link physics. |

## 2. Quantities, units and literals

IR fields carry SI base units (00-overview). To keep LLM-authored designs readable and to prevent unit slips, the
authoring parser accepts either a bare number (base units) or a quantity string with a unit suffix. The canonical
writer always emits bare numbers in base units.

| Rust newtype | Base unit | Accepted suffixes (case-sensitive) | Example |
|---|---|---|---|
| `Bytes` (u64) | B | `B`, `KB`,`MB`,`GB`,`TB` (10^3n), `KiB`,`MiB`,`GiB`,`TiB` (2^10n) | `"40MiB"`, `"16GB"` |
| `Bits` (u64, only in fields named `*_bits`) | bit | `b`, `Kib`, `Mib` | `1024` |
| `Hz` (f64) | Hz | `Hz`,`kHz`,`MHz`,`GHz` | `"1410MHz"` |
| `BytesPerSec` (f64) | B/s | `B/s`,`KB/s`,`MB/s`,`GB/s`,`TB/s`,`GiB/s`,`TiB/s` | `"1555GB/s"` |
| `BitsPerSec` (f64, only in `*_bits_per_s`) | bit/s | `b/s`,`Mb/s`,`Gb/s`,`Gbps` | `"2.43Gbps"` |
| `Seconds` (f64) | s | `s`,`ms`,`us`,`ns`,`ps` | `"12ns"` |
| `Joules` (f64) | J | `J`,`mJ`,`uJ`,`nJ`,`pJ`,`fJ` | `"3.9pJ"` |
| `JoulesPerByte` (f64) | J/B | `pJ/B`, `pJ/b` (converted x8) | `"3.9pJ/b"` |
| `Watts` (f64) | W | `W`,`mW`,`kW` | `"400W"` |
| `Volts` (f64) | V | `V`,`mV` | `"0.75V"` |
| `Um` (f64) | um | `um`,`mm` | `"33mm"` |
| `Mm2` (f64) | mm^2 | `mm2`,`um2` | `"826mm2"` |
| `Cycles` (f64) | cycles of the entity's clock | `cyc` | `"4cyc"` or `4` |

Rules:
- A quantity string with the wrong dimension for the field is `E-IR-0108`. A bare number is always base units.
- `Bytes` must be an integer after conversion (`"1.5KiB"` ok = 1536; `"0.3B"` error).
- Plausibility ranges (e.g. clock 1 MHz..20 GHz) produce `E-IR-0109` with the range in the message. Ranges are
  sanity bounds, not structural caps; they live in `kiln-ir/data/plausible.toml` and can be widened per run with
  `--allow-implausible <code>`.
- `Cycles` fields are converted to seconds using the clock domain of the owning entity after expansion.

```rust
#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Bytes(pub u64);
impl<'de> Deserialize<'de> for Bytes { /* number | "<num><suffix>" | "=expr" (authoring only) */ }
// same pattern for every unit newtype; Serialize always emits the bare base-unit number.
```

## 3. Document structure

A hardware document is one JSON (canonical) or JSON5/RON/YAML (authoring) object.

```rust
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwDoc {
    pub schema: SchemaVersion,                   // "kiln.hw/1.0"  (required)
    pub name: Id,                                // design name (required)
    #[serde(default)] pub meta: Meta,            // excluded from semantic hash
    #[serde(default)] pub extends: Option<ImportRef>, // inherit a whole design, then patch (sec 14.6)
    #[serde(default)] pub imports: IndexMap<Id, ImportRef>,
    #[serde(default)] pub params: IndexMap<Id, ParamValue>,
    #[serde(default)] pub templates: IndexMap<Id, Template>,
    pub tech: TechRef,                           // default process node for every die
    #[serde(default)] pub exec_model: ExecModel,  // host_launched (default) | device_queued | static_dataflow (03 sec 4.4)
    #[serde(default)] pub family: Option<Id>,     // calibration family/platform key (03 sec 9, 06): "a100_40gb", "tpu_v5e" (06 platform keys);
                                                  // hashed; forbidden in profile `search` (E-IR-1102)
    #[serde(default)] pub clocks: Vec<ClockDomain>,   // global domains; dies may add their own
    #[serde(default)] pub power: Vec<PowerDomain>,
    pub system: System,
    #[serde(default)] pub set: Vec<Patch>,        // post-expansion-of-templates patches (sec 14.5)
}

pub struct Meta {
    pub description: String,
    pub authors: Vec<String>,
    pub citations: IndexMap<Id, String>,         // id -> full citation; referenced as "src:<id>" in notes
    pub claims: Vec<Claim>,                      // published figures checked against derived values (W-IR-1801)
    pub notes: IndexMap<Path, String>,           // per-entity provenance notes ("published src:wp" / "assumed: ...")
}

pub struct Claim {
    pub metric: ClaimMetric,      // e.g. "peak_ops.bf16", "offchip_bw", "onchip_bytes.l2", "die_area", "tdp"
    pub scope: Path,              // entity the metric is computed over ("" = whole system)
    pub value: f64,               // base units
    pub rel_tol: f64,             // default 0.02
    pub source: Id,               // citation id
}
```

`exec_model` may be overridden per package (`Package.exec_model`) for heterogeneous systems. `family` scopes
platform-specific calibration residuals: a novel design never sets it, so it receives only technology-scoped or
assumed constants (03 sec 9 rule 6).

`Id` is a validated `String` matching `^[a-z][a-z0-9_-]*$` (dots are reserved as path separators, so ids
themselves contain no dots; this is the authoring-level subset of the 00-overview grammar). `Path` is a dotted
sequence of instance ids, e.g. `board0.chip0.die0.gpc3.tpc1.sm0.smsp2.tc`.

Provenance convention used in examples: every numeric field that is not derived gets a `meta.notes` entry or an
inline JSON5 comment `// pub:<citation>` or `// assumed: <reason>`. The comment form is for humans; tools read
`meta.notes` and `meta.claims`.

## 4. System hierarchy

### 4.1 Levels

```
System
 ├─ hosts[]            (CPU hosts; minimal model for offload and launch paths)
 ├─ boards[]           (node/tray: one or more packages + on-board links/switches)
 │   ├─ packages[]     (one accelerator device as sold: substrate/interposer, dies, memory stacks)
 │   │   ├─ dies[]     (one piece of silicon; chiplets are dies; 3D stacking via layers)
 │   │   │   ├─ clusters[]  (recursive: GPC > TPC > SM > sub-partition; tile; TensorCore)
 │   │   │   │   ├─ units[]      (compute: matrix | vector | scalar | special | cim)
 │   │   │   │   ├─ memories[]   (register files, scratchpads, caches, FIFOs)
 │   │   │   │   ├─ networks[]   (bus, crossbar, NoC local to this scope)
 │   │   │   │   ├─ blocks[]     (area/power-only blocks, DMA engines, controllers, PHYs)
 │   │   │   │   └─ clusters[]   (nested)
 │   │   │   ├─ ports[]     (die I/O: d2d, HBM PHY, SerDes, PCIe)
 │   │   │   └─ (units/memories/networks/blocks directly at die scope also allowed)
 │   │   ├─ mem_stacks[]    (HBM / LPDDR / GDDR / stacked DRAM; may carry a logic die with PIM units)
 │   │   ├─ links[]         (die-to-die, die-to-stack)
 │   │   └─ networks[]      (package-level networks over die ports)
 │   ├─ switches[]     (NVSwitch-like)
 │   └─ networks[]     (chip-to-chip over package ports: ICI, NVLink)
 ├─ switches[]
 └─ networks[]         (board-to-board: optical ICI, InfiniBand/Ethernet DCN)
```

A `Cluster` may appear at die scope or inside another cluster, to any depth. Units, memories, networks and blocks
may appear at any container scope (die or cluster). "Tile" is not a separate type; a tile is a cluster.

Single-chip designs may omit boards and packages using shorthand (sec 14.7): a document whose `system` has a
`package` key is wrapped into a board with id `board`; a `die` key is additionally wrapped into a package with id
`chip`. Paths in such documents therefore start `board.<package id>.`.

### 4.2 Container types

```rust
pub struct System {
    #[serde(default)] pub hosts: Vec<Host>,
    #[serde(default)] pub boards: Vec<Board>,
    #[serde(default)] pub switches: Vec<Switch>,
    #[serde(default)] pub networks: Vec<Network>,
}

pub struct Board {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,         // count, layout, vary, disabled (sec 14.1)
    pub packages: Vec<Package>,
    #[serde(default)] pub switches: Vec<Switch>,
    #[serde(default)] pub networks: Vec<Network>,
    #[serde(default)] pub host_links: Vec<HostLink>,
    #[serde(default)] pub power: Option<PowerCap>,       // inline cap (PowerDomain minus id/members), sec 12
}

pub struct Package {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub dies: Vec<Die>,
    #[serde(default)] pub mem_stacks: Vec<MemStack>,
    #[serde(default)] pub links: Vec<LinkDecl>,      // explicit point-to-point (d2d, die<->stack when no PHY ref)
    #[serde(default)] pub networks: Vec<Network>,
    #[serde(default)] pub ports: Vec<PortExport>,    // package-level names for die ports (optional aliases)
    #[serde(default)] pub substrate: Substrate,      // sec 11
    #[serde(default)] pub layers: Vec<StackLayer>,   // 3D stacking (sec 11.4)
    #[serde(default)] pub address_map: Vec<AddressMap>, // sec 8.4
    #[serde(default)] pub exec_model: Option<ExecModel>,
    #[serde(default)] pub power: Option<PowerCap>,       // inline cap (PowerDomain minus id/members), sec 12
}

pub struct Die {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    #[serde(default)] pub tech: Option<TechRef>,      // default: HwDoc.tech
    #[serde(default)] pub role: DieRole,             // compute | io | memory_base | sram | interposer_active (informational + 04 defaults)
    #[serde(flatten)] pub contents: Contents,
    #[serde(default)] pub ports: Vec<IoPort>,
    #[serde(default)] pub clocks: Vec<ClockDomain>,
    #[serde(default)] pub default_clock: Option<ClockRef>,
    #[serde(default)] pub floorplan: DieFloorplan,   // outline, shoreline sites, keepouts (sec 11)
    #[serde(default)] pub layer: Option<LayerRef>,   // which package stack layer (3D)
    #[serde(default)] pub over: Option<Ref>,         // 3D: the die directly below (placement is relative to it)
    #[serde(default)] pub stitched: bool,            // allow > reticle (stitched/wafer-scale); 04 applies yield model
    #[serde(default)] pub power: PowerOverride,
}

pub struct Cluster {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    #[serde(flatten)] pub contents: Contents,
    #[serde(default)] pub clock: Option<ClockRef>,
    #[serde(default)] pub footprint: Option<Footprint>,  // None => bounding box of children (04)
    #[serde(default)] pub placement: Placement,
    #[serde(default, rename = "use")] pub use_: Option<TemplateUse>, // sec 14.2
}

#[derive(Default)]
pub struct Contents {
    #[serde(default)] pub clusters: Vec<Cluster>,
    #[serde(default)] pub units: Vec<ComputeUnit>,
    #[serde(default)] pub memories: Vec<Memory>,
    #[serde(default)] pub networks: Vec<Network>,
    #[serde(default)] pub blocks: Vec<Block>,
}

pub struct Host {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub mem_capacity: Bytes,
    pub mem_bandwidth: BytesPerSec,
    #[serde(default)] pub launch_overhead: Seconds,   // per kernel/program launch (00-overview runtime term); default 0
}

pub struct HostLink { pub host: Ref, pub to: Selector, pub link: LinkSpec }  // e.g. PCIe Gen4 x16 per chip
```

Every container and leaf also accepts `use`, `with`, `set` (templating, sec 14) and `notes` (string, not hashed).

### 4.3 Blocks

`Block` covers everything placeable that is not compute, memory, or network: controllers, PHYs, DMA engines,
command processors, uncore filler.

```rust
pub struct Block {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub kind: BlockKind,
    #[serde(default)] pub footprint: Option<Footprint>,   // required for kind=misc
    #[serde(default)] pub placement: Placement,
    #[serde(default)] pub clock: Option<ClockRef>,
    #[serde(default)] pub power: PowerOverride,
}

#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockKind {
    MemController(MemControllerSpec),   // sec 8.3
    Phy(PhySpec),                       // sec 10.6
    Dma(DmaSpec),                       // sec 10.8
    Sequencer { issue_overhead: Seconds },  // command processor / instruction fetch; adds per-launch latency
    Misc { power: Option<Watts> },      // area/power filler (uncore, I/O, test, NVDEC...)
}
```

## 5. Compute units

### 5.1 Common shape

```rust
pub struct ComputeUnit {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    #[serde(flatten)] pub kind: ComputeKind,           // "kind": "matrix" | "vector" | "scalar" | "special" | "cim"
    pub precisions: Vec<PrecisionMode>,                // at least one (E-IR-0301)
    #[serde(default)] pub ops: Option<Vec<OpClass>>,   // None => default set for kind (table 6.3)
    #[serde(default)] pub feeds: IndexMap<OperandRole, Feed>, // where each operand comes from / goes to
    #[serde(default)] pub local: Vec<LocalBuffer>,     // in-unit buffers (weight regs, accumulators)
    #[serde(default)] pub pipeline: Pipeline,          // fill/drain/issue latencies
    #[serde(default)] pub near: Option<NearBinding>,   // near-memory placement (sec 9)
    #[serde(default)] pub clock: Option<ClockRef>,     // default: enclosing scope's clock
    #[serde(default)] pub footprint: Option<Footprint>,// None => derived by 04 from kind/geometry/precisions/node
    #[serde(default)] pub placement: Placement,
    #[serde(default)] pub power: PowerOverride,        // per-op energy / leakage overrides
}

#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComputeKind {
    Matrix(MatrixSpec),
    Vector(VectorSpec),
    Scalar(ScalarSpec),
    Special(SpecialSpec),
    Cim(CimSpec),            // in-memory compute macro (sec 9.3)
}
```

Throughput of a unit instance for precision mode `p` and op class `c`:

```
ops_per_s(p, c) = base_ops_per_cycle(kind) * p.rate * class_rate(c) * f_clock(unit)
```

where `base_ops_per_cycle` is MACs/cycle for matrix units (geometry product) and lane-ops/cycle for vector,
scalar and special units. CIM units have no single base: their MACs/cycle are derived per precision mode from the
array (sec 9.3), and `p.rate` only de-rates them. `f_clock` is the effective clock after power-cap throttling (sec 12; computed by 04/03).
A FLOP count is `2 * MACs` for MAC-based classes; reporting conventions are 06's.

### 5.2 Matrix units

```rust
pub struct MatrixSpec {
    pub geometry: Geometry,
    #[serde(default)] pub dataflow: DataflowChoice,      // default per geometry, table below
    #[serde(default)] pub sparsity: Vec<SparsitySupport>,
    #[serde(default)] pub accumulate_in: AccumulateIn,   // where partial sums live between k-steps
    #[serde(default)] pub operand_run: BTreeMap<OperandRole, u32>, // per input role: most consecutive temporal
                                                         // steps the operand stays in the unit between feed reads
                                                         // (an instruction's span; unlisted = unlimited)
}

#[serde(rename_all = "snake_case")]
pub enum Geometry {
    /// rows x cols PEs, one MAC per PE per cycle; reduction runs along rows.
    Systolic { rows: u32, cols: u32 },
    /// Per-cycle MMA tile (tensor-core style): m*n*k MACs per cycle.
    Mma { m: u32, n: u32, k: u32 },
    /// rows x cols outer-product array.
    OuterProduct { rows: u32, cols: u32 },
    /// Generic spatial unrolling over named loop dims (ZigZag "operational array"):
    /// e.g. {"k": 32, "c": 16}. Product = MACs/cycle. Dims use 02's canonical loop-dim names.
    Spatial { dims: IndexMap<LoopDim, u32> },
}

#[serde(untagged)]
pub enum DataflowChoice {
    One(Dataflow),
    Set(Vec<Dataflow>),          // mapper picks per op among these
}
#[serde(rename_all = "snake_case")]
pub enum Dataflow { WeightStationary, OutputStationary, InputStationary, RowStationary, Any }

pub struct SparsitySupport {
    pub pattern: SparsityPattern,  // "2:4", "1:4", "block:<n>", "unstructured"
    pub operand: OperandRole,      // which operand may be sparse
    pub speedup: f64,              // effective rate multiplier when the pattern holds (e.g. 2.0)
    #[serde(default)] pub metadata_bits_per_nz: u32,
}

#[serde(rename_all = "snake_case")]
pub enum AccumulateIn { Local, Feed, Any }  // Local: unit-local accumulators; Feed: via feed 'c'/'o' memory
```

Geometry to loop-dim mapping (consumed by 03's intra-core cost model):

| Geometry | Spatial unrolling (02 loop dims `m` rows-of-A, `n` cols-of-B, `k` reduction) | Default dataflow |
|---|---|---|
| `systolic{rows,cols}` | `k=rows`, `n=cols`; `m` streams temporally | `weight_stationary` |
| `mma{m,n,k}` | `m`, `n`, `k` as given | `output_stationary` |
| `outer_product{rows,cols}` | `m=rows`, `n=cols` | `output_stationary` |
| `spatial{dims}` | as given | `any` |

`any` means the mapper may choose stationarity per op; 04 still prices the hardware for the declared set (a unit
supporting `[ws, os]` pays the area of both operand-holding structures, see 04).

### 5.3 Vector units

```rust
pub struct VectorSpec {
    pub lanes: u32,                   // SIMD width in elements of the 32-bit reference precision
    #[serde(default = "one")] pub sublanes: u32,   // TPU-style (8,128) => sublanes 8, lanes 128
    #[serde(default)] pub class_rates: IndexMap<OpClass, f64>,  // per-class multiplier; default table 6.3
    #[serde(default)] pub reduce_tree: bool,         // cross-lane reduction in 1 pass (default false: log2 passes)
}
```

`base_ops_per_cycle = lanes * sublanes`. Precision `rate` scales it (e.g. A100 FP32 lanes rate 1.0, packed FP16
rate 4.0 per CUDA-guide throughput table).

### 5.4 Scalar and special-function units

```rust
pub struct ScalarSpec {
    #[serde(default = "one")] pub issue_width: u32,
    #[serde(default)] pub controls: Vec<Ref>,       // units this scalar core sequences (informational for 03 launch costs)
}

pub struct SpecialSpec {
    pub lanes: u32,
    pub functions: Vec<SpecialFn>,   // exp, exp2, log, log2, rsqrt, sqrt, recip, sin, cos, tanh, sigmoid, gelu, silu, erf
    #[serde(default)] pub fn_rates: IndexMap<SpecialFn, f64>,  // default 1.0 each
}
```

Special units implement op class `transcendental` for listed functions only; 03 lowers an unlisted function to
vector-unit sequences (02 provides the expansion cost per function).

### 5.5 Operand roles and feeds

Operand roles are fixed names: `a` (left / activations / input), `b` (right / weights), `c` (accumulator in),
`o` (result out), `in` (generic input, vector/special), `out` (generic output), `any` (shorthand for all roles the
unit uses).

```rust
pub struct Feed {
    pub from: Ref,                         // a Memory (or MemStack for near-memory units); sec 14.4 resolution
    #[serde(default)] pub width_bits: Option<u32>,   // per cycle of the unit clock; None => derived need (below)
    #[serde(default)] pub latency: Option<Cycles>,    // None => derived by 04 from distance + memory latency
    #[serde(default)] pub via: Option<Ref>,           // a Network, if the feed is not a dedicated port
}
```

Authoring shorthand: `feeds: { a: "rf", b: "rf", o: "rf" }` (string = `{from: ...}`); `feeds: { any: "vmem" }`.

Semantics:
- A feed is a dedicated port pair between the unit and the memory. It becomes a channel in the expanded graph whose
  bandwidth is `min(feed.width_bits, memory port width) * f_clock / 8`. If `via` is set, the feed uses that
  network's channels and contends with its other traffic.
- Derived width need (when `width_bits` is None): the bits per cycle that keep the geometry busy at its highest-rate
  precision mode, for the role's operand, under the most bandwidth-hungry declared dataflow. 04 does not use this
  for area unless the memory's ports are also derived.
- Every role that some precision mode of the unit uses must have a feed (`E-IR-0304`), except roles satisfied by a
  `local` buffer marked `holds: <role>` with `refill_from` set.
- A unit may read an operand only from memories it has a feed to. Getting data into those memories is a data
  movement the mapper schedules over the graph (sec 16.4).

### 5.6 Local buffers and pipeline

```rust
pub struct LocalBuffer {
    pub id: Id,
    pub holds: OperandRole,
    pub capacity: Bytes,
    #[serde(default)] pub double_buffered: bool,     // capacity is total; usable per phase = capacity/2
    #[serde(default)] pub refill_from: Option<Ref>,  // memory the buffer loads from (default: feed of same role)
}

#[derive(Default)]
pub struct Pipeline {
    #[serde(default)] pub fill: Option<Cycles>,     // systolic fill; None => derived (rows + cols - 1 for systolic)
    #[serde(default)] pub drain: Option<Cycles>,
    #[serde(default)] pub issue_overhead: Cycles,   // per instruction/tile issue, default 0
}
```

Invariant `E-IR-0311`: under weight-stationary dataflow a systolic unit needs a `b` local buffer of at least
`rows * cols * bytes(b)` for the widest `b` precision; output-stationary needs a `c`/`o` local buffer of at least
`rows * cols * bytes(acc)`, unless `accumulate_in = feed`.

## 6. Precisions and op classes

### 6.1 Precision registry

`Precision` is a closed enum in `kiln-ir::precision`, shared with 02 (02 tags tensors with the same names). Adding
a format is a minor schema bump.

| Name | Kind | Element bits | Block / scale | Storage bits per element (incl. scale) |
|---|---|---|---|---|
| `fp64` | float | 64 | none | 64 |
| `fp32` | float | 32 | none | 32 |
| `tf32` | float (19-bit compute, 32-bit storage) | 32 | none | 32 |
| `bf16`, `fp16` | float | 16 | none | 16 |
| `fp8_e4m3`, `fp8_e5m2` | float | 8 | per-tensor scale (02) | 8 |
| `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1` | float | 6/6/4 | per-tensor scale | 6/6/4 |
| `mxfp8_e4m3`, `mxfp8_e5m2` | OCP MX | 8 | 32 elements, E8M0 scale | 8.25 |
| `mxfp6_e3m2`, `mxfp6_e2m3` | OCP MX | 6 | 32, E8M0 | 6.25 |
| `mxfp4` | OCP MX (E2M1) | 4 | 32, E8M0 | 4.25 |
| `mxint8` | OCP MX | 8 | 32, E8M0 | 8.25 |
| `nvfp4` | block float (E2M1) | 4 | 16, E4M3 scale | 4.5 |
| `int32`, `int16`, `int8`, `int4`, `uint8`, `uint4` | integer | 32/16/8/4 | none | same |
| `int64`, `bool` | integer (workload index/mask tensors only) | 64/8 | none | 64/8 |
| `e8m0` | MX shared exponent (scale only, never an operand) | 8 | none | 8 |

Storage variants (02's scale-carrying `ElemType` shorthands, 02 §3.2). Each computes as the named compute precision;
hardware `PrecisionMode`s reference compute names only (`Precision::compute()` maps a variant to it):

| Name | Computes as | Scaling | Storage bits per element (incl. scale) |
|---|---|---|---|
| `fp8_e4m3_pt` | `fp8_e4m3` | per-tensor fp32 scale | 8 (+4 B per tensor) |
| `int8_pc` | `int8` | per-axis (row) fp32 scale | 8 (+4 B per row) |
| `int4_g128` | `int4` | block 128 along K, bf16 scale + int4 zero point | 4.156 |

`nvfp4` also carries a second-level fp32 tensor scale (02 §3.2). 02 owns the `ElemType` expansion of every name
(scaling struct); this table owns the names and `storage_bits()`.

Non-default MX block sizes are written `"mxfp4/16"` (block 16); scale format is fixed E8M0 for `mx*`. Storage bits
including scale overhead are what memories and links move (03 and 02 use `Precision::storage_bits()`).

### 6.2 Precision modes

```rust
#[serde(untagged)]
pub enum PrecisionMode {
    /// matrix / cim
    Mac { a: Precision, b: Precision, acc: Precision,
          #[serde(default)] out: Option<Precision>,  // None => acc
          #[serde(default = "one_f")] rate: f64 },   // MACs/cycle multiplier vs geometry product
    /// vector / scalar / special
    Elem { dtype: Precision, #[serde(default = "one_f")] rate: f64 },
}
```

Shorthands accepted in authoring: `"bf16*bf16+fp32"` (rate 1), `"int8*int8+int32@2"` (rate 2), `"fp32@1"`,
`"bf16@4"`. Mixed-input modes (e.g. `bf16*int4+fp32` for weight-only quantization) are ordinary entries.

Invariants: `acc` must be at least as wide as needed (integer inputs need `int32` or `int16` acc, `E-IR-0303`;
float inputs need float acc with exponent range >= input); `rate > 0`; at most one mode per `(a,b,acc,out)`.
A workload op whose precision has no exactly matching mode may run in a wider mode only if 02 marks the tensor
`upcast_ok` (02 owns that flag); otherwise the unit cannot run it.

### 6.3 Op classes

`OpClass` is shared with 02 (02 owns the op -> class lowering; 01 owns capability declarations).

| OpClass | Meaning | Default on | Default `class_rate` |
|---|---|---|---|
| `matmul` | GEMM / batched GEMM / einsum reducible to GEMM | matrix, cim | 1.0 |
| `conv` | convolution (lowered to matmul by 03 if unit lacks it) | matrix | 1.0 |
| `elementwise` | add, mul, sub, fma, max/min, select, compare | vector, scalar | 1.0 |
| `transcendental` | exp, log, tanh, rsqrt, ... | special; vector at rate 0.125 | 1.0 (special) |
| `reduction` | sum/max/mean along an axis | vector | 1.0 (+ log2 passes unless `reduce_tree`) |
| `convert` | precision cast, MX quantize/dequantize | vector | 1.0 |
| `permute` | transpose, shuffle, gather within registers | vector | 0.5 |
| `gather_scatter` | indexed memory access, embedding lookup | vector, scalar | 0.25 |
| `scan` | prefix ops (sampling, SSM scans) | vector | 0.5 |
| `sort_topk` | sort, top-k | vector | 0.125 |
| `control` | scalar control, address generation | scalar | 1.0 |
| `collective_reduce` | in-network or near-memory reduction | special / switch / PIM | 1.0 |

`softmax`, `layernorm`, `rmsnorm`, `attention` are composites that 02/03 decompose into the classes above (or map
whole to a unit declaring the composite class in a future minor version).

## 7. On-chip memories

```rust
pub struct Memory {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub kind: MemKind,
    pub capacity: Bytes,
    #[serde(default = "word32")] pub word_bits: u32,     // access granularity; multiple of 8, power of two
    #[serde(default = "one")] pub banks: u32,
    #[serde(default)] pub bank_interleave: Option<Bytes>,// default word_bits/8
    pub ports: Vec<PortSpec>,                           // at least one (E-IR-0402)
    #[serde(default)] pub operands: OperandPolicy,
    #[serde(default)] pub cache: Option<CacheSpec>,     // required iff kind = cache
    #[serde(default)] pub backing: Option<Selector>,     // next level(s) (caches: miss path; scratchpads: default fill
                                                         // source); several targets => address map picks per line
    #[serde(default)] pub implementation: MemImpl,       // auto | flop | latch_array | sram_banked | sram_macro | edram | mram (04 sec 13)
    #[serde(default)] pub bitcell: Option<Bitcell>,      // hd | hc | two_port; None => 04 default for implementation
    #[serde(default)] pub power_gated: bool,             // may be gated when idle (04 leakage model)
    #[serde(default)] pub clock: Option<ClockRef>,
    #[serde(default)] pub footprint: Option<Footprint>,
    #[serde(default)] pub placement: Placement,
    #[serde(default)] pub overrides: MemOverrides,
}

#[serde(rename_all = "snake_case")]
pub enum MemKind { RegisterFile, Scratchpad, Cache, Fifo }

pub struct PortSpec {
    pub dir: PortDir,                       // read | write | rw
    #[serde(default = "one")] pub count: u32,
    pub width_bits: u32,                    // per port per cycle of the memory clock
    #[serde(default)] pub per_bank: bool,   // true: count is per bank (multi-banked SRAM)
}

#[derive(Default)]
#[serde(rename_all = "snake_case", tag = "policy")]
pub enum OperandPolicy {
    #[default] Unified,                                     // any operand anywhere
    Partitioned { parts: IndexMap<OperandRole, Bytes> },    // fixed split; sum <= capacity
    Carveout { options: Vec<CarveOption> },                 // mapper picks one option per kernel, e.g. L1 vs SMEM
}

pub struct CarveOption {        // scratch + cache <= capacity (E-IR-0405)
    pub scratch: Bytes,         // software-managed part
    pub cache: Bytes,           // hardware-managed part; requires `cache` spec if any option has cache > 0
}

pub struct CacheSpec {
    pub line: Bytes,
    #[serde(default = "one")] pub sectors: u32,           // fill granularity = line / sectors
    pub ways: u32,
    #[serde(default)] pub write: WritePolicy,             // write_back (default) | write_through
    #[serde(default)] pub allocate: AllocPolicy,          // write_allocate (default) | no_write_allocate
    #[serde(default)] pub replacement: Replacement,       // lru (default) | plru | random | fifo
    #[serde(default)] pub coherent_with: Vec<Ref>,        // peer caches holding copies (e.g. A100 L2 partitions)
    #[serde(default)] pub pinnable: Option<Bytes>,        // max bytes the mapper may pin (persisting/residency control)
}

#[derive(Default)]
pub struct MemOverrides {   // all optional; when set they replace 04's derived value
    pub read_energy: Option<JoulesPerByte>,
    pub write_energy: Option<JoulesPerByte>,
    pub latency: Option<Cycles>,       // load-to-use from this memory's ports
    pub leakage: Option<Watts>,
    pub area: Option<Mm2>,
    pub bandwidth: Option<BytesPerSec>,// total; must be <= derived port bandwidth (E-IR-0409)
}
```

Derived port bandwidth: `sum(ports: count * (per_bank ? banks : 1) * width_bits / 8) * f_clock(memory)`.
Bank conflicts and port contention are modelled in Tier B (03) using `banks`, `bank_interleave`, `ports`.

Sharing is structural: a memory is shared by every unit that has a feed to it and every network it is an endpoint
of. Hierarchy is the graph of `feeds`, `backing`, and network reachability; there is no fixed level numbering. An
optional `level_hint: u8` exists only for visualizer grouping.

Register files are `kind: register_file`; they must be fed by at most the units of one cluster instance unless
`shared_rf: true` is set (`W-IR-0410` otherwise, a likely authoring bug).

## 8. Off-chip memory

### 8.1 Memory stacks / devices

```rust
pub struct MemStack {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub kind: DramKind,
    pub capacity: Bytes,                         // per stack/device
    pub io_width_bits: u32,                      // HBM2/2e/3/3e: 1024; HBM4: 2048; LPDDR5x x16 channel: 16 per channel...
    pub pin_rate_bits_per_s: BitsPerSec,         // per data pin
    #[serde(default)] pub channels: Option<u32>, // None => kind default (HBM2: 8, HBM3: 16, HBM4: 32)
    #[serde(default)] pub pseudo_channels_per_channel: Option<u32>, // HBM2+: 2
    #[serde(default)] pub banks_per_pc: Option<u32>,
    #[serde(default)] pub dies_high: Option<u32>,          // stack height (informational, area/power in 04)
    #[serde(default)] pub row_bytes: Option<Bytes>,
    #[serde(default)] pub timing: Option<DramTiming>,      // None => 04 table for kind + pin rate
    #[serde(default)] pub attach: StackAttach,             // how the stack connects to logic (8.2)
    #[serde(default)] pub logic_die: Option<LogicDie>,     // near-memory compute (sec 9)
    #[serde(default)] pub site: Placement,                 // package site (11.3), or stacked layer (11.4)
    #[serde(default)] pub layer: Option<LayerRef>,         // StackedDram only
    #[serde(default)] pub over: Option<Ref>,               // StackedDram only: die below/above
    #[serde(default)] pub clock: Option<ClockRef>,         // I/O clock domain (informational; bw from pin rate)
    #[serde(default)] pub footprint: Option<Footprint>,    // None => 04 per kind (HBM ~ 11 x 10 mm)
    #[serde(default)] pub overrides: StackOverrides,       // bandwidth, energy_per_byte, latency, power
}

#[serde(rename_all = "snake_case")]
pub enum DramKind { Hbm2, Hbm2e, Hbm3, Hbm3e, Hbm4, Hbm4e, Lpddr4x, Lpddr5, Lpddr5x, Lpddr6,
                    Gddr6, Gddr6x, Gddr7, Ddr5, StackedDram /* DRAM-on-logic, 3D hybrid-bonded */, Custom }

pub struct DramTiming {   // seconds; any subset; missing => 04 defaults
    pub t_rcd: Option<Seconds>, pub t_rp: Option<Seconds>, pub t_cl: Option<Seconds>, pub t_ras: Option<Seconds>,
    pub t_rrd: Option<Seconds>, pub t_faw: Option<Seconds>, pub t_rfc: Option<Seconds>, pub t_refi: Option<Seconds>,
    pub t_ccd_l: Option<Seconds>, pub t_ccd_s: Option<Seconds>,
}
```

Derived peak bandwidth per stack: `io_width_bits * pin_rate_bits_per_s / 8`. Achievable fraction (refresh,
turnaround, row misses) is a 04 calibration constant per kind, not an IR field. `overrides.bandwidth` must not exceed
the derived peak (`E-IR-0506`).

`StackedDram` (DRAM-on-logic) has no PHY: it is placed on a stack layer above or below a die (11.4) and its
interface width is derived from the bond pitch and overlap area (04) unless `io_width_bits` is given.

### 8.2 Stack attachment

```rust
#[serde(untagged)]
pub enum StackAttach {
    /// Explicit: names the die PHY ports (sec 10.6) this stack's channels terminate on.
    Phys { phys: Vec<Ref> },
    /// Shorthand: names on-die controllers; kiln synthesizes PHY blocks on the die edge facing the stack.
    Controllers { controllers: Vec<Ref> },
    /// Shorthand: a network; kiln synthesizes one controller (endpoint of that network) + PHY per stack.
    Network { network: Ref },
    /// Stacked DRAM: vertical interface to the die directly below/above.
    Vertical { die: Ref },
}
```

Synthesized controllers/PHYs appear in the expanded model with ids `<stack>_mc<i>` / `<stack>_phy` and are reported
by `kiln expand` so the author can pin them later.

### 8.3 Memory controllers

```rust
pub struct MemControllerSpec {
    pub serves: Ref,                          // stack (or stack.pc<i> range via selector)
    #[serde(default)] pub channels: Option<u32>, // channels of the stack this controller owns
    #[serde(default)] pub width_bits: Option<u32>,
    #[serde(default)] pub queue_depth: Option<u32>,
    #[serde(default)] pub scheduler: DramScheduler,  // fr_fcfs (default) | fcfs
    pub endpoint_of: Vec<Ref>,                // networks this controller is attached to
}
```

### 8.4 Address maps

Physical address placement determines which controller/partition serves a byte, which matters for the A100 L2
partitions and NUMA-like chiplet designs.

```rust
pub struct AddressMap {
    pub id: Id,
    pub targets: Selector,                  // controllers or stacks, in interleave order
    pub granule: Bytes,                     // interleave granularity
    #[serde(default)] pub hash: InterleaveHash,  // linear (default) | xor_fold
}
```

The mapper (03) allocates tensors in an address map and may choose among multiple maps (e.g. a "local" map per
chiplet and a "global" interleaved map). Default when absent: one map interleaving all stacks of the package at
4 KiB granule (assumed default).

## 9. Near-memory and in-memory compute

Three placements are first-class:

| Placement | Example | IR form |
|---|---|---|
| Logic die of a DRAM stack (PIM) | Samsung HBM-PIM / Aquabolt-XL, HBM4 custom base die | `MemStack.logic_die.units[]` with `near` binding to the stack |
| Per-bank / near-bank in DRAM | UPMEM, AiM | same as above with `granularity: per_bank` |
| On-chip SRAM near-memory | compute attached to one SRAM bank group | `ComputeUnit.near` binding to a `Memory` |
| In-SRAM compute (CIM) | digital/analog SRAM-CIM macro | `ComputeUnit { kind: cim }` (9.3) |

### 9.1 Binding

```rust
pub struct NearBinding {
    pub memory: Ref,                        // Memory or MemStack
    pub granularity: NearGranularity,       // per_instance | per_bank | per_bank_group | per_pseudo_channel | per_channel | per_stack
    #[serde(default)] pub internal_bandwidth: Option<BytesPerSec>, // per unit instance; None => derived (04: row-buffer width / tCCD for DRAM, port width for SRAM)
    #[serde(default)] pub access_mode: NearAccessMode,
    #[serde(default)] pub residency: NearResidency,
    #[serde(default)] pub command_latency: Option<Seconds>,  // host -> PIM command issue latency; None => 04 default
    #[serde(default)] pub result_path: ResultPath,
}

#[serde(rename_all = "snake_case")]
pub enum NearAccessMode {
    Concurrent,          // host accesses and near compute share bandwidth (arbitrated)
    ExclusiveAllBank,    // while near compute runs, the memory is unavailable to the host (HBM-PIM all-bank mode)
    ExclusivePerBank,    // only the banks in use are blocked
}

#[serde(rename_all = "snake_case")]
pub enum NearResidency {
    AllOperands,         // every operand must already reside in the bound memory slice
    Weights,             // role b must reside; a may be broadcast in over the channel (GEMV style, default)
}

#[serde(rename_all = "snake_case")]
pub enum ResultPath { WriteBack /* results land in bound memory */, Channel /* results stream to host over the memory channel */ }
```

Count rule: a near unit with omitted `count` gets one instance per granule of each bound memory instance (e.g. a
PIM unit with `per_pseudo_channel` on 5 stacks x 16 PCs = 80 instances). An explicit `count` must equal that
number (`E-IR-0607`).

### 9.2 How the mapper sees it (contract with 03)

- A near unit is an ordinary `ComputeUnit` node in the expanded graph. Its feeds are implicit: every role reads from
  and writes to its bound memory slice over a private channel of `internal_bandwidth`. Explicit `feeds` naming any
  other memory are an error (`E-IR-0603`).
- Its bound slice is addressable: data placed by the address map (8.4) in that bank/PC/stack is "resident" for it.
  03 must place `residency`-required operands accordingly or pay the move.
- Broadcast-in operands (`residency: weights`) arrive over the memory channel from the host side controller and
  consume normal channel bandwidth; `ExclusiveAllBank` makes that channel unavailable for unrelated traffic for the
  duration.
- Launch costs `command_latency` per kernel per granule group.

### 9.3 In-memory (CIM) units

```rust
pub struct CimSpec {
    pub rows: u32,                      // word lines (reduction depth of one column)
    pub cols: u32,                      // bit-cell columns (bit lines), not output channels
    #[serde(default = "one")] pub cell_bits: u32,            // bits stored per cell: 1 for 6T/8T SRAM; 1..=8
    #[serde(default = "one")] pub input_bits_per_cycle: u32, // activation bits applied per cycle: 1 = bit-serial
    #[serde(default)] pub parallel_rows: Option<u32>,        // rows summed per cycle; None => rows (digital); analog: ADC-bound
    #[serde(default = "one")] pub weight_sets: u32,          // stored weight sets per compute cell (cells muxed onto one multiplier)
    #[serde(default)] pub weight_capacity: Option<Bytes>,    // None => derived (below); if given must equal it (E-IR-0606)
    #[serde(default)] pub style: CimStyle,     // digital (default) | analog
    #[serde(default)] pub adc_bits: Option<u32>,  // analog only (required there, E-IR-0609)
    #[serde(default)] pub weight_write: Option<BytesPerSec>, // reprogramming bandwidth; None => derived from SRAM write ports
}
```

A CIM unit is a weight-stationary matrix unit whose `b` operand is held in the array itself (no `b` feed needed,
weight reload priced at `weight_write`). Its precisions list declares e.g. `int8*int4+int32`. CIM can also take a
`near` binding to an adjacent SRAM for activations.

Published SRAM-CIM macros are not `rows x cols` MACs per cycle. Digital macros are bit-serial in the activation
and store each multi-bit weight across several 1-bit cell columns (TSMC ISSCC 2021 16.4, 22 nm: 64 Kb, 256 inputs,
16 four-bit weight slices, 1-8 b inputs streamed serially, weights of 4-16 b built from 4 b nibbles); later macros
apply several input bits per cycle (TSMC ISSCC 2024, 3 nm: "parallel-MAC" INT12 x INT12). Analog macros decompose a
P-bit x Q-bit product into P x Q binary MAC cycles, or P/y x Q with y-bit DAC inputs, and an ADC reading a sum of
`R` rows of `y`-bit inputs needs `ceil(log2(R * (2^y - 1) + 1))` bits to be lossless (ASiM, arXiv 2411.11022: 256
rows x 4 b inputs span 0..3840, 12 b). kiln therefore derives CIM throughput and capacity:

```
cells_per_weight(m) = ceil(w_bits(m) / cell_bits)                  // w_bits = element bits of operand b
MACs/cycle(m)       = parallel_rows * floor(cols / cells_per_weight(m)) / ceil(in_bits(m) / input_bits_per_cycle)
                      * m.rate                                       // m.rate <= 1: de-rate only (E-IR-0608)
weight_capacity     = rows * cols * cell_bits * weight_sets / 8 bytes
```

Example (ember, 256 x 256, 1-bit cells, bit-serial): int8 x int4 = 256 * 64 / 8 = 2048 MACs/cycle, int8 x int8 =
1024; capacity 8 KiB per weight set. Both are per precision mode with no hand-written `@rate`: a mode's `@rate`
above 1 asserts throughput the array does not have and is unpriced (E-IR-0608, every profile). Every new field
(`cell_bits`, `input_bits_per_cycle`, `parallel_rows`, `weight_sets`, `adc_bits`) is priced by 04 §4.11, so
raising throughput through them costs area and energy in proportion; an explicit `weight_write` is an unpriced
override under `search` (E-IR-1101).

Analog note: `parallel_rows` is bounded by the ADC: below `ceil(log2(parallel_rows * (2^cell_bits - 1) *
(2^input_bits_per_cycle - 1) + 1))` bits the readout is lossy. kiln warns (W-IR-0610) and does not model the
accuracy loss (open question 12); profile `search` rejects it (E-IR-1107).

### 9.4 Logic dies on stacks

```rust
pub struct LogicDie {
    #[serde(default)] pub tech: Option<TechRef>,          // DRAM-process logic is slow; 04 tables "dram_logic_1y", "tsmc_n12"...
    #[serde(default)] pub area_budget: Option<Mm2>,        // E-IR-0605 if derived area of units exceeds it
    #[serde(default)] pub power_budget: Option<Watts>,
    #[serde(flatten)] pub contents: Contents,             // units with near bindings, small memories, networks
}
```

## 10. Interconnect

### 10.1 Networks

A `Network` declared at any scope connects endpoints within that scope's subtree. Endpoints are memories, units
(only for networks the unit's feeds name via `via`), blocks (controllers, DMA, PHYs), die ports, package ports,
switches and hosts.

```rust
pub struct Network {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,           // e.g. one per partition
    pub topology: Topology,
    pub endpoints: Vec<EndpointBinding>,
    #[serde(default)] pub router: RouterSpec,
    pub link: LinkSpec,                               // default link for every edge
    #[serde(default)] pub routing: Routing,
    #[serde(default)] pub features: NetFeatures,
    #[serde(default)] pub flit_bits: Option<u32>,     // None => link.width_bits
    #[serde(default)] pub clock: Option<ClockRef>,
}

#[serde(rename_all = "snake_case", tag = "type")]
pub enum Topology {
    /// One shared medium; all endpoints contend (bandwidth = link).
    Bus { #[serde(default)] arbitration: Arbitration },
    /// Non-blocking between distinct (input, output) pairs; per-port bandwidth = link; contention at outputs.
    Crossbar { #[serde(default)] speedup: Option<f64> },
    /// Point-to-point between exactly two endpoint instances.
    P2p,
    Ring { #[serde(default = "t")] bidirectional: bool, #[serde(default = "one")] rings: u32 },
    Mesh { dims: Vec<u32> },                             // 1D..nD; product = router count
    Torus { dims: Vec<u32>, #[serde(default)] wrap: Option<Vec<bool>> }, // wrap default all true
    Tree { arity: u32, levels: u32 },
    FatTree { arity: u32, levels: u32, #[serde(default)] taper: Option<f64> },
    Star { center: Ref },                                // all endpoints to one switch/router
    /// Networks of networks: each level's endpoints include the lower levels' gateways.
    Hierarchical { levels: Vec<Ref>, gateways: Vec<GatewaySpec> },
    /// Arbitrary graph: routers numbered 0..routers; edges may override the link.
    Custom { routers: u32, edges: Vec<CustomEdge> },
}

pub struct CustomEdge { pub a: u32, pub b: u32, #[serde(default)] pub link: Option<LinkSpec>, #[serde(default = "one")] pub count: u32 }

pub struct EndpointBinding {
    pub select: Selector,                    // e.g. "tile*.sram", "gpc[0..3].tpc*.sm*.l1"
    #[serde(default)] pub at: RouterBinding, // how instances map to routers
    #[serde(default)] pub port: Option<LinkSpec>, // endpoint injection/ejection link; default = network link
    #[serde(default)] pub multiplicity: Option<u32>, // ports per endpoint instance, default 1
    #[serde(default)] pub ports: Option<Selector>,   // direct networks: port instances of the selected entity that
                                                     // carry this network's links (e.g. "die.ici*"), see below
}

#[serde(rename_all = "snake_case", untagged)]
pub enum RouterBinding {
    Auto,                        // placer/mapper assigns (default for mesh/torus: by floorplan proximity)
    Layout,                      // instance layout coords -> router coords (grid-aligned arrays)
    Index(Vec<Vec<u32>>),        // explicit router coordinate per instance, in expansion order
    Concentrated { per_router: u32 }, // c endpoints share one router, in expansion order
    Fixed { router: Vec<u32> },       // every selected instance attaches to this router (e.g. one crossbar of a 2-router fabric)
    LayoutOffset { offset: Vec<i32> },// layout coords + offset -> router coords (a second array beside the first)
}
```

Direct networks (mesh, torus, ring, custom) whose endpoints are packages or dies with `ports`: each selected
entity *is* a router (its on-die router, e.g. the TPU ICI router), and its port instances are assigned to the
router's existing links in dimension order (+d0, -d0, +d1, -d1, ...; missing links at mesh edges are skipped). The
number of port instances must be >= the router's link count (`E-IR-0710`); extra ports stay unused (`W-IR-0726`).

Size invariant: for mesh/torus, `product(dims) * concentration >= #endpoint instances` (`E-IR-0703`). Tori with a
dimension of size 2 and `wrap = true` expand to two parallel links between the pair (double bandwidth), matching
how 2-wide torus dimensions are cabled; size-1 dimensions have no links.

### 10.2 Routers

```rust
#[derive(Default)]
pub struct RouterSpec {
    pub radix: Option<u32>,                  // None => derived from topology; explicit smaller value => E-IR-0704
    pub pipeline: Option<Cycles>,            // per-hop router latency, default 2 (assumed default, 04 may refine)
    pub input_buffer_flits: Option<u32>,     // default 4
    pub vcs: Option<u32>,                    // default 1 (mesh) / 2 (torus, ring: dateline)
    pub footprint: Option<Footprint>,
    pub power: PowerOverride,
}
```

### 10.3 Links

```rust
pub struct LinkSpec {
    #[serde(default)] pub width_bits: Option<u32>,   // per direction per cycle; required unless phys gives lanes*rate
    #[serde(default)] pub clock: Option<ClockRef>,   // default: network clock
    #[serde(default = "t")] pub full_duplex: bool,   // true: width_bits each direction
    #[serde(default = "one")] pub count: u32,        // parallel links per edge
    #[serde(default)] pub phys: LinkPhys,
    #[serde(default)] pub latency: Option<Seconds>,  // None => derived (04: wire model from placed distance, or PHY latency)
    #[serde(default)] pub energy: Option<JoulesPerByte>, // None => derived
    #[serde(default)] pub bandwidth: Option<BytesPerSec>, // per direction; override, <= derived (E-IR-0712)
}

#[derive(Default)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum LinkPhys {
    #[default] OnDie { #[serde(default)] metal: MetalClass /* auto|local|intermediate|global: class preference */,
                       #[serde(default)] repeated: Option<bool>,     // None => 04 sizing decides
                       #[serde(default)] swing: Swing,               // full (default) | low
                       #[serde(default)] pipelined: Option<bool>,    // None => 04 inserts flops if wire delay > cycle
                       #[serde(default)] sizing: Option<WireSizing> }, // delay | energy | custom{k_h,k_s}; None => energy (04 sec 7.1)
    D2d(D2dSpec),
    Serdes(SerdesSpec),
    Vertical(VerticalSpec),      // 3D bond between stacked dies (11.4)
    Optical(OpticalSpec),        // board/system level (OCS-style ICI, co-packaged optics)
}

pub struct D2dSpec {
    pub standard: D2dStandard,         // ucie_standard | ucie_advanced | bow | aib | custom
    pub modules: u32,                  // UCIe modules (lanes per module: std 16, adv 64 unless lanes_per_module)
    #[serde(default)] pub lanes_per_module: Option<u32>,
    pub pin_rate_bits_per_s: BitsPerSec,
    #[serde(default)] pub bump_pitch_um: Option<Um>,
    #[serde(default)] pub reach: Option<Um>,
}

pub struct SerdesSpec {
    pub protocol: SerdesProtocol,      // nvlink_like | ici_like | pcie | ethernet | infiniband | custom
    pub lanes: u32,                    // per direction
    pub lane_rate_bits_per_s: BitsPerSec,
    #[serde(default)] pub encoding_efficiency: Option<f64>, // e.g. PCIe Gen4 128b/130b; default 04 per protocol
    #[serde(default)] pub fec_latency: Option<Seconds>,
}

pub struct VerticalSpec { pub bond: BondKind, pub pitch_um: Um, #[serde(default)] pub signals: Option<u32> }
pub struct OpticalSpec { pub lanes: u32, pub lane_rate_bits_per_s: BitsPerSec, #[serde(default)] pub switch_latency: Option<Seconds> }
```

Derived link bandwidth per direction: OnDie and Vertical: `width_bits * f / 8` (Vertical width may be derived
from pitch); D2d: `modules * lanes_per_module * pin_rate / 8`; Serdes: `lanes * lane_rate * encoding_eff / 8`;
Optical: `lanes * lane_rate / 8`. `width_bits` on non-OnDie links is optional and, if given, must agree within 1%
with the PHY-derived value (`E-IR-0713`).

### 10.4 Routing and features

```rust
#[derive(Default)]
#[serde(rename_all = "snake_case")]
pub enum Routing { #[default] Default /* xy for mesh, dimension-order+dateline for torus, shortest for others */,
                   DimensionOrder { order: Vec<u32> }, MinimalAdaptive, Table { routes: Vec<RouteEntry> } }

#[derive(Default)]
pub struct NetFeatures {
    pub multicast: bool,                       // one injection, many ejections along a tree
    pub broadcast: bool,                       // to all endpoints
    pub in_network_reduce: Vec<Precision>,     // empty = none; switches/routers that sum in-flight (SHARP-like)
    pub ordered: bool,                         // point-to-point ordering guarantee (informational)
}
```

Routing is deterministic: `HwModel::route(src, dst, net)` returns the same channel list for the same inputs. For
`MinimalAdaptive`, 03 may choose among `HwModel::minimal_paths()`; Tier A uses an even split.

### 10.5 Die ports, die-to-die, chip-to-chip

```rust
pub struct IoPort {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,      // e.g. 12 NVLink ports
    pub kind: IoKind,                           // d2d | hbm | serdes | pcie | lpddr | optical | vertical
    pub phy: Option<Ref>,                       // Phy block on the die; None => synthesized at an auto shoreline site
    pub internal: Ref,                          // on-die endpoint: a memory, block, or network (auto-bound as endpoint)
    #[serde(default)] pub link: Option<LinkSpec>, // default link for networks/links that bind this port
}

pub struct LinkDecl {           // package/board-level explicit link between two ports
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub a: Ref, pub b: Ref,
    pub link: LinkSpec,
}

pub struct PortExport { pub id: Id, pub from: Ref }   // alias a die port at package scope ("ici0" -> "die0.ici0")
```

Chip-to-chip networks are ordinary `Network`s at board or system scope whose endpoints are package ports (or die
ports via path), with `link.phys = serdes | optical`. A chip with 4 ICI ports in a 2D torus is
`endpoints: [{select: "chip*.ici", multiplicity: 4, at: "layout"}]` (example 20.4).

### 10.6 PHY blocks

```rust
pub struct PhySpec {
    pub for_kind: IoKind,
    #[serde(default)] pub lanes: Option<u32>,
    #[serde(default)] pub shoreline: Option<Um>,   // die-edge length consumed; None => derived by 04 per kind/lanes/node
    #[serde(default)] pub site: Option<Ref>,       // named shoreline site (11.2); None => placer picks an edge
}
```

### 10.7 Switches

```rust
pub struct Switch {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub radix: u32,
    pub port: LinkSpec,
    #[serde(default)] pub latency: Option<Seconds>,     // port-to-port
    #[serde(default)] pub in_network_reduce: Vec<Precision>,
    #[serde(default)] pub reduce_bandwidth: Option<BytesPerSec>,
    #[serde(default)] pub power: Option<Watts>,
}
```

A switch is an endpoint (with `multiplicity = radix`) in networks; `Star { center: switch }` and `FatTree` use them
as internal routers.

### 10.8 DMA engines

```rust
pub struct DmaSpec {
    pub endpoint_of: Vec<Ref>,            // networks it injects into
    #[serde(default)] pub bandwidth: Option<BytesPerSec>,
    #[serde(default)] pub outstanding: Option<u32>,
    #[serde(default)] pub scope: Option<Selector>, // memories it may move between; None => all reachable
}
```

If a design declares no DMA blocks, every memory-to-memory move is initiated implicitly (unlimited initiators,
bandwidth bounded only by channels). Declaring DMA blocks makes initiators a contended resource in Tier B.

## 11. Floorplan data

Formulas (area from contents, placement, wire delay) are 04's. 01 defines what is placeable and what an author may
pin.

### 11.1 Footprints and placement

Placeable entities: dies, mem stacks, clusters, compute units, memories, blocks, routers (via `RouterSpec`).

```rust
pub struct Footprint {
    #[serde(default)] pub area: Option<Mm2>,          // either area (placer picks aspect within range) ...
    #[serde(default)] pub w: Option<Um>,              // ... or explicit w x h
    #[serde(default)] pub h: Option<Um>,
    #[serde(default)] pub aspect: Option<(f64, f64)>, // allowed w/h range with area; default (0.5, 2.0)
    #[serde(default)] pub utilization: Option<f64>,   // placement density for derived areas; default 04
}
// None everywhere => 04 derives area from contents (arrays, SRAM macros, routers, PHYs) at the die's node.

#[derive(Default)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum Placement {
    #[default] Auto,                                               // placer decides
    Pinned { x: Um, y: Um, #[serde(default)] rot: Rotation },      // lower-left corner, parent-relative
    Region { x0: Um, y0: Um, x1: Um, y1: Um },                      // placer chooses within box
    Edge { edge: Edge, #[serde(default)] offset: Option<Um> },     // n | e | s | w of the parent
    Array,                                                         // replicated instances tile as their layout grid
    Site { site: Ref },                                            // a named shoreline/package site
}
#[serde(rename_all = "snake_case")] pub enum Rotation { #[default] R0, R90, R180, R270 }
```

Coordinates are parent-relative um with origin at the parent's lower-left. `Array` placement with a `grid` layout
packs instances row-major at pitch = instance footprint (+ `layout.gap`), giving the systolic/tile arrays their
real shape so wire lengths are honest.

### 11.2 Die floorplan

```rust
#[derive(Default)]
pub struct DieFloorplan {
    #[serde(default)] pub outline: Outline,
    #[serde(default)] pub shoreline: Vec<ShorelineSite>,
    #[serde(default)] pub keepouts: Vec<Rect>,
    #[serde(default)] pub reticle_limit: Option<Mm2>,  // default from 04 node table (858 mm^2 = 26 x 33 mm)
}

#[derive(Default)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Outline {
    #[default] Auto,                                      // 04 sizes die = placed area / utilization
    Fixed { w: Um, h: Um },
    MaxArea { area: Mm2, #[serde(default)] aspect: Option<(f64, f64)> },
}

pub struct ShorelineSite {
    pub id: Id,
    #[serde(flatten)] pub rep: Replication,
    pub edge: Edge,
    #[serde(default)] pub offset: Option<Um>,            // along the edge, from the edge's start (ccw)
    #[serde(default)] pub length: Option<Um>,            // None => derived from the PHY bound to it
    pub kind: IoKind,
}
```

### 11.3 Package floorplan

```rust
#[derive(Default)]
pub struct Substrate {
    #[serde(default)] pub kind: SubstrateKind,  // organic | silicon_interposer (CoWoS-S-like) | rdl_interposer (CoWoS-R/InFO) | bridge (EMIB/LSI) | none
    #[serde(default)] pub outline: Outline,     // interposer size; Auto => bounding box + margin (04)
    #[serde(default)] pub sites: Vec<PackageSite>,  // named positions for dies / stacks
}
pub struct PackageSite { pub id: Id, #[serde(flatten)] pub rep: Replication, pub x: Um, pub y: Um, #[serde(default)] pub rot: Rotation }
```

Dies and stacks use `placement` (`Auto`, `Pinned`, `Site`) relative to the substrate. The die-to-stack distance
from placement feeds 04's HBM channel-length check; die-to-die distance feeds D2D reach checks.

### 11.4 3D stacking

```rust
pub struct StackLayer {
    pub id: Id,
    pub index: i32,                  // 0 = base (bottom); positive = above
    pub bond: BondKind,              // hybrid | microbump | tsv
    pub pitch_um: Um,
}
```

A die (or `StackedDram` mem stack) with `layer: <id>` sits on that layer, above the die named by `over` (required
for index > 0, `E-IR-0809`). Its `placement` is relative to that die. Vertical links (`LinkPhys::Vertical`) are only legal between entities on adjacent layers with
overlapping footprints (`E-IR-0810`). Upper dies must fit within the footprint of the die(s) under them unless the
layer's bond is `microbump` on an interposer (`E-IR-0809`).

## 12. Clocks, voltage, power caps, harvesting

```rust
pub struct ClockDomain {
    pub id: Id,
    pub freq: Hz,                                 // nominal (boost) operating point
    #[serde(default)] pub base: Option<Hz>,       // guaranteed sustained floor
    #[serde(default)] pub voltage: Option<Volts>, // at freq; None => 04 node nominal
    #[serde(default)] pub vf: Vec<VfPoint>,       // DVFS curve, ascending freq; empty => fixed frequency
    #[serde(default)] pub crossing_latency: Option<Cycles>, // added on channels crossing into this domain
}
pub struct VfPoint { pub freq: Hz, pub voltage: Volts }

pub struct PowerDomain {
    pub id: Id,
    pub members: Selector,                        // dies / packages / boards
    pub cap: Watts,                               // board/package power limit (TDP / TGP)
    #[serde(default)] pub policy: PowerPolicy,    // dvfs (default) | fixed | duty_cycle
    #[serde(default)] pub clocks: Vec<ClockRef>,  // domains throttled by this cap (default: all clocks of members)
    #[serde(default)] pub idle: Option<Watts>,    // static platform power counted against the cap (fans, VRM loss)
    #[serde(default)] pub thermal: Option<ThermalSpec>, // tj_max, theta_ja; used by 04's thermal check
    #[serde(default)] pub assumed: Option<CapRange>,    // unpublished cap: plausible range; never throttles (04 sec 8)
}
pub struct CapRange { pub lo: Watts, pub hi: Watts, #[serde(default)] pub basis: Option<String> } // 0 < lo <= cap <= hi
```

```rust
pub struct ThermalSpec {
    pub tj_max_c: f64,                       // deg C (unit in the field name; temperatures are not SI-base in 00)
    #[serde(default)] pub cooling: CoolingClass,  // air | air_hp | liquid_cold_plate | immersion (04 table)
    #[serde(default)] pub theta_ja: Option<f64>,  // K/W override
}

pub struct PowerCap { pub cap: Watts, #[serde(default)] pub policy: PowerPolicy,
                      #[serde(default)] pub clocks: Vec<ClockRef>, #[serde(default)] pub idle: Option<Watts>,
                      #[serde(default)] pub thermal: Option<ThermalSpec>,
                      #[serde(default)] pub level: CapLevel /* board | package | die */,
                      #[serde(default)] pub assumed: Option<CapRange> }

/// Per-entity cost overrides. Allowed only in profile `reference` and only with a citation (sec 18.3).
#[derive(Default)]
pub struct PowerOverride {
    pub area: Option<Mm2>,
    pub energy_per_op: IndexMap<String /* precision-mode shorthand or op class */, Joules>,
    pub leakage: Option<Watts>,
    pub ctrl_ge: Option<f64>,                // control-logic gate equivalents (04 area model input)
    pub source: Option<Id>,                  // citation id; required when any field is set
}
```

Semantics (computed by 04, consumed by 03): for each workload phase, power = dynamic(f, V) + leakage(V) + idle. If
power exceeds `cap` under `dvfs`, frequency steps down the `vf` curve (never below `base`, else error at
simulation time: `E-MAP-POWER-CAP`, owned by 03) until it fits; Tier A applies this per steady-state phase (03 sec 4.5), Tier B per time window. Under
`fixed`, exceeding the cap is a simulation error. This is how the A100 example reproduces the observed sustained
clock near 1290 MHz on large GEMMs at the 400 W cap (example 20.1); the measured value is a calibration target
(06), not an IR field.

Harvesting (yield): any `Replication` may list `disabled` selectors. Disabled instances keep their footprint (they
are still on the die) and contribute leakage only if `harvest_power: leak` (default `gated`: 0 W); they are absent
from the mapping graph. Networks skip them; routers of disabled tiles remain if `keep_routers: true` (default true
for mesh/torus, so routing stays regular).

## 13. Technology reference

```rust
#[serde(untagged)]
pub enum TechRef {
    Name(String),                          // "tsmc_n7", "tsmc_n5", "tsmc_n4p", "tsmc_n3e", "samsung_sf4", "dram_1b", ...
    Detailed { node: String, #[serde(default)] variant: Option<String>,   // "hp" | "hd" | "lp"
               #[serde(default)] vdd: Option<Volts>, #[serde(default)] metal_stack: Option<String> },
}
```

Names resolve against 04's versioned technology table (`kiln-phys/data/tech/*.toml`). Unknown names are
`E-IR-1001` with the list of known names. The table version contributes to the calibration-set hash, not to the
design hash. Mixed nodes are allowed per die and per logic die.

## 14. Compact authoring

### 14.1 Replication

Every entity accepts these fields (flattened):

```rust
#[derive(Default)]
pub struct Replication {
    #[serde(default)] pub count: Option<CountExpr>,   // None => 1; layout may imply count
    #[serde(default)] pub layout: Layout,
    #[serde(default)] pub vary: Vec<Variation>,       // per-instance patches
    #[serde(default)] pub disabled: Vec<Selector>,    // harvested instances (sec 12)
    #[serde(default)] pub harvest_power: HarvestPower,
}

#[derive(Default)]
#[serde(untagged)]
pub enum Layout {
    #[default] Linear,
    Grid { grid: Vec<u32>, #[serde(default)] order: GridOrder /* row_major | col_major | snake */,
           #[serde(default)] gap: Option<Um> },
    Ring { ring: u32 },
    Explicit { coords: Vec<Vec<i32>> },
}

pub struct Variation { pub select: Selector, pub set: IndexMap<FieldPath, serde_json::Value> }
```

`count` and `layout.grid` must agree (`product(grid) == count`, or count omitted). Layout coordinates are
attached to each instance (`inst.coord`) for `RouterBinding::Layout` and `Placement::Array`.

### 14.2 Templates

```rust
pub struct Template {
    pub kind: TemplateKind,                     // cluster | unit | memory | network | block | die | package | board | mem_stack
    #[serde(default)] pub params: IndexMap<Id, ParamDecl>,
    pub body: serde_json::Value,                // an entity of `kind`, may contain "=expr" strings using params
}
pub struct ParamDecl { #[serde(default)] pub default: Option<ParamValue>, #[serde(default)] pub doc: Option<String> }

pub struct TemplateUse { pub template: Ref /* "sm" or "lib.tpu_tc" */, }
```

Use site: any entity object may contain `"use": "<template>"`, `"with": {param: value}` and its own fields. Merge
order: template body (with params substituted) < fields written at the use site < `set`. The use site's `id`
always wins. Templates may use other templates (cycle => `E-IR-0208`).

### 14.3 Params and expressions

- `params` at document level and `with` at use sites bind names. Scopes are lexical: a template body sees its own
  params, then document params.
- A string beginning with `=` is an expression: `"= n_mxu * 4"`, `"= 2 * l2_slice_cap"`, `"= vmem_bw / 2"`.
- Grammar: numbers, quantity literals (`40MiB`, `1.41GHz`), param names, `+ - * / % ^`, parentheses, comparison,
  `if(cond, a, b)`, functions `min max ceil floor round sqrt log2 pow abs`. Result dimension is checked against the
  field (`1.41GHz * 4` is a Hz; assigning it to `capacity` is `E-IR-0108`). No loops, no recursion, no strings
  except quantity literals; evaluation is total and deterministic (f64, IEEE round-to-nearest; integer fields
  require an exact integer result, else `E-IR-0204`).
- Params may be quantities: `params: { clk: "1410MHz" }`, or any JSON value. An expression consisting of a single
  param name (`"=hbm_kind"`) substitutes the value verbatim, whatever its type (string enum, list, object).
- Instance variables: inside a replicated entity, expressions may use `i` (linear index), `r`, `c` (grid
  coordinates), `n` (count), and `^i`, `^r`, `^c` (enclosing replicated ancestor). Reference and selector strings
  may interpolate them with braces: `"hbm[{i}]"`, `"/chip.hbm_if{^i}.phy"`. Interpolation is evaluated per
  instance at expansion (step 8), so it is the only per-instance-varying construct besides `vary`.

### 14.4 References and selectors

`Ref` is a path string resolved lexically from the referencing entity's scope:

1. `"rf"`: sibling in the same container instance; if absent, look in the parent container, then grandparent,
   etc. (nearest wins). If the nearest match is a replicated entity with >1 instance, that is ambiguous unless the
   reference is from inside a sibling instance with matching index (`E-IR-0207`; use an explicit selector).
2. `"^.smem"`: start at the parent scope (`^.^` for grandparent).
3. `"/board0.chip0.die0.l2p0"`: absolute path.
4. `"tc"` from within `smsp2` binds to `smsp2.tc` (per-instance resolution: references inside a replicated
   template resolve per instance).

`Selector` extends a path with patterns, per segment:

| Pattern | Meaning |
|---|---|
| `sm*` | all instances of `sm` |
| `sm[3]` | instance 3 (linear) |
| `tile[1;2]` | grid instance (row 1, col 2); `tile[0..2;*]` rows 0-1, all cols. The instance id `tile1_2` also works |
| `sm[0..54]` | half-open range |
| `sm[0,5,9]` | list (on a grid, `[i]` indexes in layout order) |
| `**` | any depth of containers |
| `*.l1` | `l1` in every direct child |

A selector in an `endpoints`, `members`, `disabled` or `vary` field that matches nothing is `E-IR-0205`.

### 14.5 Patches (`set`)

`set` at the document level or a use site is a list of `{ path: Selector + "#" + field path, value }` or the
object form `{ "gpc*.tpc*.sm*.l1#capacity": "256KiB" }`. Patch selectors are absolute from the system root (shorthand
wrapping applied first, so `board.chip.die`). Patches apply after template instantiation and before expansion. Patching a field that does not exist is `E-IR-0212`. A field path ending in `+` appends to an array
(`"die#clusters+": [{...}]`), which is how a variant adds a block (example 20.3).

### 14.6 Imports and `extends`

```rust
#[serde(untagged)]
pub enum ImportRef {
    Path(String),                                  // "lib/tpu_tc.json5" relative to the importing file
    Pinned { path: String, sha256: String },       // content-pinned; mismatch => E-IR-0209
    Builtin(String),                               // "kiln:std/hbm3e_stack" (std library shipped with kiln-ir)
}
```

`imports: { lib: "lib/tpu.json5" }` makes `lib.<template>` usable. An imported template's expressions see the
imported document's `params` (lexical), overridable per use with `with`. `extends: "tpu_v5e.json5"` loads that whole
document as the base; the current document's `params` override the base's params, and its `set` patches apply on
top. `extends` is how a design variant is written in a few lines (example 20.3). The evolution loop (06) stores
designs as fully inlined documents (`kiln fmt --inline`) so a design is self-contained.

### 14.7 Shorthands

| Shorthand | Expands to |
|---|---|
| `system: { package: {...} }` | `system.boards = [{id: board, packages: [{id: chip, ...}]}]` (count 1 each) |
| `system: { die: {...} }` | as above with `packages[0].dies = [die]` |
| `feeds: { a: "rf" }` | `feeds: { a: { from: "rf" } }` |
| `precisions: ["bf16*bf16+fp32", "int8*int8+int32@2"]` | `Mac` modes |
| `link: "512b"` | `link: { width_bits: 512 }` |
| `topology: "crossbar"` | `topology: { type: "crossbar" }` |
| `endpoints: ["a*", "b"]` | `[{select: "a*"}, {select: "b"}]` |
| `mem_stacks[].attach: "die.dma"` | `StackAttach::Network { network: "die.dma" }` |

### 14.8 Recommended authoring formats

| Format | Use | Notes |
|---|---|---|
| JSON5 (`.json5`) | **Recommended for LLM authors and hand-written designs.** | Comments carry provenance (`// pub:wp`, `// assumed:`), unquoted keys, trailing commas, quantity strings, expressions. |
| Python builder `kiln.hw` (kiln-py) | Programmatic sweeps, migration of `harness/design.py` designs | Thin dataclass layer that emits the same JSON5/JSON; no semantics of its own; `kiln.hw.load(path)` round-trips. |
| JSON (`.json`) | Canonical storage, hashing, evolution archive | Emitted by `kiln fmt --canonical`. |
| RON / YAML | Accepted (00-overview) | Same data model; no extra features. |

A Rust-like DSL was considered and rejected: a new grammar costs LLM accuracy, while JSON5 is near-JSON, which LLMs
emit reliably.

## 15. Expansion pipeline and instance naming

Deterministic pipeline in `kiln-ir` (each step pure, no HashMap iteration):

| Step | Action | Errors |
|---|---|---|
| 1 | Parse to a span-annotated value tree (format by extension or `--format`) | E01xx |
| 2 | Read `schema`; migrate to current (sec 19) | E19xx |
| 3 | Resolve `extends` and `imports` (recursive, cycle-checked) | E0209 |
| 4 | Evaluate document params; instantiate templates (`use`/`with`), substitute expressions; merge use-site fields | E02xx |
| 5 | Apply shorthands, then `set` patches | E0212 |
| 6 | Typed deserialize (`deny_unknown_fields`), fill defaults | E01xx |
| 7 | **Canonical compact form** (sec 17); compute `design_hash` | |
| 8 | Expand `count`/`layout` into instances; apply `vary`; mark `disabled` | E0105, E0210, E0211 |
| 9 | Resolve refs/selectors to typed indices | E0205-E0207 |
| 10 | Synthesize implicit entities (controllers, PHYs, ports) | |
| 11 | Build networks into routers + channels; build feeds into channels | E07xx |
| 12 | Structural validation (sec 18) | E03xx-E12xx |
| 13 | Hand to kiln-phys: derive physical values, place, physical validation (E08xx physical subset, E09xx power) | |

Instance naming:
- `count: n`, linear: `<id><i>` for `i in 0..n` (`sm` x 2 -> `sm0`, `sm1`). With `count` 1 or absent the instance
  id is `<id>` (no suffix).
- Grid: `<id><r>_<c>` (2D), `<id><i>_<j>_<k>` (3D).
- An id ending in a digit with `count > 1` is `E-IR-0105` (prevents `hbm2` x 5 producing `hbm20`).
- Paths concatenate instance ids with `.`; network-internal routers are `<net>.r<i>` or `<net>.r<x>_<y>`.
- Expansion is depth-first in declaration order; typed indices (`UnitIx`, `MemIx`, `NodeIx`, `ChanIx`) are assigned
  in that order and are stable for a given canonical form.

Expansion budget: `ExpandOptions::max_instances` (default 50,000,000) guards memory; exceeding it is `E-IR-0210`,
which reports the per-entity contribution so the author can see which `count` exploded. This is a resource guard,
not a modelling cap.

## 16. Expanded hardware model (engine-facing)

```rust
pub struct HwModel {
    pub design_hash: DesignHash,
    pub schema: SchemaVersion,
    pub tree: Vec<Container>,            // boards, packages, dies, clusters (instances), parent links
    pub units: Vec<UnitInst>,
    pub memories: Vec<MemInst>,          // on-chip memories, mem-stack instances, local buffers
    pub blocks: Vec<BlockInst>,
    pub routers: Vec<RouterInst>,
    pub ports: Vec<PortInst>,            // die/package io ports, switch ports, endpoint injection ports
    pub channels: Vec<Channel>,
    pub resources: Vec<SharedResource>,
    pub clocks: Vec<ClockInst>,
    pub power_domains: Vec<PowerDomainInst>,
    pub address_maps: Vec<AddressMapInst>,
    pub index: BTreeMap<Path, NodeIx>,   // path lookup for errors/viz
}

pub enum NodeIx { Unit(UnitIx), Mem(MemIx), Block(BlockIx), Router(RouterIx), Port(PortIx), Container(ContIx) }

pub struct Channel {                     // directed
    pub src: NodeIx, pub dst: NodeIx,
    pub kind: ChannelKind,               // feed | mem_port | noc_hop | bus | d2d | serdes | vertical | optical | near | host
    pub width_bits: u32, pub clock: ClockIx,
    pub resource: ResourceIx,            // contention group (bus, shared port, link direction)
    pub network: Option<NetIx>,
    pub derived: Option<ChannelPhys>,    // filled by kiln-phys: bandwidth, latency, energy_per_byte, wire_len_um
    pub overrides: ChannelOverrides,
}

pub struct SharedResource { pub kind: ResKind /* bus | port | link_dir | router_xbar | dma | mem_bank_group */,
                            pub capacity: Option<BytesPerSec> /* filled by kiln-phys */ }

pub struct UnitInst {
    pub path: Path, pub kind: ComputeKindInst, pub precisions: Vec<PrecisionMode>, pub ops: OpSet,
    pub feeds: IndexMap<OperandRole, (MemIx, ChanIx /*read*/, Option<ChanIx> /*write*/)>,
    pub local: Vec<MemIx>, pub near: Option<NearInst>, pub clock: ClockIx, pub power_domain: Option<PdIx>,
    pub container: ContIx, pub enabled: bool,
}
```

### 16.1 Derived-value annotation

kiln-phys returns `PhysAnnotation { per_node, per_channel, floorplan, area, ... }` keyed by the indices above. Each
value is `Derived<T> { value: T, source: Source }` with `Source = Override | Model(&'static str) | Published(Id)`.
The visualizer shows the source per value.

### 16.2 Queries provided by kiln-ir (used by 03)

| Fn | Returns |
|---|---|
| `route(src, dst) -> Option<Vec<ChanIx>>` | deterministic path under declared routing |
| `minimal_paths(src, dst, k)` | up to k minimal paths (adaptive routing) |
| `reachable_from(mem) -> BitSet<MemIx>` | memories data can move to |
| `feed_memories(unit) -> &[MemIx]` | staging memories per role |
| `staging_chains(unit, role)` | all memory chains offchip -> ... -> feed memory (03 tiling search space) |
| `peak_ops(unit_set, precision, class)` | structural peak at nominal clock |
| `instances(selector)` | resolved instances |
| `summary() -> HwSummary` | per chip: structural peak ops per (a,b,acc) mode incl. sparse flag; per-class vector/special throughput; per memory level capacity and bandwidth; off-chip capacity/bandwidth; chip count and inter-chip topology; `exec_model`; host launch overhead (02 sec 16 asks a-e) |
| `level(mem) -> u8` | derived memory level: 0 local buffers, 1 memories that feed units directly, k+1 one routed move further from every unit, off-chip stacks = deepest; `level_hint` overrides for display only |
| `diff(a, b) -> HwDiff` | canonical structural diff of two designs (added/removed/changed entities and fields, by template-level path), for 05 and the evolution log |

### 16.3 Graph rules

- Each feed becomes 1 read channel (memory -> unit) and, for roles `o`/`out`/`c`, 1 write channel.
- Each network endpoint gets an injection and ejection channel to its router (bus/crossbar: to the shared medium).
- Bus: all channels share one `SharedResource`. Crossbar: one resource per output port. Mesh/torus: one resource per
  link direction.
- `backing` on a cache adds miss/fill channels only if no network already connects the two (otherwise misses route
  over the network).
- Mem stack channels: controller <-> PHY <-> stack, one resource per pseudo-channel (Tier B) aggregated per stack
  (Tier A).

### 16.4 Data-movement semantics

A tensor tile can move from memory X to memory Y iff `route(X, Y)` exists. A compute unit consumes operand role R
only from `feeds[R]`. Therefore every operand's journey is: resident location (off-chip stack, or on-chip memory)
-> chain of memories along routed channels -> feed memory -> unit. The mapper (03) chooses the chain and residency;
01 guarantees (via E0720/E0721) that at least one chain exists for every unit and role.

## 17. Canonical form and design hashing

The **canonical compact form** is the output of pipeline step 6: templates instantiated, params and expressions
evaluated, shorthands expanded, patches applied, `extends`/`imports` inlined, defaults materialized, but `count`
and `layout` retained (not instance-expanded).

Serialization rules (`kiln_ir::canon::to_canonical_json`):
1. RFC 8785 JSON Canonicalization Scheme (JCS, 00 decision 2): UTF-8, no whitespace, object keys sorted (bytewise
   equals JCS order for the ASCII keys kiln uses), arrays in declared order.
2. Arrays whose order is not semantic are sorted by canonical serialization of their elements: `precisions`, `ops`,
   `functions`, `in_network_reduce`, `disabled`, `coherent_with`, `sparsity`. Every other array (layouts, `vf`,
   `endpoints`, children) keeps order because expansion order defines indices.
3. Integers as integers; floats with shortest round-trip repr (ryu); `-0.0` -> `0`; NaN/inf rejected earlier.
4. Defaults written explicitly; `Option::None` omitted.
5. Excluded: `meta` (description, citations, claims, notes), per-entity `notes`, source spans, `schema` minor
   version (only major is hashed; minor bumps are additive with defaults, so equal designs stay equal).

`design_hash = "hw1-" + hex(sha256(canonical_json))[..32]`. sha256 is chosen to match the existing harness
(`harness/design.py`). Two authorings that differ only in template use, params, expressions, quantity literals,
key order, comments, or format (JSON5/RON/YAML) hash identically. A design written with `count: 108` and one
listing 108 SMs by hand hash differently; that is accepted (both are rare in practice, and hashing the expanded
form costs O(instances)). `kiln hash --expanded` provides an expanded-form hash for audits.

Derived physical values are not part of the design hash; they depend on the calibration set, whose hash is
recorded separately in result provenance (00-overview).

## 18. Validation

### 18.1 Error structure

```rust
pub struct HwDiag {
    pub code: &'static str,          // "E-IR-0720"
    pub severity: Severity,          // error | warning | info
    pub path: Option<Path>,          // instance or template-relative entity path
    pub span: Option<Span>,          // file, line, col in the authoring source (through templates: use-site + template span)
    pub message: String,             // what is wrong, with offending value
    pub hint: String,                // concrete edit that fixes it
    pub related: Vec<Path>,
}
```

Rules: all diagnostics of a stage are collected (no early exit within a stage); stage N+1 runs only if stage N
had no errors. Diagnostics on replicated entities are reported once per template-level entity with
`"(and 107 other instances)"` to keep LLM context small. Warnings never block evaluation unless `--deny-warnings`.
`kiln validate --format json` emits the list for the evolution loop (06).

### 18.2 Codes

Structural checks run in kiln-ir (S); physical checks run after kiln-phys annotation/placement (P). (P) checks
are emitted by kiln-phys; where 04 names the same check, the emitted code is 04's and the 01 number is an alias:
E-IR-0803 = `E-PHYS-AREA-OVERFLOW`, E-IR-0804 = `E-PHYS-RETICLE`, E-IR-0805 = `E-PHYS-SHORELINE`,
E-IR-0907 = `E-PHYS-POWER-DENSITY`.

**E01xx parse / schema (S)**

| Code | Check | Example message / hint |
|---|---|---|
| E-IR-0101 | unknown field | `memories[0] 'l1': unknown field 'size'. Allowed: capacity, banks, ports, ... Did you mean 'capacity'?` |
| E-IR-0102 | missing required field | `unit 'mxu': missing 'precisions'. Add e.g. precisions: ["bf16*bf16+fp32"]` |
| E-IR-0103 | wrong JSON type | `count must be an integer or "=expr", got "four"` |
| E-IR-0104 | id grammar | `id 'L2' invalid: ids match ^[a-z][a-z0-9_-]*$ (no dots, lowercase). Use 'l2'.` |
| E-IR-0105 | id ends in digit with count>1 | `'hbm2' has count 5; instances would be hbm20..hbm24. Rename to 'hbm'.` |
| E-IR-0106 | duplicate id in scope | |
| E-IR-0107 | unknown schema version | lists supported versions and `kiln migrate` |
| E-IR-0108 | quantity dimension mismatch | `capacity "1.4GHz": expected bytes (e.g. "40MiB")` |
| E-IR-0109 | implausible value | `clock freq 14.1 GHz outside [1 MHz, 20 GHz]; did you mean 1.41GHz?` |
| E-IR-0110 | non-integer bytes/count after conversion | |

**E02xx templates / expansion (S)**

| Code | Check |
|---|---|
| E-IR-0201 | unknown template (lists known, with nearest match) |
| E-IR-0202 | template param without default not supplied in `with` |
| E-IR-0203 | `with` names a param the template does not declare |
| E-IR-0204 | expression error (parse, unknown name, dimension, non-integer for integer field, divide by zero) |
| E-IR-0205 | selector matches nothing (shows the closest existing paths) |
| E-IR-0206 | unresolved reference |
| E-IR-0207 | ambiguous reference (lists candidates; suggests `^.` or absolute path) |
| E-IR-0208 | template or import cycle (prints the cycle) |
| E-IR-0209 | import not found or pinned sha256 mismatch |
| E-IR-0210 | expansion budget exceeded (prints largest contributors) |
| E-IR-0211 | `disabled`/`vary` index out of range for the entity's count/grid |
| E-IR-0212 | `set` patch targets a non-existent field |
| E-IR-0213 | `count` disagrees with `layout` product |

**E03xx compute units (S)**

| Code | Check |
|---|---|
| E-IR-0301 | unit has no precision modes |
| E-IR-0302 | unknown precision name (lists registry) |
| E-IR-0303 | accumulator incompatible with inputs (int inputs need int32/int16 acc; float acc narrower than inputs) |
| E-IR-0304 | role used by a precision mode has no feed and no local buffer with `refill_from` (`unit 'tc' reads operand 'b' but has no feed for 'b'. Add feeds: { b: "rf" }`) |
| E-IR-0305 | feed `from` is not a memory (or near-bound memory) |
| E-IR-0306 | op class not legal for unit kind (e.g. `matmul` on `special`) |
| E-IR-0307 | zero geometry dimension / lanes |
| E-IR-0308 | dataflow not supported by geometry (e.g. `input_stationary` on `cim`) |
| E-IR-0309 | invalid sparsity (speedup <= 1, pattern n:m with n >= m, operand not used) |
| E-IR-0310 | feed `width_bits` exceeds the memory's widest port (`feed 'a' of 'mxu' wants 4096 b/cycle; memory 'vreg' ports are 1024 b. Widen ports or lower width_bits`) |
| E-IR-0311 | local buffer too small for declared dataflow (5.6) |
| W-IR-0312 | MX block size does not divide the reduction dimension of the geometry (padding waste) |
| W-IR-0313 | duplicate precision mode (same a,b,acc,out) |
| E-IR-0314 | unit precision has no `convert` path: a mode outputs a precision no reachable vector unit can convert from, while ops downstream need a different one (only when `--strict-convert`) |

**E04xx on-chip memory (S)**

| Code | Check |
|---|---|
| E-IR-0401 | capacity not divisible by `banks * word_bits/8` |
| E-IR-0402 | memory has no ports |
| E-IR-0403 | `kind: cache` without `cache` spec; `cache` spec on a non-cache memory that has no carveout option with `cache > 0`; carveout with `cache > 0` but no `cache` spec |
| E-IR-0404 | `backing` cycle |
| E-IR-0405 | carveout option sums exceed capacity |
| E-IR-0406 | partition sums exceed capacity |
| W-IR-0407 | dead memory: no feed, network or backing reaches it |
| E-IR-0408 | `word_bits` not a power of two multiple of 8 |
| E-IR-0409 | bandwidth override exceeds derived port bandwidth |
| W-IR-0410 | register file fed by units of more than one cluster instance without `shared_rf: true` |
| E-IR-0411 | cache line not a multiple of word, or ways*line > capacity |

**E05xx off-chip memory (S, P)**

| Code | Check |
|---|---|
| E-IR-0501 | mem stack has no attach (no controller, PHY or endpoint) |
| W-IR-0502 | capacity outside kind's known range (e.g. HBM2 stack > 8 GiB) |
| W-IR-0503 | pin rate beyond kind's published maximum (04 table) |
| E-IR-0504 | stack `site` references an unknown package site |
| E-IR-0505 | PHY count/width inconsistent with stack io width (`stack 'hbm' x5 needs 5 HBM PHYs of 1024 b; die has 6 hbm sites but 4 PHYs bound`) |
| E-IR-0506 | bandwidth override exceeds derived peak `io_width_bits * pin_rate / 8` |
| E-IR-0507 | stacked DRAM `Vertical` attach to a die in another package or non-adjacent layer |
| E-IR-0508 (P) | stack-to-PHY distance exceeds the 04 reach limit for the substrate (HBM on organic substrate is illegal) |

**E06xx near / in-memory compute (S, P)**

| Code | Check |
|---|---|
| E-IR-0601 | `near.memory` is not a Memory or MemStack |
| E-IR-0602 | granularity incompatible with memory (`per_pseudo_channel` on SRAM; `per_bank` on a 1-bank memory with count>1 units) |
| E-IR-0603 | near unit declares feeds from a memory other than its bound one |
| E-IR-0604 | `ExclusiveAllBank` stack has no host-side path (controller) to issue commands |
| E-IR-0605 (P) | logic die derived area exceeds `area_budget` |
| E-IR-0606 | CIM `weight_capacity` given and != `rows*cols*cell_bits*weight_sets/8` (a cell stores `cell_bits` bits, not one weight; 9.3) |
| E-IR-0607 | explicit count != granules implied by binding |
| E-IR-0608 | CIM precision mode `@rate` > 1 (throughput beyond the array-derived rate, 9.3) |
| E-IR-0609 | CIM parameter out of range: `parallel_rows` > `rows`, `cell_bits` > 8, `input_bits_per_cycle` > 16, analog without `adc_bits`, `adc_bits` on digital |
| W-IR-0610 | analog CIM `adc_bits` below the lossless boundary for its `parallel_rows` (accuracy not modeled) |

**E07xx interconnect and connectivity (S)**

| Code | Check |
|---|---|
| E-IR-0701 | endpoint selector resolves to a non-connectable entity (e.g. a cluster) |
| E-IR-0702 | same instance bound twice to one network (use `multiplicity`) |
| E-IR-0703 | topology too small for endpoints (`mesh 4x4 with concentration 1 cannot host 20 endpoints`) |
| E-IR-0704 | router radix exceeded (topology degree + concentration > declared radix) |
| E-IR-0705 | custom graph disconnected or edge references router >= routers |
| E-IR-0706 | zero link width / lanes / rate |
| E-IR-0707 | routing deadlock-prone (dimension-order on torus/ring with vcs < 2; table routes with cyclic channel dependency) |
| E-IR-0708 | d2d link between dies of different packages (use serdes) or serdes between dies of one package without `allow` |
| E-IR-0709 | port `phy` refers to a PHY of the wrong `for_kind` |
| E-IR-0710 | endpoint multiplicity exceeds the port count the entity declares (`chip has 4 'ici' ports; torus binding needs 6`) |
| E-IR-0711 | `in_network_reduce` precision unsupported by the switch/routers |
| E-IR-0712 | link bandwidth override exceeds the PHY/width-derived bandwidth |
| E-IR-0713 | link `width_bits` disagrees (>1%) with PHY-derived bandwidth |
| E-IR-0714 | `via` names a network the unit and memory are not both endpoints of |
| E-IR-0720 | **staging unreachable**: a unit's feed memory cannot be filled from any off-chip stack or host for some input role (`unit 'mxu' (tc0) stages operand 'b' through 'vmem', but 'vmem' is not connected to any network that reaches off-chip memory 'hbm'. Add 'vmem' to the endpoints of a network containing 'hbm_mc*', or set vmem.backing`) |
| E-IR-0721 | **output cannot drain**: a unit's `o`/`out` memory has no route to off-chip memory or to another unit's input memory |
| E-IR-0722 | feed exists but no channel can be built (memory in another die without d2d path, e.g. feed crosses dies) |
| W-IR-0723 | chips in a multi-chip system with no chip-to-chip network (collectives impossible; they will be rejected by 03 if the workload is sharded) |
| W-IR-0724 | multicast/broadcast flag on a bus (redundant) |
| W-IR-0725 | host link missing (weights load path unknown; host-side costs are skipped) |
| W-IR-0726 | direct-network entity has more port instances than router links; extras unused |

**E08xx floorplan (S for pinned data, P after placement)**

| Code | Check |
|---|---|
| E-IR-0801 (P) | overlapping blocks (`'gpc3' [x 12000..18000] overlaps 'l2p1' ...; move or set placement: auto`) |
| E-IR-0802 (P) | block outside its parent outline |
| E-IR-0803 (P) | die area exceeded: sum of child footprints / utilization > outline area (prints the top area consumers) |
| E-IR-0804 (P) | die exceeds reticle limit (unless `die.stitched: true`) |
| E-IR-0805 (P) | shoreline exceeded: PHY edge lengths on an edge > edge length (`north edge 33 mm needs 3 HBM PHYs (3 x 8.7 mm) + 6 NVLink (6 x 1.4 mm) = 34.5 mm`) |
| E-IR-0806 (P) | stack not adjacent to the die edge its PHY sits on |
| E-IR-0807 (P) | dies/stacks overlap on the substrate |
| W-IR-0808 (P) | substrate exceeds the 04 limit for its kind (e.g. silicon interposer > 3.3x reticle) |
| E-IR-0809 | upper-layer die larger than the die below (hybrid bond) |
| E-IR-0810 | vertical link between non-adjacent layers or non-overlapping footprints |
| E-IR-0811 | pinned coordinates not inside the parent or negative |
| E-IR-0812 | `Placement::Array` on an entity without a grid layout |

**E09xx clocks and power (S, P)**

| Code | Check |
|---|---|
| E-IR-0901 | unknown clock reference |
| E-IR-0902 | `vf` not strictly ascending in freq, or voltage decreasing |
| E-IR-0903 | nominal freq or base outside `vf` range |
| E-IR-0904 | entity in two power domains |
| E-IR-0905 (P) | power cap below leakage + idle at the lowest V/F point (design can never run) |
| W-IR-0906 | channel crosses clock domains with no `crossing_latency` (default applied) |
| E-IR-0907 (P) | power density above 04 limit for cooling class (envelope error, 04 sec 9) |
| E-IR-0908 | `assumed` cap range does not hold `0 < lo <= cap <= hi` |

**E10xx technology (S)**

| Code | Check |
|---|---|
| E-IR-1001 | unknown tech node (lists table) |
| E-IR-1002 | memory implementation not available at node (e.g. `edram` at tsmc_n3e) |

**E18xx claims (P)**

| Code | Check |
|---|---|
| W-IR-1801 | derived metric differs from a `meta.claims` entry by more than `rel_tol` (`peak_ops.bf16 derived 296.1 TFLOP/s vs claimed 312 (src:wp); check count/geometry/clock`) |

**E19xx versioning (S)**

| Code | Check |
|---|---|
| E-IR-1901 | migration failed (prints the step and field) |
| W-IR-1902 | document migrated from older schema (suggests `kiln migrate --write`) |

### 18.3 Profiles

`kiln validate --profile <p>` (and the PyO3 `evaluate` entry point, which takes a profile) adds rules on top of the
structural checks:

| Profile | Purpose | Extra rules |
|---|---|---|
| `full` (default) | hand-written and exploratory designs | none |
| `reference` | published chips (sec 20) | `*.overrides` / `PowerOverride` allowed only with `source` citation (E-IR-1103); `meta.claims` required for peak ops and off-chip bandwidth |
| `search` | designs produced by the evolution loop (06) | **no unpriced performance** (06 "E-IR-UNPRICED" = E-IR-1101 and E-IR-1104; 00 decision 3: never rejects a change for being cost-neutral or cost-reducing): every performance override is compared with the value kiln derives from structure (kept next to the effective value in the expanded model, e.g. `bandwidth_derived`), regardless of `source` or legacy-import origin: a "more is better" field (bandwidth of on-chip memories, stacks, links/PHYs, DMA, switch `reduce_bandwidth`, `internal_bandwidth`, CIM `weight_write`, PHY `encoding_efficiency`, unit/CIM `@rate`, vector `class_rates` against table 6.3, special `fn_rates` against 1.0) is allowed at <= derived (1e-9 relative tolerance; a de-rate is cost-neutral) and rejected above it, with derived value, override and ratio in the message; a "less is better" field (latency, `command_latency`, energy, area, leakage, power, `fec_latency`, `switch_latency`, `crossing_latency`, `Pipeline` fill/drain, router pipeline) is allowed at >= derived and rejected below it; a field kiln-ir cannot derive yet (energies, area, leakage, power, most latencies until kiln-phys exists) is rejected as unverifiable, and becomes allowed under the same rule once kiln-phys derives it; `family` is rejected; every performance-bearing field must be one 04 prices (E-IR-1104 lists unpriceable fields). Assumed constants come only from versioned data files. |
| `stream_compat` | legacy subset for differential testing vs the Stream fork (06 sec 2.3) | 1 chip; matrix/vector units with one precision each feeding one shared memory; <= 2 on-chip levels + 1 DRAM; bf16*bf16+fp32 only; no near-memory, no NoC topologies other than bus/p2p (E-IR-1105 names the first violating entity) |

| Code | Check |
|---|---|
| E-IR-1101 | `search`: unpriced performance override (path, field, and "remove the override; kiln derives it from structure") |
| E-IR-1102 | `search`: `family` set (platform calibration residuals cannot be inherited by novel designs) |
| E-IR-1106 | `search`: power cap marked `assumed` (unpublished caps are for reference designs of real chips) |
| E-IR-1107 | `search`: analog CIM `adc_bits` below the lossless boundary (accuracy loss is unmodeled, open question 12) |
| E-IR-1103 | `reference`: override without `source` |
| E-IR-1104 | field kind 04 cannot price (e.g. `Custom` DRAM kind without timing/energy data) |
| E-IR-1105 | not inside `stream_compat` subset |

### 18.4 Random design generator

`kiln-ir::arb` (cargo feature `arbitrary`, used by 06's property tests and corpora) generates designs that pass
structural validation by construction: it samples a hierarchy depth, replicated clusters, unit kinds and
precisions from the registries, memories with consistent banks/ports, and networks whose sizes satisfy
E-IR-0703/0704, and it always adds a staging chain from a mem stack to each feed (E-IR-0720/0721). Parameters:
`ArbConfig { seed, max_chips, max_instances, p_multi_chip, p_near_memory, profile }`. Determinism: same seed and
config give the same canonical JSON. Generated designs that fail physical (P) checks are kept in the corpus as
negative cases.

## 19. Schema versioning and migration

- `schema: "kiln.hw/<major>.<minor>"`. Current: `kiln.hw/1.0`.
- **Minor** bump: additive only (new optional fields with defaults, new enum variants, new precisions/op classes).
  Older documents load unchanged; hash unchanged because only major is hashed and defaults that reproduce old
  behaviour are materialized identically. A minor bump that would change the canonical form of an old document is
  not allowed; make it a major bump.
- **Major** bump: breaking. `kiln-ir/src/migrate/` holds pure functions `v{N}_to_v{N+1}(Value) -> Result<Value>`
  operating on the untyped value tree, chained at load. Each migration ships with golden before/after fixtures and a
  test that every reference design under `designs/` migrates and validates. Migrations never consult physics.
- Design hashes are major-version-scoped (`hw1-` prefix). Evolution archives record both the original source and
  the migrated canonical form; a migrated design gets a new hash, and the archive keeps a `migrated_from` link.
- **`kiln.hw/0` = harness JSON.** The existing `harness/design.py` format (`clock_mhz`, `compute[]`, `memory[]`,
  `offchip`, `links[]`) is registered as schema 0 and migrated by `v0_to_v1`:

| harness v0 | kiln.hw/1 |
|---|---|
| `clock_mhz`, `tech_node` | `clocks: [{id: core, freq}]`, `tech: tsmc_<node>` |
| `ComputeUnit(kind=matrix, rows, cols, count, precision, accumulator)` | `units[]: {kind: matrix, geometry: {systolic: {rows, cols}}, precisions: ["<p>*<p>+<acc>"]}` |
| `buffer_kib/buffer_gbps` | a `local` buffer (role `any`) + feed width = buffer_gbps / clock |
| `regfile_kib/regfile_gbps` (vector) | a `register_file` memory with a feed |
| `attach` | `feeds: { any: <memory> }` |
| `MemoryUnit(size_mib, bandwidth_gbps, count)` | `memories[]: {kind: scratchpad, capacity, ports: [{dir: rw, width_bits: bandwidth/clock}]}` |
| `OffChip(capacity_gib, bandwidth_gbps, attach, kind, stacks)` | `mem_stacks: [{count: stacks, capacity: cap/stacks, kind, overrides.bandwidth: bw/stacks, attach}]` with io/pin rate set to reproduce bw |
| `Link(kind=bus, scope=global)` | network `{topology: bus}` |
| `Link(scope=per_memory)` | network with `count = memory count`, endpoints per instance |

The v0 migration exists so every harness experiment can be replayed in kiln (06 regression). `kiln import
harness-design <legacy.json>` (06 sec 9) is the CLI entry point to it; migrated legacy designs fall inside the
`stream_compat` profile by construction.

## 20. Worked examples

All examples are in the recommended JSON5 authoring form. Comment tags: `pub:<src>` = published, cited in
`meta.citations`; `derived` = arithmetic from published numbers; `assumed` = our modelling choice or unpublished
value. Examples 20.1 to 20.4 are the reference designs that replace `harness/designs/*.json`; 06 owns their
calibration against measurements.

### 20.1 NVIDIA A100-SXM4-40GB

Real structure: 8 GPC sites x 8 TPCs x 2 SMs on the GA100 die, harvested to 7 GPCs / 54 TPCs / 108 SMs; each SM
has 4 sub-partitions (SMSPs) with one tensor core, a 64 KiB register file, FP32/INT/FP64 lanes and SFUs; 192 KiB
L1/SMEM per SM; L2 in two partitions joined by an inter-partition link; 6 HBM2 sites, 5 active stacks behind 10 of
12 memory controllers.

Derived checks (`kiln validate --report` prints these; they must match the claims):
- bf16 dense: 108 SM x 4 TC x 256 MAC x 2 x 1.41 GHz = 311.9 TFLOP/s (claim 312, pub:wp).
- int8: x2 = 623.8 TOP/s (claim 624). TF32: x0.5 = 155.9 (claim 156). FP64 TC: 108 x 4 x 16 x 2 x 1.41 GHz = 19.5 (claim 19.5).
- HBM: 5 x 1024 b x 2.43 Gb/s / 8 = 1555.2 GB/s (claim 1555).
- L2: 80 slices x 64 B/clk = 5120 B/clk (pub:wp) = 7.22 TB/s at 1410 MHz.

```json5
{
  schema: "kiln.hw/1.0",
  name: "a100-sxm4-40gb",
  family: "a100_40gb",                                 // 06 platform key (profile reference)
  meta: {
    description: "NVIDIA A100-SXM4-40GB (Colab A100). GA100 die, 108 SMs active.",
    citations: {
      wp: "NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020)",
      ds: "NVIDIA A100 datasheet (SXM4 40GB: 1555 GB/s, 400 W)",
      cuda: "CUDA C++ Programming Guide: arithmetic-instruction throughput table (CC 8.0), shared memory (32 banks x 4 B), L2 persistence",
      obs: "kiln calibration/measurements/a100_2026-10-04 (sustained ~1290 MHz on large GEMMs at 400 W, user observation)",
    },
    claims: [
      { metric: "peak_ops.bf16", scope: "", value: 312e12, rel_tol: 0.01, source: "wp" },
      { metric: "peak_ops.int8", scope: "", value: 624e12, rel_tol: 0.01, source: "wp" },
      { metric: "offchip_bw",    scope: "", value: 1555e9, rel_tol: 0.01, source: "ds" },
      { metric: "onchip_bytes.cache", scope: "board.gpu.ga100.l2p*", value: 41943040, rel_tol: 0.0, source: "wp" },
      { metric: "die_area", scope: "board.gpu.ga100", value: 826, rel_tol: 0.01, source: "wp" },
    ],
  },
  tech: "tsmc_n7",                                     // pub:wp (TSMC 7nm N7)
  clocks: [
    { id: "gpc_clk", freq: "1410MHz", base: "1095MHz", // pub:ds boost/base
      vf: [ { freq: "1095MHz", voltage: "0.73V" },     // assumed V/F curve (NVIDIA does not publish voltages);
            { freq: "1290MHz", voltage: "0.81V" },     //   04 calibrates it so the 400 W cap lands near 1290 MHz
            { freq: "1410MHz", voltage: "0.87V" } ] }, //   on large GEMMs (obs)
    { id: "hbm_clk", freq: "1215MHz" },                // derived: 2.43 Gb/s DDR pins
  ],
  power: [ { id: "board_tdp", members: "board.gpu", cap: "400W", policy: "dvfs", clocks: ["gpc_clk"] } ], // pub:ds

  templates: {
    smsp: { kind: "cluster", body: {
      units: [
        { id: "tc", kind: "matrix",
          geometry: { mma: { m: 8, n: 4, k: 8 } },      // 256 dense FMA/clk per TC pub:wp; the m/n/k split is assumed
          dataflow: "output_stationary",
          accumulate_in: "feed",                        // accumulators live in registers
          precisions: [
            "fp16*fp16+fp32", "fp16*fp16+fp16", "bf16*bf16+fp32",   // pub:wp 312 TFLOPS
            "tf32*tf32+fp32@0.5",                                   // pub:wp 156
            "int8*int8+int32@2", "int4*int4+int32@4",               // pub:wp 624 / 1248
            "fp64*fp64+fp64@0.0625",                                // pub:wp 19.5
          ],
          sparsity: [ { pattern: "2:4", operand: "a", speedup: 2.0, metadata_bits_per_nz: 2 } ], // pub:wp
          feeds: { a: "rf", b: "rf", c: "rf", o: "rf" } },
        { id: "alu", kind: "vector", lanes: 16,         // 16 FP32 lanes per SMSP = 64/SM pub:cuda
          precisions: ["fp32@1", "int32@1", "fp16@4", "bf16@4", "fp64@0.5"], // pub:cuda 64/256/256/32 per SM per clk
          feeds: { any: "rf" } },
        { id: "sfu", kind: "special", lanes: 4,         // 16/SM/clk pub:cuda
          functions: ["exp2", "log2", "rsqrt", "recip", "sqrt", "sin", "cos", "tanh"],
          precisions: ["fp32@1"], feeds: { any: "rf" } },
      ],
      memories: [
        { id: "rf", kind: "register_file", capacity: "64KiB", // pub:wp 256 KB/SM
          banks: 4, word_bits: 32,                               // assumed
          ports: [ { dir: "read", count: 3, width_bits: 1024 },  // assumed: 3 operand reads x 32 lanes x 32 b
                   { dir: "write", count: 1, width_bits: 1024 } ] },
      ],
    } },

    sm: { kind: "cluster", body: {
      clusters: [ { id: "smsp", count: 4, use: "smsp" } ],      // pub:wp
      memories: [
        { id: "l1", kind: "scratchpad", capacity: "192KiB",     // pub:wp combined L1/SMEM
          banks: 32, word_bits: 32,                              // pub:cuda
          ports: [ { dir: "rw", width_bits: 1024 } ],            // 128 B/clk pub:cuda (32 banks x 4 B)
          operands: { policy: "carveout", options: [             // pub:cuda carveout sizes (SMEM KiB)
            { scratch: "0KiB",   cache: "192KiB" }, { scratch: "8KiB",   cache: "184KiB" },
            { scratch: "16KiB",  cache: "176KiB" }, { scratch: "32KiB",  cache: "160KiB" },
            { scratch: "64KiB",  cache: "128KiB" }, { scratch: "100KiB", cache: "92KiB" },
            { scratch: "132KiB", cache: "60KiB" },  { scratch: "164KiB", cache: "28KiB" } ] },
          cache: { line: "128B", sectors: 4, ways: 4 },          // ways assumed
          backing: "l2p*.slice*" },                              // lexical lookup finds die-level l2p; misses route over the fabric
      ],
      networks: [
        { id: "lsu", topology: "bus", endpoints: ["smsp*.rf", "l1"], link: "1024b" }, // ldmatrix/LSU path, shares 128 B/clk
      ],
    } },
  },

  system: { package: {
    id: "gpu",
    substrate: { kind: "silicon_interposer" },                   // pub:wp CoWoS
    dies: [ {
      id: "ga100",
      default_clock: "gpc_clk",
      floorplan: {
        outline: { type: "fixed", w: "25.6mm", h: "32.3mm" },    // area pub:wp 826 mm2; aspect assumed
        shoreline: [
          { id: "hbm_w", count: 3, edge: "w", kind: "hbm" },     // 6 HBM2 sites pub:wp; 3 per side assumed from package photos
          { id: "hbm_e", count: 3, edge: "e", kind: "hbm" },
          { id: "nvl",   count: 12, edge: "s", kind: "serdes" }, // 12 NVLink3 pub:wp; edge assumed
          { id: "pcie_site", edge: "n", kind: "pcie" },
        ],
      },
      clusters: [
        { id: "gpc", count: 8, layout: { grid: [2, 4] }, placement: { mode: "array" },  // 8 GPC sites pub:wp; arrangement assumed
          disabled: ["gpc1_3", "gpc0_0.tpc7", "gpc0_1.tpc7"],   // -> 7 GPCs, 54 TPCs, 108 SMs pub:wp; which ones: assumed
          clusters: [ { id: "tpc", count: 8, clusters: [ { id: "sm", count: 2, use: "sm" } ] } ] },
        { id: "l2p", count: 2, layout: { grid: [1, 2] },          // two L2 partitions pub:wp
          memories: [ {
            id: "slice", count: 40, kind: "cache",               // 80 slices x 512 KiB = 40 MiB: total pub:wp, slicing assumed
            capacity: "512KiB", banks: 4, word_bits: 256,          // banks assumed
            ports: [ { dir: "read", width_bits: 512 }, { dir: "write", width_bits: 512 } ], // 64 B/clk/slice: derived from 5120 B/clk pub:wp
            cache: { line: "128B", sectors: 4, ways: 16,           // ways assumed
                     coherent_with: ["^.l2p*.slice*"],             // partitions keep coherent copies pub:wp
                     pinnable: "384KiB" },                         // 75% persisting carve-out (pub:cuda, 30 MB), split per slice: assumed
          } ] },
        { id: "hbm_if", count: 6, disabled: ["hbm_if5"],         // one interface per HBM site
          blocks: [
            { id: "mc", count: 2, kind: { type: "mem_controller",  // 12 x 512-bit controllers on GA100, 10 enabled pub:wp
                serves: "/board.gpu.hbm{^i}", width_bits: 512, endpoint_of: ["fabric"] } },
            { id: "phy", kind: { type: "phy", for_kind: "hbm", lanes: 1024 } },
          ] },
      ],
      networks: [ {
        // GPC <-> L2 crossbar, one crossbar per partition (router 0, 1) plus the inter-partition link (edge 0-1).
        id: "fabric",
        topology: { type: "custom", routers: 2,
                    edges: [ { a: 0, b: 1, link: { width_bits: 10240 } } ] }, // assumed: half of one partition's L2 bw
        router: { radix: 256, pipeline: 4 },                                  // assumed
        link: { width_bits: 512 },                                            // assumed 64 B/clk per endpoint port
        endpoints: [
          { select: "gpc0_*.tpc*.sm*.l1", at: { router: [0] } },
          { select: "gpc1_*.tpc*.sm*.l1", at: { router: [1] } },
          { select: "l2p0.slice*",  at: { router: [0] } },
          { select: "l2p1.slice*",  at: { router: [1] } },
          { select: "hbm_if[0..3].mc*", at: { router: [0] } },
          { select: "hbm_if[3..6].mc*", at: { router: [1] } },
        ],
      } ],
      blocks: [
        { id: "uncore", kind: { type: "misc" }, footprint: { area: "60mm2" } }, // frontend, copy engines, NVDEC/JPEG, hub: assumed area
      ],
      ports: [
        { id: "nvlink", count: 12, kind: "serdes", internal: "fabric",
          link: { phys: { type: "serdes", protocol: "nvlink_like", lanes: 4, lane_rate_bits_per_s: "50Gbps" } } },
          // 4 lanes x 50 Gb/s = 25 GB/s/dir per link, 12 links = 600 GB/s bidirectional pub:wp
        { id: "pcie", kind: "pcie", internal: "fabric",
          link: { phys: { type: "serdes", protocol: "pcie", lanes: 16, lane_rate_bits_per_s: "16Gbps" } } }, // Gen4 x16 pub:ds
      ],
    } ],
    mem_stacks: [ {
      id: "hbm", count: 6, disabled: ["hbm5"],                  // 6 sites, 5 active pub:wp
      kind: "hbm2", capacity: "8GiB", dies_high: 8,              // 40 GB / 5; 8 dies per stack pub:wp
      io_width_bits: 1024, pin_rate_bits_per_s: "2.43Gbps",      // 5120-bit bus pub:wp; pin rate derived from 1555 GB/s
      clock: "hbm_clk",
      attach: { phys: ["/board.gpu.ga100.hbm_if{i}.phy"] },
    } ],
    address_map: [ { id: "global", targets: "ga100.hbm_if*.mc*", granule: "512B" } ], // granule assumed
  } },
}
```

Note: the power domain applies DVFS to `gpc_clk` only; HBM power is counted against the same cap.

### 20.2 Google TPU v5e (single chip)

Published (pub:v5e): 1 TensorCore with 4 MXUs, a vector unit and a scalar unit; 197 TFLOP/s bf16; 393 TOP/s int8;
16 GB HBM; HBM bandwidth 819 GB/s, assumed (the v5e docs page lists "800 GiBps" = 859 GB/s; the JAX scaling book
lists 8.2e11 B/s, pub:sbook, which matches 2 HBM2e stacks at 3.2 Gb/s; flagged in open questions); 4 ICI ports, 400 GB/s bidirectional per chip. MXU
128x128 (pub:sysarch). VMEM 128 MiB (pub:pallas). Clock is not published: derived 197e12 / (4 x 128^2 x 2) =
1.503 GHz, we use 1.5 GHz.

Derived checks: 4 x 16384 x 2 x 1.5 GHz = 196.6 TFLOP/s; int8 x2 = 393.2; HBM 2 x 1024 x 3.2 Gb/s / 8 = 819.2 GB/s;
ICI 4 ports x 2 dirs x 50 GB/s = 400 GB/s.

The document exposes `v5e_chip` as a template so other designs (20.3, 20.4) reuse it.

```json5
// file: designs/tpu_v5e.json5
{
  schema: "kiln.hw/1.0",
  name: "tpu-v5e",
  family: "tpu_v5e",                                   // 06 platform key
  meta: {
    citations: {
      v5e: "Google Cloud TPU v5e docs, docs.cloud.google.com/tpu/docs/v5e (accessed 2026-10-04)",
      sysarch: "Google Cloud TPU system architecture page (MXU 128x128 before v6e, 256x256 on v6e)",
      pallas: "JAX Pallas TPU docs (VMEM 128 MiB, vreg shape (8,128) x 32-bit)",
      sbook: "JAX scaling book, How to Think About TPUs, jax-ml.github.io/scaling-book/tpus (v5e HBM 8.2e11 B/s; accessed 2026-10-04)",
    },
    claims: [
      { metric: "peak_ops.bf16", scope: "", value: 197e12, rel_tol: 0.01, source: "v5e" },
      { metric: "peak_ops.int8", scope: "", value: 393e12, rel_tol: 0.01, source: "v5e" },
      { metric: "offchip_bw", scope: "", value: 819e9, rel_tol: 0.01, source: "sbook" },  // v5e docs: 800 GiBps (assumed 819e9)
    ],
  },
  tech: "tsmc_n5",                                    // assumed (unpublished)
  params: {
    mxu_edge: 128, n_mxu: 4, clk: "1500MHz",          // pub:sysarch, pub:v5e; clk derived
    vmem_cap: "128MiB",                               // pub:pallas
    vmem_rd_ports: 3,                                 // assumed: 3 vreg loads/cycle (~18 TB/s, cf. harness 19.5 TB/s)
    hbm_kind: "hbm2e", hbm_cap: "8GiB", hbm_pin: "3.2Gbps", n_hbm: 2,   // 2 stacks assumed; 16 GB pub
    ici_lanes: 4, ici_lane_rate: "100Gbps",           // derived: 50 GB/s/dir/port from 400 GB/s bidir / 4 ports; lane split assumed
    tdp: "200W",                                      // assumed placeholder (unpublished)
    vpu_precisions: ["fp32@1", "int32@1"],            // assumed: no bf16 VALU on v5e
  },

  templates: {
    tensorcore: { kind: "cluster", body: {
      units: [
        { id: "mxu", count: "=n_mxu", kind: "matrix",
          geometry: { systolic: { rows: "=mxu_edge", cols: "=mxu_edge" } },
          dataflow: "weight_stationary",               // pub (TPU MXUs are weight-stationary systolic arrays)
          precisions: ["bf16*bf16+fp32", "int8*int8+int32@2"],
          local: [ { id: "wreg", holds: "b", capacity: "= 2 * mxu_edge * mxu_edge * 2", double_buffered: true } ], // assumed
          accumulate_in: "feed",                       // assumed: partial sums popped to vregs (MRB modelled as feed latency)
          feeds: { a: "vreg", b: "vreg", o: "vreg" } },
        { id: "vpu", kind: "vector", lanes: 128, sublanes: 8,   // pub:pallas (8,128)
          precisions: "=vpu_precisions", feeds: { any: "vreg" } },
        { id: "eup", kind: "special", lanes: 128,               // transcendental unit; width assumed
          functions: ["exp", "log", "recip", "rsqrt", "tanh", "sigmoid"],
          precisions: ["fp32@1"], feeds: { any: "vreg" } },
        { id: "spu", kind: "scalar", issue_width: 2, controls: ["mxu", "vpu", "eup"], // assumed width
          precisions: ["int32@1", "fp32@1"], feeds: { any: "smem" } },
      ],
      memories: [
        { id: "vreg", kind: "register_file", capacity: "128KiB", // 32 vregs x 4 KiB: count assumed, shape pub:pallas
          banks: 32, word_bits: 32768,
          ports: [ { dir: "read", count: 4, width_bits: 32768 }, { dir: "write", count: 2, width_bits: 32768 } ] }, // assumed
        { id: "vmem", kind: "scratchpad", capacity: "=vmem_cap", banks: 32, word_bits: 32768, // banks assumed
          ports: [ { dir: "read", count: "=vmem_rd_ports", width_bits: 32768 }, { dir: "write", count: 1, width_bits: 32768 } ] },
        { id: "smem", kind: "scratchpad", capacity: "1MiB", word_bits: 32,  // assumed
          ports: [ { dir: "rw", count: 2, width_bits: 32 } ] },
      ],
      networks: [
        { id: "vld", topology: "p2p", endpoints: ["vmem", "vreg"],
          link: { width_bits: "= vmem_rd_ports * 32768" } },          // vld/vst path
      ],
    } },

    v5e_chip: { kind: "package", body: {
      substrate: { kind: "silicon_interposer" },                 // assumed
      power: { cap: "=tdp", policy: "dvfs" },
      dies: [ {
        id: "die",
        clocks: [ { id: "core", freq: "=clk" } ],                 // no V/F published: fixed clock
        default_clock: "core",
        clusters: [ { id: "tc", use: "tensorcore" } ],
        networks: [
          { id: "dma", topology: "crossbar", endpoints: ["tc.vmem", "tc.smem"],
            link: { width_bits: 4096 } },                         // assumed; HBM is the limiter
        ],
        ports: [
          { id: "ici", count: 4, kind: "serdes", internal: "dma",  // 4 ports pub:v5e
            link: { phys: { type: "serdes", protocol: "ici_like", lanes: "=ici_lanes", lane_rate_bits_per_s: "=ici_lane_rate" } } },
          { id: "pcie", kind: "pcie", internal: "dma",
            link: { phys: { type: "serdes", protocol: "pcie", lanes: 16, lane_rate_bits_per_s: "16Gbps" } } }, // assumed Gen4 x16
        ],
        floorplan: { outline: { type: "auto" } },                 // die size unpublished: placer sizes it
      } ],
      mem_stacks: [ {
        id: "hbm", count: "=n_hbm", kind: "=hbm_kind", capacity: "=hbm_cap",
        io_width_bits: 1024, pin_rate_bits_per_s: "=hbm_pin",
        attach: { network: "die.dma" },                           // synthesizes hbm{i}_mc + hbm{i}_phy on the die
      } ],
    } },
  },

  system: { package: { id: "chip", use: "v5e_chip" } },
}
```

### 20.3 Google TPU v6e (Trillium), as a variant

Published (pub:v6e): 918 TFLOP/s bf16, 1836 TOP/s int8, 32 GB HBM at 1638 GB/s, ICI 800 GB/s bidirectional per
chip over 4 ports, 1 TensorCore, SparseCores present, 256x256 MXUs (pub:sysarch). The v6e page states 2 MXUs per
TensorCore, which would need a 3.5 GHz clock for 918 TFLOP/s; no Google page publishes the clock (checked
2026-10-04: v6e docs, system-architecture page, JAX scaling book). Owner ruling (08 §F): follow Google's page, 2 x 256^2 at
3.5 GHz (917.5 TFLOP/s), clock derived from the published peak.

Derived checks: 2 x 65536 x 2 x 3.5 GHz = 917.5 TFLOP/s; HBM 2 x 1024 x 6.4 Gb/s / 8 = 1638.4 GB/s;
ICI 4 x 2 x 100 GB/s = 800 GB/s.

```json5
// file: designs/tpu_v6e.json5
{
  schema: "kiln.hw/1.0",
  name: "tpu-v6e",
  family: "tpu_v6e",                                   // overrides the inherited key
  extends: "tpu_v5e.json5",
  meta: {
    citations: { v6e: "Google Cloud TPU v6e docs, docs.cloud.google.com/tpu/docs/v6e (accessed 2026-10-04)" },
    claims: [
      { metric: "peak_ops.bf16", scope: "", value: 918e12, rel_tol: 0.01, source: "v6e" },
      { metric: "peak_ops.int8", scope: "", value: 1836e12, rel_tol: 0.01, source: "v6e" },
      { metric: "offchip_bw", scope: "", value: 1638e9, rel_tol: 0.01, source: "v6e" },
    ],
  },
  params: {
    mxu_edge: 256, n_mxu: 2, clk: "3500MHz",          // edge pub:sysarch; 2 MXUs pub:v6e; clock derived (08 §F)
    hbm_kind: "hbm3", hbm_cap: "16GiB", hbm_pin: "6.4Gbps",   // 32 GB, 1638 GB/s pub:v6e; 2 x HBM3 split assumed
    ici_lane_rate: "200Gbps",                          // derived: 100 GB/s/dir/port; lane split assumed
    vpu_precisions: ["fp32@1", "int32@1", "bf16@2"],   // assumed bf16 VALU on v6e
    vmem_rd_ports: 6,                                  // assumed: scaled with MXU edge (harness: 45.6 TB/s)
    tdp: "300W",                                       // assumed placeholder
  },
  set: [
    { "board.chip.die#clusters+": [ {
        id: "sc", count: 2,                            // 2 SparseCores pub:v6e (count); internals assumed
        units: [ { id: "tile", count: 16, kind: "vector", lanes: 16,   // assumed
                   precisions: ["fp32@1", "int32@1", "bf16@2"],
                   ops: ["elementwise", "gather_scatter", "reduction", "sort_topk", "convert"],
                   feeds: { any: "spmem" } } ],
        memories: [ { id: "spmem", kind: "scratchpad", capacity: "8MiB", // assumed
                      ports: [ { dir: "rw", count: 16, width_bits: 512 } ] } ],
      } ] },
    { "board.chip.die.dma#endpoints+": ["sc*.spmem"] },
  ],
}
```

### 20.4 Four-chip TPU v5e slice (2x2) with ICI

Slice shape 2x2 is a supported v5e topology (pub:v5e). Each chip's on-die ICI router is a node of a 2D torus; in
size-2 dimensions the wraparound becomes a second parallel link (sec 10.1), so all 4 ports of each chip are used.
Whether Google wires 2x2 slices with wraparound is not published (assumed; open question).

Derived: per chip 4 links x 50 GB/s/dir; bisection of the 2x2 (cut one dimension) = 2 chip pairs x 2 links x
50 GB/s/dir = 200 GB/s per direction.

```json5
// file: designs/tpu_v5e_2x2.json5
{
  schema: "kiln.hw/1.0",
  name: "tpu-v5e-2x2",
  imports: { v5e: "tpu_v5e.json5" },
  meta: { citations: { v5e: "Google Cloud TPU v5e docs (slice shapes, 8 chips per host)" } },
  tech: "tsmc_n5",                                     // assumed
  system: {
    hosts: [ { id: "host", mem_capacity: "192GiB", mem_bandwidth: "200GB/s",   // assumed host
               launch_overhead: "20us" } ],                                 // assumed; 06 calibrates
    boards: [ {
      id: "tray",
      packages: [ { id: "chip", count: 4, layout: { grid: [2, 2] }, use: "v5e.v5e_chip" } ],
      networks: [ {
        id: "ici",
        topology: { type: "torus", dims: [2, 2] },      // pub:v5e 2D torus; 2x2 wrap assumed
        endpoints: [ { select: "chip*", at: "layout", ports: "die.ici*" } ],
        link: { phys: { type: "serdes", protocol: "ici_like", lanes: 4, lane_rate_bits_per_s: "100Gbps" } },
        router: { pipeline: 8 },                        // assumed ICI hop latency in core cycles
        routing: "default",                             // dimension order + dateline (2 VCs default on torus)
        features: { multicast: false, broadcast: false, in_network_reduce: [] },  // assumed
      } ],
      host_links: [ { host: "/host", to: "chip*.die.pcie",
                      link: { phys: { type: "serdes", protocol: "pcie", lanes: 16, lane_rate_bits_per_s: "16Gbps" } } } ],
    } ],
  },
}
```

### 20.5 Exotic: "ember", chiplets + HBM4-PIM + mesh NoC + 3D SRAM + CIM

Purpose: exercise every first-class feature. All numbers assumed unless tagged; HBM4 interface numbers are
JEDEC (pub:jedec).

- 2x2 compute chiplets (tsmc_n3e) on a silicon interposer, joined by UCIe-Advanced in a 2x2 package mesh.
- Each chiplet: 8x8 mesh NoC; 64 tensor tiles (64x64 MX systolic + 32-lane vector + 1.5 MiB SRAM) and a column of
  8 SRAM-CIM tiles at mesh column 8 (`layout_offset`), with multicast/broadcast.
- One SRAM die hybrid-bonded on top of each chiplet (64 MiB "l3" scratchpad, vertical link into the NoC).
- One HBM4 stack per chiplet, with a logic die (tsmc_n12) carrying one GEMV PIM unit per pseudo-channel.
- 2 harvested tiles on one chiplet; DVFS under a 700 W package cap.

```json5
// file: designs/ember.json5
{
  schema: "kiln.hw/1.0",
  name: "ember",
  meta: { citations: { jedec: "JEDEC JESD270-4 HBM4 (2048-bit interface, up to 8 Gb/s/pin)" } },
  tech: "tsmc_n3e",
  params: { tiles: 8, arr: 64, clk: "1.6GHz", tile_sram: "1.5MiB" },
  templates: {
    tile: { kind: "cluster", body: {
      units: [
        { id: "mx", kind: "matrix", geometry: { systolic: { rows: "=arr", cols: "=arr" } },
          dataflow: ["weight_stationary", "output_stationary"],       // mapper picks per op
          precisions: ["mxfp8_e4m3*mxfp8_e4m3+fp32", "mxfp6_e2m3*mxfp6_e2m3+fp32@2",
                       "mxfp4*mxfp4+fp32@4", "bf16*bf16+fp32@0.5", "int8*int8+int32", "bf16*mxfp4+fp32@0.5"],
          local: [ { id: "w", holds: "b", capacity: "= 2 * arr * arr", double_buffered: true },
                   { id: "acc", holds: "o", capacity: "= arr * arr * 4" } ],
          feeds: { a: "sram", b: "sram", o: "sram" } },
        { id: "vec", kind: "vector", lanes: 32, precisions: ["fp32@1", "bf16@2", "mxfp8_e4m3@2"],
          reduce_tree: true, feeds: { any: "sram" } },
        { id: "sfu", kind: "special", lanes: 8, functions: ["exp2", "rsqrt", "recip", "silu", "gelu"],
          precisions: ["fp32@1"], feeds: { any: "sram" } },
      ],
      memories: [ { id: "sram", kind: "scratchpad", capacity: "=tile_sram", banks: 16, word_bits: 512,
                    ports: [ { dir: "rw", count: 2, width_bits: 512, per_bank: false } ] } ],
    } },
    cim_tile: { kind: "cluster", body: {
      units: [ { id: "cim", kind: "cim", rows: 256, cols: 256, weight_sets: 4, weight_capacity: "32KiB",
                 style: "digital", precisions: ["int8*int4+int32", "int8*int8+int32"],   // 2048 / 1024 MAC/cycle (9.3)
                 near: { memory: "act", granularity: "per_bank" } } ],   // one CIM macro per act bank: count 16 implied
      memories: [ { id: "act", kind: "scratchpad", capacity: "512KiB", banks: 16, word_bits: 256,
                    ports: [ { dir: "rw", count: 2, width_bits: 1024 } ] } ],
    } },
    chiplet: { kind: "die", body: {
      clocks: [ { id: "core", freq: "=clk", base: "1.0GHz",
                  vf: [ { freq: "1.0GHz", voltage: "0.60V" }, { freq: "1.6GHz", voltage: "0.80V" } ] } ],
      default_clock: "core",
      floorplan: { outline: { type: "max_area", area: "400mm2", aspect: [0.8, 1.25] } },
      clusters: [
        { id: "t", use: "tile", layout: { grid: ["=tiles", "=tiles"] }, placement: { mode: "array" } },
        { id: "c", use: "cim_tile", layout: { grid: ["=tiles", 1] } },
      ],
      networks: [ {
        id: "noc", topology: { type: "mesh", dims: ["=tiles", "= tiles + 1"] },
        endpoints: [ { select: "t*.sram", at: "layout" },
                     { select: "c*.act", at: { layout_offset: [0, "=tiles"] } } ],
        link: { width_bits: 1024 },          // latency/energy derived from placed tile pitch (04)
        router: { pipeline: 2, vcs: 2, input_buffer_flits: 8 },
        routing: "default",
        features: { multicast: true, broadcast: true, in_network_reduce: ["fp32", "bf16"] },
      } ],
      ports: [
        { id: "ucie", count: 2, kind: "d2d", internal: "noc" },
        { id: "up", kind: "vertical", internal: "noc" },
        { id: "xlink", count: 8, kind: "serdes", internal: "noc",
          link: { phys: { type: "serdes", protocol: "custom", lanes: 8, lane_rate_bits_per_s: "224Gbps" } } },
      ],
    } },
  },
  system: { package: {
    id: "pkg",
    substrate: { kind: "silicon_interposer" },
    power: { cap: "700W", policy: "dvfs" },
    layers: [ { id: "base", index: 0, bond: "microbump", pitch_um: 36 },
              { id: "top", index: 1, bond: "hybrid", pitch_um: 9 } ],
    dies: [
      { id: "cc", use: "chiplet", layout: { grid: [2, 2] }, layer: "base",
        disabled: ["cc0_0.t3_5", "cc0_0.t6_1"] },                       // harvested tiles; routers kept
      { id: "sram", layout: { grid: [2, 2] }, layer: "top", tech: "tsmc_n5", role: "sram",
        over: "cc[{i}]", placement: { mode: "pinned", x: 0, y: 0 },     // relative to the die below
        floorplan: { outline: { type: "fixed", w: "18mm", h: "18mm" } },
        memories: [ { id: "l3", kind: "scratchpad", capacity: "64MiB", banks: 64, word_bits: 512,
                      ports: [ { dir: "rw", count: 8, width_bits: 1024 } ] } ],
        ports: [ { id: "down", kind: "vertical", internal: "l3" } ] },
    ],
    links: [
      { id: "v", count: 4, a: "cc[{i}].up", b: "sram[{i}].down",
        link: { phys: { type: "vertical", bond: "hybrid", pitch_um: 9 }, width_bits: 8192 } },
    ],
    networks: [ {
      id: "d2d", topology: { type: "mesh", dims: [2, 2] },
      endpoints: [ { select: "cc*", at: "layout", ports: "ucie*" } ],
      link: { phys: { type: "d2d", standard: "ucie_advanced", modules: 4, pin_rate_bits_per_s: "32Gbps" } },
      // 4 modules x 64 lanes x 32 Gb/s / 8 = 1024 GB/s per direction
    } ],
    mem_stacks: [ {
      id: "hbm", count: 4, kind: "hbm4", capacity: "36GiB", dies_high: 12,
      io_width_bits: 2048, pin_rate_bits_per_s: "8Gbps",                 // pub:jedec -> 2.048 TB/s per stack
      attach: { network: "cc[{i}].noc" },                               // controllers on the chiplet's NoC
      logic_die: {
        tech: "tsmc_n12", area_budget: "30mm2", power_budget: "8W",
        units: [ { id: "pim", kind: "matrix", geometry: { mma: { m: 1, n: 16, k: 16 } },  // GEMV engine
                   precisions: ["bf16*bf16+fp32", "fp8_e4m3*fp8_e4m3+fp32@2", "int8*int8+int32@2"],
                   clock: "pim_clk",
                   near: { memory: "hbm[{^i}]", granularity: "per_pseudo_channel", access_mode: "exclusive_per_bank",
                           residency: "weights", result_path: "write_back" } } ],
      },
    } ],
  } },
  clocks: [ { id: "pim_clk", freq: "500MHz" } ],
}
```

`near.memory: "hbm[{^i}]"` binds each PIM unit to its own stack (`^i` is the enclosing stack instance; a logic
die is not a path segment, its contents live directly under the stack instance). With
HBM4 defaults (32 channels x 2 pseudo-channels), each stack gets 64 PIM instances (count omitted, sec 9.1).

## 21. Cross-section dependencies

Assumptions this section makes about other sections' interfaces. Each owner should confirm or object.

| # | Other section | Assumption |
|---|---|---|
| D1 | 02 workload IR | `Precision` (6.1) and `OpClass` (6.3) are single enums in `kiln-ir`, shared by hardware and workload. 01 owns the names and storage-bit rules; 02 owns tensor tagging, per-tensor scaling metadata, and the `upcast_ok` flag. |
| D2 | 02 | 02 lowers every op to `OpClass` instances and decomposes composites (softmax, norms, attention) into classes; 02 provides per-`SpecialFn` vector-sequence costs for functions a special unit lacks. |
| D3 | 02 | Canonical loop-dim names for matmul are `m, n, k` (and conv dims for `Spatial` geometry); `Geometry` maps onto them per table 5.2. |
| D4 | 03 mapping | 03 consumes only `HwModel` (sec 16) plus kiln-phys annotations, never authoring structs. It uses `route`, `minimal_paths`, `staging_chains`, `feed_memories`, `peak_ops`. |
| D5 | 03 | 03 makes every "mapper decides" choice: dataflow from the declared set, carveout option per kernel, address-map choice and tensor placement, endpoint-to-router binding when `auto` and not placed by 04, near-memory residency, multicast use. |
| D6 | 03 | Power-cap throttling (sec 12) is applied by 03 per steady-state phase (Tier A, 03 sec 4.5) or per window (Tier B) using 04's power model; violations at `base` are `E-MAP-POWER-CAP` (owned by 03). |
| D7 | 03 | Implicit DMA (unlimited initiators) when no DMA blocks are declared; declared DMA blocks are contended resources in Tier B. |
| D8 | 03 | Near-memory units are ordinary compute nodes with implicit private feeds, constraints per sec 9.2 (`ExclusiveAllBank` blocks host traffic on that stack/PC for the kernel duration). |
| D9 | 04 physical | kiln-phys derives, per entity: area (when no footprint), access energy/latency/leakage for memories, per-op energy for units, link latency/energy/bandwidth from placed wire length and PHY kind, DRAM timing defaults and achievable-bandwidth fraction per `DramKind`, PHY shoreline length per kind/lanes/node, D2D reach, substrate size limits, reticle limit, default router pipeline/buffers. It returns `Derived<T>` with `Source` (16.1). |
| D10 | 04 | Technology node names in 13 resolve against 04's table; macro kinds (`sram_hd`, `edram`, ...) are 04 table keys; `dram_logic` nodes exist for PIM logic dies. |
| D11 | 04 | The placer honours `Placement` variants (`auto`, `pinned`, `region`, `edge`, `array`, `site`) and `DieFloorplan`; it reports E08xx (P) diagnostics in the `HwDiag` format. |
| D12 | 04 | V/F curves are inputs; when only `freq` is given, 04 supplies a node-default voltage and treats the clock as fixed. Voltages in reference designs are calibration targets. |
| D13 | 05 visualizer | Reads `HwModel.tree`, instance layout coordinates, placed floorplan, `level_hint`, `Derived` sources, and `meta.notes`/citations for provenance display. |
| D14 | 06 validation/API | 06 runs `meta.claims` checks (W-IR-1801) in CI for every reference design, stores evolution designs as inlined canonical JSON with `design_hash`, uses `kiln validate --format json` diagnostics as LLM feedback, and owns calibration of every `assumed` number in sec 20. Harness `kiln.hw/0` designs replay via the v0 migration. |
| D15 | 06 | Design-space search knobs (ranges for counts, sizes) are not part of the hardware IR; the evolution loop emits concrete designs. |
| D16 | 00 overview | Ids in 01 exclude `.` (path separator) and uppercase; this narrows the 00 grammar `[a-z0-9_.-]+` and requires a leading letter. |
| D17 | 07 prior art | The Stream/ZigZag "operational array + per-operand memory hierarchy" concepts map onto `Geometry::Spatial`, `feeds` and `local`; a Stream hardware YAML importer is not planned (harness v0 import covers our existing designs). |
| D18 | 03 | `exec_model` (`host_launched`, `device_queued`, `static_dataflow`) is a document/package attribute here; overhead constants per model live in 03/06 calibration data. |
| D19 | 04 | Field set aligned with 04 sec 13 "needs from 01": memory `implementation` kinds use 04's names; links carry `metal` (class preference), `repeated` (sizing), `swing`, `pipelined`; `PowerCap.level` and `ThermalSpec` (tj_max, cooling class) form the envelope; harvesting = spares; `PowerOverride` carries `area`, `energy_per_op`, `ctrl_ge` with `source`. 04's default die aspect [0.75, 1.33] applies to `Outline::Auto`; 01's (0.5, 2.0) applies to soft blocks. |
| D20 | 06 | 06's `E-IR-*` names map to 01 codes (`E-IR-UNPRICED` = E-IR-1101 unpriced override, E-IR-1104 unpriceable field); `stream_compat` and `search` profiles are defined in 18.3; `kiln-ir::arb` in 18.4; `family` field is 06's "calibration family". |
| D21 | 05 | 05 gets instance naming (sec 15), `level()`, layout coordinates, and `diff()` from 16.2. |

## 22. Open questions

1. **TPU v6e MXU count/clock.** Google's v6e page says 2 MXUs per TensorCore, which needs 3.5 GHz for 918 TFLOP/s.
   Resolved by owner ruling (08 §F): 2 x 256^2 at 3.5 GHz. Needs a measured clock or a better source; it changes vreg/VMEM
   bandwidth per MXU.
2. **TPU v5e HBM bandwidth.** Current docs say "800 GiBps" (859 GB/s); the JAX scaling book and the harness use
   819 GB/s (8.2e11). 819 GB/s is kept, marked assumed. Measured GEMV reaches 0.72-0.77 TB/s (06 sec 1), which does
   not discriminate. Resolve with the v5e measurements in progress.
3. **2x2 slice wiring.** Are size-2 torus dimensions wired with a doubled link, a single link, or no wraparound on
   v5e 2x2 slices? Affects the 4-chip example's bisection by 2x.
4. **A100 L2 internals.** Inter-partition link bandwidth, per-SM crossbar port width, slice count/ways, and which
   TPCs are harvested on a given part are assumed. Microbenchmarks on the Colab A100 (06) should pin the first two.
5. **Hash granularity.** Hashing the compact canonical form (sec 17) means `count: 108` and 108 hand-written SMs hash
   differently. Acceptable for evolution dedup? The alternative (expanded hash) costs O(instances) per design.
6. **Instruction-issue limits.** GPU tensor-core utilization is often bounded by warp issue / operand-collector
   limits rather than the structures modelled here. Is `Pipeline.issue_overhead` plus 04/06 calibration enough,
   or does 01 need an explicit issue model (schedulers per SMSP, issue slots)?
7. **Composite op classes.** Should units be able to declare fused composites (`softmax`, `attention`) at a rate, for
   designs with dedicated attention engines, or must everything decompose into the 12 base classes?
8. **Cache coherence semantics.** `coherent_with` (A100 L2 copies) is declarative only. Tier A needs a rule for
   effective capacity when partitions duplicate lines (assume no duplication? worst case?). Owner: 03.
9. **Per-die power caps in chiplet packages.** Current model: one cap per power domain with DVFS over listed clocks.
   Do we need per-die thermal coupling (hotspot-driven throttling) in v0, or is package-level enough?
10. **Implicit DMA default.** Unlimited initiators flatter designs that would need many DMA engines in silicon (TPU
    has a fixed number of DMA engines). Should the default synthesize one DMA per memory controller instead?
11. **Register-file and vreg assumptions for TPUs.** Vreg count, VMEM port count and SMEM size are unpublished;
    they bound vector-heavy ops (softmax, norms) on v5e/v6e. Calibrate against measured per-op times.
12. **Analog CIM.** `CimSpec.adc_bits` is required and priced, and an ADC below the lossless boundary warns
    (W-IR-0610), but accuracy/noise effects are out of scope; should analog CIM be rejected in v0 to avoid rewarding
    designs whose numerics would fail?
13. **Host model.** `Host` is minimal (capacity, bandwidth, launch overhead). Is that enough for KV-cache offload or
    weight streaming studies, or does 02/03 need host compute?
14. **Precision registry ownership.** Resolved by 00 decision 4: merged table in 6.1 (storage variants included).
