# 03: Mapping and engines (kiln-cost, kiln-map, kiln-sim)

Status: spec v0. Follows 00-overview.md. Covers the intra-unit cost model, the mapper, Tier A (analytical) and
Tier B (event-driven) engines, multi-chip execution, near-memory execution, invariants, the result model handed
to 05/06, and calibration hooks.

Types below are Rust sketches: field names and semantics are normative, exact signatures are not. Where this
section assumes a type owned by 01/02/04 it says so under "Cross-section dependencies"; those sections are the
authority for the type, this section is the authority for how the engines use it.

---

## 0. Design rules learned from Stream and the A100 comparison

Each rule below is a direct response to a defect we hit. Each has an enforcement point named in this section.

| # | What went wrong (Stream fork / A100 v0) | kiln rule | Enforced in |
|---|---|---|---|
| L1 | Single-node throughput bound took max over cores, ignored link occupancy: decode GEMV reported 3x faster than HBM allows | Every resource the mapping touches (each directional link, each NoC hop, each memory port, each HBM channel, each inter-chip lane) carries busy time; op time is bounded by the max over **all** of them | 4.2, invariant I3 |
| L2 | Results depended on PYTHONHASHSEED | No hash-order iteration; integer time; stable tie-breaks by typed index; result hash tested across thread counts | 5.6, I11 |
| L3 | gcd-based core splitting left cores idle | Splits are explicit slice lists; uneven (floor/ceil) splits are first-class; no factorization requirement | 2.3, 3.3 |
| L4 | Every memory tile routed to every compute core; equal-bandwidth buses merged into one resource | Transfers are derived from affine footprint intersection per (producer slice, consumer slice); routes are explicit link-id paths; resources are identified by instance id, never by attribute equality | 3.5, I4 |
| L5 | Loop order emitted by dim position pinned outputs and became infeasible | Loop order is a searched decision with explicit partial-sum accounting; infeasibility is reported with the binding memory and the loop that caused it | 2.4 |
| L6 | MILP scheduling with AIE column assumptions | No MILP in any inner loop; list scheduling + search over mapping moves; no architecture-family assumptions in the scheduler | 3.8 |
| L7 | Decode ~20% too fast (A100) | DRAM efficiency is a structured model (refresh, turnaround, row misses, residual) calibrated on streaming microbenchmarks, not peak | 5.3, 9 |
| L8 | Tiny ops 2-3x too fast (no per-launch overhead) | Execution-model overhead terms: launch, minimum kernel duration, inter-kernel gap, sync, collective setup | 4.4 |
| L9 | Compute-bound shapes 0.90x to 1.36x | Structural terms first: wave/array-shape quantization across units, pipeline fill, clock under power cap. Only then a bounded residual | 2.3, 4.5, 9 |
| L10 | Real GPU throttles to ~1290 MHz at 400 W | Power-capped DVFS fixed point is part of every result; clock is an output, not an input | 4.5 |

---

## 1. Crate responsibilities and data flow

```
kiln-ir (HwDoc -> HwModel, WorkloadDoc)  +  kiln-wl (bind, lower, partition -> PartitionedProgram, 02)
   |                      \
   v                       v
kiln-phys (floorplan -> link lat/bw/energy, per-access energy, V/f, P_static, thermal)
   |
   v
kiln-map  --uses-->  kiln-cost (intra-unit loop-nest mapping, memoized)
   |  produces Mapping (serializable) + LoweredGraph (tasks over resources)
   v
kiln-sim  (Tier A: analytical, Tier B: event-driven), shared ResourceTable and LoweredGraph
   |  produces SimResult (+ Trace for Tier B)
   v
kiln-trace (result + trace model)  -->  kiln-viz (05), kiln-py / kiln-cli (06)
```

The key interface is the **LoweredGraph**: a DAG of `Task`s (compute slice, transfer chunk-group, NMP command
batch, reduce, sync, launch) each annotated with the resources it occupies and its demand on each. Tier A and
Tier B consume the same LoweredGraph. This is what makes the Tier A / Tier B bound relation (4.6) provable
rather than hoped for: they disagree only in how they compose identical per-resource demands.

Crate split decision: keep the overview's split. `kiln-cost` stays separate from `kiln-map` because its cache is
keyed by unit template, not by design, and is reused across designs in an evolution run (and shipped as a
persistent cache file). `kiln-sim` holds both tiers because they share the ResourceTable, LoweredGraph, clock
model, and invariant checker.

---

## 2. Intra-unit cost model (`kiln-cost`)

### 2.1 Decision: Rust-native port of ZigZag/LOMA concepts; Python ZigZag as differential oracle

We port the concepts (loop-relevance, per-operand memory hierarchy, spatial unrolling, LOMA loop-order
enumeration, port-sharing stall model) to Rust and do not embed Python ZigZag in the engine.

Reasons:
1. Speed. Tier A must finish a design-layer in <= 50 ms. A Python ZigZag call costs tens of ms to seconds per
   layer-core pair; one design-layer may need 10-50 distinct (unit, shape) evaluations on cold cache.
2. Determinism. The Stream fork showed hash-order sensitivity in the same Python stack. Rust with IndexMap and
   integer cycle arithmetic removes the class of bug.
3. Structural freedom. ZigZag's hierarchy assumes per-operand trees with fixed operand names (I1/I2/O) and
   regular arrays. kiln needs ragged spatial splits, MX scale streams, more than two inputs (fused ops), NMP
   units, and sparse/irregular array shapes. Extending ZigZag means forking it.
4. Embedding Python in the Rust core violates hard requirement 7.

ZigZag remains a **differential oracle** (06 owns the harness): on a corpus of (accelerator, op) pairs that
ZigZag can express, kiln-cost run with the mapping ZigZag chose must reproduce ZigZag's per-level access counts
exactly and its energy and latency within 1%. Separately, kiln-cost's searched optimum must be no worse than
ZigZag's optimum by more than 2% on the same objective (it may be better: our search space is a superset). Any
disagreement is triaged to a named cause before merging.

### 2.2 Inputs

```rust
/// One affine op instance as seen by one compute unit (after kiln-map has sliced it).
pub struct OpNest {
    pub kind: OpKind,                    // Gemm, BatchedGemm, Conv, Elementwise, Reduce, Softmax, AttnFused, ...
    pub dims: SmallVec<[LoopDim; 8]>,    // name, size (u64), kind: Parallel | Reduction
    pub operands: SmallVec<[Operand; 4]>,// >= 1 inputs, exactly 1 output (fused groups lower to several OpNests)
    pub macs_per_point: u32,             // 1 for GEMM; >1 for e.g. complex; 0 for pure data movement
    pub vector_ops_per_point: u32,       // non-MAC work (exp, max, scale) for vector units
}
pub struct Operand {
    pub tensor: TensorId,
    pub role: Role,                      // Input | Weight | Output | Accum
    pub access: AffineMap,               // dims -> tensor coords (from 02); gives relevance per dim
    pub dtype: Precision,                // 01 §6.1 registry, incl. MX block formats
    pub source: Residency,               // which memory level/instance the slice lives in before the op
    pub sink: Option<Residency>,         // for outputs: where it must end up
}
pub enum Relevance { Relevant, Irrelevant, PartiallyRelevant { stride: u32 } } // per (operand, dim), derived from AffineMap
```

The unit template (`ComputeUnitTemplate`) is built by kiln-cost from 01's `HwModel` `UnitInst` (`Geometry`,
`PrecisionMode`s with `macs_per_cycle` = geometry product x `rate`, `Pipeline`, `local` buffers, `staging_chains`): spatial axes (e.g. `rows`, `cols`, `lanes`, any count
of axes, any sizes), allowed dim-to-axis bindings (or "any"), supported MAC modes per precision with
`macs_per_cycle`, accumulator precision, pipeline depth, and an ordered per-operand memory hierarchy
(`MemLevel { instance, capacity_B, read_bw_Bpc, write_bw_Bpc, ports: [PortId], served_operands, double_buffer:
bool, e_read_J_per_B, e_write_J_per_B, latency_cycles }`). Bandwidth is per port per cycle of the level's clock
domain; energies come from 04.

### 2.3 Spatial mapping (incl. ragged splits and array-shape quantization)

A spatial mapping binds each array axis `a` to a list of `(dim, unroll)` pairs with `prod(unroll) <= size(a)`.
Multiple dims may share an axis (e.g. `cols <- N:16 x B:4`), and a dim may span axes.

Quantization and raggedness are explicit, never folded into a scalar efficiency:
- **Array-shape quantization.** If dim `d` with size `S_d` is unrolled by `u_d`, the number of spatial
  iterations is `ceil(S_d / u_d)`. The last iteration has `r_d = S_d - (ceil(S_d/u_d) - 1) * u_d` active lanes.
  Cycles count every iteration as full; useful MACs count actual points. `spatial_util = useful_macs /
  (issued_macs)` is reported, not assumed.
- **Ragged spatial splits.** For non-divisible sizes, kiln-cost evaluates two temporal tile classes (full and
  remainder) and sums their costs, instead of padding the whole nest (ZigZag pads). The remainder class is
  costed with its own temporal mapping (often a different loop order is optimal for a sliver).
- **Wave quantization across units.** When kiln-map splits an op over `n` identical units into slices that are
  not all equal, each slice size forms its own cost class, and op time is the max over units (4.2). This is the
  structural explanation of GPU "wave" effects (108 SMs, tile counts not divisible by 108), and must be
  modelled before any residual efficiency is fitted.

Candidate generation (per unit template, per OpNest):
```
for each injective-ish binding of dims to axes allowed by the template:          // |D|^|A|, D<=8, A<=4
    for each axis: choose unroll for each bound dim from
        { min(S_d, size(a)) , largest divisor of S_d <= size(a), size(a)/k for k in {1,2,4} }   // <= 3 options
    drop candidates whose spatial_util < 0.5 * best_util_seen    (keep >= 1 per binding)
keep top-K_s (default 8) by (spatial_util, then lower peak per-cycle input bandwidth demand)
```
Complexity: O(|D|^|A| * 3^|A|) cheap evaluations, bounded at ~10^4, typically < 300 after the allowed-binding
filter.

### 2.4 Temporal mapping, loop order, partial sums, double buffering

Representation (LOMA-style): the temporal loop nest is a sequence of `(dim, factor)` from innermost to outermost
whose product per dim equals `ceil(S_d / spatial_unroll_d)` (or the full/remainder class size). Each operand
assigns a contiguous range of the sequence to each of its memory levels, innermost first ("memory allocation").

Allocation rule (bottom-up fill, as LOMA): for each operand, walk loops innermost-out, greedily keep a loop at
the current level while the tile footprint (from `AffineMap`, accounting for partial relevance/halo) fits that
level's capacity, divided by 2 if the level is double-buffered for this operand and buffering is chosen on.
Levels shared by several operands are filled jointly: capacity is split by the candidate's per-level
`capacity_share` vector, searched over {even, proportional-to-footprint, output-first} (3 options).

Loop order is a decision, not a consequence of dim position (L5). Order matters only through: (a) which loops
sit above each operand's level boundary and are irrelevant to it (reuse), and (b) for outputs, whether a
reduction loop sits above the output's residency level (partial sums).

**Partial-sum accounting.** If any Reduction loop with factor `f > 1` is above the output's level `l`, the
output tile at `l` is written `f` times and read back `f - 1` times **at accumulator precision** (fp32 for
bf16/fp8/MX MACs unless the template declares in-array accumulation at lower precision), and the final write
converts to the output dtype (conversion cost from the unit's vector path, or zero if the template declares
fused down-conversion). Both directions consume bandwidth at level `l+1` and its link, and energy at both.
Infeasibility rule: an output tile whose accumulator footprint exceeds every level that can hold psums is
infeasible; the error names the level, capacity, required footprint, and the reduction loop that forced it,
with hint "move <dim> inside or split <dim> across units".

**Order enumeration with pruning.**
```
fn search_temporal(nest, spatial, hier) -> best:
    factors  = prime factorization of each temporal dim size (full + remainder classes)
    classes  = merge identical primes of the same dim  (multiset permutations, not permutations)
    LB       = compute_cycles(spatial) ; also traffic LB = compulsory bytes per level / bw
    best     = heuristic_seed(nest)          // output-stationary + reduction-innermost, weight-stationary, input-stationary
    for perm in multiset_permutations(classes) with prefix pruning:
        // prefix pruning: after fixing the inner k loops, the allocation of levels whose
        // range is fully determined is fixed; compute partial cost (stalls+energy so far) + LB of rest
        if partial_cost(prefix) + LB_rest >= best.cost: prune subtree
        // equivalence pruning: two adjacent loops that are both relevant to every operand at the
        // same allocation level commute; canonicalize by dim index to visit one representative
        alloc = allocate(perm); cost = evaluate(alloc)
        best = min(best, cost) by (objective, then canonical order key)   // stable tie-break
    return best
```
Worst case is multinomial in the number of prime factors (LOMA's known blow-up). Bounds: (i) primes of the same
dim larger than the innermost level's capacity are coarsened into one loop; (ii) a node budget (default 20k
evaluated permutations per (spatial candidate)); when exhausted, the search returns the best found plus a flag
`search_truncated = true` that propagates to the result. Typical LLM GEMMs (power-of-two-heavy dims, 3-4 dims)
finish in 1-5 ms per cold key.

**Double buffering.** Per (operand, level) the mapping chooses buffered or not. Buffered halves usable capacity
and lets the refill of tile `i+1` overlap compute on tile `i`; not buffered serializes refill and compute.

### 2.5 Latency per access level (stall model)

For each operand `o` and level boundary `l -> l-1` (refill direction; also writeback for outputs):
- `period_cycles(o,l)` = compute cycles between successive refills of `o` at level `l-1` (product of loops
  inside level `l-1`'s range for `o`, times cycles per innermost iteration).
- `bytes_per_period(o,l)` = tile footprint (with halo) at `l-1`, plus psum read/write at accumulator width.
- Per physical port `p` of the boundary: `demand_cycles(p) = sum over operands sharing p of bytes_per_period /
  bw(p)` (port sharing is summed per period, the ZigZag "combine ports" behavior, but computed on actual
  periods aligned to the least common period).
- Stall per period = `max(0, demand_cycles - period_cycles)` if every operand on the port is double-buffered,
  else `demand_cycles` for the non-buffered operands plus the buffered excess.
- Plus onload (first fill) and offload (last drain) of every level once, not overlapped.

`unit_cycles = issue_cycles + sum stalls + onload + offload + pipeline_depth`, where
`issue_cycles = prod(ceil-based temporal factors) * ceil(macs_per_point / macs_per_cycle_per_lane)`.

kiln-cost returns per-boundary byte totals **separately from cycles**. kiln-map attaches those bytes to the
actual physical resources (memory ports, the tile-local links, and, for the outermost level, the NoC route to
the home memory). The intra-unit cycles assume the boundary into the unit's outermost level is served at
`bw_assumed`; Tier A and Tier B then re-check that boundary against the real shared resources (L1). The
intra-unit model never claims bandwidth it does not own.

### 2.6 Energy per access level

`E = sum_ops(issued_macs * e_mac(mode)) + sum_{o,l} (reads(o,l) * bytes * e_read(l) + writes(o,l) * bytes *
e_write(l)) + vector_ops * e_vec + conversion_ops * e_conv`. Padding MACs from quantization are charged at
`e_mac_idle(mode)` (clock-gated lanes; value from 04, default `0.1 * e_mac`, marked assumed). Static energy is
not added here; it is a system term applied over makespan (4.7).

### 2.7 Precision, mixed precision, MX overheads

- Each template lists `MacMode { a: Precision, b: Precision, acc: Precision, macs_per_cycle_per_lane, e_mac_J }`. An
  op's mode is the cheapest listed mode whose input types are both >= the operand types in a
  precision-dominance order the template provides. If none exists the op is unmappable on that unit (structured
  error), never silently upcast. Explicit upcast is a mapping choice (`CastPolicy`) that adds a vector pass.
- Bytes of a tensor slice of `n` elements: `ceil(n * bits / 8)` plus MX scale bytes.
- MX formats (MXFP8/6/4, MXINT8): block size `k` (default 32, along the block axis from 02), one E8M0 scale per
  block, so scale bytes = `ceil(n / k)`. The scale tensor is a separate operand stream in the hierarchy: same
  loops, footprint divided by `k` along the block dim. Tile boundaries along the block dim must be multiples of
  `k` (constraint in allocation; the remainder class handles the tail).
- MX compute: if the template declares `mx_native` for the mode, scales are applied in-array (cost is in
  `macs_per_cycle` and `e_mac`) plus scale-read energy. Otherwise kiln-cost lowers to integer/fp MACs per block
  plus a vector pass per (output element, K-block): `vector_ops += M*N*ceil(K/k)` multiply-adds at the
  accumulator precision.
- fp8 e4m3 vs e5m2 differ only in mode lookup; precision is per tensor.

### 2.8 Memoization

```rust
pub struct CostKey {
    unit_template: TemplateHash,   // content hash of the template incl. hierarchy, energies, clocks domain ratios
    shape: ShapeClass,             // op kind + exact dims + relevance signature + dtypes + Residency levels + cast policy
    objective: Objective,          // Latency | Energy | EDP | Weighted(w_lat, w_e) quantized
}
pub struct CostResult {
    pub spatial: SpatialMapping, pub temporal: TemporalMapping,
    pub issue_cycles: u64, pub stall_cycles: u64, pub fill_drain_cycles: u64,
    pub useful_macs: u64, pub issued_macs: u64,
    pub accesses: Vec<LevelAccess>,          // (level, operand, rd/wr, bytes) per boundary
    pub energy_J: EnergyBreakdown,
    pub search: SearchStats,                 // evaluated, pruned, truncated flag, wall time
}
```
Shape classes use **exact** dims, not buckets: bucketing hides quantization effects (L9). Hit rates stay high
because slicing over identical units produces few distinct shapes (at most 2 per split dim with floor/ceil).
The cache is a per-thread LRU in front of a global `RwLock<IndexMap>`; insertion
races are harmless because the search is a pure function of the key (identical values). The cache serializes
to a versioned file keyed by kiln-cost version so an evolution run warms once. Templates edited by the evolution
loop produce new hashes; unchanged templates in a mutated design hit.

---

## 3. Mapper (`kiln-map`)

### 3.1 Mapping representation (serializable)

The mapping is a plain data object, JSON-canonical like the IRs, produced by the default heuristic, by search,
by the RL/learned mapper, or by hand, and validated before use.

```rust
pub struct Mapping {
    pub version: u32,
    pub design_hash: Hash, pub workload_hash: Hash,
    pub mesh: MeshEmbedding,                   // logical parallel mesh axes -> physical chip ids (3.2)
    pub ops: IndexMap<OpId, OpPlacement>,
    pub tensors: IndexMap<TensorId, TensorPlacement>,
    pub groups: Vec<ExecGroup>,                // fusion + pipelining + launch boundaries (3.6)
    pub collectives: IndexMap<OpId, CollectivePlan>,
    pub routing: RoutingPolicy,                // default policy; per-transfer overrides in `routes`
    pub routes: IndexMap<TransferId, Route>,   // explicit overrides only
    pub priorities: Option<Vec<(TaskSelector, i32)>>,
    pub unit_overrides: IndexMap<(OpId, SliceIx), UnitMappingOverride>, // optional fixed intra-unit mapping
}
pub struct OpPlacement {
    pub target: Target,                        // Units(UnitSetId) | Nmp(NmpSetId) | Host (unmodelled; error in v0)
    pub split: Vec<SplitAxis>,                 // ordered; product of parts = #slices
    pub slice_to_unit: Vec<UnitId>,            // len = #slices; many slices may share a unit (time-multiplexed)
}
pub struct SplitAxis { pub dim: DimName, pub parts: Vec<u64> }   // explicit sizes, sum = dim size; uneven allowed
pub struct TensorPlacement {
    pub home: Vec<(TensorRegion, MemInstanceId)>,  // regions partition the tensor; region = box in tensor coords
    pub interleave: Option<Interleave>,            // over channels/banks/stacks: granule_B + instance list
    pub lifetime: LifetimePolicy,                  // Resident | Streamed | Spilled{to}
}
pub enum Route { Path(Vec<LinkId>), Split(Vec<(Vec<LinkId>, Ratio)>) }     // Ratio sums to 1
pub enum RoutingPolicy { DimensionOrder, ShortestWeighted { tie: TieBreak::LowestLinkIx }, Ecmp { max_paths: u8 } }
pub struct CollectivePlan {
    pub algo: CollectiveAlgo,                  // Ring | BiRing | Tree | DoubleBinaryTree | Ring2D{rows} | Hierarchical(Vec<CollectiveAlgo>) | Direct
    pub order: Vec<ChipId>,                    // ring/tree order over participants
    pub chunks: u32,                           // pipelining chunks
    pub reduce_on: ReduceTarget,               // ComputeUnit | DmaEngine | Nmp | SwitchInNetwork
}
```
Validation (structured errors, LLM-readable): splits partition every dim exactly; regions partition each tensor;
every route is a connected path from source instance to destination instance in the expanded interconnect
graph; units support the op kind and precision; capacities hold under the liveness schedule (3.7); collective
participants match the parallelism group from 02.

### 3.2 Pipeline overview

```
map(design, workload, policy) -> (Mapping, LoweredGraph)
 1. mesh embedding                  (3.2.1)  multi-chip only
 2. per-chip instantiation of 02's PartitionedProgram (collectives already inserted by kiln-wl::partition);
    only 02 §9.1's volume-preserving rewrites allowed
 3. grouping: fusion + pipelining choices (3.6)
 4. per-op partition across units  (3.3)
 5. tensor placement + KV cache     (3.4)
 6. transfer derivation + routing   (3.5)
 7. intra-unit costing via kiln-cost for every distinct (template, slice shape)
 8. scheduling: priorities + liveness + capacity check (3.7)
 9. lowering to LoweredGraph (tasks x resources)
```
Steps 3-6 are where policies make choices; steps 7-9 are deterministic functions of the Mapping. A policy may
return a full Mapping or only some fields; missing fields are filled by the default heuristic.

#### 3.2.1 Mesh embedding

Logical parallel axes (TP, SP/CP, EP, PP, DP; sizes from 02) are embedded into the physical chip graph (01).
Objective: minimize `sum_axis volume(axis) * dilation(axis)`, where `volume` is bytes per step from the
collective plan's analytical cost and `dilation` is the mean bottleneck hop count between ring/tree neighbours.
Algorithm: for <= 16 chips exhaustive over axis-to-dimension assignments of the physical topology's natural
dims (torus/mesh/switch tiers), else greedy: assign highest-volume axis to the highest-bandwidth tier first
(e.g. TP inside an NVLink/ICI domain), then recurse. O(axes! * chips) for the small case.

### 3.3 Op partitioning across units (spatial)

Candidates for an op on a unit set of `n` units (identical or heterogeneous):
- Split dims: parallel output dims first (M/N/batch/heads); a reduction-dim split (K) is allowed and inserts a
  reduce task (on a unit, an NMP site, or in-network) plus the extra psum traffic at accumulator precision.
- Part counts: for each split dim set, choose part counts `p_1 * ... * p_j <= n` maximizing used units, **not**
  requiring divisibility: for dim size `S` and `p` parts, sizes are `floor(S/p)` and `ceil(S/p)` (L3), with
  alignment to the unit's spatial unroll granule and MX block size when that reduces padding.
- Heterogeneous units: parts proportional to each unit's effective throughput for this op (from kiln-cost),
  rounded to granules, then fixed-point corrected so the slowest slice is minimized (water-filling, O(n log n)).
- Time-multiplexing: `#slices > #units` is allowed (e.g., to make the double buffer at the shared level fit).

Scoring uses a **local Tier A estimate** (4.2 restricted to the op): max over the op's resources of busy time.
Default heuristic tries <= 32 candidates per op; complexity O(32 * cost lookups) per op.

### 3.4 Tensor placement and KV cache

- Weights: home is chosen by the residency policy: if total weights + KV fit in on-chip SRAM across units
  (SRAM-rich designs), place each weight slice in the memory local to the unit that consumes it (Resident);
  else home in off-chip memory, interleaved over channels/stacks with the design's interleave granule, and
  Streamed into unit memories. The mapper never places a slice so that its consumer's route crosses a link the
  slice does not need (L4).
- Activations: home is the memory nearest (by route cost) to the producing slice; consumers pull their
  footprint. If a fusion group keeps an intermediate on-chip, it has no off-chip home.
- KV cache (02 declares KV tensors, shapes, sequence length, paging granularity): placement options are
  `OffChip{interleave}`, `OnChipSram{units}`, `NmpBanks{sites}`, and split by head across the TP group (follows
  the attention split). KV append writes are explicit transfers each decode step; KV reads are streamed with
  the attention op. Paged KV is modelled as page-granular regions with a configurable fragmentation fraction
  (capacity only, no performance effect in v0).
- Capacity is checked under the liveness intervals of 3.7; on overflow the heuristic spills the largest
  lowest-reuse tensor to the next level and records `Spilled{to}`; if nothing fits, structured error naming the
  memory and its live set at the peak.
- KV-capacity query for 02's continuous mode (02 §8.4): `kv_capacity(design, plan) -> bytes per rank` = capacity
  of the KV home memories minus resident weights and reserved activations under the chosen placement.

### 3.5 Transfer derivation and routing (no all-to-all fan-out)

For each edge (producer op P, consumer op C, tensor T):
```
for each consumer slice c:
    need_c = footprint(C.access_T, slice_box(c))              // affine image, exact boxes; halos included
    for each region r of T's current location (producer slice outputs, or home regions):
        overlap = need_c ∩ r                                    // box intersection, O(dims)
        if overlap nonempty: emit Transfer{src: loc(r), dst: input_level(c), bytes: |overlap| * bytes/elem}
```
Complexity: O(#producer regions x #consumer slices x dims); with sorted-interval sweep per split dim it drops to
O((P + C) log(P + C)) for the common grid-split case. Multicast: identical `overlap` boxes going to several
consumers from one source become one MulticastTransfer whose route is a tree (Steiner heuristic: union of
shortest paths, deduplicated per link) if the interconnect declares multicast support, else separate unicasts.

Routing over the actual expanded interconnect graph (01): nodes are memory instances, unit ports, routers,
switches, chip I/O; edges are **directional link instances** with bandwidth, latency, energy from 04. Default
`ShortestWeighted` uses Dijkstra with edge weight `bytes / bw + latency`, tie-broken by lowest link index; for
mesh NoCs that declare it, `DimensionOrder` (XY) is used to match hardware. `Ecmp` splits a transfer across up
to `k` edge-disjoint shortest paths in proportion to bottleneck bandwidth. Links are never merged by attribute
equality (L4); two buses with equal bandwidth are two resources. Per-hop router traversal is a resource too
(router crossbar port), so NoC hops derived from placement are all charged.

### 3.6 Fusion, layer pipelining, execution groups

```rust
pub struct ExecGroup {
    pub ops: Vec<OpId>,
    pub kind: GroupKind,          // Single | Fused { on_chip: Vec<TensorId> } | Pipelined { stages: Vec<Stage>, microbatches: u32 }
    pub launch: LaunchKind,       // HostLaunch | DeviceQueued | StaticProgram  (selects overhead terms, 4.4)
    pub barrier_after: bool,
}
```
- Fusion: elementwise/normalization/activation ops fuse into their producer when the intermediate tile fits the
  producer unit's output level (implemented for stacks with `fuse_elementwise`, 02 §7.4.1: one group per
  contraction with its elementwise neighbours, cross-node edges inside it streaming as intra-op edges do); attention may fuse (QK^T, softmax, AV) into one flash-style nest when the unit
  template can hold the running max/sum state; the fused nest is lowered to multiple OpNests sharing loops.
- Layer pipelining (spatial dataflow): consecutive ops assigned to disjoint unit sets with tiles streaming
  between them; requires that the per-stage tile order is compatible (consumer's outer loops consume in
  producer's emission order). The mapper checks compatibility from the temporal mappings and inserts a reorder
  buffer requirement otherwise (capacity cost).
- Layer-by-layer: each group uses all units, barrier between groups. Both are always evaluated by the default
  heuristic for each candidate region of the graph (two candidates, pick lower Tier A).

### 3.7 Scheduling and liveness (no MILP)

Scheduling is list scheduling over the LoweredGraph with priority = longest remaining path (Tier A per-task
durations), tie-broken by task index. Liveness intervals come from that schedule; capacity checks use them.
Why not MILP: (1) it is exponential in the worst case and its wall-time varies with solver heuristics, breaking
the Tier A budget and determinism; (2) the Stream MILP encoded AIE column assumptions that silently do not hold
for other topologies; (3) list scheduling has a known quality bound (Graham: <= (2 - 1/m) x optimal for
identical machines) and, more importantly, the bound relation in 4.6 is stated against the produced schedule.
Search over schedule alternatives happens in the outer loop via `priorities` moves (3.8), where it composes
with every other mapping decision. A MILP/CP solver may be used offline in a test (e.g., to measure the
heuristic's gap on small cases) and never in the engine.

### 3.8 Search interface

```rust
pub trait MappingPolicy: Send + Sync {
    fn name(&self) -> &str;
    fn propose(&self, ctx: &MapCtx, seed: u64) -> Result<Mapping, MapError>;
}
pub trait MappingSearch: Send + Sync {
    fn search(&self, ctx: &MapCtx, init: Mapping, eval: &dyn Fn(&Mapping) -> Score, budget: Budget, seed: u64)
        -> SearchOutcome;   // best mapping, its score, history (for inspection), stats
}
pub enum Move {                              // the shared action space for annealing, beam, evolution, RL
    Resplit { op: OpId, split: Vec<SplitAxis> }, Reassign { op: OpId, slice: SliceIx, unit: UnitId },
    Rehome { tensor: TensorId, home: TensorPlacement }, Reroute { transfer: TransferId, route: Route },
    ToggleFusion { a: OpId, b: OpId }, SetGroupKind { group: GroupIx, kind: GroupKind },
    SetCollective { op: OpId, plan: CollectivePlan }, SetTarget { op: OpId, target: Target },
    SetPriority { sel: TaskSelector, prio: i32 }, Reembed { mesh: MeshEmbedding },
}
```
Implementations: `HeuristicMapper` (default, deterministic, the steps above), `BeamSearch` (width W, moves from a
move generator ranked by Tier A bottleneck attribution: moves that touch the binding resource first),
`Annealing` (seeded ChaCha RNG, geometric schedule), `LearnedPolicy` (calls a PyO3 callback with an observation:
per-op features, per-resource utilization from the last Tier A result, bottleneck attribution; returns Moves or
a full Mapping). Every search is deterministic given `seed` and thread count independence (evaluations in
parallel, reductions in candidate-index order). `Score` is computed by Tier A; elites may be re-scored by Tier B.

Mapping quality guard: for every evaluated design, 06 records the heuristic mapping's score and the searched
score; a design's reported score is always from a validated Mapping, and the mapping is stored with the
result so it can be audited.

---

## 4. Tier A analytical engine (`kiln-sim::analytic`)

### 4.1 Resources

`ResourceTable` (built once per design from expanded instances):
- `ComputeUnit(u)`: one per unit instance (capacity 1 task at a time unless the template declares concurrency).
- `Link(l, dir)`: every directional link instance, incl. unit-to-local-memory links, NoC links, router
  crossbar ports, memory-controller ports, PHY/shoreline links, inter-chip lanes, switch ports.
- `MemPort(m, p)`: every port of every memory instance; `Bank(m, b)` groups when 01 declares banking.
- `DramChannel(c)`: each HBM/DRAM (pseudo-)channel, with `peak_Bps` and efficiency model (5.3).
- `NmpSite(s)`: NMP execution resource per bank group/channel; also locks the corresponding DramChannel/Banks.
- `Sequencer(chip)`: host/launch/command queue, for launch and sync overheads.
- `DmaEngine(e)`: copy/collective engines where declared.

Each Task in the LoweredGraph carries `demands: SmallVec<[(ResourceIx, f64 /*busy seconds at nominal clock*/,
ClockDomain); 8]>` and an uncontended latency `lat` (pipeline fill, hop latencies, DRAM access latency).

### 4.2 Per-group busy time and the bound family

For an ExecGroup `G` (tasks `T_G`), at clock assignment `f` (4.5):
- `B_r(G) = sum_{t in T_G} demand(t, r, f)` for every resource `r`.
  Compute demand scales with the unit's domain clock: `cycles / f_domain`; link and DRAM demands use their own
  clocks/bandwidths.
- `L(G)` = longest path through `T_G` using per-task duration `max_r demand(t,r) + lat(t)`.
- Bound family (each adds constraints, so monotone):
  - `A0(G) = max(FLOPs / peak_compute(f), offchip_bytes / sum DRAM peak*eta)`: classic roofline (reported for
    reference only, never used as a score).
  - `A1(G) = max_r B_r(G)`: occupancy bound over **all** resources (fixes L1).
  - `A2(G) = max(A1(G), L(G)) + O_exposed(G)`: adds dependencies and exposed overheads (4.4).
  - `A_est(G) = A2(G) + C(G)`: adds a contention correction (4.3); this is the score, and it is not a bound.

Graph time: groups separated by barriers add; groups without barriers on a shared stream overlap per 4.3.
`T_A2 = sum over barrier-separated segments of A2(segment)`.

### 4.3 Overlap and contention model

- Within a group, perfect overlap across resources is assumed by A1, and imperfect overlap is captured by
  `C(G)`: for each resource with utilization `rho_r = B_r / A2`, a queueing delay on tasks of the critical path
  that traverse `r`: `C(G) = max over critical-path resources of sum_{t on path, t uses r} demand(t,r) *
  rho_r / (2 (1 - min(rho_r, rho_max)))` (M/D/1 waiting time, `rho_max = 0.95`). It is cheap, monotone in load,
  and zero for idle resources. Its form is fixed; only regression against Tier B (never against hardware)
  may adjust `rho_max` and an optional scalar on it, and that fit is reported in the result provenance.
- Across groups without a barrier (async prefetch, e.g. next layer's weights streamed during current compute):
  segments are merged and evaluated as one group, so overlap is credited only when resources actually differ.
- Pipelined groups: `A2 = (stages + microbatches - 1) * max_stage_time` style closed form derived from
  per-stage busy sums, with fill/drain explicitly reported as pipeline bubble.

### 4.4 Per-launch, sync, and setup overheads

The execution model is a design attribute from 01 (`exec_model: HostLaunched | DeviceQueued | StaticDataflow`),
and each group selects `LaunchKind`:
- `HostLaunched` (GPU-like): `duration(G) = max(A2(G), t_min_kernel) + t_gap`, with `t_launch` hidden behind the
  previous kernel when the queue runs ahead: `exposed_launch = max(0, t_launch - duration(prev))`.
- `DeviceQueued`: per-group `t_dispatch` only.
- `StaticDataflow` (compiled program, TPU-like): per-program `t_program`, per-sync `t_sync`.
- Every barrier: `t_sync(scope)` where scope is unit set, chip, or system. Collectives: `t_coll_setup(algo)`.
- A group with no tasks (a layout view) issues no kernel and pays no launch, gap or dispatch term.
- **Software-stack kernels** (02 §7.4.1, 08 §F): the recipe's non-primary kernels of a node are charged at the
  group holding the node's first op, after the group's own kernels on the same queue. Each costs
  `t_mem + max(0, t_min_kernel - t_mem) + t_gap` on `HostLaunched` (`t_mem + t_dispatch` on `DeviceQueued`,
  `t_mem` in a `StaticDataflow` program), where `t_mem` is its bytes over the serving level's resources
  (interleaved by bandwidth share, calibrated efficiency; off chip with the `t_dram_ramp` term per kernel).
  The bytes enter resource busy time, traffic and energy; attribution puts `t_mem` on `dram` / `port` and the
  rest on `overhead`. The floor (A2) uses peak bandwidth.
Overhead constants live in calibration data files, per execution model and platform (9). A design with an
unknown platform gets the "assumed" defaults with a provenance flag.

### 4.5 Power cap and DVFS clock model

Inputs from 04: V/f table per clock domain, `P_static(T, V)`, thermal limit, power cap `P_cap` (board/package),
power-management window `tau_pm` (calibrated; default 1 ms assumed).

Energy per group is split into clock-dependent dynamic (compute + on-domain SRAM + NoC in the core domain),
scaling as `E_dyn(f) = E_dyn(f0) * (V(f)/V(f0))^2`, and clock-independent (DRAM, PHY, I/O). Power over a
window: `P(f) = (E_dyn(f) + E_indep) / T(f) + P_static(V(f))`, with `T(f)` from 4.2 (compute demands scale with
`1/f`, memory demands do not).
```
fn solve_clock(window) -> f:
    if P(f_max) <= P_cap: return f_max
    binary search f in [f_min, f_max] over the V/f table for the largest f with P(f) <= P_cap   // P monotone in f
    // O(log |table| * cost of T(f)); T(f) recomputation is O(#resources) since demands are cached per domain
```
If no `f >= base` (01 §12) satisfies the cap, the result is error `E-MAP-POWER-CAP` naming the domain and the
power at `base`.
Windows: Tier A evaluates steady state over the repeated layer (the common case), giving one clock per phase
(prefill, decode). Tier B applies the controller over sliding windows of `tau_pm` on the simulated power trace
(5.5), so short low-power phases can run at `f_max` while dense GEMMs throttle. The A100 400 W case must come
out near 1290 MHz on dense GEMM with calibrated power coefficients and must stay at max clock on decode GEMV;
both are in 06's regression corpus.

### 4.6 Tier A / Tier B relation (normative)

Same Mapping, same LoweredGraph, same calibration set, same clock assignment (Tier B's clock trace is fed to
Tier A in the comparison mode, or both use the same steady-state clock):

1. **Floor relation (provable):** `T_A2 <= T_B`. Proof sketch: in Tier B every resource serves at most its
   capacity per unit time, so `T_B >= B_r` for every `r` within a barrier segment; Tier B honors every
   dependency edge and charges at least `lat(t)` per task, so `T_B >= L`; overheads are charged identically and
   are exposed in Tier B at least as much as `O_exposed` (Tier A credits the maximum possible hiding). DRAM: Tier
   A uses `eta_max(pattern)` from the same DRAM model (the best efficiency Tier B can attain for that access
   class) in A2, so B cannot beat it. Hence `T_A2 <= T_B`. Violation is a bug in one engine; it is an error.
2. **Estimate relation (empirical):** `|T_A_est - T_B| / T_B <= eps_AB` on 06's regression corpus, with
   `eps_AB = 10%` per layer (06 §2.4 applies it as the p95, with median <= 5% and max <= 30%) and `5%` for
   end-to-end model latency (06 owns the numbers). Tier A
   uses `eta_mean(pattern)` in `A_est`.
3. **Ordering consistency:** on corpus pairs of designs, Kendall tau between Tier A and Tier B rankings >= 0.9
   (what matters for evolution).

### 4.7 Tier A energy

`E_total = sum_tasks E_dyn(task, f) + sum E_indep + P_static(V) * T_A_est + sum links (bytes * e_bit_link from
04 via bits = bytes*8 only inside 04's tables)`. Idle-unit energy is in `P_static` (with power gating as a 04
attribute: gated units contribute their gated static power).

### 4.8 Complexity and budget

Lowering for Tier A uses **task aggregation**: a slice's transfers over a route are one task with per-link
demand (not chunked). Per layer: tasks ~ #slices x (1 compute + ~3 transfers), typically 1e3-1e4; route lengths
<= ~20 hops; so ~1e5 demand updates, plus <= ~50 cold kiln-cost keys (1-5 ms each, then cached), plus the
clock solve (~10 iterations x O(#resources)). Target <= 50 ms cold-cache per single-chip (design, layer) for designs up to
~1k units (8-chip: <= 100 ms p95, 06 §8); <= 5 ms warm. For larger systems, identical-chip symmetry (same mapping per chip in a
TP/DP group) is exploited by evaluating one representative chip plus the inter-chip resources, which the
mapping validator certifies.

### 4.9 Whole-step evaluation (scoring mode)

Scores come from whole steps (02 §12.5), so whole-step evaluation is a first-class engine mode, not a sum of
per-op or per-layer results:
- One LoweredGraph covers the phase instance end to end: prologue (embedding, first norm), the repeat, and
  epilogue (final norm, head, sample). Groups may span layer boundaries, so next-layer weight prefetch,
  activations kept on chip across the boundary, fusion across residual/norm, and cross-layer overlap of
  collectives are credited exactly as within a layer (4.3). Launch, gap, dispatch and sync terms (4.4) are
  charged per kernel as the execution model and the software-stack recipe issue them over the whole step (e.g.
  one `t_launch` for a captured `HostLaunched` step plus `t_gap` per kernel, the recipe's unfused kernels
  included; one `t_program` for `StaticDataflow`).
- Repeat extrapolation (02 §7.3 steady-state contract): the step is lowered with a window of `w` consecutive
  iterations (default `w = 3`: first, middle, last) between prologue and epilogue and simulated as one graph.
  `T_mid` is the middle iteration's completion-to-completion interval (end of the previous iteration's last
  group to end of its own), and `T_step = T_window + (L - w) * T_mid`. Designs whose steady state spans `k`
  layers (cross-layer pipelining, alternating bodies) use `w = 2k + 1`; for `L <= w` the whole repeat is
  lowered. `w` is in provenance.
- Tier B simulates the same window with the same rule, so I9 (`T_A2 <= T_B`) holds per window and per step.
- `scope: layer` (02 §12.5) reports `T_mid` plus a `1/L` share of prologue and epilogue from the same evaluation.
- `eval_mode: isolated` (a `barrier` on every node, nothing resident between nodes) is a calibration mode; its
  results carry `scope: op` and are never scored.
- The per-phase clock (4.5) is solved over the window's steady state. Budget: about `w` design-layers plus
  prologue/epilogue, with identical iterations sharing kiln-cost and fragment-cache entries; targets in 06 §8.

---

## 5. Tier B event-driven engine (`kiln-sim::event`)

### 5.1 Model

Discrete-event simulation over the same LoweredGraph, refined:
- Transfers are split into **chunks** (default 4 KiB on-chip, 64 KiB inter-chip, adaptive, see 5.6). A chunk
  traverses its route with virtual cut-through: at each hop it acquires the link (serialization `bytes/bw`),
  the router output port and a credit in the downstream input buffer (finite, from 01; default 4 chunks),
  then incurs hop latency. Credits give backpressure and head-of-line blocking where hardware has it.
- Compute slices occupy their unit for the kiln-cost cycles, but their outermost-level refills are emitted as
  real transfer chunks paced by kiln-cost's period schedule (tile `i+1` refill issued at start of tile `i` if
  double-buffered), so stalls caused by shared NoC/HBM appear in the timeline instead of being assumed away.
  Compute of tile `i` cannot start until its refill chunks arrive.
- Memory ports and banks are servers with per-request occupancy; arbitration is round-robin over requesters by
  index (deterministic).
- Collectives are task graphs (5.4). NMP command batches occupy NMP sites and lock their banks (6).
- Launch/sync overheads are tasks on the `Sequencer` resource.

### 5.2 Event core and determinism

- Time is `u64` picoseconds. Durations are computed in f64 and rounded once per task with round-half-even;
  accumulations are integer. No float comparison ever orders events.
- The event queue is a binary heap keyed by `(time_ps, phase, resource_ix, task_ix, seq)`. Every collection that
  can influence order is `Vec` or `IndexMap`. RNG is not used by the engine (it is deterministic); stochastic
  DRAM effects, if enabled, use a ChaCha stream seeded from `(design_hash, workload_hash, channel_ix)`.
- Result hash (over canonical serialized result) must be identical across thread counts and platforms (x86-64,
  aarch64): f64 ops are restricted to IEEE basic arithmetic (no fused multiply-add via auto-vectorized
  intrinsics in result-affecting code; `-C target-feature` pinned), checked by I11.

### 5.3 DRAM model decision

Decision: **calibrated structured statistical model in the engine (per channel, with bank-group/row-buffer
locality classes); Ramulator 2 itself (C++, MIT) run offline as the table generator and oracle**. We do not port
Ramulator 2 and do not run a command-level model in the loop (consistent with 07: the only Rust DRAM crate,
`ramu_rs`, lacks HBM and refresh).

Rationale: the requests Tier B generates are tile-granular streams with known access classes (sequential
stream, strided, gather), not real address traces, so a command-level model inside the loop would add cost
(10-100x event count) without information to exploit; meanwhile the A100 miss (L7) is a sustained-efficiency
problem that a structured model captures. Offline, a script in 06's harness drives Ramulator 2 with synthetic
streams per (DRAM standard, access class, rw mix, request size, active banks) and writes the `eta`/`lat` tables
as versioned calibration data files (HBM2/2E/3/3E, DDR5, LPDDR5X, GDDR6/7 timing sets with datasheet citations).
If a design's access pattern falls outside the table classes, Tier B reports `dram_class_extrapolated` in
provenance.

Statistical model per channel: a server with service rate
`peak_Bps * eta(class, rw_mix, req_size, banks_active)` and access latency `lat(class)`, where
`eta = eta_refresh * eta_turnaround(rw_mix) * eta_rowmiss(class, req_size) * eta_res`:
- `eta_refresh = 1 - tRFC/tREFI` (from timing, not fitted),
- `eta_turnaround` from tWTR/tRTW and the read/write mix (computed),
- `eta_rowmiss` from tRC, row size, locality class (computed from the command-level tables),
- `eta_res` in [0.85, 1.0]: the only fitted DRAM term (9). Refresh is also applied as explicit periodic channel
  unavailability in Tier B so it shows on timelines.
`eta_max(class)` (used by Tier A A2) = the table's best case for the class with `eta_res = 1`; `eta_mean` uses
the fitted `eta_res`. For reference, the A100-40GB measurements show weight-streaming GEMV at 1300-1370 GB/s of
1555 GB/s spec (84-88%) on large shapes and ~75% on 32 MiB shapes (launch and tail effects included), which is
the scale of correction the v0 harness lacked.

### 5.4 Collectives as task graphs

`kiln-map` lowers each collective with its `CollectivePlan` into transfer + reduce tasks over explicit routes:
- Ring all-reduce over `p` participants, `c` chunks: reduce-scatter `p-1` steps then all-gather `p-1` steps;
  each step sends `N / (p * c)`-sized pieces per chunk pipeline stage to the ring successor, reduce task on
  `reduce_on` resource (reading/writing local memory, so it contends with compute for HBM; this is modelled).
- BiRing: two rings in opposite directions, half the data each (uses both link directions).
- Tree / double binary tree: reduce up, broadcast down, chunk-pipelined.
- Ring2D / hierarchical: reduce-scatter along axis 1, all-reduce along axis 2, all-gather along axis 1; the
  generic `Hierarchical(Vec<algo>)` composes over topology tiers (intra-package then inter-package).
- Direct (all-to-all for MoE dispatch/combine): one transfer per (src, dst) pair with exact per-pair bytes from
  the routing table of 02's expert assignment.
- In-network reduction where the switch declares it (`SwitchInNetwork`).
Tier A uses the same lowered task graph's per-link busy sums (so the alpha-beta closed forms emerge from the
demands rather than being separately hand-coded); closed forms are kept only as unit-test oracles
(e.g., ring AR per-link bytes = `2 (p-1)/p * N`).

### 5.5 Outputs

Tier B emits a `Trace` (owned by kiln-trace, 05 renders it):
- Spans: `(resource_ix, task_ix, op_id, slice, kind, start_ps, end_ps, bytes or macs, energy_J)`.
- Counters sampled per resource at event boundaries (step functions, not fixed-rate): link utilization, queue
  depth, credits, DRAM channel bandwidth, unit occupancy, per-domain power, clock.
- Flows: chunk-level arrows from source span to sink span (optionally decimated).
- Critical path: list of `(task_ix, reason: Compute | Link(l) | Port | Dram | Dependency | Overhead | Queue)`
  covering `[0, T_B]` without gaps (I8).
Perfetto export is in kiln-trace.

### 5.6 Performance and parallelism

Target <= 10 s per single-chip (design, layer) single-threaded on designs up to ~1k units; <= 30 s for 8 chips
(06 §8 owns the numbers).
- **Adaptive chunking:** choose chunk size so total chunk-hop events per layer <= 5e6 (estimate from Tier A
  route demands before running), never below the link flit size and never above 1/16 of a transfer (so
  pipelining across hops remains). The chosen size is in provenance; changing it is a result-affecting input.
- **Coalescing:** a run of chunks of one transfer on an uncontended path (no other demand on any route link in
  the interval, known from the reservation table) is advanced as one event; exact because cut-through
  timing for an uncontended train is closed form.
- **Parallelism (rayon):** (1) across independent jobs: designs in a batch, phases (prefill/decode), distinct
  layers when layers are independent replicas: embarrassingly parallel, primary mode; (2) within a job, a
  conservative windowed parallel DES partitioned by chip, window = minimum inter-chip link latency (YAWNS):
  each window, chips advance in parallel, cross-chip messages are exchanged at the barrier in sorted
  `(time, src, seq)` order. Results are bit-identical to the sequential engine (I11 tests both). This mode
  is post-M4 (not scheduled in 06 §10) and off by default (07 recommends parallelism across designs, not inside one eval); it ships only
  once the bit-identity property test passes on the full corpus. Intra-chip partitioning is not done (NoC
  lookahead too small to pay off).
- Memory: spans are stored columnar (struct of arrays) with optional decimation for very long traces.

---

## 6. Near-memory compute execution

An NMP site is an 01 `ComputeUnit` with a `near` binding (01 §9) at granularity `per_bank`, `per_bank_group`,
`per_pseudo_channel`, `per_channel` or `per_stack` (logic-die units live in `MemStack.logic_die`); in-memory compute
is 01's `kind: cim`. Parameters: ALUs per site (geometry), supported ops and precisions, `internal_bw_per_site_Bps`
(01 `internal_bandwidth`, typically several x the external share), command overhead `t_cmd` (01
`command_latency`), and from the bound stack's timing / 04 defaults: command-bus bandwidth (commands/s per
channel), mode-switch time `t_mode` (host mode to compute mode), row size, tRC; local buffer/register capacity
(01 `local`), reduction tree across sites.

Execution model for an op offloaded to NMP sites `S` (e.g. GEMV `y = W x`, weights resident row-wise in banks):
```
T_nmp = t_mode_in
      + T_broadcast(x)                                        // host -> sites over command/data bus or global buffer
      + max(  rows_activated_per_site * tRC / bank_parallelism_per_site,
              bytes_per_site / internal_bw_per_site,
              macs_per_site / (alus_per_site * f_nmp),
              n_commands / cmd_bus_rate )                      // the command-issue bound is often the binder
      + T_reduce_across_sites + T_gather(y) + t_mode_out
```
Layout constraint: an NMP op requires its stationary operand to be laid out in the sites' banks in the
NMP-native layout (row-aligned, element-interleaved per the unit's rule). The mapper either places the tensor
there persistently (KV cache, weights) or inserts a relayout transfer whose cost is charged. While a channel is
in compute mode, host accesses to its banks are blocked (or, if 01 declares bank partitioning, only the locked
banks are). Tier A charges the NMP time as busy time on both `NmpSite` and the locked `DramChannel`s; Tier B
holds a lock.

Offload decision (default heuristic, per op):
```
offload iff T_nmp(op) + T_relayout_amortized(op) < T_host(op) and capacity/layout constraints hold
```
evaluated with Tier A local estimates; `T_host` is the host mapping's local Tier A time including its HBM
occupancy. Expected regime where NMP wins: arithmetic intensity below `internal_bw_total / host_peak_compute`
crossover with operands resident (decode GEMV at small batch, attention over KV at decode, embedding lookups,
elementwise on resident tensors). It loses at batch sizes where the host becomes compute-bound, when commands
per byte are high (short rows, small tensors), and when relayout cannot be amortized. Search may override with
`SetTarget`. Result reports per-op host vs NMP estimates so the decision is auditable.

---

## 7. Multi-chip execution

- **Parallelism input (02):** TP, SP/CP, EP, PP (with microbatch count and schedule: GPipe, interleaved,
  inference round-robin), DP. kiln-map chooses the `ParallelPlan`, calls `kiln-wl::partition` (which propagates
  shardings and inserts collectives, 00 decision 5), instantiates the result per chip, picks collective algorithms,
  and applies only 02 §9.1's volume-preserving rewrites; kiln never invents a parallelism strategy silently (the
  search may propose a different strategy only through 02's parallelism config, so it is visible).
- **Topology:** chips, packages, links, switches from 01 with wire-derived parameters from 04 for in-package
  links and declared SerDes parameters for off-package links. Routes between chips use the same graph routing.
- **Comm/compute overlap:** each chip has compute units and DMA/collective engines as separate resources;
  overlap is credited only when 02/the mapping declares the collective asynchronous with respect to the
  consumer (e.g., TP all-reduce overlapped with the next independent GEMM chunk via decomposition).
  Contention is real in both tiers: collective reads/writes hit
  the chip's HBM and NoC, and reduce tasks take compute or vector units.
- **Pipeline parallelism:** stage boundaries are send/recv transfers; Tier A uses the pipelined-group formula
  with bubbles reported; Tier B simulates microbatches explicitly.
- **Symmetry:** identical chips executing identical shards are evaluated once in Tier A (4.8); Tier B
  simulates all chips (contention between groups sharing a switch is real and asymmetric).

---

## 8. Physical invariants and conservation laws (checked on every result; violation = error)

| Id | Invariant | Check |
|---|---|---|
| I1 | Compute floor | `T_group >= useful_macs / sum(peak macs/s of assigned units at the actual clock)` and per unit |
| I2 | Memory-level floor | per memory level, bytes moved >= compulsory footprint bytes; `T >= bytes/bw` per port/channel |
| I3 | Resource occupancy | for every resource, `busy_r <= T` (utilization in [0,1]); in Tier B, interval overlap on a resource never exceeds its declared concurrency |
| I4 | Byte conservation | per transfer, every hop carries exactly the transfer's bytes; per memory, bytes in - bytes out = net residency change; per tensor, delivered bytes to each consumer slice equal its footprint (no all-to-all fan-out) |
| I5 | Work conservation and coverage | useful MACs summed over slices = op MACs; slices partition the iteration space (each output element produced once, reductions combined once); padding MACs reported separately |
| I6 | Energy conservation | `E_total = sum component energies + static * T` exactly (same summation order); `avg power = E/T`; under a cap, windowed power <= `P_cap * (1 + tol_window)` |
| I7 | Capacity | live bytes per memory instance <= capacity at all times (Tier B checks at every allocation event) |
| I8 | Causality and critical path | no task starts before predecessors end + edge latency; critical path covers `[0,T]` with no gaps |
| I9 | Tier relation | `T_A2 <= T_B` whenever both are computed; corpus tolerance per 4.6 |
| I10 | Clock legality | every domain's clock is on its V/f table, <= `f_max`; thermal limit respected if 04 provides it |
| I11 | Determinism | result hash identical across thread counts and runs (CI property test) |
| I12 | Precision bytes | tensor bytes = `ceil(elements * bits/8)` + MX scale bytes; accumulator traffic at accumulator width |
| I13 | Little's law (Tier B) | per queue, time-averaged occupancy = arrival rate x mean sojourn within 1% |
| I14 | Collective volume | per-link bytes of a collective >= the algorithm's theoretical minimum (e.g., ring AR `2(p-1)/p * N` per participant link) |
| I15 | Roofline sanity | `T >= A0` always; a result faster than its roofline is an error with the binding term named |

Checks run in `O(#tasks + #resources)` for Tier A and `O(#events)` streaming for Tier B. Failures produce
structured errors (code, message, entity path, the two numbers, hint).

---

## 9. Calibration hooks (without fudge factors)

Principles:
1. **One physical role per parameter.** No global scale on time or energy. Each fitted parameter names the
   mechanism it represents and the term of the model it enters.
2. **Bounded.** Each has physical bounds in its data file; a fit that hits a bound fails and flags the model
   as structurally wrong for that platform instead of accepting the value.
3. **Fit on isolating microbenchmarks, evaluate on held-out workloads.** Never fit on the LLM suite used for
   accuracy claims.
4. **Structure before residual.** Quantization, clock throttle, launch overhead, DRAM refresh/turnaround are
   modelled first; residual terms are fit last and their magnitude is reported per result.
5. **Dual reporting.** Every calibrated result also reports the uncalibrated value and per-parameter
   contribution (`dT/dparam * delta_param`), so a large correction is visible, not hidden.
6. **Scoped.** Parameters apply by platform/execution-model/technology key (e.g., `exec_model=host_launched,
   platform=a100_40gb`). A novel design receives only technology-scoped or explicitly "assumed" values; it never
   inherits a GPU-specific residual.
7. **Residual acceptance test.** After fitting, residuals on held-out ops must show no structure against op
   class, shape, arithmetic intensity (06 runs the regression; slope significance -> fail).
8. **Ranges, not point values, for what is uncertain.** Every parameter carries `lower / central / upper` per
   mechanism key (never per op), derived as in 06 §3.2; results are intervals (9.1).

| Parameter | Mechanism / term | Fit from | Bounds (initial) |
|---|---|---|---|
| `eta_res(dram_kind)` | DRAM residual efficiency (5.3) | streaming copy and weight-streaming GEMV sweeps, sizes >> L2 | [0.5, 1.0] (08 §F: v6e single-program HBM 0.665 is measured behaviour) |
| `t_dram_ramp(dram_kind)` | DRAM stream fill + drain per segment: each DRAM resource's busy time `b` becomes `b + t_r` (`b >= t_r`) or `2 sqrt(b t_r)` (triangle profile), i.e. `eta_res` as a bounded function of transfer size (M2) | same sweeps, all sizes | [0, 20] us |
| `t_min_kernel`, `t_gap`, `t_launch` | launch path (4.4) | empty kernel, tiny GEMM/elementwise series (A100 shows ~5 us floor at 256^3) | [0.5, 20] us (`t_launch`); [0.1, 20] us (`t_min_kernel`, `t_gap`: the A100 empty-kernel graph chain measures their sum at 0.90 us, M2) |
| `f_cap_op(domain, power_cap)` | sustained clock of a capped domain vs MAC-pipe activity at nominal clock, a telemetry operating-point table (4.5 stand-in until the power group is calibrated; platform sets only) | NVML clock of the calib-micro GEMM/GEMV sweeps | scale [0.9, 1.1] |
| `t_sync(scope)`, `t_coll_setup` | barrier/collective setup | barrier and tiny-collective microbenchmarks | [0.1, 50] us |
| `kappa_E_*`, `kappa_clk`, `p_ll`, `p_ls` (04 §12.1; 03's former `C_eff(domain)`, 00 decision 6) | power model (4.5) | power + clock telemetry under dense GEMM at several caps | per 04 |
| `tau_pm` | power-management window | step-load telemetry | [0.1, 100] ms |
| V/f curve (`V_th`, `alpha`, 04 §4.7) | DVFS | clock + power telemetry, locked-clock runs where root allows | per 04 |
| `unit_eff(template, k_depth)` | per-unit pipeline efficiency vs. innermost reduction depth (issue bubbles, operand-delivery limits) | GEMM sweeps at locked clock, after quantization and clock are modelled | [0.7, 1.0], monotone non-decreasing in `k_depth` |
| `eta_link(kind)` | protocol efficiency of NVLink/ICI/PCIe-class links | point-to-point bandwidth tests | [0.6, 1.0] |
| `rho_max`, contention scalar | Tier A contention correction (4.3) | Tier B, never hardware | as 4.3 |

Application: parameters are loaded from versioned calibration files (hash in provenance), applied in exactly
one place each (the term named above), and the calibration-set hash enters the result. `kiln calibrate` (06)
fits them with a deterministic least-squares on log-time residuals per parameter group, in a fixed order
(clock/power first, then launch, then DRAM, then unit efficiency), each group on its own microbenchmarks.

For the A100 v0 discrepancies this plan predicts: decode GEMV error closes via `eta_res` and launch terms;
tiny-op error via `t_min_kernel/t_gap`; the compute-bound spread via clock under cap (~1290 MHz vs 1410) plus
108-SM wave quantization, with `unit_eff` absorbing only what remains. If `unit_eff` hits its bound, the
structural model is wrong and we fix structure.

### 9.1 Ranges and interval evaluation

Each parameter above, and each 04 physical constant (04 §3), carries `lower / central / upper` and a declared
pessimistic direction (`pess_dir`: the end that lowers throughput or raises area, power or energy). Ranges come
from the spread across measured devices and physically plausible bands (06 §3.2, 04 §3), not from registry
bounds or worst cases.

Propagation is deterministic corner evaluation:
- **Corners.** `central` sets every parameter to central. `low` sets every parameter to its pessimistic end at
  once (lower efficiencies, larger launch/sync/setup terms, larger energy and leakage coefficients and hence a
  lower clock under the cap, larger area coefficients and hence longer wires); `high` sets every parameter to
  its optimistic end. The `low` corner gives the low throughput and the high time, energy, power and area; `high`
  the reverse.
- **Monotonicity assumption.** Corners reuse the central Mapping and LoweredGraph and only re-cost them. At a
  fixed mapping, Tier A time is monotone in each parameter in its declared direction (every term is a sum or max
  of per-resource demands monotone in that parameter; the clock solve is monotone in power coefficients), so the
  two corners bound every interior combination. Tier B is monotone up to P3b's 2% list-scheduling slack, which
  widens the reported interval. Re-mapping at a corner can only lower its time, so the fixed-mapping `low` is
  conservative by at most the mapper's gain; S4 audits report the re-mapped `low` beside it.
- **When corners are insufficient.** (a) A parameter with no declared monotone direction for some output (e.g.
  contention `rho_max` near saturation, a clock domain whose dynamic and static power trade off under the cap):
  the engine evaluates all `2^k` vertices of the `k` such parameters (`k <= 3`) or 16 seeded Latin-hypercube
  points (`k > 3`), takes the extremes, and records `interval_method: sampled`. (b) Discrete outcomes that flip
  inside the range (feasibility, capacity, envelope, `E-MAP-POWER-CAP`) are reported as `corner_flips`; a flip at
  `low` blocks claims (06 §6.6).
- **Cost.** `corners`: central plus two re-costings at fixed mapping, at most 3x one Tier A evaluation (typically
  about 1.5x, mapping and kiln-cost loop-nest choices are reused). `sensitivity`: central plus a linearized
  interval from the per-parameter contributions already computed for dual reporting (principle 5),
  `T_low ~= T_central + sum_p max(0, dT/dp * (p_pess - p_central))`, at about 1.1x; its error against `corners`
  is tracked on 06's agreement corpus (2.4). Evolution scoring defaults to `sensitivity`; audits and claims use
  `corners` (06 §6.2, §6.6).

---

## 10. Result data model (to 05 and 06)

Owned by kiln-trace; this section defines the content.
```rust
pub struct SimResult {
    pub provenance: Provenance,            // kiln version+git, design/workload/mapping/calibration hashes, tier, chunk size, search flags
    pub tier: Tier,                        // A | B
    pub scope: Scope,                      // Step | Layer | Op (4.9, 02 §12.5); only Step and Layer are scored
    pub corner: Corner,                    // Central | Low | High (9.1); 06 assembles intervals from the three
    pub makespan_s: f64, pub t_a0_s: f64, pub t_a2_s: f64,   // floors always reported
    pub clocks: Vec<ClockSample>,          // per domain; one per phase in A, a trace in B
    pub energy: EnergyBreakdown,           // compute, per memory level, per link class, NMP, static, conversion, padding
    pub power: PowerSummary,               // avg, peak windowed, cap, throttled flag
    pub ops: Vec<OpResult>,                // per op (and per slice on request)
    pub resources: Vec<ResourceResult>,    // every resource: busy_s, bytes/macs, energy, utilization, peak queue
    pub groups: Vec<GroupResult>,          // incl. pipeline bubbles, exposed overheads
    pub collectives: Vec<CollectiveResult>,
    pub bottleneck: Bottleneck,
    pub invariants: InvariantReport,       // all checks with measured margins
    pub cost_model: CostModelSummary,      // cache hits, truncated searches
    pub calibration: CalibrationContribution, // per-parameter delta to makespan/energy
    pub trace: Option<TraceRef>,           // Tier B; Tier A when trace level `ops` is requested (00 decision 8)
}
pub struct OpResult { op: OpId, start_s: f64, end_s: f64, binding: Binding, macs_useful: u64, macs_issued: u64,
    bytes_by_level: Vec<(MemLevelRef, u64)>, energy: EnergyBreakdown, target: Target, host_vs_nmp: Option<(f64, f64)> }
pub enum Binding { Compute(UnitIx), Link(LinkIx), MemPort(PortIx), Dram(ChannelIx), Nmp(SiteIx),
    Dependency, Overhead(OverheadKind), Contention(ResourceIx), PipelineBubble }
pub struct Bottleneck {
    pub time_by_binding: Vec<(BindingClass, f64)>,   // sums to makespan: critical-path decomposition (B) or per-group binding (A)
    pub top_resources: Vec<(ResourceIx, f64 /*utilization*/, f64 /*shadow price: dT/d(1/bw) approx*/)>,
    pub slack: Vec<(ResourceIx, f64)>,               // how far each near-binding resource is from binding
    pub summary: String,                              // LLM-readable: "decode attention bound by HBM ch 3-5 (91%); ..."
}
```
Attribution rules: Tier A attributes each group's time to its argmax term in A2 (ties by fixed order Compute <
Dram < Link < Port < Dependency < Overhead) and lists the runner-up with its ratio; Tier B uses the critical
path. `time_by_binding` sums exactly to makespan (checked). Shadow price is 1 for binding resources, 0 for others,
estimated by a single re-evaluation with that resource's bandwidth +10% for the top 5 (Tier A only; cheap).

**Explanation text (single source, 00 decision 7).** `kiln_sim::explain_run(result: &SimResult, op: Option<OpId>,
max_items: usize) -> String` is the only producer of bottleneck/limiter text: deterministic templates over
`Bottleneck`, per-op `Binding` and runner-up terms, and the Mapping. It fills `Bottleneck.summary` and the per-op
explanation strings persisted in the trace; 05's views display that text verbatim and 06's `Result.explain()` /
`kiln.explain` embed it after the status, score and error lines.

---

## 11. Cross-section dependencies

- **01 hardware IR:** compute unit templates (spatial axes, MAC modes per precision, accumulator, pipeline
  depth, per-operand hierarchy with ports and double-buffer capability), memory instances with banks/ports,
  near-memory `NearBinding` and its granularity (01 §9), interconnect graph with directional links, routers (buffer depth,
  routing algorithm, multicast support), switches (in-network reduction), DMA engines, chip/package topology,
  `exec_model`, power cap. This section assumes these names; 01 is authoritative for their schema.
- **02 workload IR:** affine access maps per operand (relevance derivable), dim kinds, dtypes per tensor incl. MX
  block axis and size, KV cache tensors and paging, `ParallelPlan` and the `PartitionedProgram` from `kiln-wl::partition` (collectives inserted, 00 decision 5), MoE
  routing statistics, phases (prefill/decode) and repetition counts, async semantics for collectives.
- **04 physical model:** link bandwidth/latency/energy from wire length, per-access energy per memory
  type/size/node, `e_mac` per mode, V/f tables, `P_static`, thermal limits, DRAM timing sets. 03 provides back the
  `TrafficMatrix` for placement (04 §6.4) and a per-window `ActivityReport` for power (04 §8).
- **05 visualizer:** consumes `SimResult` + `Trace` (spans, counters, flows, critical path) and the Mapping (to
  draw placement and routes on the floorplan).
- **06 validation/API:** ZigZag differential harness (2.1), Tier A/B corpus and tolerances (4.6), calibration
  fitting and parameter ranges (9, 9.1), whole-step scoring (4.9), trust-suite metamorphic relations (06 §12), invariant tests, determinism property tests, performance targets, PyO3 surface for
  `MappingPolicy`/`Move`.
- **07 prior art (read; consistent):** ZigZag/LOMA and Timeloop (port concepts into kiln-cost; ZigZag and
  Timeloop as single-chip oracles), Stream fork (lessons in section 0, oracle on its subset), ASTRA-sim
  analytical backend (collective algebra ported as unit-test oracles in 5.4, oracle for collectives),
  Ramulator 2 (offline DRAM table generator, 5.3), LLMCompass (`third_party/LLMCompass`, GEMM/attention oracle)
  and GenZ (whole-model oracle) for Tier A, CiMLoop/NeuroSim and PrIM (UPMEM measured) for NMP/CiM constants and
  validation points (6), Chakra ET as a 02 import format whose collectives this section lowers.

---

## 12. Open questions

1. **Tier A tolerance targets.** 10% per layer / 5% end-to-end are placeholders until the first corpus exists;
   do we gate evolution on Kendall tau (ranking) rather than absolute error?
2. **Chunk size as an input.** Adaptive chunking makes Tier B results depend on a derived parameter; should we
   instead fix a per-link-class chunk size and accept slower runs on large systems?
3. **Contention correction form.** M/D/1 on critical-path resources is a guess; alternative is a small fitted
   model (still against Tier B only). Decide after the first Tier A vs Tier B corpus.
4. **`unit_eff` existence.** If structural terms (clock under cap + wave quantization + pipeline fill) explain
   the A100 compute-bound spread, drop `unit_eff` entirely. Need the TPU v5e/v6e data to judge generality.
5. **NMP layout constraints.** How much of HBM-PIM/AiM-style layout and command semantics must 01 express to be
   useful for evolution without hard-coding one vendor's scheme?
6. **Training workloads.** Closed: training is out of scope (08 §C).
7. **Learned mapper observation space.** Fixed-size feature tensors vs. graph-structured observation for the RL
   policy; affects the PyO3 surface in 06.
8. **Cross-platform bit-identity.** Forbidding FMA contraction in result-affecting code costs some speed;
   accept, or relax I11 to same-platform identity plus 1-ulp cross-platform tolerance?
9. **Symmetry shortcut in Tier A.** Certifying that chips are interchangeable needs mapping and topology
   automorphism checks; is a conservative syntactic check (same template, same shard shapes, same link classes)
   enough?

---

## 13. Error codes (kiln-cost, kiln-map, kiln-sim)

03 owns the `E-MAP-*` namespace (06 §6.5) and kiln-cost's `E-COST-*` codes. Every code is a `Diagnostic` with the
path of the offending op, tensor, unit or mapping field. Codes from the mapper's candidate filter
(`E-COST-INFEASIBLE`, `E-MAP-OP-004`, `E-MAP-PREC-002`) make a unit unmappable for that op rather than failing the
evaluation when another candidate unit remains.

| Code | Raised by | Meaning | Fix |
|---|---|---|---|
| E-COST-UNIT | kiln-cost template | unit index out of range of the design | internal: report the design |
| E-COST-CLOCK | kiln-cost template | unit has no clock (01 §12) | give the unit or an ancestor a clock |
| E-COST-GANG | kiln-cost template | no ancestor holds the gang's count of units of the kind | fix `gang` or the cluster counts |
| E-COST-CHAIN | kiln-cost model | unit has no memory chain for an operand role (§2.2) | connect the role to a memory via `feeds` |
| E-COST-RESIDENCY | kiln-cost model | an operand resides at a level that is not on its chain | home the tensor on a level of the chain |
| E-COST-MODE | kiln-cost model | precision mode index out of range | internal |
| E-COST-NEST | kiln-cost | a node names a kernel the op does not have | internal |
| E-COST-MAPPING | kiln-cost | a supplied mapping has no tile classes | supply a complete mapping |
| E-COST-UNMAPPABLE | kiln-cost search | no precision mode runs the operand pair with an accumulator holding the required one (§2.7) | add the mode, or convert (02 §5.7) |
| E-COST-INFEASIBLE | kiln-cost search | no loop nest of the slice fits the unit's memories (§2.4) | split the op further or add capacity |
| E-MAP-HW-001 | kiln-map | design has no enabled matrix or vector unit with feeds | add a compute unit with `feeds` |
| E-MAP-CAP-001 | kiln-calib predict | resident bytes of a phase exceed the capacity of a memory group | fewer layers per chip, or more capacity |
| E-MAP-CAP-002 | kiln-map placement | no memory every unit can reach (no off-chip or shared on-chip level, §3.4) | add a shared level |
| E-MAP-OP-001 | kiln-map program | a bench op kind has no kernel builder | use a supported op kind |
| E-MAP-OP-002 | kiln-map program | a kernel names an unknown tensor | internal (02 lowering) |
| E-MAP-OP-003 | kiln-map cost | contraction without operand roles | internal |
| E-MAP-OP-004 | kiln-map | no unit can run the op's kernel class, or no vector unit shares memory with the MAC unit for a contraction's conversions and scale passes | add a vector unit beside the MAC unit, or avoid the split / conversion |
| E-MAP-OP-005 | kiln-map program | an operand's rank exceeds the footprint box's 6 dimensions; dimensions are never dropped | reshape the tensor to rank <= 6 |
| E-MAP-PREC-001 | kiln-map cost | the unit has no MAC mode for the operand pair | add the mode, or convert (02 §5.7) |
| E-MAP-PREC-002 | kiln-map cost | the unit has no element mode for the op's dtypes | add the mode |
| E-MAP-ROUTE-001 | kiln-map hwview | no route between two memories (§3.5) | connect them through a network |
| E-MAP-ROUTE-002 | kiln-map mapping | routing policy is unmodelled in v0 | use `ecmp` |
| E-MAP-MOVE-001..006 | kiln-map search | a search move names an unknown op, slice, tensor or group, or fuses non-adjacent groups (§3.8) | propose a valid move |
| E-MAP-VAL-001 | kiln-map lower | placement names an unknown unit | name a unit of the design |
| E-MAP-VAL-002 | kiln-map lower | op has no placement | place every op |
| E-MAP-VAL-003 | kiln-map lower | op has no unit set | give the op a unit set |
| E-MAP-VAL-004 | kiln-map lower | host execution is unmodelled in v0 | place the op on a unit |
| E-MAP-VAL-005 | kiln-map mapping | a split names an unknown dim | split a dim of the op |
| E-MAP-VAL-006 | kiln-map mapping | split parts do not partition the extent (per segment) | parts must sum to the extent |
| E-MAP-VAL-007 | kiln-map mapping | slice count differs from the unit assignments | one assignment per slice |
| E-MAP-VAL-008 | kiln-map mapping | a slice is assigned past the end of its unit set | index within the set |
| E-MAP-VAL-009 | kiln-map mapping | a unit set mixes units that cannot run the op's kernel class | use one capable kind |
| E-MAP-VAL-010 | kiln-map mapping | placement for an unknown tensor | place only program tensors |
| E-MAP-VAL-011 | kiln-map lower | a tensor home names unknown memories | home on memories of the design |
| E-MAP-VAL-012 | kiln-map lower | an execution group names an unknown op | internal |
| E-MAP-VAL-013 | kiln-map mapping | op is in no execution group (§3.6) | cover every op |
| E-MAP-VAL-014 | kiln-map mapping | whole-step iterations must end at a barrier in v0 | end the group with a barrier |
| E-MAP-VAL-015 | kiln-map mapping | a unit set is empty | no unit supports the op's class and precision |
| E-MAP-VAL-016 | kiln-map mapping | an op uses a tensor that has no placement | place the tensor |
| E-MAP-VAL-017 | kiln-map mapping | an op reads a private tensor before any op of its span writes it | give the tensor a home, or fuse its producer into the same span |
| E-MAP-POWER-CAP | kiln-sim | a phase draws more than its power cap at the lowest V/f point (§4.5) | lower the clocks or raise the cap |
