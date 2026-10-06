# 06: Validation, calibration, evolution-loop API, CLI, performance, plan

Status: spec v0. Owner of: test strategy, calibration methodology, benchmark harness, regression corpus and CI,
`kiln-py` evaluation API, `kiln-cli`, performance targets, the trust suite, migration off the Python harness, and the phased
implementation plan for the whole project. Follows `00-overview.md` (units, ids, determinism, provenance,
floors, structured errors, calibration-as-data).

Guiding principle: kiln exists to make a claim of the form "design X beats A100/TPU-class silicon at the same
physical envelope". Every mechanism here is judged by whether it makes that claim harder to fake: by a bug,
by overfit calibration, or by an evolutionary search that finds a modelling hole. Scores are whole-step
(02 §12.5), results are intervals (6.3), and claims need the trust suite green (section 12).

---

## 1. Ground truth we hold today (inputs to this section)

| Source | What | Notes relevant to validation |
|---|---|---|
| `calibration/measurements/a100_2026-10-04.json` | A100-SXM4-40GB (Colab), 29 unique Llama-3-8B layer ops (prefill_b1, decode_b1/8/32) + `_linear` variants + GEMM sweep (34 shapes) + peaks | Timing modes: `flushed`, `unflushed`, `graph_unflushed`, `graph_cold` (canonical). NVML clock/power window per op. Peaks: copy 1.385 TB/s, read 1.427 TB/s (spec 1.555); bf16 GEMM 269-273 TFLOPS (spec 312) at 1275-1290 MHz under the 400 W cap. Large prefill GEMMs ran at 1305-1335 MHz and 403-410 W. |
| `calibration/measurements/tpuv5e_2026-10-04.json` | TPU v5e (v5 lite) single chip, same op list + sweep + HBM microbenches | Timing modes: `single` (host dispatch included, ~150 us floor on tiny ops), `pipelined`, `pipelined_rot`, `loop` (slope; canonical = min(loop, pipelined_rot)). GEMV-class ops reach 0.72-0.77 TB/s (0.88-0.94 of 819 GB/s; assumed, the v5e docs list 800 GiBps = 859 GB/s, 01 §20.2); the `read_reduce`/`copy_scale` microbenches reach only 0.57-0.72 TB/s, so microbenches are NOT the DRAM calibration target on TPU (see 3.3). bf16 peak 186.6 TFLOPS of 197. |
| TPU v6e | In progress (Colab) | Same `tpu_bench.py`. |
| `calibration/predictions/*.txt` | First uncalibrated Stream-based predictions | Decode predicted ~0.80x measured; memory-bound geomean pred/meas 0.77; tiny decode attention 0.34-0.42x (no launch overhead modelled); compute-bound 0.90-1.36x; A100 prefill `ffn_gate_up` infeasible in Stream. |
| `third_party/stream` branch `fix/sim-correctness` | Reference oracle | Plus experimental variants `stream-bus`, `stream-hashseed`, `stream-loop`, `stream-split` (to be consolidated, section 9). |
| `harness/designs/*.json` | Legacy reference designs a100, a100_40gb, tpuv4, tpuv5e, tpuv6e | Ported in section 9. |

Measurement-noise facts already visible, which bound how tight any target can be:

- GPU `flushed` vs `graph_cold` differ by 1-6% on mid/large ops and by 35-60% on tiny ops (e.g. `bmm_8_4_2048_128`:
  13.3 us vs 8.4 us). Tiny-op error is therefore dominated by what overhead a timing mode includes.
- A100 clock varies by op class (1410 MHz on memory-bound ops, 1275-1335 MHz on large GEMMs). Clock is a modelled
  quantity (power-cap solve), never a per-op constant.
- TPU `single` vs `loop` differ by up to 20x on tiny ops (host dispatch).

---

## 2. Layered validation strategy

Six layers. Each has an owner crate/tool, a cadence (section 5), and a pass criterion. A layer never substitutes
for another: L3 (differential) catches bugs, only L5 (measured hardware) establishes accuracy.

| Layer | Question answered | Evidence strength | Cadence |
|---|---|---|---|
| L1 Unit tests | Does each function do what its doc says? | Code correctness | Every commit |
| L2 Property / invariant tests | Does the model obey physics and symmetry on arbitrary designs? | Structural correctness, catches hack surfaces | Every commit (256 cases), nightly (10k) |
| L3 Differential vs Stream fork and ZigZag | On the shared subset, do we agree with independent models (or know why not)? | Bug-finding only | Nightly |
| L4 Tier A vs tier B agreement | Is the fast tier a faithful proxy for the slow one? | Internal consistency | Every commit (small corpus), nightly (full) |
| L5 Measured hardware | Do we predict real silicon? | Accuracy (the only one) | Per calibration release; hardware runs on demand |
| L6 Physical model vs published chips | Are area, power, shoreline, thermal plausible? | Physical-envelope accuracy | Per kiln-phys release |

The trust suite (section 12) adds simulated evidence on top of these layers (chip-history backtest, known-answer
and metamorphic tests, red-team campaigns, cross-simulator agreement) and gates every claim and release.

### 2.1 L1: unit tests per crate

| Crate | Required unit-test coverage (minimum) |
|---|---|
| `kiln-ir` | Every validation rule has one positive and one negative test asserting the exact error code + entity path; serde round trip for every type; JSON/RON/YAML equivalence; array expansion (`count` + layout) produces expected instance ids; hash stability (golden hash for each reference design). |
| `kiln-phys` | Technology table lookups; wire delay/energy against hand-computed values for 3 lengths per node; area roll-up equals sum of parts; placer determinism; shoreline accounting; power-cap clock solve converges and is monotone in cap. |
| `kiln-cost` | Loop-nest cost for 10 hand-worked mappings (incl. ZigZag paper examples) to exact cycle/energy values; tile-quantization (waves, partial tiles) on edge shapes (m=1, k<unit width, primes); precision-specific MAC throughput incl. MX block scaling overhead. |
| `kiln-map` | Partitioner covers every output element exactly once (checked by explicit enumeration on small shapes); capacity feasibility; seeded search reproducible; infeasible ops produce `E-MAP-*` with the offending memory path. |
| `kiln-sim` | Tier A on single-op designs equals closed-form roofline + overhead terms; tier B event queue ordering (ties broken by id); link/port contention on 2-flow and 3-flow micro-cases with analytically known answers; collectives (ring/tree all-reduce) vs alpha-beta closed forms. |
| `kiln-trace` | Trace totals equal result totals (time, energy, bytes per level); Perfetto export loads (schema check) and round-trips event counts. |
| `kiln-py` | Result dict schema conformance; GIL release; exception mapping; batch ordering. |
| `kiln-cli` | Exit codes (section 7) for each failure class; `--format json` validates against the published JSON schema. |

### 2.2 L2: physical invariant / property tests

Implemented with `proptest` against a random-design generator `kiln-ir::arb` (feature `arbitrary`), which emits
valid designs drawn from bounded but structurally open distributions: 1-8 chips, 1-256 tiles per chip, 1-5 memory
levels, 0-4 near-memory units, mixed precisions, random NoC topologies from 01's topology set, random floorplans
accepted by 04's placer. Workloads are drawn from the built-in suites plus random GEMM/BMM/elementwise ops
(`kiln-wl::arb`, 02 §11.4). Failing cases shrink and are persisted in `proptest-regressions/` (committed).

| # | Invariant | Tier | Tolerance |
|---|---|---|---|
| P1 | Time >= every physical floor: compute (per precision), bytes/bandwidth per memory level, per link, per collective (ring bound `2(N-1)/N * bytes / link_bw`) | A, B | Exact (floor uses no calibration efficiency < 1 except where 03 states otherwise) |
| P2 | Energy conservation: total = sum of per-component energies; average power = energy / time; energy >= compulsory floor (MACs x E_mac_min + compulsory bytes x E_byte_min per level) | A, B | 1e-9 relative |
| P3 | Monotonicity: raising any bandwidth, clock, unit count, or capacity (with area/power allowed to grow) never increases time | A | Exact (no increase) |
| P3b | Same as P3 | B | <= 2% increase (list-scheduling anomalies); > 2% is a bug |
| P4 | Renaming invariance: renaming ids, permuting declaration order of components/ops gives bit-identical numeric results | A, B | Bit-exact |
| P5 | Array expansion equivalence: `count: N` compact form vs N explicit instances | A, B | Bit-exact |
| P6 | Dead-hardware invariance: adding a component no op can reach leaves time unchanged (area/power increase); adding a reachable unit no op can use never makes anything faster (04 §6.2, §7.2) | A, B | Unreachable: bit-exact time and links; unusable unit: time and every link latency/energy never lower (1e-9) |
| P7 | 1-chip system expressed as a multi-chip system with one chip equals the single-chip result | A, B | Bit-exact |
| P8 | Data-parallel replication with no communication: N chips at batch N*b equals one chip at batch b | A, B | 1e-9 relative |
| P9 | Tier relation (03 4.6): `T_A2 <= T_B` under the same mapping, lowered graph, calibration and clock | A vs B | Exact; violation is an error (03 I9) |
| P10 | Thread-count invariance: results identical for 1, 2, N worker threads | A, B | Bit-exact |
| P11 | Serialization round trip: IR -> JSON -> IR identity; result -> JSON -> result identity | n/a | Exact |
| P12 | Unit sanity: every reported time, energy, bandwidth finite and >= 0; no NaN reaches a result | A, B | Exact |
| P13 | Area/power monotone in component count and size (kiln-phys) | phys | Exact |
| P14 | Precision ordering: same design, lower-precision operands never slower when the unit supports both | A | Exact |

A P-failure is an error in CI and also feeds the reward-hacking defenses: any invariant a random generator can
break, an evolutionary search will break deliberately.

### 2.3 L3: differential testing vs the Stream fork and ZigZag

Purpose: find kiln bugs and document intentional modelling differences. Neither oracle is ground truth.

**Expressible subset** (the "legacy subset", `kiln-ir` profile `profile = "stream_compat"`; `kiln validate
--profile stream_compat` checks membership):

| Dimension | Inside subset | Outside subset |
|---|---|---|
| System | 1 chip | Multi-chip, links, collectives |
| Compute | Matrix and vector units with a single precision each, attached to one shared on-chip memory level (legacy `attach`) | Near-memory/in-memory units, heterogeneous attach graphs, NoC topology |
| Memory | <= 2 on-chip levels (per-unit buffer + shared L2) + 1 off-chip DRAM | > 2 levels, multiple DRAMs, CXL |
| Precision | bf16 operands, fp32 accumulate | fp8, MX, int4 |
| Workload | Single-operator dense GEMM/BMM (legacy ONNX per op), operands initially off-chip, nothing resident across ops | Fusion, inter-op residency, elementwise/softmax, parallelism |
| Physical | Disabled (oracles have no floorplan); kiln run with `--phys none` and `wire_model = ideal` | Floorplan-driven link latency/energy |
| Overheads | Calibration set `null` (all efficiencies 1, launch overhead 0) | Calibrated |

**Comparisons and tolerances:**

| Comparison | Mode | Tolerance | On violation |
|---|---|---|---|
| kiln-cost vs ZigZag, fixed mapping (same loop nest imported via `kiln diff-test --import-mapping`) | `cost_model = zigzag_compat` | Latency and energy within 1% per op | Bug in kiln-cost unless listed in `known_diffs.json` |
| kiln-cost vs ZigZag, each with its own mapper | default | kiln best latency <= ZigZag best x 1.02; kiln energy <= ZigZag energy x 1.05 | kiln mapper regression |
| kiln tier A vs Stream fork, per op, legacy designs + 50 random subset designs | `stream_compat` | Median abs(ratio-1) <= 10%, p95 <= 25% | Triage |
| kiln tier B vs Stream fork | `stream_compat` | Median <= 10%, p95 <= 20% | Triage |
| Design ranking (Spearman over subset corpus per op) | both | rho >= 0.9 | Triage |
| Feasibility agreement | both | Every op Stream finds feasible, kiln finds feasible | kiln bug; the reverse (kiln feasible, Stream infeasible, e.g. A100 `ffn_gate_up`) is an expected Stream limitation, logged |

**Triage outcome** for every disagreement beyond tolerance is exactly one of: `kiln_bug` (fix), `oracle_bug`
(record upstream issue/commit in `known_diffs.json`), `intended` (record the modelling difference and a one-line
justification). `known_diffs.json` entries carry the oracle version and expire when the oracle version changes.
Oracles run in a pinned Python env on the nightly Linux runner, never on the user's Mac.

### 2.4 L4: tier A vs tier B agreement corpus

Corpus `corpus/agreement/` (versioned, hashed):

1. All reference designs (section 9.2), native IR versions.
2. 200 seeded random valid designs from `kiln-ir::arb` (seed list committed), 25% multi-chip.
3. Every design ever promoted to an evolution archive elite or flagged by the suspicion audit (appended
   automatically by the audit path, deduplicated by design hash). Elites are where model holes live, so the corpus
   grows exactly where it matters.

Workload set: built-in suites (02 §11.4 `evolve` set), whole step per phase (03 §4.9) with the per-layer breakdown.

| Metric | Target | Rationale |
|---|---|---|
| Per (design, layer) time, abs(T_A_est / T_B - 1) | median <= 5%, p95 <= 10% (03's `eps_AB`), max <= 30% (beyond: bug triage) | Tier A is the selection signal; p95 bounds how often a false winner survives to audit |
| Whole-step time (the scored unit) | <= 5% | 03 4.6 |
| Interval: `sensitivity` vs `corners` low/high (03 §9.1) | p95 abs difference <= 2 points of relative width | Evolution scores with the cheap linearization |
| Per (design, layer) energy | median <= 2%, p95 <= 5% | Energy is mostly count-based; contention changes static energy only |
| Rank agreement per workload | Kendall tau >= 0.9 (03 4.6), Spearman >= 0.95; top-10 overlap >= 8 | Evolution needs ordering more than absolute values |
| Ratio fidelity: (design / A100-baseline) speedup, A vs B | abs(A ratio / B ratio - 1) <= 8% at p95 | Score is a ratio; this is what the suspicion audit threshold is sized against |
| Floor relation P9 (`T_A2 <= T_B`) | 100% | Provable (03 4.6); any violation is an engine bug |

Violation of max or P9 is a bug. Median/p95 drift between releases > 2 points blocks release.

### 2.5 L5: measured-hardware validation

| Device | Access | Status | Measured content | Role |
|---|---|---|---|---|
| A100-SXM4-40GB | Colab | Done (1 session) | 29 layer ops, GEMM sweep, peaks, NVML clock/power | Fit device (platform set `platform:a100_40gb`), generic-v1 fit |
| TPU v5e (1 chip) | Colab | Done (1 session) | Same ops via JAX, HBM microbench, peaks | Held-out device for generic-v1; fit device for generic-v2 |
| TPU v6e (1 chip) | Colab | In progress | Same | Held-out device for generic-v1 and generic-v2 |
| H100-SXM5-80GB | RunPod | Planned (M2b) | Same + fp8 GEMM sweep | Held-out device (different SM generation, HBM3, 700 W cap) |
| MI300X | RunPod (if available) | Optional | Same via PyTorch ROCm | Held-out device from a different vendor and chiplet architecture; strongest generalization test |
| 8x A100 or 8x H100 NVLink/NVSwitch node | RunPod | Planned (M6) | `nccl-tests` all-reduce, all-gather, reduce-scatter, all-to-all, 1 KiB-4 GiB; tensor-parallel Llama layer (TP=2,4,8) | Multi-chip validation: links, collectives, TP overlap |
| 2-node 16-GPU (InfiniBand) | RunPod (if offered) | Optional | Same collectives inter-node | Inter-node link model |
| TPU v5e-8 (or v5litepod-8) | Kaggle TPU VM or GCP spot, if accessible | Optional | `psum`/`all_gather` sweep over ICI, sharded layer | Torus/ICI collective validation |
| MLPerf Inference published results | Public | Reference | System-level tokens/s | Weak evidence: software stack dominates; directional only, used by the trust-suite backtest (12.1) |

Evidence grades used in reports: **G1** our measurement with full provenance; **G2** our measurement with incomplete
provenance (e.g. no power telemetry on TPU); **G3** vendor/paper microbenchmarks; **G4** MLPerf and other
system-level published numbers. Acceptance targets (3.5) apply to G1/G2 only. G4 targets: phase-level within 25%,
used to catch gross errors (e.g. a modelled 8xH100 Llama-70B throughput 2x off MLPerf is a bug, not noise).

Comparison semantics: every measurement record declares which overheads its timing mode includes
(`overheads_included`: subset of `{host_dispatch, kernel_launch, inter_kernel_gap, l2_warm}`) and its residency
(`operands: cold_dram | l2_warm | resident`, which select 02 `eval_mode` `isolated{cold}`, `isolated{warm}` and
`whole_graph` respectively, 02 §12.6). kiln computes the prediction in the matching mode (03 must expose
these as evaluation modes); comparing a mode-mismatched pair is an error (`E-CAL-MODE`). Canonical modes: GPU
`graph_cold` (`kernel_launch` + `inter_kernel_gap` inside a CUDA graph, cold L2), TPU `loop` /
`pipelined_rot` minimum (no host dispatch, cold).

Phase-level validation has two forms: (a) `sum`: count-weighted sum of isolated-op measurements (available now;
calibration only); (b) `sequence`: a whole step (full Llama-3-8B decode step, all layers plus embedding, head and
sample; full prefill step) captured as one CUDA graph on GPU or one jitted function on TPU, with realistic
inter-op and inter-layer residency (added in M2), compared against kiln's whole-step prediction (03 §4.9). (b) is
the scored semantics for designs and baselines (02 §12.5); the measured A100 whole-step `sequence` result is the
baseline anchor.

### 2.6 L6: physical model vs published chips

Reference set (values live in `calibration/physical/chips.json`, each with a citation; values below are
the targets to source, flagged "verify" until cited in that file):

| Chip | Node | Die area | Power (cap / TDP) | Other checks |
|---|---|---|---|---|
| A100 (GA100) | TSMC N7 | 826 mm^2 | 400 W (SXM4) | 40 MB L2, 108 SMs enabled of 128, 5 HBM2 stacks active of 6 (shoreline) |
| H100 (GH100) | TSMC 4N | 814 mm^2 | 700 W (SXM5) | 50 MB L2, 5 HBM3 stacks active |
| TPU v4 | 7 nm | < 600 mm^2 (verify) | verify (ISCA 2023) | 4 HBM2 stacks? verify; 128 MiB CMEM |
| TPU v5e / v6e | verify | verify | verify | Use only if published |
| MI300X | N5 XCD + N6 IOD | per-chiplet (verify) | 750 W | Chiplet + 3D stacking test of 04's package model |
| Measured | | | A100 NVML power per op class (G1) | Our own sessions: ~400-410 W on large GEMM at 1275-1335 MHz |

Targets: die area within +/-15% of published; modelled power at the cap-limiting workload within +/-20% of
cap/TDP; NVML-measured power per op class (GEMM, GEMV, attention) within +/-20% (G1, A100 now, H100 later);
modelled clock under cap on the large-GEMM class within +/-5% of the NVML median (A100: 1275-1335 MHz).
Component-level sanity (SRAM area fraction, MAC area fraction) vs published die-shot analyses is G3 and reported,
not gated. Novel components (near-memory compute, MX units) have no G1 evidence; results using them carry the
`extrapolated` flag (6.6) until evidence exists.

---

## 3. Calibration methodology

### 3.1 Rules

1. **Only registered parameters.** Calibration may set only parameters from the registry in 3.2. Each is
   physically meaningful, has a unit, bounds, and a default prior. The calibration-set schema has no field keyed by
   op, op key, shape, or phase; CI rejects any set containing an unregistered parameter. A free per-op scalar is
   impossible to express, not merely discouraged.
2. **Parameters are keyed by mechanism, not by device.** Each parameter's key is the technology or execution
   model it describes (03 section 9, principle 6): `dram_kind` (e.g. `hbm2e`), `exec_model` (`host_launched`,
   `device_queued`, `static_dataflow`, 01 §3), clock/power `domain` class with process node, `unit_template`, `link_kind`.
   Two kinds of set: **platform sets** (`platform:a100_40gb`, `platform:tpu_v5e`, ...) bind keys to one known chip
   and exist only to validate mechanisms on that silicon; **generic sets** (`generic-vN`) hold values pooled over the
   fit devices per technology key and are used for every design that is not a reference chip, i.e.
   everything evolution produces. A generic set never contains a platform key. A key absent from the generic set
   (e.g. `hbm3` before any HBM3 device is measured) falls back to its prior and the result carries
   `extrapolated: [param keys]` and the key's plausible band as its range (3.2).
3. **Symmetric evaluation.** In evolution, the baseline (e.g. A100) and every candidate are evaluated with the
   same calibration set, which is `generic`. Evaluating the baseline with its own platform set and the candidate with
   `generic` is forbidden by the API (`E-CAL-ASYM`); a candidate must not win because of calibration asymmetry.
4. **Held out means held out.** Splits are fixed before fitting (3.4), versioned, and hashed. The test split is
   evaluated only by `kiln calibrate report`, which appends an entry to the set's `test_access_log`. More than
   three test accesses per set version without a version bump fails CI (forces honest reporting of iteration on
   test data).
5. **Data/parameter ratio >= 10.** A fit with fewer than 10 fit records per free parameter is refused.
6. **Calibration sets are data.** `calibration/sets/<set_id>.json`, canonical JSON, sha256 over canonical form,
   immutable once referenced by a release; changes create a new version.

### 3.2 Parameter registry

The registry is the union of 03 section 9 (performance) and 04 (power/physical); names match 03 exactly. This
table adds the key type, the fit source, and the starting estimates from our data.

| Parameter (03 name) | Key | Unit | Term it enters | Bounds | Fit on | Starting estimate |
|---|---|---|---|---|---|---|
| `eta_res` | `dram_kind` | 1 | DRAM residual efficiency after modelled refresh/turnaround/row-miss (03 5.3) | [0.5, 1.0] (08 §F) | Copy sweeps, weight-streaming GEMV sweeps at sizes >> L2 | A100 GEMV 1.26-1.42 TB/s of 1.555; v5e GEMV 0.72-0.77 of 0.819 TB/s |
| `t_launch` | `exec_model` | s | Exposed launch latency (03 4.4) | [0.5e-6, 20e-6] | Empty-kernel and tiny-op series | fit |
| `t_dram_ramp` | `dram_kind` | s | DRAM stream fill + drain per segment (03 §9): `eta_res` as a bounded function of transfer size | [0, 20e-6] | Same sweeps as `eta_res`, all sizes | fit (M2) |
| `t_min_kernel` | `exec_model` | s | Minimum kernel duration | [0.1e-6, 20e-6] (M2: the A100 empty-kernel graph chain measures `t_min_kernel + t_gap` = 0.90 us) | Same | A100 graph mode ~5 us floor at 256^3 |
| `t_gap` | `exec_model` | s | Inter-kernel gap inside a queue/graph | [0.1e-6, 20e-6] | Back-to-back tiny kernels, graph vs eager | fit |
| `f_cap_op` | `domain` x `power_cap` | 1 (scales an `[activity, Hz]` table) | Sustained clock under the power cap vs MAC-pipe activity, telemetry operating points (03 4.5 stand-in; platform sets only) | [0.9, 1.1] | NVML clock telemetry of the GEMM/GEMV sweeps | M2 |
| `t_sync` | `exec_model` x scope | s | Barrier latency per scope (unit set, chip, system) | [0.1e-6, 50e-6] | Barrier microbenchmarks | fit |
| `t_coll_setup` | `link_kind` x algorithm | s | Collective setup (alpha) | [0.1e-6, 50e-6] | Tiny-message nccl-tests / `psum` | fit (M6) |
| `kappa_E_*`, `kappa_clk`, `p_ll`, `p_ls` (04 §12.1; replaces `C_eff`, 00 decision 6) | node / technology family (04 §12.1 scope keys) | 1, W/mm^2, W/Mib | Power model and power-cap clock solve (03 4.5, 04 §8) | per 04 | NVML power + clock under dense GEMM, at several power caps where root allows (RunPod) | A100: 1275-1335 MHz at 400 W on large GEMM, 1410 MHz on GEMV |
| `tau_pm` | `domain` class | s | Power-management window | [1e-4, 0.1] | Step-load telemetry | 1 ms assumed |
| V/f curve (`V_th`, `alpha`, 04 §4.7) | `domain` class x node | V, 1 | DVFS table | per 04 | Locked-clock runs (`nvidia-smi -lgc`, RunPod root only) | 04 tables |
| `unit_eff(k_depth)` | `unit_template` | 1 | Per-unit pipeline efficiency vs innermost reduction depth, monotone non-decreasing, <= 3 knots | [0.7, 1.0] | GEMM sweeps after quantization and clock are modelled | A100 large GEMM: 273 TF vs 312 x 1290/1410 = 285 TF peak-at-clock, ~0.96 |
| `eta_link` | `link_kind` | 1 | Link protocol efficiency | [0.6, 1.0] | Point-to-point bandwidth tests | fit (M6) |
| `rho_max`, contention scalar | engine | 1 | Tier A contention correction (03 4.3) | per 03 | Tier B only, never hardware | per 03 |

Not fitted, by design: tile/wave quantization (03 computes it from shapes and unit geometry), clock (solved from
the power model and cap), refresh/turnaround/row-miss DRAM terms (computed from timing tables), cache/residency
behaviour (modelled explicitly), DRAM `lat(class)` and `eta(class)` tables (generated offline from Ramulator 2,
4.6). If a residual looks like "shape-dependent efficiency", the fix is in a mechanism (03/04), not a new
parameter. Adding a parameter requires a change to this table and to 03 section 9.

**Ranges (03 §9.1).** Every parameter in a generic set carries `lower / central / upper` per key, never per op,
shape or phase (rule 1 applies unchanged; a range is part of the parameter, not a new parameter). `central` is
the pooled fit. The range is derived, never tuned to a result:

| Evidence for the key | `lower`, `upper` | `basis` |
|---|---|---|
| Fitted on >= 2 devices | min and max of the leave-one-device-out per-device fits, each widened by its bootstrap CI95 | `device_spread` |
| Fitted on 1 device | CI95, widened to at least the plausible band's half-width around central | `single_device` |
| Not measured (`extrapolated`) | plausible band | `band` |
| Physical constants (04) | 04 §3 confidence mapping | `confidence` |

**Generic range policy (cross-chip intervals, 08 §F).** The table above gives ranges on a set's fit devices
(fitted bootstrap ranges, unchanged). On every other design (all novel designs and held-out chips, i.e. any
design whose `family` is not in `fit.fit_devices`), a generic set's `range_policy` widens each listed parameter's
range to span the spread observed across measured devices; central values are never changed. Each policy range
is exactly the span of its recorded evidence (`{device, quantity, lower, upper, n, source}`, validated): the set's
own fitted ranges, plus direct calib-micro measurements of devices the set did not fit. `eta_res`: fitted ranges,
plus achieved/peak DRAM bandwidth of weight-streaming GEMVs (m <= 16, weights >= 64 MiB) on unfitted devices
(2026-10-05: A100 fit 0.853-0.952, v5e GEMV 0.884-0.922 or fit 0.897-0.922, v6e GEMV 0.666-0.693 in `loop` mode
(0.667-0.856 under the former `best`, 08 §F.4); span 0.666-0.952). `unit_eff`: best achieved/peak bf16 FLOP/s at nominal clock on GEMMs with m, n, k >= 2048 (A100
0.877, v5e 0.975, v6e 0.836). Per-kernel launch terms (`t_min_kernel`, `t_gap`, `t_dispatch`, `t_sync`): fitted
ranges plus the marginal per-kernel cost of the launch-chain probes (A100 graph 0.89 us, v5e 0.10 us, v6e 0.12
us); per-program terms (`t_launch`, `t_program`): the single-kernel program/graph launch (v5e 2.1 us, v6e 3.2 us,
A100 6.1 us). `kiln calibrate policy` re-derives the policy without refitting; `kiln calibrate fit` writes it.

Plausible bands (initial values, assumed, reviewed per calibration release): `eta_res` [0.85, 0.97] of the
table-derived efficiency; `eta_link` [0.75, 0.95]; `unit_eff` [0.85, 1.0]; launch, gap, sync and setup terms span
the fastest to slowest measured value of the same `exec_model` class (an unmeasured class takes the full span
over measured classes); power coefficients per 04 §3. Bands are narrower than registry bounds by design: they
describe silicon that has been built, not the worst conceivable case.

Identifiability budget: about 8 performance parameters per single-chip platform plus the power group. Current A100
data (34 sweep shapes + peaks usable for fitting, after the leakage rule in 3.4) supports only 2-3 free
parameters at the 10x ratio, so M2 starts with a dedicated microbenchmark session (4.2 `calib-micro` suite,
~300 records per device) before any release-grade fit.

### 3.3 Fit procedure

1. **Staged in 03's fixed order, each group on its own microbenchmarks:** (P) clock/power (`kappa_E_*`, `kappa_clk`,
   leakage, V/f, `tau_pm`) on power/clock telemetry; (L) launch (`t_launch`, `t_min_kernel`, `t_gap`, `t_sync`) on tiny
   kernels; (D) DRAM (`eta_res`) on copy and GEMV sweeps with P and L frozen; (U) `unit_eff` on GEMM sweeps with
   P, L, D frozen. No joint refit across groups (joint fits trade mechanisms against each other).
2. **Loss:** Huber (delta = 0.05) on `log(pred / meas)`, so over- and under-prediction cost the same and outliers
   do not dominate; plus a Gaussian prior term per parameter (sigma = 1/4 of its bound width).
3. **Optimizer:** bounded deterministic least squares (Levenberg-Marquardt / L-BFGS-B, fixed iteration order) in
   `kiln calibrate fit`; tier A in the inner loop; final residuals reported for tiers A and B.
4. **Bounds are alarms:** a parameter that converges onto a bound fails the fit (03 section 9, principle 2): the
   structural model is wrong for that platform and gets fixed, not accepted.
5. **Uncertainty:** 200 seeded bootstrap resamples over fit records; 95% CI per parameter. A parameter whose
   CI spans more than half its bound range is non-identifiable: frozen at its prior, `frozen_reason` recorded.
6. **Residual structure test:** on held-out records, regress `log(pred/meas)` on op class, log arithmetic
   intensity, log bytes, and log FLOPs; any slope significant at p < 0.01 with effect > 5% across the observed
   range, or any op-class geomean bias outside [0.95, 1.05], fails the release (03 section 9, principle 7).
7. **Dual reporting:** every calibrated result also reports the uncalibrated prediction and per-parameter
   contributions (03 section 9, principle 5); the calibration report lists the largest corrections.

### 3.4 Splits and leakage rules

- **Fit on microbenchmarks, test on workloads** (03 section 9, principle 3). The `fit` split is the
  `calib-micro` suite (copies, GEMV/GEMM sweeps, tiny kernels, barriers, collectives, power steps). The LLM layer
  ops and whole-step `sequence` measurements are `test`-only, always. Split names are `fit` and `test` (not
  `train`, which kiln does not use since training is out of scope).
- **Shape-family leakage rule:** a fit record is dropped if its shape family equals that of any test op.
  Family key = (op kind, m bucket {1, 2-16, 17-128, >128}, n, k, batch bucket, dtype). Example: today's A100 sweep
  entries `gemm_{1,8,32}_4096_4096` coincide with decode `o_proj` and are excluded from the fit split.
- **Secondary diagnostic split:** within `calib-micro`, 30% of shape families held out by
  `sha256(family_key + split_salt) mod 10 < 3`, to separate "mechanism wrong" from "workload different".
- **Held-out devices:** leave-one-device-out. The generic set is fitted on fit devices only and predicts the
  held-out device using only technology/exec-model keys:

| Generic set | Fit on | Held-out devices reported | Expected extrapolated keys |
|---|---|---|---|
| generic-v1 | A100-40GB | TPU v5e, TPU v6e | `StaticDataflow` launch terms, TPU domain power (no telemetry) |
| generic-v2 | A100 + TPU v5e | TPU v6e, H100 | `hbm3` `eta_res`, Hopper `unit_template` |
| generic-v3 | A100 + v5e + v6e | H100, MI300X (if measured) | as above (on hold: 08 §F keeps v6e out of generic sets) |
| generic-v4 | all single-chip + multi-chip nodes | next new device | |

Platform sets: `platform:a100_40gb` and `platform:tpu_v6e` (08 §F: v6e calib-micro in `loop` mode, chip-state
gated; v6e is held out of every generic set).

Splits are files (`calibration/splits/<split_id>.json`: salt, family keys per side, record hashes), hashed and
referenced from each calibration set.

### 3.5 Acceptance targets

`err = pred / meas - 1` per record; phase error uses count-weighted sums of predictions and measurements.

| Metric | Target | Rationale |
|---|---|---|
| Per-op abs(err) median, LLM-suite ops (test), same device, platform set | <= 8% | Above the ~2-5% run-to-run noise; tight enough that mechanism bugs (15-30% errors today) are visible |
| Per-op abs(err) p90, LLM-suite ops, same device, platform set | <= 20% | Tiny ops have the noisiest measurements (35-60% mode sensitivity) |
| Per-op-class geomean bias (pred/meas) | within [0.95, 1.05] | Bias, not scatter, is what flips design rankings |
| Phase-level abs(err), `sum` form, every phase | <= 5% | Calibration check; errors partially cancel across ops but bias does not |
| Whole-step abs(err), `sequence` form (M2+), every phase | <= 8% | The scored unit; adds inter-op and inter-layer effects |
| Whole-step interval coverage: measured value inside predicted `[low, high]` (generic set, held-out devices) | >= 80% of steps | Ranges must be plausible, not decorative; coverage far above 95% with wide intervals means bands are too loose |
| Held-out device, generic set, per-op median | <= 15% | Transfer to unseen silicon is the actual use case |
| Held-out device, generic set, phase-level | <= 12% | Same |
| Cross-device ratio fidelity: predicted vs measured (device X / A100) per phase | within 10% | Evolution claims are ratios; with the suspicion threshold (6.6; 1.15x until M2b, then 1.3x), a 10% ratio error cannot turn an audited win into a loss |
| Multi-chip (M6): collective time vs nccl-tests per message size, generic set | median <= 10%, p90 <= 20% | Alpha-beta models are well understood; larger errors indicate a topology/algorithm bug |
| Noise floor rule | target >= 2 x measured cross-session CV | A target below noise is unverifiable; if a device's noise floor exceeds 4%, its targets relax proportionally and the report says so |

Current state for comparison (uncalibrated, Stream-based): memory-bound geomean 0.77, decode phase ~0.80, tiny
attention 0.34-0.42. M2 is gated on the targets above.

### 3.6 Calibration-set file

Fields (canonical JSON, schema `kiln.calib/1`): `id` (e.g. `generic-v2`), `version`, `kind` (`platform`, `generic`, `null`), `platform` (platform sets only),
`parameters` (list of `{name, key: {dram_kind?|exec_model?|domain?|node?|unit_template?|link_kind?|scope?},
value, unit, bounds, prior, ci95, range: {lower, upper, basis}, pess_dir, status: fit|frozen|assumed,
frozen_reason?, source?}`; the schema forbids any key
naming an op, shape, phase or workload), `fit`
(`{kiln_version, git_hash, method, loss, split_id, split_hash, fit_records: [record hashes],
test_records: [record hashes], measurement_sessions: [session hashes], residuals_by_class, bootstrap_seed}`),
`acceptance` (last report snapshot), `test_access_log`, `created`, `hash`. Every result's provenance carries the
set's `hash` (00 contract).

A `null` set (all efficiencies 1, overheads 0) exists for the L3 differential subset and for floor computation.

`kiln viz --calibration` (05 section 3.9) consumes a per-record table derived from a set plus its measurements;
`kiln calibrate report --arrow <file>` writes exactly 05 §3.9's calibration-report columns (05 owns that table).

---

## 4. Benchmark harness

### 4.1 Layout

Python package `kiln_bench` (edge code, allowed by 00): `bench/kiln_bench/{manifest.py, cuda_runner.py,
jax_runner.py, nccl_runner.py, provenance.py, remote.py}`. Runners are self-contained single files that can be
uploaded to Colab/RunPod with only the manifest (today's pattern). Nothing runs on the user's Mac; `remote.py`
wraps the `colab` CLI and `runpodctl` (with `--env PUBLIC_KEY`, `--volumePath /workspace`).

### 4.2 Ops expressed identically from the workload IR

1. `kiln bench export --workload <spec> --target gpu|tpu --out manifest.json` lowers the workload IR (02) to a
   list of `BenchOp`s. A `BenchOp` (defined here) is the canonical descriptor of one bound 02 node or kernel; its legacy name comes
   from 02 §13.2 `key_for`: op kind (gemm, bmm,
   linear, sdpa, elementwise, softmax, rmsnorm, collective), shapes, dtypes per tensor (incl. fp8/MX), layouts and
   transposes, epilogue fusion, operand residency, plus `key = sha256(canonical descriptor)`. kiln uses the same
   key for predictions, so measurement-prediction joins never rely on names.
2. Each runner implements every op kind with a fixed reference implementation and records which library call
   was used (`impl`: e.g. `torch.matmul`, `F.linear`, `torch.bmm`, `F.scaled_dot_product_attention[flash]`,
   `jnp.einsum`, `lax.psum`). An op kind with no implementation on a backend is recorded as `unsupported`, never
   silently substituted.
3. Besides workload-derived manifests, `kiln bench export --suite calib-micro` emits the fit suite of 3.4:
   copy/read/write sweeps 1 MiB-4 GiB (16 sizes); weight-streaming GEMV/skinny-GEMM sweeps at widths that are not
   LLM-suite families (e.g. n, k in {3072, 5120, 10240, 12288, 24576}, m in {1, 4, 16, 64}); square and K-heavy
   GEMM sweeps (256-16384); empty-kernel and tiny-elementwise series (1-1000 back-to-back, eager and graph);
   barrier/grid-sync kernels; power-step loads (idle -> dense GEMM -> idle) with 10 ms telemetry; on RunPod
   (root): the same GEMM set at locked clocks (`nvidia-smi -lgc`) and at 3 power caps (`nvidia-smi -pl`).
   About 300 records per device; the leakage rule of 3.4 is applied at export time.
4. Whole-step `sequence` items (2.5) are exported as ordered lists of BenchOps with tensor reuse edges covering a
   full step (`--scope step`, default; `--scope layer` for diagnostics): the CUDA runner captures the full step
   (e.g. a Llama-3-8B decode step at b1/b8/b32 and the b1 prefill step, all 32 layers plus embedding, head and
   sample, with a resident KV cache) in one CUDA graph and times replays (warmup 10, iters 50; weights exceed L2,
   so no operand rotation is needed); the JAX runner jits the full step with `donate_argnums` matching the IR and times it in `loop` mode. Records
   are tagged `operands: resident` (02 `whole_graph`).

### 4.3 Timing methodology

| Backend | Modes recorded | Canonical | Notes |
|---|---|---|---|
| CUDA (PyTorch) | `flushed` (64 MiB L2 flush per iter), `unflushed`, `graph_unflushed`, `graph_cold` (rotate over >= 512 MiB of operand copies inside one CUDA graph) | `graph_cold` | Retain current `gpu_bench.py` methods verbatim; warmup 10, iters 50; `torch.cuda._sleep` pre-fill |
| JAX / TPU | `single`, `pipelined`, `pipelined_rot`, `loop` (fori_loop slope, rotated operands >= 512 MiB, perturbation carry) | `loop` (08 §F; was min(`loop`, `pipelined_rot`), 08 E.3) | Retain current `tpu_bench.py` loop v3; `pipelined` without rotation is recorded but marked untrusted (VMEM residency); `pipelined_rot` (separate programs) and the derived `best` are diagnostic only: on v6e they run 20-27% faster than any single program (08 §F.4) |
| NCCL | `nccl-tests` busbw/algbw per size, in-place and out-of-place | out-of-place median | Ring and tree forced separately via `NCCL_ALGO` |
| JAX collectives | `psum`, `all_gather`, `ppermute` in `loop` mode | `loop` | |
| CUDA / JAX whole step | `sequence`: full step as one CUDA graph replay / one jitted function in `loop` mode (4.2 item 4) | `sequence` | Scoring anchor (2.5); test split only |

Quality gates per record (failure marks the record `quality: rejected` with reason; rejected records never enter
a fit):

- Coefficient of variation of samples <= 3% (tiny ops <= 8%).
- Clock window present (GPU); throttle reasons recorded; record flagged if SM clock min < 0.9 x median.
- Session sanity: 8192^3 GEMM and 1 GiB copy measured at session start and end; drift > 2% rejects the session.
- TPU chip-state gate (08 §F): the sanity GEMM's start and end are compared; a session whose two values differ by
  more than the 3% noise gate (the per-record CV gate above) changed TensorCore state (v6e: ~600 vs ~740 TF), and
  its compute-bound records (FLOP time >= byte time at the session's spec peaks) are excluded from fits with the
  reason recorded in the split; `kiln calibrate fit` and `report` list every TPU session's state.
- Every session repeated on >= 2 distinct instances (different Colab VMs / RunPod pods) before its data can enter
  a released calibration set; cross-session CV per op is stored as the noise floor.

Locked-clock and power-cap runs need root; Colab does not allow them, so those records come from RunPod sessions
and are tagged `requires_root`.

### 4.4 Provenance (per session)

| Group | Fields |
|---|---|
| Device | vendor, SKU string, PCI device id, vbios, memory size, SM/core count, L2 size, bus width, compute capability / TPU `device_kind`, chip count and topology |
| Clocks and power | max/app/observed SM and memory clocks, power limit (default/current/max), per-record clock and power windows (median, min, max), temperature, throttle reasons, ECC mode, MIG/persistence mode |
| Software | driver, CUDA, cuBLAS/cuBLASLt, cuDNN, NCCL, torch, jax, jaxlib, libtpu, XLA flags, Python, OS kernel |
| Host | provider (colab/runpod/kaggle/gcp), instance/pod id, region if known, hostname, CPU model |
| Method | runner file sha256, manifest hash, kiln version that exported it, timing-mode constants (warmup, iters, rotate bytes, flush bytes), start/end timestamps |

### 4.5 Storage

`calibration/measurements/<vendor>/<sku>/<YYYY-MM-DD>_<session_id>.json` (schema `kiln.meas/1`): session header
(provenance) + records (`{bench_key, impl, mode -> {median_s, min_s, p90_s, n, cv}, clock, power,
overheads_included, operands, quality}`). Append-only; files never edited after commit; a correction is a new
session that supersedes the old one in `calibration/measurements/index.json` (session hash -> status). Session
hash = sha256 of canonical JSON. Raw logs stored alongside (`.log`). Legacy files (a100_2026-10-04*.json,
tpuv5e_2026-10-04.json) are imported by `kiln bench import --legacy` into the new schema with the original file
hash in provenance; the `_v0writeflush` A100 file is imported as `quality: rejected (superseded methodology)`.

### 4.6 Offline DRAM tables (Ramulator 2)

03 section 5.3 takes DRAM `eta(class, rw_mix, req_size, banks_active)` and `lat(class)` from tables generated
offline by Ramulator 2. `bench/dram_tables/` holds the driver: synthetic streams per (DRAM standard, access class,
rw mix, request size, active banks), one run per point, output `calibration/dram/<standard>-v<N>.json` with the
Ramulator commit, config files, datasheet timing citations, and a hash. Runs on a RunPod CPU pod (never on the
Mac). Tables are inputs, not fits; `eta_res` remains the only fitted DRAM term. Validation of the tables: the
modelled A100 HBM2 and v5e copy/GEMV curves before `eta_res` must already fall within 15% of measurement; a larger
gap means the table or the access-class mapping is wrong.

---

## 5. Regression corpus and CI

### 5.1 Corpora

| Corpus | Contents | Used by |
|---|---|---|
| `corpus/golden/` | Reference designs x built-in workloads; full tier A result JSON (and tier B nightly) | Golden regression |
| `corpus/agreement/` | 2.4 | L4 |
| `corpus/differential/` | `stream_compat` subset designs + op list + `known_diffs.json` | L3 |
| `corpus/adversarial/` | Every design that ever triggered a floor violation, invariant failure, failed audit, or a hacking report; with the expected error code | Regression of reward-hacking defenses |
| `corpus/perf/` | 5 reference + 50 seeded random designs for timing | Section 8 |
| `corpus/trust/` | Backtest designs and published-ratio table with sources (12.1), metamorphic relation set (12.2), red-team regressions (12.3), cross-simulator subsets (12.4) | Section 12 |

### 5.2 Cadence

CI runs on Linux runners (GitHub Actions for per-commit; nightly on a RunPod CPU pod or a large hosted runner).
Never on the user's Mac.

| Stage | Trigger | Budget | Contents | Blocking |
|---|---|---|---|---|
| Fast | Every push / PR | <= 10 min | `cargo fmt --check`, `clippy -D warnings`, unit tests all crates, proptest 256 cases/property (fixed seed + persisted regressions), golden tier A exact-match, L4 on 20-design sub-corpus, calibration report recomputed from stored measurements (no hardware), adversarial corpus, determinism check (6.3), criterion perf smoke (fail on > 20% regression of p50) | Yes |
| Nightly | Schedule | <= 3 h | Tier B golden, full L4, L3 differential (Stream fork + ZigZag in pinned venv), proptest 10k cases/property with random seed (failures become regressions), `cargo fuzz` on IR/workload parsers 30 min each, full calibration report incl. bootstrap, full perf benchmark, cross-platform determinism (x86_64 vs aarch64 Linux), trust suite T2 and T4 (12.2, 12.4) | Opens an issue; blocks next release |
| Hardware | Manual / per calibration release | per session ~1 GPU-hr | `kiln_bench` runs on Colab/RunPod, import, quality gates | Review |
| Release | Tag | | Everything above green + acceptance targets (3.5) met on current calibration set + release trust report green (12.5) | Yes |

### 5.3 Golden numbers

- Golden files are full result JSON; comparison is bit-exact on numbers (determinism makes this possible) and
  schema-exact on structure.
- Update only via `kiln corpus run --golden corpus/golden --update --reason "<text>"`, which writes the new files
  and a diff report (`corpus/golden/CHANGES.jsonl` append: date, git hash, reason, per-metric old/new/ratio,
  max abs(ratio-1) per phase). CI fails if golden files change without a matching `CHANGES.jsonl` entry for that
  commit.
- Any golden phase metric moving > 5%, or any calibration acceptance metric moving > 1 point, requires the PR label
  `model-change-reviewed` (explicit human review), and the PR description must name the mechanism that changed.

### 5.4 Determinism checks

- Same build, same inputs: run golden corpus twice in one process, in two processes, and with `--threads 1` vs
  `--threads N`; sha256 of canonical result JSON must be identical.
- Implementation rules this enforces (cross-section): no HashMap iteration into results; parallel reductions
  collect into index-ordered vectors then reduce sequentially or in a fixed pairwise tree; transcendental functions
  via the pure-Rust `libm` crate (not platform libm) so x86_64 and aarch64 agree bit-for-bit; no `-ffast-math`
  equivalents; seeded RNG (`rand_chacha`) with seed derived from (design hash, workload hash, user seed).
- Cross-platform target: bit-exact. If a platform difference is found and cannot be removed, the tolerance becomes
  1e-12 relative and the cache key includes the target triple.

---

## 6. Evolution-loop API (`kiln-py`, PyO3)

### 6.1 Surface

Module `kiln` (wheel built with maturin, abi3, Linux x86_64/aarch64 + macOS for development only).

| Function / class | Signature (Python view) | Notes |
|---|---|---|
| `kiln.Session` | `Session(calibration="generic-v1", cache_dir=None, cache="disk", threads=None)` | Holds loaded calibration, compiled workloads, fragment caches. Thread-safe; share one per process |
| `Session.evaluate` | `evaluate(design, workload, options=None) -> Result` | `design`: JSON str, dict, or path. `workload`: built-in name (`"llama3_8b:decode_b8"`, 02 §11.4), WorkloadIR JSON/dict/path, or a `WorkloadSet` |
| `Session.evaluate_batch` | `evaluate_batch(items, options=None, max_workers=None, ordered=True) -> list[Result]` | `items`: list of `(design, workload)` or designs (with `options.workload`). Releases the GIL; rayon pool; results in input order |
| `Session.validate` | `validate(design) -> list[Error]` | IR + envelope checks only, <= 5 ms |
| `Session.baseline` | `baseline(name, workload, options=None) -> Result` | The baseline simulated under the caller's options; cached per (baseline, workload, calibration hash, simulation options, kiln version) |
| `Result.to_dict()`, `.to_json()`, `.to_arrow()` | | `to_arrow` returns pyarrow tables per 05's `kiln-trace` schema |
| `kiln.explain` | `explain(result, op=None, max_items=8) -> str` | LLM-readable summary (6.5); wraps 03's `explain_run` (kiln-sim, 00 decision 7) |
| `kiln.render` | `render(result_or_path, view, fmt="png", **opts) -> bytes` | Owned by 05 (headless `kiln-viz-render`) |
| `kiln.evaluate(...)` | module-level convenience using a default Session | |
| `kiln.schema(kind)` | `kind in {"hardware","workload","result","calibration","options"}` | JSON Schema for prompting the LLM |

### 6.2 Options (`kiln.Options` or dict, schema `kiln.options/1`)

| Field | Default | Meaning |
|---|---|---|
| `tier` | `"cascade"` | `"validate"`, `"A"`, `"B"`, `"cascade"` |
| `calibration` | Session's | Must equal the baseline's (rule 3.1.3) |
| `fitness` | `{"kind": "matched_envelope", "baseline": "a100_40gb"}` | 6.4 |
| `workloads` | `"evolve"` | Which workload set produces the score (02 §11.4; 6.6 held-out) |
| `interval` | `"sensitivity"` | `"none"`, `"sensitivity"` (linearized, about 1.1x cost) or `"corners"` (at most 3x); audits and claims force `"corners"` (03 §9.1) |
| `seeds` | `[0]` | Mapper seeds; > 1 means multi-seed scoring (median) |
| `timeout_s` | `{"A": 5, "B": 300}` | Wall-clock per evaluation per tier; cooperative cancellation |
| `audit` | `{"suspicion_ratio": 1.15, "random_rate": 0.02, "seeds": 3, "heldout": true}` | 6.6 |
| `trace` | `"summary"` | `"none"`, `"summary"`, `"ops"`, `"full"` (05 §3.3; Tier A emits `ops` only on request, 00 decision 8); `ops`/`full` traces written to the cache and referenced by handle |
| `features` | standard set | Descriptor names to compute (6.3) |
| `invalid_score` | `"zero"` | `"zero"` or `"graded"` (6.4) |
| `profile` | `"search"` | 01 §18.3 validation profile applied at S0 (`full`, `reference`, `search`, `stream_compat`) |
| `parallelism` | `"across"` | `"across"` (each eval single-threaded, batch parallel) or `"within"` (one eval uses the pool) |
| `stack` | `"kiln_ideal"` | Software-stack recipe (02 §7.4.1) both sides of `score` run under: a built-in id (`kiln_ideal`, `pytorch_cuda_graph_sdpa`, `xla_tpu_fused`) or a `kiln.stack/1` file, or `"own"` (each design under its execution model's default, i.e. the realistic score). Part of the scoring basis and the cache key (by recipe hash) |

### 6.3 Result (schema `kiln.result/1`)

| Field | Content |
|---|---|
| `status` | `ok`, `invalid` (IR/validation), `envelope` (physical envelope violation), `infeasible` (no mapping), `floor_violation` (simulator bug, see below), `pruned` (cascade bound), `timeout`, `internal_error` |
| `score` | float; the fitness (central by default, 6.4); 0 unless `status == ok` and audits (if any) passed; `graded` mode gives negative values for invalid (6.4) |
| `score_interval` | `{low, central, high}`: candidate's corner ratios against the baseline's central value (6.4) |
| `score_components` | per-phase ratios (each `{low, central, high}`), weights, aggregation, baseline id + hash, `candidate_stack` / `baseline_stack` (`id@hash` of the recipe each side ran under) |
| `score_realistic` | `{score, interval: {low, central, high}, candidate_stack, baseline_stack}`: the **realistic-stack** score, formed exactly like `score`/`score_interval` but with each side under its own execution model's default stack (PyTorch for a host-launched GPU baseline, XLA for a TPU); reported next to `score`, never used as the fitness unless `options.stack = "own"` (6.4). Absent (with warning `W-FIT-REALISTIC`) when either side does not evaluate under its own stack |
| `interval` | `{method: none|sensitivity|corners|sampled, corner_flips, drivers: top parameters by contribution to width, low_remapped?}` (03 §9.1) |
| `stage_reached`, `tier` | cascade stage and tier that produced the score |
| `phases` | per phase: `scope` (`step` or `layer`, 02 §12.5), `time_s`, `tokens_per_s`, `energy_j`, `tokens_per_j`, `avg_power_w`, `clock_hz` (solved), each as `{low, central, high}`; `floor_s` per floor kind, `roofline_frac`, `bound_breakdown` (fractions of time bound by compute / each memory level / links / overhead), `per_layer` breakdown, `trusted` |
| `ops` | per op (optional, `trace >= summary`): time, energy, bound, mapping id, floors |
| `physical` | from 04: `die_mm2` per chip, `package_mm2`, `peak_power_w`, `power_density_max_w_mm2` (each `{low, central, high}`), `hbm_shoreline_used_mm`/available, `tdp_w`, node, envelope margins at central and at the pessimistic corner |
| `features` | MAP-Elites descriptors (table below) |
| `violations` | structured errors that zeroed the score |
| `errors`, `warnings` | structured (6.5) |
| `audit` | `{status: not_run|pending|passed|failed, reasons, tier_b_ratio, seed_spread, heldout_gap, extrapolated_components, trust_report?}` (trust report handle for claims, 12.5) |
| `trace` | `{id, tier, path, viewer_url?}` handle for `kiln viz` (05) |
| `provenance` | kiln version + git hash, design hash, workload hash, calibration hash, options hash, tier, seeds |
| `timing` | wall seconds per cascade stage, cache hits |
| `calibration` | uncalibrated prediction per phase and per-parameter contributions (03 section 9, dual reporting); `extrapolated` keys |
| `invariants` | 03's `InvariantReport` (I1-I15) with margins; any failure sets `status = floor_violation` |
| `sim` (optional, `trace >= summary`) | 03's `SimResult` per phase, unmodified |

Standard descriptors (fixed ranges so archives are comparable; binning is the evolution loop's choice):

| Descriptor | Unit | Range |
|---|---|---|
| `die_mm2_total` | mm^2 | [10, 8 x 858] (reticle-limited per die) |
| `power_w` | W | [5, 8 x 1000] |
| `onchip_bytes` | B | [2^20, 2^34] (log) |
| `peak_flops_bf16` | FLOP/s | [1e12, 1e17] (log) |
| `machine_balance` | FLOP/B (peak flops / off-chip bw) | [1, 4096] (log) |
| `n_chips` | count | [1, 1024] (log) |
| `compute_tiles` | count | [1, 2^16] (log) |
| `near_mem_flop_frac` | 1 | [0, 1] fraction of workload FLOPs executed by near-memory units |
| `memory_levels` | count | [1, 8] |
| `energy_split` | 1 (3 values) | compute / memory / interconnect fractions of energy |
| `bound_frac_mem` | 1 | fraction of decode time bound by off-chip memory |
| `score_rel_width` | 1 | [0, 2] `(score_interval.high - score_interval.low) / central`: how much of the score rests on uncertain parameters |

### 6.4 Fitness definitions (configuration)

| `fitness.kind` | Definition | Envelope |
|---|---|---|
| `matched_envelope` (default) | geomean over phases of `tokens_per_s(candidate) / tokens_per_s(baseline)`, whole-step | Candidate must satisfy the baseline's envelope (limits below): total die area, per-die reticle limit, TDP/cap, process node, package size, HBM shoreline (from the baseline's `physical` result), and off-chip memory bandwidth and capacity (from the baseline's design summary); each overridable by `fitness.envelope` |
| `explicit_envelope` | same ratio | `fitness.envelope` given explicitly (e.g. `{die_mm2: 826, power_w: 400, node: "n7", offchip_bw: 1.6e12}`); physical limits it omits are unchecked, off-chip limits it omits still come from the baseline |
| `perf_per_watt` | geomean of `tokens_per_j` ratios | Envelope still enforced (prevents tiny-chip wins) |
| `perf_per_area` | geomean of `tokens_per_s / die_mm2` ratios | Envelope enforced |
| `pareto` | vector `[throughput ratio, tokens_per_j ratio, -die_mm2, -power_w]` | For NSGA-style loops; `score` = first component |
| `baseline_relative` | ratio vs baseline at the baseline's own (unmatched) envelope | Diagnostics only; refused when `audit.claim = true` |

Envelope limits (`fitness.envelope` keys; a violation sets `status = envelope` with `{value, limit, unit}` and an
actionable hint, and scores per `invalid_score`):

| Key | Default limit | Checked on the candidate's | Code |
|---|---|---|---|
| `die_mm2` | baseline total die area (central) | `physical.die_mm2` summed | `E-ENV-0001` |
| `reticle_mm2` | 858 mm^2 | each die | `E-ENV-0002` |
| `power_w` | baseline TDP | `physical.tdp_w` | `E-ENV-0003` |
| `node` | baseline node | `physical.node` | `E-ENV-0004` |
| `package_mm2` | baseline package area | `physical.package_mm2` | `E-ENV-0005` |
| `hbm_shoreline_mm` | baseline shoreline used | `physical.hbm_shoreline_used_mm` | `E-ENV-0006` |
| `offchip_bw` | baseline `HwSummary.offchip_bandwidth` (B/s) | design summary, feature `offchip_bw` | `E-ENV-0007` |
| `offchip_bytes` | baseline `HwSummary.offchip_capacity` (B) | design summary, feature `offchip_bytes` | `E-ENV-0008` |

Off-chip memory is bought, not designed (08 §F): adding stacks or raising the pin rate is not a design win. Both
off-chip limits are exact design arithmetic (sum over enabled mem stacks of `io_width_bits x pin rate / 8`, and of
capacity; 1e-9 relative tolerance), so they are enforced without a `physical` result, under every enveloped kind
(including `explicit_envelope`, which may set other values); `null` disables one explicitly. Near-memory (PIM)
units' `near.internal_bandwidth` is the stack's internal bank bandwidth, not off-chip bandwidth, and is not counted.
Until 04 produces `physical`, the physical rows are skipped with `W-ENV-0001`; stack count is covered by shoreline
then. Reference designs with more off-chip bandwidth or capacity than the baseline (v6e 1638 GB/s, H100, ember vs
the A100-40GB's 1555 GB/s) fail its matched envelope by design; cross-reference comparisons use
`baseline_relative`.

Every `tokens_per_s` above is from a whole-step evaluation (02 §12.5, 03 §4.9; `scope: layer` members
use its steady-state layer); isolated per-op timings never enter a score (they are calibration data, 3.4). **Both
sides of every ratio are simulated** (08 §F scoring basis): the baseline is the reference design simulated under the
candidate's execution model, calibration set, tier, interval method, seeds and kiln version, cached per (baseline,
workload, calibration hash, simulation options); measured data validates the simulator (2.5, 3.5) and is never the
denominator of a score or claim. A baseline result whose provenance differs from the candidate's in calibration
set, tier, seeds or kiln build, or that is not a simulated whole-step result, is refused (`E-CAL-ASYM`). The ratio is formed per corner against the
baseline's central value: `score_interval.{low, central, high}` uses the candidate's `low`, `central`, `high`
corner. `fitness.interval_basis` selects the scalar `score`: `central` (default; `score_rel_width` is reported
as a feature so the loop can see how much of a win is uncertain) or `low` (pessimistic search).

**Software stack in scoring** (08 §F owner ruling). `score` runs candidate and baseline under the same stack
recipe, `options.stack` (default `kiln_ideal`: a hardware-only comparison, no framework kernel overhead on either
side). A score whose two sides ran under different recipes (provenance flag `stack`) is refused (`E-CAL-ASYM`),
except under `options.stack = "own"`, which declares the realistic comparison. Every result that scores also
carries `score_realistic`: the candidate under its own execution model's default recipe (`host_launched` ->
`pytorch_cuda_graph_sdpa`, `static_dataflow` -> `xla_tpu_fused`, `device_queued` -> `kiln_ideal`) over the
baseline under its own (an A100 baseline -> PyTorch), with the same corner formation and interval basis. The
fitness is the `kiln_ideal` score unless `options.stack` says otherwise; with `"own"` the two coincide. The
candidate's own-stack metrics are reused when its default recipe is the one `score` used (e.g. `device_queued`
under `kiln_ideal`); otherwise the realistic score costs one more simulation of the candidate (the own-stack
baseline is cached like the other).

Common options: `phase_weights` (default equal), `aggregation` (`geomean` default, `min`, `weighted_harmonic`),
`invalid_score` (`zero`; or `graded` = `-(1 + sum of normalized violation magnitudes)`, which orders invalid
designs below every valid one while still giving a gradient), `baseline` (reference design id; simulated in the
same session with the same calibration, kiln version, seeds and simulation options; cached). An identical design
as candidate and baseline scores exactly 1.0 at `central` for every kind; its `low` is below 1.0 (a design never
claims to beat itself).

### 6.5 Structured errors for LLMs

Error object: `{code, severity, message, path, hint, value?, limit?, unit?, section}`. Code namespaces:
`E-IR-*` (01), `E-WL-*` (02), `E-MAP-*` (03), `E-PHYS-*` (04), `E-ENV-*` envelope, `E-FLOOR-*` floors,
`E-AUDIT-*` audits, `E-CAL-*` calibration, `E-TIMEOUT`, `E-INTERNAL`. This section owns `E-ENV`, `E-FLOOR`,
`E-AUDIT`, `E-CAL`; other sections own theirs.

Rules: the message names the entity by dotted path; `hint` states a concrete change ("reduce `chip0.tile*.sram`
size_bytes by >= 3.1 MiB or remove 2 HBM stacks"); numbers carry units; at most one error per root cause
(downstream errors suppressed and counted). `Result.explain()` renders: status line, score with baseline,
top errors with hints, the dominant bound per phase ("decode_b1: 81% of time bound by chip0.hbm bandwidth"),
and the largest envelope margins; the bound/limiter lines are the text of 03's `explain_run` (00 decision 7). Length
<= 1500 chars by default. Prompts to the evolution LLM use text by default; 05's image bundle (floorplan,
bottleneck, compare-vs-parent) is attached only for archive elites and when the loop config enables multimodal
input, since its 0.5 s render budget is 10x the tier A budget.

`E-FLOOR-<Ixx>` (any failure of 03's invariants I1-I15, e.g. `E-FLOOR-I1` compute floor) is a **kiln bug**, never a design property: the result gets
`status = floor_violation`, score 0, the design is appended to `corpus/adversarial/`, and the loop is told
`"simulator bug; this design is quarantined"` so it does not learn to chase it.

### 6.6 Cascade and reward-hacking defenses

Cascade (each stage can terminate with a status; `stage_reached` reports how far it got):

| Stage | Work | Budget | Exit |
|---|---|---|---|
| S0 validate | Parse, schema, IR rules (01), workload binding | <= 5 ms | `invalid` |
| S1 envelope | kiln-phys area/power/shoreline/power density/reticle | <= 20 ms | `envelope` |
| S2 bound | Design-level floor bound per phase (no mapping): upper bound on achievable score | <= 5 ms | `pruned` if bound < `fitness.prune_below` (set by the loop, e.g. current cell elite) |
| S3 tier A | Whole-step mapping + analytical sim (03 §4.9), floors, features, intervals per `options.interval` | Section 8 design-step targets | `ok` with `audit.status = not_run` |
| S4 audit | Tier B + multi-seed + held-out workloads + `corners` intervals, triggered by rules below | Section 8 Tier B targets | `audit.status = passed/failed` |

Defenses:

| Defense | Mechanism | Effect on score |
|---|---|---|
| Physical floors | Every op and phase checked against compute, per-level bandwidth, link, collective, capacity floors and energy conservation (P1, P2) at S3 and S4 | Violation: `floor_violation`, score 0, quarantine |
| No unpaid performance | Schema rule, not a design rule: every performance-bearing IR field (bandwidth, ports, banks, link width) must have its cost computed by 04 (area, energy, shoreline, wire pitch); a field 04 cannot price is rejected by 01 validation (`E-IR-UNPRICED`). Designs whose changes are cost-neutral or cost-reducing (re-placement, topology, memory splits, dataflow) are valid and are exactly what the search should find; only performance asserted without a computed cost is blocked | Invalid |
| Suspicion audit | Triggered when score > `suspicion_ratio` (1.15 until the H100 held-out check of M2b passes, then 1.3; owner ruling C.4) x baseline, or > `suspicion_ratio` x current archive best, or on every new archive elite, plus a 2% random sample | Score replaced by the minimum over {tier A, tier B, each seed}; `audit.failed` if tier B / tier A ratio < 0.85 or seed spread > 10% |
| Multi-seed mapping | Mapper re-run with 3 seeds (S4) | Median used; spread reported |
| Held-out workloads | Workload set `heldout` (never exposed to the loop's prompts or scores): e.g. Llama-3-70B layer at TP=8, Mixtral-8x7B MoE layer, 32k-context decode, batch 128 decode, an encoder (ViT-L) layer. Defined in 02 | Reported as `heldout_gap = score_heldout / score_evolve - 1`; gap < -25% flags `overfit_workload`; claims require held-out score >= 1.1 |
| Extrapolation flag | Components or operating points outside the calibration domain (near-memory units, MX formats, links not yet measured, SRAM bandwidth density above any calibrated chip, clock above node table) | `extrapolated_components` listed; their keys carry plausible-band ranges (3.2), so the uncertainty shows in `score_interval`; optional `extrapolation_penalty` multiplies score; claims carry the flag |
| Caching hygiene | Cache key includes kiln git hash, calibration hash, options affecting metrics (including the stack recipe hash) | Stale results cannot leak |
| Sandboxing | kiln-py accepts only data (JSON/dict). Loops that let the LLM write Python `build()` programs (today's `evaluate_program`) must execute them in a subprocess with timeout, memory limit and no network before calling kiln | n/a |
| Red-team evolution | Campaigns per release and per claim with adversarial objectives (12.3); every hole found becomes a permanent regression in `corpus/adversarial/` | Hardens the model |

A **claim** (a design reported as beating a baseline) requires: tier B audit passed, 3 seeds, `interval =
corners`; **interval rule:** the candidate's `low` whole-step score >= 1.0 against the simulated baseline's
central value (both simulated under the same execution model, calibration set and options; measured `sequence`
anchors, 2.5, validate the simulator and never serve as the baseline) under **both** software-stack bases (08 §F):
`score_interval.low >= 1.0` with both sides under `kiln_ideal` (`options.stack = "kiln_ideal"`, same recipe on
both sides) AND `score_realistic.interval.low >= 1.0` with each side under its own default stack (a win that
exists only with framework overhead removed, or only because the baseline's framework is slow, is not a claim), at matched envelope with the
envelope satisfied at the candidate's pessimistic (high area, high power) corner and no `corner_flips` at `low`;
held-out score >= 1.1 (central) and >= 1.0 (`low`); no `extrapolated` components or an explicit statement of
them; calibration set meeting 3.5; trust suite green for this claim (12.5); and reproduction by
`kiln eval --tier B` from the committed design JSON.

### 6.7 Caching

- Result cache: content-addressed, key = sha256(kiln model version (crate versions + git hash of `kiln-cost`,
  `kiln-map`, `kiln-sim`, `kiln-phys`), canonical expanded design, workload hash, calibration hash,
  metric-affecting options (tier, seeds, trace level, interval method, profile, software stack by recipe hash)). Fitness config is excluded: score is recomputed from cached
  metrics. Stored at `<cache_dir>/v1/<ab>/<hash>.json.zst`, atomic write-rename; `ok`, `invalid`, `envelope`,
  `infeasible` cached; `timeout` and `internal_error` never cached.
- Fragment cache (in-process LRU + disk): mapping results keyed by (hash of the hardware sub-graph an op's mapping
  depends on, BenchOp key, seed). Evolution mutations are local, so most ops of a child reuse parent mappings.
  03 must define the dependency footprint of a mapping so this key is sound; a fragment-cache hit must give a
  bit-identical result to a miss (checked in CI by running the golden corpus with the cache disabled).
- Baseline results are cached like any other result.

### 6.8 Timeouts and failure isolation

Cooperative cancellation: mapper and event loop check a deadline every N iterations; timeout returns
`status = timeout` with partial diagnostics (stage, op in progress). A Rust panic inside one evaluation is caught
at the evaluation boundary (`catch_unwind`) and returned as `internal_error` with a backtrace hash; the batch
continues. Per-worker memory budget (default 1 GiB) enforced by estimating instance counts at S0 (`E-IR-0210`).

### 6.9 Archive output

Evolution drivers that use kiln write archives in 05's schema (05 section 3.9: `archive.json`, `designs.arrow`,
`generations.arrow`, `runs/<design_id>.kiln`), owned by 05 (00 decision 7). Columns this section requires (listed
in 05 §3.9): `trust_level` (section 10), `audit_status`, `heldout_score`, `calibration_set_hash`, `kiln_git_hash`,
`stage_reached`, `extrapolated` (list of parameter keys or component paths), `fitness_low`, `fitness_high`,
`interval_method`. `kiln-py` provides
`kiln.Archive(dir).append(result, parent_ids, mutation_summary, operator)` so drivers do not hand-write Arrow.
The audit path appends elites to `corpus/agreement/` from the archive (2.4).

---

## 7. CLI (`kiln-cli`)

Binary `kiln`. Global flags: `--calib <set_id|path>`, `--cache-dir`, `--no-cache`, `--threads N`,
`--format text|json|jsonl|llm`, `--log-level`, `--out <path>`.

| Command | Purpose | Key flags |
|---|---|---|
| `kiln eval <design>` | Evaluate one design; prints `score` (one stack on both sides) and `realistic` (each side under its own stack) | `--workload`, `--tier A|B|cascade`, `--fitness <json|path>`, `--baseline`, `--seeds`, `--timeout`, `--trace none|summary|ops|full`, `--stack <id|path|own>` (default `kiln_ideal`), `-v` per-op table |
| `kiln compare <design>...` | Relative table vs the first design | as eval |
| `kiln validate <file>` | IR/workload validation only | `--profile full|reference|search|stream_compat`, `--kind hardware|workload|calibration|measurement`, `--report`, `--allow-implausible <code>`, `--deny-warnings`, `--strict-convert` (01 §18) |
| `kiln viz <design|result|trace>` | Open the visualizer (05) | 05 owns flags |
| `kiln explain <result.json>` | LLM-readable summary | `--max-items` |
| `kiln calibrate split` | Create a deterministic split | `--measurements <glob>`, `--salt`, `--holdout-phase` |
| `kiln calibrate fit` | Fit a set | `--scope`, `--measurements`, `--split`, `--stages P,L,D,U` (3.3), `--out` |
| `kiln calibrate report` | Acceptance table; logs test access | `--set`, `--devices`, `--tier A|B` |
| `kiln bench export` | Workload IR -> bench manifest | `--workload`, `--target gpu|tpu|nccl`, `--sequence`, `--scope step|layer` |
| `kiln bench import` | Runner output -> measurement session; quality gates | `--runner cuda|jax|nccl`, `--legacy` |
| `kiln diff-test` | L3 differential | `--oracle stream|zigzag`, `--corpus`, `--oracle-python <venv>` |
| `kiln agree` | L4 agreement report | `--corpus` |
| `kiln trust run|report` | Trust suite (section 12) and trust report | `--release`, `--claim <design>`, `--parts T1,T2,T3,T4`, `--baseline` |
| `kiln corpus run` | Golden regression | `--corpus`, `--tier`, `--update --reason` |
| `kiln perf` | Throughput / latency benchmark (section 8) | `--corpus perf`, `--workers`, `--repeat` |
| `kiln trace info|validate|export|pack|unpack|upgrade|recover` | Trace container ops and exports | 05 section 8 owns flags (`--perfetto`, `--chrome-json`, `--parquet`, `--csv`) |
| `kiln viz render` | Headless PNG/SVG | 05 section 7.2 |
| `kiln fmt`, `kiln expand`, `kiln hash`, `kiln migrate` | Canonical form, expansion report, design hash, schema migration | 01 sections 14.6, 8.2, 17, 19 own flags (`--canonical`, `--inline`, `--expanded`, `--write`) |
| `kiln phys export-def|import-def` | OpenROAD audit path | 04 section 6.5 |
| `kiln schema <kind>` | Print JSON Schema | |
| `kiln import harness-design <legacy.json>` | Legacy design -> hardware IR (section 9) | |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Success; all evaluated designs `ok` (and audits passed if run) |
| 1 | Completed, but at least one design `invalid`, `envelope`, `infeasible`, `pruned`, or audit failed |
| 2 | Usage error (bad flags) |
| 3 | Input unreadable or schema-invalid (file, JSON parse, unknown workload name) |
| 4 | Check failed: calibration acceptance, agreement, differential, trust suite, perf regression |
| 5 | Golden mismatch |
| 6 | Timeout |
| 7 | `floor_violation` (simulator bug detected) |
| 70 | Internal error (panic) |

Output formats: `text` (aligned tables like today's harness), `json` (one document, `kiln.result/1` or a list),
`jsonl` (one result per line, streaming for batches), `llm` (the `explain` rendering). JSON goes to stdout or
`--out`; diagnostics to stderr.

---

## 8. Performance targets and measurement

Definitions: a **design-layer** is one design x one transformer layer of one phase (all ops of that layer,
including collectives for multi-chip). A **design-step** is one design x one whole step of one phase (03 §4.9:
a `w = 3` iteration window plus prologue and epilogue), the scored unit. The standard suite (Llama-3-8B:
prefill_b1, decode_b1/8/32) is 4 design-steps. Times are wall-clock on one core, calibration loaded, cold result
and fragment caches unless stated, `interval = none` unless stated.

| Target | Value | Today (Stream harness) |
|---|---|---|
| S0 validate | p50 <= 2 ms, p95 <= 5 ms | n/a |
| S1 envelope (kiln-phys incl. placer) | p95 <= 20 ms per chip design | n/a |
| Tier A, single-chip design-layer, cold | p50 <= 20 ms, p95 <= 50 ms | ~25-30 s (106-123 s per suite) |
| Tier A, single-chip design-step, cold | p50 <= 60 ms, p95 <= 150 ms | |
| Tier A, standard suite, fragment cache warm (typical mutated child) | p50 <= 80 ms (4 design-steps) | |
| Interval overhead on Tier A | `sensitivity` <= 1.1x, `corners` <= 3x of `none` | |
| Tier A, 8-chip design-layer | p95 <= 100 ms | n/a |
| Tier B, single-chip design-layer | p95 <= 10 s | |
| Tier B, single-chip design-step | p95 <= 40 s | |
| Tier B, 8-chip design-layer | p95 <= 30 s | |
| kiln-py call overhead | <= 1 ms | |
| Memory per worker | <= 1 GiB (typical <= 200 MiB) | 1-2 GB per Stream process |
| Throughput, tier A cascade, standard suite, 32-core Linux box, `parallelism = across` | >= 100k design evaluations / hour (realistic mix: 30% rejected at S0/S1, 70% to S3) | ~120 / hour at 4 workers |

Measurement: `kiln perf` on `corpus/perf/` (5 reference + 50 seeded random designs), 5 repeats, reports p50/p95
per stage and evaluations/hour at `--workers 1, 8, N`. Criterion micro-benchmarks for S0, S1, single-op tier A.
Reference machine is a documented RunPod CPU pod type (recorded in the perf report's provenance); CI perf smoke
compares against the last nightly on the same runner class with a 20% regression threshold.

---

## 9. Migration from the Python harness

### 9.1 Mapping of existing code

| Today | Fate | When |
|---|---|---|
| `harness/design.py` (Design schema, validation) | Superseded by `kiln-ir`. `kiln import harness-design` converts legacy JSON to hardware IR (matrix/vector units, `attach`, L2, off-chip) | M0 |
| `harness/designs/*.json` (a100, a100_40gb, tpuv4, tpuv5e, tpuv6e) | Converted to `designs/legacy/*.json` (stream_compat profile, used in L3); then re-authored natively in `designs/reference/` with 01's richer components (SM/TC hierarchy, L2 partitions, NoC, HBM stacks, floorplan) | M0 / M3 |
| `harness/workloads.py`, `workloads/llm.py` | Become built-in workload suites in 02 (`llama3_8b:{prefill_b1,decode_b1,decode_b8,decode_b32}`), identical shapes and counts; residency semantics made explicit (`operands: cold_dram`, matching v1 isolated-op approximation) | M0 |
| `calibration/oplist.json`, `export_oplist.py` | Replaced by `kiln bench export` | M0 |
| `harness/evaluate.py` `evaluate`, `check_floors`, `score` | Floors and score semantics port to `kiln-sim`/`kiln-py`. The Stream-calling path becomes `oracle/stream_oracle.py`, used only by `kiln diff-test` | M1 |
| `harness/evaluate.py` `evaluate_program` | Kept as a thin shim calling `kiln.Session.evaluate` and returning the legacy keys (`combined_score`, `features`, `valid`, `phases`) so the evolution loop keeps working; deleted at M3 | M1-M3 |
| `harness/physics.py` | Superseded by `kiln-phys`; its constants move to data files tagged `assumed` with their [H14]/[WC]/[SR]/[TS]/[OC] citations; a migration test asserts kiln-phys with the legacy constants reproduces `physics.estimate` within 1% | M1 |
| `harness/compile_stream.py`, `stream_worker.py` | Move under `oracle/` | M1 |
| `calibration/gpu_bench.py`, `tpu_bench.py`, `launch_colab.sh` | Become `kiln_bench/cuda_runner.py`, `jax_runner.py`, `remote.py`, reading manifests; timing code kept verbatim; output in `kiln.meas/1` | M0-M2 |
| `calibration/compare.py` | Replaced by `kiln calibrate report` | M2 |
| `calibration/measurements/*` | Imported via `kiln bench import --legacy`; originals kept read-only | M0 |
| `tests/test_harness.py` | Floor/validation/score tests port to Rust unit tests; Stream compile tests stay with the oracle | M1 |
| `third_party/stream` (`fix/sim-correctness`) | Pinned oracle | |
| `third_party/stream-{bus,hashseed,loop,split}` | Fixes merged into the oracle branch or dropped; directories removed | M1 |
| `.cache/harness`, `.hashseed_scratch` | Deleted | M1 |

### 9.2 Reference designs (post-port)

`a100_40gb`, `a100_80gb`, `h100_sxm`, `tpu_v4`, `tpu_v5e`, `tpu_v6e`, and multi-chip `dgx_a100_8x`,
`hgx_h100_8x`, `tpu_v5e_8`. Each carries its platform key (for platform sets) and a `sources` list. A reference design must
reproduce its own measured phase times, including the whole-step `sequence` times where measured, within the
3.5 platform-set targets before it may serve as an evolution baseline.

---

## 10. Phased implementation plan (whole project)

Principle: the evolution loop starts as early as M1, but in **shadow mode**: its results are labelled by trust
level and no design is reported as a win until the gates for that trust level pass. Trust levels appear in every
result's provenance: `uncalibrated` -> `calibrated_single_chip` -> `calibrated_physical` -> `audited` ->
`calibrated_multi_chip`.

Effort in engineer-weeks with agent assistance (rough).

| Milestone | Scope (sections) | Gate to exit | Effort | Evolution status after |
|---|---|---|---|---|
| **M0 Foundations** | Cargo workspace; `kiln-ir` core types + serde + validation for the legacy-expressible subset and the full schema skeleton (01); workload IR + Llama suites (02); `kiln-trace` result types; `kiln bench export/import`, legacy measurement import; legacy design import | Legacy designs and measurements round-trip; IR schema published; L1 for kiln-ir green; determinism rules in place (lint for HashMap in result paths) | 2 | none |
| **M1 Tier A single-chip + API** | `kiln-cost` (loop-nest cost, ZigZag-equivalent), `kiln-map` (single-chip partition/tiling, seeded search), tier A in `kiln-sim` with whole-step mode (03 §4.9), floors, `kiln-py` evaluate/batch/cache, `kiln-cli eval/compare/validate`, legacy-constant `kiln-phys` stub (03, 04 partial); trust suite T2 relations (12.2) | L1 + L2 (P1-P6, P10-P12) green; T2 green on reference designs; L3 vs ZigZag fixed-mapping 1%; L3 vs Stream on stream_compat median <= 10%; tier A p95 <= 50 ms per design-layer and <= 150 ms per design-step; uncalibrated A100 no worse than Stream baseline (memory geomean >= 0.77) | 4 | **Shadow mode** (`uncalibrated`): loop runs to exercise API, find hack surfaces (red-team campaigns), populate adversarial corpus. No claims |
| **M2 Calibration single-chip** | Registry, staged fit, splits, `kiln calibrate *`, overhead/launch modelling and power-cap clock solve in 03/04; TPU v6e session; second A100 and v5e sessions (noise floor); whole-step `sequence` bench mode (4.2); **A100 whole-step anchor** measured on Colab (CUDA-graph captured full Llama-3-8B decode step at b1/b8/b32 and full prefill step at b1, G1, two sessions); TPU jitted full step on v6e (v5e only if the model fits its 16 GB HBM); parameter ranges and `corners`/`sensitivity` intervals (3.2, 03 §9.1) | 3.5 targets: A100 platform set on LLM-suite ops; whole-step `sequence` error and interval coverage on A100 and TPU; generic-v1 held-out TPU v5e/v6e; generic-v2 held-out v6e | 3 | `calibrated_single_chip`: scores trusted for ranking; claims still blocked (no physical model) |
| **M2b H100 held-out** | RunPod H100 session (~2 GPU-hr), optional MI300X | generic-v2 held-out H100 meets 3.5 transfer targets, or the failure is diagnosed to a mechanism and fixed without fitting on H100 first | 1 (parallel) | |
| **M3 Physical model** | Full `kiln-phys`: floorplan/placer, wire model, area/power/thermal, node tables, shoreline, power density (04); native reference designs; matched-envelope fitness; `E-IR-UNPRICED` enforcement; trust suite T1 backtest designs from published specs (12.1) | L6 targets (area +/-15%, power +/-20%, clock-under-cap +/-5%); P13; reference designs reproduce own measurements; T1 single-chip backtest within tolerance | 4 | `calibrated_physical`: matched-envelope scores meaningful; shim `evaluate_program` removed |
| **M4 Tier B + audit** | Event-driven tier (contention on ports/banks/links/NoC), timelines, Perfetto export, cascade S4, suspicion audit, held-out workloads, multi-seed; trust suite T3 campaigns, T4 cross-simulator runs, trust reports (12) | L4 targets (2.4); P9; tier B p95 <= 10 s per design-layer; release trust report green (12.5) | 4 | `audited`: **first claims possible** (single chip, each with a green claim trust report) |
| **M5 Visualizer** | `kiln-viz` v1 on kiln-trace (05 phases V2-V4; V0 lands in M0/M4, V1 in M3) | 05's acceptance criteria; opens any golden trace | 3 (parallel with M4) | Elites inspectable |
| **M6 Multi-chip** | Links, packages, collectives, TP/PP/DP/EP parallelism in 02/03; RunPod 8x A100/H100 node sessions (nccl-tests + TP layers); TPU v5e-8 if accessible; T1 extended to published multi-chip systems | Collective targets (3.5); TP layer phase <= 10%; P7, P8; G4 MLPerf directional checks within 25%; multi-chip T1 within tolerance | 5 | `calibrated_multi_chip`: multi-chip claims |
| **M7 Near-memory compute + precisions** | Near-memory/in-memory unit type, fp8/MX/int4 paths (01, 03, 04); H100 fp8 sweep; published PIM results as G3 evidence | fp8 per-op targets on H100; near-memory units validated against at least one published G3 source or carried as `extrapolated` | 4 | Claims involving these carry `extrapolated` until G1/G2 evidence exists |
| **M8 Hardening** | Perf to 100k evals/hr, fuzzing, cross-platform determinism, docs for the evolution loop | Section 8 targets; nightly green 14 days | 2 | Production |

Critical path: M0 -> M1 -> M2 -> M3 -> M4 (about 17 engineer-weeks to first trustworthy single-chip claim).
M5 and M2b run in parallel. M6 can start after M4's event engine exists (links need contention).

Why this order: tier A + API first (M1) gives the loop something to run against and lets red-team campaigns harden
floors and pricing before calibration effort is spent; calibration (M2) precedes the physical model (M3) because
envelope matching is meaningless if the performance side is 25% off; tier B and audits (M4) are the last gate before
claims because they are the defense against tier A holes that evolution will find.

---

## 11. Cross-section dependencies

Checked against all sections by the integration pass (08-decisions.md).

| Needs from | What | Used in |
|---|---|---|
| 01 hardware IR | Random-design generator support (bounded, valid by construction); `stream_compat` profile; `E-IR-*` codes with paths and hints; `E-IR-UNPRICED` rule (every performance attribute priced); calibration family field; legacy import mapping | 2.2, 2.3, 6.5, 6.6, 9 |
| 02 workload IR | Bound nodes/kernels and `key_for` names for the BenchOp descriptor (defined here, 4.2); `eval_mode` semantics matching `cold_dram`, `l2_warm`, `resident` (02 §12.6); built-in `evolve` and `heldout` workload sets; whole-step scoring scope (§12.5); whole-step `sequence` export | 2.5, 4.2, 6.6 |
| 03 mapping engine | Calibration parameter set and fit order of its section 9 (adopted in 3.2/3.3); tier relation and `eps_AB` of 4.6 (numbers owned here, 2.4); invariants I1-I15 (mapped to `E-FLOOR-*`); `SimResult` (10); evaluation modes matching benchmark timing modes (`overheads_included`); explicit tile/wave quantization (not fitted); launch/sync overhead hooks; mapping dependency footprint for the fragment cache; seeded mapper; cooperative cancellation; ZigZag-compatible cost mode; floor computation API; tier A/B identical-mapping mode for P9; whole-step mode (4.9); corner evaluation and parameter ranges (9.1) | 2.2-2.4, 3.2, 6.6-6.8 |
| 04 physical model | Power-cap clock solve from `power.*` and V-f tables; per-event energies with a single scope-level scale; area/power/shoreline/power-density outputs; envelope check; published chip data with citations; confidence-to-range mapping (§3) | 2.6, 3.2, 6.3, 6.4, 12.1 |
| 05 visualizer | Trace handle and `.kiln` format; `kiln viz` flags; `kiln.render`; archive and calibration-view schemas (adopted in 6.9 and 3.6 with nullable additions); interval display (error bars, bands) | 3.6, 6.3, 6.5, 6.9, 7 |
| 07 prior art | Pinned Stream/ZigZag versions and their known limitations, feeding `known_diffs.json`; LLMCompass and GenZ as cross-simulator references | 2.3, 12.4 |

Provides to all sections: test/CI obligations, calibration parameter registry (the only fittable quantities),
error-code ownership for `E-ENV/E-FLOOR/E-AUDIT/E-CAL`, the result schema, and milestone gates.

---

## 12. Trust suite

Purpose: evidence, produced in simulation, that kiln's results can be trusted beyond the few devices we can
measure. It needs no hardware, so it runs per release and per claim. It never fits anything: no parameter, band
or mechanism is tuned on its outcomes; a failure is fixed by a mechanism change that is independently justified
and passes 3.5 (the 3.3 rule 4 discipline), logged in `corpus/trust/CHANGES.jsonl`. Gate: claims (6.6) and
releases (5.2) require the trust suite green (12.5).

### 12.1 T1: chip-history backtest

- **Designs.** `designs/backtest/`: V100-SXM2, A100-SXM4, H100-SXM5, TPU v4, TPU v5e, TPU v6e, built from
  published specs only (datasheets, vendor whitepapers, ISCA/Hot Chips papers; G3), every field sourced or marked
  `assumed`. No measured data and no platform set: each is evaluated with the current generic set, exactly like an
  evolved design. Separate from `designs/reference/` (9.2), which may use our measurements.
- **Workloads.** Whole-step LLM inference (02 §12.5) matching each published comparison: the MLPerf Inference
  datacenter LLM benchmarks where both chips of a pair have results, modelled as the submitted system (chip count,
  precision, batch or scenario as published), and Llama-3-8B whole steps for vendor-published generation ratios.
- **Checks.** Generation-over-generation throughput ratios (A100/V100, H100/A100, v5e/v4, v6e/v5e, plus
  cross-vendor pairs within one MLPerf round) and rankings, against `corpus/trust/published_ratios.json` (each
  entry: source, grade, round, system description, ratio).

| Check | Tolerance | Grade |
|---|---|---|
| Central predicted ratio vs published ratio | abs(pred/pub - 1) <= 20% (G3 vendor numbers), <= 25% (G4 MLPerf) | G3/G4 |
| Ranking of pairs whose published ratio is >= 1.3x | no order flips | G3/G4 |
| Published ratio inside the predicted ratio interval (`corners`), widened by the tolerance above | >= 80% of entries | G3/G4 |

A100 and v5e are fit devices of the generic sets (3.4), so pairs involving them are reported but flagged
`fit_device`; the independent evidence is the pairs among V100, H100, v4 and v6e and the cross-generation ratios.
Published ratios include software-stack gains; the tolerances absorb that and the report says so.

### 12.2 T2: known-answer and metamorphic tests

Quantitative responses physics implies, run on all reference and backtest designs plus 20 seeded `arb` designs per
relation, in tiers A and B, at the `central`, `low` and `high` corners. L2 (2.2) checks inequalities; T2 checks
magnitudes.

| # | Transform | Expected response | Tolerance | Mechanism checked |
|---|---|---|---|---|
| M1 | Double off-chip bandwidth (all channels, priced by 04) on a memory-bound decode step | time = `T_other + T_offchip / 2` from the result's `bound_breakdown` (about 2x speedup when fully memory-bound) | 5% | Bandwidth enters time only through DRAM resource demands |
| M2 | Double peak compute (units or rate) on a memory-bound op or step | time ratio in [0.97, 1.0] | as stated | No hidden compute dependence on bandwidth-bound paths |
| M3 | Double peak compute on a compute-bound GEMM (8192^3) | time = `T_other + T_compute / 2` after wave quantization is recomputed | 5% | Compute demand and quantization |
| M4 | Halve one link's wire length (move endpoints, re-place) | link latency and energy per bit change as 04 §7.1 predicts for the new length; bandwidth unchanged unless the pipelining stage count changes | 1% vs closed form | Placement to wire to link cost path |
| M5 | Add an idle unit (reachable, no work assigned) | time never decreases; area and static power increase | exact (time), strict (area, power) | Leakage and cap coupling; no free performance |
| M6 | Permute declaration order, rename ids | bit-identical results (P4) | bit-exact | Determinism, no name or order dependence |
| M7 | Split a memory into two with the same total capacity and total bandwidth | time and energy change only through floorplan and wire terms; with `wire_model = ideal` and fixed placement, unchanged | 1e-9 relative (ideal); residual not explained by link-cost deltas <= 1% | No free ports or bandwidth from partitioning |
| M8 | Halve weight bytes (fp8 vs bf16 weights) on a weight-streaming decode step | time = `T_other + T_weights / 2` | 5% | Precision bytes (I12) feed the memory floor |

Failures are bugs, persisted to `corpus/adversarial/` with the relation id.

### 12.3 T3: red-team evolution campaigns

Evolution runs whose fitness rewards finding model holes, per release and per claim (seeded from the claim's
lineage). Budget per campaign: 20k Tier A evaluations, top 50 audited at Tier B with `corners`.

| Objective | Fitness | A hole is |
|---|---|---|
| Tier disagreement | maximize `abs(T_A_est / T_B - 1)` and the A-vs-B score-ratio gap | beyond 2.4's max (30%) or ratio fidelity (8%) |
| Floor margin | minimize `min over floors (T / T_floor - 1)` on a whole step | margin < 1% with score > 1, unless triaged as genuinely floor-bound |
| Extrapolation leverage | maximize score gain per unit of `extrapolated` components (contribution of extrapolated keys to the score gain) | > 50% of a score gain over 1.0 resting on extrapolated keys |
| Metamorphic violation | maximize violation of any T2 relation | any T2 failure |

Every hole found is fixed in a mechanism (or explicitly accepted with a justification) and its design becomes a
permanent regression in `corpus/adversarial/` with the expected error code or corrected result; regressions are
never removed and run in fast CI (5.2). The campaign's final report states the residual maximum of each objective
after fixes.

### 12.4 T4: cross-simulator agreement

On the subsets each tool can express: our Stream fork (`stream_compat`, 2.3), ZigZag (single-op intra-unit),
LLMCompass (GPU-like GEMM, attention and layer timing), GenZ (whole-model LLM inference, analytical). Agreement
is a trust signal, never ground truth and never a fit target; disagreements are triaged as in 2.3 (`kiln_bug`,
`oracle_bug`, `intended`).

| Metric (per oracle, over its subset) | Target |
|---|---|
| Whole-step or per-op time ratio kiln/oracle, median | within 15% |
| Design ranking, Spearman | >= 0.8 |
| Untriaged disagreements beyond 2x | 0 |

### 12.5 T5: trust report

`kiln trust report --release` and `kiln trust report --claim <design>` write `trust/<id>/report.json` (plus the
`llm` text form): T1 table (published vs predicted ratio and interval, tolerance, pass, `fit_device` flag,
coverage), T2 pass/fail per relation and design, T3 campaign summaries and regressions added, T4 agreement
statistics, and snapshots of L4 (2.4) and L5 (3.5) acceptance, calibration-set hash and kiln git hash. A claim
report adds the candidate's own T2 relations, a T3 campaign seeded from its lineage, T4 where the candidate is
inside an oracle's subset, and its `score_interval`, interval drivers and `extrapolated` keys. **Green** = every
gated row above passes. Releases need T1, T2 and T4 green and the latest T3 campaign without open holes; claims
need the full claim report green.

---

## 13. Open questions

1. **Generic DRAM efficiency transfer.** Resolved by owner ruling (08 §C): ranges, not a point choice; central =
   pooled fit, `lower`/`upper` from the device spread or plausible band (3.2), claims on the `low` corner (6.6).
2. **Launch overhead for novel designs.** Kernel-launch and sync costs are a property of the control/runtime
   design, which evolution can change. Proposed: 01 models a sequencer/control component whose overhead is priced;
   `generic` launch overhead is a floor no design can go below without paying for it. Needs agreement with 01/03.
   The value question is resolved by ranges (3.2); the priced-sequencer question stays open.
3. **Phase-level ground truth for decode.** Resolved by owner ruling (08 §C): whole-step `sequence` is the scored
   semantics for designs and baselines (2.5, 6.4); isolated ops are calibration data.
4. **TPU v5e-8 access.** Colab offers single-chip v5e/v6e; is a v5e-8 (Kaggle TPU VM or GCP spot) acceptable cost?
   Otherwise TPU multi-chip validation relies on G4 evidence only.
5. **Held-out workload secrecy.** If the same person/agent writes the loop prompts and reads held-out scores, the
   held-out set leaks over time. Rotate a fresh held-out set per claim?
6. **Suspicion threshold.** Closed by owner ruling C.4: 1.15x until the M2b H100 held-out check passes, then 1.3x. (Was: should it be lower (1.15x) until M2b
   proves transfer to H100?
7. **CI host.** No git commits exist yet; GitHub Actions (private repo) for fast CI and a RunPod CPU pod for nightly
   are assumed. Confirm budget and repo hosting.
8. **Bit-exact cross-platform determinism.** `libm` plus no FMA contraction should suffice; if the PyO3 build on
   macOS (development only) differs, is macOS excluded from golden checks?
9. **MI300X availability on RunPod** and whether a different-vendor held-out device is worth the porting cost of
   the ROCm runner in M2b vs M8.
10. **Power telemetry on TPU.** None on Colab; TPU power validation rests on published numbers (G3/G4) unless a GCP
    VM with power metrics is used.
