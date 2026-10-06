# 08: Integration decisions and open questions

Integration pass over 00-07 (2026-10-04). Part A logs every inconsistency resolved and how. Part A.2 records the fact check. Part B consolidates every
open question from every section, de-duplicated and tagged. Part C records the owner's rulings; Part D records how
they were applied to the sections.

Rule applied: the owning section's definition wins (ownership per 00's section and crate tables); the 8 binding
decisions in 00 override any section. Edits are minimal and in the section that was wrong.

## A. Decision log

| # | Sections | Conflict | Resolution | Why |
|---|---|---|---|---|
| 1 | 01 §6.1, 02 §3.2 | Two precision tables; 02's storage variants (`fp8_e4m3_pt`, `int8_pc`, `int4_g128`) and `int64`/`bool`/`e8m0` absent from 01's registry; 01 OQ14 open | Added them to 01 §6.1 as storage variants mapping to compute names via `Precision::compute()`; 02 §3.2 points to 01 | 00 decision 4 |
| 2 | 02 §15, 06 §6.5, 01 §18 | 02 used `W-*` for errors (01 uses `W-` for warnings); 06 assigns `E-WL-*` to 02 | 02 codes renamed `E-WL-<area>-NNN`; the two warnings become `W-WL-*` | 06 owns namespaces |
| 3 | 01 §12/D6, 03 | 01 cited an undefined `SIM-E` code in 03 | Defined `E-MAP-POWER-CAP` in 03 §4.5; 01 references it | 03 owns kiln-sim errors (06 namespace `E-MAP`) |
| 4 | 01 §18.3/D20, 06 §6.6 | 01 mapped `E-IR-UNPRICED` to E-IR-1101 (overrides) only; 06 also rejects unpriceable fields | `E-IR-UNPRICED` = E-IR-1101 + E-IR-1104, with 00's "never rejects cost-neutral or cost-reducing changes" stated in 01 | 00 decision 3 |
| 5 | 01 §18.2, 04 §13 | Same physical checks had two codes (E-IR-0803/0804/0805/0907 vs `E-PHYS-*`); power density was a warning in 01, an error in 04 and 06 S1 | kiln-phys emits `E-PHYS-*`; 01 numbers are aliases; E-IR-0907 made an error | 04 performs the check; 06 envelope stage treats it as an error |
| 6 | 04 §7.1-7.2, 01 §10.3 | 04 relied on a per-link `sizing` override absent from 01 | Added `OnDie.sizing: Option<WireSizing>` to 01 | 01 owns IR fields; 04's need was listed in 04 §13 and claimed aligned in 01 D19 |
| 7 | 04 §4.2, §4.7, §5, §6.1, §7.2, §11, §13 | 04 used field names 01 does not define: `dot_len`, `macs_per_cycle_per_lane`, `area_mm2`/`e_op_J`, `spare`, cap level `chip`, outline `w_mm/h_mm`, layouts `row/auto`, placements `left_of/abut`, link `protocol` | Rewritten to 01's names: `Geometry`, `PrecisionMode.rate`, `PowerOverride.area/energy_per_op`, `disabled`, `CapLevel::die`, `Outline::Fixed{w,h}`, `Layout`, `Placement`, `LinkPhys` | 01 owns IR types |
| 8 | 04 §14 | Stated that 01 caps power "per op" | Now cites per-phase (Tier A) / per-window (Tier B) | 00 decision 1 (01 already followed it) |
| 9 | 03 §9, 04 §7.6, 06 §3.2-3.3 | `C_eff` listed as a fitted parameter (unit F); "V/f curve points" fitted | Replaced by 04 §12.1's `kappa_E_*`, `kappa_clk`, `p_ll`, `p_ls` and V/f shape `V_th`, `alpha` | 00 decision 6; 04 owns power parameters |
| 10 | 03 §3.2 step 2, §7, §11; 02 §16 | 03 said kiln-map shards ops and inserts collectives | 03 instantiates 02's `PartitionedProgram`, picks algorithms, applies only 02 §9.1 volume-preserving rewrites | 00 decision 5 |
| 11 | 03 §6, §11 | 03 assumed an 01 `NmpUnit` type with granularities `Bank/BankGroup/PseudoChannel/LogicDie/Subarray` | Rewritten to 01's `ComputeUnit.near: NearBinding` granularities, `logic_die`, and `kind: cim`; non-IR NMP timing from stack timing / 04 defaults | 01 owns IR types |
| 12 | 03 §2.2, §2.7 | `ComputeUnitTemplate` "from 01" (not in 01); `DType` (not a registry type) | Template is built by kiln-cost from `HwModel.UnitInst`; dtypes are 01 `Precision` | 01 owns HwModel and the registry |
| 13 | 03 §10, 05 §3.8/§6.5/§7.2, 06 §6.1/§6.5 | Explanation text produced in 3 places (`kiln_trace::analysis::explain`, `Bottleneck.summary`, `Result.explain`) | Single `kiln_sim::explain_run` defined in 03 §10; 05 displays persisted text; 06 wraps it | 00 decision 7 |
| 14 | 05 §3.3, 06 §6.2/§7, 03 §10 | 06 exposed `none|summary|full` (no `ops`), default `none`; 03 said traces are Tier B only | 06 exposes `none|summary|ops|full`, default `summary`; 03 `SimResult.trace` allows Tier A at `ops` | 00 decision 8 |
| 15 | 05 §3.9, 06 §6.9 | 06 added archive columns as its own amendment | Columns listed in 05's schema; 06 references 05 | 00 decision 7 (05 owns archive) |
| 16 | 05 §3.9, 06 §3.6 | Calibration-report columns differed (`op_key` vs `bench_key`, `measured_std_s` vs `measured_cv`, split `heldout` vs `test`) | 06 writes exactly 05's columns; 05's split enum uses 06's names (`train`, `test`) | 05 owns the table; 06 owns split terminology |
| 17 | 03 §4.8, §5.6, 06 §8 | 03 targets covered 8 chips at 50 ms / 10 s; 06 sets 100 ms / 30 s for 8 chips | 03 states single-chip targets and defers 8-chip numbers to 06 | 06 owns performance targets |
| 18 | 03 §4.6, 06 §2.4 | `eps_AB` = 10% "on the corpus" vs 06's median 5% / p95 10% / max 30% | 03 cites 06's statistics | 06 owns the numbers (03 said so) |
| 19 | 02 §12.6, 06 §2.5, §4.3 | 02: `graph_cold` = "no launch gaps", `launch=false`; 06: `graph_cold` includes in-graph launch and gap. Also 64 copies vs 512 MiB rotation, 64M-element flush vs 64 MiB | 02 defines `launch` as the host term only, in-queue terms always apply; bench methods deferred to 06 §4.3 | 06 owns measurement semantics |
| 20 | 06 §4.2, §11 | 06 said 02 defines `BenchOp` and residency semantics; 02 defines neither | `BenchOp` defined in 06; `cold_dram/l2_warm/resident` mapped to 02 `eval_mode` | 06 owns the bench harness; 02 owns eval modes |
| 21 | 06 §2.2, 02 §11.4 | Random workloads in `kiln-ir::arb_workload` vs `kiln-wl::arb` | `kiln-wl::arb` | 02 owns `kiln-wl` |
| 22 | 06 §6.8, 01 §15 | `E-IR-TOO-LARGE` undefined | E-IR-0210 (expansion budget) | 01 owns `E-IR-*` |
| 23 | 06 §7, 01, 04, 05 §8 | CLI: `--stages S1..S5` vs fit stages P/L/D/U; `kiln trace export --format` vs 05's flags; `import-legacy` vs `import --legacy`; 01's `fmt/expand/hash/migrate`, 04's `export-def/import-def`, 05's `trace *`/`viz render` missing; `kiln validate --json` vs global `--format json` | 06 table fixed and extended, flags owned by the defining section; 01 uses `--format json` | 06 owns the CLI table; flags belong to the feature owner |
| 24 | 01 §18.3, 06 §6.2 | 01: evaluate takes a validation profile; 06 options had none | Added `profile` option, default `search` | Needed for `E-IR-UNPRICED` enforcement in evolution |
| 25 | 06 §6.1, 02 §16 | Workload name `llama3_8b:decode` not defined in 02 | `llama3_8b:decode_b8` | 02 §11.4 owns names |
| 26 | 04 §12.2, 06 §4.6, 07 §2 | A100-40GB memory called HBM2e | HBM2 (A100 datasheet: 40GB HBM2, 1,555 GB/s) | Fact check |
| 27 | 01 §20.2, 06 §1 | v5e HBM 819 GB/s cited to the v5e page, which says "800 GiBps" | 819 GB/s kept, marked assumed, claim cited to the JAX scaling book (8.2e11 B/s) | Fact check (A.2) |
| 28 | 05 §3.4, 02 | `ops.phase` enum (forward/backward/optimizer), `ops.kind` (matmul), path example not 02's | 02's `PhaseKind`, `Op` tag, dotted node path | 02 owns op taxonomy |
| 29 | 05 §11, 06 §10 | 05 phases V0-V4 had no mapping to 06 milestones | Mapping column in 05 (V0: M0 and M4, V1: M3, V2-V4: M5); 06 M5 row cites it | 06 owns the plan |
| 30 | 02 §10, 03 §5.6 | "phase 2" undefined | Marked as not scheduled in 06 §10 (open question B.1) | No milestone exists |
| 31 | 01 §3, 03 §9, 06 §3.1/§9.2 | Platform key `a100` vs `a100_40gb`; reference examples lacked `family` | 06's keys used; `family` added to 01's A100, v5e, v6e examples | 06 owns platform keys |
| 32 | 06 §3.1 | `exec_model` keys in CamelCase vs 01's serialized snake_case | snake_case | 01 owns the serialized form |
| 33 | 01 §17 | Canonical serialization not stated as JCS | Stated as RFC 8785 JCS | 00 decision 2 |
| 34 | 02 §11.5 | Hash prefix `wl1:` vs 01 `hw1-` | `wl1-` | Consistency |
| 35 | 03 §3.4, §11; 02 §16; 04 §13 | 02 needs a KV-capacity query; 04 needs `TrafficMatrix` and `ActivityReport` from 03; 03 provided neither | Added `kv_capacity` (03 §3.4) and the two outputs (03 §11) | Required interfaces of the consumers |
| 36 | 03 §1, 07 §2, 02 §16, 06 §11, 04 §14, 05 §3.4 | Stale or loose text: data flow omitted `kiln-wl`; 07 DRAM plan contradicted 03 §5.3; stale notes on 03 §7 and on 02/04 "not yet written"; "`width_bits` is the only `_bits` field"; `nmp_unit` kind undefined | Corrected in place | Owners' decisions already made |
| 37 | Open questions | Resolved by 00 or another section but still open: 01 OQ14, 02 OQ1/OQ10, 03 OQ6, 05 OQ1/OQ4/OQ10/OQ11 | Marked resolved in place | 00 decisions 4, 5, 7, 8; 00 crate table; 02 §9.2/§10.2 |

Checked and consistent (no edit): Tier A <= 50 ms and Tier B <= 10 s single-chip (00, 03, 06); `T_A2 <= T_B` (03 §4.6,
I9, 06 P9); A100 numbers (01, 03, 04, 06); v6e 918 TFLOP/s, 1638 GB/s, 800 GB/s ICI; 01 §12 already per-phase;
06 §6.6 `E-IR-UNPRICED` row; result provenance fields (00, 03, 05, 06); temperature units (01/04 degC fields, 05 K
columns, both named by suffix).

### A.2 Published facts checked

| Fact | Spec value | Source (accessed 2026-10-04) | Status |
|---|---|---|---|
| A100-SXM4-40GB memory BW | 1555 GB/s, HBM2 | NVIDIA A100 datasheet: "40GB HBM2", "1,555GB/s" | Confirmed; HBM2e references fixed |
| A100 bf16 dense | 312 TFLOP/s | same datasheet | Confirmed |
| TPU v5e bf16 / int8 | 197 TFLOP/s / 393 TOP/s | docs.cloud.google.com/tpu/docs/v5e | Confirmed |
| TPU v5e HBM BW | 819 GB/s | v5e page: "800 GiBps" (859 GB/s); JAX scaling book: 8.2e11 B/s | Conflicting sources; 819 kept, marked assumed |
| TPU v5e MXUs | 4 per TensorCore, 128x128 | v5e page; system-architecture page | Confirmed |
| TPU v6e bf16 / int8 | 918 TFLOP/s / 1836 TOP/s | docs.cloud.google.com/tpu/docs/v6e | Confirmed |
| TPU v6e HBM | 32 GB, 1638 GB/s | v6e page | Confirmed |
| TPU v6e MXU size | 256x256 | system-architecture page | Confirmed |
| TPU v6e MXUs per TC | page says 2 | v6e page | 2 x 256^2 needs 3.5 GHz for 918 TFLOP/s; 4 at 1.75 GHz kept, assumed |
| TPU v6e clock | 1.75 GHz | not published (v6e page, architecture page, scaling book) | Assumed |
| TPU v5e clock | 1.5 GHz | not published; derived from 197 TFLOP/s | Derived |

## B. Open questions

### B.1 [user decision], ordered by impact (recommended default in bold)

1. **Hardware access budget** (06 M2b, M6, OQ4, OQ9, OQ10; 04 OQ4). RunPod sessions are needed for H100 held-out,
   root-only locked-clock and power-cap runs (separate `V_th/alpha` from `kappa_E`), and an 8-GPU node for
   collectives; TPU v5e-8, GCP power telemetry and MI300X are optional. **Fund H100 (~2 GPU-hr) plus one 8x H100 node
   session; skip MI300X and v5e-8 until M6 results need them.**
2. *Closed by ruling C.2.* **Scored phase semantics** (06 OQ3, 02 OQ3). Score the baseline on `sum` of isolated `graph_cold` ops or on
   `sequence` (whole-layer capture)? **`graph_cold` per-op for calibration; `sequence` for scoring once M2 lands.**
3. *Closed by ruling C.3 (except the priced-sequencer part of 06 OQ2).* **Calibration conservatism for novel designs** (06 OQ1, OQ2; 01 OQ10). Generic DRAM efficiency (mean vs minimum
   over devices), launch/sync floor for designs without a priced sequencer, implicit unlimited DMA. **Minimum of
   calibrated devices; generic launch overhead as a floor unless a priced sequencer is modelled; synthesize one DMA
   per memory controller by default.**
4. **Claim gating** (06 OQ6, OQ5). Suspicion threshold 1.3x vs 1.15x until H100 transfer is proven; rotate held-out
   sets per claim. **1.15x until M2b passes, then 1.3x; rotate a fresh held-out set per claim.**
5. *Training items closed by ruling C.5 (out of scope).* **v0 workload scope** (02 OQ6, OQ7, §10; 03 §5.6). Training has no milestone; speculative decoding, prefix
   caching across requests, ONNX/Chakra import and the parallel DES are unscheduled. **Inference only through M6;
   add a training milestone after M6; speculative decoding and prefix caching deferred; ONNX/Chakra after M6.**
6. **v0 hardware scope** (01 OQ7, OQ12, OQ13, OQ9; 04 OQ7, 05 OQ5). Fused composite op classes (attention
   engines), analog CIM, host compute for offload, per-die thermal coupling, transient thermal. **All out of v0:
   decompose composites, reject analog CIM, minimal host, package-level cap, steady-state thermal per phase.**
7. **Evolution eligibility and reporting** (02 OQ12, OQ4, OQ5, OQ9). Opaque nodes, MFU headline, MoE routing
   default, uneven sharding. **Opaque nodes make a workload ineligible for scoring; kiln mask-aware MFU headline with
   PaLM beside it; mild Zipf routing default; SPMD-only with `allow_uneven` padding.**
8. **CI hosting and budget** (06 OQ7). **Private GitHub repo + Actions for fast CI, RunPod CPU pod nightly.**
9. **Images in LLM prompts** (05 OQ2). **Text only until a V1 ablation shows images help.**

### B.2 [resolve during implementation]

| Question | Refs |
|---|---|
| TPU v6e MXU count and clock; v5e HBM bandwidth (819 vs 859 GB/s); TPU vreg count, VMEM ports, SMEM size | 01 OQ1, OQ2, OQ11 |
| v5e 2x2 slice torus wiring (wraparound, doubled link) | 01 OQ3 |
| A100 L2 internals (inter-partition link, crossbar port width, slices, harvested TPCs) | 01 OQ4 |
| Compact vs expanded design hash for evolution dedup | 01 OQ5 |
| Whether `unit_eff` and an explicit GPU issue model are needed after structural terms | 01 OQ6, 03 OQ4 |
| Effective capacity rule for coherent duplicated L2 lines | 01 OQ8 |
| Tier A tolerance gating on Kendall tau vs absolute error | 03 OQ1 |
| Adaptive vs fixed Tier B chunk size | 03 OQ2 |
| Contention correction form (M/D/1 vs fitted) | 03 OQ3 |
| NMP layout and command semantics needed in 01 | 03 OQ5 |
| Learned-mapper observation space (PyO3 surface) | 03 OQ7 |
| Cross-platform bit identity (no FMA, `libm`) and whether macOS is excluded from goldens | 03 OQ8, 06 OQ8 |
| Tier A chip-symmetry certification check | 03 OQ9 |
| Mapping dependency footprint for 06's fragment cache; floor-computation API; identical-mapping A/B mode | 06 §6.7, §11 |
| Per-op runner-up terms, lane assignment and limiter table emission from 03 for 05 | 05 deps (03 row) |
| HBM PHY shoreline/area from die shots; SerDes numbers from ISSCC papers | 04 OQ1, OQ9 |
| TPU v4 192 W includes HBM or not | 04 OQ2 |
| Activity factors from tensor statistics | 04 OQ3 |
| Second cross-node calibration point | 04 OQ5 |
| Placer-chosen 3D tiers (v1) | 04 OQ6 |
| `ctrl_ge` reference core for programmable units | 04 OQ8 |
| Power-delivery (bump current) constraint | 04 OQ10 |
| ASAP7 audit to N7 mapping policy | 04 OQ11 |
| Decode sampling near capacity cliffs | 02 OQ2 |
| Exact per-document counting for packed training: closed by ruling C.5 (training out of scope) | 02 OQ11 |
| ONNX opset 23 / `com.microsoft` op verification | 02 OQ8 |
| Trace tick below 1 ps | 05 OQ3 |
| Perfetto track cap; localhost `?url=` behaviour in v58 | 05 OQ6, OQ7 |
| wasm64 for large web traces | 05 OQ8 |
| Heuristic block matching for structural diffs | 05 OQ9 |
| ATLAS/VOXEL/DeepStack repo status, LUMINA simulator, GenZ/DeepStack/SCALE-Sim licenses | 07 §5 |

## C. Owner rulings (2026-10-05)

| B.1 item | Ruling |
|---|---|
| 1 Hardware access | Deferred; revisit later. |
| 2 Score timing basis | Whole steps. Calibrate on isolated ops and microbenchmarks; score designs (and the baseline) on whole-step timing (full decode step, full prefill layer). Add a measured whole-step A100 anchor (CUDA-graph Llama-3-8B decode step). |
| 3 Conservatism for novel designs | Conservative enough that results are accepted, not so conservative they are absurd. Uncertain parameters carry ranges (from the spread across measured devices and physically plausible efficiency bands), and results are reported as intervals (low / central / high), not single numbers. |
| 4 Claim gating | Accepted (1.15x suspicion threshold until H100 held-out passes; fresh held-out workloads per claim), plus a stronger simulated test suite so results can be trusted without relying on hardware access alone. |
| 5 Workload scope | Inference only. Training is out of scope (not deferred): remove from v0 milestones; keep the IR extensible but do not specify or implement training paths. |
| 6 Hardware scope v0 | Accepted as recommended. |
| 7 Evolution rules | Accepted: opaque-node workloads simulated but unscored; masked (causal-aware) MFU as headline with dense MFU always reported alongside; mild Zipf MoE routing default, configurable; even (SPMD) sharding only in v0. |
| 8 CI hosting | Dropped for now. |
| 9 Images in prompts | Text only until an ablation shows images help. |

## D. Rulings applied (2026-10-05)

| Ruling | Sections edited | Change |
|---|---|---|
| C.5 Training out of scope | 00 non-goals; 02 intro, D7, §1, §3.4, §3.6, §4.1, §5.1-5.3, §7.1, §7.4-7.5, §8.2-8.4, §9.1-9.2, §9.5, §9.7, §10, §11.1, §12.4, §13.1, OQ11; 03 §7, OQ6; 05 §3.4, §6.4; 06 §2.5, §3; README | Removed autodiff/VJP, optimizer, recompute, ZeRO/FSDP, 1F1B, gradient/optimizer-state classes, `cross_entropy`, `train_step` and the train scenario; §10 now states training is out of scope and the IR is extensible to it; dense MFU restated for inference (2N + 4LHQT per token) |
| C.5 naming | 02 §11.4, §16; 05 §3.9, §6.8; 06 §2.4, §3, §6.2, §6.6, §11 | Evolution score set `train` renamed `evolve`; calibration splits renamed `fit` / `test`; "training device/records" renamed "fit device/records" |
| C.2 Whole-step scoring | 02 D9, §8.3, §11.4, §12.5 (new, replaces training metrics), §12.6, §13.1, OQ3; 03 §4.9 (new), §10; 06 §2.4, §2.5, §3.4, §3.5, §4.2, §4.3, §6.3, §6.4, §6.6, §7, §8, §9.2, §10 (M1, M2), OQ3 | Scores come from whole steps (`scope: step`, `scope: layer` only where the resident set exceeds the baseline's memory); engine window-of-3 steady-state extrapolation with boundary effects; isolated ops are calibration only; `sequence` bench mode captures full Llama-3-8B decode and prefill steps (CUDA graph, jitted TPU step); M2 measures the A100 whole-step anchor on Colab; design-step performance targets |
| C.3 Interval results | 03 §9 principle 8, §9.1 (new), §10; 04 §3, §12.5, §13; 05 §3.4, §3.5, §3.9, §6.4, §6.6, §6.7; 06 §2.4, §3.1, §3.2, §3.5, §3.6, §6.2-6.6, §6.9, OQ1, OQ2 | Parameters carry `lower/central/upper` per key (device spread, CI, plausible band, 04 confidence); deterministic corner evaluation at fixed mapping with stated monotonicity, sampled fallback and `corner_flips`; cost <= 3x (`corners`) or about 1.1x (`sensitivity`); results and score are intervals; fitness defaults to central with `score_rel_width` reported (configurable to `low`); claim needs candidate `low` >= baseline measured/central at matched envelope; viz error bars and bands |
| C.4 Trust suite | 06 §1 intro, §2, §5.1, §5.2, §6.3, §6.6, §7, §10 (M1, M3, M4, M6), §11, §12 (new; open questions now §13); README | T1 chip-history backtest (V100, A100, H100, TPU v4/v5e/v6e from published specs, whole-step ratios and rankings vs MLPerf/vendor, G3/G4, no fitting), T2 metamorphic relations M1-M8, T3 red-team objectives with permanent regressions, T4 Stream/ZigZag/LLMCompass/GenZ agreement as signal, T5 per-release and per-claim trust report; claims and releases gated on green |
| Structure | README | Glossary entries for whole step, interval result, trust suite; section scopes updated |

## E. M0 implementation notes

Spec issues the three M0 agents (hw, wl, trace) and the integration pass hit, one line each with the choice made.
Only 02 §11.5 was edited in place (hash rendering); everything else is recorded here for the owning section's next
revision.

### E.1 Hardware IR (kiln-ir `hw`)

- A100 L2 instance naming: the two L2 partitions are instances `l2p[0]` / `l2p[1]` of one `l2p` template, selected as `l2p[i].slice*`.
- L2 bandwidth: 01 §7's port formula sums all ports (read + write, 128 B/clk/slice); 01 §20.1's 7.22 TB/s counts reads only. kiln derives the all-ports figure; the test checks the read-only figure separately.
- A100 int8 peak is 623.7 TFLOPS (2 x 311.87), not 01 §20.1's rounded 623.8; the derived value stands.
- `meta.claims` added to the TPU v5e 2x2 slice and `ember` so they pass `reference`; a missing claim is E-IR-1103.
- `tsmc_n12` added to the known technology nodes (used by `ember`).
- `#[serde(flatten)]` + `deny_unknown_fields` do not compose in serde; a custom deserializer keeps unknown-field rejection.
- Fields added to 01's types: `Die.placement`, `shared_rf`, `level_hint`, `source` on overrides, a minimal `GatewaySpec` / `RouteEntry`, and `PowerCap.level` made optional.
- Codes: E-IR-0100 is a JSON5 syntax error; W-IR-0799 marks a topology that M0 does not expand into channels.
- Template merge is shallow.
- Patch selectors that address only some instances of a replicated entity are rejected (E-IR-0212).
- W-IR-0725 (chip without host link) and W-IR-0906 (clock-domain crossing without `crossing_latency`) narrowed in scope.
- CIM throughput and capacity are derived (01 §9.3): MACs/cycle = `parallel_rows * floor(cols / ceil(w_bits / cell_bits)) / ceil(in_bits / input_bits_per_cycle)`, capacity = `rows * cols * cell_bits * weight_sets / 8`; new fields `cell_bits`, `input_bits_per_cycle`, `parallel_rows`, `weight_sets` (priced in 04 §4.11), `@rate` > 1 on CIM is E-IR-0608, E-IR-0606 now checks the derived capacity; ember's int8*int4 peak 107,374 -> 3,355.4 TOPS (old rule counted rows*cols MACs, 32x).
- `search` checks every performance override against the derived value stored in the expanded model (`MemInst.bandwidth_derived`, `Channel.bandwidth_derived`): throughput fields allowed at <= derived (de-rates are cost-neutral, 00 decision 3), latency/energy/area/power fields at >= derived, fields kiln-ir cannot derive yet rejected as unverifiable (E-IR-1101); it now also scans CIM `weight_write`, PHY constants, switch, DMA and clock-crossing fields. Imported harness designs pass (their overrides equal the derived values).

### E.2 Workload IR (kiln-ir `wl`, kiln-wl)

- Tensor id `hL` in 02 §14.1 is not a valid id (`[a-z0-9_.-]`); implemented as `hl`.
- Harness compatibility (02 §13.1) needs bf16 logits (`harness_compat` model option); the standard model keeps fp32 logits, so its `lm_head` descriptor does not join the bf16 harness record.
- `layer_norm` is counted as 4 adds + 3 muls per element.
- New codes: E-WL-REF-001 (undeclared tensor or graph reference), E-WL-ID-001 (id error), E-WL-OP-001 (unsupported op or plan feature), E-WL-ZOO-001 (invalid zoo preset or overrides).
- Kernels are emitted bound (concrete extents), not symbolic.
- `Node` unknown fields are not denied (serde `flatten` limitation); other workload types deny them.
- Open for 03: 02 §4.2's op-class split may double count fused elementwise work; 03 decides.
- Gather bytes are counted as an upper bound.
- Hash truncation is 32 hex everywhere (`hw1-`, `wl1-`, `bop1-`, `meas1-`, `res1-`); 02 §11.5 text corrected.

### E.3 Results, traces, measurements (kiln-trace, kiln-cli)

- Unimplemented commands exit 69 with `E-NOT-IMPLEMENTED` (sysexits EX_UNAVAILABLE; not in 06 §7's table).
- `overheads_included` and `operands` live on session-level `timing_modes`, each with a `launch` path field, not per record.
- Residency is excluded from the BenchOp key (a record holds several timing modes).
- TPU sessions get a derived canonical mode `best` (minimum over the measured modes).
- Interval corners are ordered numerically (`low <= central <= high`); the low-parameter corner gives the high time.
- `SimResult` gained `phase` and the `kiln.sim/1` schema tag.
- 03's binding tie-break order lacked `Nmp`, `Contention`, `PipelineBubble`; kiln-trace adds them.
- `deterministic_hash` excludes wall-clock timing fields.
- Trace and result errors use the `E-TRACE-*` namespace.
- Sums the spec calls exact (attribution, energy) are checked to 1e-9 relative tolerance.
- 06 §1 counts corrected: 33 GEMM sweep shapes; 27 unique LLM keys.
- The CUDA runner's L2 flush buffer is 256 MiB, not 64 MiB.

### E.4 Integration decisions

- One BenchOp descriptor, `kiln_ir::bench::BenchOp`, used by bench export (kiln-wl) and measurement import (kiln-trace): `kind` (gemm / linear / bmm / sdpa / elementwise / softmax / rmsnorm / collective / copy / read_reduce / scale / other), `op` (IR op name, only where the kind does not determine the computation), `dims` (BTreeMap), per-operand `dtype` / `layout` / `shape` (BTreeMap), `mask`, `epilogue` (sorted and deduplicated in the key). `key = "bop1-" + hex(sha256(JCS))[..32]`.
- Not in the key: residency (a timing-mode property; `kiln_ir::bench::Residency`, carried by kiln-wl's `Occurrence` and kiln-trace's `TimingMode`) and occurrence data (workload, source, role, weight, count, FLOPs, min bytes; kiln-wl `Occurrence`).
- Contraction layouts use index letters: gemm `a: mk, b: kn, out: mn`; linear `b: nk`; bmm `bmk, bkn, bmn`. Other ops name operands by position (`in0..`, `out0..`) with `row_major` and explicit shapes, so descriptors do not depend on tensor ids.
- dtypes use registry names (`bf16`, `fp32`, `uint8`); kiln-trace's earlier `f32` / `u8` spellings are gone.
- `legacy_key()` is the op-list key (`gemm_M_N_K` for both weight layouts, `bmm_B_M_N_K`); `legacy_name()` is the record that measures exactly the descriptor (`gemm_M_N_K_linear` for `[n,k]` weights).
- Unfused attention BMMs are exported as dense contractions with no `mask`; `mask` is for fused (`sdpa`) kernels. Batched contractions are canonicalized to `bkn` (the harness's `torch.bmm` layout) even when the model stores K as `[n,k]`.
- `kiln bench export --suite legacy` is exactly what `gpu_bench.py` runs: the 30 op-list keys plus the `_linear` twin of every weight GEMM, 52 descriptors, joining the 52 A100 `llm_ops` records one-to-one. Manifest schema `kiln.bench/1`: unique descriptors in first-occurrence order, each with its `uses`.
- JCS number rule moved from `hw::canon` into `common::canonical_json` (integral floats as integers, `-0.0` as `0`), so every hash follows it. Changed: `meas1-` of the A100 session (integral clock and power floats), `wl1-` model hashes (e.g. llama3_8b `wl1-9278...` -> `wl1-a904...`, from integral floats such as rope theta), and `result_min.json`'s canonical bytes. `hw1-` unchanged (the rule already applied).
- `Diagnostic` gained optional `related: Vec<String>` and `span: Option<Box<Span>>` (`file`, 1-based `line` / `col`, optional byte `start` / `end`), omitted when empty (01 §18.1). `clippy.toml` sets `large-error-threshold = 160` for the larger error type.
- New CLI-level codes: E-CLI-0001 (`validate --kind auto` cannot classify the document) and E-WL-DOC-001 (file is not a workload document). `validate --kind calibration` and `bench export --sequence/--scope` exit 69 until M2.

## F. Owner rulings (2026-10-05, continued)

| Topic | Ruling |
|---|---|
| TPU v6e MXU structure | Follow Google's v6e page verbatim ("Each TensorCore has 2 matrix-multiply units (MXU)", docs.cloud.google.com/tpu/docs/v6e, checked 2026-10-05): model 2 x 256x256 MXUs at 3.5 GHz (derived from 918 TFLOPS bf16). VMEM read bandwidth held at the prior 45.6 TB/s assumption (3 ports at the 2x clock). Supersedes the 4 x 1.75 GHz assumption in 01 §20.3/§22 q1 and A.2. |
| Scoring and claims basis (M2 finding) | Simulated candidate vs simulated baseline under the same execution model; measured data validates the simulator and is never the baseline in a score or claim (supersedes "baseline measured/central" in the 06 claim rule). |
| Kernel decomposition | Model the real per-op kernel decomposition and launch costs of the baseline software stack (e.g. PyTorch eager/graph splits RMSNorm, RoPE, residual, KV append into several kernels) so absolute A100 whole steps approach measured values. |
| Cross-chip intervals | For devices not in the fit set, parameter ranges span the spread observed across measured devices (e.g. DRAM efficiency about 0.68-0.95), so held-out intervals cover reality. |
| Cross-chip intervals: implementation | `range_policy` in kiln.calib/1 generic sets (06 §3.2): ranges on non-fit designs span measured evidence across A100, v5e, v6e (`eta_res` 0.667-0.952); fit devices keep fitted ranges; generic-v1/v2 bumped to v2 without refitting point estimates. |
| Off-chip memory in the matched envelope | HBM is bought, not designed: the matched envelope pins off-chip bandwidth and capacity to the baseline's (`offchip_bw`, `offchip_bytes`; `E-ENV-0007/0008`, from the design summary, enforced without kiln-phys), and stack count/shoreline once `physical` exists. PIM internal stack bandwidth is not off-chip. Closes evolution winning by adding stacks or pin rate (06 §6.4). |
| t_gap / t_min_kernel lower bounds | Lowered from 0.5 us to 0.1 us: the measured A100 empty-kernel CUDA-graph chain (0.90 us total) makes 0.5 us each physically impossible. |

### F.1 Kernel decomposition (implemented 2026-10-05)

- Mechanism: software-stack recipes (02 §7.4.1, 03 §4.4), data files in `kiln/stacks/`. The engine charges
  each unfused kernel its traffic and the launch terms already fitted on calib-micro (`t_min_kernel`, `t_gap`,
  `eta_res`, `t_dram_ramp`); nothing is fitted to whole steps or the LLM suite.
- `pytorch_cuda_graph_sdpa` is derived from `calibration/gpu_bench.py`'s op sequence under PyTorch semantics and
  checked against the A100 profiler kernel counts (SDPA variant: 45 kernels per layer at decode b1/b8, 43 at
  decode b32 and prefill; kiln's graph issues 12). RMSNorm is 8 kernels (fp32 round trip), RoPE 7 per tensor,
  KV append 2, SiLU*mul 2, gate/up 2 GEMMs, flash split-KV + combine up to 8 sequences, prompt flash + memset,
  cuBLAS split-K reduce on `attn.o` (<= 8 rows) and `mlp.down` (<= 32 rows). kiln's `logits_select` stays one
  kernel (a view in PyTorch): +1 kernel per step.
- `xla_tpu_fused` has no rules (one fusion per node, no extra traffic): TPU results are unchanged.
- Default per execution model (`host_launched` -> PyTorch, `static_dataflow` -> XLA, `device_queued` ->
  `kiln_ideal`); scores compare candidate and baseline under the same execution model and hence the same
  recipe; the recipe used is in every result's provenance.
- Groups with no tasks (layout views such as the qkv split) no longer pay a launch.
- Effect (platform:a100_40gb, central, whole step vs measured CUDA-graph SDPA step): decode b1 -18.0% -> -5.6%,
  b8 -20.3% -> -9.4%, b32 -25.9% -> -18.8%, prefill b1 -15.4% -> -4.1%; TPU v5e/v6e and all per-op suite
  statistics unchanged. The b32 remainder is matmul-side, not decomposition: in the step, the qkv GEMM runs
  ~100 us (cuBLAS 256x64 tile) against 42 us isolated, and flash_fwd 249 us against kiln's 200 us.
| Software stack in scoring | Evolution fitness compares candidate and baseline under the SAME stack, `kiln_ideal` (hardware-only comparison). Results also report realistic-stack scores (each design under its own exec model's real stack: PyTorch for host-launched GPUs, XLA for TPUs). Claims must win under both. |

### F.2 Software stack in scoring (implemented 2026-10-05)

- `kiln.options/1` field `stack` (CLI `--stack <id|path|own>`), default `kiln_ideal` on both sides; it is in the
  baseline basis key and the metric cache key by recipe hash. `own` runs each side under its execution model's
  default (F.1).
- `fitness::apply` refuses (`E-CAL-ASYM`) a score whose sides' provenance `stack` labels differ, unless the
  options declare `own`; `score_components` records both labels.
- Every scoring result carries `score_realistic` (`{score, interval, candidate_stack, baseline_stack}`, 06 §6.3,
  §6.4); the fitness stays the `kiln_ideal` score.
- `fitness::claim_interval_rule` requires `options.stack = kiln_ideal`, equal stacks on both sides,
  `score_interval.low >= 1.0` and `score_realistic.interval.low >= 1.0` (06 §6.6).
- F.1's "same execution model hence same recipe" applied to the old per-model default; with this ruling the
  default `score` no longer charges framework kernels to either side.

### F.3 TPU prefill below kiln's optimistic corner (investigated 2026-10-05)

Measured best-software Llama-3-8B prefill b1 (splash attention) was below kiln's optimistic corner on both TPUs
(generic-v1 central +8.2% v6e, +10.5% v5e). Root causes, all structural (nothing fitted):
- Every node was its own barrier segment, so XLA-fused elementwise work (SiLU*mul, norms, RoPE, residuals, KV
  append) serialized after the GEMMs: about 455 us per v5e layer. Fix: `fuse_elementwise` on `xla_tpu_fused`
  (02 §7.4.1, 03 §3.6). Measured evidence: v5e layer 5.24 ms against 4.97 ms of isolated GEMMs alone.
- v6e's vector set mixed the TensorCore VPU with 32 SparseCore tiles under equal round-robin slices, so every
  vector op ran at a tile's pace (attention softmax bound on SparseCore scratchpads). Fix: balanced unit sets.
- `o_proj` contracts over (heads, 128): one contracting dim per array axis left v6e's 256-row MXUs half used.
  Fix: coalesced contracting dims in kiln-cost's spatial search.
Effect (generic-v1, central [optimistic, pessimistic], ms): v5e prefill 94.24 [89.64, 108.04] -> 86.92
[82.59, 100.16] (measured 85.25); v6e prefill 54.48 [51.41, 62.78] -> 40.90 [38.09, 47.98] (measured 50.37).
A100 bit-identical; per-op suite statistics unchanged. The v6e step is now below measurement because v6e
GEMMs run below kiln's MXU model (per-op prefill GEMMs measured/predicted 0.66-0.87, best large GEMM 0.84 of
peak in calib-micro against central `unit_eff` 0.95), which the overcharges had hidden; v6e decode is likewise
0.76-0.79 of measured. Open: kiln_ideal does not fuse elementwise nodes (XLA now beats it on TPUs); v5e attention
is VPU-bound on emulated exp (the EUP special unit is unused by the mapper, 01 §5.4).
| kiln_ideal fusion | `kiln_ideal` sets `fuse_elementwise: true`: an ideal stack must be at least as good as the best real stack. Changes fitness for every design (intended). |

### F.4 TPU v6e 20% too fast at decode and prefill (investigated 2026-10-05)

kiln (generic-v1 central) predicts v6e decode b1/b8/b32 at 0.785/0.785/0.770 and prefill at 0.812 of measured.
The gap is per-op, not a whole-step effect; v6e is a held-out device, and both relevant efficiencies are priors far
above what a v6e program achieves (data: `calibration/measurements/tpuv6e_2026-10-05_diag{1,2,3_suite,4_state}`,
scripts `calibration/diag_v6e*.py`, Colab v6e-1, jax 0.7.2 / libtpu 0.0.21.1):
- **HBM inside one program is ~1090 GB/s (0.665 of 1638) at every size** from 32 MiB to 2 GiB (read sweep flat
  1060-1110 GB/s; decode GEMVs 4096x3072 ... 4096x57344 all 1070-1097 GB/s; m = 1/8/32 equal; 2 GiB footprint
  equal), matching the whole step (layer weights 436 MB / 0.400 ms = 1090 GB/s; a chain of one layer's GEMVs in one
  program 396 us; 16 layers 1109 GB/s) on both jax 0.7.2 and 0.11.2. kiln's `eta_res` for `hbm3` is the
  extrapolated prior 0.91 (v6e is held out of generic-v1/v2), i.e. 1490 GB/s. What-if `eta_res` 0.665: decode
  14.53/16.25/22.17 ms (+6.1/+6.2/+4.5%). 0.665 is below the registry bound 0.85, so a v6e fit would stop on the
  bound alarm (06 §3.3 item 4): the structure that limits one v6e program to 2/3 of the HBM spec is not modelled.
  v5e reaches 0.92 in the same harness.
- **The "v6e GEMV 0.87" figure is a timing-mode artifact.** Canonical `best = min(loop, pipelined_rot)` (E.3); on
  v6e `pipelined_rot` (separate jitted calls over distinct copies) runs 20-27% faster than the in-program `loop`
  on 89/241 micro and 30/86 suite records (v5e: 0/240), e.g. gemm_1_14336_4096 86.7 vs 106.5 us, reproduced in a
  second session (86.2 vs 107.7) but not by an equivalent rotation in diag_v6e_modes.py (106.0 vs 106.6). Whole
  steps are one program and never see it. Per-op suite pred/meas against `loop` only: memory-bound geomean 0.736
  (best: 0.793), decode GEMVs uniformly 0.75-0.76, which is the whole-step ratio; the 0.92 "good" GEMVs were all
  `pipelined_rot`. The range-policy evidence "v6e GEMV 0.667-0.856" has the same upper-end inflation.
- **MXU throughput inside a program is bimodal, ~590-610 or ~730-767 TF (0.64-0.66 / 0.80-0.84 of 918)**, with
  HBM unaffected. 8192^3 is steady over 2 s and 4 min at 733 TF in one state; every reference v6e session (micro,
  suite, seq, seq_fused) reads ~590-600 TF on its 8192^3 sanity GEMM at start and end; one session switched from
  586 to 733 TF mid-run (diag3: every compute-bound or VPU op after 23:09 ran at 0.65-0.76x its earlier time,
  GEMVs unchanged). Consistent with a TensorCore clock/power state, which kiln cannot see (no TPU clock telemetry,
  `f_cap_op` is platform-only). Prefill per-op GEMMs in the reference session: 0.63-0.87 of kiln. Either state is
  far below central `unit_eff` 0.95 (v5e: 0.92-0.98).
- Not causes: JAX version (decode steps 13.69-13.79 ms on both), HBM spec (Google: 1638 GBps, matches the design;
  generation and stack count unpublished, and kiln's DRAM model has no per-generation timing terms, so the
  `hbm_kind` choice changes only the calibration key), tiling of the 2 x 256^2 MXU structure (the shortfall is
  shape-independent at 8192^3), operand perturbation or output copies in the `loop` timer (a variant without
  either matched it within 2% in the same chip state).
Open, for the owner: (1) a mechanism for single-program HBM concurrency on v6e (or an `eta_res` bound that admits
it), then generic-v3 fitting v6e; (2) canonical TPU per-op mode = `loop` (in-program, like the scored step), with
`pipelined_rot` kept as a diagnostic; (3) record and gate on the chip state (sanity GEMM per session) before using
v6e compute-bound records.
| TPU per-op timing mode (from §F.4) | Canonical TPU per-op mode is `loop` (in-program, like scored whole steps); `pipelined_rot` is diagnostic only (on v6e it is 20-27% faster via cross-program effects a single program never gets). |
| TPU chip-state gate (from §F.4) | Each TPU session records an 8192^3 sanity GEMM at start and end; compute-bound records from sessions whose state changes (v6e ~600 vs ~740 TF states) are flagged and excluded from fits. |
| eta_res bound | Widened from [0.85, 1.0] to [0.5, 1.0]: v6e single-program HBM efficiency 0.665 is measured physical behaviour. |
| v6e calibration | Add `platform:tpu_v6e` (fit on v6e loop-mode micro only); v6e stays out of generic sets so it remains a held-out device. |

### F.5 TPU timing mode, chip-state gate, eta_res bound, v6e platform set (implemented 2026-10-05)

- Canonical TPU per-op mode `loop` everywhere (`kiln_trace::meas::runner::TPU_OP_MODES`; kiln-calib devices, the
  legacy importer's `canonical_mode` and `method.settings.canonical_op_mode`, `m1_report`, `analyze_m2.py`,
  `tpu_bench.py` header/`canonical_s`); `pipelined_rot` and the derived `best` stay recorded as `diagnostic` modes.
  `loop` was slower than `pipelined_rot` on 8/252 v5e micro records (at most 2.4%; two of them fit records, 0.08%
  and 0.05%), so v5e inputs did change slightly; on v6e on 109/252 micro and 33/90 suite records.
- Chip-state gate: start/end sanity GEMM per TPU session (all three `tpu_bench.py` suites already record both);
  gate = |end/start - 1| > 3% (06 §4.3 per-record noise gate); compute-bound records (roofline at the session's
  spec peaks) of a changed session are excluded from fits (split reason `chip state: ...`), and fit notes, `kiln
  calibrate fit` and `report` list every session. All reference sessions are steady (v6e micro +2.5%, suite +2.0%,
  seq_fused -0.1%; v5e <= 0.3%); diag3_suite (-20.0%) is the only changed session and gates 8 records.
- `eta_res` bound [0.5, 1.0] in the registry. Stored sets keep the bounds they were fitted with (validation takes
  the union with the registry); `platform:a100_40gb` is not refit.
- `platform:tpu_v6e` v1 (cal1-92eee612): `eta_res(hbm3)` 0.723 [CI 0.713, 0.737], `unit_eff(v6e MXU)` 0.897
  [0.841, 0.983], `t_sync(static_dataflow)` 2.42 us; `t_dram_ramp(hbm3)` converged onto its lower bound 0 (in-program
  v6e HBM bandwidth is flat with size, §F.4) and is frozen at its prior 2 us with the `E-CAL-BOUND` alarm recorded
  in the set: open for the owner. Report (same device): per-op median 6.2% (target 8%), p90 27.5% (FAIL, 20%),
  decode b1/b8/b32 -0.5/-0.6/-2.7%, prefill -14.1% (FAIL, MXU state), coverage 100%.
- generic-v2 refit -> v3 (v5e inputs changed): `eta_res(hbm2e)` 0.90981 -> 0.90963 (engine drift since the v2 fit
  0.90965, the rest mode + bound prior), `eta_res(hbm2)` 0.90252 -> 0.90225 (bound prior), `t_sync(static_dataflow)`
  1.244 -> 1.254 us (engine drift). generic-v1 not refit (A100-only inputs unchanged); its range policy re-derived
  -> v3 (v6e GEMV evidence 0.667-0.856 -> 0.666-0.693, v6e best GEMM 0.839 -> 0.836); its `fit.test_records`
  still lists the TPU suite record hashes taken under `best`. generic-v1 on v6e (held out): per-op median 13.5% ->
  22.8% (geomean 0.862 -> 0.797; the earlier figure was `pipelined_rot` inflation), whole steps and coverage (75%)
  unchanged.
| Range policy vs held-out purity | Production range policies (novel designs) use evidence from ALL measured devices. Held-out validation reports must use leave-one-device-out ranges (exclude the held-out device's own evidence), so held-out coverage is a clean test. |
| Parameter at a physical zero bound | v6e `t_dram_ramp` converging to 0 (flat in-program bandwidth) stays frozen at its prior with E-CAL-BOUND recorded; revisit when a second device shows the same. 06 §3.4: generic-v3 does NOT include v6e (on hold). |

### F.6 Engine readiness before design search (implemented 2026-10-05)

- `kiln_ideal` fuses elementwise nodes (ruling above). Goldens run each design's own stack: unchanged.
- Vector work on a gang (an SM's four SMSP ALUs) is divided over its members, as MAC work already was (it ran at
  one SMSP's 16 lanes per SM since vector units were ganged).
- Special-function units (01 §5.4): a vector unit hands the transcendentals its special units implement (exp, or
  exp2 plus a vector multiply; log/log2; recip; rsqrt; tanh; erf; sin/cos) to the special units on its own feed
  memory, in the share that balances both (a wide bf16 VPU may keep some); each special unit is a compute
  resource. Unlisted functions stay on the vector unit at the transcendental class rate. v5e prefill -2.8%,
  v6e prefill -1.9%; an idle special unit leaves a step bit-identical.
- Low precision (02 §5.7, 03 §2.7): workloads take `+weights=`, `+kv=`, `+acts=<dtype>` (also on `standard`).
  kiln-wl inserts an explicit `convert` map kernel per contraction operand no MAC mode of the design runs (native:
  same registry name, or the element type of an MX/scaled operand; else the fastest lossless widening, never a
  narrowing; a stack with `dequantize` converts even where a mixed mode exists). The mapper fuses each convert into
  its contraction (operand fusion): the contraction reads the stored low-precision source and converts each tile
  on the vector unit beside its MAC unit. Modes match on registry names (`mxfp4`, `fp8_e4m3_pt`), which kiln-cost
  now receives, so MX scale streams and mixed MX modes are modelled. Trust M8 passes on 6 of 7 designs (v6e decode
  also becomes MXU weight-load bound). H100 SXM, llama3_8b, generic-v1 central, kiln_ideal:

  | step | bf16 | fp8 weights | + fp8 activations (W8A8) | + fp8 KV |
  |---|---|---|---|---|
  | decode b1 | 5.40 ms | 2.94 | 2.94 | 2.90 |
  | decode b8 | 6.02 | 3.56 | 3.56 | 3.21 |
  | decode b32 | 8.14 | 6.05 | 5.68 | 4.27 |
  | prefill b1 (2048) | 67.6 | 66.5 | 43.5 | 42.8 |

- Multi-die single package (ember 0.082 -> 0.458 of A100): gangs need a private level above the feeds (ember's 254
  tiles were one gang running every op on one tile's array); the level above a memory without a declared backing
  is the largest entity one level up (the stacked L3, not the CIM activation buffers); fewer slices than units
  spread over the set (all four chiplets); ember's L3 die and vertical links had no clock (0 B/s) and now run at
  the chiplet clock (assumed). `exec_model: static_dataflow` on ember (scratchpads only, compiler-scheduled) and
  tpu_v5e_2x2. Open, design-level: each chiplet's HBM controller and its L3 attach to the mesh at one router
  (205 GB/s each against a 2.048 TB/s stack), so ember is link-bound well below its claimed 8.2 TB/s; interleaving
  over all four stacks sends 3/4 of the traffic over UCIe (die-aware homes would remove that, about 10% here);
  CIM and HBM-PIM units are near-memory and stay unmapped until NMP execution (03 §6).

## G. M5 implementation notes (visualizer, 2026-10-05)

Recorded in full in 05 §12; the cross-section points:

- egui/eframe pinned at 0.35.0, not 05's 0.36.2: 0.36 requires rustc 1.95, the workspace is on 1.92.
- The native app is opt-in (`kiln-cli --features gui`) so default builds and `cargo test --workspace` stay lean;
  `kiln viz <run>` in a default build exits 69 with a hint, `kiln viz render` always works.
- 03's Tier A op envelopes do not cover their group's traffic; per-op roofline points use execution groups and
  per-op times are group time split by the group's binding term (05 §12). 03 may want to report per-op attributed
  times in `OpResult` directly.
- `SimResult.ops` is empty at `trace: summary`; `kiln eval -o run.kiln` asks the engine for ops regardless.
- Archive schema `kiln.archive/1` adopted the spellings `kiln_evo` already wrote (`schema`, `axes`,
  `descriptor_values`, `wall_time`, JSON-string `fitness_components`); 05 §3.9 lists both accepted forms.
- Floorplans use kiln-phys's placement since 2026-10-06 (trace schema 1.1, 05 §12): kiln-phys arranges dies
  without compute-unit area, so units are filled in by area inside their placed parent; the unplaced hierarchy
  layout remains only as the labelled fallback when kiln-phys fails. `kiln viz <design>` needs no simulation.


### F.7 Out-of-sample A100 check of the PyTorch stack recipe (2026-10-05)

Predictions frozen at a27f6f6 before measuring (calibration/predictions/oos_a100_predictions.json); measured calibration/measurements/a100_2026-10-05_seq_oos.json; no recipe changes. Central (high corner) vs measured:

| Workload | Predicted ms | Measured ms | Error | Covered |
|---|---|---|---|---|
| GPT-J-6B decode b1 | 11.11 (12.92) | 12.92 | -14.0% | edge |
| GPT-J-6B decode b8 | 15.80 (17.89) | 18.07 | -12.5% | no |
| GPT-J-6B decode b32 kv1024 | 21.10 (23.41) | 24.63 | -14.3% | no |
| GPT-J-6B prefill b1 | 114.9 (135.2) | 143.2 | -19.8% | no |
| Llama-2-70B per layer decode b1 | 1.331 | 1.373 | -3.0% | |
| Llama-2-70B per layer decode b8 | 1.373 | 1.444 | -4.9% | |
| Llama-2-70B per layer decode b32 | 1.513 | 1.809 | -16.3% | |
| Llama-2-70B per layer prefill b1 | 17.83 | 15.95 | +11.8% | |

Kernels: Llama-2-70B L1 step 57 measured = 57 predicted (exact); GPT-J step 1180 measured vs 789 predicted (~42/layer vs 28), as foreseen (gelu_new Python formula, separate q/k/v, interleaved partial rotary, biases). Conclusion: the recipe transfers within the Llama op family at small batch (decode b1/b8 within 5%), not across architectures, and the b32 matmul-side gap and the prefill error are not recipe issues. Fitness is unaffected (kiln_ideal on both sides); realistic scores and absolute numbers for non-Llama architectures carry this error until per-architecture recipe rules exist.


### F.8 Engine correctness before design campaigns: H100 GEMM efficiency and register-file traffic (2026-10-05)

- H100 dense GEMMs ran at ~37% of peak because the design fed A and B to the tensor cores through the register
  file (A100's `mma.sync`): at 2x A100's MACs per SM the accumulator writes alone filled the assumed 1024-bit RF
  write port, and the A/B fills stalled behind them. The design now follows Hopper `wgmma` (PTX ISA): `a`, `b`
  from shared memory, accumulators in registers (`rf.backing: l1` for the epilogue); SMEM 128 B/clk/SM and L2
  128 B/clk/slice from Luo et al. arXiv 2501.12084 (L2 derived: A100's 5120 B/clk x the measured H800/A100
  ratio 2.2). Engine support: kiln-map chains a unit from the feed memory whose chain holds its other feeds,
  stages each operand only to its own feed level and charges feed bytes per role; kiln-cost follows a declared
  `backing`; MMA modes narrower than 16 bits span `16/bits` k per lane (PTX k32 fp8/int8, k64 4-bit).
- `operand_run` (01 §5.2): the most consecutive temporal steps an input stays in a matrix unit between feed
  reads. A100 `mma.sync` re-reads A and B every instruction (1); H100 keeps A over one `wgmma`'s N <= 256 (32 n8
  steps) and reads B per instruction. Before, level-0 stationarity was unlimited and the latency search stopped at
  the first floor-reaching order, so A100 RF reads per MAC differed 1.6-2.7x between GEMM 8192^3 and 16384^3.
- Results (central, power-capped clock): H100 GEMM 4096^3/8192^3/16384^3 369/382/373 -> 760/778/686 TFLOPS at a
  solved 1.66-1.69 GHz (673/663 W board); Llama-3-8B prefill (m1, nominal clock) 79.1 -> 55.2 ms. A100 per-op
  compute-bound geomean pred/meas 0.845 -> 0.857, prefill 135.4 -> 136.9 ms, decode unchanged; A100 GEMM 16384^3
  power 323 -> 354 W. v5e/v6e unchanged.
- Open: A100 GEMM 16384^3 is mapped as 108 n-strips (16384 x 152 per SM) and binds on the SM's L1 port at 223
  TFLOPS; the 2D candidates score worse under the quick candidate search. The SASS operand reuse cache is not
  modelled (A100 RF reads are an upper bound). fp64 DMMA on H100 reads registers, not modelled per mode.

## H. Physical-model fixes before design campaigns (2026-10-05)

| Topic | Ruling |
|---|---|
| Unpublished power caps | A cap with no published value is `assumed: { lo, hi, basis }` (01 §12): reported, never enforced, so no reference design throttles under a placeholder; the `search` profile rejects it (E-IR-1106) and physical findings on such a real chip are residuals (W-PHYS-RESIDUAL), as on a chip with a published die area. TPU v5e: nominal 200 W in [120, 250] W (TPUv4i 175 W TDP, TPU v4 192 W measured max; Google publishes no v5e figure). TPU v6e: nominal 450 W in [300, 700] W (v5e's range x 4.7 peak compute / 1.67 energy efficiency, Google's Trillium announcement). Published caps and measured maxima (A100, H100, TPU v4) stay enforced. The engine's envelope margins against `lo` (pessimistic) are left to the engine owner. |
| v6e prefill after the cap fix | Throttling under the 300 W placeholder was hiding F.4's v6e error. llama3_8b prefill_b1 (own stack): generic-v1 48.39 -> 40.18 ms, platform:tpu_v6e 50.16 -> 43.29 ms, measured 50.37 ms (-20.2% / -14.1%, the F.4 gap: v6e HBM at 0.665 of spec inside one program, GEMMs below the MXU model). Fix belongs to those parameters, not to a cap. |
| P6 | Physical, one-sided: links are measured on the live geometry (04 §6.2, §7.2). Unreachable hardware (no channel path from a unit, harvested instances, misc area) leaves links and time bit-identical; a reachable unit no op can use only stretches distances (die arrangement without unit area, then stretched; intra-cluster wires 0.5 sqrt(live cluster area)). Tests: kiln-sim `unreachable_or_incapable_units_change_nothing` (proptest, exact / never faster within 1e-9), kiln-phys `p6_dead_area_never_shortens_a_link` (A100, TPU v4, v5e: misc block and orphan memory bit-identical, unused int4 unit never shortens a link). Trust M2 now lists H100/v6e (2x units stretch the die, ratio 1.0004x / 1.0000x) instead of v4/v5e. |
| TPU v4i sources | Verified against Jouppi et al. ISCA 2021 (doi 10.1109/ISCA52012.2021.00010): Table 1 die < 400 mm2, 7 nm, 16 B transistors, 1050 MHz, TDP 175/275 W, idle 55 W, 1 core, 4 x 128x128 MXUs, 138 TFLOPS, 144 MB on-chip / 8 GB at 614 GB/s, 2 x 400 Gbit/s ICI; Fig. 6 / text: CMEM 28% of the die, MXUs 11% (also Table 7). The die target is one-sided (< 400; the former [330, 400] lower edge was not published), likewise TPU v4 (< 600, ISCA 2023 Table 4). |
| GP100 in the fit | p100_sxm2_16gb is an area-calibration design (Pascal whitepaper: 610 mm2, 15.3 B, 60 SMs / 56, no tensor cores): the N16 die and the fixed SIMT control. V100 stays held out. |
| Calibration reproducibility | `phys_calibrate::run` is the single fit; kiln-sim `phys_calibration_reproduces` refits and compares with the committed sets (values to 1e-9, inputs hash, no parameter at a bound). Any change to designs, tables, targets or engine traffic that moves the fit needs `WRITE=1 cargo run -p kiln-sim --release --example phys_calibrate`. |

