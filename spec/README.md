# kiln spec v0

`kiln` is a Rust simulator for structurally open AI-accelerator systems (any number of memories, tiles, NoC nodes,
chips; near-memory compute; per-tensor precisions; floorplan-driven wire costs). It is fast enough for LLM-driven
evolutionary search (Tier A, <= 50 ms per design-layer) and accurate enough, through calibration against measured
A100 and TPU runs, an event-driven audit tier and a simulated trust suite, that a design beating A100/TPU-class
silicon in kiln at the same physical envelope would beat it on silicon. Scope is LLM inference (training is out of
scope); designs are scored on whole steps and results are reported as intervals. This directory is the full design
spec; implementation follows 06 §10.

## Reading order

1. `00-overview.md` (binding contract and cross-section decisions), then `08-decisions.md` (how conflicts were
   resolved, open questions, owner rulings and how they were applied).
2. `01` and `02` (the two IRs), `03` (engines), `04` (physics), `05` (traces and viz), `06` (validation, API, plan).
3. `07` for prior art when judging reuse or novelty.

## Sections

| File | Scope | Owner crate(s) |
|---|---|---|
| 00-overview.md | Requirements, conventions, crate layout, binding decisions, fidelity tiers | (all) |
| 01-hardware-ir.md | Hardware entities, JSON5 authoring language, expansion, `HwModel`, design hash, `E-IR-*` validation, precision registry | `kiln-ir` |
| 02-workload-ir.md | Tensors, ops, kernels, graphs, scenarios, `ParallelPlan` and partition, frontends, metrics and whole-step scoring basis, `E-WL-*` | `kiln-ir` (types), `kiln-wl` (passes) |
| 03-mapping-engine.md | Intra-unit cost model, mapper, Tier A and Tier B engines, whole-step mode, NMP, multi-chip, invariants, `SimResult`, `explain_run`, calibration hooks, parameter ranges and corner evaluation | `kiln-cost`, `kiln-map`, `kiln-sim` |
| 04-physical-model.md | Node tables, block area/energy, floorplan and placer, wires, power, thermal, cost, physical calibration, `E-PHYS-*` | `kiln-phys` |
| 05-visualizer.md | Trace data model and `.kiln` container, archive schema, views, headless rendering, Perfetto export | `kiln-trace`, `kiln-viz-render`, `kiln-viz` |
| 06-validation-api.md | Test layers, calibration methodology, bench harness, CI, Python API, CLI, performance targets, milestones, trust suite | `kiln-py`, `kiln-cli` (plus test obligations on all) |
| 07-prior-art.md | Existing simulators and models; reuse, oracle and ignore decisions | none |
| 08-decisions.md | Integration decision log, fact checks, consolidated open questions, owner rulings and their application | none |

## Glossary

| Term | Meaning | Defined in |
|---|---|---|
| Design | One hardware document (`HwDoc`, JSON5 authoring, canonical JSON for hashing; `design_hash = hw1-...`) | 01 §3, §17 |
| `HwModel` | Expanded, validated hardware graph (instances, channels, shared resources) that engines consume | 01 §16 |
| Workload | `WorkloadDoc`: model graphs + scenarios + plans; `workload_hash = wl1-...` | 02 §1, §11.5 |
| Scenario | Snapshot, static batch or continuous serving; expands to phase instances | 02 §8 |
| Kernel | Affine iteration space + operand index maps + scalar body; what every op lowers to | 02 §4 |
| `ParallelPlan` | Logical mesh axes, sharding template/annotations, pipeline plan | 02 §9.2 |
| Whole step | The scored unit: a full decode or prefill step simulated end to end (`scope: step`; `scope: layer` where 02 §11.4 says so); isolated per-op timing is calibration only | 02 §12.5, 03 §4.9 |
| Interval result | `{low, central, high}` for times, tokens/s, power, area and score, from deterministic corner evaluation over parameter ranges | 03 §9.1, 06 §6.3 |
| `PartitionedProgram` | Output of `kiln-wl::partition`: per-stage SPMD graphs with collectives inserted | 02 §9.7 |
| Mapping | Serializable placement of ops, tensors, routes, groups and collective plans; produced by heuristic or search | 03 §3.1 |
| LoweredGraph | DAG of tasks annotated with per-resource demands; shared input of Tier A and Tier B | 03 §1 |
| Tier A | Analytical engine. A0 = roofline (reference only), A1 = max resource occupancy, A2 = A1 with dependencies and exposed overheads (provable floor of Tier B), A_est = A2 + contention correction (the score) | 03 §4.2 |
| Tier B | Event-driven engine with contention, credits, DRAM model and full timelines; audit tier | 03 §5 |
| `T_A2 <= T_B` | Provable tier relation under the same mapping, graph, calibration and clock; violation is a bug | 03 §4.6 |
| `SimResult` | Per-run result (times, floors, energy, power, ops, resources, bottleneck, invariants, provenance) | 03 §10 |
| `explain_run` | Single source of LLM-readable bottleneck text, in `kiln-sim` | 03 §10 |
| Invariants I1-I15 | Physical floors and conservation laws checked on every result (`E-FLOOR-*` on failure) | 03 §8 |
| `.kiln` trace | Single-file container: manifest + Arrow IPC tables; levels `summary`, `ops`, `full` | 05 §3 |
| Archive | MAP-Elites archive directory (`archive.json`, `designs.arrow`, `generations.arrow`, runs) | 05 §3.9 |
| Calibration set | Versioned, hashed parameter file (`platform:*`, `generic-vN`, or `null`) of registered mechanism parameters, each with a range; splits are `fit` / `test` | 06 §3 |
| Trust suite | T1 chip-history backtest, T2 known-answer/metamorphic tests, T3 red-team campaigns, T4 cross-simulator agreement, T5 trust report; gates claims and releases | 06 §12 |
| Evidence grades | G1 own measurement with full provenance, G2 own with incomplete provenance, G3 vendor/paper microbenchmarks, G4 MLPerf and system-level published numbers | 06 §2.5 |
| Trust levels | `uncalibrated` -> `calibrated_single_chip` -> `calibrated_physical` -> `audited` -> `calibrated_multi_chip` | 06 §10 |
| Milestones | M0 foundations, M1 Tier A + API, M2 single-chip calibration, M2b H100 held-out, M3 physical model, M4 Tier B + audit, M5 visualizer (05 V2-V4), M6 multi-chip, M7 near-memory + precisions, M8 hardening | 06 §10 |
| Profiles | Validation profiles `full`, `reference`, `search` (no unpriced performance, `E-IR-UNPRICED`), `stream_compat` | 01 §18.3 |
| Envelope | Die area, reticle, power cap, node, memory technology and shoreline a candidate must match for `matched_envelope` fitness | 06 §6.4 |

## End-to-end data flow

```
design (JSON5) ---------+
workload (JSON / zoo) --+--> parse, migrate, templates, canonical JSON + hashes      kiln-ir (01, 02 types)
scenario + plan --------+            |
                                     v
                         expand + validate (E-IR-*, E-WL-*, profile)                 kiln-ir
                                     |
                                     v
                 bind -> lower to kernels -> partition (collectives inserted)        kiln-wl (02)
                                     |
                                     v
             map: plan choice, mesh embedding, splits, tensor homes, routes,         kiln-map + kiln-cost (03)
                  fusion groups, intra-unit loop nests -> Mapping + LoweredGraph
                                     |
                                     v
       place + characterize: floorplan, wire/link costs, area, energy tables         kiln-phys (04)
       (two-round place-map loop driven by kiln-map/kiln-sim with TrafficMatrix)
                                     |
                                     v
       Tier A (analytical, per-phase DVFS)  |  Tier B (event-driven, windowed DVFS)  kiln-sim (03) + kiln-phys power
       whole step per phase (03 §4.9); central, low, high parameter corners (03 §9.1)
                                     |
                                     v
       invariants I1-I15 + physical floors (violation = E-FLOOR-*, quarantine)       kiln-sim
                                     |
                                     v
       SimResult + .kiln trace (summary / ops / full) + explain_run text            kiln-trace (05), kiln-sim
                                     |
              +----------------------+-----------------------+
              v                      v                       v
      kiln viz / render       kiln-py Result, score,     kiln-cli eval/compare,
      (05)                    cascade, audit, archive    calibrate, corpus (06)
                              (06; archive schema 05)
```
