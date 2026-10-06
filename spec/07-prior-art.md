# 07 Prior art (as of 2026-10-05)

Method: abstract/README level reads via WebFetch/WebSearch on 2026-10-04/05. "(unv.)" = not verified this session (background knowledge or secondary summary). No tool was run; accuracy numbers are the papers' own claims.

## 1. Verdict

No simulator found covers most of our requirement set (structurally open + multi-chip + near-memory + floorplan/wire-aware + validated on A100/TPU LLM runs). Each covers 2-3 of the 5. The closest are ATLAS and DeepStack, both 3D-DRAM-specific. The gap is real but narrow: the 3D-DRAM family is converging on parts of it, so we should not claim a gap on "multi-chip + near-memory + thermal".

| Tool | Covers | Accuracy evidence (own claims) | License / code | Use |
|---|---|---|---|---|
| ATLAS, [2604.08044](https://arxiv.org/abs/2604.08044) | Hybrid-bonded 3D-DRAM accelerators; multi-chip; BookSim2 NoC (fixed 2D mesh); HotSpot-7 thermal; FP16 only; ~5.4K LoC C++ | <=8.57% error vs own 7nm test chip, not A100/TPU | "Will be open-sourced upon publication", no repo URL found (unv. whether released) | Ignore as code. Cite as validation-against-silicon precedent; thermal-via-HotSpot design is a reference |
| DeepStack, [2604.04750](https://arxiv.org/abs/2604.04750) | 3D-DRAM-stacked distributed LLM inference, thermal/power/area constraints, 2.5e14-point space | 12.92% vs vLLM on 8xB200; 2.12% vs NS-3 | Code + MICRO artifact badges (repo URL not captured) | Best differential oracle candidate for stacked-memory systems; read before spec 03 |
| VOXEL, [2604.26821](https://arxiv.org/abs/2604.26821) | 3D-stacked AI chip, compiler-aware; tile-to-core and tensor-to-bank mapping, NoC, thermal | "validated vs silicon emulation" (not real silicon); MICRO'26 | Open-sourced (repo not located) | Reference for mapping-to-bank; not an oracle |
| SCALE-Sim EVA, [2608.12354](https://arxiv.org/abs/2608.12354) | Command/tensor-based, component-graph, trace visualisation; Python | None reported; no code link, no multi-chip, no physical model (read of HTML) | Paper CC-BY; code not found | Closest in spirit to our IR + visualizer. Watch; do not depend |
| SCALE-Sim v3, [2504.15377](https://arxiv.org/abs/2504.15377) | Systolic, multi-core, sparsity, Ramulator + Accelergy hooks; Python | Cycle-accurate; no LLM-hardware validation seen | github.com/scalesim-project/SCALE-Sim (license unv.) | Oracle for systolic-array GEMM only |
| Timeloop/Accelergy/CiMLoop, [github.com/NVlabs/timeloop](https://github.com/NVlabs/timeloop), [mit-emze/cimloop](https://github.com/mit-emze/cimloop) | Loop-nest mapping + energy; CiMLoop (v4) adds CiM statistical energy; BSD-3, C++/Python | CiMLoop ISPASS'24; no multi-chip, no floorplan | Active-ish (938 commits); needs isl/barvinok | Port the mapping/energy model idea (kiln-cost); run as oracle on single-chip subset; CiMLoop's CiM component energy tables to port |
| ZigZag / Stream, [KULeuven-MICAS/stream](https://github.com/KULeuven-MICAS/stream) | Multi-core layer-fused scheduling, MILP (TETRA), ONNX, AIE emit; MIT, Python 3.12, 1.2k commits | Edge DNN focus; LLM-scale validation not seen | Active | Already our fixed-fork oracle. Keep as oracle for subset it expresses |
| ASTRA-sim 2.0/3.0, [github.com/astra-sim/astra-sim](https://github.com/astra-sim/astra-sim), [2606.10440](https://arxiv.org/abs/2606.10440) | Collectives, Chakra ET workloads, analytical + NS-3 + Garnet backends; 3.0 adds cache-line GPU model, InfraGraph, MSCCL++ collectives | 3.0 validation not extracted from abstract (unv.) | Open, 700 stars, 82 open issues | Port analytical collective algebra to Rust; oracle for collectives; adopt Chakra ET as an import format |
| LLMCompass, [2312.03134](https://arxiv.org/abs/2312.03134) | Hardware eval for LLM inference, area/cost model, mapper, systolic sim | 10.4% operator / 4.1% end-to-end vs A100, MI210, TPUv3 | BSD-3, Python, 279 stars, 7 commits (stale-ish) | Best methodological template (A100+TPU validation + area/cost). Oracle on GEMM/attention. Do not fork |
| GenZ, [2406.01698](https://arxiv.org/abs/2406.01698) | Analytical LLM platform model; MoE, Mamba, chunked prefill, spec decode | Max geomean error 5.82% vs real platforms | Public GitHub (license unv.) | Oracle for Tier A on whole-model latency |
| Calculon, [github.com/calculon-ai](https://github.com/calculon-ai) | Analytical LLM training model, parallelism search | SC'23 | Open, Python | Parallelism-strategy reference; training-side only |
| LLMServingSim 2.0, [2602.23036](https://arxiv.org/abs/2602.23036) | Serving-level; heterogeneous/disaggregated; profile-based hardware plug-ins | 0.95% avg error vs real deployments (profile-fed) | casys-kaist repo | Out of scope (serving layer); profile-driven means no new-hardware prediction |
| Vidur, [2405.05465](https://arxiv.org/abs/2405.05465) | Serving scheduler sim from profiled operators | <9% latency error | MIT, Python | Ignore (needs real-hardware profiles) |
| GPU-Tile-Sim, [2607.11262](https://arxiv.org/abs/2607.11262) | Warp-tile graph GPU sim for LLM kernels | MAPE 1.22-8.71% on A100, H100 | MICRO'26; code unv. | Evidence that tile-graph abstraction hits <9% on A100. Supports our Tier A design |
| Accel-Sim 2.0 (Aug 2026), [accel-sim.github.io](https://accel-sim.github.io/) | Cycle-level NVIDIA; H100 | 99% corr, 13.4% MAE vs H100 (site claim) | License not stated | Not an oracle for us (too slow, GPU-only) |
| NeuroSim V1.5, [2505.02314](https://arxiv.org/abs/2505.02314) | CiM circuit-level PPA, transformer hybrid ACIM/DCIM | Device/circuit level | github.com/neurosim/NeuroSim | Source of CiM macro area/energy constants only |
| gem5-based (PIMSys, gem5-SALAM) | PIM-HBM full-system | n/a | open | Ignore (cycle-level, too slow) |
| Rust sims: SimTPU ([repo](https://github.com/kristopherpaul/SimTPU)), Akita (Go, [2604.28073](https://arxiv.org/abs/2604.28073)) | SimTPU: toy ISA-level, 1 star, LLM-training-restricted license | none | n/a | Ignore. No serious Rust accelerator-system simulator found |

Single biggest competitor to watch: DeepStack + ATLAS pair, plus any Chiplet/Beacon-style evaluator. Nothing found combines open structure (arbitrary graph of memories/tiles/chips) with floorplan-driven wire costs.

## 2. Component models

| Need | Candidate | Plan |
|---|---|---|
| DRAM/HBM timing | Ramulator 2.0 (MIT, C++; HBM3/DDR5/LPDDR5/GDDR6; 2606.14566 rebuts the Mess accuracy criticism as misconfiguration); DRAMsim3 (unv.). Rust: `ramu_rs` only DDR4+FCFS, no HBM/refresh ([lib.rs](https://lib.rs/crates/ramu_rs)) | Do not port. Both tiers: per-channel structured statistical model with row-buffer locality classes, calibrated on A100 HBM2 (03 §5.3); Ramulator 2 as offline table generator and oracle |
| NoC | BookSim2 (used by ATLAS), Garnet (via ASTRA-sim) | Rust analytical hop/serialisation model + simple flit-level Tier B; BookSim2 as oracle |
| Collectives | ASTRA-sim analytical backend | Port formulas to Rust; oracle |
| SRAM | CACTI 7, OpenRAM (unv. versions) | Fit CACTI sweeps to tables in data files; do not link |
| Wires/repeaters | Textbook RC/repeater models (Bakoglu; ITRS/IRDS interconnect tables) (unv.) | Implement analytically; constants cited per node |
| Placement | DREAMPlace (GPU, PyTorch), OpenROAD RePlAce, DG-RePlAce ([2404.13049](https://arxiv.org/abs/2404.13049)) | Too heavy for the inner loop. Use own coarse analytical/force-directed block placer; DREAMPlace-class only for elite audit (unv. fit) |
| PIM | Samsung HBM-PIM (16-lane FP16 SIMD per bank pair; ">2x perf, >70% energy" [press](https://news.samsungsemiconductor.com/global/samsung-develops-industrys-first-high-bandwidth-memory-with-ai-processing-power/)); SK hynix GDDR6-AiM (per-bank multipliers + adder tree; 16 Gbps; "16x" on select ops, 80% power cut, press [hothardware](https://hothardware.com/news/sk-hynix-ai-accelerating-pim-memory)); UPMEM PrIM ([2105.03814](https://arxiv.org/abs/2105.03814)) | Use as PIM unit parameter presets and validation points. Vendor claims are marketing; use PrIM measured UPMEM numbers as the only real measurement |
| MX hardware cost | 2511.06313 (MXINT8 657, MXFP8/6 1438-1675, MXFP4 4065 GOPS/W, own synthesis); VMXDOTP 2603.04979 (7.2% area overhead, 843/1632 GFLOPS/W MXFP8/4 at 1GHz 0.8V) | Cost tables per format = assumed + cite; flag as synthesis results at unknown node, not silicon |
| PIM simulators | NeuPIMs 2403.00579, AiM sim (Ramulator-2-based, gitee), uPIMulator, Ramulator-PIM | Oracles for PIM unit microbenchmarks only |

## 3. Rust ecosystem (crates.io API, 2026-10-05)

| Crate | Version / date | Note |
|---|---|---|
| egui | 0.36.2 (Sep 2026) | Pre-1.0, breaking changes every minor; pin exact version. Immediate mode suits timelines/floorplans; large traces need own culling/LOD |
| wgpu | 30.0.1 (Aug 2026) | Fast major churn (29 to 30 in ~1 month); pin; eframe pins its own wgpu |
| tauri | 2.12.1 stable, 3.0 alpha | Webview UI; only if we want a web frontend; avoid 3.0 alpha |
| pyo3 | 0.29.3 (Sep 30 2026) | Pre-1.0 churn; use abi3 + maturin; release the GIL around `evaluate` |
| arrow / parquet | 60.0.0 (Sep 2026) | Very fast major cadence; MSRV 1.88; OK for traces and calibration tables |
| rkyv | 0.8.18 | Zero-copy design snapshots; unsafe-adjacent, keep for caches, JSON canonical |
| petgraph | 0.8.3 (Sep 2025) | Stable; no stable-order guarantees on hash-based variants, use `StableGraph`/sorted ids |
| rayon | 1.12.0 | Parallel across designs, not inside one eval (determinism) |
| proptest | 1.11.0 (Mar 2026) | Good for physical-floor invariants and Tier A vs B agreement |
| DES crates | simrs 0.2 (2022), desim 0.4 (2023, GPL-3), others small | Nothing credible. Hand-roll the event queue (binary heap with (time, seq) key) for determinism |

## 4. LLM-driven hardware search: scoop check

| Paper | What it is | Scoops us? |
|---|---|---|
| Agentic Architect 2604.25083 | AlphaEvolve-style code evolution + cycle-accurate sim; cache replacement, branch pred, prefetch (IPC) | No: policies, not structurally open accelerators; no hardware calibration |
| MicroEvo 2608.06183 | LLM-guided MCTS, CPU microarchitecture DSE, repo GEAR-SEU/MicroEvo-ICCAD-26 | No: parametric CPU PPA |
| LUMINA 2603.05904 | LLM GPU DSE over 4.7M configs, 6 designs beat A100 in perf+area in 20 steps | Partial: same narrative on a fixed parametric GPU space; simulator unspecified in abstract, check full paper |
| Beacon 2608.30932 | Multi-agent LLM DSE for heterogeneous multi-chiplet DL accelerators, report-driven | Partial: multi-chiplet + LLM; parametric; no hardware validation seen |
| AgentDSE 2606.21836 | Coding agent reasons over simulator, 100x fewer evals (MLArchSys@ISCA'26) | No: treats existing simulators |
| CHIA 2606.27350 | Open loop framework: gem5/ChampSim/FireSim + AlphaEvolve/AdaEvolve | Infrastructure competitor for the loop; not for the simulator |
| A3D 2605.15237 | Agentic HLS accelerator generation (Claude + Catapult) | No |
| AlphaEvolve 2506.13131, OpenEvolve, ShinkaEvolve 2509.19349 | General evolve frameworks | Use as loop engines |

No hit combining MAP-Elites (quality-diversity) + structurally open accelerator graph + calibrated physical simulator + A100/TPU baselines. I did not find a scoop, but the search is of search-engine depth (arXiv listing not exhaustively crawled). LUMINA and Beacon are the nearest and must be cited and read in full.

## 5. Bottom line

Reuse:
- Run as oracle: our Stream fork (MIT), Timeloop (single-chip subset), LLMCompass (GEMM/attention, A100/TPUv3), GenZ (whole-model Tier A), ASTRA-sim analytical (collectives), Ramulator 2 (DRAM microbenchmarks), DeepStack (stacked-memory systems, if code builds).
- Port to Rust: ASTRA-sim analytical collective formulas; Timeloop/ZigZag loop-nest cost idea (already planned as kiln-cost); CACTI/CiMLoop/NeuroSim constants into data files.
- Adopt formats: Chakra ET import, Perfetto export (already planned).
- Methodology to copy: LLMCompass (error reporting per-operator and end-to-end vs real A100/TPU); GPU-Tile-Sim tile-graph; ATLAS/DeepStack validation-against-silicon and thermal coupling.
- Ignore: serving sims (Vidur, LLMServingSim), Accel-Sim, gem5 PIM, Rust DES crates, SimTPU.

Three biggest risks:
1. **Accuracy ceiling.** Best-in-class claims are ~4-13% error and mostly on fixed hardware, often with profile feeds (LLMServingSim, Vidur). A structurally open simulator cannot be profile-fed, so beating-A100 claims on novel designs rest on extrapolation we cannot validate. Only A100 is measured and H100 is absent; TPU measurements are pending. Evolution will exploit simulator error (cf. LUMINA/Beacon report wins only in their own simulators).
2. **Scooping on adjacent claims.** 3D-DRAM + multi-chip + thermal + LLM DSE is crowded (ATLAS, DeepStack, VOXEL, Beacon, LUMINA). "Multi-chip + near-memory" alone is not novel; the defensible pieces are open structure, wire/floorplan-driven costs inside the loop, and quality-diversity against measured A100/TPU. DeepStack's 100,000x-faster claim sets the speed bar for Tier A.
3. **Rust ecosystem churn and no sim base.** There is no Rust accelerator-sim or DES to build on, and egui/wgpu/pyo3/arrow all ship breaking majors monthly. Pin versions, isolate UI and bindings behind thin crates, hand-roll the event core.

Unverified/open: ATLAS and VOXEL repo status; SCALE-Sim EVA code; DeepStack repo URL; ASTRA-sim 3.0 validation numbers; LUMINA's simulator; licenses of GenZ/DeepStack/SCALE-Sim; CACTI/OpenRAM/BookSim2/wire-model details (background knowledge only).
