# 04. Physical model (crate `kiln-phys`)

Status: spec v0, implementation-ready. Follows the shared contract in `00-overview.md` (units, ids, determinism,
provenance, structured errors, calibration constants in versioned data files with citations).

Scope: technology node tables, block-level area and energy models (datapath, SRAM, register files, NoC routers,
PHYs), floorplan model (dies, packages, interposers, 3D stacks, shoreline), placer, wire model, area roll-up,
power model inputs for the DVFS solve, thermal model, optional cost model, and the calibration of all of the
above. Not in scope: cycle timing of the mapped workload (03), DRAM timing (03 §5.3 / 01), RTL.

Why the current placeholder fails (`harness/physics.py`, A100 predicted 306 mm2 / 249 W vs real 826 mm2 / 400 W):
it models only MAC arrays, the 40 MB L2 and 5 HBM PHYs. It omits (a) ~27 MB of register file (108 SMs x 256 KiB)
and ~21 MB of L1/shared memory (108 x 192 KiB), i.e. more storage than the L2; (b) all non-tensor SM logic (FP32/
INT32/FP64 lanes, SFUs, schedulers, LSU, texture); (c) the 20 fused-off SMs and spare L2/HBM sites of the full
GA100 die (128 SMs, 6 HBM sites); (d) 12 NVLink3 links and PCIe Gen4 PHYs; (e) the SM-to-L2 crossbar and its
wiring; (f) multi-mode tensor-core overhead (fp64/tf32/fp16/bf16/int8/int4 in one datapath); (g) small-macro SRAM
periphery (array efficiency far below 0.4 for 32-256 KiB macros with wide ports); (h) HBM DRAM power on the board
TDP. This section's model is structured so each of these is an explicit term, not a scalar fudge.

---

## 1. Design principles

1. **Structure first, then calibrate.** Every area/energy term is a physical mechanism with a formula and a
   tabulated coefficient. Calibration adjusts named coefficients within bounds (03 §9 rules apply verbatim: one
   physical role per parameter, bounded, fit on isolating data, dual reporting, scoped).
2. **Relative accuracy matters most.** kiln's claim is "beats A100/TPU at the same envelope". Iso-node
   comparisons share node constants, so node-table errors largely cancel. Cross-node comparisons (e.g. N7 design
   vs H100 on N4) carry the node-scaling uncertainty, which every result reports (§12.5).
3. **Every number is sourced.** Each table value carries `q` (quality: `pub` published, `der` derived from
   published by a stated formula, `asm` assumed, `fit` produced by calibration) and `c` (confidence: `H`, `M`,
   `L`). A result's provenance lists the count and total sensitivity of `asm`/`L` values it depends on.
4. **Determinism.** All iteration orders are by typed index; floating-point reductions are in fixed order; all
   iterative solvers run a fixed iteration count or a deterministic convergence test; no threads in Tier A
   reductions (Tier B may parallelize over independent dies with fixed merge order).
5. **Speed.** Per design (not per layer): characterize + Tier A floorplan + link table <= 3 ms for designs up to
   ~2,000 placeable blocks after array collapsing (§6.1). The result is cached by design hash and reused for every
   layer, so it is amortized across the 50 ms/(design, layer) Tier A budget. Power/thermal evaluation per
   activity report <= 0.2 ms (Tier A), <= 50 ms (Tier B thermal grid).

---

## 2. Crate layout

`kiln-phys` (no split proposed; the modules share the `TechNode` and `Floorplan` types and nothing here is reused
independently by another crate the way `kiln-cost` is):

| Module | Content |
|---|---|
| `tech` | `TechNode` loading, schema validation, interpolation, node scaling, source registry |
| `blocks` | Area/energy/leakage models: datapath (MAC modes, vector, SFU), SRAM macro, register file, router, PHY, misc |
| `floorplan` | `Floorplan`, `Die`, `Package`, `Layer3d`, shoreline assignment, keep-outs, outlines |
| `place` | Tier A slicing placer, Tier B analytical placer + legalizer + refinement, DEF import |
| `wire` | Repeated-wire model, pipelining, link derivation, congestion/bisection (RUDY) |
| `noc` | Router area/energy/latency vs radix/VCs/buffers/width |
| `power` | Dynamic energy tables, leakage `P_static(T,V)`, clock tree, V/f curves, board overheads |
| `thermal` | Tier A density check, Tier B steady-state grid solver (2D and stacked 3D) |
| `cost` | Optional wafer/yield/package/HBM cost model |
| `calib` | Parameter registry, priors, bounds, fitting targets, MAP fit, LOO validation (driven by `kiln calibrate`, 06) |

Data directory (versioned, hashed into provenance and into the calibration-set hash per 01 §13): `kiln-phys/data/`.
Ids match 01's `TechRef` names (`tsmc_n7`, `tsmc_n4p`, ...); aliases are listed in `tech/aliases.json`
(`tsmc_n4` and `tsmc_n4p` -> the N4 table, `nvidia_4n` -> N4 with a note). JSON is canonical for hashing; TOML
(as 01 §13 writes), RON and YAML are accepted and normalized on load.
```
sources.json                 citation registry (key -> full citation, url, accessed date)
tech/{tsmc_n16,tsmc_n7,tsmc_n5,tsmc_n4,tsmc_n3e,tsmc_n2,asap7,sky130,gf180mcu,ihp_sg13g2}.json
tech/{dram_logic_1y,dram_1b}.json         DRAM-process logic for near-memory units (01 §9)
tech/_xcheck/{intel7,intel4}.json         optional, cross-check only
tech/aliases.json
dram/{hbm2,hbm2e,hbm3,hbm3e,hbm4,lpddr5x,gddr6,gddr7,ddr5,stacked_dram}.json   timing/energy defaults (4.8)
phy/{hbm2e,hbm3,hbm3e,ucie_std,ucie_adv,serdes_56g,serdes_112g,nvlink_c2c,pcie5,hybrid_bond}.json
package/{cowos_s,cowos_l,organic,emib,soic}.json
cost/{wafer,defect,hbm,package}.json
calib/<platform>-<date>.json              fitted parameter sets (hash enters provenance)
targets/{dies,power,fractions}.json       calibration targets with sources
```

---

## 3. Data-file format

JSON canonical (RON/YAML accepted, per 00). Every leaf numeric is a `Sourced` object; bare numbers are a
validation error (code `E-PHYS-UNSOURCED`).

```json
{
  "schema": "kiln.phys.tech/1",
  "id": "tsmc_n7",
  "kind": "foundry",
  "family": "tsmc_n7",
  "derived_from": null,
  "logic": {
    "nand2_area_um2":  {"v": 0.0439, "q": "der", "c": "M", "src": "wikichip_n7", "note": "4 / 91.2 MTr/mm2"},
    "util_std_cell":   {"v": 0.65,   "q": "asm", "c": "L", "bounds": [0.45, 0.85], "fit": true}
  },
  "sram": {
    "bitcell_hd_um2":  {"v": 0.027, "q": "pub", "c": "H", "src": "tsmc_iedm2016"}
  }
}
```

Rust sketch:
```rust
pub enum Quality { Pub, Der, Asm, Fit }
pub enum Conf { H, M, L }
pub struct Sourced<T> { pub v: T, pub q: Quality, pub c: Conf, pub src: Option<SourceKey>,
                        pub bounds: Option<(T, T)>, pub fit: bool, pub note: Option<String> }
pub struct TechNode {
    pub id: NodeId, pub kind: NodeKind /* Foundry | Predictive | OpenPdk */,
    pub vdd: VddParams, pub logic: LogicParams, pub sram: SramParams, pub rf: RegFileParams,
    pub datapath: DatapathCoeffs, pub wires: Vec<WireClass>, pub vias3d: Option<Vertical3d>,
    pub leakage: LeakageParams, pub dvfs: DvfsParams, pub flop: FlopParams, pub misc: MiscParams,
}
```
Loading validates: all required fields present, units match the field suffix, values within bounds, every `src`
resolves in `sources.json`. Load time target < 1 ms per node (files are small; parse once per process).

**Ranges from confidence.** Every `Sourced` value has a `lower / central / upper` range used by 03 §9.1's corner
evaluation, with `central = v` and the pessimistic direction declared per field in the registry (larger area,
energy or leakage coefficient is pessimistic; larger utilization or packing efficiency is optimistic):

| Source of the value | Range |
|---|---|
| Explicit `range: [lo, hi]` in the file (e.g. a published interval such as TPU v4 die < 600 mm2) | as given |
| `fit: true` after calibration (§12.3) | central `exp(±1 posterior sigma)` |
| otherwise, by confidence `c` | central `exp(±sigma_c)`, sigma `H` 0.1, `M` 0.3, `L` 0.7 (the §12.3 prior sigmas) |
| published facts used as targets or envelope (die area, TDP, stack count, node) | zero width |

Every range is clipped to `bounds`. A one-sigma band is a plausible band, not a worst case; quadrature is not
used because shared node constants move together.

Data-build tooling (Python, offline, at the edge per 00): scripts that extract cell areas from open-PDK LEF,
internal power from Liberty, and macro sizes from open SRAM LEFs, and that fit CACTI 7 sweeps into the SRAM
coefficients of §4.3. Outputs are committed JSON with `q: "der"` and the script + PDK commit as `src`.

---

## 4. Technology node tables

### 4.1 Node summary (logic, SRAM, supply)

`nand2_area_um2` is the area of one NAND2-equivalent gate (GE) at the published peak logic density
`D` (MTr/mm2), `a_GE = 4 / D` um2 (D in MTr/mm2 gives transistors per um2 = D; 4 transistors per GE).
Peak densities are best-case library densities; real designs are lower (A100: 54.2 B / 826 mm2 = 65.6 MTr/mm2
including SRAM). The gap is absorbed by `util_std_cell` (fit) and by the SRAM model, never by changing `a_GE`.

| Node | Peak logic density MTr/mm2 | a_GE um2 | HD 6T bitcell um2 | Large-macro density | Vdd nom (V) | E-scale vs N7 `s_E` |
|---|---|---|---|---|---|---|
| N16 (16FF+) | 28.9 (pub, M) [wikichip_n16] | 0.138 (der) | 0.074 (pub, M) [tsmc_iedm2014] | der from model | 0.80 (asm, M) | 2.5 (pub-mkt, M) [tsmc_n7_press] |
| N7 | 91.2 (pub, M) [wikichip_n7] | 0.0439 (der) | 0.027 (pub, H) [tsmc_iedm2016] | der from model | 0.75 (asm, M) | 1.0 |
| N5 | 138.2 (pub, M) [wikichip_n5] | 0.0289 (der) | 0.021 (pub, H) [tsmc_iedm2019] | der from model | 0.75 (asm, M) | 0.70 (pub-mkt, M) [tsmc_n5_press] |
| N4 / 4N (H100) | ~147 (asm, L: N5 x 1/0.94) [tsmc_n4_press] | 0.0272 (der) | 0.021 (asm, M) | der | 0.75 (asm, M) | 0.65 (asm, L) |
| N3E | ~197 (asm, L) [wikichip_n3] | 0.0203 (der) | 0.021 (pub, H) [wikichip_iedm2022]; N3B 0.0199 (pub, H) | 31.8 Mib/mm2 (pub, M) [wikichip_iedm2022] | 0.75 (asm, M) | 0.48 (pub-mkt, L) [tsmc_n3e_press] |
| N2 (optional row) | n/a | n/a | 0.0175 (pub, H) [tsmc_isscc2025] | 38.1 Mb/mm2 (pub, H) [tsmc_isscc2025] | 0.70 (asm, L) | 0.36 (asm, L) |
| ASAP7 (predictive) | read from lib (asm, L ~ 70-90) | from LEF (see note) | from PDK SRAM cell (asm, L: 0.027) | der | 0.70 (pub, H) [asap7_mej2016] | from Liberty (der) |
| SKY130 | n/a (GE-based) | 3.75 (pub, H: `sky130_fd_sc_hd__nand2_1` 1.38 x 2.72 um) | ~1.9 (asm, L: `sky130_fd_bd_sram__sram_sp_cell`) | 38,299 bit/mm2 at 512 B OpenRAM macro (pub, H) [orram_2607.12244] | 1.8 (pub, H) | from Liberty (der); prior 35 (asm, L) |
| GF180MCU | n/a | from LEF (asm, L ~ 10-12) | from `gf180mcu_fd_ip_sram` LEF (asm, L) | from LEF | 3.3 / 5.0 (pub, H) | from Liberty; prior 200 (asm, L) |
| IHP SG13G2 | n/a | from LEF (asm, L ~ 7-8) | from `RM_IHPSG13_*` LEF (asm, L) | from LEF | 1.2 (pub, H) | from Liberty; prior 15 (asm, L) |
| Intel 7 (x-check) | 100.8 (pub, M) [intel_bohr2017] | 0.0397 | 0.0312 (pub, H) [intel_iedm2017] | | | |
| Intel 4 (x-check) | ~2x Intel 7 HP (pub-mkt, L) | | 0.024 (pub, M) [intel_vlsi2022] | | | |

Notes:
- `s_E` is the foundry's iso-speed chip-power claim (N7 "-60% vs N16", N5 "-30% vs N7", N3E "-30..35% vs N5").
  It scales all switched-capacitance energies (datapath, SRAM, RF, wire repeaters, clock). Marketing numbers are
  `M`/`L` confidence by policy. For the open PDKs, `s_E` is replaced by direct characterization from Liberty
  `internal_power` of a reference set (INV, NAND2, DFF, FA) plus wire `C` from the tech LEF.
- ASAP7: the PDK's LEF/GDS have historically been drawn at 4x linear scale for academic-tool compatibility; the
  data-build script must detect the scale from the cell height (7.5-track cell = 270 nm at 1x) and divide areas
  by 16 when needed (confidence M; verify on the OpenROAD `asap7` release in use). Metal pitches (pub, H)
  [asap7_mej2016]: M1-M3 36 nm, M4-M5 48 nm, M6-M7 64 nm, M8-M9 80 nm; CPP 54 nm. ASAP7 absolute numbers are
  "not credible for silicon comparison" (memory note from the prior-art scan); kiln uses ASAP7 for **relative**
  sensitivity checks and for designs that will be audited via OpenROAD (§6.5), not for envelope claims.
- The open PDKs (SKY130, GF180MCU, SG13G2) are included for completeness and for small tapeout-scale studies.
  Their rows are populated by the data-build tooling from the PDK, not by hand.
- *Node transfer of `util_std_cell`.* The area group fits `util_std_cell` at N7 only. A node without its own fit
  uses `util(N7) x density_realization(node)`, never the generic 0.65 prior (which made N16 logic 17% sparser than
  the fitted N7 for no physical reason). `density_realization` is the node's realized/peak density over N7's, from
  a same-class product outside the held-out set: N16 = (GP100 15.3 B / 610 mm2 / 28.9) / (GA100 54.2 B / 826 mm2 /
  91.2) = 1.206 (der M, [nvidia_p100_wp], [nvidia_a100_wp]: peak-density ratios overstate the realized N16 -> N7
  shrink of HBM GPUs, 2.6x vs 3.16x); N7 = 1 by definition; other nodes 1.0 (asm L, no such product). The value
  is an effective utilization against the published peak and may exceed 1 on N16.
- TSMC 12FFN (GV100) uses the N16 table without a density adjustment: GV100 realizes 25.9 MTr/mm2 against
  GP100's 25.1 on 16FF+ (both [nvidia_a100_wp] table), i.e. no density gain on NVIDIA's HBM parts.
- Reticle limit (all optical nodes, pub H [asml_reticle]): 26 x 33 mm = 858 mm2 per monolithic die. High-NA EUV
  (optional flag, N2-class and beyond): 26 x 16.5 mm. Violation = error `E-PHYS-RETICLE`.

### 4.2 Datapath (MAC / dot-product) model

Anchors at 45 nm, 0.9 V from Horowitz ISSCC 2014 [horowitz_isscc2014] (pub, H for 45 nm):

| Op (45 nm) | Energy pJ | Area um2 |
|---|---|---|
| 8b int add | 0.03 | 36 |
| 32b int add | 0.1 | 137 |
| 8b int mult | 0.2 | 282 |
| 32b int mult | 3.1 | 3495 |
| 16b fp add | 0.4 | 1360 |
| 32b fp add | 0.9 | 4184 |
| 16b fp mult | 1.1 | 1640 |
| 32b fp mult | 3.7 | 7700 |
| 8 KiB SRAM 64b read | 10 | |
| 32 KiB SRAM 64b read | 20 | |
| 1 MiB SRAM 64b read | 100 | |
| DRAM 64b read | 1300-2600 | |

Derived per-bit coefficients (der, M): partial-product energy `e_pp = 0.2 pJ / 64 = 3.1 fJ/bit^2` (32b mult gives
3.0, consistent); `a_pp = 4.4 um2/bit^2` (int8) and 3.4 (int32, Booth/Wallace benefit; use 4.4 for <= 12-bit
operands, 3.4 above); adder `e_add = 3.1 fJ/bit`, `a_add = 4.3 um2/bit`.

**Structural dot-product unit (DPU) model.** Tensor units are fused dot-product units of length `L` (derived from 01 `Geometry`: `mma` k, `systolic` 1 per PE
with in-place accumulation; A100 tensor core ~ 4-8).
For operand formats `a`, `b` with significand widths incl. hidden bit `M_a`, `M_b`, exponent widths `E_a`, `E_b`,
accumulator format `acc` with width `W_acc` (fp32: 24-bit significand + 8-bit exp; int32: 32), per lane:

```
A_lane = a_pp * M_a * M_b                                     // multiplier
       + [fp] a_add * (max(E_a,E_b) + 1)                        // exponent add
       + [fp] a_mux * W_al * ceil(log2(W_al))                   // alignment shifter, W_al = M_a+M_b+g
       + a_add * (W_al + ceil(log2 L)) * (L-1)/L                // adder-tree share
       + (1/L) * ( a_add * W_acc                                // final accumulate
                 + [fp acc] a_mux * W_acc * ceil(log2 W_acc)    // normalize (LZC + shift)
                 + [fp] a_cmp * L * max(E_a,E_b) )              // max-exponent tree
       + a_ff * (b_a + b_b + W_acc / L) * r_pipe                // operand/pipeline registers
       + [mx] (1/K) * (a_add * 9)                               // E8M0 scale add per K-block (K=32)
E_lane = same expression with e_* coefficients, times activity factor alpha_dp (default 0.5, asm)
```
Guard bits `g = 3`. For integer formats the `[fp]` terms vanish and `M = bits`. Then node scaling and overhead:
```
A_mac(node) = kappa_pe * A_lane * area_scale(45 -> node)        area_scale = a_GE(node) / a_GE(45nm), a_GE(45) = 0.80 um2 (pub, H: Nangate 45 nm NAND2_X1 0.798 um2)
E_mac(node) = kappa_E_dp * E_lane * s_45->N7 * s_E(node) * (V/V_nom)^2
```
`s_45->N7 = 0.14` (asm, M: C.V^2 argument, 0.9 V -> 0.75 V gives 0.69 and capacitance per gate ~0.2;
cross-check against Stillmaker & Baas 2017 scaling equations [stillmaker_2017], to be transcribed into the
data file by the data-build step). `kappa_pe` (area) and `kappa_E_dp` (energy) are fitted (§12), priors 1.5 and
1.3, bounds [1.0, 3.0] and [1.0, 2.5]. Additional assumed coefficients at 45 nm: `a_mux = 1.2 um2/bit-stage`,
`e_mux = 1.0 fJ/bit-stage`, `a_cmp = 4.3 um2/bit`, `a_ff = 6.0 um2/bit`, `e_ff = 8 fJ/bit/cycle incl. local
clock` (all asm, L; refine from ASAP7/Liberty characterization).

Format table (significand incl. hidden bit `M`, exponent `E`); relative multiplier term `M_a*M_b`:

| Format | M | E | M^2 | Source |
|---|---|---|---|---|
| fp32 | 24 | 8 | 576 | IEEE 754 |
| tf32 | 11 | 8 | 121 | NVIDIA A100 whitepaper |
| fp16 | 11 | 5 | 121 | IEEE 754 |
| bf16 | 8 | 8 | 64 | |
| fp8 e4m3 / e5m2 | 4 / 3 | 4 / 5 | 16 / 9 | OCP FP8 spec |
| MXFP8 (e4m3/e5m2 elems) | 4 / 3 | 4 / 5 + shared E8M0 per 32 | 16 / 9 | OCP MX v1.0 [ocp_mx_v1] |
| MXFP6 e2m3 / e3m2 | 4 / 3 | 2 / 3 | 16 / 9 | OCP MX v1.0 |
| MXFP4 e2m1 | 2 | 2 | 4 | OCP MX v1.0 |
| MXINT8 | 8 (int) | shared E8M0 | 64 | OCP MX v1.0 |
| int8 / int4 | 8 / 4 | - | 64 / 16 | |

Consequence the model must reproduce qualitatively (check in tests): int8 MAC < bf16 MAC < fp16 MAC in area and
energy, fp8 ~ 0.4-0.5x bf16, MXFP4 dominated by alignment/tree not the multiplier. Published synthesis numbers
to cross-check (07): 2511.06313 (MXINT8 657, MXFP8/6 1438-1675, MXFP4 4065 GOPS/W), VMXDOTP 2603.04979
(7.2% area overhead for MX support, 843/1632 GFLOPS/W MXFP8/MXFP4 at 1 GHz 0.8 V). These are synthesis at
unknown or different nodes; they enter as **ratio** checks only (q `pub`, c `L` for absolute).

**Multi-mode units.** A template listing several `MacMode`s (03 §2.7) on one datapath:
`A = max_m A_lane(m) * (1 + mm_area * (n_modes - 1))`, `E(m) = E_lane(m) * (1 + mm_energy)`; `mm_area = 0.08`,
`mm_energy = 0.10` (asm, L, fit-eligible with bounds [0, 0.3]). A mode whose multiplier is a sub-multiple of the
widest (e.g. 2x fp8 per bf16 lane) declares `rate` > 1 in its 01 `PrecisionMode`; area is not double counted.

**Idle/padding MAC energy** (03 §2.6): `e_mac_idle = 0.1 * e_mac` (asm, L) for clock-gated lanes; units without
declared clock gating use `0.35 * e_mac` (asm, L; clock + register toggling only).

**Vector and special-function units.** fp32 FMA lane = DPU model with L = 1, `acc = fp32`, normalize per op.
SFU (exp, recip, rsqrt, sin) lane: `A = 3.0 * A_fma32`, `E = 4.0 * E_fma32` per op (asm, L). Conversion
(cast) op: `E = 0.25 * E_fma32` (asm, L). 01 may override any unit with an explicit `PowerOverride.area` / `energy_per_op`
(e.g. from synthesis), recorded with its own `source`.

Example evaluated values (N7, nominal V, before `kappa` factors; asm/der, for sanity only): bf16 x bf16 -> fp32
lane at L = 8 ~ 0.05-0.06 pJ/MAC, ~ 45 um2 (operand/pipeline registers are ~ 1/4 of it); int8 lane ~ 0.03 pJ,
~ 27 um2; fp8 ~ 0.03 pJ. A100 dense fp16 312 TFLOPS
= 1.56e14 MAC/s, so datapath-only power at these numbers is ~ 10-15 W of the 400 W board; operand delivery,
SRAM/RF, wires, control, clock, leakage and HBM dominate. This is consistent with published accelerator energy
breakdowns (MAC energy a small fraction of total), and is why the SRAM, wire, and overhead terms below must be
structural.

### 4.3 SRAM macro model

Parameters per node (`sram` block): HD bitcell `a_cell` (table 4.1), high-current cell `a_cell_hc = 1.25 a_cell`
(asm, M), bitcell aspect `w_c / h_c = 2.0` (asm, M; FinFET HD cells are wide and short), port factors
`f_port`: 1RW 1.0, 1R1W (8T) 1.4, 2RW 2.0 (asm, M), periphery parameters below.

Macro of capacity `C` bits, word width `w`, ports `p`, internal banks `B` (01 fields; defaults B = 1). Subarray
rows `R` (default min(512, power of 2 fitting); 01 may pin), columns `Cc = C / (B * n_sub * R)`, column mux
`m = max(1, Cc / w)` limited to {1,2,4,8,16}:
```
A_sub   = (R * h_c + H_p) * (Cc * w_c + W_p)          h_c * w_c = a_cell * f_port
H_p     = n_hp * h_c        // sense amps, column mux, write drivers, local IO   (n_hp prior 60, bounds [20, 150])
W_p     = n_wp * w_c        // row decoder, wordline drivers                     (n_wp prior 30, bounds [10, 100])
A_macro = n_sub_total * A_sub * (1 + g_ctrl) + A_fix   // g_ctrl prior 0.06; A_fix = control/BIST/redundancy/pins
eta     = C * a_cell / A_macro                         // reported array efficiency
```
`A_fix` per node: N7 2,000 um2 (asm, L, bounds [500, 10,000]); scales with `a_GE` across nodes.

Fitting targets for `n_hp, n_wp, A_fix, g_ctrl` (in this order, by the data-build step, not by `kiln calibrate`):
(i) large-macro published densities: N3E 31.8 Mib/mm2 at 0.021 um2 cell gives eta = 0.70; N2 38.1 Mb/mm2 at
0.0175 gives eta = 0.67 (pub, H); (ii) SKY130 OpenRAM 512 B macro at 38,299 bit/mm2 (eta ~ 0.07, periphery
dominated; pub, H) [orram_2607.12244]; (iii) CACTI 7 sweeps at 22 nm/16 nm scaled (der, M) [cacti7]; (iv) open
SRAM LEFs (GF180, SG13G2) over capacity. Sanity check the shape: R = 512, Cc = 256 gives subarray efficiency
512*256 / ((512+60)(256+30)) = 0.80; a 4 Kib macro (64 x 64) gives 0.35 before `A_fix`.

Energy per access (read; write = 1.1x read, asm M), with Horowitz anchor `E_64b(8 KiB) = 10 pJ` at 45 nm and the
observed sqrt scaling (10, 20, 100 pJ at 8 KiB, 32 KiB, 1 MiB fits `10 * sqrt(C/8KiB)` within 13%):
```
E_read_bit(bank) = e_s0 * sqrt(C_bank_bits / 65536) * f_port_E + e_htree(bank)
e_s0 = 0.156 pJ/bit at 45 nm (der, M) -> times s_45->N7 * s_E(node) * kappa_E_sram
e_htree = wire energy (§7.3) over 0.5 * sqrt(A_macro) at the macro's internal metal class
f_port_E: 1RW 1.0, 1R1W 1.15, 2RW 1.3 (asm, L)
```
`kappa_E_sram` prior 1.0, bounds [0.5, 2.0]. N7 example: 8 KiB bank ~ 0.022 pJ/bit, 256 KiB bank ~ 0.12
pJ/bit (der/asm). Banked structures pay per-bank energy plus the wire from the bank to the port, which the
floorplan supplies when the memory is placed as several macros (§6.1).

Latency: `t_acc = t_s0 + t_s1 * log2(C_bank_bits) + t_wire(0.5 sqrt(A_macro))`; N7 `t_s0 = 0.20 ns`,
`t_s1 = 0.015 ns/doubling` (asm, L; fit to CACTI 7). Reported as `latency_cycles = ceil(t_acc * f_domain)` plus
the macro's declared output register stage.

Leakage: `P_leak_sram = p_ls * C_bits * f_port` with N7 `p_ls = 1.0 mW/Mib` at 0.75 V, 85 C (asm, L; bounds
[0.2, 5]); retention/power-gated banks per 01 use `0.2 * p_ls` (asm, L).

### 4.4 Register files and flop arrays

01 chooses the implementation per storage instance: `flop`, `latch_array`, `sram_banked` (GPU-style RF).
- `sram_banked` uses §4.3 with 1R1W cells and the declared bank count.
- `flop` / `latch_array`: area per bit `a_rf = a_bit0 * (1 + k_w * (P_r + P_w - 1))^2` (cell grows linearly in
  both dimensions with port count, wire-dominated; Rixner et al. HPCA 2000 [rixner_hpca2000]); N7 `a_bit0` =
  0.25 um2 (flop with write mux, ~ 5.5 GE, asm M) and 0.12 um2 (latch array, asm L); `k_w = 0.5` (asm, M).
- Energy per bit access: `e_rf = e_rf0 * (1 + 0.3 * (P - 1)) * sqrt(entries / 64)`; N7 `e_rf0 = 4 fJ/bit`
  (asm, L), times `s_E`. Write = 1.2x read.

### 4.5 Wire classes per node

Each node declares wire classes with geometry and per-length R, C. Values below are N7 (asm, L-M; derived from
ASAP7 pitches and a size-effect resistivity model); other nodes scale by pitch with the same procedure in the
data-build step. `r = rho_eff(w) / (w * t)` with `rho_eff` including barrier and surface scattering
(Cu, 7 nm-class: 7, 5, 3, 2.2 uOhm.cm for widths 18, 32, 80, 400 nm; asm M). `c ~ 0.18-0.21 fF/um` for all
classes (asm, M: total line capacitance is nearly scale-invariant).

| Class | Layers (count) | Pitch nm | r Ohm/um | c fF/um | Use | Max unrepeated length (§7.1) |
|---|---|---|---|---|---|---|
| local | M1-M3 (3) | 36 | 108 | 0.18 | intra-cell, intra-block | ~ 30 um |
| intermediate | M4-M7 (4) | 48-64 | 20-45 | 0.19 | intra-tile, block-to-block < 0.5 mm | ~ 120 um |
| semi-global | M8-M11 (4) | 80-160 | 2.3 | 0.20 | buses, NoC links, tile-to-tile | ~ 0.8 mm |
| global | top 2 thick (2) | 720 | 0.07 | 0.22 | clock, power, long low-count signals | ~ 3 mm (inductance-limited) |

Per class: `routing_frac` = fraction of tracks available for inter-block signals after power grid and block-
internal use: local 0.0, intermediate 0.15, semi-global 0.5, global 0.15 (asm, L; fit-eligible from congestion
audits in §6.5). Over-the-block routing: semi-global and global may pass over logic blocks; over SRAM macros only
global (macros block up to their top routing layer, 01 may override).

Driver parameters per node: min inverter output resistance `R0`, input cap `C0`, parasitic ratio `gamma = Cp/C0`
(N7: `R0*C0 = 2.9 ps`, from FO4 ~ 10 ps = 3.45 R0C0; `gamma = 1`; asm M).

### 4.6 Off-die interfaces (PHY tables)

All per-PHY values: area on die (mm2), die-edge length consumed (mm, "shoreline"), bandwidth per direction,
energy per bit (PHY + link, excl. DRAM core), latency, and package requirements.

| PHY | Shoreline / unit | BW density | Energy | Area | Latency | Source / quality |
|---|---|---|---|---|---|---|
| HBM2e PHY, 1024 data bits, 3.2-3.6 Gb/s/pin | 6.0 mm/stack (asm, M) | 410-460 GB/s/stack (pub, H) [jedec_hbm2e] | PHY 0.6 pJ/b (asm, L); DRAM core+IO 3.9 pJ/b total (pub, M) [oconnor_micro2017] | 12 mm2/stack incl. controller at N7 (asm, L) | ~ 100 ns load-to-use (03 owns) | stack 7.75 x 11.87 mm (pub, M) |
| HBM3 PHY, 1024 bits, 6.4 Gb/s/pin | 6.5 mm/stack (asm, L) | 819 GB/s/stack (pub, H) [jedec_hbm3] | total 3.0-3.5 pJ/b (asm, L) | 14 mm2 at N5 (asm, L) | | stack ~ 11 x 11 mm (pub, M) [chipscalereview_2023]; microbump pitch 96 um (pub, M) |
| HBM3e, up to 9.6 Gb/s/pin | 6.5 mm (asm, L) | 1.2 TB/s/stack (pub, M) [synopsys_hbm3_phy] | 2.5-3.0 pJ/b (asm, L) | 15 mm2 (asm, L) | | |
| UCIe standard pkg (x16, 100-130 um bump) | per module | 28-224 GB/s/mm (pub, H) [ucie_1.0] | 0.5 pJ/b (pub, H) | der from module | < 2 ns (pub, H) | |
| UCIe advanced pkg (x64, 25-55 um bump) | per module | 165-1317 GB/s/mm (pub, H) [ucie_1.0] | 0.25 pJ/b (pub, H) | der | < 2 ns (pub, H) | |
| NVLink-C2C-class D2D | | 900 GB/s total (pub, H) | 1.3 pJ/b (pub, M) [nvidia_gh200] | asm | asm | |
| 112G PAM4 LR SerDes lane | 0.4 mm/lane (asm, L) | 14 GB/s/lane/dir | 5 pJ/b (asm, L) | 0.5 mm2/lane incl. PLL share (asm, L) | ~ 20-50 ns incl. FEC (asm, L) | ISSCC 112G papers to transcribe |
| 56G PAM4 SerDes lane (A100 NVLink3 class: 50 Gb/s) | 0.35 mm/lane (asm, L) | 6.25 GB/s/lane/dir | 4 pJ/b (asm, L) | 0.35 mm2/lane (asm, L) | asm | NVIDIA A100 whitepaper for lane count |
| PCIe Gen4/5 x16 | 6 mm (asm, L) | 32/64 GB/s/dir (pub, H) | 5-6 pJ/b (asm, L) | 8 mm2 (asm, L) | ~ 500 ns (asm) | |
| Hybrid bond (SoIC-class, 9 um pitch) | area, not edge | 1/p^2 = 12,300 /mm2 (der, M) | 0.05 pJ/b (asm, L) | bond pad area within block | < 1 ns | [tsmc_soic] |
| Microbump 3D (40 um pitch) | area | 625 /mm2 | 0.15 pJ/b (asm, L) | | | |

DRAM energy split: `E_dram_core` (in the stack, counted in board power, not die power) and `E_phy` (on die).
DRAM timing, refresh and efficiency belong to 03/01; kiln-phys supplies only energy, PHY area and shoreline.

### 4.7 Leakage, V/f, clock

**V/f curve** (alpha-power law, Sakurai-Newton 1990 [sakurai_1990]): `f(V) = k_f * (V - V_th)^alpha / V`,
per clock domain, defined on `[V_min, V_max]`. N7 priors: `V_th = 0.30 V`, `alpha = 1.3`, `V_min = 0.60`,
`V_max = 0.95` (asm, L). `k_f` is set so `f(V_at_boost) = f_boost` (01 `ClockDomain.freq` and optional `voltage`; default
`V_at_boost = V_max`). If 01 lists explicit `vf` points they are used verbatim (piecewise-linear, monotone check)
and the alpha-power law is only used to extrapolate below the lowest point down to `base`; 01's `base` is the
floor below which 03 raises a simulation error. The curve is tabulated to a 32-point `VfTable` for 03's
`solve_clock`. Calibration fits `V_th`, `alpha` per platform from (clock, power) telemetry (§12).

**Leakage**: `P_static(V, T) = sum_blocks P_leak0_block * (V/V_nom) * exp(k_V (V - V_nom)) * exp(k_T (T - T0))`
with `T0 = 85 C`. Block `P_leak0`: logic `p_ll * A_logic`, N7 `p_ll = 0.04 W/mm2` at 85 C (asm, L; bounds
[0.01, 0.15]); SRAM per §4.3; PHYs fixed per PHY table (asm). `k_V = 3 /V`, `k_T = 0.02 /K` (asm, M: leakage
roughly doubles per ~35 K). Power gating (01 flag per block) multiplies idle-block leakage by `0.05` when 03
reports the block idle for a window > `t_pg_min` (asm 10 us).

**Clock tree**: `P_clk = (c_clk_flop * N_flops + c_clk_area * A_clocked) * V^2 * f * (1 - g_cg)`; N7
`c_clk_flop = 0.6 fF`, `c_clk_area = 50 pF/mm2` (asm, L); `g_cg` = clock-gated fraction, from activity (idle
units gated if 01 declares fine-grain gating). `N_flops` comes from the block models (pipeline registers,
RF flops, router buffers). Fitted scalar `kappa_clk`, prior 1.0, bounds [0.3, 3.0].

**Board/package overheads**: VR efficiency `eta_vr = 0.90` (asm, M) applied to core rails; board fixed power
`P_board` (fans excluded, misc ICs) prior 10 W for GPU modules (asm, L); HBM power counted on board. 01's
`PowerDomain.idle` (fans, VRM loss), when given, replaces `P_board_fixed`; `PowerDomain.thermal` (`tj_max`,
`theta_ja`) replaces the §9 defaults. 01 declares
whether the envelope cap is `die`, `package` or `board` (`PowerCap.level`), and kiln-phys reports all three
(`P_chip` below is the die-level power).

### 4.8 DRAM kind defaults (01 §8.1 `timing: None`, `footprint: None`)

01 delegates DRAM timing defaults, stack footprints and per-kind energies to this section's data files
(`dram/<kind>.json`). Values are typical JESD235/JESD238 figures for the fastest standard speed bin and must be
transcribed per bin in the data build (q `pub` once transcribed; until then `asm`, c `M`).

| Kind | Channels x PC | tRCD / tRP / tCL (ns) | tRAS (ns) | tRFC (ns) / tREFI (us) | tCCD_S / tCCD_L (tCK) | Energy, pJ/bit (core+IO) | Activate (nJ) | Footprint (mm) |
|---|---|---|---|---|---|---|---|---|
| HBM2 / HBM2e | 8 x 2 | 14 / 14 / 14 | 33 | 350 / 3.9 | 2 / 4 | 3.9 (pub, M) [oconnor_micro2017] | 0.9 (asm, L) | 7.75 x 11.87 (pub, M) |
| HBM3 | 16 x 2 | 14 / 14 / 14 | 33 | 350 / 3.9 | 2 / 4 | 3.2 (asm, L) | 0.8 (asm, L) | 11 x 11 (pub, M) |
| HBM3e | 16 x 2 | 14 / 14 / 14 | 33 | 350 / 3.9 | 2 / 4 | 2.8 (asm, L) | 0.8 (asm, L) | 11 x 11 (asm, M) |
| HBM4 | 32 x 2 | asm | asm | asm | asm | 2.5 (asm, L) | asm | 11 x 11 (asm, L) |
| LPDDR5x / GDDR6 / GDDR7 / DDR5 | per JEDEC | per JEDEC | | | | 4-8 (asm, L) | | package per kind |
| StackedDram (hybrid-bonded DRAM-on-logic) | from bond area | 01 or asm | | | | 0.5-1.0 (asm, L) incl. bond | asm | = overlap area |

The **achievable bandwidth fraction** per kind that 01 §8.1 assigns to "a 04 calibration constant" is the same
parameter 03 §9 calls `eta_res(dram_kind)`: it is stored in this crate's calibration file, fitted by 03's
procedure, and applied once in 03 §5.3 (single owner of the term: 03; single storage location: here).

Near-memory defaults for 01 §9 when not given: `internal_bandwidth` per bank group = `row_bytes / tCCD_L` x
bank groups active in parallel (der); `command_latency` = 20 ns (asm, L); logic on a DRAM-process die uses
`dram_logic_1y` (a_GE = 10x N16, energy 3x N16 at 1.1 V; asm, L, citing the general observation that DRAM-process
logic has 3-4 metal layers and slow transistors; refine from published HBM-PIM / AiM area reports).

### 4.9 Dataflow-dependent operand storage (01 §5.2)

A matrix unit supporting a set of stationarity modes pays, per PE, one holding register per supported mode:
weight-stationary `b_w` bits, output-stationary `W_acc` bits, input-stationary `b_a` bits (plus a bypass mux
`a_mux * width` per extra mode). `any` prices all three. These registers enter `A_lane` via the `a_ff` term and
their clock energy via `e_ff`, so declaring more flexibility than needed costs area and power in the search.

### 4.10 Protocol encoding efficiency defaults (01 link `encoding_efficiency: None`)

PCIe Gen3-5 128b/130b = 0.985 (pub, H); PCIe Gen6 FLIT 242/256 = 0.945 (pub, H); UCIe raw = 1.0 with flit
overhead per mode (asm 0.94, M); NVLink/ICI-class = 0.94 (asm, L); on-die wires 1.0. Protocol efficiency
beyond encoding (packet headers, credits, retries) is 03's `eta_link(kind)` and is not applied here.

### 4.11 CIM macro model (01 §9.3)

A CIM unit is priced as its bit-cell array plus a compute periphery that scales with the bits it processes per
cycle, so every 01 `CimSpec` field that raises throughput raises area and energy in proportion. With `R = rows`,
`P = parallel_rows`, `C = cols`, `b_c = cell_bits`, `b_i = input_bits_per_cycle`, `S = weight_sets`:

```
Array        : §4.3 subarray model with R rows, C * S columns, cell area a_cell * f_cim * max(1, b_c / 2)
               f_cim = 1.4 (8T decoupled read; 1R1W, asm M); multi-bit cells (b_c > 1) are analog/eNVM-style
               and pay the max(...) term (asm, L). Weight write energy/bandwidth: §4.3 write path.
Weight mux   : S > 1: a_mux * C * b_c * ceil(log2 S) per macro (selects the active set)
Bit products : n_pp = P * C * b_c * b_i  1b x 1b products per cycle
               A_pp = a_pp * n_pp,  E_pp/cycle = alpha_dp * e_pp * n_pp      (§4.2 coefficients, M = 1 bit)
Digital      : per column an adder tree over P inputs of (b_c + b_i) bits, then shift-and-add across the
               cells_per_weight columns and the ceil(in_bits / b_i) input cycles into W_acc:
               A_tree = a_add * C * (P - 1) * (b_c + b_i + ceil(log2 P) / 2)
               A_sacc = (a_add + a_ff) * (C / cells_per_weight_min) * W_acc
               E/cycle = alpha_dp * (e_add * C * (P - 1) * (b_c + b_i + ceil(log2 P) / 2) + e_ff * C * W_acc / cells_per_weight_min)
Analog       : the tree is replaced by one ADC conversion per column per cycle, plus a b_i-bit DAC per row if b_i > 1:
               A_adc(adc_bits) = a_adc0 * 2^(adc_bits/2),  E_adc = e_adc0 * 2^adc_bits  (Walden FOM form; a_adc0,
               e_adc0 per node, asm L); A = C / adc_share * A_adc + R * [b_i > 1] * A_dac(b_i); adc_share default 1.
               Shift-and-add as digital. Wordline drive: e_wl * P per cycle.
Leakage      : array per §4.3 (C_bits = R * C * b_c * S); periphery per §4.7 on its gate count.
```

`e_mac(mode)` for 03 is `E/cycle / MACs_per_cycle(mode)` with 01 §9.3's derivation, so a bit-serial macro pays
`ceil(in_bits / b_i)` cycles of periphery energy per MAC and a fully parallel one pays a `b_i`-times wider tree
once. Throughput per cycle and periphery area both scale with `P * C * b_c * b_i`: a fully parallel CIM costs
proportionally more area and energy per cycle than a bit-serial one of the same array. `cells_per_weight_min`
uses the narrowest weight among the unit's modes. Unpriced: an explicit `weight_write` (01 E-IR-1101 under
`search`); accuracy effects of `adc_bits` (01 open question 12).

---

## 5. Block library (what kiln-phys computes per IR component)

`characterize(design: &ExpandedHw, tech) -> Characterized` (pure, cached by design hash). For each component
template (not each instance; arrays share), compute:

```rust
pub struct BlockModel {
    pub template: TemplateId,
    pub area_mm2: f64,                       // incl. kappa factors and std-cell utilization
    pub area_parts: Vec<(AreaPart, f64)>,    // datapath, sram, rf, control, router, phy, misc
    pub shape: ShapeConstraint,              // hard (w,h) | soft { aspect_min, aspect_max } | grid(array)
    pub n_flops: u64, pub sram_bits: u64,
    pub leak_w_nom: f64,                     // at V_nom, 85 C
    pub energies: BlockEnergies,             // e_mac per MacMode, e_mac_idle, e_vec per class, e_read/e_write per
                                             // storage instance (J per byte at access width), e_router_flit, ...
    pub latencies: BlockLatencies,           // storage latency_cycles, router pipeline, PHY latency
    pub power_density_hint_w_mm2: f64,       // at full activity, for placement/thermal
}
pub enum ShapeConstraint { Hard { w_um: f64, h_um: f64, rotatable: bool },
                           Soft { aspect_min: f64, aspect_max: f64 },
                           Grid { rows: u32, cols: u32, cell: Box<ShapeConstraint>, pitch_overhead: f64 } }
```

Area per component kind:

| IR component (01) | Area formula |
|---|---|
| Compute unit (matrix) | `n_lanes * A_mac(mode set) / util_std_cell + A_ctrl_unit` |
| Compute unit (cim) | §4.11: bit-cell array (§4.3) + weight-set mux + bit-product, adder-tree/ADC and shift-accumulate periphery scaled by `parallel_rows * cols * cell_bits * input_bits_per_cycle`, + `A_ctrl_unit` |
| Compute unit (vector/SFU) | `lanes * A_lane(kind) / util_std_cell + A_ctrl_unit` |
| Control per unit (`A_ctrl_unit`) | `kappa_ctrl * (A_datapath)` for fixed-function; for programmable cores `GE_ctrl * a_GE / util` with 01-declared `ctrl_ge` (e.g. GPU SM scheduler/LSU/TEX) else `kappa_ctrl` prior 0.15, bounds [0.03, 0.6] |
| SIMT sub-core control (host-launched designs: a cluster whose matrix/vector units feed from a register file inside it) | `(simt_ctrl_ge + simt_operand_ge_per_bit * B_op) * a_GE / util` per sub-core instance, harvested ones included; `B_op` = operand bits per cycle its matrix units consume at their widest mode (sum over units of `ops_per_cycle * (bits_a + bits_b)`). The fixed part is scheduling/dispatch/collectors/LSU/TEX; the second is matrix-operand delivery (register-bank reads, operand staging, collector-to-tensor-core network). GP100 (no tensor cores) and GA100 separate them |
| Storage | §4.3 / §4.4 per instance, macros counted separately (they are placed) |
| NoC router | §7.4 |
| Near-memory compute site (in logic die of a DRAM stack or on DRAM die) | uses the node of the host die it sits on; DRAM-process logic uses `dram_logic` node entry (asm, L: 10x a_GE of N16, 3x energy) since DRAM-process logic is far less dense |
| PHY | §4.6 table, per instance, pinned to edge |
| Misc chip-level (fuses, PLLs, management CPU, PCIe controller, test) | `A_misc` per chip, prior 15 mm2 at N7 (asm, L), scales by `a_GE` |
| Redundancy / harvesting | instances listed in 01 `disabled` (harvesting, 01 §12) (e.g. GA100: 128 SMs, 108 enabled) are placed and counted in area; leakage per 01 §12 `harvest_power` (`gated` default: 0 W, `leak`: full leakage); absent from performance |

---

## 6. Floorplan model and placer

### 6.1 Floorplan entities

```rust
pub struct Floorplan {
    pub package: PackageFp,                 // substrate/interposer outline, dies, HBM stacks, bridges
    pub dies: Vec<DieFp>,                   // one per die (chiplet)
    pub links: Vec<PhysLink>,               // every link instance with geometry (from 7.2)
    pub hash: Hash, pub tier: PlaceTier, pub stats: PlaceStats,
}
pub struct DieFp {
    pub id: DieId, pub node: NodeId, pub outline_um: Rect,       // w, h; origin lower-left
    pub layers3d: Vec<Layer3d>,                                  // bottom to top; 1 for 2D dies
    pub edges: [EdgeBudget; 4],                                  // S, E, N, W: length, assigned PHYs, used
    pub blocks: Vec<PlacedBlock>, pub keepouts: Vec<Rect>,
    pub seal_ring_um: f64,                                       // default 50 (asm)
}
pub struct PlacedBlock { pub inst: BlockIx, pub layer: u8, pub rect_um: Rect, pub orient: Orient,
                         pub pinned: bool, pub array_slot: Option<(ArrayIx, u32, u32)> }
pub struct Layer3d { pub die_ref: DieId, pub bond: BondKind /* Hybrid{pitch_um} | Microbump{pitch_um} */,
                     pub thickness_um: f64, pub face: Face }
pub struct EdgeBudget { pub len_um: f64, pub reserved: Vec<(PhyIx, Interval)>, pub corner_keepout_um: f64 }
```

Inputs from 01 (per die): node, optional fixed outline (`Outline::Fixed { w, h }`, um) or aspect bounds (default
[0.75, 1.33]), target utilization `u_place` (default from calibration), pinned blocks (01 `Placement`: `pinned`,
`region`, `edge`, `array`, `site`), 3D layer assignment per block (v0:
declared, not inferred), PHY instances with their edge preference, keep-outs, power-delivery region overhead.

**Array collapsing.** An array of `N` identical blocks with a layout rule (01 `Layout`: `linear`, `grid`,
`ring`, `explicit`) is placed as one soft macro whose candidate shapes are the factorizations `r x c >= N` (with
`r*c - N <= 0.05 N` dummy slots allowed) times the cell's shape options. Instances get slot coordinates
inside; intra-array link lengths follow from the cell pitch. This keeps the placer's problem at tens to low
thousands of macros even for designs with 10^4-10^5 tiles.

### 6.2 Die sizing and whitespace

```
A_blocks  = sum placed block areas (incl. spares, PHYs, routers)
A_channel = routing channel area from §7.5 congestion (0 if over-the-block capacity suffices)
A_core    = (A_blocks + A_channel) / u_place                 // u_place = block-level packing utilization (fit; prior 0.85, bounds [0.70, 0.95])
die w,h   = solve: w*h >= A_core + seal ring + edge PHY depth effects; aspect within bounds;
            sum over edges of PHY shoreline (incl. corner keep-outs, default 1 mm/corner asm) fits;
            w,h <= reticle (26 x 33 mm) else E-PHYS-RETICLE
```
**Area vs geometry (06 P6).** The roll-up above counts every block. The geometry links are measured on (§7.2) is
built from *live* blocks only: enabled compute units, everything enabled they reach over channels, and the
containers holding those. Harvested instances, blocks without channels (misc area) and memories nothing connects
to count in area, leakage and the outline checks, but take no place in the geometry (an optimal placer parks dead
area where it lengthens nothing), so adding them leaves every link bit-identical. The die arrangement (macro
areas, Tier A, nested layout, ordering by a structural signature of the arranged content) further leaves out
compute-unit subtrees and the SIMT control they bring; the arranged die is then stretched uniformly by
`sqrt(env(live) / env(arranged))` (auto outlines; a fixed outline already holds them). A unit no op can use
therefore only stretches distances. Positions inside a cluster use §7.2's cluster rule.

If shoreline needs exceed the perimeter of the area-derived die, the die grows (shoreline-limited) and the
result reports `shoreline_limited = true` with the binding edge. If the declared fixed outline is too small,
error `E-PHYS-AREA-OVERFLOW` with the deficit in mm2 and the top-3 area consumers.

### 6.3 Package and 2.5D/3D

- Interposer/substrate outline: dies + HBM stacks + spacing (default 0.1 mm die-to-die, 0.5 mm die-to-HBM;
  asm), checked against package limits: CoWoS-S <= ~3.3 x reticle (~2,800 mm2) (pub-mkt, M) [tsmc_cowos],
  CoWoS-L larger (asm), organic substrate per 01.
- Each HBM stack must sit adjacent to its PHY's edge interval; a stack footprint (e.g. 11 mm wide HBM3) may
  exceed its PHY shoreline (6.5 mm) and may overhang die corners, but stacks on the same side cannot overlap:
  check `sum stack widths + gaps <= die edge + 2 * overhang_max` (overhang_max 3 mm asm). This is the
  constraint that limits A100/H100-class dies to 3 stacks per long edge.
- Die-to-die links: PHYs on facing edges; package trace length = gap + PHY depth; latency/energy from the PHY
  table (2.5D) rather than the on-die wire model.
- 3D: blocks on different layers overlap in x,y; vertical links use the bond kind's density (pads/mm2 times
  the overlap area of the two blocks' footprints, times `bond_signal_frac` 0.5 asm) to bound bandwidth, and
  `e_bit` from the table. Thermal model stacks layers (§9).

### 6.4 Placer, Tier A (fast, deterministic, ms-level)

Objective (both tiers):
```
Phi = sum_links w_l * WL_l                        // WL = HPWL of the link's endpoints (pin = block center unless 01 gives port side)
    + lambda_th * sum_bins (q_bin - q_mean)^2     // power-density spreading, q = W/mm2 from power_density_hint (smoothed)
    + lambda_ov * overlap + lambda_bd * boundary
w_l = traffic_bytes_per_s(l) * (e_bit_mm + lambda_lat * crit(l))   // traffic from 03; crit = 1 if the link is on a reported critical path
```
Traffic source (chicken-and-egg resolution): (1) first placement uses the design's **declared** link bandwidths
as traffic proxy (`w_l = bw_l`); (2) 03 maps the workload and returns a `TrafficMatrix` (bytes per link
endpoint pair, summed over all layers of the workload weighted by repetition); (3) re-place with real traffic;
(4) re-derive links; 03 re-maps once. Tier A stops after (4) (two placements); if makespan changes by more than
2% between the two mappings, the result carries `place_unconverged` (warning in the result, not an error).
Tier B iterates to a fixed point (max 4 rounds).

Algorithm (Tier A): **recursive min-cut bisection into a slicing tree, then shape-curve sizing.**
1. Assign PHYs to edges (deterministic greedy: PHYs sorted by shoreline desc then id; each goes to the edge with
   the most remaining length among those allowed by 01; ties by S, E, N, W order). PHY blocks are pinned.
2. Build the block graph (macros after array collapsing; pinned PHYs as fixed terminals at their edge).
3. Recursive bisection: at each node, split blocks into two sets balancing area to the ratio of the region
   (tolerance 10%), minimizing weighted cut + terminal-propagation pull toward pinned blocks. Initial split:
   Fiedler vector of the weighted Laplacian (deterministic Lanczos, fixed 40 iterations, start vector = sorted
   index ramp, sign fixed so the lowest-index block is negative); then Fiduccia-Mattheyses refinement
   [fm_1982], fixed 4 passes, ties by block index. Cut direction = perpendicular to the region's longer side.
4. Shape curves (Stockmeyer [stockmeyer_1983]): bottom-up combine soft-block shape curves (aspect samples: 7 per
   soft block, all factorizations for arrays), top-down pick the minimum-area shape fitting the die aspect.
5. Thermal spreading: if any leaf's power density > `1.5 * q_mean`, swap with the coolest sibling subtree of
   similar area if it does not increase `Phi` by more than 5% (one pass).
Complexity O(n log n) for n macros; target <= 2 ms at n = 2,000, <= 0.2 ms at n = 100.

### 6.5 Placer, Tier B (quality) and audit import

1. Global: quadratic placement with the bound-to-bound net model (Spindler et al., Kraftwerk2 [kraftwerk2_2008])
   solved by preconditioned CG (fixed tolerance 1e-6, max 200 iterations, Jacobi preconditioner), plus an
   electrostatic density penalty in the style of ePlace [eplace_2015] (FFT-based on a 64 x 64 bin grid,
   Nesterov steps, fixed 300 iterations with deterministic step schedule), plus the thermal term.
2. Legalization: macro legalization by constraint-graph compaction (sequence pair derived from global
   positions [murata_1995]), then left/bottom compaction honoring pinned blocks.
3. Refinement: simulated annealing on the sequence pair with seeded RNG (seed from design hash), fixed
   20,000 moves, temperature schedule fixed; accept only legal states.
4. Congestion: RUDY map [rudy_2007] per wire class (§7.5); overflow bins inflate their blocks' spacing and
   re-run steps 2-3 once.
Target <= 2 s at n = 2,000 macros.

**OpenROAD audit path** (elites only, 06): `kiln-phys export-def` writes a block-level DEF + LEF abstracts
(macros with pins at block edges); `kiln-phys import-def` accepts an OpenROAD (RePlAce/macro placer, or full
P&R with ASAP7) placement, overrides block positions, and recomputes links. Imported floorplans are flagged
`placement_source = openroad` in provenance. Post-route wire lengths (from SPEF/ODB reports) may replace the
HPWL-based lengths per link if provided.

---

## 7. Wire and on-chip interconnect model

### 7.1 Repeated wire

Elmore stage delay for repeater size `h` (multiple of min inverter) and segment length `s` on a class with `r, c`:
```
t_seg / s = 0.69 R0 (C0 + Cp) / s + 0.69 R0 c / h + 0.38 r c s + 0.69 r h C0
delay-optimal:  h* = sqrt(R0 c / (r C0)),   s* = sqrt(0.69 R0 (C0+Cp) / (0.38 r c))
                tau* = sqrt(R0 C0 r c) * (2 sqrt(0.262 (1+gamma)) + 1.38)   (= 2.83 sqrt(R0 C0 r c) at gamma = 1)
```
(Bakoglu [bakoglu_1990]; Ho, Mai, Horowitz [ho_wires_2001]). Repeater capacitance per length at the delay
optimum `= c * sqrt(0.38/0.69) * sqrt(1+gamma) ~ 1.05 c`. kiln uses **energy-aware sizing** by default:
`h = k_h h*`, `s = k_s s*` with `k_h = 0.5`, `k_s = 1.5` (Banerjee & Mehrotra [banerjee_2002]; asm M), giving
~ +10-15% delay and repeater cap ~ 0.35 c. Implementation evaluates the closed forms with these factors (no
iterative solve). Per-link override 01 `OnDie.sizing: delay | energy | custom{k_h, k_s}`.

Resulting N7 values (der from §4.5 asm inputs; c L): semi-global ~ 105 ps/mm delay-optimal, ~ 120 ps/mm
energy-aware; intermediate ~ 300 ps/mm; global ~ 20-30 ps/mm (time-of-flight floor 6.7 ps/mm in SiO2-class
dielectric applies: `tau >= sqrt(eps_r)/c0`).

Max unrepeated length: `L_unrep = s` of the chosen sizing (a wire longer than one segment gets repeaters; this
is a design rule, reported per class).

### 7.2 Link derivation from placement

For each IR link instance (01: endpoints, `width_bits`, direction, clock domain, `OnDie.metal` class preference,
optional `pipelined`, `phys` kind):
```
L_um        = Manhattan distance between endpoint port positions (block-edge port nearest the other end;
              block center if unspecified) + detour factor (1.0 Tier A over-the-block; RUDY-derived in Tier B)
class       = cheapest class (by energy) that meets the clock budget per stage and has track capacity (§7.5)
t_wire      = tau_class(sizing) * L
T_avail     = (1 - f_logic) / f_domain - t_setup_clkq      // f_logic = fraction of cycle used by endpoint logic, default 0.3 (asm)
n_stages    = max(0, ceil(t_wire / T_avail) - 1)            // pipeline registers inserted
latency_cyc = 1 + n_stages + endpoint_latency_cycles        // endpoint: SRAM, router, PHY latencies from §4
bw_Bps      = width_bits * f_domain / 8                     // raw wire capacity; protocol efficiency is 03's eta_link
e_J_per_B   = 8 * E_bit                                     // E_bit below
```
Wire energy per bit (SI): `E_bit = a_t * (c_w + c_rep) * L * V^2 + n_stages * e_ff_bit + e_endpoint_bit`, where
`a_t` = energy-weighted transition factor: for random data, toggle probability 0.5 and energy `C V^2 / 2` per
transition gives `a_t = 0.25` (der, H as physics; data-dependent activity may later come from 02 tensors, open
question). N7 semi-global with energy-aware repeaters: `c_w + c_rep ~ 0.27 pF/mm`, `V = 0.75` gives
`~ 0.038 pJ/bit/mm` plus flops; this sits at the low end of the commonly quoted ~ 0.05-0.15 pJ/bit/mm for
recent nodes (asm, L) and is multiplied by the fitted `kappa_E_wire` (prior 1.5, bounds [0.8, 4]) that
represents shielding, coupling (Miller), and repeater short-circuit power not in the formula.

**Inside a cluster** (the lowest common ancestor of the two endpoints is a cluster container below the die),
`L_um = 0.5 * sqrt(A_live(cluster))`, half the side of the cluster's live area: the slicing layout inside a
cluster is not a placement, and this length is monotone in added area, which 06 P6 needs (a unit added to a
cluster may lengthen its wires, never shorten them). Links whose endpoints meet at the die or package use the
placed positions of the stretched arrangement (§6.2).

Bit-serial or low-swing links (01 option `swing: low`, `V_swing = 0.2 V`): `E = a_t (c_w) V_swing V_dd L +
e_rx` with receiver `e_rx = 20 fJ/bit` (asm, L) [ho_wires_2001].

Output to 03 per directional link instance:
```rust
pub struct LinkCost {
    pub link: LinkIx, pub class: WireClassId, pub length_um: f64,
    pub latency_cycles: u32, pub latency_s: f64, pub pipeline_stages: u32,
    pub bw_bytes_per_s: f64,                // raw wire capacity at domain clock (before protocol efficiency)
    pub e_j_per_byte: f64,                  // dynamic, at V_nom; 03 rescales by (V/V_nom)^2 under DVFS
    pub e_j_per_byte_idle_clock: f64,       // pipeline flop clock energy per cycle when idle (if not gated)
    pub domain: ClockDomainId, pub source: LinkSource,   // Wire | NocHop | Phy(kind) | Bond3d | Package
}
```

### 7.3 Memory access cost table (to 03)

Per storage instance, as 03's `MemLevel` energy/latency fields (03 §2.2): `e_read_J_per_B`, `e_write_J_per_B`
at the declared access width (§4.3/4.4, including in-macro H-tree), `latency_cycles` (macro + output stage),
and per-bank data. Accesses that traverse a placed path (e.g. SM to L2 slice) are not folded in here; they are
links with their own `LinkCost`, so 03 charges them per route (avoids double counting; invariant: an energy
term is charged by exactly one resource).

### 7.4 NoC routers

Router with `P` ports (radix), `V` VCs per port, `B` flit buffers per VC, flit width `W` bits, `n_pipe` stages
(ORION 2.0 [orion2_2009] and DSENT [dsent_2012] structure; coefficients asm L until fit to DSENT runs):
```
A_buf   = P * V * B * W * a_buf_bit            (flop/latch FIFO; a_buf_bit = a_rf for 1R1W)
A_xbar  = (P * W * p_x)^2                       (matrix crossbar, wire-limited; p_x = semi-global pitch * 2 for shielding)
A_alloc = a_alloc * (P * V)^2 * a_GE            (separable VC + switch allocators; a_alloc = 12 GE asm L)
A_router = (A_buf + A_alloc) / util_std_cell + A_xbar
E_flit  = W * (e_buf_wr + e_buf_rd) + E_xbar(W, P) + e_arb(P, V)
E_xbar  = W * a_t * c_semi * (P * W * p_x) * 2 * V^2        (input line + output line length)
t_cycle_min = (t_alloc0 + t_alloc1 * log2(P * V)) FO4       (t_alloc0 = 8, t_alloc1 = 3 FO4; asm L)
```
If `t_cycle_min > 1/f_domain`, the router gets an extra pipeline stage (reported) or 01's declared `n_pipe` is
rejected with `E-PHYS-ROUTER-TIMING`. Router latency = `n_pipe` cycles; link latency from §7.2. Router leakage =
logic + buffer leakage. Both router crossbar ports and links are resources for 03 (03 §3.5).

### 7.5 Wiring capacity, bisection, congestion

Track supply per class per um of cut: `n_tracks/um = n_layers * routing_frac / pitch` (one direction per
layer, alternate-direction layering assumed so half the layers serve each direction).
- **Tier A bisection check:** for each slicing-tree cut (all internal nodes) the total `width_bits` (x 1.1 for
  shielding/spares, asm) of links crossing the cut must be <= cut length x track supply summed over eligible
  classes. Violation: grow the region (channel area added to `A_channel`) and report `wire_limited` with the
  cut location; error only if die exceeds reticle.
- **Tier B RUDY:** per link, spread `width_bits * (w+h)/(w*h)` demand over its bounding box; per bin, demand vs
  supply; overflow > 1.0 in any bin triggers spacing inflation (§6.5 step 4), persistent overflow is reported
  with a heat map for the visualizer (05).
- Achievable link bandwidth is never above what its allocated tracks support; links that share a channel do
  not share bandwidth (dedicated wires), only space, so congestion converts to area or longer detours, not to
  contention. Contention exists only at routers, shared buses, and ports, which 03 models.

### 7.6 How these feed 03's resources

| 03 resource | kiln-phys provides |
|---|---|
| `Link(l, dir)` (wires, NoC links, PHY links, inter-chip lanes) | `LinkCost` (bw, latency, energy) |
| Router crossbar port | per-port bw (`W * f / 8`), latency `n_pipe`, `E_flit / (W/8)` per byte |
| Memory port / level | `MemLevel` energies and latency (§7.3) |
| DRAM channel | PHY energy per byte, DRAM core energy per byte (stack), activate energy, timing defaults (4.8) |
| Compute unit | `e_mac` per `MacMode`, `e_mac_idle`, `e_vec`, `e_conv` |
| Clock domain | `VfTable`, `kappa_E_*`-scaled dynamic energy (03's former `C_eff`, 00 decision 6), `P_static(T, V)`, `P_cap`, thermal limit |

---

## 8. Power model

kiln-phys provides pure functions; the clock solve loop lives in kiln-sim (03 §4.5) and calls them.

**Inputs from 03** (`ActivityReport`, one per window: Tier A per phase, Tier B per `tau_pm` window):
```rust
pub struct ActivityReport {
    pub window_s: f64, pub f_by_domain: Vec<(ClockDomainId, f64)>, pub v_by_domain: Vec<(ClockDomainId, f64)>,
    pub macs: Vec<(UnitIx, MacModeIx, u64 /*issued*/, u64 /*padding*/)>,
    pub vec_ops: Vec<(UnitIx, VecClass, u64)>,
    pub mem: Vec<(StorageIx, u64 /*read B*/, u64 /*write B*/)>,
    pub link_bytes: Vec<(LinkIx, u64)>,           // kiln-phys multiplies by per-link energy (length already inside)
    pub router_flits: Vec<(RouterIx, u64)>,
    pub dram: Vec<(DramIx, u64 /*rd B*/, u64 /*wr B*/, u64 /*activates*/)>,
    pub io_bytes: Vec<(PhyIx, u64)>,
    pub busy_cycles: Vec<(UnitIx, u64)>,          // for clock gating and power gating
    pub temperature_c: Option<Vec<f64>>,          // per die from the previous thermal solve (Tier B)
}
```
**Energy**:
```
E_dyn  = sum macs * e_mac(mode) + padding * e_mac_idle + sum vec * e_vec + sum mem * e_rw
       + sum link_bytes * e_link + sum flits * E_flit + E_clk(window) + io_bytes * e_phy
E_dram = sum dram bytes * e_dram_core + activates * e_act            (board-level)
E_core_domain scales with (V/V_nom)^2; PHY/DRAM/IO energies are clock-independent (03 §4.5 split)
P_chip  = E_dyn / window + P_static(V, T)
P_pkg   = P_chip + P_dram_stack(E_dram/window + stack background)
P_board = P_pkg / eta_vr + P_board_fixed
```
The `C_eff(domain)` that 03 §9 calibrates is not a separate free scalar: it is `E_dyn_core / (V^2 * cycles)`
derived from the structural terms, and the calibrated quantities are the `kappa_E_*` factors with bounds. This
keeps 03's "one physical role per parameter" rule.

**Power cap and DVFS** (contract with 03 §4.5): kiln-phys exposes
```rust
fn power(&self, act: &ActivityReport, v: &DomainVoltages, t: &Temps) -> PowerBreakdown;
fn vf_table(&self, domain: ClockDomainId) -> &VfTable;   // monotone, f_max = f_boost
fn p_static(&self, v: &DomainVoltages, t: &Temps) -> f64;
fn envelope(&self) -> Envelope { p_cap_w, cap_level: Chip|Package|Board, t_j_max_c, q_max_w_mm2 }
```
Example (the A100 case, 06 regression): at f = 1410 MHz on a large bf16 GEMM, `P_board(1410) > 400 W`;
03's bisection on the V/f table lands near 1290 MHz with calibrated `kappa_E_*`, `V_th`, `alpha`. Decode GEMV
stays at 1410 MHz because `E_dyn_core/window` is low and HBM energy is clock independent.

**Assumed caps** (01 §12 `assumed: { lo, hi, basis }`). A cap with no published value (Google publishes no TPU
v5e/v6e power) is a range, not a number: it is reported (`tdp_assumed_w` = nominal, lo, hi) and never enforced, so
the clock solve does not throttle a reference design under a placeholder. Its pessimistic end (`lo`) is where a
real cap could bind; the engine's envelope margins may compare peak power against `hi` (central) and `lo`
(pessimistic). Published caps (A100 400 W, H100 700 W) and published measured maxima used as caps (TPU v4
192 W, [tpuv4_isca2023]) are enforced as before. Rule recorded in 08 §H.

**Thermal throttling** (Tier B): if the thermal solve gives `T_j > t_j_max`, the controller reduces f until
satisfied; leakage-temperature coupling iterates (fixed point, max 8 iterations); divergence (dP_static/dT *
R_th >= 1) is error `E-PHYS-THERMAL-RUNAWAY`.

**Floors and conservation** (00 rule, 03 invariant I6): kiln-phys checks per window that every energy term is
non-negative, that `P_static >= 0`, and reports the breakdown in the same summation order as 03.

---

## 9. Thermal model

**Tier A** (per design, per window, O(blocks)):
- Average density `q_avg = P_chip / A_die <= q_avg_max` (default 1.0 W/mm2 air, 1.5 liquid; asm M. Reference:
  A100 0.48, H100 0.86 W/mm2 at TDP, der from pub).
- Block density `q_block = P_block / A_block <= q_block_max` (default 2.0 W/mm2, asm L).
- Junction estimate `T_j = T_inlet + P_pkg * R_th_ja`, with `R_th_ja = R''_ja / A_die`, `R''_ja = 90 K mm2/W`
  air-cooled HGX-class (der, L: A100 at 400 W with ~45 K rise over 826 mm2), 40 K mm2/W cold plate (asm, L).
Violations of q limits are errors in the envelope check (`E-PHYS-POWER-DENSITY`), with the block named.

**Tier B** (steady state, HotSpot-style RC network [hotspot_2006]):
- Grid `G x G` per die layer (default 64), layers: active Si (each 3D tier), bond layer, bulk Si, TIM,
  spreader, effective sink conductance. Lateral conductance `k_Si = 120 W/mK at 85 C`, vertical per layer
  `k A / t`; sink `A_cell / R''_sink`.
- Power map: each block's power spread uniformly over its rect on its layer.
- Solve `K T = P + G_amb T_amb` with CG + incomplete-Cholesky (IC0) preconditioner, fixed tolerance 1e-8,
  max 500 iterations, deterministic ordering. Target <= 50 ms for 64 x 64 x 6 layers.
- Outputs: per-bin T map (to 05), per-block max T, hotspot list, leakage update.
- 3D: upper tiers see the bond layer's conductance and the lower tier's power; this is where stacked-memory or
  stacked-logic designs are rejected or throttled.
- Transient thermal: out of scope v0 (the power cap is averaged over `tau_pm`, thermal time constants are
  seconds, so steady state per sustained phase is the right model; open question for bursty workloads).

---

## 10. Area roll-up and reporting

```
A_die = A_core_placed (incl. channels and whitespace) + seal ring
A_core_placed = sum_blocks A_block / u_place  (Tier A), or the legalized bounding outline (Tier B)
A_block = A_datapath + A_storage + A_ctrl + A_router + A_phy + A_misc share
```
Report: per block, per part (datapath / SRAM / RF / control / NoC / PHY / misc / whitespace), per die, package
area, shoreline usage per edge, transistor-count estimate (`GE * 4 + sram_bits * 6 * f_port_tx`) for the
transistor-count cross-check (A100 54.2 B, H100 80 B, TPU v4 22 B; pub H/M).

---

Monotonicity (06 property P13): the **envelope area** used for fitness is the analytic roll-up above (Tier A
form), which is monotone non-decreasing in component count and size by construction (all terms non-negative,
`u_place` fixed per calibration). Tier B's legalized outline is reported as `die_mm2_legalized` and must be
>= 0.97 x the envelope area (else the Tier B placer has a bug); it never replaces the envelope value, so placer
heuristics cannot make a larger design look smaller. Same rule for power: per-component energies are
non-negative and additive.

**Legacy profile** (06 migration test): `model_profile = "legacy_harness_v0"` evaluates exactly the formulas of
`harness/physics.py` with its constants (tagged `asm` with the [H14]/[WC]/[SR]/[TS]/[OC] citations) so 06 can
assert agreement within 1%. It is never used for fitness.

## 11. Cost model (optional, for envelope comparisons)

Per die: dies per wafer (300 mm, `d` = 300, `A` incl. scribe 0.1 mm per edge):
`DPW = floor(pi (d/2)^2 / A - pi d / sqrt(2 A))`.
Yield: negative binomial `Y = (1 + A D0 / alpha_c)^(-alpha_c)`, `alpha_c = 3` (asm M), `D0` per node:
N7 0.09 /cm2, N5 0.10, N3 0.12, N16 0.06 (pub-mkt/asm, L: TSMC symposium statements of "< 0.1" for N7/N5).
**Harvesting**: with `n` identical units of which `k` must work (`n` minus 01 `disabled` count), per-unit yield
`y_u = Y(A_unit)`, die yield = `Y(A_non_redundant) * P[Binomial(n, y_u) >= k]` (e.g. GA100: k = 108 of 128 SMs).
Wafer prices (USD, CSET 2020 [cset_2020], pub M as estimates): N16/12 3,984; N10 5,992; N7 9,346; N5 16,988;
N3 ~20,000 (asm, L; press). Die cost `= wafer / (DPW * Y)`; KGD test + sort `+5%` (asm).
Package: interposer cost per mm2 (CoWoS-S asm 0.02 USD/mm2 ... placeholder L), assembly yield 0.98 per die
attach (asm L), HBM cost per GB by generation (asm L, from public analyst estimates, dated). Optional NRE
(mask set by node, asm). Chiplet vs monolithic comparison = same formulas per die + D2D PHY area + package.
All cost outputs carry `c = L` and are excluded from any claim except relative envelope comparison.

---

## 12. Calibration

### 12.1 Parameters (fit set), roles and bounds

| Param | Role (single term it enters) | Prior | Bounds |
|---|---|---|---|
| `util_std_cell(node)` | std-cell placement utilization in logic blocks | 0.65 | [0.45, 0.85] |
| `u_place` | fraction of the die covered by declared blocks (packing plus undeclared fill) | 0.85 | [0.50, 0.95] (lower edge: TPUv4i's floorplan has interconnect "stretched to fill space", [tpuv4i_isca2021] Fig. 6) |
| `kappa_pe` | PE area overhead over the structural lane (local wiring, muxing, gating) | 1.5 | [1.0, 3.0] |
| `kappa_ctrl` | control area per fixed-function unit | 0.15 | [0.03, 0.6] |
| `A_misc` | chip-level misc area (N7 mm2; scales like logic, incl. `util` transfer) | 15 mm2 | [5, 60] |
| `simt_ctrl_ge` | fixed SIMT sub-core control, GE per sub-core | 1.5 M | [0.2 M, 6 M] |
| `simt_operand_ge_per_bit` | SIMT matrix-operand delivery, GE per operand bit per cycle | 500 | [50, 5000] |
| `kappa_sram_area` | product-SRAM array overhead beyond the compiled-macro periphery model | 1.0 | [1.0, 4.0] |
| `kappa_E_dp` | datapath energy overhead | 1.3 | [1.0, 2.5] |
| `kappa_E_sram`, `kappa_E_rf` | storage energy | 1.0 | [0.5, 2.0] |
| `kappa_E_wire` | wire coupling/shielding/short-circuit | 1.5 | [0.8, 4.0] |
| `kappa_clk` | clock tree | 1.0 | [0.3, 3.0] |
| `p_ll`, `p_ls` | leakage densities | 0.04 W/mm2, 1 mW/Mib | [0.01, 0.15], [0.2, 5] |
| `V_th`, `alpha` | V/f curve shape | 0.30, 1.3 | [0.2, 0.45], [1.0, 2.0] |
| `e_dram_core(kind)` | DRAM energy per bit | 3.9 / 3.2 pJ | [2.0, 6.0] |
| SRAM periphery `n_hp, n_wp, A_fix, g_ctrl` | fitted in data build (§4.3), frozen for `kiln calibrate` | | |

Scope keys (03 §9 rule 6): `node`-scoped (util_std_cell, p_ll, p_ls, SRAM, wire), `technology-family`-scoped
(kappa_* datapath/storage/wire/clock), `platform`-scoped only for things that describe a specific product
(e.g. A100's `ctrl_ge` per SM, `A_misc`, V/f). A novel design inherits only node- and family-scoped values.

### 12.2 Targets

Area (targets/dies.json):

| Chip | Node | Die mm2 | Other published facts used | Use |
|---|---|---|---|---|
| A100 (GA100) | N7 | 826 (pub, H) | 54.2 B transistors; 128 SMs on die / 108 enabled; 40 MB L2; 6 HBM2 sites (5 used); 12 NVLink3; PCIe4 | fit |
| GP100 (P100) | N16 | 610 (pub, H) [nvidia_p100_wp] | 15.3 B transistors; 60 SMs on die / 56 enabled, two 32-lane processing blocks per SM, no tensor cores; 4 MB L2; 4 HBM2 | fit (the N16 die; constrains the fixed SIMT control) |
| TPU v4 | N7 | < 600 (pub, H) [tpuv4_isca2023] Table 4 | 22 B transistors; 2 TC x 4 MXU 128x128; CMEM 128 MiB; 1050 MHz | fit (one-sided: hinge above 600; the former lower edge 500 was not published) |
| TPU v4i | N7 | < 400 (pub, H) [tpuv4i_isca2021] Table 1, Fig. 6 | 16 B transistors (Table 1); CMEM 28% of the die (Fig. 6 caption, text), MXUs 11% (text, Table 7); 1 TC, 4 MXU 128x128, 1050 MHz, 175 W TDP | fit (one-sided below 400; fractions +- 4 pts as block area over die area, whitespace excluded) |
| TPU v6e | n/p | < 858 (physical: monolithic die within the reticle) | | fit (hinge above the reticle only) |
| V100 (GV100) | N12 -> N16 table | 815 (pub, H) | 21.1 B transistors; 84 SMs / 80 enabled | held out (cross-node GPU check; residual documented in §17) |
| H100 SXM (GH100) | N4 | 814 (pub, H) | 80 B transistors; 144 SMs on die / 132 enabled; 50 MB L2 enabled (60 MB on die, pub M) | **held out** (node transfer N7 -> N4) |
| TPU v5e | n/p | 300-350 (unofficial, L) | | weak held-out check only |
| TPU v6e | n/p | not published | | no area target |
| MI300X (06 §L6) | N5 XCD + N6 IOD, 3D | per-chiplet areas to verify from AMD disclosures | 8 XCD on 4 IOD, 8 HBM3 | held-out package/3D model check only (structure, shoreline, package area), once areas are sourced |

Power and clock (targets/power.json):

| Chip | Facts | Use |
|---|---|---|
| A100 SXM4 40 GB (Colab, our telemetry via `calibration/gpu_bench.py` ClockLog: SM clock, power, temperature; summarized in 06 §2) | 400 W cap; bf16 GEMM sweep at 1275-1290 MHz under the cap; large prefill GEMMs 1305-1335 MHz at 403-410 W; 1410 MHz on GEMV; idle power from the same log | fit (V/f, kappa_E, leakage) |
| A100 GEMM sweep at multiple shapes, with per-op power | equations separating MAC/SRAM energy (compute-bound) from HBM energy (memory-bound) | fit; ~half held out by shape class |
| TPU v4 | idle/min/mean/max 90/121/170/192 W (pub, H) [tpuv4_isca2023]; whether HBM is included is unclear (open question) | weak fit (idle -> leakage + always-on), max as upper bound on GEMM |
| H100 SXM | 700 W TDP, 1980 MHz max boost (pub, H) | held-out sanity: dense GEMM must throttle, predicted clock reported |
| TPU v5e / v6e | measured performance only (power not exposed on Cloud TPU, assumed) | none for power |

### 12.3 Procedure

1. Data-build fits (SRAM periphery, wire R/C, Liberty-derived energies) are frozen first.
2. Area group: MAP estimate in log-space, residuals `log(A_pred/A_target)` (interval targets use a hinge
   loss: zero inside the interval, quadratic outside), fractions as absolute-point residuals. Priors
   log-normal centered at the table prior with sigma by confidence (H 0.1, M 0.3, L 0.7). Levenberg-Marquardt,
   fixed 50 iterations, deterministic. Constraints: A100 and GP100 die areas, TPU v4/v4i/v6e one-sided die bounds,
   2 v4i fractions, TPU v4/v4i transistor counts as weak terms (sigma 0.25) vs 8 free area parameters; the priors
   make it well-posed and the report says so. GPU transistor counts are reported checks, not fit terms: the IR does
   not describe the SM logic they count (GA100 needs about 12 B GE against the model's about 7 B), so as fit terms
   they only push `util_std_cell` to its upper bound. The transistor estimate counts SRAM macro periphery as logic.
3. Power group (after 03's clock/launch calibration order: clock/power first): fit `V_th, alpha, kappa_E_*,
   p_ll, kappa_clk, e_dram_core` to the A100 telemetry (sustained clock and power per op class, idle power).
4. Report: fitted values, posterior sigma (Laplace), which params hit bounds (hitting a bound = fail, 03 §9
   rule 2), per-target residuals, leave-one-chip-out errors.
5. Held-out checks (must pass for a calibration set to be accepted):
   - H100 area predicted from its IR on N4 node tables with N7-fitted family parameters: within +-12%.
   - V100 (held out, N12 on the N16 table) reported with its residual (§17).
   - The committed calibration sets are what `phys_calibrate` produces from the tree they are committed with
     (kiln-sim test `phys_calibration_reproduces`; the power group depends on engine traffic, so an engine change
     that moves it requires a refit).
   - Fit-set die areas within +-5%; TPU v4/v4i inside their intervals; v4i fractions within +-4 pts.
   - Transistor-count estimates within +-25% (weak).
   - A100 held-out shapes: sustained clock within +-3%, power within +-8%.
   - H100 dense GEMM: predicted clock < 1980 MHz (throttles) and > 1400 MHz (sanity band, asm).
6. Calibration set file `calib/<platform>-<date>.json` with all values, targets hash, source list, fit
   diagnostics; its hash enters every result's provenance (00).

Honest limitation (reported in `kiln calibrate` output): three-to-four chips cannot identify every parameter;
the node-scoped and family-scoped parameters are mostly prior-driven. Iso-node relative comparisons (a design
vs A100 both at N7 under the A100 calibration) are the claim kiln supports with stated tolerance; absolute and
cross-node numbers carry the propagated uncertainty (§12.5).

### 12.4 Expected first-fit diagnosis for A100 (what must appear as terms)

| Missing in placeholder | Estimated contribution (asm, to be confirmed by fit) |
|---|---|
| RF 27 MB + L1/shared 21 MB as small banked macros | ~ 120-180 mm2 |
| Non-tensor SM logic (FP32/INT/FP64 lanes, SFU, schedulers, LSU, TEX) via `ctrl_ge` | ~ 150-250 mm2 |
| 20 spare SMs + spare L2/HBM site | ~ 15% of SM area + 1 PHY |
| NVLink3 (12 links x 4 lanes x 2 dir) + PCIe4 PHYs | ~ 30-50 mm2 |
| Crossbar/NoC + channels | ~ 30-60 mm2 |
| HBM DRAM power on board | ~ 50-70 W at full bandwidth (2 TB/s x 3.9 pJ/b = 62 W) |
These estimates are only for sanity; the calibration must reach 826 mm2 through these terms without any
parameter at a bound.

### 12.5 Uncertainty propagation

Result intervals for area, power and clock (and, through clock and wire lengths, performance) come from 03
§9.1's corners using the ranges of §3. Each result also includes `phys_uncertainty`: first-order sensitivity of
die area, power, and clock to every `asm`/`L` parameter (`d metric / d log param * sigma_param`) and the top-5
contributors, which names the drivers of the interval width. Computed by forward finite differences in Tier B or
on request; Tier A reports a cached per-design value from the last Tier B run, or none.

---

## 13. Interfaces summary

**kiln-phys needs from 01 (hardware IR):** per die: node id, outline or aspect bounds, envelope (`p_cap_w`,
cap level, `t_j_max_c`, cooling class), clock domains with `f_boost` (and optional `V_at_boost`, V/f points);
per component: kind, template, `count` + layout rule, MAC modes (01 `PrecisionMode`: formats, `rate`,
accumulator; dot length from `Geometry`), vector lanes and classes, storage instances (capacity, word width, ports, banks, implementation
`flop | latch_array | sram_banked | sram_macro`, power-gating flag), router params (P, V, B, W, n_pipe), link
instances (endpoints, `width_bits`, domain, direction, class preference, sizing, swing, pipelined), PHY
instances (kind, count/lanes, edge preference), spares/harvesting, pins/keep-outs/3D layer assignment, package
(kind, interposer, HBM stacks and their PHY binding), optional per-block overrides (01 `PowerOverride`: `area`, `energy_per_op`,
`ctrl_ge`) with `source`.

**kiln-phys needs from 03:** `TrafficMatrix` (bytes per link, workload-weighted) for placement; per-window
`ActivityReport` for power; critical-path link ids (optional) for the placement objective.

**kiln-phys provides to 03:** `LinkCost` for every link instance; router port parameters; `MemLevel`
energy/latency per storage instance; `e_mac` per `MacMode`, `e_mac_idle`, `e_vec`, `e_conv`; `VfTable` per
domain; `power()`, `p_static()`, `envelope()`; thermal limit and (Tier B) temperatures.

**To 05 (visualizer):** `Floorplan` (dies, blocks, layers, PHYs, edges, package), per-link geometry, RUDY maps,
power-density and temperature maps, area breakdown, shoreline usage.

**To 06:** calibration API (`calib::fit(group, targets) -> CalibSet`), envelope violations as structured
errors, `phys_uncertainty`, per-constant ranges (§3), performance counters (characterize/place/link/thermal times).

Structured error codes (all with entity path and hint): `E-PHYS-UNSOURCED`, `E-PHYS-RETICLE`,
`E-PHYS-AREA-OVERFLOW`, `E-PHYS-SHORELINE` (PHYs do not fit edges; hint names stacks and edges), `E-PHYS-WIRE-LIMITED`
(only if unresolvable within reticle), `E-PHYS-ROUTER-TIMING`, `E-PHYS-POWER-DENSITY`, `E-PHYS-THERMAL-RUNAWAY`,
`E-PHYS-NODE-MISSING-FIELD`, `E-PHYS-PACKAGE-OVERFLOW`, `E-PHYS-CALIB-BOUND`. Aliases of 01 (P) codes:
`E-PHYS-AREA-OVERFLOW` = E-IR-0803, `E-PHYS-RETICLE` = E-IR-0804, `E-PHYS-SHORELINE` = E-IR-0805,
`E-PHYS-POWER-DENSITY` = E-IR-0907; kiln-phys emits the `E-PHYS-*` name.

Performance targets (per design, single core): characterize <= 0.5 ms; Tier A place + links <= 2.5 ms at 2,000
macros; power() <= 20 us per window; Tier B place <= 2 s; thermal <= 50 ms; full calibration fit <= 10 s.

---

## 14. Cross-section dependencies

- **01 Hardware IR:** the cap is solved per steady-state phase in Tier A and per `tau_pm` window in Tier B (00
  decision 1; 01 §12 and 03 §4.5 agree); kiln-phys evaluates any window. 01's default block aspect range (0.5, 2.0) is used for soft blocks; die aspect default here is
  [0.75, 1.33].
- **01 Hardware IR (fields):** must carry every field in §13 "needs from 01"; arrays with layout rules; storage
  implementation kind; bit-denominated fields named `*_bits` per 00 (e.g. link `width_bits`); harvesting (`disabled`); pins; 3D
  layer assignment; envelope with cap level; package description including HBM stack binding to PHYs.
- **02 Workload IR:** none directly. Optional future: per-tensor data statistics for wire/datapath activity
  factors (open question 3).
- **03 Mapping engine:** owns `solve_clock`, DRAM timing, contention, protocol efficiencies (`eta_link`); uses
  `LinkCost`, `MemLevel`, `MacMode` energies, `VfTable`, `P_static`. Provides `TrafficMatrix` and
  `ActivityReport`. The two-round place-map loop (§6.4) is driven by kiln-sim/kiln-map, not kiln-phys. The
  `C_eff` 03 once listed is realized here as the structural `kappa_E_*` factors (§8, 00 decision 6); 03 §9
  references them.
- **05 Visualizer:** consumes `Floorplan`, link geometry, congestion, power and thermal maps.
- **06 Validation/API:** its L6 acceptance (die area +-15%, cap-limiting power +-20%) is the outer gate; §12.3
  tolerances here are tighter internal goals. Error codes use 06's `E-PHYS-*` prefix. Runs `kiln calibrate` (§12), holds the targets, adds the A100 power/clock telemetry
  suite (already logged by `calibration/gpu_bench.py`), enforces held-out acceptance, exposes OpenROAD audit.
- **07 Prior art:** CACTI 7 and OpenRAM data are fitted into tables (not linked); own placer for the inner
  loop, OpenROAD/DREAMPlace only for audits; HotSpot is re-implemented (steady state) rather than linked; MX
  synthesis papers used as ratio checks.

---

## 15. Sources (registry keys; full entries live in `kiln-phys/data/sources.json`)

- `horowitz_isscc2014`: M. Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014.
- `wikichip_n16/n7/n5/n3`, `wikichip_iedm2022`: WikiChip fuse articles on TSMC N16/N7/N5 densities and
  "IEDM 2022: Did we just witness the death of SRAM?" (N3B 0.0199, N3E 0.021 um2, 31.8 Mib/mm2).
- `tsmc_iedm2014/2016/2019`: TSMC platform papers (16FF+, N7, N5) with HD bitcell sizes.
- `tsmc_isscc2025`: TSMC N2 HD SRAM, 38.1 Mb/mm2, ISSCC 2025 session 29.
- `tsmc_n7_press`, `tsmc_n5_press`, `tsmc_n3e_press`, `tsmc_n4_press`: TSMC public node claims.
- `intel_bohr2017`, `intel_iedm2017`, `intel_vlsi2022`: Intel 10 nm density (100.8 MTr/mm2), 10 nm SRAM, Intel 4.
- `asap7_mej2016`: L. T. Clark et al., "ASAP7: A 7-nm finFET predictive process design kit", Microelectronics
  Journal 53, 2016; OpenROAD `asap7` repo.
- `orram_2607.12244`: ORRAM (OpenROAD-integrated RAM generator), arXiv 2607.12244, OpenRAM SKY130 density.
- `cacti7`: Balasubramonian et al., "CACTI 7", ACM TACO 2017.
- `rixner_hpca2000`: Rixner et al., "Register organization for media processing", HPCA 2000.
- `stillmaker_2017`: A. Stillmaker, B. Baas, "Scaling equations for the accurate prediction of CMOS device
  performance from 180 nm to 7 nm", Integration 58, 2017.
- `bakoglu_1990`: H. B. Bakoglu, "Circuits, Interconnections, and Packaging for VLSI", 1990.
- `ho_wires_2001`: R. Ho, K. Mai, M. Horowitz, "The future of wires", Proc. IEEE 2001.
- `banerjee_2002`: K. Banerjee, A. Mehrotra, "A power-optimal repeater insertion methodology for global
  interconnects", IEEE TED 2002.
- `orion2_2009`: A. Kahng et al., "ORION 2.0", DATE 2009. `dsent_2012`: C. Sun et al., "DSENT", NOCS 2012.
- `rudy_2007`: P. Spindler, F. Johannes, "Fast and accurate routing demand estimation...", DATE 2007.
- `kraftwerk2_2008`: P. Spindler et al., "Kraftwerk2", IEEE TCAD 2008. `eplace_2015`: J. Lu et al., ePlace,
  ACM TODAES 2015. `fm_1982`: Fiduccia & Mattheyses, DAC 1982. `stockmeyer_1983`: L. Stockmeyer, "Optimal
  orientations of cells in slicing floorplan designs", Information and Control 1983. `murata_1995`: H. Murata
  et al., sequence pair, ICCAD 1995.
- `hotspot_2006`: W. Huang et al., "HotSpot: a compact thermal modeling methodology", IEEE TVLSI 2006.
- `sakurai_1990`: T. Sakurai, A. R. Newton, alpha-power law MOSFET model, IEEE JSSC 1990.
- `oconnor_micro2017`: M. O'Connor et al., "Fine-grained DRAM", MICRO 2017 (HBM2 ~3.9 pJ/bit).
- `jedec_hbm2e`, `jedec_hbm3`: JESD235 family; `synopsys_hbm3_phy`: Synopsys HBM3/3E PHY datasheet (9.6 Gb/s);
  `chipscalereview_2023`: HBM3 package width 11 mm, 96 um microbump pitch.
- `ucie_1.0`: UCIe 1.0 specification / D. Das Sharma overview (28-224 GB/s/mm standard, 165-1317 GB/s/mm
  advanced, 0.5 / 0.25 pJ/b, < 2 ns).
- `nvidia_a100_wp`, `nvidia_h100_wp`, `nvidia_gh200`: NVIDIA architecture whitepapers (die sizes, SM counts,
  L2, NVLink, C2C).
- `tpuv4_isca2023`: N. Jouppi et al., "TPU v4: An optically reconfigurable supercomputer...", ISCA 2023
  (< 600 mm2, 7 nm, 22 B transistors, 1050 MHz, 90/121/170/192 W).
- `tpuv4i_isca2021`: N. Jouppi et al., "Ten lessons from three generations shaped Google's TPUv4i", ISCA 2021
  (< 400 mm2, CMEM 28%, MXU ~11%).
- `cset_2020`: S. Khan, A. Mann, "AI Chips: What they are and why they matter", CSET 2020 (wafer prices).
- `tsmc_cowos`, `tsmc_soic`, `asml_reticle`: public TSMC packaging and ASML reticle-field statements.
- MX: `ocp_mx_v1` (OCP Microscaling Formats v1.0, 2023); 2511.06313; VMXDOTP 2603.04979 (from 07).

---

## 16. Open questions

1. **HBM PHY shoreline and area** are assumed (6-6.5 mm/stack, 12-15 mm2). Die-shot analyses (TechInsights,
   Yole) are paywalled. Can we measure from public A100/H100 die shots (pixel measurement against known die
   dimensions)? This term decides whether shoreline or area binds for 5-6 stack designs.
2. **TPU v4 192 W:** chip-only or including HBM? Changes the leakage/idle fit for TPU and whether v4 is usable
   as a power target.
3. **Activity factors:** wire and datapath energy use random-data toggle factors. Real LLM activations/weights
   (bf16, many near-zero exponents, sparse int8) toggle less. Should 02 carry per-tensor toggle statistics?
4. **A100 voltage:** NVML does not expose core voltage on A100; V/f is then fit only from (clock, power). Do we
   lock clocks at several values (`nvidia-smi -lgc`, permitted on Colab?) to separate `V_th/alpha` from
   `kappa_E`? 03 §9 assumes "measured clock/voltage at locked clocks".
5. **Node transfer validation** relies on one held-out chip (H100, N4). Is a second cross-node point available
   (e.g. TPU v5e die area from a credible source, or an open-PDK tapeout of our own)?
6. **Layer assignment in 3D** is declared in v0. Should the placer choose tiers (thermal-aware) in v1?
7. **Transient thermal** for bursty multi-phase workloads (prefill/decode alternation) is out of scope; is
   steady state per phase enough for the evolution-loop envelope check?
8. **Control logic of programmable units** (`ctrl_ge`) for GPUs dominates area and is platform-scoped; novel
   designs with programmable cores need a defensible default (e.g. from open RISC-V vector core synthesis at
   ASAP7). Which reference core?
9. **SerDes numbers** (area/lane, pJ/b, shoreline) are all assumed L. Transcribe from 2-3 ISSCC 112G/224G papers
   before multi-chip envelope claims.
10. **Power delivery:** bump current density and package power-delivery limits (e.g. ~1,000 A at 0.75 V for a
    700 W die) are not modelled. Add a C4/bump-count constraint?
11. **ASAP7 vs N7 discrepancy policy:** when an elite is audited through OpenROAD on ASAP7, how are its
    ASAP7-absolute area/energy mapped back to the N7 tables (ratio to an ASAP7-implemented A100-like reference
    block)?

---

## 17. Implementation notes (M3, 2026-10-05)

What `kiln-phys` implements and where it departs from the text above; each departure is a model decision, recorded
here so the next reader does not have to rediscover it.

**Implemented.** Sourced data files (`kiln-phys/data/`, embedded, hashed as `phys1-...`, bare numbers rejected as
`E-PHYS-UNSOURCED`): TSMC N16/N7/N5/N4/N3E/N2 and DRAM-process logic node tables (aliases `nvidia_4n`, `tsmc_n4p` ->
N4, `tsmc_n12` -> N16), Horowitz 45 nm datapath anchors, DRAM kinds, PHYs, packages, the parameter registry
(`params.json`), calibration targets and fitted sets (`calib/`). Block models (§4.2-§4.4, §4.11, §7.4), per-instance
characterization with template caching, floorplan with array collapsing, shoreline sites/greedy edge assignment,
die sizing (fixed / max-area / auto, reticle-aware), Tier A placer (§6.4) and a Tier B placer, hierarchical layout
of every instance, package frame with HBM stacks and pinned dies, repeated-wire model and `LinkCost` per channel
(§7.1-§7.2), memory energies/latencies (§7.3), V/f tables, leakage, clock tree, board/VR overheads (§8), the 03 §4.5
cap solve, Tier A thermal estimate plus a Tier B steady-state grid solver (§9), envelope roll-up with corner bands
(§10), MAP calibration (§12), E-IR-UNPRICED pricing (01 §18.3).

**Departures.**
1. *Datapath fp anchors.* The bit-level partial-product form underestimates fp multipliers and fp accumulators about
   3x against Horowitz's own rows; fp multipliers scale the fp16 multiply by `(M_a M_b / 121)` and fp adders scale the
   fp16/fp32 add rows by `(bits/16)^1.62` (area) and `^1.17` (energy). Integer, alignment, tree and register terms are
   as §4.2. mma dot length is `min(k, 8)` (§4.2's "A100 ~ 4-8").
2. *Crossbars.* The matrix form `(P W p_x)^2` stacks its buses over the intermediate + semi-global layers of one
   direction (divide the side by that layer count); a radix above 8 is `ceil(P/8)` radix-8 tiles. Bus topologies are
   a mux tree plus arbiter, no buffers.
3. *Two parameters beyond §12.1.* `kappa_sram_area` (area group: ECC/redundancy bits, bank channels, macro abutment
   of product SRAM beyond the compiled-macro periphery model; the TPUv4i CMEM 28% fraction needs it) and
   `alpha_ctrl` (power group: switching of control logic while a phase runs; GPU power at decode is dominated by it,
   §4.2's "operand delivery, control, ..."). Also registered: `simt_ctrl_ge` and `simt_operand_ge_per_bit`
   (family-scoped SIMT sub-core control, §5), `cache_pipeline_cycles` (cache hit pipeline beyond the data array, ~190
   cycles from GPU microbenchmarks), `misc_block_mm2`. The operand-delivery GE is left out of `alpha_ctrl`'s
   while-running control switching: it toggles with the operand traffic, which the feed links charge per byte.
4. *Tier B placer* is simulated annealing over the Polish expression of the Tier A slicing tree (Wong-Liu moves,
   seeded from the design hash, `min(400 n, 20000)` moves), not the quadratic/ePlace/sequence-pair flow of §6.5;
   shape curves (§6.4 step 4) are replaced by area-proportional regions with a soft-aspect penalty in the objective.
   The Tier A cut ordering is a 1-D quadratic embedding with terminal propagation (Fiedler vector by power iteration
   when no terminal pulls a set), FM refinement as specified.
5. *Canonical order.* Every reduction and placement iterates nodes in a structural order (subtree signature of kind,
   own area, instance index and coordinates), so permuting or renaming declarations gives bit-identical numbers (P4).
6. *P6 with a floorplan* (revised 2026-10-05, see the fixes below). Links are measured on the live geometry (§6.2):
   unreachable hardware leaves every link and time bit-identical; a reachable unit no op can use only stretches
   wires (§6.2 arrangement, §7.2 cluster rule). The former 2% two-sided tolerance is gone.
7. *Envelope area* is the analytic roll-up (blocks / `u_place` + seal ring); a fixed outline is a constraint
   (`E-PHYS-AREA-OVERFLOW` above `(1 + area_overflow_tol)` x outline, tol 5% = the fit-set tolerance). Findings on a
   design that carries a published `die_area` claim (a real chip) are reported as `W-PHYS-RESIDUAL`, not errors.
8. *Cap solve.* `ClockMode::PowerCapped` is the default; a platform telemetry clock table (`f_cap_op`, A100) still
   overrides the solved clock. A phase over the cap at the lowest V/f point runs there and reports `E-MAP-POWER-CAP`.
   Energies the engine charges are board energies: core-domain events at `(V/V_nom)^2`, and `static_j` holds the
   time-proportional terms (leakage at V and T_j, clock tree, control switching, DRAM background, board power, VR
   loss). The 04 §9 junction limit is a Tier A warning (`W-PHYS-THERMAL`); Tier B throttling is not wired.
9. *Power calibration* converts energy to power with the measured run time (isolates energy coefficients from
   engine timing error); V/f is not fitted (the A100 IR declares its V/f points; NVML exposes no voltage, §16 q4).

**Fixes before design campaigns (2026-10-05).** Root causes and changes; numbers from `phys_calibrate`.
- *Calibration starved for structure.* At priors every chip was 30-45% under its die (A100 467 vs 826 mm2), so
  the fit pushed GE-scaled knobs to their bounds (`simt_ctrl_ge` 6 M, `a_misc_mm2` 60, `u_place` 0.70) and those
  scale x3.2 to N16: V100 came out 1509 mm2 (+85%). Causes and fixes: harvested SIMT sub-cores paid no control area
  (the 20 spare GA100 SMs; fixed); SIMT control split into a fixed and a matrix-operand part (§5); `util_std_cell`
  transfer to unfitted nodes (§4.1; N16 had the 0.65 prior against N7's fit); chip misc scales like logic; the
  transistor estimate counts SRAM periphery; TPU v4/v4i die targets had unpublished lower edges ([500, 600], [330,
  400]) and are one-sided as published; v4i transistors (16 B, Table 1) added; v4i fractions are block area over
  die area; GP100 (N16, no tensor cores) joined the fit; v6e's 16-port SparseCore scratchpad (one 170 mm2 bank) is
  banked; `u_place` lower bound 0.50 (TPUv4i fill, §12.1); v6e must fit the reticle.
- *Result (area group, no parameter at a bound):* util 0.645, u_place 0.633, kappa_pe 1.06, kappa_ctrl 0.144,
  a_misc 18.3 mm2, simt_ctrl_ge 1.73 M, simt_operand 370 GE/bit, kappa_sram_area 2.00. A100 832 (+0.8%), GP100 600
  (-1.6%), TPU v4 563 (< 600), v4i 379 (< 400; CMEM 26.2%, MXU 11.1%), v6e 347; held out: H100 882 (+8.4%, within
  12%), V100 1026 (+25.9%), v5e 284 (-5.5% of the unofficial 300-350).
- *V100 residual (+26%, documented, not fitted).* The model gives a GV100 SM about twice a GP100 SM at the same node
  (four 16-lane sub-cores, each with the fixed SIMT control and its own RF macro, against two 32-lane blocks),
  while the published dies give about equal die area per SM (815/84 vs 610/60 mm2). Scaling the fixed control with
  issue width instead fixes V100 (+22%) but breaks the held-out H100 (32-wide sub-cores, +17.6%), so two fitted GPUs
  do not identify how sub-core control scales; GPU transistor counts (A100 28.7 vs 54.2 B) say the IR leaves most SM
  logic undescribed. Cross-node GPU area is good to about +-25%.
- *Power group:* `kappa_e_rf` sat at 2.0 because control switching charged all SIMT control while running (decode
  overpredicted, GEMMs underpredicted); with operand delivery charged through its traffic: kappa_e_dp 1.66, sram
  1.04, rf 1.82, wire 3.38, clk 0.76, p_ll 0.037, e_dram 3.76, alpha_ctrl 0.050, p_board 8.6 W, none at a bound;
  fit records within 8%, held-out gemm 16384^3 -16.7% (engine time ratio 1.22), decode_b8 +6.2%, prefill_b1 -1.0%.
  The old committed set (kappa_e_dp 1.93) predated 37e2480's engine physics; `phys_calibration_reproduces` now fails
  whenever the committed set is not what this tree fits, and whenever a parameter sits at a bound.

**Deferred.** Open-PDK / ASAP7 node tables (data-build step), RUDY congestion and channel area, DEF import/export,
the cost model (§11), Tier B thermal coupling into the clock controller, `phys_uncertainty` sensitivities (§12.5),
per-block power-density checks, the `legacy_harness_v0` profile.
