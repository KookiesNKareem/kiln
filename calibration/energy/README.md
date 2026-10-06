# A100 per-component energy ground truth

Measured board-level energy per unit of work on one Colab `NVIDIA A100-SXM4-40GB` (400 W limit, driver 580.82.07,
torch 2.11 / CUDA 13.0), for validating kiln-phys. All compute ran on Colab.

## Files

- `kernels.cu`: microbenchmark kernels (compiled with `nvcc -arch=sm_80 -cubin` on the VM, launched through the CUDA
  driver API on torch's context): `spin` (globaltimer busy loop or `__nanosleep` loop), `k_ffma` (8 chaotic
  `x = x*x - 1.9` FMA chains per thread), `smem_rd` (`ld.shared.v4`), `l1_rd` (`ld.global.ca.v4`, 4 KiB tile, 99.95%
  L1 hit in ncu), `gld_cg` (`ld.global.cg.v4` grid-stride, 16 MiB set = L2-resident, 2 GiB set = HBM), `gcopy`.
  The load loops move their base by a runtime zero each iteration, since ptxas otherwise hoists or merges
  repeated loads.
- `energy_bench.py`: `run <out.json> [reps]` is the suite. `check` prints throughputs. `once <bench> <level>` runs one
  short launch for ncu. Env: `ENERGY_LGC` sets the locked SM clock (default 1410). `ENERGY_ONLY` is a comma list of
  benches.
- `ncu_check.sh`: one short launch per bench under Nsight Compute, to confirm the analytic work counts.
- `analyze.py <run.json> [ncu.log [ref_summary.json]]` writes `<run>_summary.json` and `<run>_table.md`.
- `job.sh`: Colab helper (`new`, `launch`, `sh`, `get`).
- Results are in `../measurements/energy/`. `e1.*` is the full suite at a locked 1410 MHz, 3 reps. `e2.*` is a subset
  at a locked 1095 MHz, 3 reps, below the power cap. `ncu_all.log` holds the counter confirmation.

## Method

- Clocks: `nvidia-smi -lgc F,F` worked on Colab and was reset with `-rgc` at the end. Memory clock is fixed at
  1215 MHz. NVML samples SM/mem clock, temperature and clock-event reasons for every window.
- Energy: `nvmlDeviceGetTotalEnergyConsumption` read before and after a 5.5 s steady window, after 1 s of warmup. The
  queue is kept full in 0.2 s batches. The counter updates at 10 Hz on this board, so endpoint quantization is at most
  about 2% per window. A 200 Hz `nvmlDeviceGetPowerUsage` mean is kept as a cross-check: its mean difference from the
  counter is +0.3 W (sd 4.3 W, larger under power capping).
- Work: computed analytically per kernel, (grid x 1024 threads x iters x per-iteration count).
  ncu confirms the counts:
  - FFMA thread-instructions are 227.4M vs 226.5M analytic.
  - Shared loads show 4 wavefronts per LDS.128, which means no bank conflicts.
  - L1 hit rate is 99.95%.
  - `hbm_read` DRAM bytes equal the requested bytes (905,976,832 vs 905,969,664).
  - The L2 bench reads DRAM only for the cold fill.
  - cuBLAS bf16 8192³ issues exactly 268,435,456 HMMA.16816 (= MACs / 2048).
- Each custom kernel runs 1024-thread blocks with 100 KiB of dynamic smem, so exactly one block lands per SM.
  `full` = 108 blocks, `half` = 54 SMs. `duty50`/`duty25` alternate the work kernel with a 1-thread globaltimer spin
  of equal or triple length, which keeps the clocks up.
- Derived energies, all per unit of work. Rep spread is min-max over 3 reps.
  - over busy-spin = (P_full − P_spin_busy_full) / throughput. `spin_busy` keeps every warp slot on every SM issuing a
    globaltimer poll loop. That cancels static, clock-tree, control and issue power for an all-SMs-active chip, so
    this is the datapath plus memory energy of the work. **Headline number.**
  - all-in duty slope = (P_full − P_duty50) / (thr_full − thr_duty50). This is the marginal energy including the cost
    of turning the SMs on (clock ungating, control, issue). This is what a workload that toggles between work and
    idle pays.
  - half-SM slope: valid for the SM-bound benches only (FFMA, SMEM, L1), where it agrees with the duty slope within
    2%. For L2 and HBM, bandwidth barely drops with half the SMs, so that slope is meaningless.

## Schema (`kiln.energy/1`)

`e*.json`: `device`, `software`, `method` (window, grid, `lgc_mhz`), `clock_lock` (nvidia-smi output),
`energy_counter` (update rate), `iters`, `gemm_call_s`, `sass_*`, and `records[]`. Each record has:

- `rep`, `bench`, `level`, `steps`, `wall_s`, `energy_j`, `power_w` (counter-based).
- `power_sampled_w`, `power_sampled_sd_w`, `n_power`.
- `sm_mhz_mean`, `sm_mhz_min`, `sm_mhz_max`, `mem_mhz`, `temp_c_start`, `temp_c_end`.
- `clock_event_reasons_union` (NVML bitmask; 4 = SW power cap).
- `work` {unit: count} and `throughput` {unit: count/s}.
- `grid`, `iters`, `kernel_s` (custom kernels), or `gemm_n`, `calls_per_step` (GEMMs).

Units: `fp32_fma` = thread FMA. `smem_byte`/`l1_byte`/`l2_byte` = bytes delivered to registers. `hbm_byte` = bytes
read (copy: read + written). `mac` = multiply-accumulate.

`e*_summary.json`:

- `baselines` {name: power/clock/temperature spreads}.
- `components` {bench: `over_spin_pJ`, `duty_pJ`, `half_pJ`, `duty50_vs_25_pJ`, `over_active_idle_pJ`, each
  {mean,min,max,sd,n}; per-level power/throughput/clock spreads; `per_rep`}.
- `ncu` {bench: metric: value}.
- GEMM only: `traffic_per_call`, `memory_parts`, `datapath_residual_pJ_per_mac`.
- `kiln`: the kiln values compared against.

## Results (board power, NVML)

Baselines (W):

| state | 1410 MHz | 1095 MHz | kiln (1410 MHz) |
|---|---|---|---|
| idle, no kernel (clocks locked) | 70.2 (67.2-71.7) | 52.3 | 69.1 (busy=0) |
| active idle, 1-thread spin | 79.9 (78.5-80.6) | 56.8 | 69.1 |
| all SMs resident, `__nanosleep` | 83.8 | - | - |
| busy spin, 54 SMs | 121.4 | 76.4 | - |
| busy spin, 108 SMs (issue + control, minimal datapath) | 164.0 (161.8-165.2) | 96.4 | 156.8 (busy=1, activity=0); 250.4 at activity=1 |

Per-unit energies:

| component | measured pJ/unit, over busy-spin, 1410 MHz (min-max) | all-in duty slope, 1410 | over busy-spin, 1095 MHz | method notes | kiln A100 (V_nom 0.75 V) |
|---|---|---|---|---|---|
| FP32 FMA (CUDA core, register-resident) | 8.47 (8.12-8.74) /FMA | 17.9 (17.2-18.5) | 4.19 (4.11-4.23) | 9.73e12 FMA/s = 99.8% of peak | fp32 op 0.766; + RF 12 B rd + 4 B wr = 6.16 |
| SMEM read | 2.60 (2.51-2.68) /B | 6.92 | 1.44 | 19.4 TB/s; includes RF write and ≤1.06 pJ/B LOP3 reduction (≤0.52 at 1095) | 0.160 L1/SMEM array (+0.362 RF wr) |
| L1 hit (global ld.ca) | 3.74 (3.63-3.81) /B | 8.00 | - | 19.4 TB/s; tag + LSU on top of SMEM; same LOP3 bound | 0.160 (+0.362 RF wr) |
| L2 hit (16 MiB set) | 24.8 (24.0-26.4) /B | 52.7 | - | 3.0 TB/s; includes SM↔L2 crossbar and A100 near/far-partition traffic (ncu lts bytes = 1.43× requested) | 0.863 SRAM read (no NoC/crossbar term) |
| HBM read (2 GiB stream, incl. L2 path) | 79.2 (77.3-82.8) /B | 132 | 70.6 | 1.49 TB/s (96% of 1555). Minus the L2 hit cost this leaves about 54 pJ/B for DRAM + PHY + memory controller | 36.8 (32.0 core + 4.8 PHY) |
| HBM copy (read + write bytes) | 68.6 (67.6-70.6) /B | 127 | - | 1.37 TB/s | 36.8 |
| bf16 tensor-core GEMM (cuBLAS 8192³) | 1.75 (1.73-1.77) /MAC at 1256 MHz (power-capped) | 1.51; duty 50 vs 25 = 3.09 | **1.66 (1.64-1.68)**; all-in 1.98-2.01 | 1410 lock hits the 400 W cap (clock 1256 MHz). Minus ncu-counted SMEM/L2/HBM traffic = 0.86 /MAC (tensor datapath + RF + cp.async staging) | 0.123 /MAC datapath only |
| fp16 GEMM | 1.85 (1.83-1.86) at 1223 MHz | 1.63 | - | power-capped | 0.182 |
| tf32 GEMM | 3.68 (3.65-3.70) at 1216 MHz | 3.11 | - | power-capped | 0.210 |
| int8 GEMM (`torch._int_mm`, cutlass i16832) | 1.17 (1.15-1.19) at 1281 MHz | 1.03 | 1.00 (0.99-1.00); all-in 1.21 | minus traffic = 0.38 /MAC | 0.0447 |

Kiln column: `Phys::new` on `designs/reference/a100_sxm4_40gb.json5`, read with an out-of-tree probe, working tree of
2026-10-06 (kiln was not modified). Kiln's other terms:

- clock tree 117.7 W at activity 1 (35.3 W gated)
- control 77.5 W when busy
- leakage 12.6 W
- DRAM background 7.5 W
- board fixed 8.9 W
- VR efficiency 0.90
- cap 400 W

## Caveats

- **NVML is board power.** It includes HBM2 stacks, VRs (about 10% loss) and other board overhead (SXM boards have no fans). The per-unit
  numbers are therefore board-level marginal energies. Divide by about 1.1 for the die-plus-HBM rail, and more for the
  die alone.
- **Clock behavior.** With the clock locked at 1410 MHz, every GEMM ran into the 400 W cap (reason bit 4) at
  1216-1281 MHz. That mixes V/f points across levels. The uncapped GEMM numbers are the 1095 MHz ones.
  - Everything else held 1410 or 1095 exactly.
  - Per-op energy roughly halves from 1410 to 1095 MHz for SM-side work (FFMA 8.5 → 4.2, SMEM 2.6 → 1.4), which is
    consistent with a steep V(f).
  - HBM scales less (79 → 71), because the DRAM and PHY sit on a fixed clock.
  - Kiln quotes energies at V_nom 0.75 V. Which measured clock corresponds to that is a kiln V/f question.
- **Baseline choice.** The over-busy-spin numbers assume the spin's issue and control power is also paid by the real
  kernel. Memory-bound kernels issue less often than the spin, so the per-byte numbers may be slightly low. The
  duty-slope numbers bound the other side by charging all SM activation to the work. The gap between the two columns
  is the "SM busy" overhead:
  - busy spin − active idle = 84 W at 1410 MHz.
  - It is linear in active SMs: 0.78 W per SM.
  - Resident-but-sleeping warps cost only 4 W.
- **Temperature drift.** Full-intensity windows ended at 50-67 °C, against 39-46 °C for the baselines. The leakage
  rise of a few watts is attributed to the work, a bias of a few percent.
- **Microbenchmark overheads.** The SMEM and L1 numbers include the 16 B register write per load and 1/8 LOP3 per byte
  (bounded by the FFMA energy). The FFMA number includes the RF operand reads and the instruction issue beyond the
  spin's.
- **L2 is not just SRAM.** The L2 figure is the full SM↔L2 round trip (crossbar, both partitions, tags). The A100
  far-partition effect is real: ncu `lts__t_bytes` was 1.43× the requested bytes. Kiln's 0.863 pJ/B is the slice array
  only, so the comparison needs kiln's NoC/link energy added.
- **GEMM decomposition is approximate.**
  - It takes ncu traffic per call: bf16 = 34.7 GB of SMEM from ldmatrix wavefronts × 128 B, 12.9 GB from L2 to SM,
    1.48 GB of DRAM.
  - It uses the 1410 MHz per-byte energies, while the GEMM itself ran throttled at about 1256 MHz.
  - cp.async SMEM fills, RF traffic and epilogue stores are not subtracted.
- **Single board, single session.** Rep spread is mostly under ±3%, but board-to-board variation is not captured.
