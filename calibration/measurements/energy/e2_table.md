| baseline | measured board W (min-max) | SM MHz | kiln board W |
|---|---|---|---|
| idle (no kernel) | 52.3 (51.4-52.7) | 1095 | 69 (clocks gated, busy=0) |
| active idle (1-thread spin) | 56.8 (56.7-57.0) | 1095 | 69 |
| all SMs busy spin, half SMs | 76.4 (76.3-76.4) | 1095 | - |
| all SMs busy spin (issue+control) | 96.4 (96.4-96.4) | 1095 | 157-250 (busy=1, act 0..1, no op energy) |

| component | over busy-spin: (P_full - P_spin)/thr, pJ (min-max) | all-in duty slope (full vs 50%) | half-SM slope | full: W, throughput/s, SM MHz | kiln A100 (V_nom 0.75 V) |
|---|---|---|---|---|---|
| ffma (fp32_fma) | 4.19 (4.11-4.23) | 9.7 (9.45-9.94) | 9.38 (9.29-9.44) | 128 W, 7.563e+12, 1095 | fp32 op 0.766; with RF 12 B rd + 4 B wr: 6.16 |
| smem_read (smem_byte) | 1.44 (1.34-1.49); LOP3 share <= 0.52 | 4.21 (4.02-4.33) | 4.13 (4.02-4.28) | 118 W, 1.511e+13, 1095 | 0.160 (+RF wr 0.362) |
| hbm_read (hbm_byte) | 70.6 (70.5-70.8) | 94.9 (93.2-96.3) | 102 (85.5-132) | 201 W, 1.486e+12, 1095 | 36.8 (+L2 path) |
| gemm_bf16 (mac) | 1.66 (1.64-1.68) | 1.98 (1.96-2.01); 50% vs 25%: 2.01 (1.97-2.05) | - | 288 W, 1.152e+14, 1095 | 0.123 per MAC (datapath only) |
| gemm_int8 (mac) | 0.999 (0.994-1) | 1.21 (1.18-1.25) | - | 267 W, 1.707e+14, 1095 | 0.045 per MAC (datapath only) |
