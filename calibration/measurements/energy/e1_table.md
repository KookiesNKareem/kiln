| baseline | measured board W (min-max) | SM MHz | kiln board W |
|---|---|---|---|
| idle (no kernel) | 70.2 (67.2-71.7) | 1410 | 69 (clocks gated, busy=0) |
| active idle (1-thread spin) | 79.9 (78.5-80.6) | 1410 | 69 |
| all SMs resident, nanosleep | 83.8 (83.1-84.2) | 1410 | 157 (busy=1, activity=0) |
| all SMs busy spin, half SMs | 121.4 (119.8-122.3) | 1410 | - |
| all SMs busy spin (issue+control) | 164.0 (161.8-165.2) | 1410 | 157-250 (busy=1, act 0..1, no op energy) |

| component | over busy-spin: (P_full - P_spin)/thr, pJ (min-max) | all-in duty slope (full vs 50%) | half-SM slope | full: W, throughput/s, SM MHz | kiln A100 (V_nom 0.75 V) |
|---|---|---|---|---|---|
| ffma (fp32_fma) | 8.47 (8.12-8.74) | 17.9 (17.2-18.5) | 17.7 (16.7-18.2) | 246 W, 9.729e+12, 1410 | fp32 op 0.766; with RF 12 B rd + 4 B wr: 6.16 |
| smem_read (smem_byte) | 2.6 (2.51-2.68); LOP3 share <= 1.06 | 6.92 (6.85-6.98) | 7.01 (6.86-7.27) | 215 W, 1.944e+13, 1410 | 0.160 (+RF wr 0.362) |
| l1_read (l1_byte) | 3.74 (3.63-3.81); LOP3 share <= 1.06 | 8 (7.97-8.03) | 8.13 (7.83-8.45) | 236 W, 1.937e+13, 1410 | 0.160 (+RF wr 0.362) |
| l2_read (l2_byte) | 24.8 (24-26.4) | 52.7 (50.5-55.2) | 108 (81.9-135) | 239 W, 3.010e+12, 1410 | 0.863 (+NoC, +L1 fill) |
| hbm_read (hbm_byte) | 79.2 (77.3-82.8) | 132 (127-137) | 261 (230-316) | 282 W, 1.491e+12, 1410 | 36.8 (+L2 path) |
| hbm_copy (hbm_byte) | 68.6 (67.6-70.6) | 127 (126-129) | 75.5 (-77.4-367) | 258 W, 1.372e+12, 1410 | 36.8 (+L2 path) |
| gemm_bf16 (mac) | 1.75 (1.73-1.77); minus ncu-counted SMEM/L2/HBM traffic: 0.86 | 1.51 (1.49-1.54); 50% vs 25%: 3.09 (2.97-3.25) | - | 396 W, 1.321e+14, 1256 | 0.123 per MAC (datapath only) |
| gemm_fp16 (mac) | 1.85 (1.83-1.86) | 1.63 (1.56-1.67) | - | 401 W, 1.279e+14, 1223 | 0.182 per MAC (datapath only) |
| gemm_int8 (mac) | 1.17 (1.15-1.19); minus ncu-counted SMEM/L2/HBM traffic: 0.38 | 1.03 (0.989-1.05) | - | 397 W, 1.989e+14, 1281 | 0.045 per MAC (datapath only) |
| gemm_tf32 (mac) | 3.68 (3.65-3.7) | 3.11 (2.93-3.21) | - | 397 W, 6.344e+13, 1216 | 0.210 per MAC (datapath only) |
