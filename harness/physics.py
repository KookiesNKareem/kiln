"""First-order die area and peak power, plus an envelope check.

ALL CONSTANTS ARE PLACEHOLDERS to be replaced by ASAP7-calibrated numbers. Sources:
  [H14] M. Horowitz, "Computing's energy problem", ISSCC 2014: 45 nm energy (pJ) and area (um^2) of
        fp16 mul 1.1/1640, fp32 add 0.9/4184, fp32 mul 3.7/7700, int8 mul 0.2/282, int32 add 0.1/137.
  [WC]  TSMC logic density (WikiChip): N16 28.9, N7 91.2, N5 138.2, N3 ~197 MTr/mm^2.
  [SR]  TSMC high-density 6T bitcell: N16 0.074, N7 0.027, N5 0.021, N3 0.0199 um^2 (IEDM 2014/2016/
        2019/2022 platform papers).
  [TS]  TSMC press: N7 -60% power vs N16, N5 -30% vs N7, N3 -25..30% vs N5 at iso speed.
  [OC]  O'Connor et al., "Fine-grained DRAM", MICRO 2017: HBM2 ~3.9 pJ/bit.
Assumed without a source (placeholders): 45 nm -> N7 MAC area /20 and energy x0.25, PE overhead 1.5x,
SRAM array efficiency 0.4, SRAM 0.1 pJ/bit at N7, 50% SRAM-bandwidth activity at peak, HBM PHY
12 mm^2 and 7 mm shoreline per stack, uncore (NoC, control, IO, PCIe/ICI) 1.6x of the modelled area,
leakage 0.05 W/mm^2.
"""

from __future__ import annotations

import math

from harness.design import Design

N7_MAC = {  # (um^2, pJ) per MAC with fp32 (int32) accumulate, at N7, incl. PE overhead
    "bf16": ((1640 + 4184) / 20 * 1.5, (1.1 + 0.9) * 0.25),
    "fp16": ((1640 + 4184) / 20 * 1.5, (1.1 + 0.9) * 0.25),
    "fp8": ((900 + 4184) / 20 * 1.5, (0.6 + 0.9) * 0.25),
    "int8": ((282 + 137) / 20 * 1.5, (0.2 + 0.1) * 0.25),
    "int4": ((90 + 137) / 20 * 1.5, (0.07 + 0.1) * 0.25),
    "fp32": ((7700 + 4184) / 20 * 1.5, (3.7 + 0.9) * 0.25),
}
NODE = {  # logic area factor vs N7 [WC], energy factor vs N7 [TS], SRAM bitcell um^2 [SR]
    "n16": (91.2 / 28.9, 2.5, 0.074),
    "n7": (1.0, 1.0, 0.027),
    "n5": (91.2 / 138.2, 0.7, 0.021),
    "n3": (91.2 / 197.0, 0.5, 0.0199),
}
SRAM_ARRAY_EFF = 0.4
SRAM_PJ_PER_BIT_N7 = 0.1
SRAM_ACTIVITY = 0.5
HBM_PJ_PER_BIT = 3.9
HBM_PHY_MM2 = 12.0
HBM_SHORELINE_MM = 7.0
UNCORE_FACTOR = 1.6
LEAKAGE_W_PER_MM2 = 0.05

ENVELOPE = {"die_mm2": 826.0, "power_w": 400.0}


def estimate(d: Design) -> dict:
    area_f, energy_f, bitcell = NODE[d.tech_node]
    mac_mm2 = sum(c.macs * N7_MAC[c.precision][0] for c in d.compute) * area_f / 1e6
    mac_w = sum(c.macs * N7_MAC[c.precision][1] for c in d.compute) * energy_f * 1e-12 * d.clock_hz
    sram_bits = d.onchip_bytes * 8
    sram_mm2 = sram_bits * bitcell / SRAM_ARRAY_EFF / 1e6
    sram_bw_bits = 8e9 * (sum(2 * m.bandwidth_gbps * m.count for m in d.memory)
                          + sum((c.buffer_gbps + c.regfile_gbps) * c.count for c in d.compute))
    sram_w = sram_bw_bits * SRAM_ACTIVITY * SRAM_PJ_PER_BIT_N7 * energy_f * 1e-12
    phy_mm2 = d.offchip.stacks * HBM_PHY_MM2
    logic_mm2 = (mac_mm2 + sram_mm2 + phy_mm2) * UNCORE_FACTOR
    # HBM PHYs sit on the two long edges; a die too small for its stacks grows to fit them.
    shoreline_mm2 = (d.offchip.stacks * HBM_SHORELINE_MM / 2) ** 2
    die_mm2 = max(logic_mm2, shoreline_mm2)
    hbm_w = d.offchip_bytes_per_s * 8 * HBM_PJ_PER_BIT * 1e-12
    power_w = mac_w + sram_w + hbm_w + die_mm2 * LEAKAGE_W_PER_MM2
    out = {
        "die_mm2": die_mm2, "mac_mm2": mac_mm2, "sram_mm2": sram_mm2, "hbm_phy_mm2": phy_mm2,
        "shoreline_limited": shoreline_mm2 > logic_mm2,
        "power_w": power_w, "mac_w": mac_w, "sram_w": sram_w, "hbm_w": hbm_w,
        "peak_tflops_per_w": d.peak_flops() / 1e12 / power_w,
    }
    out["violations"] = [f"{k} {out[k]:.0f} > envelope {v:.0f}" for k, v in ENVELOPE.items()
                         if not math.isfinite(out[k]) or out[k] > v]
    return out
