"""Reference designs. `python -m harness.designs.reference` regenerates the JSON next to this file.

A100-SXM4-80GB: NVIDIA A100 whitepaper (2020) and Jouppi et al., "TPU v4", ISCA 2023, Table 5:
108 SMs, 1410 MHz boost, 312 TFLOPS dense bf16, 40 MiB L2, 80 GiB HBM2e at 2039 GB/s, 7 nm, 826 mm2,
400 W. Mirrors hw/a100/a100.yaml exactly (16 SM clusters of 6.75 SMs, 4 L2 slices on one crossbar
bus); see the comments there for why. 5 active HBM2e stacks (6 sites).

TPU v4: Jouppi et al., ISCA 2023 (arXiv 2304.01433), Sec. 2 and Table 4; Google Cloud TPU v4 docs.
2 TensorCores x 4 MXUs of 128x128 bf16, 1050 MHz -> 2*4*16384*2*1.05e9 = 275.3 TFLOPS (published
275). Per TC a 16 MiB VMEM and a VPU of 128 lanes x 16 ALUs; the two TCs share a 128 MiB CMEM.
0.25 MiB register file per core. 32 GiB HBM2 at 1200 GB/s, 4 stacks. 7 nm, < 600 mm2, 192 W max.
Assumed (unpublished): VMEM feed bandwidth, scaled from the TPU v7 Stream example (57 TB/s per TC for
8x256^2 MXUs) by array edge length (systolic I/O scales with rows+cols) -> 14.3 TB/s per TC;
CMEM<->VMEM bus at half the VMEM rate.
Stream-driven modelling choices:
  * Each TC's published 16 MiB VMEM is split as 4 x 2 MiB MXU-local staging + 8 MiB shared VMEM core.
    Stream's capacity tiler fits tiles into a compute core's local buffer (50% fill, <= 1024
    steady-state tiles) and silently gives up otherwise, so a 512 KiB MXU buffer made every large
    prefill GEMM infeasible; the split keeps total on-chip SRAM equal to the published figure.
  * One bus per TensorCore (scope="per_memory"), as on silicon. Before fork commit cd20655 Stream
    routed every VMEM to every MXU and merged equal-bandwidth buses, halving decode throughput here.

TPU v5e (device_kind "TPU v5 lite"): Google Cloud TPU v5e docs (docs.cloud.google.com/tpu/docs/v5e):
1 TensorCore per chip with 4 MXUs, a vector unit and a scalar unit; 197 TFLOPS bf16, 16 GiB HBM at
819 GB/s. 128x128 MXUs (Cloud TPU system-architecture page: "128 x 128 (TPU versions prior to v6e)").
128 MiB VMEM per TensorCore (JAX Pallas TPU hardware reference). Clock derived, not published:
197e12 / (4 * 128^2 * 2) = 1503 MHz -> 1500 MHz gives 196.6 TFLOPS. No CMEM.
TPU v6e / Trillium (device_kind "TPU v6 lite"): Google Cloud TPU v6e docs: 1 TensorCore per chip,
918 TFLOPS bf16, 32 GiB HBM at 1640 GB/s (Pallas reference: 1640 GB/s, 128 MiB VMEM); 256x256 MXUs
(system-architecture page). MXU count conflicts: the v6e page says 2 MXUs per TensorCore, which needs
a 3.5 GHz clock for 918 TFLOPS; we use 4 x 256^2 at 1750 MHz -> 917.5 TFLOPS (the clock cited by
third-party spec sheets, and Google's "4.7x over v5e from larger MXUs and higher clock" = 4x MACs x
1.75/1.5). SparseCores omitted (no dense-matmul role).
Assumed for v5e/v6e (unpublished): VMEM feed bandwidth = Ironwood Stream example's 416974 bits/cycle
for 8x256 MXU edge, scaled by MXU edge length (4x128 -> 1/4, 4x256 -> 1/2) at each chip's own clock:
19.5 TB/s (v5e), 45.6 TB/s (v6e); cross-check "VMEM ~22x HBM" (scaling book) gives 18 / 36 TB/s.
VPU 8 sublanes x 128 lanes (Pallas vreg shape), register file and VPU buffer as TPU v4. Tech node n5
HBM generation (HBM2e / HBM3) and 2 stacks for both. VMEM split as TPU v4: 4 x 16 MiB MXU-local + 64 MiB shared core.
"""

from pathlib import Path

from harness.design import ComputeUnit, Design, Link, MemoryUnit, OffChip

A100_CLK = 1410.0
A100_XBAR_GBPS = 40960 / 8 * A100_CLK / 1e3


def a100() -> Design:
    return Design(
        name="a100",
        clock_mhz=A100_CLK,
        tech_node="n7",
        compute=[
            ComputeUnit("sm_cluster", "matrix", 72, 96, count=16, attach="l2", buffer_kib=3024,
                        buffer_gbps=27648 / 8 * A100_CLK / 1e3),
            ComputeUnit("cuda_cores", "vector", 27, 64, count=4, attach="l2", precision="fp32", buffer_kib=2048,
                        buffer_gbps=A100_XBAR_GBPS, regfile_kib=256, regfile_gbps=52131 / 8 * A100_CLK / 1e3),
        ],
        memory=[MemoryUnit("l2", 10, A100_XBAR_GBPS, count=4)],
        offchip=OffChip(80, 2039, attach=["l2"], kind="HBM2e", stacks=5),
        links=[Link("l2_xbar", ["sm_cluster", "cuda_cores", "l2"], A100_XBAR_GBPS)],
        notes="NVIDIA A100-SXM4-80GB; 16 tensor-core clusters of 6.75 SMs; 4 L2 slices on one crossbar bus.",
    )


def a100_40gb() -> Design:
    d = a100()
    d.name = "a100_40gb"
    d.offchip = OffChip(40, 1555, attach=["l2"], kind="HBM2", stacks=5)
    d.notes = "NVIDIA A100-SXM4-40GB (the Colab A100): as a100 but 40 GiB HBM2 at 1555 GB/s."
    return d


TPU4_CLK = 1050.0
TPU4_VMEM_GBPS = 416974 / 8 * 1.1 * (4 * 128) / (8 * 256)


def tpuv4() -> Design:
    return Design(
        name="tpuv4",
        clock_mhz=TPU4_CLK,
        tech_node="n7",
        compute=[
            ComputeUnit("mxu", "matrix", 128, 128, count=8, attach="vmem", buffer_kib=2048, buffer_gbps=TPU4_VMEM_GBPS),
            ComputeUnit("vpu", "vector", 16, 128, count=2, attach="vmem", precision="fp32", buffer_kib=512,
                        buffer_gbps=TPU4_VMEM_GBPS, regfile_kib=256, regfile_gbps=TPU4_VMEM_GBPS),
        ],
        memory=[MemoryUnit("vmem", 8, TPU4_VMEM_GBPS, count=2), MemoryUnit("cmem", 128, TPU4_VMEM_GBPS / 2)],
        offchip=OffChip(32, 1200, attach=["vmem", "cmem"], kind="HBM2", stacks=4),
        links=[
            Link("tc_bus", ["mxu", "vpu", "vmem"], TPU4_VMEM_GBPS, scope="per_memory"),
            Link("cmem_bus", ["vmem", "cmem"], TPU4_VMEM_GBPS / 2),
        ],
        notes="Google TPU v4: 2 TensorCores x (4 MXU 128x128 + VPU + 16 MiB VMEM), shared 128 MiB CMEM.",
    )


IRONWOOD_VMEM_BITS_PER_EDGE = 416974 / (8 * 256)


def _tpu_single_tc(name: str, clock_mhz: float, mxu_edge: int, hbm_gib: float, hbm_gbps: float, kind: str,
                   notes: str) -> Design:
    vmem_gbps = IRONWOOD_VMEM_BITS_PER_EDGE * 4 * mxu_edge / 8 * clock_mhz / 1e3
    return Design(
        name=name,
        clock_mhz=clock_mhz,
        tech_node="n5",
        compute=[
            ComputeUnit("mxu", "matrix", mxu_edge, mxu_edge, count=4, attach="vmem", buffer_kib=16384,
                        buffer_gbps=vmem_gbps),
            ComputeUnit("vpu", "vector", 8, 128, count=1, attach="vmem", precision="fp32", buffer_kib=512,
                        buffer_gbps=vmem_gbps, regfile_kib=256, regfile_gbps=vmem_gbps),
        ],
        memory=[MemoryUnit("vmem", 64, vmem_gbps)],
        offchip=OffChip(hbm_gib, hbm_gbps, attach=["vmem"], kind=kind, stacks=2),
        links=[Link("tc_bus", ["mxu", "vpu", "vmem"], vmem_gbps)],
        notes=notes,
    )


def tpuv5e() -> Design:
    return _tpu_single_tc("tpuv5e", 1500.0, 128, 16, 819, "HBM2e",
                          "Google TPU v5e: 1 TensorCore x (4 MXU 128x128 + VPU + 128 MiB VMEM), 16 GiB HBM.")


def tpuv6e() -> Design:
    return _tpu_single_tc("tpuv6e", 1750.0, 256, 32, 1640, "HBM3",
                          "Google TPU v6e Trillium: 1 TensorCore x (4 MXU 256x256 + VPU + 128 MiB VMEM), 32 GiB HBM.")


REFERENCES = {"a100": a100, "a100_40gb": a100_40gb, "tpuv4": tpuv4, "tpuv5e": tpuv5e, "tpuv6e": tpuv6e}

if __name__ == "__main__":
    for name, fn in REFERENCES.items():
        (Path(__file__).parent / f"{name}.json").write_text(fn().validate().to_json() + "\n")
