"""Workload suites: Llama-3-8B layer ops for prefill and decode, each with its roofline quantities.

v1 approximation: every op is its own single-operator ONNX, evaluated in isolation (inputs start in
off-chip memory, nothing stays resident between ops), and a phase's time is sum(count * op time).
No inter-op fusion or overlap, no elementwise ops (RMSNorm, RoPE, SiLU*, softmax), causal masking
ignored (prefill attention is computed dense, 2x the useful FLOPs). Prefill attention is split per
KV head (Stream finds the 8-head batched AV matmul structurally infeasible; heads are independent).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workloads"))

from llm import LLAMA3_8B, Op, build_onnx, decode_ops, prefill_ops  # noqa: E402

ONNX_DIR = ROOT / "workloads" / "build"
WORD_BYTES = 2


@dataclass(frozen=True)
class Workload:
    phase: str  # e.g. "prefill_b1", "decode_b8"
    tokens: int  # tokens produced per phase execution (prefill: seq*batch, decode: batch)
    op: Op

    @property
    def name(self) -> str:
        return f"{self.phase}/{self.op.name}"

    @property
    def flops(self) -> int:
        return self.op.flops

    def compulsory_bytes(self, onchip_bytes: float) -> float:
        """Off-chip bytes no schedule can avoid: operands start off-chip, and outputs that cannot all
        stay on chip must leave. For matmuls also the Irony-Toledo-Tiskin I/O bound
        Q >= mnk / (2 sqrt(2 S)) - S words per independent product (S = all on-chip words)."""
        o = self.op
        s_words = onchip_bytes / WORD_BYTES
        itt = o.batch * max(0.0, o.m * o.n * o.k / (2 * math.sqrt(2 * s_words)) - s_words) * WORD_BYTES
        return max(o.bytes_a + o.bytes_b + max(0.0, o.bytes_out - onchip_bytes), itt)

    def onnx(self) -> Path:
        return build_onnx(self.op, ONNX_DIR)


def _prefill(seq=2048, batch=1):
    ops = []
    for o in prefill_ops(seq, batch):
        if not o.weight:
            o = Op(o.name, o.m, o.n, o.k, batch=1, weight=False, count=o.count * o.batch)
        ops.append(Workload(f"prefill_b{batch}", seq * batch, o))
    return ops


def _decode(batches=(1, 8, 32), kv_len=2048):
    return [Workload(f"decode_b{b}", b, o) for b in batches for o in decode_ops(b, kv_len)]


SUITES = {
    # gemm_16x8192x8192 is the canary for Stream reporting single-op groups below the HBM floor.
    "smoke": lambda: [Workload("smoke", 1024, Op("gemm_1024", 1024, 1024, 1024)),
                      Workload("smoke", 16, Op("gemm_16_8192", 16, 8192, 8192))],
    "prefill": _prefill,
    "decode": _decode,
    "all": lambda: _prefill() + _decode(),
}


def suite(name: str) -> list[Workload]:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {list(SUITES)}")
    return SUITES[name]()


def resident_bytes(phase: str, kv_len: int = 2048) -> float:
    """Weights + KV cache that must fit in off-chip memory for a phase."""
    s = LLAMA3_8B
    params = s.n_layers * (s.d_model * (s.n_heads + 2 * s.n_kv_heads) * s.head_dim
                           + s.n_heads * s.head_dim * s.d_model + 3 * s.d_model * s.d_ff) + 2 * s.vocab * s.d_model
    batch = int(phase.split("_b")[1]) if "_b" in phase else 1
    kv = 2 * s.n_layers * batch * kv_len * s.n_kv_heads * s.head_dim
    return (params + kv) * WORD_BYTES
