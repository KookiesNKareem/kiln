"""Llama-3-8B-shaped matmul ops as single-op ONNX files Stream can parse.

Projections are Gemm with a shape-only bf16 weight initializer (as make_gemm.py). Attention is a
batched MatMul with both operands as graph inputs (the KV cache / activations live off-chip). GQA is
exact: the 4 query heads sharing a KV head are folded into M, so batch = batch_size * n_kv_heads.
"""

from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from make_gemm import make_gemm

BF16 = TensorProto.BFLOAT16
DTYPE_BYTES = 2


@dataclass(frozen=True)
class ModelShape:
    name: str = "llama3_8b"
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 14336
    vocab: int = 128256


LLAMA3_8B = ModelShape()


@dataclass(frozen=True)
class Op:
    """C[b, m, n] = A[b, m, k] @ B[b?, k, n]; B is a weight (shared over b) iff weight."""

    name: str
    m: int
    n: int
    k: int
    batch: int = 1
    weight: bool = True
    count: int = 1

    @property
    def key(self) -> str:
        return f"gemm_{self.m}_{self.n}_{self.k}" if self.weight else f"bmm_{self.batch}_{self.m}_{self.n}_{self.k}"

    @property
    def flops(self) -> int:
        return 2 * self.batch * self.m * self.n * self.k

    @property
    def bytes_a(self) -> int:
        return self.batch * self.m * self.k * DTYPE_BYTES

    @property
    def bytes_b(self) -> int:
        return (1 if self.weight else self.batch) * self.k * self.n * DTYPE_BYTES

    @property
    def bytes_out(self) -> int:
        return self.batch * self.m * self.n * DTYPE_BYTES


def prefill_ops(seq: int = 2048, batch: int = 1, s: ModelShape = LLAMA3_8B) -> list[Op]:
    t, g, L = seq * batch, s.n_heads // s.n_kv_heads, s.n_layers
    qkv = (s.n_heads + 2 * s.n_kv_heads) * s.head_dim
    return [
        Op("qkv", t, qkv, s.d_model, count=L),
        Op("attn_score", g * seq, seq, s.head_dim, batch=batch * s.n_kv_heads, weight=False, count=L),
        Op("attn_av", g * seq, s.head_dim, seq, batch=batch * s.n_kv_heads, weight=False, count=L),
        Op("o_proj", t, s.d_model, s.n_heads * s.head_dim, count=L),
        Op("ffn_gate_up", t, s.d_ff, s.d_model, count=2 * L),
        Op("ffn_down", t, s.d_model, s.d_ff, count=L),
        Op("lm_head", batch, s.vocab, s.d_model),
    ]


def decode_ops(batch: int, kv_len: int = 2048, s: ModelShape = LLAMA3_8B) -> list[Op]:
    g, L = s.n_heads // s.n_kv_heads, s.n_layers
    qkv = (s.n_heads + 2 * s.n_kv_heads) * s.head_dim
    return [
        Op("qkv", batch, qkv, s.d_model, count=L),
        Op("attn_score", g, kv_len, s.head_dim, batch=batch * s.n_kv_heads, weight=False, count=L),
        Op("attn_av", g, s.head_dim, kv_len, batch=batch * s.n_kv_heads, weight=False, count=L),
        Op("o_proj", batch, s.d_model, s.n_heads * s.head_dim, count=L),
        Op("ffn_gate_up", batch, s.d_ff, s.d_model, count=2 * L),
        Op("ffn_down", batch, s.d_model, s.d_ff, count=L),
        Op("lm_head", batch, s.vocab, s.d_model),
    ]


def build_onnx(op: Op, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{op.key}.onnx"
    if path.exists():
        return path
    if op.weight:
        return make_gemm(op.m, op.n, op.k, out_dir)
    b, m, n, k = op.batch, op.m, op.n, op.k
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"], name="MatMul")],
        op.key,
        [helper.make_tensor_value_info("A", BF16, [b, m, k]), helper.make_tensor_value_info("B", BF16, [b, k, n])],
        [helper.make_tensor_value_info("Y", BF16, [b, m, n])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path
