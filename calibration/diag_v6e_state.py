"""TPU compute-state probe: every ~5 s for DURATION s, in-program 8192^3 bf16 GEMM TFLOP/s and 256 MiB read GB/s.

Usage: python diag_v6e_state.py out.json [duration_s]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

F32, BF16 = jnp.float32, jnp.bfloat16
OUT, DUR = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 180


def fill(R, shape, seed):
    i = lax.iota(jnp.uint32, R * shape[0] * shape[1]).reshape((R, *shape))
    h = (i + jnp.uint32(seed)) * jnp.uint32(2654435761)
    return (((h ^ (h >> 15)) >> 8).astype(F32) * (2.0 / 2**24) - 1.0).astype(BF16)


def looped(f, stacks, R):
    @jax.jit
    def run(K, st):
        def body(i, s):
            o = lax.optimization_barrier(f(*[lax.dynamic_index_in_dim(x, i % R, keepdims=False) for x in st]))
            return s + o.reshape(-1)[0].astype(F32) * 1e-30
        return lax.fori_loop(0, K, body, jnp.zeros((), F32))

    def t(K1, K2):
        jax.block_until_ready(run(K1, stacks)); jax.block_until_ready(run(K2, stacks))
        a = time.perf_counter(); jax.block_until_ready(run(K1, stacks)); a = time.perf_counter() - a
        b = time.perf_counter(); jax.block_until_ready(run(K2, stacks)); b = time.perf_counter() - b
        return (b - a) / (K2 - K1)
    return t


g = looped(lambda a, b: a @ b, [fill(2, (8192, 8192), 1), fill(2, (8192, 8192), 2)], 2)
r = looped(lambda x: jnp.sum(x, axis=0, dtype=F32), [fill(4, (131072, 1024), 3)], 4)
res = {"device": jax.devices()[0].device_kind, "jax": jax.__version__, "samples": []}
t0 = time.time()
while time.time() - t0 < DUR:
    tg = g(5, 25)
    tr = r(20, 100)
    s = {"t_s": round(time.time() - t0, 1), "gemm_tflops": 2 * 8192**3 / tg / 1e12, "read_gbps": 2**28 / tr / 1e9}
    res["samples"].append(s)
    print(time.strftime("%H:%M:%S"), f"{s['t_s']:6.1f}s gemm {s['gemm_tflops']:6.1f} TF  read {s['read_gbps']:6.0f} GB/s", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    time.sleep(4)
