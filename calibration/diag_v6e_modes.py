"""TPU timing-mode diagnostics: one op timed as (a) in-program loop without input perturbation (output kept live by an
optimization barrier), (b) pipelined over the same buffers, (c) pipelined over R distinct buffers (tpu_bench
`pipelined_rot`), (d) in-program loop issuing 2 independent copies of the op per iteration.

Usage: python diag_v6e_modes.py out.json
"""
import json
import math
import statistics
import sys
import time

import jax
import jax.numpy as jnp
from jax import lax

OUT = sys.argv[1]
MiB = 2**20
BF16, F32 = jnp.bfloat16, jnp.float32
RES = {"device": jax.devices()[0].device_kind, "jax": jax.__version__, "records": []}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def block(x):
    return jax.block_until_ready(x)


def fill(R, shape, seed):
    i = lax.iota(jnp.uint32, R * math.prod(shape)).reshape((R, *shape))
    h = (i + jnp.uint32(seed)) * jnp.uint32(2654435761)
    h = h ^ (h >> 15)
    return ((h >> 8).astype(F32) * (2.0 / 2**24) - 1.0).astype(BF16)


def slope(run, args, est, target=0.05, pairs=5):
    K2 = int(min(4000, max(8, target / est)))
    K1 = max(2, K2 // 4)
    block(run(K1, *args)); block(run(K2, *args))
    sl = []
    for _ in range(pairs):
        t = time.perf_counter(); block(run(K1, *args)); a = time.perf_counter() - t
        t = time.perf_counter(); block(run(K2, *args)); b = time.perf_counter() - t
        sl.append((b - a) / (K2 - K1))
    return statistics.median(sl), statistics.pstdev(sl) / statistics.median(sl)


def first(o):
    return jax.tree.leaves(o)[0].reshape(-1)[0].astype(F32)


def modes(name, f, shapes, nbytes, flops, est):
    R = max(2, min(16, math.ceil(512 * MiB / nbytes)))
    stacks = [jax.jit(fill, static_argnums=(0, 1, 2))(R, tuple(s), 3 + j) for j, s in enumerate(shapes)]
    r = {"name": name, "bytes": nbytes, "flops": flops, "R": R}

    def loopn(copies):
        @jax.jit
        def run(K, stacks):
            def body(i, s):
                for c in range(copies):
                    args = [lax.dynamic_index_in_dim(st, (copies * i + c) % R, keepdims=False) for st in stacks]
                    s = s + first(lax.optimization_barrier(f(*args))) * 1e-30
                return s
            return lax.fori_loop(0, K, body, jnp.zeros((), F32))
        return run
    for copies in (1, 2):
        try:
            t, cv = slope(loopn(copies), (stacks,), est * copies)
            r[f"loop_x{copies}"] = {"per_op_s": t / copies, "cv": cv}
        except Exception as e:
            r[f"loop_x{copies}"] = {"error": repr(e)[:200]}
    fj = jax.jit(f)
    sets = [[st[j] for st in stacks] for j in range(R)]
    for label, pick in (("pipelined_same", lambda i: sets[0]), ("pipelined_rot", lambda i: sets[i % R])):
        for p in sets:
            block(fj(*p))
        n = int(min(4000, max(R, 0.05 / est)))
        out = []
        for _ in range(5):
            t = time.perf_counter()
            for i in range(n):
                o = fj(*pick(i))
            block(o)
            out.append((time.perf_counter() - t) / n)
        r[label] = {"per_op_s": statistics.median(out), "cv": statistics.pstdev(out) / statistics.median(out)}
    del sets, stacks
    msg = " ".join(f"{k}={v['per_op_s']*1e6:8.1f}us({nbytes / v['per_op_s'] / 1e9:5.0f}GB/s,{flops / v['per_op_s'] / 1e12:5.0f}TF)"
                   for k, v in r.items() if isinstance(v, dict) and "per_op_s" in v)
    log(f"{name:<22} {msg}")
    RES["records"].append(r)
    json.dump(RES, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    for k, n in ((4096, 4096), (14336, 4096), (4096, 14336)):
        modes(f"gemv_1x{k}x{n}", lambda x, w: x @ w, [(1, k), (k, n)], 2 * (k + k * n + n), 2 * k * n, 2 * k * n / 1.1e12)
    nb = 256 * MiB
    modes("read_256MiB", lambda x: jnp.sum(x, dtype=F32), [(nb // 2048, 1024)], nb, nb // 2, nb / 1.1e12)
    for m, n, k in ((2048, 4096, 4096), (2048, 6144, 4096), (2048, 14336, 4096), (2048, 4096, 14336), (4096, 4096, 4096),
                    (8192, 8192, 8192), (8192, 4096, 4096)):
        modes(f"gemm_{m}x{n}x{k}", lambda a, b: a @ b, [(m, k), (k, n)], 2 * (m * k + k * n + m * n), 2 * m * n * k,
              2 * m * n * k / 800e12)
    # sustained 8192^3 for ~2 s in-program
    st = [fill(2, (8192, 8192), 1), fill(2, (8192, 8192), 2)]

    @jax.jit
    def run(K, st):
        def body(i, s):
            a, b = (lax.dynamic_index_in_dim(x, i % 2, keepdims=False) for x in st)
            return s + first(lax.optimization_barrier(a @ b)) * 1e-30
        return lax.fori_loop(0, K, body, jnp.zeros((), F32))
    for tgt in (0.02, 2.0):
        t, cv = slope(run, (st,), 2 * 8192**3 / 800e12, target=tgt, pairs=3)
        log(f"gemm_8192^3 sustained {tgt}s: {t*1e6:.1f}us {2*8192**3/t/1e12:.0f}TF cv={cv:.3f}")
        RES["records"].append({"name": f"gemm_8192^3_sustain_{tgt}s", "per_op_s": t, "cv": cv})
    RES["done"] = True
    json.dump(RES, open(OUT, "w"), indent=1)
