"""TPU v6e HBM diagnostics (decode GEMV shapes, read-size sweep, chained decode layers); GEMMs: diag_v6e_modes.py.

Usage: python diag_v6e.py out.json [parts]   parts: comma list of gemv,read,chain (default all)
Timer: slope of a jitted fori_loop between K1 and K2 iterations over R distinct operand copies (as tpu_bench.py
`loop`), 5 pairs, median; the first operand is perturbed by the previous iteration's output so nothing is hoisted.
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
PARTS = (sys.argv[2] if len(sys.argv) > 2 else "gemv,read,chain").split(",")
MiB, GiB = 2**20, 2**30
BF16, F32 = jnp.bfloat16, jnp.float32
DEV = jax.devices()[0]
RES = {"device": DEV.device_kind, "jax": jax.__version__, "records": []}
H, FF, V, NQ, NKV, HD = 4096, 14336, 128256, 32, 8, 128


def dump():
    json.dump(RES, open(OUT, "w"), indent=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def block(x):
    return jax.block_until_ready(x)


def fill(R, shape, seed):
    i = lax.iota(jnp.uint32, R * math.prod(shape)).reshape((R, *shape))
    h = (i + jnp.uint32(seed)) * jnp.uint32(2654435761)
    h = h ^ (h >> 15)
    return ((h >> 8).astype(F32) * (2.0 / 2**24) - 1.0).astype(BF16)


def loop_time(f, shapes, nbytes, R, target_s=0.03, pairs=5, k_min=8, est=None):
    stacks = [jax.jit(fill, static_argnums=(0, 1, 2))(R, tuple(s), 7 + j) for j, s in enumerate(shapes)]

    @jax.jit
    def run(K, stacks):
        out0 = jax.eval_shape(f, *[s[0] for s in stacks])

        def body(i, s):
            args = [lax.dynamic_index_in_dim(st, i % R, keepdims=False) for st in stacks]
            args[0] = args[0] + s.astype(BF16)
            o = f(*args)
            return s + jax.tree.leaves(o)[0].reshape(-1)[0].astype(F32) * 1e-30
        del out0
        return lax.fori_loop(0, K, body, jnp.zeros((), F32))

    est = est or max(nbytes / 1.6e12, 2e-6)
    K2 = int(min(4000, max(k_min, target_s / est)))
    K1 = max(2, K2 // 4)
    block(run(K1, stacks))
    block(run(K2, stacks))
    sl = []
    for _ in range(pairs):
        t = time.perf_counter(); block(run(K1, stacks)); a = time.perf_counter() - t
        t = time.perf_counter(); block(run(K2, stacks)); b = time.perf_counter() - t
        sl.append((b - a) / (K2 - K1))
    del stacks
    med = statistics.median(sl)
    return {"median_s": med, "cv": statistics.pstdev(sl) / med, "K1": K1, "K2": K2, "R": R, "samples": sl}


def rec(name, f, shapes, nbytes, foot=512 * MiB, **kw):
    R = max(1, math.ceil(foot / nbytes))
    R = min(R, max(1, int(0.3 * 31 * GiB / nbytes)))
    try:
        r = loop_time(f, shapes, nbytes, R, **kw)
    except Exception as e:
        r = {"error": repr(e)[:300]}
    r.update(name=name, bytes=nbytes)
    if "median_s" in r:
        r["gbps"] = nbytes / r["median_s"] / 1e9
        log(f"{name:<44} {r['median_s']*1e6:9.2f} us  {r['gbps']:7.0f} GB/s  cv={r['cv']:.3f} R={R}")
    else:
        log(name, r["error"])
    RES["records"].append(r)
    dump()
    return r


def gemv_part():
    for m in (1, 8, 32):
        for k, n in ((4096, 6144), (4096, 4096), (4096, 14336), (14336, 4096), (4096, 28672), (4096, 3072), (4096, 8192),
                     (4096, 10240), (4096, 20480), (4096, 57344)):
            if m > 1 and (k, n) not in ((4096, 6144), (4096, 4096), (14336, 4096)):
                continue
            nb = 2 * (m * k + k * n + m * n)
            rec(f"gemv/kn/{m}x{k}x{n}", lambda x, w: x @ w, [(m, k), (k, n)], nb)
    # lm_head as in the step: logits = x @ W.T with W stored [V, H]
    for m in (1, 8):
        nb = 2 * (m * H + V * H + m * V)
        rec(f"gemv/nk/{m}x{H}x{V}", lambda x, w: x @ w.T, [(m, H), (V, H)], nb)
    # footprint sensitivity at decode sizes: same op, 2 GiB of copies instead of 512 MiB
    for k, n in ((4096, 4096), (4096, 14336)):
        nb = 2 * (k + k * n + n)
        rec(f"gemv/kn/1x{k}x{n}/foot2G", lambda x, w: x @ w, [(1, k), (k, n)], nb, foot=2 * GiB)


def read_part():
    for s in (16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 768, 1024, 2048):
        nb = s * MiB
        rec(f"read/sum_rows1024/{s}MiB", lambda x: jnp.sum(x, dtype=F32), [(nb // 2048, 1024)], nb)
    for s in (64, 256, 512, 1024):
        nb = s * MiB
        rec(f"read/colsum_rows1024/{s}MiB", lambda x: jnp.sum(x, axis=0, dtype=F32), [(nb // 2048, 1024)], nb)
        rec(f"read/sum_rows8192/{s}MiB", lambda x: jnp.sum(x, dtype=F32), [(nb // 16384, 8192)], nb)
    # one 512 MiB buffer read as 4 x 128 MiB slices in one program vs one kernel
    nb = 512 * MiB
    rec("read/sum_512MiB_as_4x128", lambda x: sum(jnp.sum(x[i * 65536:(i + 1) * 65536], dtype=F32) for i in range(4)),
        [(nb // 2048, 1024)], nb)
    for s in (128, 512):
        nb = s * MiB
        rec(f"read/sum_rows1024/{s}MiB/foot2G", lambda x: jnp.sum(x, dtype=F32), [(nb // 2048, 1024)], nb, foot=2 * GiB)


def chain_part():
    """Decode-layer GEMV chain (no attention): qkv, o, gate, up, silu*mul, down, residuals; L distinct layers."""
    def mk(L, m):
        def f(x, *W):
            for i in range(L):
                wqkv, wo, wg, wu, wd = W[5 * i:5 * i + 5]
                h = x * lax.rsqrt(jnp.mean(x.astype(F32) ** 2, -1, keepdims=True) + 1e-5).astype(BF16)
                q = (h @ wqkv)[:, :H]
                x = x + q @ wo
                h = x * lax.rsqrt(jnp.mean(x.astype(F32) ** 2, -1, keepdims=True) + 1e-5).astype(BF16)
                x = x + (jax.nn.silu(h @ wg) * (h @ wu)) @ wd
            return x
        return f
    per = [(H, (NQ + 2 * NKV) * HD), (H, H), (H, FF), (H, FF), (FF, H)]
    for L in (1, 4, 16):
        for m in (1,):
            shapes = [(m, H)] + per * L
            nb = 2 * L * sum(a * b for a, b in per)
            rec(f"chain/L{L}/m{m}", mk(L, m), shapes, nb, foot=nb, k_min=4)
    # weight streams only (independent GEMVs, no dependency between them) for one layer
    def ind(x, a, b, c, d, e):
        return (x @ a)[:, :8].sum() + (x @ b)[:, :8].sum() + (x @ c)[:, :8].sum() + (x @ d)[:, :8].sum() + (jnp.tile(x, (1, 4))[:, :FF] @ e)[:, :8].sum()
    rec("chain/L1/independent", ind, [(1, H)] + per, 2 * sum(a * b for a, b in per), foot=0)


if __name__ == "__main__":
    log(DEV, getattr(DEV, "num_cores", None), DEV.memory_stats().get("bytes_limit"))
    t0 = time.time()
    for p in PARTS:
        {"gemv": gemv_part, "read": read_part, "chain": chain_part}[p]()
        log(f"part {p} done at {time.time() - t0:.0f}s")
    RES["done"] = True
    dump()
