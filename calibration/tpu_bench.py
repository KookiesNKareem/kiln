"""kiln TPU ground-truth runner (JAX, one chip), schema kiln.meas/1 (spec 06 §4).

Usage: python tpu_bench.py <micro|seq|suite> out.json [oplist.json]

  micro  calib-micro fit suite: HBM read/write/copy 1 MiB-4 GiB, VMEM-resident sweeps, GEMV/skinny and GEMM sweeps
         (LLM shape families dropped, 06 §3.4), batched-matmul sweep, elementwise sweep, dispatch/launch probes,
         session sanity.
  seq    whole-step `sequence`: Llama-3-8B decode b1/b8/b32 (kv 2048) and prefill b1 (seq 2048) as one jitted step
         per attention implementation (env SEQ_IMPLS, see IMPLS: unfused einsum f32/bf16 softmax, dot_product_attention xla,
         Pallas paged / ragged-paged / flash / splash; each checked against an f32 reference first; SEQ_PHASES filters)
         (embedding, all layers, final norm, lm_head, argmax; KV caches donated), timed in `loop` mode (fori_loop over
         steps, slope), plus `single` and `pipelined`; 1-layer steps. If the full model does not fit HBM the step uses
         the largest layer count that fits and records it (`n_layers`), so per-layer extrapolation is explicit.
  suite  LLM per-op suite (oplist.json), legacy GEMM sweep, isolated non-matmul ops of the step.

Timers per record: `single` (one jitted call, block each; includes host dispatch), `pipelined` (back-to-back calls on
the same buffers; untrusted: libtpu can keep VMEM-sized operands resident), `pipelined_rot` (cycling over distinct
operand copies), `loop` (slope of a jitted fori_loop between K1 and K2 iterations with a traced trip count; each
iteration reads a different operand copy and writes a different output slot, copies totalling >= 512 MiB, capped by
memory not count; the first operand is perturbed by a scalar read from the previous output so nothing is hoisted).
Canonical per-op mode = `loop` (in-program, like a scored whole step; 08 §F). `pipelined_rot` is diagnostic only (on
v6e it runs 20-27% faster than any single program, 08 §F.4); `best_s` = min(loop, pipelined_rot) is kept as a
diagnostic, `canonical_s` is the loop median. Samples are stored per mode (loop: one slope per K1/K2 pair).
Every suite records the chip-state sanity (8192^3 GEMM, 1 GiB copy, loop mode) at session start and end; kiln gates
compute-bound records of sessions whose sanity GEMM moved by more than 3% (08 §F chip-state gate).
"""

import datetime
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from functools import lru_cache, partial
from importlib import metadata

import jax
import jax.numpy as jnp
from jax import lax

SUITE, OUT = sys.argv[1], sys.argv[2]
OPLIST = sys.argv[3] if len(sys.argv) > 3 else "oplist.json"
MiB, GiB = 2**20, 2**30
N_SINGLE = 30
LOOP_TARGET_S = 0.03
LOOP_PAIRS = 7
ROTATE_BYTES = 512 * MiB
ROT_SLICES_MAX = 1024
SPEC = {"TPU v5 lite": (197e12, 819e9), "TPU v6 lite": (918e12, 1640e9), "TPU v6e": (918e12, 1640e9)}
BF16, F32 = jnp.bfloat16, jnp.float32

DEV = jax.devices()[0]
PEAK, BW = SPEC.get(DEV.device_kind, (200e12, 800e9))
try:
    HBM_LIMIT = DEV.memory_stats()["bytes_limit"]
except Exception:
    HBM_LIMIT = 15 * GiB


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return f"ERR {e}"


def ver(p):
    try:
        return metadata.version(p)
    except Exception:
        return None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stats(samples, extra=None):
    ts = sorted(samples)
    mean = statistics.fmean(ts)
    sd = statistics.pstdev(ts) if len(ts) > 1 else 0.0
    r = {"median_s": statistics.median(ts), "min_s": ts[0], "p90_s": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
         "mean_s": mean, "n": len(ts), "cv": abs(sd / mean) if mean else None, "samples_s": [float(f"{x:.4g}") for x in samples]}
    if extra:
        r.update(extra)
    return r


def block(x):
    return jax.block_until_ready(x)


def _t(f, *a):
    t = time.perf_counter()
    block(f(*a))
    return time.perf_counter() - t


# ---------------------------------------------------------------- provenance

def host_info():
    cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
    mem = [ln for ln in open("/proc/meminfo") if ln.startswith("MemTotal")]
    md = lambda p: sh(f"curl -s -m 2 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/{p}", 5).strip()
    return {"provider": "colab", "hostname": socket.gethostname(), "cpu_model": cpu[0] if cpu else None, "cpu_count": os.cpu_count(),
            "mem_total": mem[0].split(":")[1].strip() if mem else None, "os_kernel": platform.release(),
            "os": sh("cat /etc/os-release | head -2").strip(), "python": platform.python_version(),
            "gce_zone": md("zone")[:200], "gce_machine_type": md("machine-type")[:200], "gce_id": md("id")[:64],
            "accelerator_type": md("attributes/accelerator-type")[:100],
            "env": {k: v for k, v in os.environ.items() if k.startswith(("COLAB_", "TPU_", "XLA_", "LIBTPU", "JAX_"))}}


def session_header(suite):
    try:
        ms = {k: v for k, v in DEV.memory_stats().items() if isinstance(v, (int, float))}
    except Exception:
        ms = None
    return {"schema": "kiln.meas/1", "suite": suite, "runner": os.path.basename(__file__),
            "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "oplist_sha256": hashlib.sha256(open(OPLIST, "rb").read()).hexdigest() if os.path.exists(OPLIST) else None,
            "started": now(),
            "device": {"vendor": "google", "sku": DEV.device_kind, "device_kind": DEV.device_kind, "platform": DEV.platform,
                       "devices": [str(d) for d in jax.devices()], "n_devices": jax.device_count(),
                       "coords": getattr(DEV, "coords", None), "core_on_chip": getattr(DEV, "core_on_chip", None),
                       "memory_stats": ms, "hbm_bytes_limit": HBM_LIMIT, "spec_bf16_flops": PEAK, "spec_hbm_bps": BW,
                       "power_telemetry": None},
            "software": {"jax": jax.__version__, "jaxlib": ver("jaxlib"), "libtpu": ver("libtpu") or ver("libtpu-nightly"),
                         "python": platform.python_version(), "xla_flags": os.environ.get("XLA_FLAGS"),
                         "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS")},
            "host": host_info(),
            "method": {"n_single": N_SINGLE, "loop_target_s": LOOP_TARGET_S, "loop_pairs": LOOP_PAIRS,
                       "rotate_bytes": ROTATE_BYTES, "rotate_rule": "R = min(ceil(512MiB/bytes), 0.35*hbm_limit/alloc)",
                       "rot_slices_max": ROT_SLICES_MAX},
            "canonical_op_mode": "loop",
            "timing_modes": {
                "single": {"overheads_included": ["host_dispatch", "kernel_launch"], "operands": "resident", "launch": "jit_call"},
                "pipelined": {"overheads_included": ["kernel_launch"], "operands": "resident", "launch": "jit_call",
                              "note": "untrusted: VMEM residency across executions"},
                "pipelined_rot": {"overheads_included": ["kernel_launch"], "operands": "cold_dram", "launch": "jit_call",
                                  "diagnostic": True, "note": "separate programs; on v6e faster than any single program (08 F.4)"},
                "loop": {"overheads_included": ["inter_kernel_gap"], "operands": "cold_dram", "launch": "fori_loop", "canonical": True},
                "best": {"derived": "min(loop, pipelined_rot)", "diagnostic": True, "note": "canonical until 08 F"},
                "seq_loop": {"overheads_included": ["inter_kernel_gap"], "operands": "resident", "launch": "fori_loop",
                             "note": "whole step inside fori_loop, caches carried, slope over K"}},
            "records": []}


class Session:
    def __init__(self, suite):
        self.res = None
        if os.path.exists(OUT):
            try:
                prev = json.load(open(OUT))
                if prev.get("suite") == suite and prev.get("device", {}).get("device_kind") == DEV.device_kind:
                    self.res = prev
                    redo = os.environ.get("REDO")
                    if redo:
                        import re
                        self.res["records"] = [r for r in prev["records"] if not re.search(redo, r["name"])]
                        self.res.setdefault("redone", []).append(redo)
                    self.res.setdefault("resumed", []).append(now())
            except Exception:
                pass
        if self.res is None:
            self.res = session_header(suite)
        self.done = {r["name"] for r in self.res["records"]}
        self.dump()

    def dump(self):
        with open(OUT + ".tmp", "w") as f:
            json.dump(self.res, f, indent=0, separators=(",", ":"))
        os.replace(OUT + ".tmp", OUT)

    def add(self, r):
        self.res["records"].append(r)
        self.done.add(r["name"])
        self.dump()
        m = r.get("modes", {})
        b = r.get("canonical_s", r.get("best_s", float("nan")))
        log(f"{r['name']:44s} loop={b * 1e6:10.2f}us single={m.get('single', {}).get('median_s', float('nan')) * 1e6:9.1f}us "
            f"loop_cv={m.get('loop', {}).get('cv') or 0:.3f} {r.get('tflops', 0):7.1f}TF {r.get('gbps', 0):7.0f}GB/s {r.get('quality')}")

    def run(self, spec):
        if spec.name in self.done:
            return
        try:
            r = measure(spec)
        except Exception as e:
            r = {"name": spec.name, "kind": spec.kind, "dims": spec.dims, "error": repr(e)[:500], "modes": {}, "quality": "rejected"}
        self.add(r)

    def sanity(self, tag):
        out = {}
        for name, spec in (("gemm_8192^3", mm_spec("s", 8192, 8192, 8192)), ("copy_1GiB", mem_spec("s", "copy", GiB))):
            r = measure(spec, modes=("loop",))
            out[name] = {"loop_s": r["modes"]["loop"]["median_s"], "cv": r["modes"]["loop"]["cv"],
                         "tflops": r.get("tflops"), "gbps": r.get("gbps")}
        self.res.setdefault("sanity", {})[tag] = out
        if tag == "end" and "start" in self.res["sanity"]:
            s = self.res["sanity"]
            self.res["sanity"]["drift"] = {k: s["end"][k]["loop_s"] / s["start"][k]["loop_s"] - 1 for k in out}
        self.dump()
        log("sanity", tag, out)

    def close(self):
        self.sanity("end")
        self.res["finished"] = now()
        self.dump()
        log("DONE")


# ---------------------------------------------------------------- generic op timing

class Spec:
    """f(*operands) with operand shapes/dtypes; nbytes = bytes touched per call (in + out)."""

    def __init__(self, name, kind, f, shapes, nbytes, flops=0, dims=None, alloc=None, modes=None, impl="", **meta):
        self.name, self.kind, self.f, self.shapes, self.nbytes, self.flops = name, kind, f, shapes, nbytes, flops
        self.dims, self.alloc, self.impl, self.meta = dims or {}, alloc or nbytes, impl, meta
        self.modes = modes or ("single", "pipelined", "pipelined_rot", "loop")


@partial(jax.jit, static_argnums=(1, 2, 3))
def _hash_fill(seed, R, shape, dtype):
    """Cheap pseudo-random fill (threefry over 512 MiB stacks takes seconds on TPU)."""
    i = lax.iota(jnp.uint32, R * math.prod(shape)).reshape((R, *shape))
    h = (i + seed.astype(jnp.uint32)) * jnp.uint32(2654435761)
    h = h ^ (h >> 15)
    if jnp.issubdtype(dtype, jnp.integer):
        return (h % 1000).astype(dtype)
    return ((h >> 8).astype(F32) * (2.0 / 2**24) - 1.0).astype(dtype)


def rand_stack(key, R, shape, dtype):
    return _hash_fill(key[-1], R, tuple(shape), dtype)


def make_stacks(spec, R, seed=1):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(spec.shapes))
    return [rand_stack(k, R, s, d) for k, (s, d) in zip(keys, spec.shapes)]


def make_loop(f, R):
    @jax.jit
    def run(K, stacks):
        outs0 = jax.tree.map(lambda o: jnp.zeros((R, *o.shape), o.dtype), jax.eval_shape(f, *[s[0] for s in stacks]))

        def body(i, carry):
            s, outs = carry
            j = i % R
            prev = jax.tree.leaves(outs)[0]
            s = s + lax.dynamic_index_in_dim(prev, (i + R - 1) % R, keepdims=False).reshape(-1)[0].astype(F32) * 1e-30
            args = [lax.dynamic_index_in_dim(st, j, keepdims=False) for st in stacks]
            if jnp.issubdtype(args[0].dtype, jnp.floating):
                args[0] = args[0] + s.astype(args[0].dtype)
            else:
                args[0] = args[0] + (s > 1e30).astype(args[0].dtype)
            c = f(*args)
            return s, jax.tree.map(lambda O, x: lax.dynamic_update_index_in_dim(O, x, j, 0), outs, c)
        return lax.fori_loop(0, K, body, (jnp.zeros((), F32), outs0))
    return run


def time_loop_fn(run, args, est):
    K2 = int(min(4000, max(8, LOOP_TARGET_S / est)))
    K1 = max(2, K2 // 4)
    block(run(K1, *args))
    block(run(K2, *args))
    pairs = []
    for _ in range(LOOP_PAIRS):
        a = _t(run, K1, *args)
        b = _t(run, K2, *args)
        pairs.append((a, b))
    slopes = [(b - a) / (K2 - K1) for a, b in pairs]
    return stats(slopes, {"K1": K1, "K2": K2, "t_K1": statistics.median(p[0] for p in pairs),
                          "t_K2": statistics.median(p[1] for p in pairs),
                          "intercept_s": statistics.median(p[0] for p in pairs) - K1 * statistics.median(slopes)})


def measure(spec, modes=None):
    modes = modes or spec.modes
    r = {"name": spec.name, "kind": spec.kind, "impl": spec.impl, "dims": spec.dims, "flops": spec.flops, "bytes": spec.nbytes,
         "started": now(), **spec.meta, "modes": {}}
    est = max(spec.flops / PEAK, spec.nbytes / BW, 1e-6)
    f = jax.jit(spec.f)
    if any(m in modes for m in ("single", "pipelined")):
        args = [s[0] for s in make_stacks(spec, 1)]
        for _ in range(3):
            block(f(*args))
        if "single" in modes:
            r["modes"]["single"] = stats([_t(f, *args) for _ in range(N_SINGLE)])
        if "pipelined" in modes:
            n = int(min(200, max(10, 0.02 / est)))
            out = []
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(n):
                    o = f(*args)
                block(o)
                out.append((time.perf_counter() - t) / n)
            r["modes"]["pipelined"] = stats(out, {"calls_per_sample": n})
            o = None
        del args
    want = math.ceil(ROTATE_BYTES / max(spec.nbytes, 1))
    mem_cap = max(1, int(0.35 * HBM_LIMIT / max(spec.alloc, 1)))
    R = max(1, min(want, mem_cap))
    r["rotation"] = {"R": R, "rotate_bytes": R * spec.nbytes, "limit": "want" if R == want else "memory"}
    if "loop" in modes or "pipelined_rot" in modes:
        stacks = make_stacks(spec, R)
        if "loop" in modes:
            try:
                r["modes"]["loop"] = time_loop_fn(make_loop(spec.f, R), (stacks,), est)
                est = max(1e-7, r["modes"]["loop"]["median_s"])
            except Exception as e:
                r["modes"]["loop"] = {"error": repr(e)[:300]}
        if "pipelined_rot" in modes:
            try:
                Rr = min(R, ROT_SLICES_MAX)
                sets = [[st[j] for st in stacks] for j in range(Rr)]
                for p in sets[:4]:
                    block(f(*p))
                n = int(min(max(Rr, 20), max(Rr, 0.03 / est), 4000))
                out = []
                for _ in range(5):
                    t = time.perf_counter()
                    for i in range(n):
                        o = f(*sets[i % Rr])
                    block(o)
                    out.append((time.perf_counter() - t) / n)
                o = None
                r["modes"]["pipelined_rot"] = stats(out, {"R": Rr, "calls_per_sample": n, "cold_complete": Rr * spec.nbytes >= ROTATE_BYTES})
                del sets
            except Exception as e:
                r["modes"]["pipelined_rot"] = {"error": repr(e)[:300]}
        del stacks
    return finish(r)


def finish(r):
    cands = [r["modes"][m]["median_s"] for m in ("loop", "pipelined_rot") if "median_s" in r["modes"].get(m, {})]
    if not cands:
        cands = [r["modes"][m]["median_s"] for m in r["modes"] if "median_s" in r["modes"][m]]
    if cands:
        r["best_s"] = min(cands)
        best = r["canonical_s"] = r["modes"].get("loop", {}).get("median_s") or r["best_s"]
        r["canonical_mode"] = "loop" if "median_s" in r["modes"].get("loop", {}) else "fallback:min"
        if r.get("flops"):
            r["tflops"] = r["flops"] / best / 1e12
        if r.get("bytes"):
            r["gbps"] = r["bytes"] / best / 1e9
        lp = r["modes"].get("loop", {})
        lim = 0.08 if best < 20e-6 else 0.03
        flags = [f"loop cv {lp['cv']:.3f} > {lim}"] if lp.get("cv") is not None and lp["cv"] > lim else []
        r["quality"], r["quality_flags"] = ("rejected" if flags else "ok"), flags
    else:
        r["quality"], r["quality_flags"] = "rejected", ["no timing"]
    r["finished"] = now()
    return r


# ---------------------------------------------------------------- builders

def mm(a, b):
    return jnp.einsum("bmk,bkn->bmn", a, b) if a.ndim == 3 and b.ndim == 3 else jnp.matmul(a, b)


def mm_spec(name, m, n, k, batch=1, weight=True, **meta):
    a = (batch, m, k) if batch > 1 else (m, k)
    b = (batch, k, n) if batch > 1 and not weight else (k, n)
    nbytes = 2 * (batch * m * k + (1 if weight else batch) * k * n + batch * m * n)
    return Spec(name, "gemm" if weight else "bmm", mm, [(a, BF16), (b, BF16)], nbytes, 2 * batch * m * n * k,
                dims={"m": m, "n": n, "k": k, "batch": batch}, impl="jnp.matmul/einsum", **meta)


def flat2d(numel):
    """1-D bf16 arrays get a pathological TPU layout (v5e: 188 vs 750 GB/s read); use (numel/1024, 1024)."""
    return (numel // 1024, 1024) if numel % 1024 == 0 and numel >= 1024 else (numel,)


def mem_spec(name, kind, nbuf, **meta):
    numel = nbuf // 2
    shp = flat2d(numel)
    if kind == "read":
        return Spec(name, "read_reduce", lambda x: jnp.sum(x, dtype=F32), [(shp, BF16)], nbuf, numel,
                    dims={"bytes": nbuf}, impl="jnp.sum(bf16->f32)", **meta)
    if kind == "write":
        return Spec(name, "write", lambda x: jnp.broadcast_to(x[0], shp), [((1,), BF16)], nbuf,
                    dims={"bytes": nbuf}, impl="broadcast into output slot", **meta)
    return Spec(name, "copy", lambda x: x * 2, [(shp, BF16)], 2 * nbuf, dims={"bytes": nbuf},
                impl="x*2 into output slot", **meta)


def ew_spec(name, op, nbuf, **meta):
    numel = nbuf // 2
    shp = flat2d(numel)
    if op == "add":
        return Spec(name, "elementwise", lambda x, y: x + y, [(shp, BF16)] * 2, 3 * nbuf, numel, dims={"numel": numel, "op": op}, **meta)
    f = {"scale": lambda x: x * 2, "silu": jax.nn.silu}[op]
    return Spec(name, "elementwise", f, [(shp, BF16)], 2 * nbuf, numel, dims={"numel": numel, "op": op}, **meta)


def bucket(x):
    return "1" if x <= 1 else "2-16" if x <= 16 else "17-128" if x <= 128 else ">128"


def family(kind, m, n, k, batch=1):
    return ("bmm" if kind == "bmm" else "gemm", bucket(m), n, k, bucket(batch), "bf16")


def llm_families():
    return {family("gemm" if o["weight"] else "bmm", o["m"], o["n"], o["k"], o["batch"])
            for o in json.load(open(OPLIST)) if o["suite"] == "all"}


# ---------------------------------------------------------------- Llama-3-8B

H, NL, NQ, NKV, HD, FF, V, EPS, THETA = 4096, 32, 32, 8, 128, 14336, 128256, 1e-5, 500000.0
SCALE = HD ** -0.5
LAYER_W_BYTES = 2 * (H * (NQ + 2 * NKV) * HD + H * H + 3 * H * FF + 2 * H)
FIXED_W_BYTES = 2 * (2 * V * H + H)


def rmsnorm(x, w):
    xf = x.astype(F32)
    return (xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + EPS)).astype(x.dtype) * w


def rope_tables(pos):
    inv = 1.0 / THETA ** (jnp.arange(0, HD, 2, dtype=F32) / HD)
    f = pos.astype(F32)[:, None] * inv[None]
    return jnp.cos(f).astype(BF16)[:, None, :], jnp.sin(f).astype(BF16)[:, None, :]


def rope(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def silu_mul(g, u):
    return jax.nn.silu(g) * u


KV_S = 2048
CAUSAL = lambda S: jnp.where(jnp.arange(S)[None, :] > jnp.arange(S)[:, None], -jnp.inf, 0.0)

# attention implementations for the seq step; name -> layout, phases, kernel params
IMPLS = {
    "einsum_f32": {"layout": "bnsd", "phases": "dp", "desc": "einsum GQA (no repeat), f32 softmax, unfused XLA"},
    "einsum_bf16sm": {"layout": "bnsd", "phases": "dp", "desc": "einsum GQA (no repeat), bf16 softmax, unfused XLA"},
    "dpa_xla": {"layout": "bsnd", "phases": "dp", "desc": "jax.nn.dot_product_attention(implementation='xla'), GQA native"},
    "pallas_paged_ps64_pb16": {"layout": "paged", "phases": "d", "ps": 64, "ppcb": 16,
                               "desc": "pallas.ops.tpu.paged_attention, page 64, 16 pages/compute block"},
    "pallas_paged_ps128_pb8": {"layout": "paged", "phases": "d", "ps": 128, "ppcb": 8,
                               "desc": "pallas.ops.tpu.paged_attention, page 128, 8 pages/compute block"},
    "pallas_ragged_ps128": {"layout": "ragged", "phases": "dp", "ps": 128, "kvpb": 8, "qpb": {"d": 8, "p": 128}, "vmem": 64 * MiB,
                            "desc": "pallas.ops.tpu.ragged_paged_attention, page 128, 8 kv pages/block, q block 8 (decode) / 128 (prefill), vmem 64 MiB"},
    "pallas_flash_b128": {"layout": "bnsd", "phases": "p", "blk": None,
                          "desc": "pallas.ops.tpu.flash_attention causal, default blocks (KV repeated to 32 heads)"},
    "pallas_flash_b512": {"layout": "bnsd", "phases": "p", "blk": (512, 512, 512),
                          "desc": "pallas.ops.tpu.flash_attention causal, 512 blocks (KV repeated to 32 heads)"},
    "pallas_flash_b1024": {"layout": "bnsd", "phases": "p", "blk": (1024, 1024, 512),
                           "desc": "pallas.ops.tpu.flash_attention causal, q/k-major 1024, k 512 (KV repeated to 32 heads)"},
    "pallas_flash_b2048": {"layout": "bnsd", "phases": "p", "blk": (2048, 2048, 512),
                           "desc": "pallas.ops.tpu.flash_attention causal, q/k-major 2048, k 512 (KV repeated to 32 heads)"},
    "pallas_splash_b512": {"layout": "bnsd", "phases": "p", "blk": (512, 512, 512),
                           "desc": "pallas.ops.tpu.splash_attention MQA kernel vmapped over 8 KV heads, causal, q/kv 512"},
    "pallas_splash_b1024": {"layout": "bnsd", "phases": "p", "blk": (1024, 1024, 512),
                            "desc": "pallas.ops.tpu.splash_attention MQA kernel vmapped over 8 KV heads, causal, q/kv 1024 (compute 512)"},
    "pallas_splash_b2048": {"layout": "bnsd", "phases": "p", "blk": (2048, 2048, 512),
                            "desc": "pallas.ops.tpu.splash_attention MQA kernel vmapped over 8 KV heads, causal, q/kv 2048 (compute 512)"},
}


def cache_shapes(impl, B):
    c = IMPLS[impl]
    if c["layout"] == "bnsd":
        return [(B, NKV, KV_S, HD)] * 2
    if c["layout"] == "bsnd":
        return [(B, KV_S, NKV, HD)] * 2
    P = KV_S // c["ps"]
    if c["layout"] == "paged":
        return [(NKV, B * P, c["ps"], HD)] * 2
    return [(B * P, c["ps"], 2 * NKV, HD)]


def pack_cache(impl, K, V):
    """Logical K, V [B, S, NKV, D] -> the impl's cache layout (used by the correctness check)."""
    c = IMPLS[impl]
    B = K.shape[0]
    if c["layout"] == "bnsd":
        return (jnp.swapaxes(K, 1, 2), jnp.swapaxes(V, 1, 2))
    if c["layout"] == "bsnd":
        return (K, V)
    ps, P = c["ps"], KV_S // c["ps"]
    if c["layout"] == "paged":
        pg = lambda X: X.reshape(B, P, ps, NKV, HD).transpose(3, 0, 1, 2, 4).reshape(NKV, B * P, ps, HD)
        return (pg(K), pg(V))
    return (jnp.stack([K, V], 3).reshape(B * P, ps, 2 * NKV, HD),)


def kv_write(impl, cache, k, v, pre, pos):
    """k, v [T, NKV, D]; decode: T = B rows at position pos; prefill: B = 1, T = S rows from 0."""
    c = IMPLS[impl]
    T = k.shape[0]
    if c["layout"] == "bnsd":
        kc, vc = cache
        if pre:
            return (lax.dynamic_update_slice(kc, jnp.swapaxes(k, 0, 1)[None], (0, 0, 0, 0)),
                    lax.dynamic_update_slice(vc, jnp.swapaxes(v, 0, 1)[None], (0, 0, 0, 0)))
        return (lax.dynamic_update_slice(kc, k[:, :, None, :], (0, 0, pos, 0)),
                lax.dynamic_update_slice(vc, v[:, :, None, :], (0, 0, pos, 0)))
    if c["layout"] == "bsnd":
        kc, vc = cache
        if pre:
            return (lax.dynamic_update_slice(kc, k[None], (0, 0, 0, 0)), lax.dynamic_update_slice(vc, v[None], (0, 0, 0, 0)))
        return (lax.dynamic_update_slice(kc, k[:, None], (0, pos, 0, 0)), lax.dynamic_update_slice(vc, v[:, None], (0, pos, 0, 0)))
    ps, P = c["ps"], KV_S // c["ps"]
    if c["layout"] == "paged":
        kc, vc = cache
        if pre:
            pg = lambda X: jnp.swapaxes(X, 0, 1).reshape(NKV, P, ps, HD)
            return (lax.dynamic_update_slice(kc, pg(k), (0, 0, 0, 0)), lax.dynamic_update_slice(vc, pg(v), (0, 0, 0, 0)))
        pages = jnp.arange(T) * P + pos // ps
        return (kc.at[:, pages, pos % ps, :].set(jnp.swapaxes(k, 0, 1)), vc.at[:, pages, pos % ps, :].set(jnp.swapaxes(v, 0, 1)))
    (kv,) = cache
    kvn = jnp.stack([k, v], 2).reshape(T, 2 * NKV, HD)
    if pre:
        return (lax.dynamic_update_slice(kv, kvn.reshape(P, ps, 2 * NKV, HD), (0, 0, 0, 0)),)
    return (kv.at[jnp.arange(T) * P + pos // ps, pos % ps].set(kvn),)


@lru_cache(None)
def splash_kernel(S, blk):
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk, splash_attention_mask as sm
    mask = sm.MultiHeadMask([sm.CausalMask((S, S))] * (NQ // NKV))
    with jax.ensure_compile_time_eval():  # mask-info arrays must be concrete, not tracers of the first caller
        return sk.make_splash_mqa_single_device(mask, block_sizes=sk.BlockSizes(block_q=blk[0], block_kv=blk[1],
                                                                                block_kv_compute=blk[2]))


def attend(impl, q, cache, k, v, pre):
    """q [T, NQ, D] (decode: T = B, one query per sequence; prefill: B = 1, T = S, causal) -> [T, H]."""
    T = q.shape[0]
    c = IMPLS[impl]
    if impl.startswith("einsum"):
        f32 = impl == "einsum_f32"
        kc, vc = cache
        if pre:
            s = jnp.einsum("sgjd,gtd->gjst", q.reshape(T, NKV, 4, HD), kc[0])
            s = s.astype(F32) * SCALE + CAUSAL(T).astype(F32) if f32 else s * jnp.asarray(SCALE, BF16) + CAUSAL(T).astype(BF16)
            p = jax.nn.softmax(s, -1).astype(BF16)
            return jnp.einsum("gjst,gtd->sgjd", p, vc[0]).reshape(T, H)
        s = jnp.einsum("bgjd,bgsd->bgjs", q.reshape(T, NKV, 4, HD), kc)
        s = s.astype(F32) * SCALE if f32 else s * jnp.asarray(SCALE, BF16)
        p = jax.nn.softmax(s, -1).astype(BF16)
        return jnp.einsum("bgjs,bgsd->bgjd", p, vc).reshape(T, H)
    if impl == "dpa_xla":
        kc, vc = cache
        if pre:
            return jax.nn.dot_product_attention(q[None], kc, vc, scale=SCALE, is_causal=True, implementation="xla").reshape(T, H)
        return jax.nn.dot_product_attention(q[:, None], kc, vc, scale=SCALE, implementation="xla").reshape(T, H)
    if c["layout"] == "paged":
        from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
        kc, vc = cache
        P = KV_S // c["ps"]
        o = paged_attention((q.astype(F32) * SCALE).astype(BF16), kc, vc, jnp.full((T,), KV_S, jnp.int32),
                            jnp.arange(T * P, dtype=jnp.int32).reshape(T, P), pages_per_compute_block=c["ppcb"])
        return o.reshape(T, H)
    if c["layout"] == "ragged":
        from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention
        (kv,) = cache
        P = KV_S // c["ps"]
        nseq = 1 if pre else T
        cu = jnp.array([0, T], jnp.int32) if pre else jnp.arange(T + 1, dtype=jnp.int32)
        o = ragged_paged_attention(q, kv, jnp.full((nseq,), KV_S, jnp.int32), jnp.arange(nseq * P, dtype=jnp.int32).reshape(nseq, P),
                                   cu, jnp.array([nseq], jnp.int32), sm_scale=SCALE, num_kv_pages_per_block=c["kvpb"],
                                   num_queries_per_block=c["qpb"]["p" if pre else "d"], vmem_limit_bytes=c["vmem"])
        return o.reshape(T, H)
    if impl.startswith("pallas_flash"):
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, BlockSizes
        b = c["blk"]
        bs = BlockSizes(block_q=b[0], block_k_major=b[1], block_k=b[2], block_b=1) if b else None
        rep = lambda X: jnp.repeat(jnp.swapaxes(X, 0, 1), NQ // NKV, axis=0)[None]
        o = flash_attention(jnp.swapaxes(q, 0, 1)[None], rep(k), rep(v), causal=True, sm_scale=SCALE, block_sizes=bs)
        return jnp.swapaxes(o[0], 0, 1).reshape(T, H)
    if impl.startswith("pallas_splash"):
        qs = (q.astype(F32) * SCALE).astype(BF16).reshape(T, NKV, NQ // NKV, HD).transpose(1, 2, 0, 3)
        o = jax.vmap(splash_kernel(T, c["blk"]))(qs, jnp.swapaxes(k, 0, 1), jnp.swapaxes(v, 0, 1))
        return o.transpose(2, 0, 1, 3).reshape(T, H)
    raise ValueError(impl)


def attention(impl, q, k, v, cache, pre, pos):
    cache = kv_write(impl, cache, k, v, pre, pos)
    return attend(impl, q, cache, k, v, pre), cache


def attn_check(impl, phase):
    """Write + attend through the impl's cache layout vs an f32 reference on the logical K/V."""
    pre = phase.startswith("prefill")
    B = 1 if pre else int(phase.split("_b")[1])
    S, T = KV_S, KV_S if pre else B
    ks = jax.random.split(jax.random.PRNGKey(11), 3)
    K = jax.random.normal(ks[0], (B, S, NKV, HD), BF16)
    V = jax.random.normal(ks[1], (B, S, NKV, HD), BF16)
    q = jax.random.normal(ks[2], (T, NQ, HD), BF16)
    if pre:
        k, v = K[0], V[0]
        cache = pack_cache(impl, jnp.zeros_like(K), jnp.zeros_like(V))
    else:
        k, v = K[:, S - 1], V[:, S - 1]
        cache = pack_cache(impl, K.at[:, S - 1].set(0), V.at[:, S - 1].set(0))
    o = jax.jit(lambda q, k, v, cache: attention(impl, q, k, v, cache, pre, S - 1)[0])(q, k, v, cache).astype(F32)
    Kr = jnp.repeat(K.astype(F32), NQ // NKV, 2)
    Vr = jnp.repeat(V.astype(F32), NQ // NKV, 2)
    if pre:
        s = jnp.einsum("tnd,snd->nts", q.astype(F32), Kr[0]) * SCALE + CAUSAL(S)
        ref = jnp.einsum("nts,snd->tnd", jax.nn.softmax(s, -1), Vr[0]).reshape(T, H)
    else:
        s = jnp.einsum("bnd,bsnd->bns", q.astype(F32), Kr) * SCALE
        ref = jnp.einsum("bns,bsnd->bnd", jax.nn.softmax(s, -1), Vr).reshape(T, H)
    err = jnp.abs(o - ref)
    rel = float(jnp.sqrt(jnp.mean(err ** 2)) / jnp.sqrt(jnp.mean(ref ** 2)))
    return {"max_abs_err": float(err.max()), "rel_rms_err": rel, "ref_rms": float(jnp.sqrt(jnp.mean(ref ** 2))), "ok": rel < 2e-2}


def layer_fwd(x, Wl, cache, cos, sin, pos, pre, impl):
    T = x.shape[0]
    h = rmsnorm(x, Wl["ln1"])
    qkv = h @ Wl["wqkv"]
    q = qkv[:, :NQ * HD].reshape(T, NQ, HD)
    k = qkv[:, NQ * HD:(NQ + NKV) * HD].reshape(T, NKV, HD)
    v = qkv[:, (NQ + NKV) * HD:].reshape(T, NKV, HD)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    o, cache = attention(impl, q, k, v, cache, pre, pos)
    x = x + o @ Wl["wo"]
    h = rmsnorm(x, Wl["ln2"])
    return x + silu_mul(h @ Wl["wg"], h @ Wl["wu"]) @ Wl["wd"], cache


def make_weights(nl):
    t0 = time.time()
    key = jax.random.PRNGKey(0)
    init = jax.jit(lambda k, shape: jax.random.normal(k, shape, BF16) * 0.02, static_argnums=1)
    ks = iter(jax.random.split(key, 8 * nl + 4))
    W = {"emb": init(next(ks), (V, H)), "lm": init(next(ks), (V, H)), "lnf": jnp.ones((H,), BF16), "layers": []}
    for _ in range(nl):
        W["layers"].append({"ln1": jnp.ones((H,), BF16), "wqkv": init(next(ks), (H, (NQ + 2 * NKV) * HD)), "wo": init(next(ks), (H, H)),
                            "ln2": jnp.ones((H,), BF16), "wg": init(next(ks), (H, FF)), "wu": init(next(ks), (H, FF)),
                            "wd": init(next(ks), (FF, H))})
    block(W)
    log(f"weights {nl} layers in {time.time() - t0:.1f}s")
    return W


def make_step(L, pre, S, impl):
    def step(params, caches, tok):
        cos, sin = rope_tables(jnp.arange(S) if pre else jnp.array([S - 1]))
        x = params["emb"][tok]
        new = []
        for i in range(L):
            x, c = layer_fwd(x, params["layers"][i], caches[i], cos, sin, S - 1, pre, impl)
            new.append(c)
        if pre:
            x = x[-1:]
        logits = rmsnorm(x, params["lnf"]) @ params["lm"].T
        nt = jnp.argmax(logits, -1).astype(jnp.int32)
        return (jnp.zeros_like(tok).at[: nt.shape[0]].set(nt) + tok * 0 if pre else nt), new
    return step


def make_layer1_step(pre, S, impl):
    def step(params, caches, tok):
        cos, sin = rope_tables(jnp.arange(S) if pre else jnp.array([S - 1]))
        x = params["emb"][:tok.shape[0]] + tok[:, None].astype(BF16) * 0
        x, c = layer_fwd(x, params["layers"][0], caches[0], cos, sin, S - 1, pre, impl)
        return tok + (x[0, 0] > 1e30).astype(tok.dtype), [c]
    return step


def time_step(step, params, caches, tok, iters_single=10):
    stepj = jax.jit(step, donate_argnums=(1,))

    @partial(jax.jit, donate_argnums=(1,))
    def loopj(params, caches, tok, K):
        return lax.fori_loop(0, K, lambda i, c: step(params, c[1], c[0]), (tok, caches))

    t = time.perf_counter()
    tok, caches = block(stepj(params, caches, tok))
    compile_s = time.perf_counter() - t
    single = []
    for _ in range(iters_single):
        t = time.perf_counter()
        tok, caches = block(stepj(params, caches, tok))
        single.append(time.perf_counter() - t)
    est = statistics.median(single)
    n = int(min(30, max(5, 0.3 / est)))
    pipe = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            tok, caches = stepj(params, caches, tok)
        block(tok)
        pipe.append((time.perf_counter() - t) / n)
    modes = {"single": stats(single, {"compile_s": compile_s}), "pipelined": stats(pipe, {"calls_per_sample": n})}
    K1, K2 = 2, int(min(400, max(6, 0.3 / est)))
    try:
        t = time.perf_counter()
        tok, caches = block(loopj(params, caches, tok, K1))
        loop_compile_s = time.perf_counter() - t
        tok, caches = block(loopj(params, caches, tok, K2))
        slopes = []
        for _ in range(4):
            t = time.perf_counter()
            tok, caches = block(loopj(params, caches, tok, K1))
            a = time.perf_counter() - t
            t = time.perf_counter()
            tok, caches = block(loopj(params, caches, tok, K2))
            b = time.perf_counter() - t
            slopes.append((b - a) / (K2 - K1))
        modes["loop"] = stats(slopes, {"K1": K1, "K2": K2, "compile_s": loop_compile_s})
    except Exception as e:
        # fori_loop carrying a cache the body cannot update in place (paged scatter, bsnd DUS) can exceed HBM
        modes["loop"] = {"error": repr(e)[:500]}
    return modes, caches


def run_seq():
    """SEQ_IMPLS (comma list, default einsum_f32) picks attention implementations; SEQ_PHASES filters phases."""
    ses = Session("seq")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    S = KV_S
    impls = os.environ.get("SEQ_IMPLS", "einsum_f32").split(",")
    phases = os.environ.get("SEQ_PHASES", "decode_b1,decode_b8,decode_b32,prefill_b1").split(",")
    kv_layer = lambda B: 2 * 2 * B * NKV * S * HD
    act = 1.2 * GiB
    budget = 0.88 * HBM_LIMIT - FIXED_W_BYTES - act
    plan = {}
    for phase in ("decode_b1", "decode_b8", "decode_b32", "prefill_b1"):
        B = 1 if phase.startswith("prefill") else int(phase.split("_b")[1])
        plan[phase] = int(min(NL, budget // (LAYER_W_BYTES + kv_layer(B))))
    Lw = min(plan.values())
    plan = {k: Lw for k in plan}
    ses.res["model"] = {"name": "llama-3-8b (random bf16 weights)", "hidden": H, "layers_full": NL, "q_heads": NQ, "kv_heads": NKV,
                        "head_dim": HD, "ffn": FF, "vocab": V, "kv_len": S, "attn_impl": "per record (attn_impl)",
                        "attn_impls": {k: IMPLS[k]["desc"] for k in impls},
                        "layer_weight_bytes": LAYER_W_BYTES, "fixed_weight_bytes": FIXED_W_BYTES, "hbm_bytes_limit": HBM_LIMIT,
                        "layers_per_phase": plan, "full_model_fits": all(v == NL for v in plan.values())}
    ses.dump()
    log("plan", plan, "impls", impls, "phases", phases)
    checks = ses.res.setdefault("attn_checks", {})
    todo = []
    for phase in phases:
        for impl in impls:
            if ("p" if phase.startswith("prefill") else "d") not in IMPLS[impl]["phases"]:
                continue
            key = f"{phase}/{impl}"
            if key not in checks:
                try:
                    checks[key] = attn_check(impl, phase)
                except Exception as e:
                    checks[key] = {"ok": False, "error": repr(e)[:800]}
                ses.dump()
                log("check", key, checks[key])
            if checks[key]["ok"]:
                todo.append((phase, impl))
    W = make_weights(Lw)
    for phase in phases:  # embedding + final norm + lm_head + argmax only: per-layer slope = (step - step0) / L
        name = f"seq/{phase}/step0"
        if name in ses.done:
            continue
        pre = phase.startswith("prefill")
        B = 1 if pre else int(phase.split("_b")[1])
        r = {"name": name, "kind": "sequence", "scope": "step0", "phase": phase, "batch": B, "kv_len": S, "tokens": S if pre else B,
             "n_layers": 0, "n_layers_full": NL, "started": now(), "modes": {}}
        try:
            tok = jax.random.randint(jax.random.PRNGKey(3), (S if pre else B,), 0, V, jnp.int32)
            params = {"emb": W["emb"], "lm": W["lm"], "lnf": W["lnf"], "layers": []}
            r["modes"], _ = time_step(make_step(0, pre, S, "einsum_f32"), params, [], tok)
            r["best_mode"] = "loop" if "median_s" in r["modes"]["loop"] else "pipelined"
            r["best_s"] = r["modes"][r["best_mode"]]["median_s"]
        except Exception as e:
            r["error"] = repr(e)[:800]
        lp = r["modes"].get(r.get("best_mode", "loop"), {})
        r["quality"] = "ok" if lp.get("cv") is not None and lp["cv"] <= 0.03 else "rejected"
        r["finished"] = now()
        ses.add(r)
    for phase, impl in todo:
        L = plan[phase]
        pre = phase.startswith("prefill")
        B = 1 if pre else int(phase.split("_b")[1])
        for scope, nl in (("step", L), ("layer1", 1)):
            name = f"seq/{phase}/{scope}" + ("" if impl == "einsum_f32" else f"/{impl}")
            if name in ses.done:
                continue
            r = {"name": name, "kind": "sequence", "scope": "step" if scope == "step" else "layer", "phase": phase, "batch": B,
                 "kv_len": S, "tokens": S if pre else B, "n_layers": nl, "n_layers_full": NL, "attn_impl": impl,
                 "attn_impl_desc": IMPLS[impl]["desc"], "attn_check": checks[f"{phase}/{impl}"], "started": now(), "modes": {}}
            try:
                params = {"emb": W["emb"], "lm": W["lm"], "lnf": W["lnf"], "layers": W["layers"][:nl]}
                keys = jax.random.split(jax.random.PRNGKey(7), 2 * nl)
                caches = [tuple(jax.random.normal(keys[2 * i + j], s, BF16) for j, s in enumerate(cache_shapes(impl, B)))
                          for i in range(nl)]
                tok = jax.random.randint(jax.random.PRNGKey(3), (S if pre else B,), 0, V, jnp.int32)
                step = make_step(nl, pre, S, impl) if scope == "step" else make_layer1_step(pre, S, impl)
                r["modes"], caches = time_step(step, params, caches, tok)
                r["best_mode"] = "loop" if "median_s" in r["modes"]["loop"] else "pipelined"
                r["best_s"] = r["modes"][r["best_mode"]]["median_s"]
                if scope == "step":
                    r["tokens_per_s"] = r["tokens"] / r["best_s"]
                del caches
            except Exception as e:
                r["error"] = repr(e)[:800]
            lp = r["modes"].get(r.get("best_mode", "loop"), {})
            r["quality"] = "ok" if lp.get("cv") is not None and lp["cv"] <= 0.03 else "rejected"
            r["finished"] = now()
            ses.add(r)
    ses.close()


# ---------------------------------------------------------------- micro / suite

def launch_probe(ses, kname, N):
    name = f"launch/{kname}_x{N}"
    if name in ses.done:
        return
    d = {"add_1elem": 1, "mm_16": 16, "mm_64": 64, "mm_128": 128}[kname]

    def chain(x):
        for _ in range(N):
            x = x + 1 if kname == "add_1elem" else jnp.tanh(x @ x)
        return x
    shape = (1,) if kname == "add_1elem" else (d, d)
    spec = Spec(name, "launch_probe", chain, [(shape, BF16)], 4 * math.prod(shape) * N, dims={"kernel": kname, "chain": N},
                modes=("single", "loop"))
    r = measure(spec)
    for m in r["modes"].values():
        if "median_s" in m:
            m["per_op"] = {k: m[k] / N for k in ("median_s", "min_s")}
    ses.add(r)


def run_micro():
    ses = Session("micro")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    fams = llm_families()
    dropped = ses.res.setdefault("leakage_dropped", [])

    def gated(spec):
        d = spec.dims
        if family(spec.kind, d["m"], d["n"], d["k"], d["batch"]) in fams:
            if spec.name not in dropped:
                dropped.append(spec.name)
            return
        ses.run(spec)

    if "launch/empty_jit" not in ses.done:
        f = jax.jit(lambda x: x)
        x = jnp.ones((1,), BF16)
        for _ in range(5):
            block(f(x))
        ses.add(finish({"name": "launch/empty_jit", "kind": "launch_probe", "modes": {"single": stats([_t(f, x) for _ in range(200)])}}))
    for kname, Ns in (("add_1elem", (1,)), ("mm_16", (1, 10, 100, 1000)), ("mm_64", (1, 10, 100)), ("mm_128", (1, 10, 100))):
        for N in Ns:
            launch_probe(ses, kname, N)

    sizes = sorted([2**i * MiB for i in range(13)] + [48 * MiB, 384 * MiB, 1536 * MiB])
    for s in sizes:
        for kind in ("read", "write", "copy"):
            ses.run(mem_spec(f"hbm/{kind}_{s // MiB}MiB", kind, s, group="hbm"))

    for kib in (256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768):
        for kind in ("read", "copy"):
            sp = mem_spec(f"vmem/{kind}_{kib}KiB", kind, kib * 1024, group="vmem_resident")
            name = sp.name
            if name in ses.done:
                continue
            r = measure(sp, modes=("pipelined",))
            try:
                lp = time_loop_fn(make_loop(sp.f, 1), (make_stacks(sp, 1),), sp.nbytes / BW)
                r["modes"]["loop_r1"] = lp
            except Exception as e:
                r["modes"]["loop_r1"] = {"error": repr(e)[:200]}
            r["note"] = "no rotation (R=1): operands may stay VMEM-resident"
            ses.add(finish(r))

    W = (3072, 5120, 10240, 12288, 24576)
    for m in (1, 4, 16, 64):
        for n in W:
            for k in W:
                gated(mm_spec(f"gemv/gemm_{m}_{n}_{k}", m, n, k, group="gemv"))
    gm = [(s, s, s) for s in (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384)]
    gm += [(m, m, k) for m in (256, 1024, 2048) for k in (8192, 16384, 32768, 65536)]
    gm += [(m, n, k) for m in (512, 2048, 8192) for n in (3072, 5120, 10240) for k in (3072, 5120, 10240)]
    for m, n, k in gm:
        gated(mm_spec(f"gemm/gemm_{m}_{n}_{k}", m, n, k, group="gemm"))
    for b in (16, 128):
        for m in (4, 1024):
            for n, k in ((1024, 128), (4096, 64), (128, 4096)):
                gated(mm_spec(f"bmm/bmm_{b}_{m}_{n}_{k}", m, n, k, batch=b, weight=False, group="bmm"))
    for op in ("add", "scale", "silu"):
        for p in range(12, 31, 2):
            ses.run(ew_spec(f"ew/{op}_{2**p // 1024}KiB", op, 2**p, group="elementwise"))
    ses.close()


def aux_specs(phase):
    pre = phase.startswith("prefill")
    B = 1 if pre else int(phase.split("_b")[1])
    T, S = (2048, 2048) if pre else (B, 2048)
    cos, sin = rope_tables(jnp.arange(T) if pre else jnp.array([S - 1]))
    w = jnp.ones((H,), BF16)
    out = []

    def add(name, f, shapes, nbytes, count, alloc=None):
        out.append((Spec(f"{phase}/{name}", "aux", f, shapes, nbytes, alloc=alloc, phase=phase, op=name, count=count,
                         impl="jax (as seq, unfused in isolation)"), count))
    add("rmsnorm", lambda x: rmsnorm(x, w), [((T, H), BF16)], 4 * T * H, 2 * NL + (0 if pre else 1))
    if pre:
        add("rmsnorm_final", lambda x: rmsnorm(x, w), [((1, H), BF16)], 4 * H, 1)
    add("rope_qk", lambda q, k: (rope(q, cos, sin), rope(k, cos, sin)), [((T, NQ, HD), BF16), ((T, NKV, HD), BF16)],
        4 * T * (NQ + NKV) * HD, NL)
    if pre:
        mask = lambda: jnp.where(jnp.arange(S)[None, :] > jnp.arange(S)[:, None], -jnp.inf, 0.0).astype(F32)
        add("softmax", lambda s: jax.nn.softmax(s.astype(F32) * SCALE + mask(), -1).astype(BF16), [((NKV, 4, S, S), BF16)],
            4 * NKV * 4 * S * S, NL)
    else:
        add("softmax", lambda s: jax.nn.softmax(s.astype(F32) * SCALE, -1).astype(BF16), [((B, NKV, 4, S), BF16)], 4 * B * NKV * 4 * S, NL)
    add("residual_add", lambda a, b: a + b, [((T, H), BF16)] * 2, 6 * T * H, 2 * NL)
    add("silu_mul", silu_mul, [((T, FF), BF16)] * 2, 6 * T * FF, NL)
    add("argmax", lambda x: jnp.argmax(x, -1), [((1 if pre else B, V), BF16)], 2 * (1 if pre else B) * V, 1)
    return out


def kv_write_record(phase):
    """Decode KV append (prefill: full-sequence write) as an in-place DUS inside a fori_loop carrying the cache."""
    pre = phase.startswith("prefill")
    B = 1 if pre else int(phase.split("_b")[1])
    S = 2048
    nb = 2 * (S if pre else 1) * B * NKV * HD
    R = max(1, min(4096, math.ceil(ROTATE_BYTES / nb)))
    ks = jax.random.normal(jax.random.PRNGKey(5), (R, B, NKV, S if pre else 1, HD), BF16)
    cache = jnp.zeros((B, NKV, S, HD), BF16)

    @jax.jit
    def run(K, ks, cache):
        def body(i, c):
            k = lax.dynamic_index_in_dim(ks, i % R, keepdims=False)
            return lax.dynamic_update_slice(c, k, (0, 0, 0 if pre else i % S, 0))
        return lax.fori_loop(0, K, body, cache)
    lp = time_loop_fn(lambda K, ks: run(K, ks, cache), (ks,), max(nb / BW, 1e-6))
    r = {"name": f"{phase}/kv_write", "kind": "aux", "phase": phase, "op": "kv_write", "count": 2 * NL, "bytes": nb,
         "impl": "lax.dynamic_update_slice in fori_loop (cache carried)", "rotation": {"R": R}, "modes": {"loop": lp}}
    return finish(r)


def run_suite():
    ses = Session("suite")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    seen = set()
    for o in json.load(open(OPLIST)):
        if o["key"] in seen:
            continue
        seen.add(o["key"])
        ses.run(mm_spec(o["key"], o["m"], o["n"], o["k"], o["batch"], o["weight"], phase=o["phase"], op=o["op"], count=o["count"],
                        group="llm_op"))
    for phase in ("decode_b1", "decode_b8", "decode_b32", "prefill_b1"):
        for spec, _ in aux_specs(phase):
            ses.run(spec)
        if f"{phase}/kv_write" not in ses.done:
            try:
                ses.add(kv_write_record(phase))
            except Exception as e:
                ses.add({"name": f"{phase}/kv_write", "error": repr(e)[:300], "modes": {}, "quality": "rejected"})
        name = f"{phase}/embedding"
        if name not in ses.done:
            T = 2048 if phase.startswith("prefill") else int(phase.split("_b")[1])
            sp = Spec(name, "aux", lambda t, emb: emb[t % V], [((T,), jnp.int32), ((V, H), BF16)], 4 * T * H, alloc=2 * V * H,
                      phase=phase, op="embedding", count=1, impl="gather (table rotated as memory allows)")
            ses.run(sp)
    shapes = [(s, s, s) for s in (256, 512, 1024, 2048, 4096, 8192)]
    shapes += [(m, nk, nk) for nk in (4096, 8192, 14336) for m in (1, 8, 16, 32, 64, 128, 256)]
    for m, n, k in shapes:
        ses.run(mm_spec(f"sweep/gemm_{m}_{n}_{k}", m, n, k, group="legacy_sweep"))
    ses.close()


if __name__ == "__main__":
    {"micro": run_micro, "suite": run_suite, "seq": run_seq}[SUITE]()
