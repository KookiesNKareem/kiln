"""kiln CUDA ground-truth runner, schema kiln.meas/1 (spec 06 §4).

Usage: python gpu_bench.py <micro|seq|seq_oos|suite> out.json [oplist.json]

  micro  calib-micro fit suite (06 §3.4, §4.2 item 3): HBM read/write/copy 1 MiB-4 GiB, L2-resident sweeps,
         GEMV/skinny and GEMM sweeps (LLM shape families dropped), bmm sweep, elementwise sweep, launch-overhead
         probes, power-step telemetry, session sanity.
  seq_oos out-of-sample sequences for the stack recipe: GPT-J-6B whole steps (HF modeling_gptj op structure, SDPA)
         and Llama-2-70B 1/2/4-layer whole steps plus layer-only graphs; full profiler kernel lists.
  seq    whole-step `sequence` (06 §4.2 item 4): Llama-3-8B decode b1/b8/b32 (kv 2048) and prefill b1 (seq 2048),
         full 32 layers + embedding + lm_head + argmax as one CUDA graph; 1-layer sequences; eager and SDPA variants.
  suite  LLM per-op suite (oplist.json, plus `_linear` twins), legacy GEMM sweep, and isolated non-matmul ops of the
         step (rmsnorm, rope, kv write, softmax, silu*mul, residual add, embedding, argmax, permutes) so per-phase
         sums of isolated ops can be compared with the whole step.

Timing modes (per record): `flushed` (256 MiB L2 flush before each iteration), `unflushed`, `graph_unflushed`
(same operands, many calls in one CUDA graph), `graph_cold` (canonical: one CUDA graph cycling over distinct operand
copies totalling >= max(512 MiB, 2 x L2), capped by memory). Eager modes store per-iteration CUDA-event samples;
graph modes store one sample per replay (mean per call within the replay). NVML clock/power windows cover only the
timed region of each mode. Output is rewritten after every record; a rerun resumes and skips finished records.
"""
import ctypes
import datetime
import glob
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from importlib import metadata

import torch
import torch.nn.functional as F

SUITE, OUT = sys.argv[1], sys.argv[2]
OPLIST = sys.argv[3] if len(sys.argv) > 3 else "oplist.json"
MiB, GiB = 2**20, 2**30
WARMUP = 10
MODE_S = 0.25
MIN_SAMPLES, MAX_EAGER = 20, 1000
GRAPH_MIN_REPLAYS, GRAPH_MAX_REPLAYS, GRAPH_MIN_REPLAY_S = 10, 200, 2e-3
MAX_GRAPH_CALLS = 16384
FLUSH_BYTES = 256 * MiB
ROT_MIN = 512 * MiB
NVML_DT = 0.01
BF = dict(dtype=torch.bfloat16, device="cuda")
SPEC_GBPS = {"A100-SXM4-80GB": 2039, "A100-SXM4-40GB": 1555, "A100-PCIE-40GB": 1555, "A100 80GB PCIe": 1935}
SPEC_TFLOPS = {"A100": 312}

PROPS = torch.cuda.get_device_properties(0)
L2 = getattr(PROPS, "L2_cache_size", 40 * MiB) or 40 * MiB
COLD_TARGET = max(ROT_MIN, 2 * L2)
FLUSH = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def sh(cmd, timeout=30):
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


# ---------------------------------------------------------------- telemetry

class Telemetry:
    """10 ms NVML sampler: (t, sm_mhz, mem_mhz, power_w, temp_c, throttle_bits)."""

    def __init__(self):
        self.samples, self.stop, self.error = [], False, None
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or nv.nvmlDeviceGetCurrentClocksThrottleReasons
            while not self.stop:
                self.samples.append((time.time(), nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                                     nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), nv.nvmlDeviceGetPowerUsage(h) / 1e3,
                                     nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), int(reasons(h))))
                time.sleep(NVML_DT)
        except Exception as e:
            self.error = repr(e)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        sparse = len(s) < 3
        if sparse:
            s = sorted(self.samples, key=lambda x: abs(x[0] - (t0 + t1) / 2))[:1]
        if not s:
            return None
        col = lambda i: [x[i] for x in s]
        bits = 0
        for b in col(5):
            bits |= b
        return {"n": len(s), "sparse": sparse, "t0": t0, "t1": t1, "sm_mhz": statistics.median(col(1)),
                "sm_mhz_min": min(col(1)), "sm_mhz_max": max(col(1)), "mem_mhz": statistics.median(col(2)),
                "power_w": statistics.median(col(3)), "power_w_max": max(col(3)), "temp_c": statistics.median(col(4)),
                "throttle_bits_union": bits}

    def raw(self, t0, t1):
        return [[round(x[0] - t0, 4), x[1], round(x[3], 1), x[4], x[5]] for x in self.samples if t0 <= x[0] <= t1]


TEL = Telemetry()


# ---------------------------------------------------------------- provenance

def cublas_info():
    out = {"pip": {p: ver(p) for p in ("nvidia-cublas-cu12", "nvidia-cublas-cu13", "nvidia-cublas", "nvidia-cudnn-cu12",
                                      "nvidia-cudnn-cu13", "nvidia-nccl-cu12", "nvidia-nccl-cu13") if ver(p)}}
    a = torch.randn(64, 64, **BF)
    (a @ a).sum().item()
    paths = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libcublas" in ln})
    out["loaded"] = paths
    for p in paths:
        try:
            lib = ctypes.CDLL(p)
            if "libcublasLt" in p:
                lib.cublasLtGetVersion.restype = ctypes.c_size_t
                out["cublasLt_version"] = int(lib.cublasLtGetVersion())
            else:
                v = []
                for prop in range(3):
                    x = ctypes.c_int()
                    lib.cublasGetProperty(prop, ctypes.byref(x))
                    v.append(x.value)
                out["cublas_version"] = ".".join(map(str, v))
        except Exception as e:
            out.setdefault("errors", []).append(f"{p}: {e!r}"[:200])
    out["preferred_blas"] = str(getattr(torch.backends.cuda, "preferred_blas_library", lambda: None)())
    return out


def host_info():
    cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
    mem = [ln for ln in open("/proc/meminfo") if ln.startswith("MemTotal")]
    md = lambda p: sh(f"curl -s -m 2 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/{p}", 5).strip()
    return {"provider": "colab", "hostname": socket.gethostname(), "cpu_model": cpu[0] if cpu else None,
            "cpu_count": os.cpu_count(), "mem_total": mem[0].split(":")[1].strip() if mem else None,
            "os_kernel": platform.release(), "os": sh("cat /etc/os-release | head -2").strip(), "python": platform.python_version(),
            "gce_zone": md("zone")[:200], "gce_machine_type": md("machine-type")[:200], "gce_id": md("id")[:64],
            "colab_env": {k: v for k, v in os.environ.items() if k.startswith(("COLAB_", "TPU_", "CUDA_V", "NV_"))}}


def device_info():
    q = ("name,memory.total,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,clocks.applications.graphics,power.limit,"
         "power.default_limit,power.max_limit,pcie.link.gen.max,pcie.link.width.max,driver_version,vbios_version,"
         "pci.device_id,ecc.mode.current,persistence_mode,mig.mode.current,temperature.gpu")
    smi = dict(zip(q.split(","), sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader").strip().split(", ")))
    name = smi.get("name", PROPS.name).replace("NVIDIA ", "")
    return {"vendor": "nvidia", "sku": name, "torch_name": PROPS.name, "smi": smi, "sm_count": PROPS.multi_processor_count,
            "total_mem_bytes": PROPS.total_memory, "l2_bytes": L2, "cc": f"{PROPS.major}.{PROPS.minor}",
            "mem_bus_width": getattr(PROPS, "memory_bus_width", None), "mem_clock_khz": getattr(PROPS, "memory_clock_rate", None),
            "spec_hbm_gbps": SPEC_GBPS.get(name), "spec_bf16_dense_tflops": SPEC_TFLOPS.get(name[:4]),
            "smi_q": sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,ECC,TEMPERATURE")}


def software_info():
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "nccl": ".".join(map(str, torch.cuda.nccl.version())) if hasattr(torch.cuda, "nccl") else None,
            "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").strip(),
            "cublas": cublas_info(), "nvcc": sh("nvcc --version | tail -2").strip(), "python": platform.python_version()}


# ---------------------------------------------------------------- timing primitives

def ev():
    return torch.cuda.Event(enable_timing=True)


def prefill(seconds):
    s = min(0.06, max(0.002, seconds))
    torch.cuda._sleep(int(s * PROPS.clock_rate * 1e3 if hasattr(PROPS, "clock_rate") else s * 1.41e9))
    return s


def stats(samples, extra=None):
    ts = sorted(samples)
    mean = statistics.fmean(ts)
    sd = statistics.pstdev(ts) if len(ts) > 1 else 0.0
    r = {"median_s": statistics.median(ts), "min_s": ts[0], "p90_s": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
         "mean_s": mean, "n": len(ts), "cv": sd / mean if mean else None,
         "samples_s": [float(f"{x:.4g}") for x in samples]}
    if extra:
        r.update(extra)
    return r


def pilot(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = ev(), ev()
    a.record()
    for _ in range(5):
        fn()
    b.record()
    torch.cuda.synchronize()
    return max(a.elapsed_time(b) * 1e-3 / 5, 1e-6)


def flush():
    FLUSH.add_(1.0)


def time_eager(fn, est, do_flush=False, iters=None):
    per = est + (FLUSH_BYTES * 2 / 1.3e12 if do_flush else 0)
    iters = iters or int(min(MAX_EAGER, max(MIN_SAMPLES, MODE_S / per)))
    for _ in range(max(WARMUP, min(200, int(0.02 / per)))):
        fn()
    torch.cuda.synchronize()
    evs = [(ev(), ev()) for _ in range(iters)]
    t0 = time.time()
    lead = prefill(iters * (12e-6 if do_flush else 8e-6) + 1e-3)
    for a, b in evs:
        if do_flush:
            flush()
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    t1 = time.time()
    return stats([a.elapsed_time(b) * 1e-3 for a, b in evs], {"clock": TEL.window(t0 + lead, t1), "iters": iters})


def capture(calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in calls[: min(len(calls), 3)]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in calls:
            f()
    return g


def time_graph(calls, est_call, replays=None, extra=None):
    """One sample per replay = replay time / len(calls)."""
    g = capture(calls)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    rep_s = est_call * len(calls)
    replays = replays or int(min(GRAPH_MAX_REPLAYS, max(GRAPH_MIN_REPLAYS, MODE_S / rep_s)))
    evs = [(ev(), ev()) for _ in range(replays)]
    t0 = time.time()
    lead = prefill(replays * 15e-6 + 1e-3)
    for a, b in evs:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    t1 = time.time()
    del g
    return stats([a.elapsed_time(b) * 1e-3 / len(calls) for a, b in evs],
                 {"clock": TEL.window(t0 + lead, t1), "calls_per_replay": len(calls), **(extra or {})})


def bank(copies, shape, dtype=torch.bfloat16, init="randn"):
    """`copies` distinct tensors as 256-byte-aligned views of one buffer."""
    numel = math.prod(shape)
    esz = torch.empty((), dtype=dtype).element_size()
    stride = -(-numel * esz // 256) * 256 // esz
    buf = torch.empty(copies * stride, dtype=dtype, device="cuda")
    if init == "randn":
        buf.normal_()
    elif init == "ones":
        buf.fill_(1)
    elif init == "zeros":
        buf.zero_()
    return [buf[i * stride:i * stride + numel].view(shape) for i in range(copies)]


# ---------------------------------------------------------------- records

class Spec:
    """A measurable op: make(copies) -> list of zero-arg callables over distinct operand sets."""

    def __init__(self, name, kind, make, nbytes, flops=0, alloc=None, impl="", dims=None, modes=None, rot_cap=None, **meta):
        self.name, self.kind, self.make, self.nbytes, self.flops = name, kind, make, nbytes, flops
        self.alloc = alloc or nbytes
        self.impl, self.dims = impl, dims or {}
        self.modes = modes or ("flushed", "unflushed", "graph_unflushed", "graph_cold")
        self.rot_cap, self.meta = rot_cap, meta


def free_bytes():
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def measure(spec):
    r = {"name": spec.name, "kind": spec.kind, "impl": spec.impl, "dims": spec.dims, "flops": spec.flops,
         "bytes": spec.nbytes, "started": now(), **spec.meta, "modes": {}}
    fns = spec.make(1)
    est = pilot(fns[0])
    r["pilot_s"] = est
    for mode in spec.modes:
        try:
            if mode == "flushed":
                r["modes"][mode] = time_eager(fns[0], est, True)
            elif mode == "unflushed":
                r["modes"][mode] = time_eager(fns[0], est, False)
            elif mode == "graph_unflushed":
                reps = int(min(2000, max(5, math.ceil(GRAPH_MIN_REPLAY_S / est))))
                r["modes"][mode] = time_graph([fns[0]] * reps, est)
        except Exception as e:
            r["modes"][mode] = {"error": repr(e)[:300]}
    del fns
    if "graph_cold" in spec.modes:
        try:
            want = math.ceil(COLD_TARGET / max(spec.nbytes, 1))
            mem_cap = max(1, int(0.45 * free_bytes() / max(spec.alloc, 1)))
            copies = max(1, min(want, mem_cap, MAX_GRAPH_CALLS, spec.rot_cap or MAX_GRAPH_CALLS))
            fns = spec.make(copies)
            cycles = max(1, math.ceil(math.ceil(GRAPH_MIN_REPLAY_S / est) / copies))
            if copies * cycles > MAX_GRAPH_CALLS:
                cycles = 1
            calls = [fns[i % copies] for i in range(copies * cycles)]
            rot = copies * spec.nbytes
            r["modes"]["graph_cold"] = time_graph(calls, est, extra={
                "copies": copies, "rotate_bytes": rot, "cold_complete": rot >= 2 * L2 or copies == want,
                "copy_limit": "want" if copies == want else ("memory" if copies == mem_cap else "calls")})
            del fns, calls
        except Exception as e:
            r["modes"]["graph_cold"] = {"error": repr(e)[:300]}
    torch.cuda.empty_cache()
    return finish(r)


def finish(r):
    canon = next((r["modes"][k] for k in ("graph_cold", "graph_unflushed", "graph") if "median_s" in r["modes"].get(k, {})),
                 next((m for m in r["modes"].values() if "median_s" in m), {}))
    if "median_s" in canon:
        t = canon["median_s"]
        if r.get("flops"):
            r["tflops"] = r["flops"] / t / 1e12
        if r.get("bytes"):
            r["gbps"] = r["bytes"] / t / 1e9
        lim = 0.08 if t < 20e-6 else 0.03
        flags = []
        if canon.get("cv") is not None and canon["cv"] > lim:
            flags.append(f"cv {canon['cv']:.3f} > {lim}")
        c = canon.get("clock")
        if c and not c.get("sparse") and c["sm_mhz_min"] < 0.9 * c["sm_mhz"]:
            flags.append("sm clock min < 0.9 median")
        r["quality"] = "rejected" if any(f.startswith("cv") for f in flags) else "ok"
        r["quality_flags"] = flags
    else:
        r["quality"] = "rejected"
        r["quality_flags"] = ["no canonical timing"]
    r["finished"] = now()
    return r


# ---------------------------------------------------------------- op builders

def mm_spec(name, m, n, k, batch=1, weight=True, linear=False, **meta):
    if weight:
        a_s, b_s, c_s = (m, k), ((n, k) if linear else (k, n)), (m, n)
    else:
        a_s, b_s, c_s = (batch, m, k), (batch, k, n), (batch, m, n)
    flops = 2 * batch * m * n * k
    nbytes = 2 * (batch * m * k + (1 if weight else batch) * k * n + batch * m * n)

    def make(c):
        A, B, C = bank(c, a_s), bank(c, b_s), bank(c, c_s, init="zeros")
        if not weight:
            return [lambda A=A, B=B, C=C: torch.bmm(A, B, out=C) for A, B, C in zip(A, B, C)]
        if linear:
            return [lambda A=A, B=B: F.linear(A, B) for A, B in zip(A, B)]
        return [lambda A=A, B=B, C=C: torch.matmul(A, B, out=C) for A, B, C in zip(A, B, C)]
    impl = "torch.bmm" if not weight else ("F.linear" if linear else "torch.matmul")
    kind = "bmm" if not weight else ("linear" if linear else "gemm")
    dims = {"m": m, "n": n, "k": k, "batch": batch}
    return Spec(name, kind, make, nbytes, flops, impl=impl, dims=dims, **meta)


def mem_spec(name, kind, nbytes_buf, **meta):
    numel = nbytes_buf // 2
    if kind == "read":
        def make(c):
            X = bank(c, (numel,), init="ones")
            return [lambda x=x: torch.sum(x, dtype=torch.float32) for x in X]
        return Spec(name, "read_reduce", make, nbytes_buf, numel, impl="torch.sum(bf16->f32)", dims={"bytes": nbytes_buf}, **meta)
    if kind == "write":
        def make(c):
            X = bank(c, (numel,), init=None)
            return [lambda x=x: x.fill_(1.0) for x in X]
        return Spec(name, "write", make, nbytes_buf, impl="Tensor.fill_", dims={"bytes": nbytes_buf}, **meta)
    def make(c):
        X, Y = bank(c, (numel,), init="ones"), bank(c, (numel,), init=None)
        return [lambda x=x, y=y: y.copy_(x) for x, y in zip(X, Y)]
    return Spec(name, "copy", make, 2 * nbytes_buf, impl="Tensor.copy_", dims={"bytes": nbytes_buf}, alloc=2 * nbytes_buf, **meta)


def ew_spec(name, op, nbytes_buf, **meta):
    numel = nbytes_buf // 2
    if op == "add":
        def make(c):
            X, Y, Z = bank(c, (numel,)), bank(c, (numel,)), bank(c, (numel,), init=None)
            return [lambda x=x, y=y, z=z: torch.add(x, y, out=z) for x, y, z in zip(X, Y, Z)]
        return Spec(name, "elementwise", make, 3 * nbytes_buf, numel, impl="torch.add(out=)", dims={"numel": numel, "op": op}, **meta)
    fn = {"scale": lambda x, z: torch.mul(x, 2.0, out=z), "silu": lambda x, z: torch.ops.aten.silu.out(x, out=z)}[op]
    def make(c):
        X, Z = bank(c, (numel,)), bank(c, (numel,), init=None)
        return [lambda x=x, z=z: fn(x, z) for x, z in zip(X, Z)]
    return Spec(name, "elementwise", make, 2 * nbytes_buf, numel, impl=f"aten.{op}.out", dims={"numel": numel, "op": op}, **meta)


# ---------------------------------------------------------------- leakage rule (06 §3.4)

def bucket(x):
    return "1" if x <= 1 else "2-16" if x <= 16 else "17-128" if x <= 128 else ">128"


def family(kind, m, n, k, batch=1):
    kind = "bmm" if kind == "bmm" else "gemm"
    return (kind, bucket(m), n, k, bucket(batch), "bf16")


def llm_families():
    fams = set()
    for o in json.load(open(OPLIST)):
        if o["suite"] == "all":
            fams.add(family("gemm" if o["weight"] else "bmm", o["m"], o["n"], o["k"], o["batch"]))
    return fams


# ---------------------------------------------------------------- Llama-3-8B pieces (shared by seq and suite)

H, NL, NQ, NKV, HD, FF, V, EPS, THETA = 4096, 32, 32, 8, 128, 14336, 128256, 1e-5, 500000.0
SCALE = HD ** -0.5


def rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


def rope(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def rope_tables(positions):
    inv = 1.0 / THETA ** (torch.arange(0, HD, 2, device="cuda").float() / HD)
    f = positions.float()[:, None] * inv[None]
    return f.cos().to(torch.bfloat16)[:, None, :], f.sin().to(torch.bfloat16)[:, None, :]


def softmax_dec(s):
    return (s.float() * SCALE).softmax(-1).to(torch.bfloat16)


def softmax_pre(s, mask):
    S = mask.shape[0]
    return (s.view(NKV, 4, S, S).float() * SCALE + mask).softmax(-1).to(torch.bfloat16).view(NKV, 4 * S, S)


def silu_mul(g, u):
    return F.silu(g) * u


def make_weights(nl):
    t0 = time.time()
    init = lambda *s: torch.randn(*s, **BF).mul_(0.02)
    W = {"emb": init(V, H), "lm": init(V, H), "lnf": torch.ones(H, **BF), "layers": []}
    for _ in range(nl):
        W["layers"].append({"ln1": torch.ones(H, **BF), "wqkv": init((NQ + 2 * NKV) * HD, H), "wo": init(H, H),
                            "ln2": torch.ones(H, **BF), "wg": init(FF, H), "wu": init(FF, H), "wd": init(H, FF)})
    torch.cuda.synchronize()
    log(f"weights {nl} layers in {time.time() - t0:.1f}s, alloc {torch.cuda.memory_allocated() / GiB:.2f} GiB")
    return W


def decode_layer(x, Wl, kc, vc, cos, sin, pos, attn):
    B = x.shape[0]
    S = kc.shape[2]
    h = rmsnorm(x, Wl["ln1"])
    qkv = F.linear(h, Wl["wqkv"])
    q = qkv[:, :NQ * HD].view(B, NQ, HD)
    k = qkv[:, NQ * HD:(NQ + NKV) * HD].view(B, NKV, HD)
    v = qkv[:, (NQ + NKV) * HD:].view(B, NKV, HD)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    kc[:, :, pos] = k
    vc[:, :, pos] = v
    if attn == "sdpa":
        o = F.scaled_dot_product_attention(q.view(B, NQ, 1, HD), kc, vc, enable_gqa=True).reshape(B, H)
    else:
        s = torch.bmm(q.view(B * NKV, 4, HD), kc.view(B * NKV, S, HD).transpose(1, 2))
        o = torch.bmm(softmax_dec(s), vc.view(B * NKV, S, HD)).view(B, H)
    x = x + F.linear(o, Wl["wo"])
    h = rmsnorm(x, Wl["ln2"])
    return x + F.linear(silu_mul(F.linear(h, Wl["wg"]), F.linear(h, Wl["wu"])), Wl["wd"])


def prefill_layer(x, Wl, kc, vc, cos, sin, mask, attn):
    S = x.shape[0]
    h = rmsnorm(x, Wl["ln1"])
    qkv = F.linear(h, Wl["wqkv"])
    q = qkv[:, :NQ * HD].view(S, NQ, HD)
    k = qkv[:, NQ * HD:(NQ + NKV) * HD].view(S, NKV, HD)
    v = qkv[:, (NQ + NKV) * HD:].view(S, NKV, HD)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    kc[0].copy_(k.transpose(0, 1))
    vc[0].copy_(v.transpose(0, 1))
    if attn == "sdpa":
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kc, vc, is_causal=True, enable_gqa=True)
        o = o[0].transpose(0, 1).reshape(S, H)
    else:
        qg = q.view(S, NKV, 4, HD).permute(1, 2, 0, 3).reshape(NKV, 4 * S, HD)
        s = torch.bmm(qg, kc[0].transpose(1, 2))
        o = torch.bmm(softmax_pre(s, mask), vc[0])
        o = o.view(NKV, 4, S, HD).permute(2, 0, 1, 3).reshape(S, H)
    x = x + F.linear(o, Wl["wo"])
    h = rmsnorm(x, Wl["ln2"])
    return x + F.linear(silu_mul(F.linear(h, Wl["wg"]), F.linear(h, Wl["wu"])), Wl["wd"])


# ---------------------------------------------------------------- sessions

def session_header(suite):
    return {"schema": "kiln.meas/1", "suite": suite, "runner": os.path.basename(__file__),
            "runner_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "oplist_sha256": hashlib.sha256(open(OPLIST, "rb").read()).hexdigest() if os.path.exists(OPLIST) else None,
            "started": now(), "device": device_info(), "software": software_info(), "host": host_info(),
            "method": {"warmup": WARMUP, "mode_target_s": MODE_S, "min_samples": MIN_SAMPLES, "max_eager_iters": MAX_EAGER,
                       "graph_min_replays": GRAPH_MIN_REPLAYS, "graph_min_replay_s": GRAPH_MIN_REPLAY_S,
                       "max_graph_calls": MAX_GRAPH_CALLS, "flush_bytes": FLUSH_BYTES, "flush_impl": "fp32 add_ over 256 MiB",
                       "rotate_target_bytes": COLD_TARGET, "rotate_rule": "copies = min(ceil(target/bytes), 0.45*free/alloc, max_graph_calls)",
                       "nvml_dt_s": NVML_DT, "clock_window": "timed region of each mode only (after warmup and queue prefill)"},
            "timing_modes": {
                "flushed": {"overheads_included": ["kernel_launch"], "operands": "cold_dram", "launch": "eager",
                            "note": "256 MiB L2-flush kernel before each iteration; samples are per-iteration CUDA events"},
                "unflushed": {"overheads_included": ["kernel_launch", "l2_warm"], "operands": "l2_warm", "launch": "eager"},
                "graph_unflushed": {"overheads_included": ["kernel_launch", "inter_kernel_gap", "l2_warm"], "operands": "l2_warm", "launch": "cuda_graph"},
                "graph_cold": {"overheads_included": ["kernel_launch", "inter_kernel_gap"], "operands": "cold_dram", "launch": "cuda_graph",
                               "canonical": True},
                "graph": {"overheads_included": ["kernel_launch", "inter_kernel_gap"], "operands": "resident", "launch": "cuda_graph",
                          "note": "seq: one sample per whole-step replay"},
                "eager": {"overheads_included": ["host_dispatch", "kernel_launch", "inter_kernel_gap"], "operands": "resident", "launch": "eager"}},
            "records": []}


class Session:
    def __init__(self, suite):
        self.res = None
        if os.path.exists(OUT):
            try:
                prev = json.load(open(OUT))
                if prev.get("suite") == suite:
                    self.res = prev
            except Exception:
                pass
        if self.res is None:
            self.res = session_header(suite)
        self.res.setdefault("resumed", []).append(now()) if self.res["records"] else None
        self.done = {r["name"] for r in self.res["records"]}
        self.dump()
        fn = mm_spec("w", 8192, 8192, 8192).make(1)[0]
        t = time.time()
        while time.time() - t < 3.0:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        del fn
        torch.cuda.empty_cache()

    def dump(self):
        with open(OUT + ".tmp", "w") as f:
            json.dump(self.res, f, indent=0, separators=(",", ":"))
        os.replace(OUT + ".tmp", OUT)

    def add(self, r):
        self.res["records"].append(r)
        self.done.add(r["name"])
        self.dump()
        c = r["modes"].get("graph_cold") or r["modes"].get("graph") or {}
        log(f"{r['name']:44s} {c.get('median_s', float('nan')) * 1e6:10.2f}us cv={c.get('cv') or 0:.3f} "
            f"{r.get('tflops', 0):7.1f}TF {r.get('gbps', 0):7.0f}GB/s {r['quality']}")

    def run(self, spec):
        if spec.name in self.done:
            return
        try:
            r = measure(spec)
        except Exception as e:
            r = {"name": spec.name, "kind": spec.kind, "dims": spec.dims, "error": repr(e)[:500], "modes": {}, "quality": "rejected"}
            torch.cuda.empty_cache()
        self.add(r)

    def sanity(self, tag):
        out = {}
        for name, spec in (("gemm_8192^3", mm_spec("s", 8192, 8192, 8192)), ("copy_1GiB", mem_spec("s", "copy", GiB))):
            fns = spec.make(1)
            est = pilot(fns[0])
            t = time_eager(fns[0], est, False, iters=max(MIN_SAMPLES, int(0.5 / est)))
            t.pop("samples_s")
            out[name] = {**t, "tflops" if "gemm" in name else "gbps":
                         (spec.flops / t["median_s"] / 1e12) if "gemm" in name else spec.nbytes / t["median_s"] / 1e9}
            del fns
            torch.cuda.empty_cache()
        self.res.setdefault("sanity", {})[tag] = out
        if tag == "end" and "start" in self.res["sanity"]:
            s = self.res["sanity"]
            self.res["sanity"]["drift"] = {k: s["end"][k]["median_s"] / s["start"][k]["median_s"] - 1 for k in out}
        self.dump()
        log("sanity", tag, {k: round(v.get("tflops", v.get("gbps", 0)), 1) for k, v in out.items()})

    def close(self):
        self.sanity("end")
        self.res["finished"] = now()
        self.res["telemetry_error"] = TEL.error
        self.res["smi_after"] = sh("nvidia-smi -q -d CLOCK,POWER,PERFORMANCE")
        self.dump()
        log("DONE")


# ---------------------------------------------------------------- micro suite

def power_step(name, fn, est, load_s=3.0):
    torch.cuda.synchronize()
    time.sleep(0.6)
    t0 = time.time()
    time.sleep(0.5)
    n = int(load_s / est)
    a, b = ev(), ev()
    t_load = time.time()
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    t_end = time.time()
    time.sleep(1.0)
    t1 = time.time()
    raw = TEL.raw(t0, t1)
    steady = [x for x in raw if t_load - t0 + 1.0 <= x[0] <= t_end - t0]
    return {"name": name, "kind": "power_step", "modes": {}, "quality": "ok", "load_calls": n,
            "load_s": a.elapsed_time(b) * 1e-3, "t_load_start": round(t_load - t0, 4), "t_load_end": round(t_end - t0, 4),
            "steady": {"sm_mhz": statistics.median(x[1] for x in steady) if steady else None,
                       "power_w": statistics.median(x[2] for x in steady) if steady else None,
                       "n": len(steady)},
            "series_cols": ["t_s", "sm_mhz", "power_w", "temp_c", "throttle_bits"], "series": raw}


def launch_probe(ses, kname, mk, N):
    name = f"launch/{kname}_x{N}"
    if name in ses.done:
        return
    fn = mk()
    r = {"name": name, "kind": "launch_probe", "dims": {"kernel": kname, "chain": N}, "started": now(), "modes": {}}
    chain = lambda: [fn() for _ in range(N)]
    est = pilot(chain) / N
    try:
        iters = int(min(MAX_EAGER, max(MIN_SAMPLES, MODE_S / (est * N))))
        for _ in range(WARMUP):
            chain()
        torch.cuda.synchronize()
        evs = [(ev(), ev()) for _ in range(iters)]
        t0 = time.time()
        prefill(iters * N * 8e-6 + 1e-3)
        for a, b in evs:
            a.record()
            chain()
            b.record()
        torch.cuda.synchronize()
        r["modes"]["eager_chain"] = stats([a.elapsed_time(b) * 1e-3 / N for a, b in evs], {"clock": TEL.window(t0, time.time())})
        r["modes"]["graph_chain"] = time_graph([fn] * N, est)
        if N <= 100:
            hs = []
            for _ in range(10):
                torch.cuda.synchronize()
                t = time.perf_counter()
                chain()
                hs.append((time.perf_counter() - t) / N)
            r["modes"]["host_submit"] = stats(hs)
        if N == 1:
            rt = []
            for _ in range(100):
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                rt.append(time.perf_counter() - t)
            r["modes"]["sync_roundtrip"] = stats(rt)
            g = capture([fn])
            rt = []
            for _ in range(100):
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                rt.append(time.perf_counter() - t)
            r["modes"]["graph_sync_roundtrip"] = stats(rt)
            del g
    except Exception as e:
        r["error"] = repr(e)[:300]
    r["modes"]["graph_cold"] = r["modes"].get("graph_chain", {})
    ses.add(finish(r))


def run_micro():
    ses = Session("micro")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    fams = llm_families()
    dropped = ses.res.setdefault("leakage_dropped", [])

    def gated(spec):
        d = spec.dims
        if spec.kind in ("gemm", "linear", "bmm") and family(spec.kind, d["m"], d["n"], d["k"], d["batch"]) in fams:
            if spec.name not in dropped:
                dropped.append(spec.name)
            return
        ses.run(spec)

    # launch-overhead probes
    tiny = {
        "sleep0": lambda: (lambda: torch.cuda._sleep(0)),
        "add_1elem": lambda: (lambda a=torch.ones(1, **BF), b=torch.ones(1, **BF), c=torch.empty(1, **BF): torch.add(a, b, out=c)),
        "gemm_16": lambda: (lambda a=torch.randn(16, 16, **BF), c=torch.empty(16, 16, **BF): torch.matmul(a, a, out=c)),
        "gemm_64": lambda: (lambda a=torch.randn(64, 64, **BF), c=torch.empty(64, 64, **BF): torch.matmul(a, a, out=c)),
        "gemm_128": lambda: (lambda a=torch.randn(128, 128, **BF), c=torch.empty(128, 128, **BF): torch.matmul(a, a, out=c)),
    }
    for kname, mk in tiny.items():
        for N in (1, 10, 100, 1000):
            launch_probe(ses, kname, mk, N)

    # power steps (clock under sustained load)
    for name, spec in (("power_step/gemm_8192", mm_spec("p", 8192, 8192, 8192)),
                       ("power_step/gemm_16384", mm_spec("p", 16384, 16384, 16384)),
                       ("power_step/copy_1GiB", mem_spec("p", "copy", GiB))):
        if name not in ses.done:
            fns = spec.make(1)
            r = power_step(name, fns[0], pilot(fns[0]))
            del fns
            torch.cuda.empty_cache()
            ses.add(r)

    # HBM read/write/copy 1 MiB .. 4 GiB (16 sizes)
    sizes = [2**i * MiB for i in range(13)] + [48 * MiB, 384 * MiB, 1536 * MiB]
    for s in sorted(sizes):
        for kind in ("read", "write", "copy"):
            ses.run(mem_spec(f"hbm/{kind}_{s // MiB}MiB", kind, s, group="hbm"))

    # L2-resident sweeps
    for kib in (256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768):
        for kind in ("read", "copy"):
            sp = mem_spec(f"l2/{kind}_{kib}KiB", kind, kib * 1024, group="l2")
            sp.modes = ("unflushed", "graph_unflushed")
            ses.run(sp)

    # GEMV / skinny weight streaming (non-LLM widths)
    W = (3072, 5120, 10240, 12288, 24576)
    for m in (1, 4, 16, 64):
        for n in W:
            for k in W:
                gated(mm_spec(f"gemv/gemm_{m}_{n}_{k}", m, n, k, group="gemv"))

    # GEMM: square, K-heavy, rectangular grid
    gm = [(s, s, s) for s in (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384)]
    gm += [(m, m, k) for m in (256, 1024, 2048) for k in (8192, 16384, 32768, 65536)]
    gm += [(m, n, k) for m in (512, 2048, 8192) for n in (3072, 5120, 10240) for k in (3072, 5120, 10240)]
    for m, n, k in gm:
        gated(mm_spec(f"gemm/gemm_{m}_{n}_{k}", m, n, k, group="gemm"))

    # bmm (attention-like, non-LLM families)
    for b in (16, 128):
        for m in (4, 1024):
            for n, k in ((1024, 128), (4096, 64), (128, 4096)):
                gated(mm_spec(f"bmm/bmm_{b}_{m}_{n}_{k}", m, n, k, batch=b, weight=False, group="bmm"))

    # elementwise sweeps
    for op in ("add", "scale", "silu"):
        for p in range(12, 31, 2):
            ses.run(ew_spec(f"ew/{op}_{2**p // 1024}KiB", op, 2**p, group="elementwise"))

    ses.close()


# ---------------------------------------------------------------- per-op suite

def aux_specs(phase):
    """Isolated non-matmul ops of one step, same implementations and shapes as seq. Returns (spec, count)."""
    pre = phase.startswith("prefill")
    B = 1 if pre else int(phase.split("_b")[1])
    T, S = (2048, 2048) if pre else (B, 2048)
    cos1, sin1 = rope_tables(torch.arange(T, device="cuda") if pre else torch.tensor([S - 1], device="cuda"))
    specs = []

    def add(name, make, nbytes, count, alloc=None, rot_cap=None):
        specs.append((Spec(f"{phase}/{name}", "aux", make, nbytes, alloc=alloc, rot_cap=rot_cap, phase=phase, op=name, count=count,
                           impl="torch eager (as seq)"), count))
    w = torch.ones(H, **BF)
    add("rmsnorm", lambda c: [lambda x=x: rmsnorm(x, w) for x in bank(c, (T, H))], 4 * T * H, 2 * NL + (0 if pre else 1))
    if pre:
        add("rmsnorm_final", lambda c: [lambda x=x: rmsnorm(x, w) for x in bank(c, (1, H))], 4 * H, 1)
    add("rope_qk", lambda c: [lambda q=q, k=k: (rope(q, cos1, sin1), rope(k, cos1, sin1))
                              for q, k in zip(bank(c, (T, NQ, HD)), bank(c, (T, NKV, HD)))], 4 * T * (NQ + NKV) * HD, NL)
    if pre:
        def kvw(c):
            KC, K = bank(c, (1, NKV, S, HD), init=None), bank(c, (S, NKV, HD))
            return [lambda kc=kc, k=k: kc[0].copy_(k.transpose(0, 1)) for kc, k in zip(KC, K)]
        add("kv_write", kvw, 4 * S * NKV * HD, 2 * NL)
        add("q_permute", lambda c: [lambda q=q: q.view(S, NKV, 4, HD).permute(1, 2, 0, 3).reshape(NKV, 4 * S, HD)
                                    for q in bank(c, (S, NQ, HD))], 4 * S * H, NL)
        add("o_permute", lambda c: [lambda o=o: o.view(NKV, 4, S, HD).permute(2, 0, 1, 3).reshape(S, H)
                                    for o in bank(c, (NKV, 4 * S, HD))], 4 * S * H, NL)
        mask = torch.full((S, S), float("-inf"), device="cuda").triu_(1)
        add("softmax", lambda c: [lambda s=s: softmax_pre(s, mask) for s in bank(c, (NKV, 4 * S, S))], 4 * NKV * 4 * S * S, NL)
    else:
        def kvw(c):
            kcs = bank(max(1, c // S + 1), (B, NKV, S, HD), init=None)
            K = bank(c, (B, NKV, HD))
            return [lambda kc=kcs[i // S], p=i % S, k=K[i]: kc[:, :, p].copy_(k) for i in range(c)]
        add("kv_write", kvw, 2 * B * NKV * HD, 2 * NL, alloc=2 * B * NKV * HD + 2 * B * NKV * S * HD // S)
        add("softmax", lambda c: [lambda s=s: softmax_dec(s) for s in bank(c, (B * NKV, 4, S))], 4 * B * NKV * 4 * S, NL)
    add("residual_add", lambda c: [lambda a=a, b=b: a + b for a, b in zip(bank(c, (T, H)), bank(c, (T, H)))], 6 * T * H, 2 * NL)
    add("silu_mul", lambda c: [lambda g=g, u=u: silu_mul(g, u) for g, u in zip(bank(c, (T, FF)), bank(c, (T, FF)))], 6 * T * FF, NL)
    emb = torch.randn(V, H, **BF)
    add("embedding", lambda c: [lambda t=t: F.embedding(t, emb) for t in
                                [torch.randint(0, V, (T,), device="cuda") for _ in range(c)]], 4 * T * H, 1, alloc=4 * T * H, rot_cap=1024)
    Bh = 1 if pre else B
    add("argmax", lambda c: [lambda x=x: x.argmax(-1) for x in bank(c, (Bh, V))], 2 * Bh * V, 1)
    return specs


def run_suite():
    ses = Session("suite")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    seen = set()
    for o in json.load(open(OPLIST)):
        if o["key"] in seen:
            continue
        seen.add(o["key"])
        kw = dict(m=o["m"], n=o["n"], k=o["k"], batch=o["batch"], weight=o["weight"])
        meta = dict(phase=o["phase"], op=o["op"], count=o["count"], group="llm_op")
        ses.run(mm_spec(o["key"], **kw, **meta))
        if o["weight"]:
            ses.run(mm_spec(o["key"] + "_linear", **kw, linear=True, **meta))
    for phase in ("decode_b1", "decode_b8", "decode_b32", "prefill_b1"):
        for spec, _ in aux_specs(phase):
            ses.run(spec)
            torch.cuda.empty_cache()
    shapes = [(s, s, s) for s in (256, 512, 1024, 2048, 4096, 8192)]
    shapes += [(m, nk, nk) for nk in (4096, 8192, 14336) for m in (1, 8, 16, 32, 64, 128, 256)]
    shapes += [(256, 256, 65536), (1024, 1024, 32768), (2048, 2048, 16384), (128, 4096, 65536), (4096, 128, 65536), (64, 64, 262144)]
    for m, n, k in shapes:
        ses.run(mm_spec(f"sweep/gemm_{m}_{n}_{k}", m, n, k, group="legacy_sweep"))
    ses.close()


# ---------------------------------------------------------------- whole-step sequence

def kernel_profile(fn, full=False):
    try:
        from torch.profiler import ProfilerActivity, profile
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")]
        if not evs:
            return {"n_kernels": 0}
        by = {}
        for e in evs:
            d = by.setdefault(e.name[:120], [0, 0.0])
            d[0] += 1
            d[1] += e.time_range.elapsed_us() * 1e-6
        busy = sum(v[1] for v in by.values())
        span = (max(e.time_range.end for e in evs) - min(e.time_range.start for e in evs)) * 1e-6
        gem = sum(v[1] for k, v in by.items() if any(s in k.lower() for s in ("gemm", "cutlass", "xmma", "sm80", "ampere", "flash", "fmha")))
        top = sorted(by.items(), key=lambda kv: -kv[1][1])[:20]
        r = {"n_kernels": sum(v[0] for v in by.values()), "kernel_busy_s": busy, "kernel_span_s": span,
             "matmul_like_busy_s": gem, "top": [[k, v[0], v[1]] for k, v in top]}
        if full:
            r["all"] = [[k, v[0], v[1]] for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])]
        return r
    except Exception as e:
        return {"error": repr(e)[:300]}


def time_step_graph(fn, iters=50, warmup=10, full=False):
    g = capture([fn])
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    evs = [(ev(), ev()) for _ in range(iters)]
    t0 = time.time()
    lead = prefill(2e-3)
    for a, b in evs:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    t1 = time.time()
    prof = kernel_profile(g.replay, full)
    del g
    return stats([a.elapsed_time(b) * 1e-3 for a, b in evs], {"clock": TEL.window(t0 + lead, t1), "kernels": prof})


def run_seq():
    ses = Session("seq")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    W = make_weights(NL)
    S = 2048
    ses.res["model"] = {"name": "llama-3-8b (random bf16 weights)", "hidden": H, "layers": NL, "q_heads": NQ, "kv_heads": NKV,
                        "head_dim": HD, "ffn": FF, "vocab": V, "kv_len": S, "attn_impls": ["bmm", "sdpa"],
                        "weight_bytes": sum(t.numel() * 2 for t in [W["emb"], W["lm"]] + [x for l in W["layers"] for x in l.values()]),
                        "notes": "one weight copy; weights (~16 GB) >> L2 so every step streams weights from HBM; "
                                 "lm_head on last token only for prefill; sample = argmax"}
    ses.dump()
    for phase in ("decode_b1", "decode_b8", "decode_b32", "prefill_b1"):
        pre = phase.startswith("prefill")
        B = 1 if pre else int(phase.split("_b")[1])
        kcs = [torch.zeros(B, NKV, S, HD, **BF) for _ in range(NL)]
        vcs = [torch.zeros(B, NKV, S, HD, **BF) for _ in range(NL)]
        for kc in kcs + vcs:
            kc.normal_()
        if pre:
            tok = torch.randint(0, V, (S,), device="cuda")
            cos, sin = rope_tables(torch.arange(S, device="cuda"))
            mask = torch.full((S, S), float("-inf"), device="cuda").triu_(1)
        else:
            tok = torch.randint(0, V, (B,), device="cuda")
            cos, sin = rope_tables(torch.tensor([S - 1], device="cuda"))
            mask = None
        out = {}

        def step(attn, layers=range(NL)):
            def f():
                x = F.embedding(tok, W["emb"])
                for i in layers:
                    if pre:
                        x = prefill_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, mask, attn)
                    else:
                        x = decode_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, S - 1, attn)
                if pre:
                    x = x[-1:]
                out["tok"] = F.linear(rmsnorm(x, W["lnf"]), W["lm"]).argmax(-1)
            return f

        def layer_only(attn, i=0, reps=1):
            x0 = torch.randn(S if pre else B, H, **BF)
            def f():
                x = x0
                for _ in range(reps):
                    if pre:
                        x = prefill_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, mask, attn)
                    else:
                        x = decode_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, S - 1, attn)
                out["x"] = x
            return f

        for attn in ("bmm", "sdpa"):
            sfx = "" if attn == "bmm" else "_sdpa"
            name = f"seq/{phase}/step{sfx}"
            if name not in ses.done:
                r = {"name": name, "kind": "sequence", "scope": "step", "phase": phase, "attn": attn, "batch": B, "kv_len": S,
                     "tokens": S if pre else B, "n_layers": NL, "started": now(), "modes": {}}
                try:
                    f = step(attn)
                    r["modes"]["graph"] = time_step_graph(f)
                    est = r["modes"]["graph"]["median_s"]
                    r["modes"]["eager"] = time_eager(f, est, False, iters=int(min(100, max(20, 2.0 / est))))
                    r["modes"]["eager"]["kernels"] = kernel_profile(f)
                    r["tokens_per_s"] = r["tokens"] / est
                except Exception as e:
                    r["error"] = repr(e)[:500]
                r["modes"]["graph_cold"] = r["modes"].get("graph", {})
                ses.add(finish(r))
                r["modes"].pop("graph_cold", None)
                ses.dump()
                torch.cuda.empty_cache()
            name = f"seq/{phase}/layer1{sfx}"
            if name not in ses.done:
                r = {"name": name, "kind": "sequence", "scope": "layer", "phase": phase, "attn": attn, "batch": B, "kv_len": S,
                     "n_layers": 1, "started": now(), "modes": {}}
                try:
                    f = layer_only(attn)
                    est = pilot(f)
                    r["modes"]["graph_replay1"] = time_step_graph(f, iters=int(min(300, max(50, 0.5 / est))))
                    f32 = layer_only(attn, reps=NL)
                    t = time_step_graph(f32, iters=int(min(50, max(10, 0.5 / (est * NL)))))
                    for kk in ("median_s", "min_s", "p90_s", "mean_s"):
                        t[kk] /= NL
                    t["samples_s"] = [x / NL for x in t["samples_s"]]
                    t["note"] = "layer 0 repeated 32x in one graph (same weights, 436 MB/layer >> L2), per-layer time"
                    r["modes"]["graph_x32"] = t
                except Exception as e:
                    r["error"] = repr(e)[:500]
                r["modes"]["graph_cold"] = r["modes"].get("graph_x32", {})
                ses.add(finish(r))
                r["modes"].pop("graph_cold", None)
                ses.dump()
                torch.cuda.empty_cache()
        del kcs, vcs
        torch.cuda.empty_cache()
    ses.close()


# ---------------------------------------------------------------- out-of-sample sequences (stack recipe check)

GJ = dict(H=4096, NL=28, NH=16, HD=256, ROT=64, FF=16384, V=50400, EPS=1e-5, MAXPOS=2048)
L2C = dict(H=8192, NQ=64, NKV=8, HD=128, FF=28672, V=32000, EPS=1e-5, THETA=10000.0)


def gptj_weights(nl):
    t0 = time.time()
    c = GJ
    init = lambda *s: torch.randn(*s, **BF).mul_(0.02)
    W = {"wte": init(c["V"], c["H"]), "lm": init(c["V"], c["H"]), "lm_b": init(c["V"]),
         "lnf": torch.ones(c["H"], **BF), "lnf_b": torch.zeros(c["H"], **BF), "layers": []}
    for _ in range(nl):
        W["layers"].append({"ln": torch.ones(c["H"], **BF), "ln_b": torch.zeros(c["H"], **BF),
                            "wq": init(c["H"], c["H"]), "wk": init(c["H"], c["H"]), "wv": init(c["H"], c["H"]),
                            "wo": init(c["H"], c["H"]), "fi": init(c["FF"], c["H"]), "fi_b": init(c["FF"]),
                            "fo": init(c["H"], c["FF"]), "fo_b": init(c["H"])})
    inv = 1.0 / (10000 ** (torch.arange(0, c["ROT"], 2, dtype=torch.int64, device="cuda").float() / c["ROT"]))
    sinusoid = torch.einsum("i,j->ij", torch.arange(c["MAXPOS"], dtype=torch.int64, device="cuda").float(), inv)
    W["embed_positions"] = torch.cat((torch.sin(sinusoid), torch.cos(sinusoid)), dim=1).to(torch.bfloat16)
    torch.cuda.synchronize()
    log(f"gptj weights {nl} layers in {time.time() - t0:.1f}s, alloc {torch.cuda.memory_allocated() / GiB:.2f} GiB")
    return W


def gelu_new(x):
    # transformers.activations.NewGELUActivation (ACT2FN["gelu_new"]), as written there
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(t, sin, cos):
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (t * cos) + (rotate_every_two(t) * sin)


def gptj_layer(x, Wl, kc, vc, pos_ids, embed_positions, pos):
    """transformers GPTJBlock + GPTJAttention op sequence, bf16; static KV cache written in place (pos = int for
    decode, None for prefill writing positions [0, S)), attention through SDPA instead of the fp32 _attn."""
    c = GJ
    B, S, _ = x.shape
    residual = x
    h = F.layer_norm(x, (c["H"],), Wl["ln"], Wl["ln_b"], c["EPS"])
    q = F.linear(h, Wl["wq"]).view(B, S, c["NH"], c["HD"])
    k = F.linear(h, Wl["wk"]).view(B, S, c["NH"], c["HD"])
    v = F.linear(h, Wl["wv"]).view(B, S, c["NH"], c["HD"]).permute(0, 2, 1, 3)
    ep = embed_positions.repeat(pos_ids.shape[0], 1, 1)
    rep = pos_ids.unsqueeze(-1).repeat(1, 1, ep.shape[-1])
    sincos = torch.gather(ep, 1, rep)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    R = c["ROT"]
    k_rot = apply_rotary_pos_emb(k[:, :, :, :R], sin, cos)
    q_rot = apply_rotary_pos_emb(q[:, :, :, :R], sin, cos)
    k = torch.cat([k_rot, k[:, :, :, R:]], dim=-1).permute(0, 2, 1, 3)
    q = torch.cat([q_rot, q[:, :, :, R:]], dim=-1).permute(0, 2, 1, 3)
    if pos is None:
        kc.copy_(k)
        vc.copy_(v)
        o = F.scaled_dot_product_attention(q, kc, vc, is_causal=True)
    else:
        kc[:, :, pos] = k[:, :, 0]
        vc[:, :, pos] = v[:, :, 0]
        o = F.scaled_dot_product_attention(q, kc, vc)
    o = o.permute(0, 2, 1, 3).contiguous().view(B, S, c["H"])
    a = F.linear(o, Wl["wo"])
    m = F.linear(gelu_new(F.linear(h, Wl["fi"], Wl["fi_b"])), Wl["fo"], Wl["fo_b"])
    return a + m + residual


def l2_weights(nl):
    t0 = time.time()
    c = L2C
    init = lambda *s: torch.randn(*s, **BF).mul_(0.02)
    W = {"emb": init(c["V"], c["H"]), "lm": init(c["V"], c["H"]), "lnf": torch.ones(c["H"], **BF), "layers": []}
    for _ in range(nl):
        W["layers"].append({"ln1": torch.ones(c["H"], **BF), "wqkv": init((c["NQ"] + 2 * c["NKV"]) * c["HD"], c["H"]),
                            "wo": init(c["H"], c["H"]), "ln2": torch.ones(c["H"], **BF), "wg": init(c["FF"], c["H"]),
                            "wu": init(c["FF"], c["H"]), "wd": init(c["H"], c["FF"])})
    torch.cuda.synchronize()
    log(f"llama2-70b weights {nl} layers in {time.time() - t0:.1f}s, alloc {torch.cuda.memory_allocated() / GiB:.2f} GiB")
    return W


def l2_rope(x, cos, sin):
    hd = L2C["HD"]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def l2_rope_tables(positions):
    hd = L2C["HD"]
    inv = 1.0 / L2C["THETA"] ** (torch.arange(0, hd, 2, device="cuda").float() / hd)
    f = positions.float()[:, None] * inv[None]
    return f.cos().to(torch.bfloat16)[:, None, :], f.sin().to(torch.bfloat16)[:, None, :]


def l2_rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + L2C["EPS"])).to(x.dtype) * w


def l2_layer(x, Wl, kc, vc, cos, sin, pos):
    """decode_layer / prefill_layer (SDPA) with Llama-2-70B shapes; pos None = prefill."""
    c = L2C
    T = x.shape[0]
    NQ, NKV, HD = c["NQ"], c["NKV"], c["HD"]
    h = l2_rmsnorm(x, Wl["ln1"])
    qkv = F.linear(h, Wl["wqkv"])
    q = qkv[:, :NQ * HD].view(T, NQ, HD)
    k = qkv[:, NQ * HD:(NQ + NKV) * HD].view(T, NKV, HD)
    v = qkv[:, (NQ + NKV) * HD:].view(T, NKV, HD)
    q, k = l2_rope(q, cos, sin), l2_rope(k, cos, sin)
    if pos is None:
        kc[0].copy_(k.transpose(0, 1))
        vc[0].copy_(v.transpose(0, 1))
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kc, vc, is_causal=True, enable_gqa=True)
        o = o[0].transpose(0, 1).reshape(T, c["H"])
    else:
        kc[:, :, pos] = k
        vc[:, :, pos] = v
        o = F.scaled_dot_product_attention(q.view(T, NQ, 1, HD), kc, vc, enable_gqa=True).reshape(T, c["H"])
    x = x + F.linear(o, Wl["wo"])
    h = l2_rmsnorm(x, Wl["ln2"])
    return x + F.linear(F.silu(F.linear(h, Wl["wg"])) * F.linear(h, Wl["wu"]), Wl["wd"])


def oos_record(ses, name, meta, f, iters=50):
    if name in ses.done:
        return
    r = {"name": name, "kind": "sequence", "attn": "sdpa", "started": now(), "modes": {}, **meta}
    try:
        r["modes"]["graph"] = time_step_graph(f, iters=iters, full=True)
        r["tokens_per_s"] = r.get("tokens", 0) / r["modes"]["graph"]["median_s"]
    except Exception as e:
        r["error"] = repr(e)[:500]
    r["modes"]["graph_cold"] = r["modes"].get("graph", {})
    ses.add(finish(r))
    r["modes"].pop("graph_cold", None)
    ses.dump()
    torch.cuda.empty_cache()


def run_gptj(ses):
    c = GJ
    W = gptj_weights(c["NL"])
    ses.res.setdefault("models", {})["gptj_6b"] = {
        "name": "gpt-j-6b (random bf16 weights)", "hidden": c["H"], "layers": c["NL"], "heads": c["NH"], "head_dim": c["HD"],
        "rotary_dim": c["ROT"], "ffn": c["FF"], "vocab": c["V"], "norm": "LayerNorm w+b", "act": "gelu_new (python formula)",
        "notes": "transformers modeling_gptj op structure (separate q/k/v Linear without bias, per-layer embed_positions repeat + sincos gather, "
                 "rotate_every_two partial rotary, parallel attn+MLP from ln_1, fc_in/fc_out/lm_head with bias); bf16 "
                 "throughout incl. embed_positions; static KV cache written in place; SDPA (flash) instead of fp32 _attn; "
                 "lm_head on last token only for prefill; sample = argmax"}
    ses.dump()
    for phase, B, S in (("decode_b1", 1, 2048), ("decode_b8", 8, 2048), ("decode_b32", 32, 2048), ("decode_b32_kv1024", 32, 1024),
                        ("prefill_b1", 1, 2048)):
        pre = phase.startswith("prefill")
        if phase == "decode_b32_kv1024" and "seq_oos/gptj_6b/decode_b32/step" in ses.done and \
                "error" not in next(r for r in ses.res["records"] if r["name"] == "seq_oos/gptj_6b/decode_b32/step"):
            continue
        need = 2 * c["NL"] * B * c["NH"] * S * c["HD"] * 2
        free = torch.cuda.mem_get_info()[0]
        if need > free - 1.5 * GiB:
            ses.add(finish({"name": f"seq_oos/gptj_6b/{phase}/step", "kind": "sequence", "phase": phase, "model": "gptj_6b",
                            "error": f"kv cache {need / GiB:.1f} GiB > free {free / GiB:.1f} GiB - 1.5", "modes": {}}))
            continue
        kcs = [torch.empty(B, c["NH"], S, c["HD"], **BF).normal_() for _ in range(c["NL"])]
        vcs = [torch.empty(B, c["NH"], S, c["HD"], **BF).normal_() for _ in range(c["NL"])]
        if pre:
            tok = torch.randint(0, c["V"], (1, S), device="cuda")
            pos_ids, pos = torch.arange(S, device="cuda")[None], None
        else:
            tok = torch.randint(0, c["V"], (B, 1), device="cuda")
            pos_ids, pos = torch.full((B, 1), S - 1, device="cuda", dtype=torch.long), S - 1
        out = {}

        def step(layers=range(c["NL"])):
            def f():
                x = F.embedding(tok, W["wte"])
                for i in layers:
                    x = gptj_layer(x, W["layers"][i], kcs[i], vcs[i], pos_ids, W["embed_positions"], pos)
                if pre:
                    x = x[:, -1:]
                x = F.layer_norm(x, (c["H"],), W["lnf"], W["lnf_b"], c["EPS"])
                out["tok"] = F.linear(x, W["lm"], W["lm_b"]).argmax(-1)
            return f

        def layer_only(reps):
            x0 = torch.randn(B, S if pre else 1, c["H"], **BF)
            def f():
                x = x0
                for _ in range(reps):
                    x = gptj_layer(x, W["layers"][0], kcs[0], vcs[0], pos_ids, W["embed_positions"], pos)
                out["x"] = x
            return f

        meta = {"phase": phase, "model": "gptj_6b", "batch": B, "kv_len": S, "tokens": S if pre else B}
        oos_record(ses, f"seq_oos/gptj_6b/{phase}/step", {**meta, "scope": "step", "n_layers": c["NL"]}, step())
        oos_record(ses, f"seq_oos/gptj_6b/{phase}/layer1", {**meta, "scope": "layer", "n_layers": 1}, layer_only(1), iters=200)
        name = f"seq_oos/gptj_6b/{phase}/layer_x{c['NL']}"
        oos_record(ses, name, {**meta, "scope": "layer", "n_layers": c["NL"],
                               "note": "layer 0 repeated NL times in one graph (same weights, 402 MB/layer >> L2); divide by NL"},
                   layer_only(c["NL"]), iters=30)
        del kcs, vcs
        torch.cuda.empty_cache()
    del W
    torch.cuda.empty_cache()


def run_l2_70b(ses):
    c = L2C
    NLmax = 4
    W = l2_weights(NLmax)
    ses.res.setdefault("models", {})["llama2_70b"] = {
        "name": "llama-2-70b layers (random bf16 weights)", "hidden": c["H"], "q_heads": c["NQ"], "kv_heads": c["NKV"],
        "head_dim": c["HD"], "ffn": c["FF"], "vocab": c["V"], "rope_theta": c["THETA"],
        "notes": "same op sequence as the Llama-3-8B seq (decode_layer/prefill_layer, SDPA); whole steps with 1, 2, 4 "
                 "distinct layers (embedding, layers, final norm, lm_head, argmax) and layer-only graphs; 1.71 GB/layer >> L2"}
    ses.dump()
    for phase, B in (("decode_b1", 1), ("decode_b8", 8), ("decode_b32", 32), ("prefill_b1", 1)):
        pre = phase.startswith("prefill")
        S = 2048
        kcs = [torch.empty(B, c["NKV"], S, c["HD"], **BF).normal_() for _ in range(NLmax)]
        vcs = [torch.empty(B, c["NKV"], S, c["HD"], **BF).normal_() for _ in range(NLmax)]
        if pre:
            tok = torch.randint(0, c["V"], (S,), device="cuda")
            cos, sin = l2_rope_tables(torch.arange(S, device="cuda"))
            pos = None
        else:
            tok = torch.randint(0, c["V"], (B,), device="cuda")
            cos, sin = l2_rope_tables(torch.tensor([S - 1], device="cuda"))
            pos = S - 1
        out = {}

        def step(n):
            def f():
                x = F.embedding(tok, W["emb"])
                for i in range(n):
                    x = l2_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, pos)
                if pre:
                    x = x[-1:]
                out["tok"] = F.linear(l2_rmsnorm(x, W["lnf"]), W["lm"]).argmax(-1)
            return f

        def layers_only(idx):
            x0 = torch.randn(S if pre else B, c["H"], **BF)
            def f():
                x = x0
                for i in idx:
                    x = l2_layer(x, W["layers"][i], kcs[i], vcs[i], cos, sin, pos)
                out["x"] = x
            return f

        meta = {"phase": phase, "model": "llama2_70b", "batch": B, "kv_len": S, "tokens": S if pre else B}
        for n in (1, 2, 4):
            oos_record(ses, f"seq_oos/llama2_70b/{phase}/step_l{n}", {**meta, "scope": "step", "n_layers": n}, step(n), iters=100)
        oos_record(ses, f"seq_oos/llama2_70b/{phase}/layer1", {**meta, "scope": "layer", "n_layers": 1}, layers_only([0]), iters=200)
        oos_record(ses, f"seq_oos/llama2_70b/{phase}/layers2", {**meta, "scope": "layer", "n_layers": 2}, layers_only([0, 1]), iters=200)
        oos_record(ses, f"seq_oos/llama2_70b/{phase}/layer_x8", {**meta, "scope": "layer", "n_layers": 8,
                                                                 "note": "layer 0 repeated 8x in one graph; divide by 8"},
                   layers_only([0] * 8), iters=100)
        del kcs, vcs
        torch.cuda.empty_cache()
    del W
    torch.cuda.empty_cache()


def run_seq_oos():
    ses = Session("seq_oos")
    if "start" not in ses.res.get("sanity", {}):
        ses.sanity("start")
    for m in os.environ.get("OOS_MODELS", "gptj_6b,llama2_70b").split(","):
        {"gptj_6b": run_gptj, "llama2_70b": run_l2_70b}[m](ses)
    ses.close()


if __name__ == "__main__":
    {"micro": run_micro, "suite": run_suite, "seq": run_seq, "seq_oos": run_seq_oos}[SUITE]()
