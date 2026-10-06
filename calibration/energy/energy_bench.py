"""A100 per-component energy microbenchmarks (schema kiln.energy/1, see README.md).

Usage:
  python energy_bench.py run <out.json> [reps]       full suite, rewritten after every record
  python energy_bench.py once <bench> <level>         one short launch of a bench (for ncu)
"""
import ctypes
import datetime
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import pynvml
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WARM_S, MEAS_S, STEP_S, BATCH_S = 1.0, 5.5, 0.025, 0.2
GRID_FULL, BLOCK, DYN_SMEM = 108, 1024, 100 * 1024
MiB = 2**20
LGC = int(os.environ.get("ENERGY_LGC", "1410"))
ONLY = [x for x in os.environ.get("ENERGY_ONLY", "").split(",") if x]

# ---------------- CUDA driver API via ctypes on torch's primary context ----------------
cu = ctypes.CDLL("libcuda.so.1")


def ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} -> CUresult {r}")


def build():
    cubin = "/tmp/energy_kernels.cubin"
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_80", "-cubin", "-o", cubin, os.path.join(HERE, "kernels.cu")],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    torch.zeros(1, device="cuda")
    mod = ctypes.c_void_p()
    ck(cu.cuModuleLoad(ctypes.byref(mod), cubin.encode()), "cuModuleLoad")
    fns = {}
    for name in ["spin", "k_ffma", "smem_rd", "l1_rd", "gld_cg", "gcopy"]:
        f = ctypes.c_void_p()
        ck(cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), name)
        ck(cu.cuFuncSetAttribute(f, 8, DYN_SMEM), "maxdynsmem")  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        fns[name] = f
    return fns, sass


def launch(f, grid, block, smem, *args):
    holders = [a if isinstance(a, ctypes._SimpleCData) else ctypes.c_uint64(a) for a in args]
    ptrs = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    ck(cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, None, ptrs, None), "launch")


# ---------------- NVML telemetry ----------------
pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)


def energy_mj():
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(H)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.on, self.rows, self.stop_flag = False, [], False

    def run(self):
        k = 0
        while not self.stop_flag:
            if self.on:
                row = {"t": time.time(), "p": pynvml.nvmlDeviceGetPowerUsage(H) / 1e3}
                if k % 5 == 0:
                    row["sm"] = pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)
                    row["mem"] = pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_MEM)
                    row["temp"] = pynvml.nvmlDeviceGetTemperature(H, pynvml.NVML_TEMPERATURE_GPU)
                    row["thr"] = pynvml.nvmlDeviceGetCurrentClocksEventReasons(H)
                self.rows.append(row)
                k += 1
            time.sleep(0.005)

    def window(self):
        self.rows = []
        self.on = True

    def close(self):
        self.on = False
        r = self.rows
        p = [x["p"] for x in r]
        sm = [x["sm"] for x in r if "sm" in x]
        thr = 0
        for x in r:
            thr |= x.get("thr", 0)
        return {"n_power": len(p), "power_sampled_w": statistics.fmean(p) if p else None,
                "power_sampled_sd_w": statistics.pstdev(p) if len(p) > 1 else None,
                "sm_mhz_mean": statistics.fmean(sm) if sm else None, "sm_mhz_min": min(sm) if sm else None,
                "sm_mhz_max": max(sm) if sm else None,
                "mem_mhz": statistics.fmean([x["mem"] for x in r if "mem" in x]) if sm else None,
                "temp_c_start": next((x["temp"] for x in r if "temp" in x), None),
                "temp_c_end": next((x["temp"] for x in reversed(r) if "temp" in x), None),
                "clock_event_reasons_union": thr}


SAMP = Sampler()
SAMP.start()


def counter_granularity():
    e0 = energy_mj()
    t0 = time.time()
    changes, last = [], e0
    while time.time() - t0 < 1.0:
        e = energy_mj()
        if e != last:
            changes.append(time.time())
            last = e
    dts = [b - a for a, b in zip(changes, changes[1:])]
    return {"updates_per_s": len(changes), "median_update_interval_s": statistics.median(dts) if dts else None}


# ---------------- benches ----------------
class Bench:
    """step(): enqueue one work kernel (~STEP_S); work: dict of unit -> count per step."""


def setup(fns):
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    l1buf = torch.randint(-2**31, 2**31 - 1, (4 * 1024 // 4,), dtype=torch.int32, device="cuda")
    l2n = 16 * MiB // 16
    l2buf = torch.randint(-2**31, 2**31 - 1, (l2n * 4,), dtype=torch.int32, device="cuda")
    hbn = 2048 * MiB // 16
    hbuf = torch.randint(-2**31, 2**31 - 1, (hbn * 4,), dtype=torch.int32, device="cuda")
    cpn = 1024 * MiB // 16
    csrc = torch.randint(-2**31, 2**31 - 1, (cpn * 4,), dtype=torch.int32, device="cuda")
    cdst = torch.empty_like(csrc)
    keep = [out, l1buf, l2buf, hbuf, csrc, cdst]
    T = BLOCK

    def k_spin(nap):
        return lambda g, it: launch(fns["spin"], g, T, DYN_SMEM, ctypes.c_uint64(int(it)), ctypes.c_int(nap))

    benches = {
        "spin_busy": dict(fn=k_spin(0), unit=None, per_iter=None, kind="ns"),
        "spin_nap": dict(fn=k_spin(1), unit=None, per_iter=None, kind="ns"),
        "ffma": dict(fn=lambda g, it: launch(fns["k_ffma"], g, T, DYN_SMEM, out.data_ptr(), ctypes.c_float(-1.9),
                                             ctypes.c_int(it)),
                     unit="fp32_fma", per_iter=lambda g: g * T * 32,
                     bytes_note="register-resident; 3 RF reads + 1 RF write of 4 B per FMA"),
        "smem_read": dict(fn=lambda g, it: launch(fns["smem_rd"], g, T, DYN_SMEM, out.data_ptr(), ctypes.c_int(it), ctypes.c_uint32(0)),
                          unit="smem_byte", per_iter=lambda g: g * T * 8 * 16,
                          alu_per_unit=1 / 8, alu_note="LOP3 per byte (2 LOP3 per 16 B load)"),
        "l1_read": dict(fn=lambda g, it: launch(fns["l1_rd"], g, T, DYN_SMEM, l1buf.data_ptr(), out.data_ptr(),
                                                ctypes.c_int(it), ctypes.c_uint64(0)),
                        unit="l1_byte", per_iter=lambda g: g * T * 8 * 16, alu_per_unit=1 / 8),
        "l2_read": dict(fn=lambda g, it: launch(fns["gld_cg"], g, T, DYN_SMEM, l2buf.data_ptr(), out.data_ptr(),
                                                ctypes.c_int(it), ctypes.c_uint32(l2n - 1)),
                        unit="l2_byte", per_iter=lambda g: g * T * 8 * 16, working_set_bytes=16 * MiB),
        "hbm_read": dict(fn=lambda g, it: launch(fns["gld_cg"], g, T, DYN_SMEM, hbuf.data_ptr(), out.data_ptr(),
                                                 ctypes.c_int(it), ctypes.c_uint32(hbn - 1)),
                         unit="hbm_byte", per_iter=lambda g: g * T * 8 * 16, working_set_bytes=2048 * MiB),
        "hbm_copy": dict(fn=lambda g, it: launch(fns["gcopy"], g, T, DYN_SMEM, csrc.data_ptr(), cdst.data_ptr(),
                                                 ctypes.c_int(it), ctypes.c_uint32(cpn - 1)),
                         unit="hbm_byte", per_iter=lambda g: g * T * 4 * 16 * 2, working_set_bytes=2048 * MiB,
                         bytes_note="read + write bytes"),
    }
    gem = {}
    for dt in ["bf16", "fp16", "tf32", "int8"]:
        n = 8192
        if dt == "int8":
            a = torch.randint(-128, 127, (n, n), dtype=torch.int8, device="cuda")
            b = torch.randint(-128, 127, (n, n), dtype=torch.int8, device="cuda").t().contiguous().t()
            f = (lambda a, b: lambda: torch._int_mm(a, b))(a, b)
        else:
            tdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}[dt]
            a = torch.randn(n, n, device="cuda", dtype=tdt)
            b = torch.randn(n, n, device="cuda", dtype=tdt)
            f = (lambda a, b: lambda: torch.matmul(a, b))(a, b)
        keep += [a, b]
        gem[f"gemm_{dt}"] = dict(call=f, unit="mac", macs=n**3, n=n)
    torch.backends.cuda.matmul.allow_tf32 = True
    return benches, gem, keep


def ev_time(fn, reps=3):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / 1e3 / reps


def measure(step, gap_ns, sleep_fn):
    """Run step (+ optional 1-thread spin gap) back-to-back; NVML energy over the steady window."""
    def one():
        step()
        if gap_ns:
            sleep_fn(gap_ns)

    def phase(seconds):
        n, evs, t0 = 0, [], time.time()
        while time.time() - t0 < seconds:
            for _ in range(nb):
                one()
            n += nb
            ev = torch.cuda.Event()
            ev.record()
            evs.append(ev)
            if len(evs) > 2:
                evs.pop(0).synchronize()
        torch.cuda.synchronize()
        return n

    t_one = ev_time(one, 2)
    nb = max(1, int(BATCH_S / t_one))
    phase(WARM_S)
    SAMP.window()
    e0, t0 = energy_mj(), time.time()
    n = phase(MEAS_S)
    e1, t1 = energy_mj(), time.time()
    tel = SAMP.close()
    dt = t1 - t0
    return dict(steps=n, wall_s=dt, energy_j=(e1 - e0) / 1e3, power_w=(e1 - e0) / 1e3 / dt, **tel)


def run(out_path, reps):
    fns, sass = build()
    benches, gem, keep = setup(fns)
    sleep1 = lambda ns: launch(fns["spin"], 1, 1, 0, ctypes.c_uint64(int(ns)), ctypes.c_int(0))

    def smi(*a):
        r = subprocess.run(["nvidia-smi", *a], capture_output=True, text=True)
        return (r.stdout + r.stderr).strip()

    doc = {"schema": "kiln.energy/1", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "device": {"name": pynvml.nvmlDeviceGetName(H), "torch_name": torch.cuda.get_device_name(0),
                      "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(H) / 1e3,
                      "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                      "max_sm_mhz": pynvml.nvmlDeviceGetMaxClockInfo(H, pynvml.NVML_CLOCK_SM),
                      "max_mem_mhz": pynvml.nvmlDeviceGetMaxClockInfo(H, pynvml.NVML_CLOCK_MEM)},
           "software": {"torch": torch.__version__, "cuda": torch.version.cuda, "driver": pynvml.nvmlSystemGetDriverVersion()},
           "method": {"warm_s": WARM_S, "window_s": MEAS_S, "step_s": STEP_S, "grid_full": GRID_FULL, "block": BLOCK,
                      "dyn_smem_bytes_for_1_block_per_sm": DYN_SMEM, "energy_source": "nvmlDeviceGetTotalEnergyConsumption"},
           "records": []}
    assert "A100" in doc["device"]["name"], doc["device"]["name"]
    doc["clock_lock"] = {"lgc": smi("-lgc", f"{LGC},{LGC}"), "query_after": smi("--query-gpu=clocks.sm,clocks.mem,clocks.applications.graphics", "--format=csv")}
    doc["energy_counter"] = counter_granularity()
    doc["sass_ffma_count"] = sass.count("FFMA")
    doc["sass_excerpt"] = {k: [l.strip() for l in sass.split("\n") if k in l][:3] for k in ["LDS.128", "LDG.E.128", "STG.E.128"]}

    def save():
        json.dump(doc, open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)

    # calibrate iterations so each work kernel takes ~STEP_S at full grid
    iters = {}
    for name, b in benches.items():
        if b.get("kind") == "ns":
            continue
        it = 64
        while True:
            t = ev_time(lambda: b["fn"](GRID_FULL, it), 2)
            if t > 2e-3:
                break
            it *= 4
        iters[name] = max(1, int(it * STEP_S / t))
    gem_t = {k: ev_time(g["call"], 3) for k, g in gem.items()}
    doc["iters"], doc["gemm_call_s"] = iters, gem_t
    save()

    plan = []
    plan.append(("idle", "none", None))
    plan.append(("spin_1thread", "full", None))
    plan += [("spin_nap", "full", None), ("spin_busy", "full", None), ("spin_busy", "half", None)]
    for k in gem:
        plan += [(k, "full", None), (k, "duty50", None)] + ([(k, "duty25", None)] if k == "gemm_bf16" else [])
    for k in ["ffma", "smem_read", "l1_read", "l2_read", "hbm_read", "hbm_copy"]:
        plan += [(k, "full", None), (k, "half", None), (k, "duty50", None)]
    if ONLY:
        plan = [p for p in plan if p[0] in ONLY]
    doc["method"]["lgc_mhz"], doc["method"]["only"] = LGC, ONLY

    for rep in range(reps):
        for name, level, _ in plan:
            rec = {"rep": rep, "bench": name, "level": level}
            if name == "idle":
                torch.cuda.synchronize()
                time.sleep(2.0)
                SAMP.window()
                e0, t0 = energy_mj(), time.time()
                time.sleep(MEAS_S)
                e1, t1 = energy_mj(), time.time()
                rec.update(wall_s=t1 - t0, energy_j=(e1 - e0) / 1e3, power_w=(e1 - e0) / 1e3 / (t1 - t0), **SAMP.close())
                rec["work"] = {}
            elif name == "spin_1thread":
                rec.update(measure(lambda: sleep1(STEP_S * 1e9), 0, sleep1))
                rec["work"] = {}
            elif name.startswith("spin_"):
                g = GRID_FULL if level == "full" else GRID_FULL // 2
                rec.update(measure(lambda: benches[name]["fn"](g, STEP_S * 1e9), 0, sleep1))
                rec["grid"], rec["work"] = g, {}
            elif name in gem:
                gm = gem[name]
                reps_per_step = max(1, round(STEP_S / gem_t[name]))
                step = lambda: [gm["call"]() for _ in range(reps_per_step)]
                busy = gem_t[name] * reps_per_step
                gap = {"full": 0, "duty50": busy, "duty25": 3 * busy}[level] * 1e9
                m = measure(step, gap, sleep1)
                rec.update(m)
                rec["work"] = {"mac": m["steps"] * reps_per_step * gm["macs"]}
                rec["gemm_n"], rec["calls_per_step"] = gm["n"], reps_per_step
            else:
                b = benches[name]
                g = GRID_FULL if level in ("full", "duty50") else GRID_FULL // 2
                it = iters[name]
                busy_s = ev_time(lambda: b["fn"](g, it), 2)
                gap = busy_s * 1e9 if level == "duty50" else 0
                m = measure(lambda: b["fn"](g, it), gap, sleep1)
                rec.update(m)
                rec["grid"], rec["iters"], rec["kernel_s"] = g, it, busy_s
                rec["work"] = {b["unit"]: m["steps"] * b["per_iter"](g) * it}
                if "alu_per_unit" in b:
                    rec["lop3_per_unit"] = b["alu_per_unit"]
            rec["throughput"] = {u: c / rec["wall_s"] for u, c in rec["work"].items()}
            doc["records"].append(rec)
            save()
            thr = ", ".join(f"{u} {v:.3e}/s" for u, v in rec["throughput"].items())
            print(f"[{rep}] {name:12s} {level:6s} P={rec['power_w']:.1f} W  sm={rec['sm_mhz_mean']} "
                  f"T={rec['temp_c_end']} {thr}", flush=True)
    doc["clock_reset"] = smi("-rgc")
    doc["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    SAMP.stop_flag = True
    save()


def once(name, level):
    fns, _ = build()
    benches, gem, keep = setup(fns)
    if name in gem:
        gem[name]["call"]()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        gem[name]["call"]()
        torch.cuda.profiler.stop()
    else:
        g = GRID_FULL if level == "full" else GRID_FULL // 2
        benches[name]["fn"](g, int(sys.argv[4]) if len(sys.argv) > 4 else 64)
    torch.cuda.synchronize()


def check():
    fns, sass = build()
    benches, gem, keep = setup(fns)
    print("sass FFMA", sass.count("FFMA"), "LDS.128", sass.count("LDS.128"), "LDG.E.128", sass.count("LDG.E.128"))
    for name, b in benches.items():
        if b.get("kind") == "ns":
            t = ev_time(lambda: b["fn"](GRID_FULL, 2e6), 2)
            print(name, f"{t*1e3:.2f} ms for 2 ms spin")
            continue
        for it in [16, 256]:
            t = ev_time(lambda: b["fn"](GRID_FULL, it), 3)
            print(name, it, f"{t*1e3:.3f} ms", f"{b['per_iter'](GRID_FULL)*it/t:.3e} {b['unit']}/s")
    for k, g in gem.items():
        t = ev_time(g["call"], 5)
        print(k, f"{t*1e3:.3f} ms {2*g['macs']/t/1e12:.1f} TOPS")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        once(sys.argv[2], sys.argv[3])
