"""evaluate(design, suite) -> validated metrics. Each Stream call runs in its own subprocess."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from harness import physics
from harness.compile_stream import UnsupportedDesign, compile_design, layout
from harness.design import Design, DesignError
from harness.workloads import ROOT, Workload, resident_bytes, suite as get_suite

STREAM_DIR = Path(os.environ.get("HARNESS_STREAM_SRC", ROOT / "third_party" / "stream"))  # override to A/B Stream versions
CACHE_DIR = ROOT / ".cache" / "harness"
MAX_WORKERS = 4  # 16 GB Mac: each Stream process peaks around 1-2 GB
FLOOR_TOL = 0.01
DESIGNS_DIR = Path(__file__).parent / "designs"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@lru_cache(maxsize=1)
def stream_version() -> str:
    """Commit plus a hash of uncommitted changes: other agents patch Stream in place."""
    git = ["git", "-C", str(STREAM_DIR)]
    head = subprocess.run([*git, "rev-parse", "--show-toplevel", "--short=12", "HEAD"], capture_output=True, text=True)
    top, _, head = head.stdout.strip().partition("\n")
    if Path(top).resolve() != STREAM_DIR.resolve():
        return f"src:{_sha(str(STREAM_DIR.resolve()).encode())}"
    diff = subprocess.run([*git, "diff", "HEAD"], capture_output=True).stdout
    return f"{head}+{_sha(diff)}" if diff else head


def load_design(ref: str | Path | Design) -> Design:
    if isinstance(ref, Design):
        return ref.validate()
    p = Path(ref)
    if not p.exists() and (DESIGNS_DIR / f"{ref}.json").exists():
        p = DESIGNS_DIR / f"{ref}.json"
    return Design.load(p)


def run_stream(hw_yaml: Path, onnx: Path, timeout: float) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": os.pathsep.join([str(STREAM_DIR), str(ROOT)])}
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "harness.stream_worker", str(hw_yaml), str(onnx)],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Stream exceeded {timeout:.0f}s", "sim_wall_s": time.time() - t0}
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("HARNESS_RESULT ")), None)
    if line is None:
        return {"status": "crash", "error": f"exit {p.returncode}: {p.stderr[-2000:]}", "sim_wall_s": time.time() - t0}
    return {**json.loads(line.removeprefix("HARNESS_RESULT ")), "sim_wall_s": time.time() - t0}


def _cached_run(hw_yaml: Path, hw_hash: str, w: Workload, timeout: float, use_cache: bool) -> dict:
    onnx = w.onnx()
    key = _sha(f"{hw_hash}|{w.op.key}|{_sha(onnx.read_bytes())}|{stream_version()}".encode())
    path = CACHE_DIR / "runs" / f"{key}.json"
    if use_cache and path.exists():
        return {**json.loads(path.read_text()), "cached": True}
    r = run_stream(hw_yaml, onnx, timeout)
    if r["status"] in ("ok", "infeasible"):  # deterministic outcomes only; timeouts/crashes may be load
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(r))
        tmp.replace(path)
    return {**r, "cached": False}


def check_floors(cycles: float, w: Workload, d: Design) -> dict:
    """Physical lower bounds on cycles; a simulator result below either is wrong, not fast."""
    compute = w.flops / (d.peak_flops() / d.clock_hz)
    hbm = w.compulsory_bytes(d.onchip_bytes) / (d.offchip_bytes_per_s / d.clock_hz)
    ok = math.isfinite(cycles) and cycles > 0
    return {
        "floor_compute_cycles": compute,
        "floor_hbm_cycles": hbm,
        "compute_floor": ok and cycles >= compute * (1 - FLOOR_TOL),
        "hbm_floor": ok and cycles >= hbm * (1 - FLOOR_TOL),
        "roofline_frac": max(compute, hbm) / cycles if ok else None,
    }


def explain_error(err: str, cores: list[tuple[str, int]]) -> str:
    """Name the design unit behind Stream's core ids, and hint at the usual cause of capacity failures."""
    err = re.sub(r"Core (\d+)", lambda m: f"Core {m[1]} ({cores[int(m[1])][0]}#{cores[int(m[1])][1]})"
                 if int(m[1]) < len(cores) else m[0], err)
    if "on-chip memory of Core" in err:
        err += (" | hint: Stream tiles each op to fit the compute units' buffer_kib (50% fill, <= 1024 tiles); "
                "when a memory unit overflows, the compute buffers are usually too small for this op")
    return err[:1000]


def _hw_hash(hw_dir: Path) -> str:
    return _sha(b"".join(p.read_bytes() for p in sorted(hw_dir.rglob("*.yaml"))))


def evaluate(design: str | Path | Design, suite: str = "decode", workers: int = MAX_WORKERS,
             timeout: float = 120.0, use_cache: bool = True) -> dict:
    t0 = time.time()
    try:
        d = load_design(design)
    except DesignError as e:
        return {"valid": False, "stage": "validate", "errors": e.errors, "wall_s": time.time() - t0}
    base = {"design": d.name, "design_hash": d.hash(), "suite": suite, "stream_version": stream_version(),
            "clock_mhz": d.clock_mhz, "peak_tflops": d.peak_flops() / 1e12, "offchip_gbps": d.offchip.bandwidth_gbps,
            "onchip_mib": d.onchip_bytes / 2**20, "physics": physics.estimate(d)}
    hw_dir = CACHE_DIR / "hw" / d.hash()
    try:
        hw_yaml = compile_design(d, hw_dir)
    except UnsupportedDesign as e:
        return {**base, "valid": False, "stage": "compile", "errors": e.errors, "wall_s": time.time() - t0}
    hw_hash = _hw_hash(hw_dir)

    wls = get_suite(suite)
    unique = {w.op.key: w for w in wls}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futs = {k: pool.submit(_cached_run, hw_yaml, hw_hash, w, timeout, use_cache) for k, w in unique.items()}
        runs = {k: f.result() for k, f in futs.items()}

    ops, violations, failures = [], [], []
    phases = defaultdict(lambda: {"time_s": 0.0, "flops": 0.0, "floor_s": 0.0, "failed_ops": [], "floor_violations": []})
    cores = layout(d)[0]
    for w in wls:
        r = runs[w.op.key]
        row = {"phase": w.phase, "op": w.op.name, "key": w.op.key, "count": w.op.count, "flops": w.flops,
               "status": r["status"], "cached": r["cached"], "sim_wall_s": r.get("sim_wall_s")}
        ph = phases[w.phase]
        ph["tokens"] = w.tokens
        if r["status"] != "ok":
            row["error"] = explain_error(r.get("error", ""), cores)
            failures.append(f"{w.name}: {r['status']}")
            ph["failed_ops"].append(w.op.name)
        else:
            cyc = r["cycles"]
            chk = check_floors(cyc, w, d)
            secs = cyc / d.clock_hz
            row |= {"cycles": cyc, "time_s": secs, "tflops": w.flops / secs / 1e12, **chk}
            for name in ("compute_floor", "hbm_floor"):
                if not chk[name]:
                    fl = chk[f"floor_{name.split('_')[0]}_cycles"]
                    ph["floor_violations"].append(w.op.name)
                    violations.append(f"{w.name} [{w.op.key}]: {name} violated, {cyc:.0f} cycles < floor {fl:.0f} "
                                      f"({cyc / fl:.2f}x)")
            ph["time_s"] += w.op.count * secs
            ph["flops"] += w.op.count * w.flops
            ph["floor_s"] += w.op.count * max(chk["floor_compute_cycles"], chk["floor_hbm_cycles"]) / d.clock_hz
        ops.append(row)

    for name, ph in phases.items():
        ph["complete"] = not ph["failed_ops"]
        ph["trusted"] = ph["complete"] and not ph["floor_violations"]
        ok = ph["complete"] and ph["time_s"] > 0
        ph["tokens_per_s"] = ph["tokens"] / ph["time_s"] if ok else None
        ph["tflops"] = ph["flops"] / ph["time_s"] / 1e12 if ok else None
        ph["roofline_frac"] = ph["floor_s"] / ph["time_s"] if ok else None
        need = resident_bytes(name) if name != "smoke" else 0
        ph["fits_offchip"] = need <= d.offchip.capacity_gib * 2**30
        if not ph["fits_offchip"]:
            violations.append(f"{name}: weights+KV {need / 2**30:.1f} GiB exceed off-chip {d.offchip.capacity_gib} GiB")

    return {**base, "valid": not violations and not failures, "violations": violations, "failures": failures,
            "phases": dict(phases), "ops": ops, "n_stream_calls": sum(not r["cached"] for r in runs.values()),
            "wall_s": time.time() - t0}


def score(result: dict, baseline: dict) -> float:
    """Geometric-mean tokens/s speedup over the baseline across phases; 0 for anything not trustworthy."""
    if not result.get("valid") or result["physics"]["violations"]:
        return 0.0
    ratios = [result["phases"][p]["tokens_per_s"] / b["tokens_per_s"]
              for p, b in baseline["phases"].items() if b.get("tokens_per_s")]
    return math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else 0.0


def evaluate_program(program_path: str | Path, suite: str = "decode", baseline: str = "a100", **kw) -> dict:
    """Entry point for the evolution loop: the program defines build() -> Design (or a dict).
    Returns the full result plus `combined_score` (speedup vs baseline, 0 if invalid) and MAP-Elites
    feature candidates."""
    try:
        spec = importlib.util.spec_from_file_location("candidate", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        obj = mod.build()
        design = Design.from_dict(obj) if isinstance(obj, dict) else obj
        if not isinstance(design, Design):
            raise TypeError(f"build() must return a Design or dict, got {type(design).__name__}")
    except DesignError as e:
        return {"valid": False, "stage": "validate", "errors": e.errors, "combined_score": 0.0}
    except Exception as e:
        return {"valid": False, "stage": "load", "errors": [f"{type(e).__name__}: {e}"], "combined_score": 0.0}
    res = evaluate(design, suite, **kw)
    if "phases" not in res:
        return {**res, "combined_score": 0.0}
    res["combined_score"] = score(res, evaluate(baseline, suite, **kw))
    res["features"] = {"die_mm2": res["physics"]["die_mm2"], "power_w": res["physics"]["power_w"],
                       "onchip_mib": res["onchip_mib"], "peak_tflops": res["peak_tflops"],
                       "flops_per_byte": res["peak_tflops"] * 1e12 / (res["offchip_gbps"] * 1e9)}
    return res
