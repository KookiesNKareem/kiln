"""Measured GPU op times vs cached Stream predictions (never runs Stream).
Usage: python calibration/compare.py measurements/a100_X.json [--design a100] [--predictions eval.json]
  [--mode graph_cold|flushed|unflushed|graph_unflushed]"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.evaluate import CACHE_DIR, _hw_hash, _sha, load_design, stream_version  # noqa: E402
from harness.workloads import ONNX_DIR, suite  # noqa: E402


def predictions(design):
    d = load_design(design)
    hw_dir = CACHE_DIR / "hw" / d.hash()
    if not hw_dir.exists():
        return d, {}
    hw_hash, ver, out = _hw_hash(hw_dir), stream_version(), {}
    for s in ("all", "smoke"):
        for w in suite(s):
            onnx = ONNX_DIR / f"{w.op.key}.onnx"
            if not onnx.exists():
                continue
            p = CACHE_DIR / "runs" / f"{_sha(f'{hw_hash}|{w.op.key}|{_sha(onnx.read_bytes())}|{ver}'.encode())}.json"
            if p.exists():
                r = json.loads(p.read_text())
                out[w.op.key] = r["cycles"] / d.clock_hz * 1e6 if r["status"] == "ok" else r["status"]
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurements")
    ap.add_argument("--design", default="a100")
    ap.add_argument("--mode", default="graph_cold")
    ap.add_argument("--predictions", help="evaluate() result JSON to use instead of the run cache")
    a = ap.parse_args()
    meas = json.loads(Path(a.measurements).read_text())
    d, pred = predictions(a.design)
    if a.predictions:
        res = json.loads(Path(a.predictions).read_text())
        res = res[0] if isinstance(res, list) else res
        pred = {o["key"]: o["time_s"] * 1e6 if o["status"] == "ok" else o["status"] for o in res["ops"]}
    ridge = d.peak_flops() / d.offchip_bytes_per_s
    print(f"device {meas['device']['torch_name']} vs design {d.name} (ridge {ridge:.0f} flop/B), mode={a.mode}, "
          f"{len(pred)} cached predictions")
    oplist = json.loads((ROOT / "calibration" / "oplist.json").read_text())
    errs = {"compute": [], "memory": []}
    phase_t = {}
    print(f"{'op':28s} {'bound':7s} {'meas_us':>9s} {'pred_us':>9s} {'pred/meas':>9s} {'meas_TF':>8s} {'meas_GB/s':>9s}")
    for r in meas["ops"]:
        if r["name"].endswith("_linear") or "error" in r:
            continue
        m = r[a.mode]["median_us"]
        bound = "compute" if r["intensity"] >= ridge else "memory"
        p = pred.get(r["name"])
        if isinstance(p, float):
            ratio = p / m
            errs[bound].append(ratio)
            ps, rs = f"{p:9.1f}", f"{ratio:9.3f}"
        else:
            ps, rs = f"{p or 'missing':>9s}", ""
        print(f"{r['name']:28s} {bound:7s} {m:9.1f} {ps} {rs:>9s} {r[a.mode]['tflops']:8.1f} {r[a.mode]['gbps']:9.0f}")
    for o in oplist:
        if o["suite"] != "all":
            continue
        r = next((x for x in meas["ops"] if x["name"] == o["key"] and "error" not in x), None)
        ph = phase_t.setdefault(o["phase"], {"meas": 0.0, "pred": 0.0, "missing": 0, "tokens": o["tokens"]})
        if r:
            ph["meas"] += o["count"] * r[a.mode]["median_us"]
        p = pred.get(o["key"])
        if isinstance(p, float):
            ph["pred"] += o["count"] * p
        else:
            ph["missing"] += 1
    print("\nphase       meas_ms   tok/s(meas)   pred_ms   pred/meas  missing")
    for k, v in phase_t.items():
        rr = v["pred"] / v["meas"] if v["pred"] and not v["missing"] else float("nan")
        print(f"{k:10s} {v['meas'] / 1e3:9.3f} {v['tokens'] / v['meas'] * 1e6:12.0f} {v['pred'] / 1e3:9.3f} {rr:10.3f} {v['missing']:4d}")
    print()
    for b, rs in errs.items():
        if not rs:
            print(f"{b}: no predicted ops")
            continue
        ape = [abs(x - 1) * 100 for x in rs]
        gm = math.exp(statistics.mean(math.log(max(e, 1e-9)) for e in ape))
        gmr = math.exp(statistics.mean(map(math.log, rs)))
        print(f"{b}: n={len(rs)} median|%err|={statistics.median(ape):.1f} geomean|%err|={gm:.1f} "
              f"geomean pred/meas={gmr:.3f}")


if __name__ == "__main__":
    main()
