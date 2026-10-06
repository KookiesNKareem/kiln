"""M2 ground-truth summary: whole-step vs per-op sums, peaks, noise, clocks. Reads JSON only.

Usage: python analyze_m2.py [date]   (default 2026-10-05)
"""
import json
import statistics
import sys
from pathlib import Path

D = Path(__file__).parent / "measurements"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-05"
OPS = [o for o in json.load(open(Path(__file__).parent / "oplist.json")) if o["suite"] == "all"]
PHASES = ("decode_b1", "decode_b8", "decode_b32", "prefill_b1")
NL = 32


def load(dev, suite):
    p = D / f"{dev}_{DATE}_{suite}.json"
    return json.load(open(p)) if p.exists() else None


def canon(r, gpu):
    if gpu:
        for k in ("graph_cold", "graph_unflushed", "graph", "graph_x32"):
            if "median_s" in r["modes"].get(k, {}):
                return r["modes"][k]["median_s"]
        return None
    return r["modes"].get("loop", {}).get("median_s")  # 08 §F: canonical TPU per-op mode is loop


def per_op_sum(suite, phase, gpu, layers=NL):
    rec = {r["name"]: r for r in suite["records"]}
    mm = aux = 0.0
    missing = []
    for o in OPS:
        if o["phase"] != phase:
            continue
        name = o["key"] + ("_linear" if gpu and o["weight"] else "")
        t = canon(rec[name], gpu) if name in rec else None
        if t is None:
            missing.append(name)
            continue
        cnt = o["count"] if o["op"] == "lm_head" else o["count"] * layers / NL
        mm += cnt * t
    for r in suite["records"]:
        if r.get("kind") == "aux" and r.get("phase") == phase:
            t = canon(r, gpu)
            if t is None:
                missing.append(r["name"])
                continue
            fixed = r["op"] in ("embedding", "argmax", "rmsnorm_final")
            cnt = r["count"]
            if not fixed:
                if r["op"] == "rmsnorm" and not phase.startswith("prefill"):
                    cnt = 2 * layers + 1
                else:
                    cnt = r["count"] * layers / NL
            aux += cnt * t
    return mm, aux, missing


def step_table():
    print("\n## Whole step vs sum of isolated per-op times (ms)\n")
    print("| device | phase | attn | layers | step (seq) | sum matmul ops | sum aux ops | sum all | step/sum_all | step/sum_matmul | "
          "layer1 x L | 32-layer step (extrap.) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for dev, gpu in (("a100", True), ("tpuv6e", False), ("tpuv5e", False)):
        seq, suite = load(dev, "seq"), load(dev, "suite")
        if not seq:
            continue
        rec = {r["name"]: r for r in seq["records"]}
        for phase in PHASES:
            for sfx, attn in (("", "bmm" if gpu else "einsum"), ("_sdpa", "sdpa")):
                st = rec.get(f"seq/{phase}/step{sfx}")
                if not st or not st.get("modes"):
                    continue
                L = st.get("n_layers", NL)
                t = st["modes"]["graph"]["median_s"] if gpu else st["modes"]["loop"]["median_s"]
                l1 = rec.get(f"seq/{phase}/layer1{sfx}")
                l1t = None
                if l1 and l1.get("modes"):
                    l1t = l1["modes"]["graph_x32"]["median_s"] if gpu else l1["modes"]["loop"]["median_s"]
                s = per_op_sum(suite, phase, gpu, L) if suite else (None, None, [])
                ext = t + (NL - L) * l1t if (l1t and L < NL) else t
                fmt = lambda x: f"{x * 1e3:.3f}" if x else "-"
                line = [dev, phase, attn, str(L), fmt(t)]
                if s[0] is not None and sfx == "":
                    tot = s[0] + s[1]
                    line += [fmt(s[0]), fmt(s[1]), fmt(tot), f"{t / tot:.3f}", f"{t / s[0]:.3f}"]
                else:
                    line += ["-"] * 5
                line += [fmt(l1t * L) if l1t else "-", fmt(ext)]
                print("| " + " | ".join(line) + " |")
                if s[2] and sfx == "":
                    print(f"<!-- missing {dev} {phase}: {s[2]} -->")


def peaks():
    print("\n## Achievable peaks (canonical mode)\n")
    for dev, gpu in (("a100", True), ("tpuv6e", False), ("tpuv5e", False)):
        m = load(dev, "micro")
        if not m:
            continue
        rs = [r for r in m["records"] if canon(r, gpu)]
        best = lambda pred, key: max(((r.get(key) or 0), r["name"]) for r in rs if pred(r)) if any(pred(r) for r in rs) else None
        print(f"- {dev}: GEMM {best(lambda r: r.get('group') == 'gemm', 'tflops')} TFLOPS; "
              f"GEMV {best(lambda r: r.get('group') == 'gemv', 'gbps')} GB/s; "
              f"HBM read {best(lambda r: r.get('kind') == 'read_reduce' and r.get('group') == 'hbm', 'gbps')}, "
              f"write {best(lambda r: r.get('kind') == 'write' and r.get('group') == 'hbm', 'gbps')}, "
              f"copy {best(lambda r: r.get('kind') == 'copy' and r.get('group') == 'hbm', 'gbps')} GB/s")
        if gpu:
            l2 = [(r["modes"]["graph_unflushed"]["median_s"], r) for r in m["records"] if r.get("group") == "l2"]
            l2b = max((r["bytes"] / t / 1e9, r["name"]) for t, r in l2)
            print(f"  L2-resident (graph_unflushed) best {l2b[0]:.0f} GB/s ({l2b[1]})")
        print(f"  sanity: {json.dumps({k: v for k, v in m.get('sanity', {}).get('drift', {}).items()})}")


def noise():
    print("\n## Noise: median CV per mode per group\n")
    for dev in ("a100", "tpuv6e", "tpuv5e"):
        for suite in ("micro", "suite", "seq"):
            m = load(dev, suite)
            if not m:
                continue
            by = {}
            for r in m["records"]:
                g = r.get("group") or r.get("kind")
                for mode, v in r.get("modes", {}).items():
                    if isinstance(v, dict) and v.get("cv") is not None:
                        by.setdefault((g, mode), []).append(v["cv"])
            q = {}
            for (g, mode), cvs in sorted(by.items()):
                q.setdefault(g, []).append(f"{mode}={statistics.median(cvs):.4f}")
            n_rej = sum(r.get("quality") == "rejected" for r in m["records"])
            print(f"- {dev} {suite} ({len(m['records'])} records, {n_rej} rejected):")
            for g, s in q.items():
                print(f"    {g}: {', '.join(s)}")


def clocks():
    print("\n## A100 clocks per mode (timed region only)\n")
    m, s, q = load("a100", "micro"), load("a100", "suite"), load("a100", "seq")
    if not m:
        return
    for r in m["records"]:
        if r.get("kind") == "power_step":
            print(f"- {r['name']}: steady sm {r['steady']['sm_mhz']} MHz, {r['steady']['power_w']} W over {r['steady']['n']} samples")
    rows = [r for r in m["records"] + (s["records"] if s else []) if r.get("kind") in ("gemm", "linear") and (r.get("flops") or 0) >= 1e11]
    print("\n| record | TFLOPS | mode: sm MHz median/min @ W median (n) |")
    print("|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["flops"])[:14]:
        cells = []
        for mode in ("flushed", "unflushed", "graph_unflushed", "graph_cold"):
            c = r["modes"].get(mode, {}).get("clock")
            if c:
                cells.append(f"{mode}: {c['sm_mhz']:.0f}/{c['sm_mhz_min']:.0f} @ {c['power_w']:.0f} W ({c['n']}{' sparse' if c['sparse'] else ''})")
        print(f"| {r['name']} | {r.get('tflops', 0):.1f} | {'; '.join(cells)} |")
    if q:
        print()
        for r in q["records"]:
            for mode in ("graph", "eager"):
                c = r["modes"].get(mode, {}).get("clock")
                if c:
                    print(f"- {r['name']} {mode}: {c['sm_mhz']:.0f} MHz (min {c['sm_mhz_min']:.0f}) @ {c['power_w']:.0f} W, throttle bits {c['throttle_bits_union']}")


def fused_table():
    """Best-software step per phase: min over attention impls that passed the f32 reference check (seq_fused files)."""
    print("\n## Whole step per attention implementation (seq_fused; best_s, ms)\n")
    print("32-layer A = step + (32 - L) x layer1 (as before); B = step0 + 32 x (step - step0) / L (step0 = embedding + lm_head + argmax).\n")
    print("| device | phase | impl | layers | step | layer1 | 32-layer A | 32-layer B | A vs einsum_f32 (same run) | A vs unfused seq run |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for dev in ("tpuv6e", "tpuv5e"):
        fz, old = load(dev, "seq_fused"), load(dev, "seq")
        if not fz:
            continue
        rec = {r["name"]: r for r in fz["records"]}
        oldrec = {r["name"]: r for r in old["records"]} if old else {}
        sw = fz["software"]
        print(f"<!-- {dev}: jax {sw['jax']}, libtpu {sw['libtpu']} -->")

        def full(phase, impl, rs):
            sfx = "" if impl == "einsum_f32" else f"/{impl}"
            st, l1 = rs.get(f"seq/{phase}/step{sfx}"), rs.get(f"seq/{phase}/layer1{sfx}")
            if not st or st.get("best_s") is None or st.get("quality") != "ok":
                return None
            L = st["n_layers"]
            l1t = l1.get("best_s") if l1 else None
            if L < NL and l1t is None:
                return None
            return st["best_s"], l1t, L, st["best_s"] + (NL - L) * (l1t or 0)
        for phase in PHASES:
            impls = sorted({r["attn_impl"] for r in fz["records"] if r.get("phase") == phase and r.get("attn_impl")})
            rows = {i: full(phase, i, rec) for i in impls if fz["attn_checks"].get(f"{phase}/{i}", {}).get("ok")}
            rows = {i: v for i, v in rows.items() if v}
            if not rows:
                continue
            ref = rows.get("einsum_f32")
            o = full(phase, "einsum_f32", oldrec)
            best = min(rows, key=lambda i: rows[i][3])
            s0 = rec.get(f"seq/{phase}/step0", {}).get("best_s")
            for i, (t, l1t, L, ext) in sorted(rows.items(), key=lambda kv: kv[1][3]):
                f = lambda x: f"{x * 1e3:.3f}" if x else "-"
                extb = s0 + NL * (t - s0) / L if s0 else None
                print(f"| {dev} | {phase} | {i}{' **best**' if i == best else ''} | {L} | {f(t)} | {f(l1t)} | {f(ext)} | {f(extb)} | "
                      f"{ext / ref[3]:.3f} | {ext / o[3] if o else float('nan'):.3f} |")


if __name__ == "__main__":
    step_table()
    fused_table()
    peaks()
    noise()
    clocks()
