import argparse
import json
import sys

from harness.evaluate import MAX_WORKERS, evaluate


def _fmt(x, spec):
    return "-" if x is None else format(x, spec)


def print_result(r: dict, verbose: bool = False) -> None:
    if "phases" not in r:
        print(f"{r.get('design', '?')}: REJECTED at {r['stage']}")
        for e in r["errors"]:
            print(f"  - {e}")
        return
    p = r["physics"]
    print(f"{r['design']} [{r['design_hash']}] suite={r['suite']} stream={r['stream_version']} "
          f"peak={r['peak_tflops']:.0f} TFLOPS, HBM {r['offchip_gbps']:.0f} GB/s, on-chip {r['onchip_mib']:.0f} MiB, "
          f"~{p['die_mm2']:.0f} mm2 ~{p['power_w']:.0f} W (placeholder model)")
    print(f"  {'phase':<12}{'time/step':>12}{'tok/s':>12}{'TFLOPS':>9}{'roofline':>10}  status")
    for name, ph in r["phases"].items():
        status = (f"FAILED {ph['failed_ops']}" if not ph["complete"]
                  else f"BELOW PHYSICAL FLOOR {ph['floor_violations']}" if ph["floor_violations"] else "ok")
        print(f"  {name:<12}{_fmt(ph['time_s'] * 1e3 if ph['complete'] else None, '>9.3f')} ms"
              f"{_fmt(ph['tokens_per_s'], '>12.1f')}{_fmt(ph['tflops'], '>9.1f')}{_fmt(ph['roofline_frac'], '>10.2f')}"
              f"  {status}")
    if verbose:
        print(f"  {'op':<24}{'key':<26}{'x':>4}{'us':>10}{'TFLOPS':>8}{'floorC us':>10}{'floorM us':>10}  checks")
        clk = r["clock_mhz"]
        for o in r["ops"]:
            if o["status"] != "ok":
                print(f"  {o['phase'] + '/' + o['op']:<24}{o['key']:<26}{o['count']:>4}  {o['status']}: {o['error'][:80]}")
                continue
            flags = ",".join(n for n in ("compute_floor", "hbm_floor") if not o[n]) or "ok"
            print(f"  {o['phase'] + '/' + o['op']:<24}{o['key']:<26}{o['count']:>4}{o['time_s'] * 1e6:>10.1f}"
                  f"{o['tflops']:>8.1f}{o['floor_compute_cycles'] / clk:>10.1f}{o['floor_hbm_cycles'] / clk:>10.1f}"
                  f"  {flags}")
    print(f"  valid={r['valid']}  stream calls={r['n_stream_calls']}  wall={r['wall_s']:.1f}s")
    for v in r["violations"] + r["failures"] + [f"physics: {v}" for v in p["violations"]]:
        print(f"  ! {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "compare"):
        s = sub.add_parser(name)
        s.add_argument("designs", nargs="+" if name == "compare" else 1, help="JSON path or reference name")
        s.add_argument("--suite", default="decode")
        s.add_argument("--workers", type=int, default=MAX_WORKERS)
        s.add_argument("--timeout", type=float, default=120.0)
        s.add_argument("--no-cache", action="store_true")
        s.add_argument("-v", "--verbose", action="store_true", help="per-op table")
        s.add_argument("--json", help="write full results here")
    a = ap.parse_args(argv)
    results = [evaluate(d, a.suite, a.workers, a.timeout, not a.no_cache) for d in a.designs]
    for r in results:
        print_result(r, a.verbose)
    if a.cmd == "compare" and all("phases" in r for r in results):
        ref = results[0]
        print(f"\n  tokens/s relative to {ref['design']}:")
        print("  " + f"{'phase':<12}" + "".join(f"{r['design']:>12}" for r in results))
        for ph in ref["phases"]:
            vals = [r["phases"].get(ph, {}).get("tokens_per_s") for r in results]
            base = vals[0]
            print("  " + f"{ph:<12}" + "".join(_fmt(v / base if v and base else None, ">12.2f") for v in vals))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0 if all(r.get("valid") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
