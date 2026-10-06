"""Dump the harness op suite (no Stream runs) to calibration/oplist.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.workloads import suite  # noqa: E402

rows, seen = [], set()
for s in ("all", "smoke"):
    for w in suite(s):
        o = w.op
        rows.append({"suite": s, "phase": w.phase, "op": o.name, "key": o.key, "m": o.m, "n": o.n, "k": o.k,
                     "batch": o.batch, "weight": o.weight, "count": o.count, "tokens": w.tokens, "flops": o.flops,
                     "min_bytes": o.bytes_a + o.bytes_b + o.bytes_out})
for m, n, k in [(1024, 1024, 1024), (4096, 4096, 4096), (16, 8192, 8192)]:
    rows.append({"suite": "make_gemm", "phase": "gemm", "op": f"gemm_{m}_{n}_{k}", "key": f"gemm_{m}_{n}_{k}",
                 "m": m, "n": n, "k": k, "batch": 1, "weight": True, "count": 1, "tokens": m,
                 "flops": 2 * m * n * k, "min_bytes": 2 * (m * k + k * n + m * n)})
(Path(__file__).parent / "oplist.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "ops,", len({r["key"] for r in rows}), "unique keys")
