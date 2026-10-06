"""Writes this fixture with pyarrow the way an evolution driver would (05 section 3.9): archive.json, then
designs.arrow / generations.arrow as appended IPC streams (one stream per append), int64 and large types
where pandas/polars would produce them. Read back by kiln-trace's archive reader in tests/archive_interop.rs.
Run: python3 gen.py (pyarrow >= 14)."""
import json
import os

import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))

meta = {
    "schema": "kiln.archive/1",
    "axes": [
        {"name": "die_mm2_total", "unit": "mm^2", "range": [10, 6864], "bins": 10},
        {"name": "power_w", "unit": "W", "log": True, "range": [5, 8000], "bins": 8},
    ],
    "fitness": {"name": "matched_envelope", "direction": "maximize"},
    "run_config": {"seed": 0, "driver": "pyarrow-fixture"},
}

designs_schema = pa.schema([
    ("design_id", pa.string()),
    ("generation", pa.int64()),
    ("parent_ids", pa.list_(pa.string())),
    ("mutation_summary", pa.string()),
    ("operator", pa.string()),
    ("descriptor_values", pa.list_(pa.float64())),
    ("cell", pa.list_(pa.int64())),
    ("fitness", pa.float64()),
    ("fitness_components", pa.string()),
    ("status", pa.dictionary(pa.int8(), pa.string())),
    ("run_path", pa.string()),
    ("thumbnail_path", pa.string()),
    ("design_hash", pa.string()),
    ("wall_time", pa.float64()),
    ("trust_level", pa.string()),
    ("audit_status", pa.string()),
    ("heldout_score", pa.float64()),
    ("calibration_set_hash", pa.string()),
    ("kiln_git_hash", pa.string()),
    ("stage_reached", pa.string()),
    ("extrapolated", pa.list_(pa.string())),
    ("fitness_low", pa.float64()),
    ("fitness_high", pa.float64()),
    ("interval_method", pa.string()),
    ("driver_note", pa.string()),
])


def row(i, g, parents, fit, status):
    return {
        "design_id": f"d{i}", "generation": g, "parent_ids": parents,
        "mutation_summary": f"change {i}", "operator": "crossover" if len(parents) > 1 else "mutate",
        "descriptor_values": [100.0 + 50 * i, 100.0 + 20 * i], "cell": [i % 10, i % 8], "fitness": fit,
        "fitness_components": json.dumps({"decode_b1": fit, "prefill_b1": fit * 1.1}), "status": status,
        "run_path": f"runs/d{i}.kiln", "thumbnail_path": None, "design_hash": f"hw1-{i:04d}",
        "wall_time": 0.25, "trust_level": "uncalibrated", "audit_status": None, "heldout_score": None,
        "calibration_set_hash": None, "kiln_git_hash": "deadbeef", "stage_reached": "S2",
        "extrapolated": None, "fitness_low": fit * 0.9, "fitness_high": fit * 1.1,
        "interval_method": "corners", "driver_note": "extra column, ignored by kiln",
    }


gens_schema = pa.schema([
    ("generation", pa.int64()), ("best", pa.float64()), ("median", pa.float64()), ("qd_score", pa.float64()),
    ("coverage", pa.float64()), ("evaluations", pa.int64()), ("invalid_count", pa.int64()),
])


def append(path, schema, rows):
    with open(path, "ab") as f, pa.ipc.new_stream(f, schema) as w:
        w.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))


for name in ("designs.arrow", "generations.arrow"):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        os.remove(p)
with open(os.path.join(HERE, "archive.json"), "w") as f:
    json.dump(meta, f, indent=2)
append(os.path.join(HERE, "designs.arrow"), designs_schema, [row(0, 0, [], 1.0, "elite")])
append(os.path.join(HERE, "designs.arrow"), designs_schema,
       [row(1, 1, ["d0"], 1.3, "elite"), row(2, 1, ["d0", "d1"], 0.0, "invalid")])
append(os.path.join(HERE, "generations.arrow"), gens_schema,
       [{"generation": 0, "best": 1.0, "median": 1.0, "qd_score": 1.0, "coverage": 0.0125,
         "evaluations": 1, "invalid_count": 0}])
append(os.path.join(HERE, "generations.arrow"), gens_schema,
       [{"generation": 1, "best": 1.3, "median": 0.65, "qd_score": 2.3, "coverage": 0.025,
         "evaluations": 2, "invalid_count": 1}])
