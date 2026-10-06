"""MAP-Elites archive on fixed descriptor ranges (06 §6.3), per island plus a global grid, persisted in 05 §3.9's
archive layout (`archive.json`, `designs.arrow`, `generations.arrow`) next to the loop's own append-only records."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import kiln

# Descriptors the loop derives from kiln's standard features (fixed ranges, like 06 §6.3's).
DERIVED = {
    "flops_per_tile": {"unit": "FLOP/s per compute unit", "low": 1e9, "high": 1e15, "log": True,
                       "doc": "compute granularity: peak_flops_bf16 / compute_tiles"},
    "onchip_bytes_per_flop": {"unit": "B per FLOP/s", "low": 1e-8, "high": 1e-3, "log": True,
                              "doc": "on-chip capacity per unit of peak compute"},
}
WORKLOAD_DEPENDENT = ("bound_frac_mem", "score_rel_width", "energy_split")


@dataclass(frozen=True)
class Axis:
    name: str
    bins: int
    low: float
    high: float
    log: bool
    unit: str

    def position(self, x: float) -> float:
        if x is None or not math.isfinite(x):
            return float("nan")
        if self.log:
            t = (math.log(max(x, 1e-300)) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        else:
            t = (x - self.low) / (self.high - self.low)
        return min(1.0, max(0.0, t))

    def bin(self, x: float) -> int | None:
        t = self.position(x)
        if math.isnan(t):
            return None
        return min(self.bins - 1, int(t * self.bins))

    def edges(self) -> list[float]:
        out = []
        for i in range(self.bins + 1):
            t = i / self.bins
            out.append(math.exp(math.log(self.low) + t * (math.log(self.high) - math.log(self.low))) if self.log
                       else self.low + t * (self.high - self.low))
        return out


def _standard() -> dict:
    out = {}
    for d in kiln.descriptors():
        if d.get("dims", 1) == 3:
            for part in ("compute", "memory", "interconnect"):
                out[f"{d['name']}_{part}"] = {**d, "name": f"{d['name']}_{part}", "dims": 1}
        else:
            out[d["name"]] = d
    return out


def make_axes(specs: list[dict]) -> list[Axis]:
    std = _standard()
    axes = []
    for s in specs:
        name = s["name"]
        base = std.get(name) or DERIVED.get(name)
        if base is None and not {"low", "high"} <= s.keys():
            raise ValueError(f"descriptor {name!r} is neither standard ({sorted(std)}) nor derived "
                             f"({sorted(DERIVED)}); give low/high/log to define it")
        base = base or {}
        axes.append(Axis(name, int(s.get("bins", 8)), float(s.get("low", base.get("low"))),
                         float(s.get("high", base.get("high"))), bool(s.get("log", base.get("log", False))),
                         s.get("unit", base.get("unit", ""))))
    return axes


def feature_values(features: dict) -> dict:
    """Flattens kiln features (energy_split -> 3 keys) and adds the derived descriptors."""
    out = {}
    for k, v in (features or {}).items():
        if isinstance(v, list):
            for part, x in zip(("compute", "memory", "interconnect"), v):
                out[f"{k}_{part}"] = x
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    flops, tiles, onchip = out.get("peak_flops_bf16"), out.get("compute_tiles"), out.get("onchip_bytes")
    if flops and tiles:
        out["flops_per_tile"] = flops / tiles
    if flops and onchip:
        out["onchip_bytes_per_flop"] = onchip / flops
    return out


class Archive:
    def __init__(self, axes: list[Axis], n_islands: int):
        self.axes = axes
        self.islands: list[dict[tuple, str]] = [{} for _ in range(n_islands)]
        self.grid: dict[tuple, str] = {}
        self.records: dict[str, dict] = {}

    def cell(self, features: dict) -> tuple | None:
        vals = feature_values(features)
        out = []
        for a in self.axes:
            b = a.bin(vals.get(a.name, float("nan")))
            if b is None:
                return None
            out.append(b)
        return tuple(out)

    def normalized(self, features: dict) -> dict:
        vals = feature_values(features)
        return {a.name: a.position(vals[a.name]) for a in self.axes if a.name in vals}

    def fitness(self, rid: str) -> float:
        return self.records[rid]["fitness"]

    def elite_fitness(self, cell: tuple, island: int | None = None) -> float | None:
        g = self.grid if island is None else self.islands[island]
        rid = g.get(cell)
        return None if rid is None else self.fitness(rid)

    def would_place(self, rec: dict) -> str | None:
        """'new_cell' / 'improved' in the global grid, without inserting."""
        cell = rec.get("cell") and tuple(rec["cell"])
        if not cell or not rec.get("eligible"):
            return None
        cur = self.elite_fitness(cell)
        if cur is None:
            return "new_cell"
        return "improved" if rec["fitness"] > cur else None

    def insert(self, rec: dict) -> str | None:
        self.records[rec["id"]] = rec
        cell = rec.get("cell") and tuple(rec["cell"])
        if not cell or not rec.get("eligible"):
            return None
        event = self.would_place(rec)
        if event:
            self.grid[cell] = rec["id"]
        isl = self.islands[rec["island"]]
        cur = isl.get(cell)
        if cur is None or rec["fitness"] > self.fitness(cur):
            isl[cell] = rec["id"]
        return event

    def migrate(self, src: int, dst: int, k: int) -> list[str]:
        top = sorted(self.islands[src].items(), key=lambda kv: (-self.fitness(kv[1]), kv[1]))[:k]
        moved = []
        for cell, rid in top:
            cur = self.islands[dst].get(cell)
            if cur is None or self.fitness(rid) > self.fitness(cur):
                self.islands[dst][cell] = rid
                moved.append(rid)
        return moved

    def elites(self, island: int | None = None) -> list[dict]:
        g = self.grid if island is None else self.islands[island]
        # Ties break by id: dict order differs between a live run and a restored snapshot (sorted by cell).
        return sorted((self.records[r] for r in g.values()), key=lambda r: (-r["fitness"], r["id"]))

    def best(self) -> dict | None:
        e = self.elites()
        return e[0] if e else None

    def coverage(self) -> float:
        total = math.prod(a.bins for a in self.axes)
        return len(self.grid) / total

    def qd_score(self) -> float:
        return sum(max(0.0, self.fitness(r)) for r in self.grid.values())

    def snapshot(self) -> dict:
        enc = lambda g: [[list(c), r] for c, r in sorted(g.items())]  # noqa: E731
        return {"grid": enc(self.grid), "islands": [enc(g) for g in self.islands]}

    def restore(self, snap: dict) -> None:
        dec = lambda xs: {tuple(c): r for c, r in xs}  # noqa: E731
        self.grid = dec(snap["grid"])
        self.islands = [dec(g) for g in snap["islands"]]


def atomic_write(path: Path, data: bytes | str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


ELITE_STATUSES = ("elite", "displaced", "invalid", "failed")


def write_05_archive(out: Path, cfg: dict, archive: Archive, generations: list[dict], heldout: dict,
                     kiln_info: dict) -> list[str]:
    """Writes 05 §3.9's archive.json, designs.arrow and generations.arrow (rewritten whole at each checkpoint)."""
    notes = []
    axes = [{"name": a.name, "unit": a.unit, "bins": a.bins, "edges": a.edges(), "log": a.log} for a in archive.axes]
    meta = {"schema": "kiln.archive/1", "axes": axes, "fitness": {"name": cfg["fitness"]["kind"],
                                                                  "direction": "maximize"},
            "run_config": {k: v for k, v in cfg.items() if not k.startswith("_") and k != "workloads"}
            | {"workloads": {"train": cfg["workloads"]["train"], "screen": cfg["workloads"]["screen"]}},
            "kiln": kiln_info}
    atomic_write(out / "archive.json", json.dumps(meta, indent=1, default=str))
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        return ["pyarrow not installed: designs.arrow / generations.arrow not written"]
    elite_ids = set(archive.grid.values())
    rows = []
    for r in archive.records.values():
        if r.get("status") == "duplicate":
            continue
        if r["id"] in elite_ids:
            st = "elite"
        elif r["status"] == "ok":
            st = "displaced"
        elif r["status"] in ("invalid", "sandbox", "llm_error"):
            st = "invalid"
        else:
            st = "failed"
        h = heldout.get(r["id"]) or {}
        rows.append({
            "design_id": r["id"], "generation": r["gen"], "parent_ids": r.get("parents") or [],
            "mutation_summary": r.get("hypothesis") or "", "operator": r.get("operator") or "",
            "descriptor_values": [r.get("descriptors", {}).get(a.name) for a in archive.axes],
            "cell": list(r["cell"]) if r.get("cell") else None, "fitness": float(r.get("fitness") or 0.0),
            "fitness_components": json.dumps(r.get("per_workload") or {}), "status": st,
            "run_path": r.get("program_path") or "", "thumbnail_path": None, "design_hash": r.get("design_hash"),
            "wall_time": float(r.get("wall_s") or 0.0), "trust_level": "uncalibrated" if r["status"] == "ok" else None,
            "audit_status": (r.get("audit") or {}).get("status"), "heldout_score": h.get("heldout_score"),
            "calibration_set_hash": kiln_info.get("calibration_hash"), "kiln_git_hash": kiln_info.get("git_hash"),
            "stage_reached": r.get("stage"), "extrapolated": r.get("extrapolated") or [],
            "fitness_low": r.get("fitness_low"), "fitness_high": r.get("fitness_high"),
            "interval_method": r.get("interval_method"),
        })
    schema = pa.schema([
        ("design_id", pa.utf8()), ("generation", pa.uint32()), ("parent_ids", pa.list_(pa.utf8())),
        ("mutation_summary", pa.utf8()), ("operator", pa.utf8()), ("descriptor_values", pa.list_(pa.float64())),
        ("cell", pa.list_(pa.uint32())), ("fitness", pa.float64()), ("fitness_components", pa.utf8()),
        ("status", pa.dictionary(pa.int8(), pa.utf8())), ("run_path", pa.utf8()), ("thumbnail_path", pa.utf8()),
        ("design_hash", pa.utf8()), ("wall_time", pa.float64()), ("trust_level", pa.utf8()),
        ("audit_status", pa.utf8()), ("heldout_score", pa.float64()), ("calibration_set_hash", pa.utf8()),
        ("kiln_git_hash", pa.utf8()), ("stage_reached", pa.utf8()), ("extrapolated", pa.list_(pa.utf8())),
        ("fitness_low", pa.float64()), ("fitness_high", pa.float64()), ("interval_method", pa.utf8()),
    ])
    cols = {f.name: [row[f.name] for row in rows] for f in schema}
    cols["status"] = pa.array(cols["status"], pa.utf8()).dictionary_encode()
    table = pa.table({k: (v if k == "status" else pa.array(v, schema.field(k).type)) for k, v in cols.items()},
                     schema=schema)
    _write_ipc(out / "designs.arrow", table, ipc)
    gschema = pa.schema([("generation", pa.uint32()), ("best", pa.float64()), ("median", pa.float64()),
                         ("qd_score", pa.float64()), ("coverage", pa.float64()), ("evaluations", pa.uint64()),
                         ("invalid_count", pa.uint64())])
    gt = pa.table({f.name: pa.array([g[f.name] for g in generations], f.type) for f in gschema}, schema=gschema)
    _write_ipc(out / "generations.arrow", gt, ipc)
    return notes


def _write_ipc(path: Path, table, ipc) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with ipc.new_file(str(tmp), table.schema) as w:
        w.write_table(table)
    os.replace(tmp, path)
