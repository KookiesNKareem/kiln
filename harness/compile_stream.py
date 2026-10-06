"""Design -> Stream hardware directory (system YAML + one YAML per unit type).

Layout follows what Stream's MILP placer can route: each on-chip memory instance owns a column
(col < nb_cols_to_use = 4), the compute instances attached to it sit in that column above it, and the
off-chip core sits alone below. Core ids run column by column, so the reference A100 compiles to the
same core numbering as hw/a100/a100.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from harness.design import OFFCHIP, PRECISION_BITS, ComputeUnit, Design, MemoryUnit

MAX_MEMORY_COLUMNS = 4
STREAM_PRECISIONS = {"bf16", "fp32"}


class UnsupportedDesign(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("design is valid but Stream cannot simulate it yet:\n" + "\n".join(f"  - {e}" for e in errors))


def unsupported(d: Design) -> list[str]:
    e = []
    n_mem = sum(m.count for m in d.memory)
    if n_mem > MAX_MEMORY_COLUMNS:
        e.append(f"{n_mem} on-chip memory instances; Stream routes through at most {MAX_MEMORY_COLUMNS} memory "
                 "columns. Merge memories or use fewer, larger instances")
    for c in d.compute:
        if c.kind == "matrix" and c.precision != "bf16":
            e.append(f"compute '{c.name}': precision {c.precision} is not simulated; the workload suite is bf16")
        if c.kind == "vector" and c.precision not in STREAM_PRECISIONS:
            e.append(f"compute '{c.name}': vector precision {c.precision} is not simulated; use bf16 or fp32")
        if c.accumulator != "fp32" and c.kind == "matrix":
            e.append(f"compute '{c.name}': accumulator {c.accumulator} is not simulated; use fp32")
    for x in d.links:
        if OFFCHIP in x.endpoints:
            e.append(f"link '{x.name}': Stream reaches off-chip memory only through on-chip memory; "
                     "list memory units in offchip.attach instead of linking to 'offchip'")
    return e


def _bits_per_cycle(gbps: float, clock_hz: float) -> int:
    return max(1, round(gbps * 1e9 * 8 / clock_hz))


def _port(name, kind, bw, alloc, bw_min=256):
    return {"name": name, "type": kind, "bandwidth_min": min(bw_min, bw), "bandwidth_max": bw, "allocation": alloc}


_READ_ALLOC = ["I1, tl", "I2, tl", "O, tl", "O, th"]
_WRITE_ALLOC = ["I1, fh", "I2, fh", "O, fh", "O, fl"]


def _mem(size_bits, ports, served, operands=("I1", "I2", "O"), r_cost=120, w_cost=130, extra=None):
    return {"size": int(size_bits), "r_cost": r_cost, "w_cost": w_cost, "area": 0, "latency": 1,
            **(extra or {}), "operands": list(operands), "ports": ports, "served_dimensions": served}


def _rw_buffer(size_bits, bw, **kw):
    return _mem(size_bits, [_port("r_port_1", "read", bw, _READ_ALLOC), _port("w_port_1", "write", bw, _WRITE_ALLOC)],
                ["D1", "D2"], **kw)


def compute_core(c: ComputeUnit, clock_hz: float) -> dict:
    buf = _rw_buffer(c.buffer_kib * 1024 * 8, _bits_per_cycle(c.buffer_gbps, clock_hz))
    if c.kind == "matrix":
        pbits, acc = PRECISION_BITS[c.precision], c.cols * PRECISION_BITS[c.accumulator]
        mems = {
            "wreg": _mem(2 * pbits, [_port("r_port_1", "read", pbits, ["I2, tl"], 16),
                                     _port("w_port_1", "write", pbits, ["I2, fh"], 16)], [], operands=["I2"],
                         r_cost=0.02, w_cost=0.02, extra={"auto_cost_extraction": False}),
            "accumulator": _mem(acc, [_port("r_port_1", "read", acc, ["O, tl"], acc),
                                      _port("r_port_2", "read", acc, ["O, th"], acc),
                                      _port("w_port_1", "write", acc, ["O, fh"], acc),
                                      _port("w_port_2", "write", acc, ["O, fl"], acc)], ["D2"], operands=["O"],
                                r_cost=0.01, w_cost=0.01),
            "operand_buffer": buf,
        }
        energy, area = 0.02, 1
    else:
        rbw = _bits_per_cycle(c.regfile_gbps, clock_hz)
        mems = {
            "vregs": _mem(c.regfile_kib * 1024 * 8,
                          [_port("rw_port_1", "read_write", rbw, _READ_ALLOC, 512),
                           _port("rw_port_2", "read_write", rbw, _WRITE_ALLOC, 512)],
                          ["D1", "D2"], r_cost=0.05, w_cost=0.05),
            "operand_buffer": buf,
        }
        energy, area = 0.06, 0.05
    return {
        "name": c.name, "type": "zigzag.compute", "memories": mems,
        "operational_array": {"unit_energy": energy, "unit_area": area, "dimensions": ["D1", "D2"],
                              "sizes": [c.rows, c.cols]},
        "operand_precision": {"input": c.precision, "accumulator": c.accumulator},
        "operator_types": c.op_types,
    }


def memory_core(m: MemoryUnit, clock_hz: float) -> dict:
    return {"name": m.name, "type": "zigzag.memory",
            "memories": {"vmem": _rw_buffer(m.size_mib * 2**20 * 8, _bits_per_cycle(m.bandwidth_gbps, clock_hz),
                                            r_cost=200, w_cost=220)},
            "operational_array": {"unit_energy": 0, "unit_area": 0, "dimensions": ["D1", "D2"], "sizes": [0, 0]}}


def offchip_core(d: Design) -> dict:
    bw = _bits_per_cycle(d.offchip.bandwidth_gbps, d.clock_hz)
    alloc = ["I1, fh", "I1, tl", "I2, fh", "I2, tl", "O, fh", "O, tl", "O, fl", "O, th"]
    return {"name": "offchip", "type": "zigzag.offchip",
            "memories": {"dram": _mem(d.offchip.capacity_gib * 2**30 * 8,
                                      [_port("rw_port_1", "read_write", bw, alloc)], ["D1", "D2"],
                                      r_cost=4000, w_cost=4000)},
            "operational_array": {"unit_energy": 0, "unit_area": 0, "dimensions": ["D1", "D2"], "sizes": [0, 0]}}


def layout(d: Design) -> tuple[list[tuple[str, int]], dict[int, tuple[int, int]], dict[tuple[str, int], int]]:
    """Core list as (unit name, instance), coordinates, and (unit, instance) -> core id."""
    cores, coords = [], {}
    col = 0
    for m in d.memory:
        for j in range(m.count):
            row = 0
            for c in d.compute:
                if c.attach != m.name:
                    continue
                per = c.count // m.count
                for i in range(j * per, (j + 1) * per):
                    coords[len(cores)] = (col, row)
                    cores.append((c.name, i))
                    row += 1
            coords[len(cores)] = (col, row)
            cores.append((m.name, j))
            col += 1
    coords[len(cores)] = (0, max(r for _, r in coords.values()) + 1)
    cores.append((OFFCHIP, 0))
    return cores, coords, {k: i for i, k in enumerate(cores)}


def system(d: Design) -> tuple[dict, dict[str, dict]]:
    cores, coords, cid = layout(d)
    comp = {c.name: c for c in d.compute}
    mem = {m.name: m for m in d.memory}
    files = {**{c.name: compute_core(c, d.clock_hz) for c in d.compute},
             **{m.name: memory_core(m, d.clock_hz) for m in d.memory}, OFFCHIP: offchip_core(d)}
    instances = {u: [k for k in cid if k[0] == u] for u in files}

    conn = []
    for x in d.links:
        bw = _bits_per_cycle(x.bandwidth_gbps, d.clock_hz)
        if x.scope == "global":
            ids = sorted(cid[k] for p in x.endpoints for k in instances[p])
            conn.append({"type": x.kind, "cores": ids, "bandwidth": bw})
            continue
        (m,) = [p for p in x.endpoints if p in mem]
        others = [p for p in x.endpoints if p != m]
        for j in range(mem[m].count):
            ids = [cid[(m, j)]]
            for p in others:
                per = comp[p].count // mem[m].count if p in comp else None
                ids += [cid[k] for k in instances[p] if per is None or k[1] // per == j]
            conn.append({"type": x.kind, "cores": sorted(ids), "bandwidth": bw})
    hbm = sorted(cid[k] for a in d.offchip.attach for k in instances[a]) + [cid[(OFFCHIP, 0)]]
    conn.append({"type": "bus", "cores": hbm, "bandwidth": _bits_per_cycle(d.offchip.bandwidth_gbps, d.clock_hz)})

    sysd = {
        "name": d.name,
        "cores": {i: f"./cores/{u}.yaml" for i, (u, _) in enumerate(cores)},
        "core_coordinates": {i: list(coords[i]) for i in range(len(cores))},
        "offchip_core_id": cid[(OFFCHIP, 0)],
        "unit_energy_cost": 0,
        "technology_node": d.tech_node,
        "core_connectivity": conn,
    }
    return sysd, files


class _FlowLists(yaml.SafeDumper):
    pass


_FlowLists.add_representer(list, lambda r, v: r.represent_sequence("tag:yaml.org,2002:seq", v,
                                                                    flow_style=all(not isinstance(x, dict) for x in v)))


def dump(obj) -> str:
    return yaml.dump(obj, Dumper=_FlowLists, sort_keys=False, width=200)


def compile_design(d: Design, out_dir: str | Path) -> Path:
    d.validate()
    errs = unsupported(d)
    if errs:
        raise UnsupportedDesign(errs)
    out = Path(out_dir)
    (out / "cores").mkdir(parents=True, exist_ok=True)
    sysd, files = system(d)
    for name, body in files.items():
        (out / "cores" / f"{name}.yaml").write_text(dump(body))
    path = out / f"{d.name}.yaml"
    path.write_text(dump(sysd))
    return path
