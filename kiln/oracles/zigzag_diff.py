"""L3 differential oracle (06 §2.3): kiln-cost vs ZigZag (zigzag-dse==3.8.5, the Stream fork's pin) on a corpus
of (GEMM shape x single-core architecture x fixed mapping) expressible in both.

For every case this script builds the architecture once in a neutral form, emits it both as a ZigZag accelerator
and as a kiln `UnitTemplate` (same energies, bandwidths, served dimensions, port allocation), runs ZigZag's
`CostModelEvaluation` directly on the fixed spatial + temporal mapping, and writes everything (inputs and ZigZag's
numbers) to `crates/kiln-cost/tests/data/zigzag_corpus.json`. The Rust test `zigzag_corpus` replays the corpus
through `kiln_cost::evaluate` with `zigzag_compat` and checks access counts exactly, latency and energy within 1%.

Usage (repo .venv has zigzag-dse 3.8.5):
    ../.venv/bin/python oracles/zigzag_diff.py            # regenerate corpus, then run the Rust comparison
    ../.venv/bin/python oracles/zigzag_diff.py --no-run   # regenerate only
"""

import argparse
import importlib.metadata
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

from zigzag.cost_model.cost_model import CostModelEvaluation  # noqa: E402
from zigzag.datatypes import LayerDim, LayerOperand  # noqa: E402
from zigzag.hardware.architecture.memory_port import DataDirection  # noqa: E402
from zigzag.mapping.data_movement import DataMoveAttr  # noqa: E402
from zigzag.mapping.temporal_mapping import TemporalMapping, TemporalMappingType  # noqa: E402
from zigzag.parser.accelerator_factory import AcceleratorFactory  # noqa: E402
from zigzag.parser.accelerator_validator import AcceleratorValidator  # noqa: E402
from zigzag.parser.mapping_validator import MappingValidator  # noqa: E402
from zigzag.parser.workload_factory import WorkloadFactory  # noqa: E402
from zigzag.parser.workload_validator import WorkloadValidator  # noqa: E402
from zigzag.stages.mapping.spatial_mapping_conversion import SpatialMappingConversionStage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "crates/kiln-cost/tests/data/zigzag_corpus.json"
PIN = "3.8.5"

# kiln stream order is the nest operand order: A = I (activations), B = W (weights), O.
OPS = ["I", "W", "O"]
MEM_OP = {"I": "I1", "W": "I2", "O": "O"}
ROLE = {"I": "a", "W": "b", "O": "o"}
DIR = {"tl": "to_low", "fh": "from_high", "th": "to_high", "fl": "from_low"}
ZZ_DIR = {
    DataDirection.RD_OUT_TO_LOW: "to_low",
    DataDirection.WR_IN_BY_HIGH: "from_high",
    DataDirection.RD_OUT_TO_HIGH: "to_high",
    DataDirection.WR_IN_BY_LOW: "from_low",
}
DIMS = ["M", "N", "K"]  # kiln dim indices 0, 1, 2
AXES = {"I": [[0], [2]], "W": [[2], [1]], "O": [[0], [1]]}


def port(name, kind, bw, alloc):
    return {"name": name, "type": kind, "bandwidth_min": bw, "bandwidth_max": bw, "allocation": alloc}


# ----------------------------------------------------------------------------------------------- architectures
# Each memory: size (bits per instance), r/w cost (pJ per access of `bw` bits), operands, ports, served dims.
def arch_tpu_like(d1, d2):
    return {
        "name": f"tpu_like_{d1}x{d2}",
        "dims": [d1, d2],
        "unit_energy": 0.04,
        "memories": {
            "rf_w": dict(size=256, r_cost=0.02, w_cost=0.024, operands=["I2"], served=[],
                         ports=[port("r_port_1", "read", 16, ["I2, tl"]), port("w_port_1", "write", 16, ["I2, fh"])]),
            "rf_o": dict(size=64, r_cost=0.03, w_cost=0.035, operands=["O"], served=[],
                         ports=[port("r_port_1", "read", 32, ["O, tl"]), port("r_port_2", "read", 32, ["O, th"]),
                                port("w_port_1", "write", 32, ["O, fh"]), port("w_port_2", "write", 32, ["O, fl"])]),
            "sram": dict(size=8 * 1024 * 1024 * 8, r_cost=40.0, w_cost=44.0, operands=["I1", "O"], served=["D1", "D2"],
                         ports=[port("r_port_1", "read", 1024, ["I1, tl", "O, tl", "O, th"]),
                                port("w_port_1", "write", 1024, ["I1, fh", "O, fh", "O, fl"])]),
            "dram": dict(size=10_000_000_000, r_cost=700.0, w_cost=750.0, operands=["I1", "I2", "O"], served=["D1", "D2"],
                         ports=[port("rw_port_1", "read_write", 512, ["I1, fh", "I1, tl", "I2, fh", "I2, tl",
                                                               "O, fh", "O, tl", "O, fl", "O, th"])]),
        },
    }


def arch_shared_l1(d1, d2):
    return {
        "name": f"shared_l1_{d1}x{d2}",
        "dims": [d1, d2],
        "unit_energy": 0.05,
        "memories": {
            "reg_i": dict(size=64, r_cost=0.01, w_cost=0.012, operands=["I1"], served=["D2"],
                          ports=[port("r_port_1", "read", 16, ["I1, tl"]), port("w_port_1", "write", 16, ["I1, fh"])]),
            "reg_w": dict(size=64, r_cost=0.01, w_cost=0.012, operands=["I2"], served=[],
                          ports=[port("r_port_1", "read", 16, ["I2, tl"]), port("w_port_1", "write", 16, ["I2, fh"])]),
            "reg_o": dict(size=128, r_cost=0.02, w_cost=0.025, operands=["O"], served=["D1"],
                          ports=[port("r_port_1", "read", 32, ["O, tl"]), port("r_port_2", "read", 32, ["O, th"]),
                                 port("w_port_1", "write", 32, ["O, fh"]), port("w_port_2", "write", 32, ["O, fl"])]),
            "l1": dict(size=512 * 1024 * 8, r_cost=8.0, w_cost=9.0, operands=["I1", "I2", "O"], served=["D1", "D2"],
                       ports=[port("r_port_1", "read", 512, ["I1, tl", "I2, tl", "O, tl", "O, th"]),
                              port("w_port_1", "write", 512, ["I1, fh", "I2, fh", "O, fh", "O, fl"])]),
            "dram": dict(size=10_000_000_000, r_cost=500.0, w_cost=520.0, operands=["I1", "I2", "O"], served=["D1", "D2"],
                         ports=[port("rw_port_1", "read_write", 256, ["I1, fh", "I1, tl", "I2, fh", "I2, tl",
                                                               "O, fh", "O, tl", "O, fl", "O, th"])]),
        },
    }


def arch_two_level(d1, d2):
    return {
        "name": f"two_level_{d1}x{d2}",
        "dims": [d1, d2],
        "unit_energy": 0.06,
        "memories": {
            "l1": dict(size=256 * 1024 * 8, r_cost=4.0, w_cost=4.4, operands=["I1", "I2", "O"], served=["D1", "D2"],
                       ports=[port("r_port_1", "read", 256, ["I1, tl", "I2, tl", "O, tl", "O, th"]),
                              port("w_port_1", "write", 256, ["I1, fh", "I2, fh", "O, fh", "O, fl"])]),
            "dram": dict(size=10_000_000_000, r_cost=300.0, w_cost=310.0, operands=["I1", "I2", "O"], served=["D1", "D2"],
                         ports=[port("rw_port_1", "read_write", 128, ["I1, fh", "I1, tl", "I2, fh", "I2, tl",
                                                               "O, fh", "O, tl", "O, fl", "O, th"])]),
        },
    }


def zz_accelerator(arch):
    data = {
        "name": arch["name"],
        "memories": {
            name: {
                "size": m["size"], "r_cost": m["r_cost"], "w_cost": m["w_cost"], "area": 0, "latency": 1,
                "operands": m["operands"], "ports": m["ports"], "served_dimensions": m["served"],
            }
            for name, m in arch["memories"].items()
        },
        "operational_array": {"unit_energy": arch["unit_energy"], "unit_area": 1, "dimensions": ["D1", "D2"],
                              "sizes": arch["dims"]},
    }
    v = AcceleratorValidator(data)
    data = v.normalized_data
    assert v.validate(), v.validator.errors if hasattr(v, "validator") else "invalid accelerator"
    return AcceleratorFactory(data).create()


def kiln_unit(arch, prec):
    names = list(arch["memories"])
    levels = []
    for name in names:
        m = arch["memories"][name]
        inst_axes = [i for i, d in enumerate(["D1", "D2"]) if d not in m["served"]]
        n_inst = math.prod(arch["dims"][i] for i in inst_axes)
        ports = []
        for p in m["ports"]:
            serves = []
            for a in p["allocation"]:
                mop, d = [x.strip() for x in a.split(",")]
                op = next(k for k, v in MEM_OP.items() if v == mop)
                serves.append([ROLE[op], DIR[d]])
            ports.append({"dir": p["type"], "bytes_per_cycle": p["bandwidth_max"] / 8, "serves": serves})
        bw = m["ports"][0]["bandwidth_max"]
        assert all(p["bandwidth_max"] == bw for p in m["ports"]), "kiln energy/B needs one port width per memory"
        levels.append({
            "name": name, "mem": None, "capacity_bytes": m["size"] * n_inst // 8, "ports": ports, "double_buffer": True,
            "instance_axes": inst_axes, "e_read_j_per_b": m["r_cost"] * 1e-12 / (bw / 8),
            "e_write_j_per_b": m["w_cost"] * 1e-12 / (bw / 8), "latency_cycles": 0, "external": name == "dram",
        })
    chains = [{"role": ROLE[op], "levels": [i for i, n in enumerate(names) if MEM_OP[op] in arch["memories"][n]["operands"]]}
              for op in OPS]
    a, acc = prec["in"], prec["acc"]
    return {
        "name": arch["name"], "clock_hz": 1e9,
        "axes": [{"name": "D1", "size": arch["dims"][0], "allowed": []}, {"name": "D2", "size": arch["dims"][1], "allowed": []}],
        "modes": [{"a": a, "b": a, "acc": acc, "out": None, "macs_per_cycle": float(math.prod(arch["dims"])),
                   "e_mac_j": arch["unit_energy"] * 1e-12, "mx_native": False}],
        "levels": levels, "chains": chains,
        "pipeline": {"fill": 0, "drain": 0, "issue_overhead": 0},
        "psum_precision": None, "fused_down_conversion": True, "e_mac_idle_ratio": 0.0, "e_vector_op_j": 0.0,
        "energy_source": "supplied",
    }


def kiln_nest(shape, prec):
    return {
        "kind": "contraction",
        "dims": [{"name": d.lower(), "size": shape[d], "kind": "reduction" if d == "K" else "parallel"} for d in DIMS],
        "operands": [
            {"tensor": op.lower(), "role": ROLE[op], "axes": [{"terms": [[d, 1] for d in ax], "div": 1} for ax in AXES[op]],
             "dtype": prec["out"] if op == "O" else prec["in"], "is_output": op == "O", "source": None, "sink": None,
             "block_axis": None}
            for op in OPS
        ],
        "macs_per_point": 1, "vector_ops_per_point": 0, "points": None,
    }


# ----------------------------------------------------------------------------------------------------- mappings
def factorize(n):
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1
    if n > 1:
        f.append(n)
    return f


def footprint(op, ext):
    return math.prod(math.prod(ext[d] for d in ax) for ax in AXES[op])


def greedy_alloc(arch, loops, spatial, prec):
    """LOMA-like bottom-up fill per operand: each non-top level keeps loops while the tile (incl. the spatial
    unrolling it serves) fits half its per-instance size (double buffering headroom)."""
    names = list(arch["memories"])
    bits = {"I": BITS[prec["in"]], "W": BITS[prec["in"]], "O": BITS[prec["acc"]]}
    out = {}
    for op in OPS:
        lvls = [n for n in names if MEM_OP[op] in arch["memories"][n]["operands"]]
        b, lo, served = [], 0, set()
        for j, name in enumerate(lvls):
            served |= set(arch["memories"][name]["served"])
            if j == len(lvls) - 1:
                b.append(len(loops))
                break
            sp = [1, 1, 1]
            for ax, (d, u) in enumerate(spatial):
                if ["D1", "D2"][ax] in served:
                    sp[d] *= u
            k = lo
            while k < len(loops):
                ext = list(sp)
                for (d, f) in loops[: k + 1]:
                    ext[d] *= f
                if footprint(op, ext) * bits[op] * 2 > arch["memories"][name]["size"]:
                    break
                k += 1
            b.append(k)
            lo = k
        out[op] = b
    return out


BITS = {"bf16": 16, "fp32": 32, "int8": 8, "int32": 32}


def make_mapping(arch, shape, spatial, order, prec):
    """spatial: [(dim, unroll) on D1, (dim, unroll) on D2]; order: dims innermost first, each dim's temporal
    factor split into its prime factors and interleaved in that order."""
    t = list(shape[d] for d in DIMS)
    for d, u in spatial:
        assert t[d] % u == 0, "corpus uses divisible unrollings (ZigZag uses fractional ones otherwise)"
        t[d] //= u
    pf = {d: factorize(t[d]) for d in range(3)}
    loops = []
    while any(pf.values()):
        for d in order:
            if pf[d]:
                loops.append((d, pf[d].pop(0)))
    alloc = greedy_alloc(arch, loops, spatial, prec)
    return loops, alloc


# ------------------------------------------------------------------------------------------------------ ZigZag
def zz_layer(shape, prec):
    wl = [{
        "id": 0, "operator_type": "Gemm", "equation": "O[m][n]+=I[m][k]*W[k][n]", "loop_dims": DIMS,
        "loop_sizes": [shape[d] for d in DIMS],
        "operand_precision": {"O": BITS[prec["acc"]], "O_final": BITS[prec["out"]], "W": BITS[prec["in"]], "I": BITS[prec["in"]]},
        "operand_source": {"I": 0, "W": 0},
    }]
    v = WorkloadValidator(wl)
    data = v.normalized_data
    assert v.validate()
    return data


def run_zigzag(arch, shape, spatial, loops, alloc, prec):
    acc = zz_accelerator(arch)
    mapping = [{"name": "default", "spatial_mapping": {f"D{ax + 1}": [f"{DIMS[d]}, {u}"] for ax, (d, u) in enumerate(spatial)},
                "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"}}]
    mv = MappingValidator(mapping)
    mapping = mv.normalized_data
    assert mv.validate()
    layer = next(iter(WorkloadFactory(zz_layer(shape, prec), mapping).create().topological_sort()))
    layer.spatial_mapping.initialize_oa_dims(acc.operational_array.dimension_sizes)
    conv = SpatialMappingConversionStage([lambda *a, **k: None], accelerator=acc, layer=layer)
    sm, smi = conv.convert_user_spatial_mapping(layer.spatial_mapping)
    tm = {}
    for op in OPS:
        b, lo, lv = alloc[op], 0, []
        for hi in b:
            lv.append([(LayerDim(DIMS[d]), f) for d, f in loops[lo:hi]])
            lo = hi
        tm[LayerOperand(op)] = lv
    tmap = TemporalMapping(tm, layer, TemporalMappingType.UNEVEN)
    t0 = time.perf_counter()
    c = CostModelEvaluation(accelerator=acc, layer=layer, spatial_mapping=sm, spatial_mapping_int=smi, temporal_mapping=tmap)
    dt = time.perf_counter() - t0
    counts = {}
    for op in OPS:
        counts[op] = []
        for d in c.mapping.unit_mem_data_movement[LayerOperand(op)]:
            mv4 = d.get_attribute(DataMoveAttr.DATA_ELEM_MOVE_COUNT)
            counts[op].append({ZZ_DIR[x]: int(mv4.get(x)) for x in DataDirection})
    energy_levels = {op: [float(x) * 1e-12 for x in c.mem_energy_breakdown[LayerOperand(op)]] for op in OPS}
    return {
        "counts": counts,
        "mac_energy_j": c.mac_energy * 1e-12,
        "mem_energy_j": c.mem_energy * 1e-12,
        "energy_j": c.energy_total * 1e-12,
        "energy_levels_j": energy_levels,
        "ideal_temporal_cycle": float(c.ideal_temporal_cycle),
        "stall": float(c.stall_slack_comb),
        "onloading": float(c.data_onloading_cycle),
        "offloading": float(c.data_offloading_cycle),
        "latency_cycles": float(c.latency_total2),
        "zigzag_eval_s": dt,
    }


def kiln_mapping(shape, spatial, loops, alloc):
    return {"classes": [{
        "sizes": [shape[d] for d in DIMS], "count": 1,
        "spatial": {"axes": [[[d, u]] for d, u in spatial]},
        "temporal": {"loops": [{"dim": d, "factor": f} for d, f in loops], "alloc": [alloc[op] for op in OPS],
                     "double_buffer": [[False] * len(alloc[op]) for op in OPS]},
    }]}


# ------------------------------------------------------------------------------------------------------ corpus
PRECS = {
    "bf16": {"in": "bf16", "acc": "fp32", "out": "bf16"},
    "int8": {"in": "int8", "acc": "int32", "out": "int32"},
}
# Llama-3-8B-like GEMM tiles (M = tokens) small enough for ZigZag to evaluate in milliseconds.
SHAPES = [
    {"M": 64, "N": 128, "K": 256},
    {"M": 16, "N": 512, "K": 512},
    {"M": 128, "N": 256, "K": 128},
    {"M": 8, "N": 1024, "K": 256},
    {"M": 256, "N": 64, "K": 1024},
]
ARCHS = [arch_tpu_like(16, 16), arch_tpu_like(32, 8), arch_shared_l1(16, 16), arch_two_level(8, 16)]
# (D1 dim, D2 dim): reduction on rows / output columns, as in a weight-stationary array, plus an M x N array.
SPATIALS = [(2, 1), (0, 1)]
ORDERS = [[0, 2, 1], [2, 0, 1], [1, 2, 0]]


def build_cases():
    cases = []
    for arch in ARCHS:
        for si, shape in enumerate(SHAPES):
            for (d1, d2) in SPATIALS:
                sp = [(d1, min(arch["dims"][0], shape[DIMS[d1]])), (d2, min(arch["dims"][1], shape[DIMS[d2]]))]
                for oi, order in enumerate(ORDERS):
                    for pname, prec in PRECS.items():
                        if pname == "int8" and (si + oi) % 2:
                            continue
                        name = f"{arch['name']}/{'x'.join(str(shape[d]) for d in DIMS)}/{DIMS[d1]}{DIMS[d2]}/o{oi}/{pname}"
                        cases.append((name, arch, shape, sp, order, prec))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()
    ver = importlib.metadata.version("zigzag-dse")
    if ver != PIN:
        sys.exit(f"zigzag-dse {ver} installed, corpus is pinned to {PIN}")
    out = []
    t0 = time.perf_counter()
    for name, arch, shape, sp, order, prec in build_cases():
        loops, alloc = make_mapping(arch, shape, sp, order, prec)
        zz = run_zigzag(arch, shape, sp, loops, alloc, prec)
        out.append({
            "name": name, "unit": kiln_unit(arch, prec), "nest": kiln_nest(shape, prec),
            "mapping": kiln_mapping(shape, sp, loops, alloc), "zigzag": zz,
        })
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text(json.dumps({"oracle": "zigzag-dse", "version": ver, "cases": out}, separators=(",", ":")) + "\n")
    print(f"wrote {len(out)} cases to {CORPUS.relative_to(ROOT)} in {time.perf_counter() - t0:.1f}s")
    if not args.no_run:
        r = subprocess.run(["cargo", "test", "-q", "--release", "-p", "kiln-cost", "--test", "zigzag_corpus", "--",
                            "--nocapture"], cwd=ROOT)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
