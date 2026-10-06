"""Accelerator design DSL: compute units, on-chip memories, off-chip memory and the links between them.

Units are human scale (MHz, KiB/MiB/GiB, GB/s) so an LLM can write designs directly. A unit with
count > 1 is that many identical instances. Every compute unit stages operands through the memory
unit it is `attach`ed to; off-chip memory talks to the memory units in `offchip.attach`. Bandwidths
are per instance (per port for memories, per bus for links). The DSL is deliberately more general
than what Stream simulates today; compile_stream.unsupported() lists what it rejects.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

PRECISION_BITS = {"int4": 4, "int8": 8, "fp8": 8, "bf16": 16, "fp16": 16, "fp32": 32}
MATRIX_OPS = ("MatMul", "Gemm", "Conv", "Einsum")
VECTOR_OPS = ("Add", "Sub", "Mul", "Div", "Exp", "Pow", "Silu", "Sigmoid", "Gelu", "Tanh", "Relu", "Softmax",
              "ReduceMax", "ReduceSum", "ReduceMean", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool")
TECH_NODES = ("n16", "n7", "n5", "n3")
OFFCHIP = "offchip"


class DesignError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid design:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ComputeUnit:
    name: str
    kind: str  # "matrix" (systolic MAC array, rows x cols MACs) or "vector" (rows x cols lanes)
    rows: int
    cols: int
    count: int = 1
    attach: str = ""  # memory unit whose instances this unit stages through
    precision: str = "bf16"  # multiplier input precision
    accumulator: str = "fp32"
    buffer_kib: float = 2048.0  # array-local operand staging
    buffer_gbps: float = 4000.0
    regfile_kib: float = 0.0  # vector units only
    regfile_gbps: float = 0.0
    ops: list[str] | None = None  # defaults by kind

    @property
    def op_types(self) -> list[str]:
        return list(self.ops) if self.ops is not None else list(MATRIX_OPS if self.kind == "matrix" else VECTOR_OPS)

    @property
    def macs(self) -> int:
        return self.rows * self.cols * self.count


@dataclass
class MemoryUnit:
    name: str
    size_mib: float
    bandwidth_gbps: float  # per read and per write port
    count: int = 1


@dataclass
class OffChip:
    capacity_gib: float
    bandwidth_gbps: float
    attach: list[str]  # memory units with a direct path to off-chip memory
    kind: str = "HBM2e"
    stacks: int = 1


@dataclass
class Link:
    name: str
    endpoints: list[str]  # unit names; every instance of each unit joins
    bandwidth_gbps: float
    kind: str = "bus"  # "bus" (shared by all endpoints) or "link" (point to point, exactly 2 instances)
    scope: str = "global"  # "global": one bus; "per_memory": one bus per instance of the single memory endpoint,
    # joining it and the compute instances attached to it


@dataclass
class Design:
    name: str
    clock_mhz: float
    tech_node: str
    compute: list[ComputeUnit]
    memory: list[MemoryUnit]
    offchip: OffChip
    links: list[Link] = field(default_factory=list)
    notes: str = ""

    @property
    def clock_hz(self) -> float:
        return self.clock_mhz * 1e6

    def peak_flops(self, op: str = "MatMul") -> float:
        return sum(2 * c.macs for c in self.compute if op in c.op_types) * self.clock_hz

    @property
    def onchip_bytes(self) -> float:
        mem = sum(m.size_mib * 2**20 * m.count for m in self.memory)
        return mem + sum((c.buffer_kib + c.regfile_kib) * 1024 * c.count for c in self.compute)

    @property
    def offchip_bytes_per_s(self) -> float:
        return self.offchip.bandwidth_gbps * 1e9

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> Design:
        errors: list[str] = []
        top = _build(cls, d, "design", errors, nested={"compute", "memory", "offchip", "links"})
        if top is not None:
            top.compute = [_build(ComputeUnit, c, f"compute[{i}]", errors) for i, c in enumerate(d.get("compute", []))]
            top.memory = [_build(MemoryUnit, m, f"memory[{i}]", errors) for i, m in enumerate(d.get("memory", []))]
            top.offchip = _build(OffChip, d.get("offchip", {}), "offchip", errors)
            top.links = [_build(Link, x, f"links[{i}]", errors) for i, x in enumerate(d.get("links", []))]
        if errors:
            raise DesignError(errors)
        top.validate()
        return top

    @classmethod
    def from_json(cls, text: str) -> Design:
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise DesignError([f"not valid JSON: {e}"]) from None
        return cls.from_dict(d)

    @classmethod
    def load(cls, path: str | Path) -> Design:
        return cls.from_json(Path(path).read_text())

    def validate(self) -> Design:
        errors = _validate(self)
        if errors:
            raise DesignError(errors)
        return self


def _build(cls, d, where: str, errors: list[str], nested: set[str] = frozenset()):
    if not isinstance(d, dict):
        errors.append(f"{where}: expected an object, got {type(d).__name__}")
        return None
    names = {f.name for f in fields(cls)}
    required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
    unknown = set(d) - names
    if unknown:
        errors.append(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(names)}")
    missing = required - set(d)
    if missing:
        errors.append(f"{where}: missing required field(s) {sorted(missing)}")
    if unknown or missing:
        return None
    return cls(**{k: (v if k not in nested else None) for k, v in d.items()})


def _num(errors, where, value, lo, hi, integer=False):
    if isinstance(value, bool) or not isinstance(value, int | float) or (integer and not isinstance(value, int)):
        errors.append(f"{where} must be {'an integer' if integer else 'a number'}, got {value!r}")
    elif not (lo <= value <= hi) or not math.isfinite(value):
        errors.append(f"{where} = {value} is outside the plausible range [{lo}, {hi}]")


def _validate(d: Design) -> list[str]:
    e: list[str] = []
    if not d.name or not isinstance(d.name, str):
        e.append("design.name must be a non-empty string")
    _num(e, "clock_mhz", d.clock_mhz, 100, 5000)
    if d.tech_node not in TECH_NODES:
        e.append(f"tech_node {d.tech_node!r} unknown; use one of {list(TECH_NODES)}")
    if not d.compute:
        e.append("design needs at least one compute unit")
    if not any(c.kind == "matrix" for c in d.compute):
        e.append("design needs at least one compute unit with kind='matrix' (the suite is all matmuls)")
    if not d.memory:
        e.append("design needs at least one on-chip memory unit")

    names = [u.name for u in (*d.compute, *d.memory)] + [x.name for x in d.links]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        e.append(f"names must be unique across compute, memory and links; duplicated: {dupes}")
    if OFFCHIP in names:
        e.append(f"'{OFFCHIP}' is reserved for off-chip memory; rename that unit")
    mem = {m.name: m for m in d.memory}
    comp = {c.name: c for c in d.compute}

    for c in d.compute:
        w = f"compute '{c.name}'"
        if c.kind not in ("matrix", "vector"):
            e.append(f"{w}: kind must be 'matrix' or 'vector', got {c.kind!r}")
        _num(e, f"{w}.rows", c.rows, 1, 1024, integer=True)
        _num(e, f"{w}.cols", c.cols, 1, 1024, integer=True)
        _num(e, f"{w}.count", c.count, 1, 4096, integer=True)
        _num(e, f"{w}.buffer_kib", c.buffer_kib, 1, 1 << 20)
        _num(e, f"{w}.buffer_gbps", c.buffer_gbps, 1, 1e6)
        for p in ("precision", "accumulator"):
            if getattr(c, p) not in PRECISION_BITS:
                e.append(f"{w}.{p} {getattr(c, p)!r} unknown; use one of {list(PRECISION_BITS)}")
        if c.kind == "vector":
            _num(e, f"{w}.regfile_kib", c.regfile_kib, 1, 1 << 16)
            _num(e, f"{w}.regfile_gbps", c.regfile_gbps, 1, 1e6)
        elif c.regfile_kib or c.regfile_gbps:
            e.append(f"{w}: regfile_kib/regfile_gbps apply to vector units only; matrix units hold operands in buffer_kib")
        allowed = set(MATRIX_OPS if c.kind == "matrix" else VECTOR_OPS)
        bad = sorted(set(c.op_types) - allowed)
        if bad:
            e.append(f"{w}: ops {bad} are not {c.kind} ops; allowed: {sorted(allowed)}")
        if c.attach not in mem:
            e.append(f"{w}: attach={c.attach!r} must name a memory unit, one of {sorted(mem)}")
        elif isinstance(c.count, int) and c.count % mem[c.attach].count:
            e.append(f"{w}: count {c.count} must be a multiple of attached memory '{c.attach}' count "
                     f"{mem[c.attach].count} so instances spread evenly")

    for m in d.memory:
        w = f"memory '{m.name}'"
        _num(e, f"{w}.size_mib", m.size_mib, 0.0625, 1 << 16)
        _num(e, f"{w}.bandwidth_gbps", m.bandwidth_gbps, 1, 1e6)
        _num(e, f"{w}.count", m.count, 1, 256, integer=True)

    o = d.offchip
    _num(e, "offchip.capacity_gib", o.capacity_gib, 0.25, 4096)
    _num(e, "offchip.bandwidth_gbps", o.bandwidth_gbps, 1, 1e5)
    _num(e, "offchip.stacks", o.stacks, 0, 64, integer=True)
    if not o.attach:
        e.append("offchip.attach must list at least one memory unit (the path from off-chip memory onto the chip)")
    for a in o.attach:
        if a not in mem:
            e.append(f"offchip.attach entry {a!r} must name a memory unit, one of {sorted(mem)}")

    for x in d.links:
        w = f"link '{x.name}'"
        _num(e, f"{w}.bandwidth_gbps", x.bandwidth_gbps, 1, 1e6)
        if x.kind not in ("bus", "link"):
            e.append(f"{w}: kind must be 'bus' or 'link', got {x.kind!r}")
        if x.scope not in ("global", "per_memory"):
            e.append(f"{w}: scope must be 'global' or 'per_memory', got {x.scope!r}")
        unknown = [p for p in x.endpoints if p not in mem and p not in comp and p != OFFCHIP]
        if unknown:
            e.append(f"{w}: endpoints {unknown} are not units; known: {sorted([*mem, *comp, OFFCHIP])}")
            continue
        if len(x.endpoints) < 2 and (x.scope == "global" or not any(p in comp for p in x.endpoints)):
            e.append(f"{w}: needs at least two endpoints")
        if x.scope == "per_memory":
            mems = [p for p in x.endpoints if p in mem]
            if len(mems) != 1:
                e.append(f"{w}: scope='per_memory' needs exactly one memory endpoint, got {mems}")
            else:
                stray = [p for p in x.endpoints if p in comp and comp[p].attach != mems[0]]
                if stray:
                    e.append(f"{w}: compute endpoints {stray} are not attached to '{mems[0]}'")
        elif x.kind == "link":
            n = sum((mem[p].count if p in mem else comp[p].count if p in comp else 1) for p in x.endpoints)
            if n != 2:
                e.append(f"{w}: kind='link' joins exactly 2 instances, got {n}; use kind='bus' for more")

    for c in d.compute:
        if c.attach in mem and not any(c.name in x.endpoints and c.attach in x.endpoints for x in d.links):
            e.append(f"compute '{c.name}': no link joins it to its attached memory '{c.attach}'; add a bus")
    return e
