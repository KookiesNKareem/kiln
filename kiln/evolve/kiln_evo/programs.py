"""Design programs: Python files whose `build()` returns a kiln.hw/1.0 design.

Conventions the loop relies on (all optional except `build()`):
- `PARAMS = {...}`: a literal dict of knobs `build()` reads; the non-LLM parametric operator edits it.
- `PARAM_SPACE = {name: {"min", "max", "step" | "log2" | "scale"} | {"choices": [...]}}`: literal bounds for
  the parametric operator.

LLM responses carry a one-line `HYPOTHESIS:` and either SEARCH/REPLACE blocks (OpenEvolve's diff format) or one
full program in a ```python fence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import pprint
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import json5

DIFF_RE = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
HYPOTHESIS_RE = re.compile(r"^\s*\**HYPOTHESIS\**\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


class ProgramError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.hint = code, message, hint

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": "error", "message": self.message, "hint": self.hint,
                "path": "program", "section": "kiln_evo"}


def program_hash(code: str) -> str:
    return "prg-" + hashlib.sha256(code.encode()).hexdigest()[:16]


def design_hash(design) -> str:
    text = design if isinstance(design, str) else json.dumps(design, sort_keys=True, separators=(",", ":"))
    return "dsn-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _assign(code: str, name: str) -> ast.Assign | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    return None


def read_literal(code: str, name: str):
    node = _assign(code, name)
    if node is None:
        return None
    try:
        return ast.literal_eval(node.value)
    except (ValueError, SyntaxError, TypeError):
        return None


def _render_dict(name: str, d: dict) -> str:
    body = "".join(f"    {json.dumps(k)}: {_py(v)},\n" for k, v in d.items())
    return f"{name} = {{\n{body}}}"


def _py(v) -> str:
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, (int, str)):
        return json.dumps(v) if isinstance(v, str) else repr(v)
    return repr(v)


def write_literal(code: str, name: str, value: dict) -> str:
    node = _assign(code, name)
    if node is None:
        raise ProgramError("E-EVO-PARAMS", f"program has no literal {name} assignment",
                           f"define {name} = {{...}} at module level")
    lines = code.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    return "".join(lines[:start]) + _render_dict(name, value) + "\n" + "".join(lines[end:])


def _magnitude(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([0-9.]+)", str(v))
    return float(m.group(1)) if m else math.inf


def materialize_seed(code: str, designs_dir: str | Path) -> str:
    """Fills a seed template (a program with a REFERENCE design name) from `designs_dir/REFERENCE.json5`: BASE becomes
    the reference design (renamed to NAME, without family and claims), PARAMS its knob values (the template's
    reference_params(BASE)), and each ENVELOPE_KNOBS range in PARAM_SPACE is capped at the reference value. The seed
    therefore always reproduces the reference design and cannot drift from it. Programs without REFERENCE are
    returned unchanged. Runs in the trusted parent: the sandbox cannot read the designs dir."""
    ref_name = read_literal(code, "REFERENCE")
    if ref_name is None:
        return code
    ref = json5.loads((Path(designs_dir) / f"{ref_name}.json5").read_text())
    ref.pop("family", None)
    ref.get("meta", {}).pop("claims", None)
    ref["name"] = read_literal(code, "NAME") or ref["name"]
    ns: dict = {}
    exec(compile(code, f"seed:{ref_name}", "exec"), ns)
    params = ns["reference_params"](ref)
    space = read_literal(code, "PARAM_SPACE") or {}
    for k in read_literal(code, "ENVELOPE_KNOBS") or ():
        spec, cap = dict(space.get(k, {})), params[k]
        if "choices" in spec:
            spec["choices"] = [c for c in spec["choices"] if _magnitude(c) <= _magnitude(cap)] or [cap]
        else:
            spec["max"] = min(spec.get("max", cap), cap)
        space[k] = spec
    code = write_literal(code, "PARAMS", params)
    code = write_literal(code, "PARAM_SPACE", space)
    node = _assign(code, "BASE")
    lines = code.splitlines(keepends=True)
    base = "BASE = " + pprint.pformat(ref, width=118, sort_dicts=False, indent=1)
    return "".join(lines[:node.lineno - 1]) + base + "\n" + "".join(lines[node.end_lineno:])


def with_params(code: str, overrides: dict) -> str:
    params = read_literal(code, "PARAMS")
    if not isinstance(params, dict):
        raise ProgramError("E-EVO-PARAMS", "program has no literal PARAMS dict", "define PARAMS = {...}")
    unknown = sorted(set(overrides) - set(params))
    if unknown:
        raise ProgramError("E-EVO-PARAMS", f"unknown PARAMS keys {unknown}", f"known keys: {sorted(params)}")
    return write_literal(code, "PARAMS", {**params, **overrides})


def _finite(x) -> bool:
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:
        return False


def _check_spec(k: str, value, spec) -> None:
    """Rejects mutation metadata `_step` cannot use: it is program-controlled."""
    def bad(why: str):
        raise ProgramError("E-EVO-PARAMS", f"PARAM_SPACE[{k!r}] = {spec!r} for PARAMS value {value!r}: {why}",
                           "use {'min', 'max', 'step' | 'log2' | 'scale'} with finite numbers or a non-empty "
                           "'choices' list")

    if not isinstance(spec, dict):
        bad("not a dict")
    if "choices" in spec:
        if not isinstance(spec["choices"], (list, tuple)) or not spec["choices"]:
            bad("choices must be a non-empty list")
        return
    if isinstance(value, bool):
        return
    if not _finite(value):
        bad("the value is not a finite number")
    lo, hi = spec.get("min", -math.inf), spec.get("max", math.inf)
    for name, x in (("min", lo), ("max", hi)):
        if name in spec and not _finite(x):
            bad(f"{name} is not a finite number")
    if lo > hi:
        bad("min exceeds max")
    if "scale" in spec and not (_finite(spec["scale"]) and spec["scale"] > 0):
        bad("scale must be a finite number > 0")
    if "step" in spec and not _finite(spec["step"]):
        bad("step must be a finite number")


def _step(value, spec: dict, rng: random.Random):
    if "choices" in spec:
        options = [c for c in spec["choices"] if c != value] or list(spec["choices"])
        return rng.choice(options)
    lo, hi = spec.get("min", -math.inf), spec.get("max", math.inf)
    up = rng.random() < 0.5
    if isinstance(value, bool):
        return not value
    if spec.get("log2"):
        new = value * 2 if up else value / 2
    elif "scale" in spec:
        new = value * spec["scale"] if up else value / spec["scale"]
    elif "step" in spec:
        new = value + spec["step"] if up else value - spec["step"]
    else:
        new = value * rng.choice([0.5, 0.75, 1.25, 1.5, 2.0])
    new = min(hi, max(lo, new))
    if isinstance(value, int):
        new = int(round(new))
    elif isinstance(new, float):
        new = round(new, 4)
    return new


def parametric_mutation(code: str, rng: random.Random, n_changes: int = 1) -> tuple[str, str]:
    """Moves `n_changes` PARAMS entries one step inside PARAM_SPACE. Returns (new code, summary)."""
    params = read_literal(code, "PARAMS")
    if not isinstance(params, dict) or not params:
        raise ProgramError("E-EVO-PARAMS", "program has no literal PARAMS dict to mutate",
                           "parametric mutation needs PARAMS = {...}")
    space = read_literal(code, "PARAM_SPACE") or {}
    if not isinstance(space, dict):
        raise ProgramError("E-EVO-PARAMS", "PARAM_SPACE is not a literal dict",
                           "define PARAM_SPACE = {name: {...}} at module level")
    keys = [k for k in params if k in space] or [k for k, v in params.items()
                                                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not keys:
        raise ProgramError("E-EVO-PARAMS", "no mutable PARAMS entries", "add numeric PARAMS or a PARAM_SPACE")
    changes = {}
    for k in rng.sample(keys, min(n_changes, len(keys))):
        spec = space.get(k, {})
        _check_spec(k, params[k], spec)
        for _ in range(4):
            try:
                new = _step(params[k], spec, rng)
            except (ArithmeticError, TypeError, ValueError) as e:
                raise ProgramError("E-EVO-PARAMS", f"cannot step PARAMS[{k!r}] = {params[k]!r}: {e}",
                                   "keep PARAMS values and PARAM_SPACE bounds finite and moderate") from None
            if not isinstance(new, bool) and isinstance(new, (int, float)) and not _finite(new):
                raise ProgramError("E-EVO-PARAMS", f"stepping PARAMS[{k!r}] = {params[k]!r} overflows",
                                   "keep PARAMS values and PARAM_SPACE bounds finite and moderate")
            if new != params[k]:
                changes[k] = new
                break
    if not changes:
        raise ProgramError("E-EVO-PARAMS", "parametric step produced no change", "widen PARAM_SPACE")
    summary = "param " + ", ".join(f"{k} {params[k]!r} -> {v!r}" for k, v in changes.items())
    return write_literal(code, "PARAMS", {**params, **changes}), summary


@dataclass
class ParsedResponse:
    hypothesis: str | None
    kind: str  # diff | full
    code: str | None = None
    blocks: list[tuple[str, str]] = field(default_factory=list)


def parse_response(text: str) -> ParsedResponse:
    m = HYPOTHESIS_RE.search(text)
    hypothesis = m.group(1).strip()[:300] if m else None
    blocks = DIFF_RE.findall(text)
    if blocks:
        return ParsedResponse(hypothesis, "diff", blocks=blocks)
    fences = FENCE_RE.findall(text)
    full = [f for f in fences if "def build" in f]
    if full:
        return ParsedResponse(hypothesis, "full", code=max(full, key=len).rstrip() + "\n")
    return ParsedResponse(hypothesis, "none")


def _locate(code: str, search: str) -> tuple[int, int] | None:
    i = code.find(search)
    if i >= 0:
        return i, i + len(search)
    # Tolerate trailing-whitespace and indentation-width drift line by line.
    want = [ln.rstrip() for ln in search.strip("\n").splitlines()]
    if not want:
        return None
    lines = code.splitlines(keepends=True)
    norm = [ln.rstrip() for ln in lines]
    for start in range(len(lines) - len(want) + 1):
        if all(norm[start + j].strip() == want[j].strip() for j in range(len(want))):
            a = sum(len(x) for x in lines[:start])
            b = a + sum(len(x) for x in lines[start:start + len(want)])
            if b > a and code[b - 1] == "\n" and not search.endswith("\n"):
                b -= 1
            return a, b
    return None


def apply_diff(code: str, blocks: list[tuple[str, str]]) -> str:
    for n, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            raise ProgramError("E-EVO-DIFF", f"SEARCH block {n} is empty",
                               "copy the exact lines to change from the parent program into SEARCH")
        span = _locate(code, search)
        if span is None:
            first = search.strip().splitlines()[0][:120]
            raise ProgramError("E-EVO-DIFF", f"SEARCH block {n} not found in the parent program (first line: {first!r})",
                               "SEARCH must copy lines from the parent program exactly, including indentation")
        code = code[:span[0]] + replace + code[span[1]:]
    return code


def apply_response(parent_code: str, text: str, require_hypothesis: bool = True) -> tuple[str, str, str]:
    """Returns (child code, hypothesis, kind) or raises ProgramError with an LLM-readable message."""
    r = parse_response(text)
    if require_hypothesis and not r.hypothesis:
        raise ProgramError("E-EVO-HYPOTHESIS", "response has no 'HYPOTHESIS: ...' line",
                           "start the answer with one line 'HYPOTHESIS: <what you change and why it should help>'")
    if r.kind == "diff":
        child = apply_diff(parent_code, r.blocks)
    elif r.kind == "full":
        child = r.code
    else:
        raise ProgramError("E-EVO-FORMAT", "response has neither SEARCH/REPLACE blocks nor a ```python program",
                           "answer with SEARCH/REPLACE blocks against the parent, or one full program in ```python")
    if child == parent_code:
        raise ProgramError("E-EVO-NOOP", "the edit leaves the program unchanged", "change the design")
    try:
        compile(child, "program.py", "exec")
    except SyntaxError as e:
        raise ProgramError("E-EVO-SYNTAX", f"child program does not parse: {e.msg} at line {e.lineno}",
                           "fix the Python syntax") from None
    if "def build" not in child:
        raise ProgramError("E-EVO-BUILD", "child program defines no build()", "keep def build() returning a design dict")
    return child, r.hypothesis or "", r.kind
