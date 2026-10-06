"""Prompt construction for the proposer LLM. Only train-workload results reach a prompt; held-out scores never do."""

from __future__ import annotations

import difflib
from pathlib import Path

from kiln.evolve import redact

from . import programs

DOCS = (Path(__file__).parent / "docs" / "design_language.md").read_text()

SYSTEM = """You are a computer architect evolving AI-accelerator designs for LLM inference in the kiln simulator.
Each design is a Python program whose build() returns a kiln.hw/1.0 design dict. You change one idea at a time,
state it as a one-line hypothesis, and learn from the simulator's per-workload bottleneck feedback.

{docs}"""

FORMAT_DIFF = """Answer format (strict):
HYPOTHESIS: <one line: the change and why it should raise the score, citing the bottleneck>
then one or more blocks against the parent program:
<<<<<<< SEARCH
<lines copied exactly from the parent program>
=======
<replacement lines>
>>>>>>> REPLACE
Keep blocks small; SEARCH text must match the parent exactly."""

FORMAT_FULL = """Answer format (strict):
HYPOTHESIS: <one line: the change and why it should raise the score, citing the bottleneck>
then the complete new program in one ```python fenced block (it must define build())."""

ENV_WARNING = """WARNING (W-ENV-0001): kiln-phys area/power is not available yet, so the physical envelope (die area,
power, HBM shoreline, node) is NOT being checked. Do not exploit this: design within the baseline's envelope.
Wins that come from adding silicon or power will be discarded when the envelope lands. Off-chip memory IS checked
now: total mem stack bandwidth and capacity may not exceed the baseline's (E-ENV-0007/0008, score 0)."""


def _fmt_fit(r: dict) -> str:
    s = f"{r.get('fitness', 0):.3f}"
    if r.get("fitness_low") is not None and r.get("status") == "ok":
        s += f" [low {r['fitness_low']:.3f}, high {r['fitness_high']:.3f}]"
    real = r.get("realistic")
    if real:
        s += f"; realistic-stack {real['score']:.3f}"
    return s


def _results_block(r: dict, explain_chars: int) -> str:
    lines = [f"status: {r.get('status')}  fitness: {_fmt_fit(r)}"]
    for w, p in (r.get("per_workload") or {}).items():
        if p.get("status") == "ok":
            lines.append(f"- {w}: score {p['score']:.3f} [{p['low']:.3f}, {p['high']:.3f}], "
                         f"{p.get('tokens_per_s') or 0:.1f} tok/s, dominant bound: {p.get('bound')}")
        else:
            lines.append(f"- {w}: {p.get('status')}")
    for name, text in (r.get("feedback") or {}).items():
        lines.append(f"[{name}]\n{text[:explain_chars]}")
    for e in r.get("errors") or []:
        lines.append(f"error {e.get('code')}: {e.get('message')} (hint: {e.get('hint')})")
    return redact("\n".join(lines))


def _desc(r: dict) -> str:
    d = r.get("descriptors") or {}
    return ", ".join(f"{k}={v:.2f}" for k, v in d.items()) + (f" cell={r.get('cell')}" if r.get("cell") else "")


def lineage(r: dict, records: dict, depth: int = 5) -> list[str]:
    out, cur = [], r
    while cur and cur.get("parents") and len(out) < depth:
        pid = cur["parents"][0]
        parent = records.get(pid)
        if parent is None:
            break
        delta = cur.get("fitness", 0) - parent.get("fitness", 0)
        out.append(f"{cur['id']} ({delta:+.3f} vs parent): {cur.get('hypothesis') or cur.get('operator')}")
        cur = parent
    return out


def build_prompt(cfg: dict, parent: dict, parent_code: str, inspirations: list[tuple[dict, str]],
                 failures: list[dict], records: dict, archive_summary: dict, mode: str, env_unchecked: bool,
                 redteam: str | None = None) -> tuple[str, str]:
    pc = cfg["prompt"]
    system = SYSTEM.format(docs=DOCS)
    parts = []
    if redteam:
        parts.append(f"# Goal (red team, 06 §12.3)\nFind simulator holes: maximize the `{redteam}` objective. "
                     "Designs that score high here expose places where kiln's model may be wrong; they are triaged "
                     "and become regression tests. Still return physically meaningful designs.")
    else:
        parts.append(f"# Goal\nMaximize `{cfg['fitness']['kind']}` fitness: whole-step tokens/s vs the simulated "
                     f"baseline `{cfg['baseline']}` (both under the `{cfg['options'].get('stack', 'kiln_ideal')}` "
                     f"stack), {cfg['fitness']['aggregation']} over workloads: {', '.join(cfg['workloads']['train'])}.")
    if env_unchecked:
        parts.append(ENV_WARNING)
    best = archive_summary.get("best")
    parts.append(f"Archive: {archive_summary.get('cells', 0)} cells filled ({archive_summary.get('coverage', 0):.1%} "
                 f"coverage); best fitness {best:.3f}." if best is not None else "Archive: empty.")
    parts.append(f"# Parent {parent['id']} ({_desc(parent)})\n{_results_block(parent, pc['explain_chars'])}")
    lin = lineage(parent, records)
    if lin:
        parts.append("Lineage (most recent first):\n" + "\n".join(f"- {x}" for x in lin))
    parts.append(f"<parent_program>\n```python\n{parent_code}```\n</parent_program>")
    if inspirations:
        insp = ["# Inspirations: elites from other archive cells"]
        for n, (r, code) in enumerate(inspirations):
            head = f"## {r['id']} fitness {_fmt_fit(r)} ({_desc(r)}): {r.get('hypothesis') or r.get('operator')}"
            diff = "".join(difflib.unified_diff(parent_code.splitlines(keepends=True),
                                                code.splitlines(keepends=True), "parent", r["id"], n=1))
            if diff and len(diff) < 0.5 * len(code):
                body = f"diff vs parent:\n```diff\n{diff[:4000]}```"
            elif n < pc["full_program_inspirations"]:
                body = f"```python\n{code}```"
            else:
                body = f"PARAMS: {programs.read_literal(code, 'PARAMS')}"
            insp.append(f"{head}\n{body}")
        parts.append("\n\n".join(insp))
    if failures:
        fl = ["# Recent failures (avoid repeating them)"]
        for f in failures:
            err = (f.get("errors") or [{}])[0]
            fl.append(f"- {f['id']} [{f.get('status')}] {f.get('hypothesis') or f.get('operator')}: "
                      f"{err.get('code')}: {str(err.get('message'))[:300]} (hint: {err.get('hint')})")
        parts.append(redact("\n".join(fl)))
    parts.append("# Task\nPropose one change to the parent that addresses its dominant bottleneck or moves it "
                 "toward an unexplored region of the archive.\n" + (FORMAT_FULL if mode == "llm_full" else FORMAT_DIFF))
    user = "\n\n".join(parts)
    if len(user) > pc["max_chars"]:
        user = _shrink(parts, pc["max_chars"])
    return system, user


def _shrink(parts: list[str], limit: int) -> str:
    # Drop inspirations, then failures, until it fits; the parent program and task are never cut.
    keep = list(parts)
    for marker in ("# Inspirations", "# Recent failures", "Lineage"):
        if len("\n\n".join(keep)) <= limit:
            break
        keep = [p for p in keep if not p.startswith(marker)]
    return "\n\n".join(keep)
