"""`status` and `report`: read a campaign's out dir (no kiln session needed except for claim gating)."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .evaluate import claim_status

BANNER = ("!!! W-ENV-0001: the physical envelope (area, power, HBM shoreline) was NOT checked for {n} of {m} scored "
          "designs: kiln-phys does not produce EvalResult.physical yet. Scores ignore silicon/power cost "
          "(off-chip memory bandwidth/capacity IS pinned to the baseline: E-ENV-0007/0008).\n"
          "!!! No result in this campaign is a claim.")


def load(out: str | Path) -> dict:
    out = Path(out)
    state = json.loads((out / "state.json").read_text())
    recs = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r["gen"] <= state["generation"]]
    held = {}
    for x in (out / "heldout.jsonl").read_text().splitlines():
        if x.strip():
            h = json.loads(x)
            held[h["id"]] = h
    cfg = json.loads((out / "campaign.json").read_text())["config"]
    return {"out": out, "state": state, "records": {r["id"]: r for r in recs}, "heldout": held, "cfg": cfg}


def _gbps(r: dict) -> str:
    """Off-chip (mem stack) bandwidth in GB/s from kiln's `offchip_bw` feature."""
    bw = (r.get("features") or {}).get("offchip_bw")
    return f"{bw / 1e9:.1f}" if isinstance(bw, (int, float)) else "-"


def _elites(d: dict) -> list[dict]:
    ids = [rid for _, rid in d["state"]["archive"]["grid"]]
    return sorted((d["records"][i] for i in ids if i in d["records"]), key=lambda r: -r["fitness"])


def status(out) -> str:
    d = load(out)
    st, cfg = d["state"], d["cfg"]
    b = cfg["budget"]
    sp = st.get("spend", {})
    el = _elites(d)
    lines = [
        f"campaign {cfg['name']} ({cfg['mode']}), generation {st['generation']}",
        f"evaluations {st['evaluations']} / {b.get('max_evals')}  children {st['children']}",
        f"LLM {cfg['llm']['backend']}:{cfg['llm'].get('model') or 'mock'}  calls {sp.get('calls', 0)}  tokens in "
        f"{sp.get('input_tokens', 0)} out {sp.get('output_tokens', 0)}  spend ${sp.get('spent_usd', 0):.4f} / "
        f"{b.get('max_usd')}",
        f"wall {st['wall_s']:.0f} s / {b.get('max_wall_s')}  stop: {st.get('stop_reason')}",
        f"archive: {len(el)} cells, best {el[0]['fitness']:.3f} ({el[0]['id']})" if el else "archive: empty",
    ]
    if st.get("env_unchecked"):
        lines.append(f"W-ENV-0001: envelope unchecked for {st['env_unchecked']} scored designs; no claims")
    return "\n".join(lines)


def report(out, top: int = 10, tier_b: str = "unavailable") -> tuple[str, dict]:
    d = load(out)
    st, cfg, recs, held = d["state"], d["cfg"], d["records"], d["heldout"]
    el = _elites(d)
    ok = [r for r in recs.values() if r["status"] == "ok"]
    unchecked = [r for r in ok if not r.get("envelope_checked", True)]
    lines = []
    if unchecked:
        lines += [BANNER.format(n=len(unchecked), m=len(ok)), ""]
    lines.append(status(out))
    gens = st.get("generations", [])
    if gens:
        g = gens[-1]
        lines.append(f"QD-score {g['qd_score']:.3f}  coverage {g['coverage']:.1%}  best by generation: "
                     + " ".join(f"{x['best']:.3f}" for x in gens[-12:]))
    lines.append("")
    lines.append(f"Top {min(top, len(el))} elites ({cfg['fitness']['kind']} vs {cfg['baseline']}, "
                 f"{cfg['options'].get('stack', 'kiln_ideal')} stack; realistic = each design under its own stack):")
    lines.append(f"{'id':<20} {'fitness':>8} {'low':>7} {'high':>7} {'real':>7} {'audit':<8} {'held':>7} {'gap':>7} "
                 f"{'offchip':>8} {'cell':<12} hypothesis")
    for r in el[:top]:
        h = held.get(r["id"]) or {}
        real = (r.get("realistic") or {}).get("score")
        gap = h.get("gap")
        lines.append(f"{r['id']:<20} {r['fitness']:>8.3f} {r.get('fitness_low') or 0:>7.3f} "
                     f"{r.get('fitness_high') or 0:>7.3f} {real if real is not None else float('nan'):>7.3f} "
                     f"{(r.get('audit') or {}).get('status', ''):<8} "
                     f"{h.get('heldout_score', float('nan')):>7.3f} "
                     f"{gap if gap is not None else float('nan'):>+7.1%} "
                     f"{_gbps(r):>8} {str(r.get('cell')):<12} "
                     f"{(r.get('hypothesis') or r.get('operator') or '')[:70]}")
    overfit = [h for h in held.values() if h.get("overfit_workload")]
    if overfit:
        lines.append(f"overfit_workload (held-out gap < {cfg['heldout']['overfit_gap']:.0%}): "
                     + ", ".join(h["id"] for h in overfit))
    claims = [{"id": r["id"], **claim_status(r, held.get(r["id"]), tier_b)} for r in el]
    n_claims = sum(c["claim"] for c in claims)
    lines.append("")
    lines.append(f"CLAIMS: {n_claims}" + ("" if n_claims else " (none: every elite has blockers)"))
    if el:
        lines.append(f"blockers for the top elite {el[0]['id']}:")
        lines += [f"  - {b}" for b in claims[0]["blockers"]]
    lines.append("")
    lines.append("Archive coverage per descriptor (filled bins / bins):")
    axes = json.loads((d["out"] / "archive.json").read_text())["axes"] if (d["out"] / "archive.json").exists() else []
    cells = [tuple(c) for c, _ in st["archive"]["grid"]]
    for i, a in enumerate(axes):
        filled = sorted({c[i] for c in cells})
        lines.append(f"  {a['name']:<22} {len(filled)}/{a['bins']}  bins {filled}")
    lines.append("")
    by_status = collections.Counter(r["status"] for r in recs.values())
    lines.append("Status counts: " + ", ".join(f"{k} {v}" for k, v in by_status.most_common()))
    env = collections.Counter(c for r in recs.values() if r["status"] == "envelope"
                              for c in {e.get("code") for e in r.get("errors") or []}
                              if str(c).startswith("E-ENV"))
    if env:
        lines.append("Envelope rejections: " + ", ".join(f"{k} {v}" for k, v in sorted(env.items()))
                     + " (E-ENV-0007/0008: off-chip bandwidth/capacity above the baseline's)")
    ops = collections.defaultdict(lambda: [0, 0, 0])
    for r in recs.values():
        if r["operator"] == "seed":
            continue
        o = ops[r["operator"]]
        o[0] += 1
        o[1] += r["status"] == "ok"
        o[2] += bool(r.get("elite_event"))
    lines.append("Operators (children / ok / elite events): "
                 + ", ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in sorted(ops.items())))
    audited = [r for r in recs.values() if (r.get("audit") or {}).get("status") not in (None, "not_run")]
    if audited:
        ac = collections.Counter(r["audit"]["status"] for r in audited)
        lines.append("Audits: " + ", ".join(f"{k} {v}" for k, v in ac.items())
                     + f"; suspicious (>{cfg['audit']['suspicion_ratio']}x): "
                     + str(sum(1 for r in audited if r.get("suspicious"))))
    if st.get("findings"):
        lines.append(f"Findings written to {cfg['adversarial_dir']}: {st['findings']} "
                     f"(quarantined floor violations: {st.get('quarantined', 0)})")
    if cfg["mode"] == "redteam":
        holes = [r for r in recs.values() if (r.get("redteam") or {}).get("hole")]
        lines.append(f"Red team `{cfg['redteam']['objective']}`: {len(holes)} holes; residual max objective "
                     f"{max((r['fitness'] for r in ok), default=0):.4f}")
    data = {"state": {k: v for k, v in st.items() if k != "archive"}, "elites": el[:top], "claims": claims[:top],
            "heldout": held, "status_counts": dict(by_status), "envelope_unchecked": len(unchecked),
            "scored": len(ok)}
    return "\n".join(lines), data
