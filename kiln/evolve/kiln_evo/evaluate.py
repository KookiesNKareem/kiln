"""Evaluation cascade for child programs (06 §6.6): sandboxed build -> S0 validate -> tier A screen on the screen
workloads (prune against the target cell's elite) -> full train suite. Plus the suspicion audit (multi-seed,
corners; tier B pending until it exists), held-out re-scoring, red-team objectives and claim gating."""

from __future__ import annotations

import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import kiln
from kiln.evolve import run_build

from . import config as C
from . import programs

ENV_UNCHECKED = ("W-ENV-0001", "W-ENV-UNAVAILABLE")
STATUS_PRIORITY = ("floor_violation", "internal_error", "timeout", "invalid", "envelope", "infeasible", "pruned")
QUARANTINE_NOTE = "simulator bug; this design is quarantined"


@dataclass
class Child:
    id: str
    gen: int
    island: int
    parents: list
    operator: str
    hypothesis: str
    code: str
    meta: dict = field(default_factory=dict)


def _agg(values: list[float], weights: list[float], how: str) -> float:
    """Aggregates over the workloads with positive weight (config validation rejects negative weights)."""
    pairs = [(v, wi) for v, wi in zip(values, weights) if wi > 0]
    if not pairs:
        return 0.0
    values, weights = [v for v, _ in pairs], [wi for _, wi in pairs]
    if how == "min":
        return min(values)
    if any(v <= 0 for v in values):
        return 0.0
    w = sum(weights)
    if how == "weighted_harmonic":
        return w / sum(wi / v for v, wi in zip(values, weights))
    return math.exp(sum(wi * math.log(v) for v, wi in zip(values, weights)) / w)


def _first_error(d: dict) -> list[dict]:
    errs = (d.get("violations") or []) + (d.get("errors") or [])
    return [{k: e.get(k) for k in ("code", "message", "hint", "path") if e.get(k) is not None} for e in errs[:3]]


class Evaluator:
    def __init__(self, cfg: dict, session: kiln.Session, out_dir: Path):
        self.cfg, self.session, self.out = cfg, session, Path(out_dir)
        self.train = list(cfg["workloads"]["train"])
        self.screen = [w for w in cfg["workloads"]["screen"] if w in self.train]
        weights = cfg["fitness"].get("weights") or {}
        self.weights = {w: float(weights.get(w, 1.0)) for w in self.train + list(cfg["workloads"]["heldout"])}
        self.opts = C.kiln_options(cfg)
        self.workers = cfg["batch"]["workers"]
        self._tier_b: str | None = None
        self._baseline: dict = {}

    # ---- kiln calls -------------------------------------------------------------------------------------
    def _batch(self, items: list[tuple], opts: dict) -> list[dict]:
        if not items:
            return []
        rs = self.session.evaluate_batch(items, opts, max_workers=self.workers)
        out = []
        for r in rs:
            d = r.to_dict()
            d["_explain"] = r.explain(max_chars=self.cfg["prompt"]["explain_chars"])
            out.append(d)
        return out

    def baseline(self, workload: str) -> dict:
        if workload not in self._baseline:
            self._baseline[workload] = self.session.baseline(self.cfg["baseline"], workload, self.opts).to_dict()
        return self._baseline[workload]

    def tier_b_status(self) -> str:
        """'available' or 'unavailable' (probed once: kiln returns E-NOT-IMPLEMENTED until M4)."""
        if self._tier_b is None:
            r = self.session.evaluate(self.cfg["baseline"], self.train[0], {**self.opts, "profile": "full",
                                                                           "tier": "B"}).to_dict()
            nyi = any(e.get("code") == "E-NOT-IMPLEMENTED" for e in r.get("errors") or [])
            self._tier_b = "unavailable" if nyi or r["status"] != "ok" else "available"
        return self._tier_b

    # ---- aggregation ------------------------------------------------------------------------------------
    def aggregate(self, per: dict[str, dict], workloads: list[str]) -> dict:
        how = self.cfg["fitness"]["aggregation"]
        ds = [per[w] for w in workloads]
        bad = [d for d in ds if d["status"] != "ok"]
        warnings = sorted({w["code"] for d in ds for w in d.get("warnings") or []})
        agg = {"warnings": warnings, "envelope_checked": not any(c in ENV_UNCHECKED for c in warnings)}
        summary = {}
        for w, d in zip(workloads, ds):
            si = d.get("score_interval") or {}
            ph = (d.get("phases") or [{}])[0]
            bb = ph.get("bound_breakdown") or {}
            summary[w] = {"status": d["status"], "score": d.get("score", 0.0), "low": si.get("low"),
                          "high": si.get("high"), "tokens_per_s": (ph.get("tokens_per_s") or {}).get("central"),
                          "bound": max(bb, key=bb.get) if bb else None,
                          "realistic": (d.get("score_realistic") or {}).get("score")}
        agg["per_workload"] = summary
        agg["feedback"] = {w: d.get("_explain", "") for w, d in zip(workloads, ds)}
        agg["errors"] = [e for d in bad for e in _first_error(d)][:3]
        if bad:
            st = min((d["status"] for d in bad),
                     key=lambda s: STATUS_PRIORITY.index(s) if s in STATUS_PRIORITY else len(STATUS_PRIORITY))
            agg.update(status=st, fitness=0.0, fitness_low=0.0, fitness_high=0.0, realistic=None,
                       features=(ds[0].get("features") or {}))
            return agg
        ws = [self.weights.get(w, 1.0) for w in workloads]
        central = _agg([d["score"] for d in ds], ws, how)
        low = _agg([d["score_interval"]["low"] for d in ds], ws, how)
        high = _agg([d["score_interval"]["high"] for d in ds], ws, how)
        real = [d.get("score_realistic") for d in ds]
        realistic = None
        if all(real):
            realistic = {"score": _agg([r["score"] for r in real], ws, how),
                         "low": _agg([r["interval"]["low"] for r in real], ws, how),
                         "high": _agg([r["interval"]["high"] for r in real], ws, how),
                         "candidate_stack": real[0].get("candidate_stack"),
                         "baseline_stack": real[0].get("baseline_stack")}
        feats = dict(ds[0].get("features") or {})
        for k in ("bound_frac_mem",):
            vals = [d["features"][k] for d in ds if k in (d.get("features") or {})]
            if vals:
                feats[k] = statistics.fmean(vals)
        splits = [d["features"]["energy_split"] for d in ds if "energy_split" in (d.get("features") or {})]
        if splits:
            feats["energy_split"] = [statistics.fmean(x) for x in zip(*splits)]
        feats["score_rel_width"] = (high - low) / central if central > 0 else 0.0
        extrap = sorted({x for d in ds for x in (d.get("audit") or {}).get("extrapolated_components") or []}
                        | {json.dumps(x, sort_keys=True) if not isinstance(x, str) else x
                           for d in ds for x in (d.get("calibration") or {}).get("extrapolated") or []})
        agg.update(status="ok", fitness=central, fitness_low=low, fitness_high=high, realistic=realistic,
                   features=feats, extrapolated=extrap,
                   interval_method=(ds[0].get("interval") or {}).get("method"),
                   provenance={k: (ds[0].get("provenance") or {}).get(k)
                               for k in ("kiln_git_hash", "git_hash", "calibration_hash", "kiln_version")})
        return agg

    # ---- cascade ----------------------------------------------------------------------------------------
    def build_all(self, children: list[Child]) -> list:
        prog_dir = self.out / "programs"
        prog_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for c in children:
            p = prog_dir / f"{c.id}.py"
            p.write_text(c.code)
            paths.append(p)
        sb = self.cfg["sandbox"]
        hidden = [self.out, self.cfg["adversarial_dir"], getattr(self.session, "cache_dir", None)]
        hidden = [d for d in hidden if d]
        n = max(1, min(self.cfg["batch"]["max_build_workers"], len(children)))
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(lambda p: run_build(p, timeout_s=sb["timeout_s"], memory_mb=sb["memory_mb"],
                                                   deny_read=hidden), paths))

    def evaluate(self, children: list[Child], archive, seen: dict[str, str]) -> list[dict]:
        t0 = time.monotonic()
        recs = [self._base_record(c) for c in children]
        builds = self.build_all(children)
        live = []
        for rec, b in zip(recs, builds):
            if not b.ok:
                rec.update(status="sandbox", stage="build", errors=[b.error],
                           feedback={"build": f"FAILED ({b.error['code']}): {b.error['message']}\nhint: "
                                              f"{b.error['hint']}\n{b.stderr[-400:]}".strip()})
                continue
            rec["design_hash"] = programs.design_hash(b.design)
            if rec["design_hash"] in seen:
                rec.update(status="duplicate", stage="dedup", duplicate_of=seen[rec["design_hash"]])
                continue
            seen[rec["design_hash"]] = rec["id"]
            errs = [e for e in self.session.validate(b.design, self.cfg["options"].get("profile", "search"))
                    if e.get("severity") == "error"]
            if errs:
                rec.update(status="invalid", stage="S0", errors=errs[:3],
                           feedback={"validate": "\n".join(f"{e['code']} at {e.get('path', '')}: {e['message']}"
                                                           f"\n  hint: {e.get('hint', '')}" for e in errs[:4])})
                continue
            live.append((rec, b.design))
        screen_all = self._batch([(d, w) for _, d in live for w in self.screen], self.opts)
        k = 0
        screened = []
        for rec, design in live:
            sres = {}
            for w in self.screen:
                sres[w] = screen_all[k]
                k += 1
            screened.append((rec, design, sres))
        survivors = []
        prune = self.cfg["cascade"].get("prune_ratio")
        for rec, design, sres in screened:
            if not self.screen:
                survivors.append((rec, design, sres))
                continue
            agg = self.aggregate(sres, self.screen)
            if agg["status"] != "ok":
                rec.update(stage="screen", **self._final(agg))
                continue
            cell = archive.cell(agg["features"])
            elite = archive.elite_fitness(cell) if cell else None
            if (prune and self.cfg["mode"] == "evolve" and elite is not None and agg["fitness"] < prune * elite):
                rec.update(self._final(agg))
                rec.update(stage="screen", status="pruned", cell=list(cell), pruned_against=elite,
                           errors=[{"code": "E-EVO-PRUNED", "message": f"screen score {agg['fitness']:.3f} on "
                                    f"{', '.join(self.screen)} is below {prune} x the cell elite's {elite:.3f}",
                                    "hint": "a change must not lose this much on the screen workloads"}])
                continue
            survivors.append((rec, design, sres))
        rest = [w for w in self.train if w not in self.screen]
        full_items = [(d, w) for _, d, _ in survivors for w in rest]
        full = self._batch(full_items, self.opts)
        k = 0
        for rec, design, sres in survivors:
            res = dict(sres)
            for w in rest:
                res[w] = full[k]
                k += 1
            agg = self.aggregate(res, self.train)
            rec.update(stage="full", **self._final(agg))
            rec["_design"] = design
            rec["_results"] = res
        for rec in recs:
            if rec["status"] == "ok":
                rec["cell"] = list(archive.cell(rec["features"]) or []) or None
                rec["descriptors"] = archive.normalized(rec["features"])
            rec["eligible"] = rec["status"] == "ok" and rec.get("cell") is not None
            rec["wall_s"] = (time.monotonic() - t0) / max(1, len(recs))
            if rec["status"] == "floor_violation":
                rec["feedback"] = {"quarantine": QUARANTINE_NOTE}
        return recs

    def _final(self, agg: dict) -> dict:
        return {k: agg.get(k) for k in ("status", "fitness", "fitness_low", "fitness_high", "realistic", "features",
                                         "per_workload", "feedback", "errors", "warnings", "envelope_checked",
                                         "extrapolated", "interval_method", "provenance")}

    def _base_record(self, c: Child) -> dict:
        return {"id": c.id, "gen": c.gen, "island": c.island, "parents": c.parents, "operator": c.operator,
                "hypothesis": c.hypothesis, "program_path": f"programs/{c.id}.py",
                "program_hash": programs.program_hash(c.code), "status": None, "stage": None, "fitness": 0.0,
                "cell": None, "descriptors": {}, "features": {}, "errors": [], "feedback": {}, "warnings": [],
                "audit": {"status": "not_run"}, **c.meta}

    # ---- suspicion audit (06 §6.6 S4, tier B pending) ---------------------------------------------------
    def audit(self, recs: list[dict], reasons: dict[str, list[str]]) -> None:
        """Re-evaluates each record per mapper seed with `audit.interval` intervals; the score becomes the minimum
        over {tier A, each seed}; seed spread above `max_seed_spread` fails the audit. Tier B does not exist yet,
        so a clean audit is `pending`, never `passed`."""
        if not recs:
            return
        acfg = self.cfg["audit"]
        seeds = list(acfg["seeds"])
        tier_b = self.tier_b_status()
        items, keys = [], []
        for r in recs:
            for s in seeds:
                for w in self.train:
                    items.append((r["_design"], w))
                    keys.append((r["id"], s, w))
        results = {}
        by_seed: dict[int, list] = {}
        for i, (key, item) in enumerate(zip(keys, items)):
            by_seed.setdefault(key[1], []).append(i)
        for s, idx in by_seed.items():
            opts = {**self.opts, "interval": acfg["interval"], "seeds": [s]}
            out = self._batch([items[i] for i in idx], opts)
            for i, d in zip(idx, out):
                results[keys[i]] = d
        for r in recs:
            seed_fit, seed_low, failed = {}, {}, []
            for s in seeds:
                agg = self.aggregate({w: results[(r["id"], s, w)] for w in self.train}, self.train)
                if agg["status"] != "ok":
                    failed.append(f"seed {s}: status {agg['status']}")
                    seed_fit[s] = 0.0
                    continue
                seed_fit[s], seed_low[s] = agg["fitness"], agg["fitness_low"]
                if s == seeds[0]:
                    r["fitness_low"], r["fitness_high"] = agg["fitness_low"], agg["fitness_high"]
                    r["interval_method"] = acfg["interval"]
            vals = list(seed_fit.values())
            med = statistics.median(vals) if vals else 0.0
            spread = (max(vals) - min(vals)) / med if med > 0 else float("inf")
            if spread > acfg["max_seed_spread"]:
                failed.append(f"seed spread {spread:.1%} > {acfg['max_seed_spread']:.0%}")
            audited = min([r["fitness"], *vals]) if vals else 0.0
            a = {"status": "failed" if failed else ("pending" if tier_b != "available" else "pending_tier_b_run"),
                 "triggers": reasons.get(r["id"], []), "reasons": failed, "seed_scores": seed_fit,
                 "seed_low": seed_low, "seed_spread": spread, "tier_b": tier_b, "fitness_before": r["fitness"],
                 "interval": acfg["interval"]}
            if tier_b != "available" and not failed:
                a["reasons"] = ["tier B not implemented (M4): audit stays pending; no claim possible"]
            r["audit"] = a
            r["fitness"] = audited
            if failed:
                r["eligible"] = False

    # ---- held-out ---------------------------------------------------------------------------------------
    def heldout(self, recs: list[dict], designs: dict[str, dict]) -> dict[str, dict]:
        ws = list(self.cfg["workloads"]["heldout"])
        if not ws or not recs:
            return {}
        items = [(designs[r["id"]], w) for r in recs for w in ws]
        out = self._batch(items, self.opts)
        res, k = {}, 0
        gap_flag = self.cfg["heldout"]["overfit_gap"]
        for r in recs:
            per = {}
            for w in ws:
                per[w] = out[k]
                k += 1
            agg = self.aggregate(per, ws)
            h = agg["fitness"]
            gap = h / r["fitness"] - 1 if r["fitness"] > 0 else None
            res[r["id"]] = {"id": r["id"], "heldout_score": h, "heldout_low": agg.get("fitness_low"),
                            "status": agg["status"], "evolve_score": r["fitness"], "gap": gap,
                            "overfit_workload": gap is not None and gap < gap_flag,
                            "per_workload": {w: v["score"] for w, v in agg["per_workload"].items()}}
        return res

    # ---- red team (06 §12.3 T3) ---------------------------------------------------------------------------
    def _threshold(self, default: float) -> float:
        t = self.cfg["redteam"].get("hole_threshold")
        return default if t is None else t

    def redteam_value(self, rec: dict) -> tuple[float, bool, dict]:
        """(objective to maximize, is_hole, details) for a fully evaluated ok record."""
        obj = self.cfg["redteam"]["objective"]
        res = rec["_results"]
        if obj == "floor_margin":
            margins = {}
            for w, d in res.items():
                ph = d["phases"][0]
                floor = max((f["seconds"] for f in ph.get("floors") or []), default=0.0)
                if floor > 0:
                    margins[w] = ph["time_s"]["low"] / floor - 1
            m = min(margins.values()) if margins else float("inf")
            thr = self._threshold(0.01)
            return -m, m < thr and rec["fitness"] > 1.0, {"floor_margin": margins, "min_margin": m}
        if obj == "physics_bound":
            ratios = {}
            for w, d in res.items():
                b = self.baseline(w)["phases"][0]
                c = d["phases"][0]
                roof = lambda ph: next((f["seconds"] for f in ph.get("floors") or [] if f["kind"] == "roofline"),  # noqa: E731
                                       None)
                if roof(b) and roof(c):
                    ratios[w] = d["score"] / (roof(b) / roof(c))
            v = _agg(list(ratios.values()), [1.0] * len(ratios), "geomean") if ratios else 0.0
            thr = self._threshold(1.25)
            return v, v > thr, {"score_over_roofline_ratio": ratios}
        if obj == "extrapolation_leverage":
            shares = {}
            for w, d in res.items():
                cal = d.get("calibration") or {}
                ext = {json.dumps(x, sort_keys=True) for x in cal.get("extrapolated") or []}
                contrib = [c for cs in (cal.get("contributions") or {}).values() for c in cs]
                tot = sum(abs(c.get("delta_makespan_s", 0.0)) for c in contrib)
                hit = sum(abs(c.get("delta_makespan_s", 0.0)) for c in contrib
                          if json.dumps(c.get("name"), sort_keys=True) in ext
                          or json.dumps({"name": c.get("name"), "key": c.get("key")}, sort_keys=True) in ext)
                shares[w] = hit / tot if tot > 0 else 0.0
            share = statistics.fmean(shares.values()) if shares else 0.0
            gain = max(0.0, rec["fitness"] - 1.0)
            thr = self._threshold(0.5)
            return gain * share, share > thr and gain > 0, {"extrapolated_share": shares}
        raise ValueError(f"red-team objective {obj!r} needs tier B, which kiln does not implement yet")


def claim_status(rec: dict, heldout: dict | None, tier_b: str) -> dict:
    """06 §6.6 claim rule. Returns {"claim": bool, "blockers": [...]}; anything with a blocker is not a claim."""
    b = []
    if rec.get("status") != "ok":
        b.append(f"status {rec.get('status')}")
    if not rec.get("envelope_checked"):
        b.append("W-ENV-0001: physical envelope not checked (kiln-phys area/power not available)")
    a = rec.get("audit") or {}
    if tier_b != "available":
        b.append("tier B audit unavailable (M4)")
    if a.get("status") != "passed":
        b.append(f"audit status {a.get('status', 'not_run')}")
    if len(a.get("seed_scores") or {}) < 3:
        b.append("fewer than 3 mapper seeds")
    if rec.get("interval_method") != "corners":
        b.append(f"interval method {rec.get('interval_method')} (claims need corners)")
    if (rec.get("fitness_low") or 0) < 1.0:
        b.append(f"score_interval.low {rec.get('fitness_low') or 0:.3f} < 1.0 (kiln_ideal)")
    real = rec.get("realistic") or {}
    if (real.get("low") or 0) < 1.0:
        b.append(f"score_realistic.low {real.get('low') or 0:.3f} < 1.0 (own stacks)")
    if rec.get("extrapolated"):
        b.append(f"extrapolated components: {rec['extrapolated'][:3]}")
    if not heldout:
        b.append("held-out score not measured")
    else:
        if (heldout.get("heldout_score") or 0) < 1.1:
            b.append(f"held-out score {heldout.get('heldout_score') or 0:.3f} < 1.1")
        if (heldout.get("heldout_low") or 0) < 1.0:
            b.append(f"held-out low {heldout.get('heldout_low') or 0:.3f} < 1.0")
    b.append("trust suite (06 §12.5) not run for this design")
    return {"claim": not b, "blockers": b}
