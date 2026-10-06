"""The evolution loop: islands of MAP-Elites archives, LLM and parametric proposals, the evaluation cascade,
suspicion audits, held-out checkpoints, red-team findings, budgets, and resumable on-disk state.

Out-dir layout:
  campaign.json        resolved config + its hash (resume refuses a different config, budget aside)
  state.json           checkpoint after each generation (counters, spend, archive grids, generation stats, scoring
                       basis: kiln build, calibration and baseline hashes; resume refuses a different basis)
  spend.jsonl          LLM reservations and settlements, fsynced before each call (resume restores spend from it)
  evals.jsonl          one record per child (append-only; a partial generation is dropped on resume)
  heldout.jsonl        held-out re-scores (never read by prompt construction)
  programs/<id>.py     every child program; designs/<id>.json for every scored design
  llm/<id>.txt         prompt and response of every LLM call
  archive.json, designs.arrow, generations.arrow   05 §3.9 archive (rewritten at each checkpoint)
"""

from __future__ import annotations

import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kiln

from . import config as C
from . import programs, prompts
from .archive import Archive, atomic_write, make_axes, write_05_archive
from .evaluate import Child, Evaluator, claim_status
from .llm import BudgetExceeded, LLMError, Spend, make_backend

FAILED = ("invalid", "sandbox", "llm_error", "timeout", "infeasible", "internal_error", "envelope", "pruned")


class CampaignError(RuntimeError):
    pass


def _jsonable(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


class Campaign:
    def __init__(self, cfg: dict, resume: bool = False, log=print):
        self.cfg, self.log = cfg, log
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        for sub in ("programs", "designs", "llm"):
            (self.out / sub).mkdir(exist_ok=True)
        state_path = self.out / "state.json"
        if state_path.exists() and not resume:
            raise CampaignError(f"{self.out} already holds a campaign; use `resume` (or a new out_dir)")
        if resume and not state_path.exists():
            raise CampaignError(f"nothing to resume in {self.out}")
        self.hash = C.config_hash(cfg)
        if resume:
            saved = json.loads((self.out / "campaign.json").read_text())
            if saved["hash"] != self.hash:
                raise CampaignError("campaign config changed since the run started (only `budget` may change on "
                                    f"resume): saved {saved['hash']} vs {self.hash}")
            atomic_write(self.out / "campaign.json", json.dumps({"hash": self.hash, "config": cfg}, indent=1,
                                                                default=str))
        cache_dir = cfg["cache_dir"] or str(self.out / "kiln_cache")
        self.session = kiln.Session(calibration=cfg["calibration"], cache_dir=cache_dir, cache="disk")
        self.archive = Archive(make_axes(cfg["descriptors"]), cfg["islands"]["count"])
        self.evaluator = Evaluator(cfg, self.session, self.out)
        self.basis = self._basis()
        self.heldout: dict[str, dict] = {}
        self.seen: dict[str, str] = {}
        self.generations: list[dict] = []
        self.state = {"generation": -1, "evaluations": 0, "children": 0, "wall_s": 0.0, "stop_reason": None,
                      "spend": {}, "findings": 0, "quarantined": 0, "env_unchecked": 0, "kiln": {}}
        if resume:
            self._load()
            changed = {k: {"saved": self.state.get("kiln", {}).get(k), "now": v} for k, v in self.basis.items()
                       if self.state.get("kiln", {}).get(k) != v}
            if changed:
                raise CampaignError(f"scoring basis changed since the checkpoint: {changed}; archived scores and "
                                    "duplicate exclusions are not comparable with new ones, so this campaign cannot "
                                    "resume (start a new out_dir)")
        else:
            atomic_write(self.out / "campaign.json", json.dumps({"hash": self.hash, "config": cfg}, indent=1,
                                                                default=str))
            (self.out / "evals.jsonl").touch()
            (self.out / "heldout.jsonl").touch()
        sp = self.state["spend"]
        self.spend = Spend(cfg["llm"]["price_usd_per_mtok"], cfg["budget"].get("max_usd"),
                           sp.get("spent_usd", 0.0), sp.get("input_tokens", 0), sp.get("output_tokens", 0),
                           sp.get("calls", 0), ledger=self.out / "spend.jsonl")
        self.backend = make_backend(cfg["llm"], self.spend)
        if cfg["mode"] == "redteam" and cfg["redteam"]["objective"] == "tier_disagreement" \
                and self.evaluator.tier_b_status() != "available":
            raise CampaignError("red-team objective tier_disagreement needs tier B, which this kiln build does not "
                                "implement (E-NOT-IMPLEMENTED); use floor_margin, physics_bound or "
                                "extrapolation_leverage")

    # ---- persistence -------------------------------------------------------------------------------------
    def _basis(self) -> dict:
        """What every archived score depends on beyond the config: the kiln build, plus the content behind each
        name the config holds, as the baseline evaluations resolve it under the campaign's options: the effective
        calibration set, the baseline design, each train and held-out workload and the software stack recipe."""
        ws = self.evaluator.train + list(self.cfg["workloads"]["heldout"])
        prov = {w: self.evaluator.baseline(w).get("provenance") or {} for w in ws}
        p0 = prov[ws[0]]
        return {"version": kiln.KILN_VERSION, "git_hash": kiln.GIT_HASH,
                "calibration_hash": p0.get("calibration_hash"), "calibration": p0.get("calibration_id"),
                "baseline_design_hash": p0.get("design_hash"),
                "workloads": {w: p.get("workload_hash") for w, p in prov.items()},
                "stacks": sorted({str((p.get("flags") or {}).get("stack")) for p in prov.values()})}

    def _journal(self, name: str) -> list[dict]:
        """The records of journal `name`. An unparseable final line is the tail of a record torn by a crash
        mid-append (records are appended before the checkpoint that commits them), so it is dropped; corruption
        anywhere else is an error."""
        lines = [x for x in (self.out / name).read_text().splitlines() if x.strip()]
        out = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                if i == len(lines) - 1:
                    self.log(f"resume: dropped a torn final record of {name}")
                    break
                raise CampaignError(f"{name} line {i + 1} is corrupt; this campaign cannot resume") from None
        return out

    def _load(self) -> None:
        self.state = json.loads((self.out / "state.json").read_text())
        gen = self.state["generation"]
        keep, dropped = [], 0
        for r in self._journal("evals.jsonl"):
            if r["gen"] <= gen:
                keep.append(r)
            else:
                dropped += 1
        atomic_write(self.out / "evals.jsonl", "".join(json.dumps(r) + "\n" for r in keep))
        if dropped:
            self.log(f"resume: dropped {dropped} records of unfinished generation {gen + 1}")
        for r in keep:
            self.archive.records[r["id"]] = r
            if r.get("design_hash") and r["status"] != "duplicate":
                self.seen.setdefault(r["design_hash"], r["id"])
        self.archive.restore(self.state["archive"])
        missing = [rid for g in [self.archive.grid, *self.archive.islands] for rid in g.values()
                   if rid not in self.archive.records]
        if missing:
            raise CampaignError(f"state.json references records missing from evals.jsonl: {missing[:5]}")
        hk = []
        for h in self._journal("heldout.jsonl"):
            if h["gen"] <= gen:
                self.heldout[h["id"]] = h
                hk.append(h)
        atomic_write(self.out / "heldout.jsonl", "".join(json.dumps(h) + "\n" for h in hk))
        self.generations = self.state.get("generations", [])

    def _checkpoint(self) -> None:
        self.state["spend"] = self.spend.state()
        self.state["archive"] = self.archive.snapshot()
        self.state["generations"] = self.generations
        self.state["kiln"] = self.basis
        atomic_write(self.out / "state.json", json.dumps(self.state, indent=1, default=str))
        for note in write_05_archive(self.out, self.cfg, self.archive, self.generations, self.heldout,
                                     self.state["kiln"]):
            self.log(f"note: {note}")

    def _append(self, name: str, rows: list[dict]) -> None:
        with open(self.out / name, "a") as f:
            for r in rows:
                f.write(json.dumps(_jsonable(r), default=str) + "\n")

    # ---- budget ------------------------------------------------------------------------------------------
    def _stop(self) -> str | None:
        b = self.cfg["budget"]
        if b.get("max_evals") is not None and self.state["evaluations"] >= b["max_evals"]:
            return f"max_evals {b['max_evals']} reached"
        if self.spend.exhausted():
            return f"max_usd {b['max_usd']} reached"
        if b.get("max_wall_s") is not None and self.state["wall_s"] + time.monotonic() - self._mark >= b["max_wall_s"]:
            return f"max_wall_s {b['max_wall_s']} reached"
        if b.get("max_generations") is not None and self.state["generation"] >= b["max_generations"]:
            return f"max_generations {b['max_generations']} reached"
        return None

    # ---- proposals ---------------------------------------------------------------------------------------
    def _code(self, rid: str) -> str:
        return (self.out / self.archive.records[rid]["program_path"]).read_text()

    def _rng(self, *parts) -> random.Random:
        return random.Random(":".join(str(p) for p in (self.cfg["seed"], *parts)))

    def _select_parent(self, island: int, rng: random.Random) -> dict | None:
        el = self.archive.elites(island) or self.archive.elites()
        if not el:
            return None
        sel = self.cfg["selection"]
        if rng.random() < sel["exploit"]:
            pool = rng.sample(el, min(sel["tournament"], len(el)))
            return max(pool, key=lambda r: r["fitness"])
        return rng.choice(el)

    def _inspirations(self, parent: dict, rng: random.Random) -> list[tuple[dict, str]]:
        others = [r for r in self.archive.elites() if r["id"] != parent["id"] and r.get("cell") != parent.get("cell")]
        pick = rng.sample(others, min(self.cfg["selection"]["inspirations"], len(others)))
        pick.sort(key=lambda r: -r["fitness"])
        return [(r, self._code(r["id"])) for r in pick]

    def _failures(self, island: int) -> list[dict]:
        n = self.cfg["selection"]["failures_shown"]
        failed = [r for r in self.archive.records.values()
                  if r["island"] == island and r["status"] in FAILED + ("floor_violation",) and r.get("errors")]
        # Insertion order differs between a live run and a resumed one (the journal is sorted by id).
        return sorted(failed, key=lambda r: (r["gen"], r["id"]))[-n:] if n > 0 else []

    def _summary(self) -> dict:
        b = self.archive.best()
        return {"cells": len(self.archive.grid), "coverage": self.archive.coverage(),
                "best": b["fitness"] if b else None}

    def _propose_one(self, gen: int, island: int, k: int) -> Child | dict | None:
        rng = self._rng("child", gen, island, k)
        cid = f"g{gen:04d}-i{island}-c{k:02d}"
        parent = self._select_parent(island, rng)
        if parent is None:
            return None
        code = self._code(parent["id"])
        ops = self.cfg["operators"]
        op = rng.choices(list(ops), weights=list(ops.values()))[0]
        if op == "parametric":
            try:
                child, summary = programs.parametric_mutation(code, rng, n_changes=rng.choice((1, 1, 2, 3)))
                return Child(cid, gen, island, [parent["id"]], "parametric", summary, child)
            except programs.ProgramError:
                op = "llm_diff"
        insp = self._inspirations(parent, rng)
        system, user = prompts.build_prompt(
            self.cfg, parent, code, insp, self._failures(island), self.archive.records, self._summary(), op,
            env_unchecked=self.state["env_unchecked"] > 0 or not parent.get("envelope_checked", True),
            redteam=self.cfg["redteam"]["objective"] if self.cfg["mode"] == "redteam" else None)
        try:
            comp = self.backend.complete(system, user)
        except BudgetExceeded:
            raise
        except LLMError as e:
            err = {"code": "E-EVO-LLM", "message": str(e)[:500], "hint": "provider call failed"}
            return {"id": cid, "gen": gen, "island": island, "parents": [parent["id"]], "operator": op,
                    "status": "llm_error", "errors": [err], "fitness": 0.0, "hypothesis": None}
        (self.out / "llm" / f"{cid}.txt").write_text(
            f"# model {comp.model} in {comp.input_tokens} out {comp.output_tokens} ${comp.cost_usd:.5f}\n"
            f"## system\n{system}\n## user\n{user}\n## response\n{comp.text}\n")
        llm_meta = {"llm": {"model": comp.model, "input_tokens": comp.input_tokens,
                            "output_tokens": comp.output_tokens, "cost_usd": comp.cost_usd,
                            "latency_s": round(comp.latency_s, 3)}}
        try:
            child, hyp, kind = programs.apply_response(code, comp.text)
        except programs.ProgramError as e:
            hyp = programs.parse_response(comp.text).hypothesis
            return {"id": cid, "gen": gen, "island": island, "parents": [parent["id"]], "operator": op,
                    "status": "llm_error", "stage": "parse", "errors": [e.as_dict()], "fitness": 0.0,
                    "hypothesis": hyp, **llm_meta}
        return Child(cid, gen, island, [parent["id"]], f"llm_{kind}", hyp, child, meta=llm_meta)

    def _propose(self, gen: int, n: int) -> tuple[list[Child], list[dict], bool]:
        slots = []
        per = self.cfg["batch"]["children_per_island"]
        for k in range(per):
            for i in range(self.cfg["islands"]["count"]):
                slots.append((i, k))
        slots = slots[:n]
        out_children, out_fail, exhausted = [], [], False
        workers = 1 if self.cfg["llm"]["backend"] == "mock" else min(8, len(slots)) or 1
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(self._propose_one, gen, i, k) for i, k in slots]
            for f in futs:
                try:
                    r = f.result()
                except BudgetExceeded as e:
                    self.log(f"budget: {e}")
                    exhausted = True
                    continue
                if isinstance(r, Child):
                    out_children.append(r)
                elif isinstance(r, dict):
                    out_fail.append(r)
        return out_children, out_fail, exhausted

    # ---- seeds -------------------------------------------------------------------------------------------
    def _seed_children(self) -> list[Child]:
        out = []
        for n, s in enumerate(self.cfg["seeds"]):
            code = programs.materialize_seed(Path(s["path"]).read_text(), C.REFERENCE_DESIGNS)
            if s.get("params"):
                code = programs.with_params(code, s["params"])
            out.append(Child(f"g0000-seed{n:02d}", 0, n % self.cfg["islands"]["count"], [], "seed",
                             f"seed {s['name']}", code, meta={"seed_name": s["name"]}))
        return out

    # ---- generation --------------------------------------------------------------------------------------
    def _post(self, gen: int, recs: list[dict]) -> None:
        cfg = self.cfg
        ok = [r for r in recs if r["status"] == "ok"]
        if any(not r.get("envelope_checked", True) for r in ok):
            self.state["env_unchecked"] += sum(1 for r in ok if not r.get("envelope_checked", True))
        for r in recs:
            if r["status"] == "floor_violation":
                self._finding("floor_violation", r, {"expected": "E-FLOOR", "note": "simulator bug, quarantined"})
                self.state["quarantined"] += 1
        if cfg["mode"] == "redteam":
            for r in ok:
                v, hole, details = self.evaluator.redteam_value(r)
                r["redteam"] = {"objective": cfg["redteam"]["objective"], "value": v, "hole": hole,
                                "details": details, "score": r["fitness"]}
                r["fitness"] = v
                if hole:
                    self._finding(f"redteam_{cfg['redteam']['objective']}", r, r["redteam"])
        else:
            self._audit(gen, ok)
        for r in recs:
            if r.get("_design") is not None and r["status"] == "ok":
                atomic_write(self.out / "designs" / f"{r['id']}.json", json.dumps(r["_design"]))
            if r.get("seed_name") and r["status"] == "ok" and (r.get("audit") or {}).get("status") != "failed":
                r["eligible"] = True
            event = self.archive.insert(r)
            r["elite_event"] = event
            if r.get("seed_name") and r.get("eligible") and r.get("cell"):
                for isl in self.archive.islands:
                    isl.setdefault(tuple(r["cell"]), r["id"])

    def _audit(self, gen: int, ok: list[dict]) -> None:
        a = self.cfg["audit"]
        rng = self._rng("audit", gen)
        best = self.archive.best()
        reasons: dict[str, list[str]] = {}
        provisional = Archive(self.archive.axes, len(self.archive.islands))
        provisional.grid = dict(self.archive.grid)
        provisional.records = dict(self.archive.records)
        for r in ok:
            why = []
            if r["fitness"] > a["suspicion_ratio"]:
                why.append(f"score {r['fitness']:.3f} > {a['suspicion_ratio']}x baseline")
            if best and r["fitness"] > a["suspicion_ratio"] * best["fitness"]:
                why.append(f"score > {a['suspicion_ratio']}x archive best {best['fitness']:.3f}")
            if a["on_new_elite"] and r.get("eligible") and provisional.would_place(r) and r.get("operator") != "seed":
                why.append("new archive elite")
            if rng.random() < a["random_rate"]:
                why.append("random sample")
            if why:
                reasons[r["id"]] = why
                r["suspicious"] = any("baseline" in w or "best" in w for w in why)
            if r.get("eligible"):
                provisional.insert(r)
        flagged = [r for r in ok if r["id"] in reasons]
        self.evaluator.audit(flagged, reasons)
        for r in flagged:
            if r.get("features"):
                # The audit replaces the interval, so interval-dependent descriptors may move the record's cell.
                r["cell"] = list(self.archive.cell(r["features"]) or []) or None
                r["descriptors"] = self.archive.normalized(r["features"])
                r["eligible"] = r.get("eligible", False) and r["cell"] is not None
            if r["audit"]["status"] == "failed":
                self._finding("audit_failed", r, r["audit"])

    def _finding(self, kind: str, r: dict, details: dict) -> None:
        d = Path(self.cfg["adversarial_dir"]) / kind
        d.mkdir(parents=True, exist_ok=True)
        key = r.get("design_hash") or r["id"]
        body = {"schema": "kiln.adversarial/1", "kind": kind, "campaign": self.cfg["name"],
                "campaign_hash": self.hash, "design_id": r["id"], "design_hash": r.get("design_hash"),
                "status": r["status"], "fitness": r.get("fitness"), "details": details,
                "per_workload": r.get("per_workload"), "errors": r.get("errors"),
                "workloads": self.cfg["workloads"]["train"], "options": C.kiln_options(self.cfg),
                "kiln": {"version": kiln.KILN_VERSION, "git_hash": kiln.GIT_HASH},
                "program": (self.out / r["program_path"]).read_text() if r.get("program_path") else None,
                "design": r.get("_design")}
        atomic_write(d / f"{key}.json", json.dumps(body, indent=1, default=str))
        self.state["findings"] += 1

    def _heldout_pass(self, gen: int) -> None:
        top = [r for r in self.archive.elites()[: self.cfg["heldout"]["top_k"]]
               if r["id"] not in self.heldout or self.heldout[r["id"]].get("evolve_score") != r["fitness"]]
        designs = {}
        for r in top:
            p = self.out / "designs" / f"{r['id']}.json"
            if p.exists():
                designs[r["id"]] = json.loads(p.read_text())
        top = [r for r in top if r["id"] in designs]
        res = self.evaluator.heldout(top, designs)
        rows = []
        for rid, h in res.items():
            h["gen"] = gen
            self.heldout[rid] = h
            rows.append(h)
        self._append("heldout.jsonl", rows)

    def _gen_stats(self, gen: int, recs: list[dict]) -> dict:
        el = [r["fitness"] for r in self.archive.elites()]
        return {"generation": gen, "best": max(el) if el else 0.0, "median": statistics.median(el) if el else 0.0,
                "qd_score": self.archive.qd_score(), "coverage": self.archive.coverage(),
                "evaluations": self.state["evaluations"],
                "invalid_count": sum(1 for r in recs if r["status"] not in ("ok", "duplicate")),
                "children": len(recs), "new_cells": sum(1 for r in recs if r.get("elite_event") == "new_cell"),
                "improved": sum(1 for r in recs if r.get("elite_event") == "improved"),
                "spent_usd": self.spend.spent_usd}

    def _run_generation(self, gen: int, children: list[Child], failures: list[dict]) -> None:
        recs = self.evaluator.evaluate(children, self.archive, self.seen) if children else []
        self._post(gen, recs)
        for f in failures:
            f.setdefault("cell", None)
            f.setdefault("eligible", False)
            self.archive.records[f["id"]] = f
        all_recs = recs + failures
        self.state["children"] += len(all_recs)
        self.state["evaluations"] += sum(1 for r in recs if r["status"] != "duplicate")
        self._append("evals.jsonl", sorted(all_recs, key=lambda r: r["id"]))
        every = self.cfg["heldout"]["every_generations"]
        if self.cfg["mode"] == "evolve" and (gen == 0 or (every and gen % every == 0)):
            self._heldout_pass(gen)
        isl = self.cfg["islands"]
        if isl["count"] > 1 and isl["migration_interval"] and gen > 0 and gen % isl["migration_interval"] == 0:
            for i in range(isl["count"]):
                self.archive.migrate(i, (i + 1) % isl["count"], isl["migrants"])
        st = self._gen_stats(gen, all_recs)
        self.generations.append(st)
        self.state["generation"] = gen
        self.log(f"gen {gen}: {len(all_recs)} children, ok {sum(r['status'] == 'ok' for r in all_recs)}, "
                 f"new cells {st['new_cells']}, improved {st['improved']}, best {st['best']:.3f}, "
                 f"coverage {st['coverage']:.1%}, evals {self.state['evaluations']}, "
                 f"${self.spend.spent_usd:.4f}")

    def run(self) -> dict:
        """Runs until a budget stops it. State is checkpointed only at generation boundaries, so an interrupted
        run (exception, kill) resumes from the last complete generation."""
        self._mark = time.monotonic()
        self.state["stop_reason"] = None
        if self.state["generation"] < 0 and not self._stop():
            seeds = self._seed_children()
            if self.cfg["budget"].get("max_evals") is not None:
                seeds = seeds[: self.cfg["budget"]["max_evals"] - self.state["evaluations"]]
            self._run_generation(0, seeds, [])
            self._wall()
            self._checkpoint()
            if not self.archive.grid and len(seeds) == len(self.cfg["seeds"]):
                raise CampaignError("no seed produced a scored design; see evals.jsonl")
        while True:
            stop = self._stop()
            if stop:
                self.state["stop_reason"] = stop
                break
            gen = self.state["generation"] + 1
            n = self.cfg["batch"]["children_per_island"] * self.cfg["islands"]["count"]
            if self.cfg["budget"].get("max_evals") is not None:
                n = min(n, self.cfg["budget"]["max_evals"] - self.state["evaluations"])
            children, failures, exhausted = self._propose(gen, n)
            if children or failures:
                self._run_generation(gen, children, failures)
            else:
                self.state["generation"] = gen
            self._wall()
            self._checkpoint()
            if exhausted:
                self.state["stop_reason"] = (f"max_usd {self.cfg['budget']['max_usd']} reached (the next call's "
                                             "worst case would exceed it)")
                break
        if self.cfg["mode"] == "evolve":
            self._heldout_pass(self.state["generation"])
        self._wall()
        self._checkpoint()
        self.log(f"stopped: {self.state['stop_reason']}")
        return self.state

    def _wall(self) -> None:
        now = time.monotonic()
        self.state["wall_s"] += now - self._mark
        self._mark = now

    def claims(self) -> list[dict]:
        tb = self.evaluator.tier_b_status()
        return [{"id": r["id"], **claim_status(r, self.heldout.get(r["id"]), tb)} for r in self.archive.elites()]
