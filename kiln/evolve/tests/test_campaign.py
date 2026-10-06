"""End-to-end campaigns with the mock LLM against the real kiln engine."""

import importlib.util
import json
from pathlib import Path

import pytest

from kiln_evo.campaign import Campaign, CampaignError
from kiln_evo.report import report, status

DECODE = ["llama3_8b:decode_b1", "llama3_8b:decode_b8", "llama3_8b:decode_b32"]
HELDOUT = ["llama3_8b:decode_b1_kv32768", "gptj_6b:decode_b8"]


def _records(out) -> list[dict]:
    return [json.loads(x) for x in (Path(out) / "evals.jsonl").read_text().splitlines() if x.strip()]


def _quiet(*_a):
    pass


def test_smoke_decode_suite_50_evals_then_resume(make_cfg):
    cfg = make_cfg(
        "smoke", workloads={"train": DECODE, "screen": ["llama3_8b:decode_b8"], "heldout": HELDOUT},
        batch={"children_per_island": 5}, budget={"max_evals": 50},
        seeds=[{"name": "a100", "program": "kiln_evo:seeds/a100.py"},
               {"name": "a100_hbm6", "program": "kiln_evo:seeds/a100.py", "params": {"hbm_enabled": 6}},
               {"name": "a100_sram", "program": "kiln_evo:seeds/a100.py",
                "params": {"tpc_per_gpc": 4, "l2_slice_kib": 2048}},
               {"name": "tpu_v4", "program": "kiln_evo:seeds/tpu_v4.py"}],
        envelope={"offchip_bw": None, "offchip_bytes": None, "hbm_shoreline_mm": 1000.0})
    st = Campaign(cfg, log=_quiet).run()
    out = Path(cfg["out_dir"])
    assert st["evaluations"] == 50 and st["stop_reason"] == "max_evals 50 reached"
    recs = _records(out)
    assert len({r["id"] for r in recs}) == len(recs)
    assert sum(r["status"] not in ("duplicate", "llm_error") for r in recs) == 50
    assert len(st["archive"]["grid"]) >= 4, "archive should fill several cells"
    ops = {r["operator"] for r in recs}
    assert {"seed", "parametric"} <= ops and ops & {"llm_diff", "llm_full"}
    # Baseline-equal seed scores exactly 1.0; the 6-stack seed is suspicious (> 1.15x) and gets audited (off-chip
    # and shoreline limits are lifted in this campaign; test_extra_stack_breaks_the_matched_envelope covers the
    # default).
    by_name = {r.get("seed_name"): r for r in recs if r["operator"] == "seed"}
    assert by_name["a100"]["fitness"] == pytest.approx(1.0)
    hbm6 = by_name["a100_hbm6"]
    assert hbm6["suspicious"] and hbm6["audit"]["status"] == "pending"
    assert set(hbm6["audit"]["seed_scores"]) == {"0", "1"} and hbm6["audit"]["tier_b"] == "unavailable"
    assert hbm6["fitness"] <= hbm6["audit"]["fitness_before"] and hbm6["interval_method"] == "corners"
    # kiln-phys checks the physical envelope, so no W-ENV-0001; nothing is a claim yet (audits pending, no tier B).
    assert all(r["envelope_checked"] for r in recs if r["status"] == "ok")
    text, data = report(out)
    assert not text.startswith("!!! W-ENV-0001") and "CLAIMS: 0" in text
    assert data["claims"] and not any(c["claim"] for c in data["claims"])
    assert not any("W-ENV-0001" in b for c in data["claims"] for b in c["blockers"])
    # Held-out workloads are scored but never reach a prompt.
    held = [json.loads(x) for x in (out / "heldout.jsonl").read_text().splitlines()]
    assert held and all(h["status"] in ("ok", "infeasible") for h in held)
    prompts = list((out / "llm").glob("*.txt"))
    assert prompts and not any(w in p.read_text() for p in prompts for w in HELDOUT)
    arrow = ("designs.arrow", "generations.arrow") if importlib.util.find_spec("pyarrow") else ()
    for f in ("archive.json", *arrow, "state.json", "campaign.json"):
        assert (out / f).exists()
    assert "evaluations 50 / 50" in status(out)
    with pytest.raises(CampaignError, match="resume"):
        Campaign(cfg, log=_quiet)
    # Resume with a larger budget continues from the checkpoint.
    cfg["budget"]["max_evals"] = 62
    st2 = Campaign(cfg, resume=True, log=_quiet).run()
    assert st2["evaluations"] == 62 and st2["generation"] > st["generation"]
    recs2 = _records(out)
    assert recs2[: len(recs)] == recs and len({r["id"] for r in recs2}) == len(recs2)
    assert "evaluations 62 / 62" in status(out)


def test_extra_stack_breaks_the_matched_envelope(make_cfg):
    cfg = make_cfg("offchip", budget={"max_evals": 2},
                   seeds=[{"name": "a100", "program": "kiln_evo:seeds/a100.py"},
                          {"name": "a100_hbm6", "program": "kiln_evo:seeds/a100.py", "params": {"hbm_enabled": 6}}])
    Campaign(cfg, log=_quiet).run()
    by_name = {r.get("seed_name"): r for r in _records(cfg["out_dir"])}
    assert by_name["a100"]["status"] == "ok" and by_name["a100"]["fitness"] == pytest.approx(1.0)
    assert by_name["a100"]["features"]["offchip_bw"] == pytest.approx(1555.2e9)
    hbm6 = by_name["a100_hbm6"]
    assert hbm6["status"] == "envelope" and hbm6["fitness"] == 0.0
    # The sixth stack also needs more HBM shoreline than the baseline die has (E-ENV-0006, kiln-phys).
    assert {e["code"] for e in hbm6["errors"]} == {"E-ENV-0006", "E-ENV-0007", "E-ENV-0008"}
    assert all(e.get("hint") for e in hbm6["errors"])
    text, _ = report(cfg["out_dir"])
    assert "Envelope rejections: E-ENV-0006 1, E-ENV-0007 1, E-ENV-0008 1" in text and " 1555.2 " in text


def _fingerprint(out):
    st = json.loads((Path(out) / "state.json").read_text())
    return ([(r["id"], r["status"], round(r["fitness"], 12), r.get("design_hash")) for r in _records(out)],
            st["archive"], st["evaluations"])


def test_resume_after_crash_equals_uninterrupted_run(make_cfg):
    straight = make_cfg("straight", budget={"max_evals": None, "max_generations": 3})
    Campaign(straight, log=_quiet).run()
    part = make_cfg("part", budget={"max_evals": None, "max_generations": 2})
    Campaign(part, log=_quiet).run()
    out = Path(part["out_dir"])
    # A crash mid-generation 3 leaves records past the checkpoint; resume drops and redoes them.
    with open(out / "evals.jsonl", "a") as f:
        f.write(json.dumps({"id": "g0003-i0-c00", "gen": 3, "status": "ok", "fitness": 9.0}) + "\n")
    part["budget"]["max_generations"] = 3
    Campaign(part, resume=True, log=_quiet).run()
    a, b = _fingerprint(straight["out_dir"]), _fingerprint(part["out_dir"])
    assert a == b
    changed = dict(part, seed=99)
    with pytest.raises(CampaignError, match="config changed"):
        Campaign(changed, resume=True, log=_quiet)


def test_dollar_and_wall_budgets_stop(make_cfg):
    cfg = make_cfg("usd", operators={"parametric": 0.0, "llm_diff": 1.0, "llm_full": 0.0},
                   budget={"max_evals": 1000, "max_usd": 0.12})
    st = Campaign(cfg, log=_quiet).run()
    assert "max_usd" in st["stop_reason"]
    assert 0 < st["spend"]["spent_usd"] <= 0.12 and st["spend"]["calls"] >= 1
    cfg = make_cfg("wall", budget={"max_evals": 1000, "max_wall_s": 0})
    st = Campaign(cfg, log=_quiet).run()
    assert st["stop_reason"] == "max_wall_s 0 reached" and st["generation"] == -1 and st["evaluations"] == 0


def test_redteam_writes_findings(make_cfg, tmp_path):
    cfg = make_cfg("rt", mode="redteam", redteam={"objective": "physics_bound", "hole_threshold": 0.0},
                   cascade={"prune_ratio": None}, budget={"max_evals": 8})
    st = Campaign(cfg, log=_quiet).run()
    assert st["findings"] >= 1
    files = list((tmp_path / "adversarial" / "redteam_physics_bound").glob("*.json"))
    assert files
    f = json.loads(files[0].read_text())
    assert f["schema"] == "kiln.adversarial/1" and f["program"] and f["design"]["schema"] == "kiln.hw/1.0"
    assert "score_over_roofline_ratio" in f["details"]["details"]
    recs = [r for r in _records(cfg["out_dir"]) if r["status"] == "ok"]
    assert all(r["redteam"]["objective"] == "physics_bound" for r in recs)
    with pytest.raises(CampaignError, match="tier B"):
        Campaign(make_cfg("rt2", mode="redteam", redteam={"objective": "tier_disagreement"}), log=_quiet)


def test_floor_violation_is_quarantined(make_cfg, tmp_path):
    camp = Campaign(make_cfg("fv", budget={"max_evals": 0}), log=_quiet)
    prog = Path(camp.out / "programs" / "x.py")
    prog.write_text("def build():\n    return {}\n")
    rec = {"id": "x", "gen": 1, "island": 0, "status": "floor_violation", "fitness": 0.0, "cell": [1, 1, 1],
           "eligible": False, "program_path": "programs/x.py", "design_hash": "dsn-x", "errors": [
               {"code": "E-FLOOR-I1", "message": "compute floor", "hint": ""}], "parents": [], "operator": "llm_diff"}
    camp._post(1, [rec])
    assert rec["elite_event"] is None and not camp.archive.grid
    found = json.loads((tmp_path / "adversarial" / "floor_violation" / "dsn-x.json").read_text())
    assert found["details"]["expected"] == "E-FLOOR"


def test_torn_journal_tails_are_dropped_on_resume(make_cfg):
    cfg = make_cfg("torn", budget={"max_evals": None, "max_generations": 0})
    Campaign(cfg, log=_quiet).run()
    out = Path(cfg["out_dir"])
    committed = _records(out)
    for name in ("evals.jsonl", "heldout.jsonl"):
        with open(out / name, "a") as f:
            f.write(json.dumps({"id": "g0001-i0-c00", "gen": 1, "status": "ok", "fitness": 9.0}) + "\n")
            f.write('{"id": "g0001-i0-c01", "gen": 1, "sta')
    cfg["budget"]["max_generations"] = 1
    Campaign(cfg, resume=True, log=_quiet).run()
    recs = _records(out)
    assert recs[: len(committed)] == committed and all(r["fitness"] != 9.0 for r in recs)
    lines = (out / "evals.jsonl").read_text().splitlines()
    lines[0] = lines[0][:20]
    (out / "evals.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(CampaignError, match="evals.jsonl"):
        Campaign(cfg, resume=True, log=_quiet)


def test_resume_refuses_a_changed_stack_recipe(make_cfg, tmp_path):
    root = Path(__file__).resolve().parents[2]
    recipe = tmp_path / "recipe.json5"
    recipe.write_text((root / "stacks" / "kiln_ideal.json5").read_text().replace('id: "kiln_ideal"', 'id: "mine"'))
    cfg = make_cfg("stk", options={"stack": str(recipe)}, budget={"max_evals": None, "max_generations": 0},
                   operators={"parametric": 1.0, "llm_diff": 0.0, "llm_full": 0.0})
    Campaign(cfg, log=_quiet).run()
    recipe.write_text(recipe.read_text().replace("onchip_fraction: 0.25", "onchip_fraction: 0.9"))
    cfg["budget"]["max_generations"] = 1
    with pytest.raises(CampaignError, match="scoring basis changed"):
        Campaign(cfg, resume=True, log=_quiet)


def test_failures_shown_do_not_depend_on_record_order(make_cfg):
    camp = Campaign(make_cfg("fail-order", budget={"max_evals": 0}), log=lambda *_a: None)
    camp.cfg["selection"]["failures_shown"] = 1
    err = [{"code": "E-X", "message": "m"}]
    recs = {"g0001-i0-c01": {"id": "g0001-i0-c01", "gen": 1, "island": 0, "status": "invalid", "errors": err},
            "g0001-i0-c00": {"id": "g0001-i0-c00", "gen": 1, "island": 0, "status": "llm_error", "errors": err},
            "g0000-i0-c05": {"id": "g0000-i0-c05", "gen": 0, "island": 0, "status": "invalid", "errors": err}}
    picks = set()
    for order in (list(recs), sorted(recs), sorted(recs, reverse=True)):
        camp.archive.records = {k: recs[k] for k in order}
        picks.add(tuple(r["id"] for r in camp._failures(0)))
    assert picks == {("g0001-i0-c01",)}
