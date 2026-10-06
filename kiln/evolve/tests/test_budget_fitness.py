"""Regression tests for workload weights, seed budgets, seed audits, LLM reservations and the spend ledger."""

import json
import math
from pathlib import Path

import pytest

from kiln_evo import config as C
from kiln_evo.campaign import Campaign, CampaignError
from kiln_evo.evaluate import _agg
from kiln_evo.llm import BudgetExceeded, LLMError, MockBackend, Spend


def _quiet(*_a):
    pass


def test_zero_weight_workloads_are_excluded_from_every_aggregation():
    assert _agg([2.0, 0.5], [1.0, 0.0], "min") == 2.0
    assert _agg([2.0, 0.0], [1.0, 0.0], "geomean") == pytest.approx(2.0)
    assert _agg([2.0, 0.5], [1.0, 0.0], "weighted_harmonic") == pytest.approx(2.0)
    assert _agg([2.0, 0.5], [1.0, 1.0], "geomean") == pytest.approx(1.0)


@pytest.mark.parametrize("weights", [{"llama3_8b:decode_b1": -1.0}, {"llama3_8b:decode_b1": float("nan")},
                                     {"llama3_8b:decode_b1": 0.0, "llama3_8b:decode_b8": 0.0},
                                     {"gptj_6b:decode_b8": 0.0}, {"llama3_8b:decode_b1": "2"}])
def test_negative_nonfinite_or_all_zero_weights_are_rejected(make_cfg, weights):
    with pytest.raises(C.ConfigError, match="fitness.weights"):
        make_cfg("w", fitness={"weights": weights})


def test_seeds_respect_max_evals(make_cfg):
    st = Campaign(make_cfg("one", budget={"max_evals": 1}), log=_quiet).run()
    assert st["evaluations"] == 1 and st["stop_reason"] == "max_evals 1 reached"
    st = Campaign(make_cfg("zero", budget={"max_evals": 0}), log=_quiet).run()
    assert st["evaluations"] == 0 and st["stop_reason"] == "max_evals 0 reached"


def test_failed_seed_audit_is_not_eligible(make_cfg, monkeypatch):
    camp = Campaign(make_cfg("audit", budget={"max_evals": 0}), log=_quiet)
    monkeypatch.setitem(camp.cfg["audit"], "random_rate", 1.0)

    def audit(recs, reasons):
        for r in recs:
            r["audit"] = {"status": "failed", "reasons": ["seed spread 50% > 10%"]}
            r["eligible"] = False

    monkeypatch.setattr(camp.evaluator, "audit", audit)
    prog = camp.out / "programs" / "s.py"
    prog.write_text("def build():\n    return {}\n")
    rec = {"id": "g0000-seed00", "gen": 0, "island": 0, "status": "ok", "fitness": 1.3, "cell": [1, 1, 1],
           "eligible": True, "seed_name": "a100", "operator": "seed", "parents": [], "program_path": "programs/s.py",
           "design_hash": "dsn-s", "audit": {"status": "not_run"}}
    camp._post(0, [rec])
    assert rec["audit"]["status"] == "failed" and rec["eligible"] is False
    assert rec["elite_event"] is None and not camp.archive.grid and not any(camp.archive.islands)


def test_reservation_bounds_the_prompt_and_effective_output_limit():
    s = Spend({"input": 1.0, "output": 0.0}, 0.0005)
    with pytest.raises(BudgetExceeded):
        s.reserve(len(("x" * 900).encode()), 0)
    s = Spend({"input": 0.0, "output": 1.0}, 1.0)
    b = MockBackend({"max_tokens": 10, "extra_body": {"max_tokens": 100_000}}, s)
    assert b.output_limit() == 100_000
    with pytest.raises(BudgetExceeded):
        Spend({"input": 0.0, "output": 1.0}, 0.05).reserve(2, b.output_limit())
    with pytest.raises(LLMError, match="output"):
        MockBackend({"max_tokens": 10, "extra_body": {"max_tokens": None}}, s).output_limit()


def test_spend_ledger_survives_a_crash(tmp_path):
    led = tmp_path / "spend.jsonl"
    s = Spend({"input": 1.0, "output": 4.0}, 10.0, ledger=led)
    r = s.reserve(30, 100)
    s.settle(r, 1_000_000, 1_000_000)
    r = s.reserve(30, 100)
    # Crash with that call in flight: its reservation counts as spent on reload.
    t = Spend({"input": 1.0, "output": 4.0}, 10.0, ledger=led)
    assert t.spent_usd == pytest.approx(5.0 + r.usd) and t.calls == 2
    assert Spend({"input": 1.0, "output": 4.0}, 10.0, ledger=led).spent_usd == pytest.approx(t.spent_usd)


def test_resume_counts_llm_spend_lost_with_the_unfinished_generation(make_cfg):
    cfg = make_cfg("ledger", operators={"parametric": 0.0, "llm_diff": 1.0, "llm_full": 0.0},
                   budget={"max_evals": None, "max_generations": 1, "max_usd": 10.0})
    Campaign(cfg, log=_quiet).run()
    out = Path(cfg["out_dir"])
    st = json.loads((out / "state.json").read_text())
    spent = st["spend"]["spent_usd"]
    assert spent > 0
    # The process died after the calls but before the checkpoint recorded them.
    st["spend"] = {"spent_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
    (out / "state.json").write_text(json.dumps(st))
    camp = Campaign(cfg, resume=True, log=_quiet)
    assert camp.spend.spent_usd == pytest.approx(spent) and math.isfinite(spent)


@pytest.mark.parametrize("key", ["git_hash", "calibration_hash", "baseline_design_hash"])
def test_resume_refuses_a_different_scoring_basis(make_cfg, key):
    cfg = make_cfg(f"basis_{key}", budget={"max_evals": None, "max_generations": 0})
    Campaign(cfg, log=_quiet).run()
    path = Path(cfg["out_dir"]) / "state.json"
    st = json.loads(path.read_text())
    Campaign(cfg, resume=True, log=_quiet)
    st["kiln"][key] = "something-else"
    path.write_text(json.dumps(st))
    with pytest.raises(CampaignError, match="scoring basis changed"):
        Campaign(cfg, resume=True, log=_quiet)
