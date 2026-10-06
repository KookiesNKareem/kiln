import json

import pytest

import kiln
from conftest import DESIGNS

FULL = {"profile": "full"}


def test_evaluate_returns_structured_result(session):
    r = session.evaluate("a100", "llama3_8b:decode_b1", FULL)
    d = r.to_dict()
    assert d["schema"] == "kiln.result/1"
    assert r.status == "ok" and d["stage_reached"] == "S3"
    assert r.score == pytest.approx(1.0)
    assert d["phases"][0]["time_s"]["central"] > 0
    assert d["provenance"]["design_hash"].startswith("hw1-")
    assert json.loads(r.to_json()) == d
    assert r["status"] == r.status
    assert r.features["n_chips"] == 1
    assert r.explain().startswith("status: ")
    assert len(r.explain(max_chars=60)) <= 60


def test_design_forms_agree(session):
    path = DESIGNS / "reference" / "a100_sxm4_40gb.json5"
    by_name = session.evaluate("a100_40gb", "smoke", FULL)
    by_path = session.evaluate(str(path), "smoke", FULL)
    by_text = session.evaluate(path.read_text(), "smoke", FULL)
    assert by_name.deterministic_hash() == by_path.deterministic_hash() == by_text.deterministic_hash()
    legacy = json.loads((DESIGNS / "legacy" / "a100.json").read_text())
    assert session.evaluate(legacy, "smoke", FULL).provenance["design_hash"].startswith("hw1-")


def test_invalid_inputs_are_results_not_exceptions(session):
    r = session.evaluate({"schema": "kiln.hw/1.0", "name": "x"}, "smoke")
    assert r.status == "invalid" and r.score == 0.0
    assert all({"code", "message", "hint"} <= e.keys() for e in r.errors if "hint" in e)
    r = session.evaluate("a100", "nope:decode_b1")
    assert r.status == "invalid" and r.errors[0]["code"].startswith("E-WL")
    r = session.evaluate("a100", "standard")
    assert r.status == "invalid" and r.errors[0]["code"] == "E-IR-1102"
    graded = session.evaluate({"schema": "kiln.hw/1.0"}, "smoke", {"invalid_score": "graded"})
    assert graded.score <= -1.0


def test_bad_options_and_non_data_raise(session):
    with pytest.raises(ValueError, match="E-OPT-0001"):
        session.evaluate("a100", "smoke", {"fitness": {"kind": "nope"}})
    with pytest.raises(TypeError, match="JSON data only"):
        session.evaluate({"schema": object()}, "smoke")
    with pytest.raises(ValueError, match="non-finite"):
        session.evaluate({"x": float("nan")}, "smoke")


def test_batch_order_and_bare_designs(session):
    items = [("a100", "smoke"), ({"schema": "kiln.hw/1.0"}, "smoke"), ("tpu_v5e", "smoke")]
    rs = session.evaluate_batch(items, FULL, max_workers=3)
    singles = [session.evaluate(d, w, FULL) for d, w in items]
    assert [r.deterministic_hash() for r in rs] == [r.deterministic_hash() for r in singles]
    rs = session.evaluate_batch(["a100", "tpu_v5e"], {**FULL, "workload": "smoke"})
    assert len(rs) == 2
    with pytest.raises(ValueError, match="bare design"):
        session.evaluate_batch(["a100"])


def test_validate_bench_export_descriptors():
    assert kiln.validate("a100", profile="full") == [] or all(e["severity"] == "warning" for e in kiln.validate("a100", profile="full"))
    assert kiln.validate("a100")[0]["code"] == "E-IR-1102"
    m = kiln.bench_export("legacy")
    assert m["schema"] == "kiln.bench/1" and len(m["ops"]) == 52
    with pytest.raises(ValueError):
        kiln.bench_export("no_such_suite")
    names = [d["name"] for d in kiln.descriptors()]
    assert names[0] == "die_mm2_total" and "score_rel_width" in names
    n = kiln.normalize_features({"onchip_bytes": 2**27, "energy_split": [0.5, 0.25, 0.25], "n_chips": 1e9})
    assert n == pytest.approx({"onchip_bytes": 0.5, "energy_split_compute": 0.5, "energy_split_memory": 0.25,
                 "energy_split_interconnect": 0.25, "n_chips": 1.0})


def test_disk_cache_dir_and_calibration(tmp_path):
    s = kiln.Session(cache_dir=str(tmp_path), calib="generic-v1", threads=2)
    assert s.cache_dir == str(tmp_path) and s.threads == 2
    assert s.calibration["id"] == "generic-v1" and s.calibration["hash"].startswith("cal1-")
    with pytest.raises(ValueError, match="E-CAL-0001"):
        kiln.Session(calibration="not a set!")
    with pytest.raises(ValueError):
        kiln.Session(cache="ram")


def test_explain_dict_and_render_stub(session):
    r = session.evaluate("a100", "smoke", FULL)
    assert kiln.explain(r.to_dict()) == r.explain()
    with pytest.raises(NotImplementedError, match="E-NOT-IMPLEMENTED"):
        kiln.render(r, view="floorplan")


@pytest.mark.parametrize("kind", ["perf_per_area", "perf_per_watt"])
def test_score_rel_width_describes_the_final_fitness_interval(session, kind):
    d = session.evaluate("a100_40gb", "smoke", {"profile": "full", "interval": "corners",
                                                "fitness": {"kind": kind}}).to_dict()
    assert d["status"] == "ok", d["errors"]
    si = d["score_interval"]
    assert d["features"]["score_rel_width"] == pytest.approx((si["high"] - si["low"]) / si["central"])


def test_unrepresentable_timeouts_are_option_errors(session):
    for t in (1e300, 1e20):
        with pytest.raises(ValueError, match="timeout_s"):
            session.evaluate("a100_40gb", "smoke", {"profile": "full", "timeout_s": {"A": t}})
