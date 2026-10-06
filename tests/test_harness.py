import copy
import json

import pytest
import yaml

from harness import evaluate as ev
from harness import physics
from harness.compile_stream import UnsupportedDesign, compile_design, system
from harness.design import Design, DesignError, Link
from harness.designs.reference import REFERENCES, a100, tpuv4, tpuv5e, tpuv6e
from harness.workloads import ROOT, Op, Workload, suite


@pytest.mark.parametrize("name", REFERENCES)
def test_reference_json_matches_python_and_roundtrips(name):
    d = REFERENCES[name]()
    on_disk = Design.load(ROOT / "harness" / "designs" / f"{name}.json")
    assert on_disk == d
    assert Design.from_json(d.to_json()) == d


def test_reference_peaks():
    assert a100().peak_flops() / 1e12 == pytest.approx(311.9, abs=0.1)
    assert tpuv4().peak_flops() / 1e12 == pytest.approx(275.3, abs=0.1)
    assert tpuv5e().peak_flops() / 1e12 == pytest.approx(196.6, abs=0.1)
    assert tpuv6e().peak_flops() / 1e12 == pytest.approx(917.5, abs=0.1)


def _errors(d: dict) -> str:
    with pytest.raises(DesignError) as e:
        Design.from_dict(d)
    return "\n".join(e.value.errors)


def test_validation_messages():
    base = a100().to_dict()
    d = copy.deepcopy(base)
    d["compute"][0]["rowz"] = 4
    assert "unknown field(s) ['rowz']" in _errors(d)
    d = copy.deepcopy(base)
    d["compute"][0]["attach"] = "l3"
    assert "attach='l3' must name a memory unit" in _errors(d)
    d = copy.deepcopy(base)
    d["compute"][0]["count"] = 6
    assert "multiple of attached memory 'l2' count 4" in _errors(d)
    d = copy.deepcopy(base)
    d["links"][0]["endpoints"].append("dram")
    assert "endpoints ['dram'] are not units" in _errors(d)
    d = copy.deepcopy(base)
    d["links"][0]["kind"] = "link"
    assert "joins exactly 2 instances, got 24" in _errors(d)
    d = copy.deepcopy(base)
    d["clock_mhz"] = 50000
    d["tech_node"] = "n2"
    msg = _errors(d)
    assert "clock_mhz = 50000" in msg and "tech_node 'n2' unknown" in msg
    assert "not valid JSON" in "\n".join(pytest.raises(DesignError, Design.from_json, "{").value.errors)


def test_compiled_a100_matches_handwritten_yaml():
    sysd, files = system(a100())
    ref = yaml.safe_load((ROOT / "hw/a100/a100.yaml").read_text())
    for k in ("core_coordinates", "core_connectivity", "offchip_core_id"):
        assert sysd[k] == ref[k], k
    ref_core = yaml.safe_load((ROOT / "hw/a100/cores/a100_sm_cluster.yaml").read_text())
    assert {**files["sm_cluster"], "name": ref_core["name"]} == ref_core


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: setattr(d.memory[0], "count", 8) or setattr(d.compute[0], "count", 16) or
     setattr(d.compute[1], "count", 8), "memory columns"),
    (lambda d: setattr(d.compute[0], "precision", "fp8"), "precision fp8 is not simulated"),
    (lambda d: d.links.append(Link("direct", ["sm_cluster", "offchip"], 1000)), "reaches off-chip memory only"),
])
def test_compiler_rejects_unsupported(mutate, msg, tmp_path):
    d = a100()
    mutate(d)
    d.validate()
    with pytest.raises(UnsupportedDesign, match=msg):
        compile_design(d, tmp_path)


def test_compiler_round_trip_cycles(tmp_path):
    """Compiled A100 reproduces hw/a100/a100.yaml on gemm 1024^3 (runs Stream twice, ~5 s)."""
    onnx = suite("smoke")[0].onnx()
    hand = ev.run_stream(ROOT / "hw/a100/a100.yaml", onnx, 120)
    comp = ev.run_stream(compile_design(a100(), tmp_path), onnx, 120)
    assert hand["status"] == comp["status"] == "ok"
    assert comp["cycles"] == hand["cycles"]


def test_floor_check_flags_too_fast_result():
    w = Workload("smoke", 16, Op("gemm_16_8192", 16, 8192, 8192))
    d = a100()
    hbm_floor_s = 8192 * 8192 * 2 / 2039e9
    assert hbm_floor_s == pytest.approx(65.8e-6, rel=0.01)
    bad = ev.check_floors(20e-6 * d.clock_hz, w, d)  # what buggy Stream reported
    assert not bad["hbm_floor"] and bad["compute_floor"]
    good = ev.check_floors(70e-6 * d.clock_hz, w, d)
    assert good["hbm_floor"] and good["compute_floor"]
    assert not ev.check_floors(0.0, w, d)["compute_floor"]
    assert not ev.check_floors(float("nan"), w, d)["hbm_floor"]


def test_floor_counts_outputs_only_beyond_onchip():
    o = Op("x", 1024, 1024, 64, batch=1, weight=False)
    w = Workload("p", 1, o)
    assert w.compulsory_bytes(1e12) == o.bytes_a + o.bytes_b
    assert w.compulsory_bytes(1) >= o.bytes_a + o.bytes_b + o.bytes_out


@pytest.fixture
def fake_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path)
    calls = []

    def install(fn):
        def run(hw_yaml, onnx, timeout):
            calls.append(onnx.stem)
            return {**fn(onnx.stem), "sim_wall_s": 0.0}
        monkeypatch.setattr(ev, "run_stream", run)
        return calls
    return install


def test_evaluate_records_violations_and_failures(fake_stream):
    def fn(key):
        if key == "gemm_16_8192_8192":
            return {"status": "ok", "cycles": 20e-6 * 1.41e9}
        return {"status": "infeasible", "error": "Infeasible mapping — Core 5: on-chip memory of Core 5 is 10 MB"}
    fake_stream(fn)
    r = ev.evaluate(a100(), "smoke", use_cache=False)
    assert not r["valid"]
    assert any("hbm_floor violated" in v for v in r["violations"])
    assert r["failures"] == ["smoke/gemm_1024: infeasible"]
    row = next(o for o in r["ops"] if o["status"] == "infeasible")
    assert "Core 5 (l2#0)" in row["error"] and "hint" in row["error"]
    ph = r["phases"]["smoke"]
    assert not ph["complete"] and not ph["trusted"] and ph["tokens_per_s"] is None


def test_evaluate_caches_deterministic_outcomes(fake_stream):
    calls = fake_stream(lambda key: {"status": "ok", "cycles": 1e9})
    ev.evaluate(a100(), "smoke")
    n = len(calls)
    r = ev.evaluate(a100(), "smoke")
    assert len(calls) == n and r["n_stream_calls"] == 0 and r["valid"]


def test_evaluate_rejects_without_raising():
    d = a100()
    d.memory[0].count = d.compute[1].count = 8
    r = ev.evaluate(d, "smoke")
    assert r["stage"] == "compile" and not r["valid"] and "memory columns" in r["errors"][0]
    d.memory[0].count = 3
    r = ev.evaluate(d, "smoke")
    assert r["stage"] == "validate" and any("multiple of attached memory" in e for e in r["errors"])


def test_evaluate_program_entry_point(fake_stream, tmp_path):
    fake_stream(lambda key: {"status": "ok", "cycles": 1e9})
    good = tmp_path / "good.py"
    good.write_text("from harness.designs.reference import tpuv4\ndef build():\n    return tpuv4().to_dict()\n")
    r = ev.evaluate_program(good, "smoke")
    assert r["valid"] and r["combined_score"] > 0 and "die_mm2" in r["features"]
    bad = tmp_path / "bad.py"
    bad.write_text("def build():\n    return {'name': 'x'}\n")
    r = ev.evaluate_program(bad, "smoke")
    assert not r["valid"] and r["combined_score"] == 0 and r["stage"] == "validate"


def test_physics_envelope():
    for f in REFERENCES.values():
        assert physics.estimate(f())["violations"] == []
    d = a100()
    d.compute[0].rows = d.compute[0].cols = 512
    est = physics.estimate(d)
    assert any("die_mm2" in v for v in est["violations"]) and any("power_w" in v for v in est["violations"])


def test_suite_flops():
    dec = {w.op.name: w for w in suite("decode") if w.phase == "decode_b1"}
    per_layer = sum(w.flops * w.op.count for n, w in dec.items() if n != "lm_head") / 32
    assert per_layer == pytest.approx(2 * 218_103_808 + 2 * 2 * 32 * 128 * 2048, rel=1e-6)
    assert json.dumps([w.op.key for w in suite("all")])
