import random
from pathlib import Path

import json5
import pytest

from kiln_evo import config as C
from kiln_evo import programs as P

REF = C.REFERENCE_DESIGNS
A100 = P.materialize_seed((C.PKG / "seeds" / "a100.py").read_text(), REF)
TPU = P.materialize_seed((C.PKG / "seeds" / "tpu_v4.py").read_text(), REF)


def _build(code: str) -> dict:
    ns: dict = {}
    exec(compile(code, "program.py", "exec"), ns)
    return ns["build"]()


def test_seed_programs_reproduce_reference_designs():
    for code, ref_name, name in ((A100, "a100_sxm4_40gb", "a100-seed"), (TPU, "tpu_v4", "tpu-v4-seed")):
        ref = json5.loads((REF / f"{ref_name}.json5").read_text())
        ref.pop("family", None)
        ref["meta"].pop("claims")
        ref["name"] = name
        got = _build(code)
        assert got == ref


def test_seeds_cap_offchip_knobs_at_the_reference():
    for code in (A100, TPU):
        params, space = P.read_literal(code, "PARAMS"), P.read_literal(code, "PARAM_SPACE")
        for k in P.read_literal(code, "ENVELOPE_KNOBS"):
            s = space[k]
            assert (max(P._magnitude(c) for c in s["choices"]) if "choices" in s else s["max"]) \
                == P._magnitude(params[k]), (k, s)


def test_params_roundtrip_and_override():
    params = P.read_literal(A100, "PARAMS")
    assert params["hbm_enabled"] == 5
    code = P.with_params(A100, {"hbm_enabled": 6, "harvest": False})
    assert P.read_literal(code, "PARAMS") == {**params, "hbm_enabled": 6, "harvest": False}
    assert P.read_literal(code, "PARAM_SPACE") == P.read_literal(A100, "PARAM_SPACE")
    assert _build(code)["system"]["package"]["mem_stacks"][0]["disabled"] == []
    with pytest.raises(P.ProgramError, match="unknown PARAMS"):
        P.with_params(A100, {"nope": 1})


def test_parametric_mutation_stays_in_space():
    space = P.read_literal(A100, "PARAM_SPACE")
    rng = random.Random(0)
    code = A100
    for _ in range(200):
        code, summary = P.parametric_mutation(code, rng, n_changes=2)
        assert summary.startswith("param ")
        for k, v in P.read_literal(code, "PARAMS").items():
            s = space.get(k)
            if s is None:
                continue
            if "choices" in s:
                assert v in s["choices"]
            else:
                assert s["min"] <= v <= s["max"], (k, v)
    with pytest.raises(P.ProgramError):
        P.parametric_mutation("def build():\n    return {}\n", rng)


def test_parse_and_apply_diff():
    resp = ('HYPOTHESIS: more HBM bandwidth\n<<<<<<< SEARCH\n    "hbm_enabled": 5,\n=======\n'
            '    "hbm_enabled": 6,\n>>>>>>> REPLACE\n')
    child, hyp, kind = P.apply_response(A100, resp)
    assert (hyp, kind) == ("more HBM bandwidth", "diff")
    assert P.read_literal(child, "PARAMS")["hbm_enabled"] == 6
    # Indentation drift in SEARCH is tolerated line by line.
    resp2 = resp.replace('    "hbm_enabled": 5,', '"hbm_enabled": 5,   ')
    assert P.read_literal(P.apply_response(A100, resp2)[0], "PARAMS")["hbm_enabled"] == 6


def test_full_rewrite_and_errors():
    full = "HYPOTHESIS: tiny\n```python\nPARAMS = {}\ndef build():\n    return {'schema': 'kiln.hw/1.0'}\n```"
    child, hyp, kind = P.apply_response(A100, full)
    assert kind == "full" and "def build" in child
    cases = {
        "E-EVO-HYPOTHESIS": full.replace("HYPOTHESIS: tiny\n", ""),
        "E-EVO-FORMAT": "HYPOTHESIS: x\nno code here",
        "E-EVO-DIFF": "HYPOTHESIS: x\n<<<<<<< SEARCH\nnot in parent\n=======\ny\n>>>>>>> REPLACE",
        "E-EVO-SYNTAX": "HYPOTHESIS: x\n```python\ndef build(:\n    pass\n```",
        "E-EVO-NOOP": "HYPOTHESIS: x\n<<<<<<< SEARCH\nimport copy\n=======\nimport copy\n>>>>>>> REPLACE",
    }
    for code, text in cases.items():
        with pytest.raises(P.ProgramError) as e:
            P.apply_response(A100, text)
        assert e.value.code == code
        d = e.value.as_dict()
        assert d["hint"] and d["message"]


def test_design_hash_is_key_order_independent():
    assert P.design_hash({"a": 1, "b": [1, 2]}) == P.design_hash({"b": [1, 2], "a": 1})
    assert Path(C.PKG / "docs" / "design_language.md").read_text().startswith("# kiln design language")
