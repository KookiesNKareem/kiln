import kiln.evolve as ev
from kiln.evolve import evaluate_program, run_build


def test_refuses_to_run_without_os_isolation(program, tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    monkeypatch.setattr(ev, "_isolation", lambda: None)
    p = program(f"open({str(marker)!r}, 'w').write('x')\ndef build():\n    return {{}}\n")
    out = run_build(p)
    assert not out.ok and out.stage == "sandbox" and out.error["code"] == "E-SANDBOX-UNAVAILABLE"
    assert not marker.exists()
    m = evaluate_program(p)
    assert m["combined_score"] == 0.0 and m["errors"][0]["code"] == "E-SANDBOX-UNAVAILABLE"
