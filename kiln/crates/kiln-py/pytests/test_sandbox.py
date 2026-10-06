import json
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiln
import kiln.evolve
from conftest import DESIGNS
from kiln.evolve import _isolation, _supervise, evaluate_program, run_build

H100 = DESIGNS / "reference" / "h100_sxm5_80gb.json5"

pytestmark = pytest.mark.skipif(_isolation() is None, reason="no OS sandbox on this host: run_build refuses "
                                "every program (test_sandbox_refusal.py)")


def failed(m, stage, code):
    assert m["combined_score"] == 0.0 and m["valid"] == 0.0
    assert m["stage"] == stage, m
    assert m["errors"][0]["code"] == code, m["errors"]
    assert m["feedback"].startswith(f"FAILED ({code})")
    assert "hint: " in m["feedback"]


def test_good_program_reaches_kiln(program, session):
    p = program(f"""
        def build():
            return {H100.read_text()!r}
    """)
    # The H100 SXM5 has more off-chip bandwidth and capacity than the A100-40GB baseline. The program returns the
    # design as JSON5 text: the sandbox cannot read the repo.
    m = evaluate_program(p, suite="smoke", session=session, options={"profile": "full"})
    assert m["status"] == "envelope" and m["errors"][0]["code"] == "E-ENV-0007", m["feedback"]
    m = evaluate_program(p, suite="smoke", session=session, options={"profile": "full"},
                         fitness="baseline_relative")
    assert m["result"]["schema"] == "kiln.result/1"
    assert m["status"] == "ok", m["feedback"]
    assert 0.0 <= m["n_chips"] <= 1.0 and m["descriptors"]["n_chips"] == m["n_chips"]
    assert m["features"]["n_chips"] == 1
    assert m["combined_score"] > 0.0


def test_invalid_design_feedback(program, session):
    p = program("def build():\n    return {'schema': 'kiln.hw/1.0', 'name': 'x'}\n")
    m = evaluate_program(p, suite="smoke", session=session)
    assert m["combined_score"] == 0.0 and m["status"] == "invalid"
    assert m["errors"] and m["feedback"].startswith("status: invalid")


def test_timeout_kills_program(program):
    p = program("def build():\n    while True:\n        pass\n")
    m = evaluate_program(p, timeout_s=1.0)
    failed(m, "timeout", "E-SANDBOX-TIMEOUT")


def test_sleeping_child_processes_are_killed(program):
    p = program("""
        import subprocess, sys
        def build():
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            while True:
                pass
    """)
    out = run_build(p, timeout_s=1.0)
    # macOS denies fork in the sandbox; Linux runs the child in the sandbox's pid namespace.
    assert out.stage in ("timeout", "build"), out.error
    assert out.wall_s < 5


def test_detached_descendant_cannot_outlive_the_timeout(program):
    p = program("""
        import subprocess, sys
        def build():
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            while True:
                pass
    """)
    t0 = time.monotonic()
    out = run_build(p, timeout_s=1.0)
    assert out.stage in ("timeout", "build"), out.error
    assert time.monotonic() - t0 < 5


def test_pipe_held_by_an_escaped_process_does_not_block_draining():
    holder = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], " \
             "start_new_session=True)"
    p = subprocess.Popen([sys.executable, "-c", holder], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, start_new_session=True)
    t0 = time.monotonic()
    _, killed = _supervise(p, time.monotonic() + 10, None)
    assert killed is None and time.monotonic() - t0 < 3


def test_stderr_flood_is_bounded(program):
    p = program("""
        import os
        def build():
            block = b"x" * (1 << 20)
            for _ in range(400):
                os.write(2, block)
            return {"schema": "kiln.hw/1.0"}
    """)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = run_build(p, timeout_s=30.0)
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    grew_mib = grew / (1 << 20) if sys.platform == "darwin" else grew / 1024
    assert out.ok, out.error
    assert grew_mib < 200, grew_mib
    assert len(out.stderr) <= 800


@pytest.mark.parametrize("body", [
    "open(sys.argv[2], 'w').write('[]')",
    "open(sys.argv[2], 'w').write('{\"ok\": true}')",
    "open(sys.argv[2], 'w').write('{\"ok\": false, \"stage\": \"../x\", \"frames\": 3}')",
    "open(sys.argv[2], 'w').write('{\"ok\": false, \"stage\": \"build\", \"error\": 1, \"message\": \"m\"}')",
    "open(sys.argv[2], 'wb').write(b'\\xff\\xfe')",
    "open(sys.argv[2], 'w').write('[' * 100000)",
    "os.mkfifo(sys.argv[2])",
    "os.symlink('/etc/passwd', sys.argv[2])",
    "os.mkdir(sys.argv[2])",
])
def test_hostile_reports_are_structured_failures(program, body):
    p = program(f"import os, sys\ndef build():\n    {body}\n    os._exit(0)\n")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("out", run_build(p, timeout_s=5.0)), daemon=True)
    t.start()
    t.join(20)
    out = box.get("out")
    assert out is not None, "run_build hung on the report"
    assert not out.ok and out.stage == "report" and out.error["code"] == "E-SANDBOX-REPORT", out.error


def test_campaign_dirs_are_neither_readable_nor_writable(program, tmp_path_factory):
    state = tmp_path_factory.mktemp("campaign")
    (state / "state.json").write_text("{}")
    p = program(f"def build():\n    return {{'x': open({str(state / 'state.json')!r}).read()}}\n")
    out = run_build(p, deny_read=[state])
    assert out.stage == "build", out
    p = program(f"def build():\n    open({str(state / 'state.json')!r}, 'w').write('forged')\n    return {{}}\n")
    assert run_build(p, deny_read=[state]).stage == "build"
    assert run_build(p).stage == "build"
    assert (state / "state.json").read_text() == "{}"


@pytest.fixture
def home_secret():
    d = Path(tempfile.mkdtemp(prefix=".kiln-sbx-test-", dir=Path.home()))
    try:
        (d / ".env").write_text("OPENAI_API_KEY=not-a-real-key\n")
        yield d / ".env"
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("target", ["home_secret", "known_hosts", "home_listing", "repo_file", "etc"])
def test_reads_outside_the_runtime_are_denied(program, home_secret, target):
    path = {"home_secret": home_secret, "known_hosts": Path.home() / ".ssh" / "known_hosts",
            "home_listing": Path.home(), "repo_file": H100, "etc": Path("/etc/hosts")}[target]
    if target == "known_hosts" and not path.exists():
        pytest.skip("no ~/.ssh/known_hosts on this host")
    read = "os.listdir(p)" if path.is_dir() else "open(p).read()"
    p = program(f"import os\np = {str(path)!r}\ndef build():\n    return {{'leak': {read}}}\n")
    out = run_build(p)
    assert not out.ok and out.stage == "build", (out.design, out.error)
    assert out.error["message"].startswith(("PermissionError", "FileNotFoundError")), out.error
    assert "not-a-real-key" not in json.dumps(out.error)


def test_runtime_and_scratch_stay_usable(program):
    p = program("""
        import collections, copy, json, math, os, random, re
        import kiln
        def build():
            open("scratch.txt", "w").write("ok")
            return {"r": open("scratch.txt").read(), "u": len(os.urandom(4)), "k": kiln.__name__}
    """)
    out = run_build(p)
    assert out.ok and out.design == {"r": "ok", "u": 4, "k": "kiln"}, out.error


def test_secrets_in_program_output_are_redacted(program):
    fake = "sk-proj-" + "Ab1" * 12
    pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAA\n-----END OPENSSH PRIVATE KEY-----"
    p = program(f"""
        import sys
        def build():
            print({fake!r}, file=sys.stderr)
            print({pem!r}, file=sys.stderr)
            raise RuntimeError("key " + {fake!r} + " password=hunter2hunter")
    """)
    out = run_build(p)
    m = evaluate_program(p)
    for text in (out.stderr, out.error["message"], json.dumps(out.error), m["feedback"], json.dumps(m["errors"])):
        assert fake not in text and "b3BlbnNzaC1rZXktdjEAAAA" not in text and "hunter2hunter" not in text, text
    assert "[REDACTED]" in m["feedback"] and "RuntimeError" in m["feedback"]
    assert len(m["feedback"]) <= kiln.evolve.FEEDBACK_MAX_CHARS + 20


def test_redact_keeps_simulator_text():
    from kiln.evolve import redact
    text = "decode_llama3_70b_bs64: 1234.5 tokens_per_s, dominant bound: hbm_bandwidth (E-ENV-0007)"
    assert redact(text) == text
    assert redact("x" * 5000, 100).endswith("[truncated]") and len(redact("x" * 5000, 100)) < 120


def test_exception_reports_line(program):
    p = program("def build():\n    x = 0\n    return 1 / x\n")
    m = evaluate_program(p)
    failed(m, "build", "E-SANDBOX-BUILD")
    assert "ZeroDivisionError" in m["feedback"] and "line 3" in m["feedback"]


def test_syntax_error_and_missing_build(program):
    failed(evaluate_program(program("def build(:\n")), "load", "E-SANDBOX-LOAD")
    m = evaluate_program(program("x = 1\n"))
    failed(m, "build", "E-SANDBOX-BUILD")
    assert "must define build()" in m["feedback"]


def test_bad_json_return_values(program):
    failed(evaluate_program(program("def build():\n    return {'x': float('nan')}\n")), "serialize", "E-SANDBOX-SERIALIZE")
    failed(evaluate_program(program("def build():\n    return [1, 2]\n")), "serialize", "E-SANDBOX-SERIALIZE")


def test_hard_crash(program):
    p = program("import os\ndef build():\n    os._exit(3)\n")
    failed(evaluate_program(p), "crash", "E-SANDBOX-CRASH")
    p = program("import ctypes\ndef build():\n    return ctypes.string_at(0)\n")
    failed(evaluate_program(p), "crash", "E-SANDBOX-CRASH")


def test_memory_limit(program):
    p = program("def build():\n    x = bytearray(6 << 30)\n    x[::4096] = b'x' * len(x[::4096])\n    return {}\n")
    m = evaluate_program(p, memory_mb=256, timeout_s=20.0)
    assert m["combined_score"] == 0.0
    assert m["stage"] in ("memory", "crash", "build"), m
    assert m["errors"][0]["code"] in ("E-SANDBOX-MEMORY", "E-SANDBOX-CRASH", "E-SANDBOX-BUILD")


def test_network_is_blocked(program):
    p = program("""
        import socket
        def build():
            socket.create_connection(("1.1.1.1", 53), timeout=2)
            return {}
    """)
    m = evaluate_program(p)
    failed(m, "build", "E-SANDBOX-BUILD")
    assert "network access is disabled" in m["feedback"]


def test_raw_socket_and_writes_denied(program, tmp_path):
    target = tmp_path.parent / "kiln_sandbox_escape.txt"
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        p = program(f"""
            import _socket
            def build():
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect(("127.0.0.1", {port}))
                return {{}}
        """)
        failed(evaluate_program(p), "build", "E-SANDBOX-BUILD")
    p = program(f"def build():\n    open({str(target)!r}, 'w').write('x')\n    return {{}}\n")
    m = evaluate_program(p)
    failed(m, "build", "E-SANDBOX-BUILD")
    assert not target.exists()


def test_env_is_scrubbed(program, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    p = program("import os\ndef build():\n    raise RuntimeError(os.environ.get('OPENAI_API_KEY', 'absent'))\n")
    m = evaluate_program(p)
    assert "RuntimeError: absent" in m["feedback"]


def test_missing_program_and_options_error(tmp_path, program, session):
    failed(evaluate_program(tmp_path / "nope.py"), "load", "E-SANDBOX-0001")
    p = program("def build():\n    return {}\n")
    m = evaluate_program(p, fitness={"kind": "bogus"}, session=session)
    failed(m, "options", "E-SANDBOX-OPTIONS")


class FakeResult:
    def __init__(self, d):
        self.d = d

    def to_dict(self):
        return self.d

    def explain(self):
        return "status: ok\n"


class FakeSession:
    def __init__(self):
        self.calls = []

    def evaluate(self, design, workload, options):
        self.calls.append((design, workload, options))
        return FakeResult({"status": "ok", "score": 1.25, "stage_reached": "S3", "errors": [], "violations": [],
                           "features": {"die_mm2_total": 10.0, "energy_split": [1.0, 0.0, 0.0]}})


def test_success_metrics_shape(program):
    s = FakeSession()
    m = evaluate_program(program("def build():\n    return {'schema': 'kiln.hw/1.0'}\n"), suite="standard",
                         baseline="tpu_v5e", fitness="perf_per_watt", session=s)
    assert m["combined_score"] == 1.25 and m["valid"] == 1.0
    assert m["die_mm2_total"] == 0.0 and m["energy_split_compute"] == 1.0
    design, workload, options = s.calls[0]
    assert design == {"schema": "kiln.hw/1.0"} and workload == "standard"
    assert options["fitness"] == {"kind": "perf_per_watt", "baseline": "tpu_v5e"}


def test_string_hashing_is_deterministic_across_builds(program):
    p = program("def build():\n    return {'order': list({f'k{i}' for i in range(32)})}\n")
    orders = {tuple(run_build(p).design["order"]) for _ in range(3)}
    assert len(orders) == 1
