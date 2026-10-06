import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kiln_evo import config as C
from kiln_evo import llm as L
from kiln_evo import programs as P

A100 = P.materialize_seed((C.PKG / "seeds" / "a100.py").read_text(), C.REFERENCE_DESIGNS)


def _llm_cfg(**over):
    return C._merge(C.DEFAULTS["llm"], over)


def test_spend_reserves_worst_case_and_stops():
    s = L.Spend({"input": 2.0, "output": 10.0}, max_usd=0.05)
    r = s.reserve(3000, 1000)  # <= 3000 + 256 in + 1000 out tokens: $0.0165
    assert r.usd == pytest.approx(0.016512)
    s.settle(r, 500, 200)
    assert s.spent_usd == pytest.approx(0.003) and s.calls == 1
    s.reserve(3000, 1000)
    s.reserve(3000, 1000)
    with pytest.raises(L.BudgetExceeded):
        s.reserve(3000, 1000)
    assert L.Spend({}, None).reserve(10**9, 10**6).usd == 0.0


def test_mock_is_deterministic_and_parsable():
    spend = L.Spend({"input": 1.0, "output": 1.0}, None)
    m = L.make_backend(_llm_cfg(backend="mock", mock={"seed": 3, "failure_rate": 0.0, "full_rewrite_rate": 0.5}),
                       spend)
    user = f"<parent_program>\n```python\n{A100}```\n</parent_program>\n"
    outs = [m.complete("sys", user + f"variant {i}") for i in range(12)]
    again = m.complete("sys", user + "variant 0")
    assert again.text == outs[0].text
    kinds = set()
    for c in outs:
        child, hyp, kind = P.apply_response(A100, c.text)
        kinds.add(kind)
        assert hyp and child != A100
    assert kinds == {"diff", "full"}
    assert spend.calls == 13 and spend.input_tokens > 0


def test_mock_failures_are_structured():
    m = L.make_backend(_llm_cfg(backend="mock", mock={"seed": 1, "failure_rate": 1.0}), L.Spend({}, None))
    user = f"<parent_program>\n```python\n{A100}```\n</parent_program>\n"
    codes = set()
    for i in range(30):
        try:
            P.apply_response(A100, m.complete("s", user + str(i)).text)
        except P.ProgramError as e:
            codes.add(e.code)
    assert {"E-EVO-HYPOTHESIS", "E-EVO-DIFF", "E-EVO-SYNTAX"} <= codes


class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Handler.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/chat/completions"):
            out = {"choices": [{"message": {"content": "HYPOTHESIS: x"}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
        else:
            out = {"content": [{"type": "text", "text": "HYPOTHESIS: y"}],
                   "usage": {"input_tokens": 2000, "output_tokens": 50, "cache_read_input_tokens": 10}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = json.dumps({"data": [{"id": "model-b"}, {"id": "model-a"}]}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.seen = []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_and_anthropic_wire_format(server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    price = {"input": 3.0, "output": 15.0}
    spend = L.Spend(price, 1.0)
    oa = L.make_backend(_llm_cfg(backend="openai", model="m-from-catalog", base_url=server, api_key_env="TEST_KEY",
                                 price_usd_per_mtok=price), spend)
    c = oa.complete("system text", "user text")
    assert c.text == "HYPOTHESIS: x" and c.cost_usd == pytest.approx((1000 * 3 + 100 * 15) / 1e6)
    path, headers, body = _Handler.seen[-1]
    assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "m-from-catalog" and body["messages"][0] == {"role": "system", "content": "system text"}
    an = L.make_backend(_llm_cfg(backend="anthropic", model="m2", base_url=server, api_key_env="TEST_KEY",
                                 price_usd_per_mtok=price), spend)
    c = an.complete("sys", "usr")
    assert c.text == "HYPOTHESIS: y" and c.input_tokens == 2010
    path, headers, body = _Handler.seen[-1]
    assert path == "/v1/messages" and headers["x-api-key"] == "sk-test" and "anthropic-version" in headers
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "usr"}]
    assert spend.calls == 2
    assert oa.list_models() == ["model-a", "model-b"]


def test_missing_key_is_an_llm_error(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    b = L.make_backend(_llm_cfg(backend="anthropic", model="m", api_key_env="NO_SUCH_KEY", retries=0,
                                price_usd_per_mtok={"input": 1, "output": 1}), L.Spend({"input": 1, "output": 1}, 1))
    with pytest.raises(L.LLMError, match="NO_SUCH_KEY"):
        b.complete("s", "u")
    assert b.spend.calls == 0 and b.spend._reserved == 0


def test_config_refuses_placeholders_for_real_backends(tmp_path):
    base = {"llm": {"backend": "anthropic", "model": "FILL_ME", "api_key_env": "K"}}
    with pytest.raises(C.ConfigError) as e:
        C.finalize(base, tmp_path)
    msg = str(e.value)
    assert "live catalog" in msg and "pricing page" in msg and "max_usd" in msg
    ok = C._merge(base, {"llm": {"model": "x", "price_usd_per_mtok": {"input": 1, "output": 2}},
                         "budget": {"max_usd": 3}})
    C.finalize(ok, tmp_path)
    # A campaign without LLM operators needs no model or prices.
    C.finalize(C._merge(base, {"operators": {"parametric": 1.0, "llm_diff": 0, "llm_full": 0}}), tmp_path)
    with pytest.raises(C.ConfigError, match="heldout"):
        C.finalize({"workloads": {"heldout": ["llama3_8b:decode_b8"]}}, tmp_path)


class _Flaky(L.Backend):
    """Fails `fails` times with `exc` after the request may have reached the provider, then succeeds."""

    def __init__(self, cfg, spend, exc, fails):
        super().__init__(cfg, spend)
        self.exc, self.fails = exc, fails

    def _call(self, system, user):
        if self.fails:
            self.fails -= 1
            raise self.exc
        return "HYPOTHESIS: ok", 0, 100_000


def test_ambiguous_failed_attempts_are_charged_at_their_reservation(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    price = {"input": 0.0, "output": 1.0}
    cfg = _llm_cfg(backend="openai", max_tokens=100_000, retries=1)
    spend = L.Spend(price, 1.0)
    c = _Flaky(cfg, spend, TimeoutError("read timed out"), fails=1).complete("s", "u")
    assert c.cost_usd == pytest.approx(0.1)
    assert spend.spent_usd == pytest.approx(0.2) and spend._reserved == 0
    spend = L.Spend(price, 1.0)
    with pytest.raises(L.LLMError):
        _Flaky(cfg, spend, TimeoutError("read timed out"), fails=5).complete("s", "u")
    assert spend.spent_usd == pytest.approx(0.2) and spend._reserved == 0
    spend = L.Spend(price, 1.0)
    refused = L.urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    with pytest.raises(L.LLMError):
        _Flaky(cfg, spend, refused, fails=5).complete("s", "u")
    assert spend.spent_usd == 0.0 and spend._reserved == 0


def test_multiple_completions_cannot_bypass_the_reservation(tmp_path):
    cfg = _llm_cfg(backend="openai", max_tokens=100_000, extra_body={"n": 2})
    with pytest.raises(L.LLMError, match="n"):
        L.make_backend(cfg, L.Spend({"input": 0.0, "output": 1.0}, 0.1)).output_limit()
    base = {"llm": {"backend": "openai", "model": "m", "base_url": "http://x/v1", "api_key_env": "K",
                    "max_tokens": 100_000, "extra_body": {"n": 2}, "price_usd_per_mtok": {"input": 0, "output": 1}},
            "budget": {"max_usd": 0.1}}
    with pytest.raises(C.ConfigError, match="extra_body.n"):
        C.finalize(base, tmp_path)


@pytest.mark.parametrize("backend,response", [
    ("openai", {"choices": [{"message": {"content": "valid"}}]}),
    ("openai", {"choices": [{"message": {"content": "valid"}}], "usage": {"prompt_tokens": 10}}),
    ("openai", {"choices": [{"message": {"content": "valid"}}],
                "usage": {"prompt_tokens": -5, "completion_tokens": "7"}}),
    ("anthropic", {"content": [{"type": "text", "text": "valid"}]}),
    ("anthropic", {"content": [{"type": "text", "text": "valid"}],
                   "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": -1}}),
])
def test_missing_usage_is_charged_at_the_reservation(monkeypatch, backend, response):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    monkeypatch.setattr(L, "_post", lambda *a, **k: response)
    price = {"input": 1.0, "output": 1.0}
    spend = L.Spend(price, 1.0)
    b = L.make_backend(_llm_cfg(backend=backend, model="m", base_url="http://x/v1", api_key_env="TEST_KEY",
                                max_tokens=1000, price_usd_per_mtok=price), spend)
    c = b.complete("s", "u")
    worst = spend.cost(2 + L.PROMPT_OVERHEAD_TOKENS + 2, 1000)
    assert worst > 0 and c.text == "valid" and c.cost_usd == pytest.approx(worst)
    assert spend.spent_usd == pytest.approx(worst) and spend._reserved == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_non_finite_prices_and_limits_are_rejected(tmp_path, bad):
    base = {"llm": {"backend": "openai", "model": "m", "base_url": "http://x/v1", "api_key_env": "K",
                    "price_usd_per_mtok": {"input": bad, "output": 1}}, "budget": {"max_usd": 0.001}}
    with pytest.raises(C.ConfigError, match="price_usd_per_mtok.input"):
        C.finalize(base, tmp_path)
    base = C._merge(base, {"llm": {"price_usd_per_mtok": {"input": 1}}, "budget": {"max_usd": bad}})
    with pytest.raises(C.ConfigError, match="max_usd"):
        C.finalize(base, tmp_path)
    with pytest.raises(ValueError):
        L.Spend({"input": bad, "output": 1.0}, 1.0)
    with pytest.raises(ValueError):
        L.Spend({"input": 1.0, "output": 1.0}, bad)


def test_torn_ledger_tail_does_not_swallow_the_next_reservation(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    s = L.Spend({"input": 0.0, "output": 1.0}, 10.0, ledger=ledger)
    s.settle(s.reserve(0, 100_000), 0, 100_000)
    with open(ledger, "a") as f:
        f.write('{"op":')
    s = L.Spend({"input": 0.0, "output": 1.0}, 10.0, ledger=ledger)
    assert s.spent_usd == pytest.approx(0.1)
    s.reserve(0, 500_000)  # $0.50 in flight, then the process dies
    assert L.Spend({"input": 0.0, "output": 1.0}, 10.0, ledger=ledger).spent_usd == pytest.approx(0.6)
