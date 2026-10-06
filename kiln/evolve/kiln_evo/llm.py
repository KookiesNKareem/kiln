"""LLM backends: OpenAI-compatible chat completions, Anthropic Messages, and a deterministic mock.

No model id or price lives in this file: both come from the campaign config, which the owner fills from the
provider's live model catalog (`python -m kiln_evo models <campaign>` queries it) and pricing page. Every call is
priced from the provider-reported token usage, and a call whose worst case (prompt bytes + framing + the effective
output limit) would cross `budget.max_usd` is refused before it is sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import programs


PROMPT_OVERHEAD_TOKENS = 256
OUTPUT_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    latency_s: float


@dataclass
class Reservation:
    id: int
    usd: float


class Spend:
    """Token and dollar accounting with a hard stop. With a `ledger` path every reservation and settlement is
    appended (and fsynced) before the call proceeds; a reloaded ledger restores the totals, charging a reservation
    that was never settled (the process died with the call in flight) at its worst case."""

    def __init__(self, price: dict, max_usd: float | None, spent_usd: float = 0.0, input_tokens: int = 0,
                 output_tokens: int = 0, calls: int = 0, ledger: str | os.PathLike | None = None):
        self.price_in = float(price.get("input") or 0.0)
        self.price_out = float(price.get("output") or 0.0)
        self.max_usd = max_usd
        self.spent_usd, self.input_tokens, self.output_tokens, self.calls = spent_usd, input_tokens, output_tokens, calls
        self._reserved = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._ledger = Path(ledger) if ledger else None
        if self._ledger is None:
            return
        if self._ledger.exists():
            self._replay()
        elif spent_usd or input_tokens or output_tokens or calls:
            self._log({"op": "open", **self.state()})

    def _log(self, rec: dict) -> None:
        if self._ledger is None:
            return
        with open(self._ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _replay(self) -> None:
        self.spent_usd, self.input_tokens, self.output_tokens, self.calls = 0.0, 0, 0, 0
        open_ids: dict[int, float] = {}
        for line in self._ledger.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a torn final line from a crash mid-write
            op = r.get("op")
            if op == "open":
                self.spent_usd += r["spent_usd"]
                self.input_tokens += r["input_tokens"]
                self.output_tokens += r["output_tokens"]
                self.calls += r["calls"]
            elif op == "reserve":
                open_ids[r["id"]] = r["usd"]
                self._seq = max(self._seq, r["id"] + 1)
            elif op == "settle":
                open_ids.pop(r["id"], None)
                self.spent_usd += r["usd"]
                self.input_tokens += r["in"]
                self.output_tokens += r["out"]
                self.calls += int(r["called"])
        for rid, usd in open_ids.items():
            self.spent_usd += usd
            self.calls += 1
            self._log({"op": "settle", "id": rid, "usd": usd, "in": 0, "out": 0, "called": True, "orphan": True})

    def cost(self, tin: int, tout: int) -> float:
        return (tin * self.price_in + tout * self.price_out) / 1e6

    def reserve(self, prompt_bytes: int, max_tokens: int) -> Reservation:
        """Reserves the call's worst case or raises BudgetExceeded. A byte-level tokenizer never emits more tokens
        than UTF-8 bytes, so the prompt is bounded by its byte count plus the message framing."""
        worst = self.cost(prompt_bytes + PROMPT_OVERHEAD_TOKENS, max_tokens)
        with self._lock:
            if self.max_usd is not None and self.spent_usd + self._reserved + worst > self.max_usd:
                raise BudgetExceeded(f"LLM budget: spent ${self.spent_usd:.4f} + in flight ${self._reserved:.4f} + "
                                     f"worst case ${worst:.4f} for the next call exceeds max_usd "
                                     f"${self.max_usd:.2f}")
            res = Reservation(self._seq, worst)
            self._seq += 1
            self._log({"op": "reserve", "id": res.id, "usd": worst})
            self._reserved += worst
        return res

    def settle(self, res: Reservation, tin: int = 0, tout: int = 0, called: bool = True) -> float:
        c = self.cost(tin, tout)
        with self._lock:
            self._log({"op": "settle", "id": res.id, "usd": c, "in": tin, "out": tout, "called": called})
            self._reserved -= res.usd
            self.spent_usd += c
            self.input_tokens += tin
            self.output_tokens += tout
            self.calls += int(called)
        return c

    def settle_worst(self, res: Reservation) -> float:
        """Settles a call whose outcome is unknown (it may have been billed) at its reserved worst case."""
        with self._lock:
            self._log({"op": "settle", "id": res.id, "usd": res.usd, "in": 0, "out": 0, "called": True,
                       "ambiguous": True})
            self._reserved -= res.usd
            self.spent_usd += res.usd
            self.calls += 1
        return res.usd

    def exhausted(self) -> bool:
        return self.max_usd is not None and self.spent_usd >= self.max_usd

    def state(self) -> dict:
        return {"spent_usd": self.spent_usd, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "calls": self.calls}


def _unbilled(e: BaseException) -> bool:
    """True when the failed request provably was not billed: rejected with a 4xx, never connected, or failed before
    sending (a missing API key)."""
    if isinstance(e, urllib.error.HTTPError):
        return 400 <= e.code < 500
    if isinstance(e, urllib.error.URLError):
        return isinstance(e.reason, (ConnectionRefusedError, socket.gaierror))
    return isinstance(e, LLMError) and getattr(e, "unsent", False)


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json",
                                                                                  **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, headers: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Backend:
    def __init__(self, cfg: dict, spend: Spend):
        self.cfg, self.spend = cfg, spend
        self.model = cfg.get("model") or "mock"

    def output_limit(self) -> int:
        """The request's effective output-token cap: `max_tokens`, or a larger limit `extra_body` sends."""
        extra = self.cfg.get("extra_body") or {}
        limits = [self.cfg["max_tokens"], *(extra[k] for k in OUTPUT_LIMIT_KEYS if k in extra)]
        if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in limits):
            raise LLMError(f"output token limits {limits} must be positive integers: an unbounded output cannot be "
                           "reserved against the dollar budget")
        if extra.get("n", 1) != 1:
            raise LLMError(f"llm.extra_body.n = {extra['n']!r}: only one completion per call is used and reserved "
                           "against the dollar budget")
        return max(limits)

    def complete(self, system: str, user: str) -> Completion:
        """Each attempt reserves its own worst case. An attempt that fails after the request may have reached the
        provider (timeout, dropped connection, 5xx, unusable response) is charged at that reservation, since the
        provider may have billed it; only a refusal that provably was not billed (no connection, 4xx) is free."""
        extra = json.dumps(self.cfg.get("extra_body") or {})
        prompt = len(system.encode()) + len(user.encode()) + len(extra.encode())
        limit = self.output_limit()
        last = None
        for attempt in range(self.cfg.get("retries", 2) + 1):
            reserved = self.spend.reserve(prompt, limit)
            t0 = time.monotonic()
            try:
                text, tin, tout = self._call(system, user)
            except (urllib.error.URLError, TimeoutError, LLMError, KeyError, ValueError) as e:
                last = e
                if _unbilled(e):
                    self.spend.settle(reserved, called=False)
                else:
                    self.spend.settle_worst(reserved)
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404):
                    break
                if self.cfg.get("retries", 2):
                    time.sleep(min(30, 2 ** attempt))
                continue
            except BaseException:
                self.spend.settle_worst(reserved)
                raise
            cost = self.spend.settle(reserved, tin, tout)
            return Completion(text, tin, tout, cost, self.model, time.monotonic() - t0)
        detail = last.read().decode(errors="replace")[:500] if isinstance(last, urllib.error.HTTPError) else ""
        raise LLMError(f"{type(last).__name__}: {last} {detail}".strip())

    def _key(self) -> str:
        env = self.cfg.get("api_key_env")
        key = os.environ.get(env or "")
        if not key:
            err = LLMError(f"environment variable {env!r} (llm.api_key_env) is not set")
            err.unsent = True
            raise err
        return key

    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError


class OpenAIBackend(Backend):
    """Any OpenAI-compatible `/chat/completions` endpoint (`llm.base_url` ends in /v1 or equivalent)."""

    def _call(self, system, user):
        body = {"model": self.cfg["model"], "messages": [{"role": "system", "content": system},
                                                          {"role": "user", "content": user}],
                "max_tokens": self.cfg["max_tokens"]}
        if self.cfg.get("temperature") is not None:
            body["temperature"] = self.cfg["temperature"]
        body.update(self.cfg.get("extra_body") or {})
        r = _post(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                  {"authorization": f"Bearer {self._key()}"}, body, self.cfg["timeout_s"])
        if "choices" not in r:
            raise LLMError(f"no choices in response: {json.dumps(r)[:300]}")
        text = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage") or {}
        return text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    def list_models(self):
        r = _get(self.cfg["base_url"].rstrip("/") + "/models", {"authorization": f"Bearer {self._key()}"})
        return sorted(m["id"] for m in r.get("data", []))


class AnthropicBackend(Backend):
    """Anthropic Messages API (`llm.base_url` defaults to https://api.anthropic.com/v1)."""

    def _base(self):
        return (self.cfg.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")

    def _headers(self):
        return {"x-api-key": self._key(), "anthropic-version": self.cfg["anthropic_version"]}

    def _call(self, system, user):
        body = {"model": self.cfg["model"], "system": system, "max_tokens": self.cfg["max_tokens"],
                "messages": [{"role": "user", "content": user}]}
        if self.cfg.get("temperature") is not None:
            body["temperature"] = self.cfg["temperature"]
        body.update(self.cfg.get("extra_body") or {})
        r = _post(self._base() + "/messages", self._headers(), body, self.cfg["timeout_s"])
        if "content" not in r:
            raise LLMError(f"no content in response: {json.dumps(r)[:300]}")
        text = "".join(b.get("text", "") for b in r["content"] if b.get("type") == "text")
        u = r.get("usage") or {}
        tin = int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0) or 0) \
            + int(u.get("cache_read_input_tokens", 0) or 0)
        return text, tin, int(u.get("output_tokens", 0))

    def list_models(self):
        r = _get(self._base() + "/models?limit=1000", self._headers())
        return sorted(m["id"] for m in r.get("data", []))


PROGRAM_RE = re.compile(r"<parent_program>\n```python\n(.*?)```\n</parent_program>", re.DOTALL)
NUM_LINE_RE = re.compile(r"""^(\s*["'][A-Za-z0-9_]+["']\s*:\s*)(-?\d+(?:\.\d+)?)(,?\s*)$""")
SIZE_RE = re.compile(r"""(["']capacity["']\s*:\s*["'])(\d+)(KiB|MiB|GiB)(["'])""")
WIDTH_RE = re.compile(r"""(["']width_bits["']\s*:\s*)(\d+)""")


class MockBackend(Backend):
    """Deterministic stand-in for an LLM: reads the parent program from the prompt and answers with a scripted
    or random structural edit (a PARAMS value, a capacity, or a link width), as a SEARCH/REPLACE diff or a full
    rewrite, with a HYPOTHESIS line. A fraction of answers is deliberately broken (no hypothesis, a SEARCH that
    does not match, a syntax error, or a build() that raises) to exercise the failure paths."""

    def __init__(self, cfg, spend):
        super().__init__(cfg, spend)
        self.mcfg = cfg.get("mock") or {}
        self.model = "mock"

    def _rng(self, user: str) -> random.Random:
        h = hashlib.sha256(f"{self.mcfg.get('seed', 0)}|{user}".encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def _edit(self, code: str, rng: random.Random) -> tuple[str, str, str] | None:
        lines = code.splitlines(keepends=True)
        cands = []
        params = programs._assign(code, "PARAMS")
        if params is not None:
            for i in range(params.lineno, params.end_lineno - 1):
                if NUM_LINE_RE.match(lines[i].rstrip("\n")):
                    cands.append(("param", i))
        for i, ln in enumerate(lines):
            if SIZE_RE.search(ln):
                cands.append(("size", i))
            elif WIDTH_RE.search(ln):
                cands.append(("width", i))
        rng.shuffle(cands)
        for kind, i in cands:
            out = self._edit_line(lines[i], kind, rng)
            if out:
                return out
        return None

    @staticmethod
    def _edit_line(old: str, kind: str, rng: random.Random) -> tuple[str, str, str] | None:
        factor = rng.choice([0.5, 2.0]) if kind != "param" else rng.choice([0.5, 0.75, 1.5, 2.0])
        if kind == "param":
            m = NUM_LINE_RE.match(old.rstrip("\n"))
            v = m.group(2)
            nv = float(v) * factor
            nv_s = str(max(1, int(round(nv)))) if "." not in v else f"{nv:.4g}"
            new = f"{m.group(1)}{nv_s}{m.group(3)}\n"
            what = f"scale {m.group(1).strip().rstrip(':').strip()} {v} -> {nv_s}"
        elif kind == "size":
            m = SIZE_RE.search(old)
            nv = max(1, int(int(m.group(2)) * factor))
            new = old[:m.start(2)] + str(nv) + old[m.end(2):]
            what = f"capacity {m.group(2)}{m.group(3)} -> {nv}{m.group(3)}"
        else:
            m = WIDTH_RE.search(old)
            nv = max(8, int(int(m.group(2)) * factor))
            new = old[:m.start(2)] + str(nv) + old[m.end(2):]
            what = f"width_bits {m.group(2)} -> {nv}"
        if new == old:
            return None
        return old, new, what

    def _call(self, system, user):
        rng = self._rng(user)
        m = PROGRAM_RE.search(user)
        code = m.group(1) if m else ""
        edit = self._edit(code, rng) if code else None
        fail = rng.random() < self.mcfg.get("failure_rate", 0.1)
        if edit is None:
            text = "HYPOTHESIS: nothing to change\nI could not find an editable value."
        else:
            old, new, what = edit
            hyp = f"HYPOTHESIS: {what} (mock edit) to move the bottleneck"
            mode = rng.choice(["no_hypothesis", "bad_search", "syntax", "raise"]) if fail else None
            if mode == "no_hypothesis":
                hyp = "Here is my change."
            if rng.random() < self.mcfg.get("full_rewrite_rate", 0.2):
                child = code.replace(old, new, 1)
                if mode == "syntax":
                    child += "\ndef broken(:\n"
                elif mode == "raise":
                    child = child.replace("def build():", "def build():\n    raise RuntimeError('mock failure')", 1)
                text = f"{hyp}\n```python\n{child}```\n"
            else:
                search = old.rstrip("\n")
                if mode == "bad_search":
                    search = search + "  # not in parent"
                replace = new.rstrip("\n")
                if mode == "syntax":
                    replace = replace + " ((("
                elif mode == "raise":
                    replace = replace + "\nraise RuntimeError('mock failure at import')"
                text = f"{hyp}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"
        tpc = self.mcfg.get("tokens_per_char", 0.25)
        return text, int((len(system) + len(user)) * tpc), int(len(text) * tpc)

    def list_models(self):
        return ["mock"]


def make_backend(cfg: dict, spend: Spend) -> Backend:
    kind = cfg["backend"]
    if kind == "mock":
        return MockBackend(cfg, spend)
    if kind == "openai":
        return OpenAIBackend(cfg, spend)
    if kind == "anthropic":
        return AnthropicBackend(cfg, spend)
    raise ValueError(f"unknown llm backend {kind!r}")
