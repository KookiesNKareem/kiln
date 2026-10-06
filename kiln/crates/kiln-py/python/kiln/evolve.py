"""Evolution-loop adapter (spec 06 §6.6 sandboxing): run an LLM-written `build()` program in a sandboxed
subprocess, evaluate the design with kiln, and return OpenEvolve-style metrics.

Sandbox: separate `python -s -P` process in a fresh scratch dir with a scrubbed environment, CPU/file-size/core/
open-file/memory rlimits (RSS polling on macOS, where RLIMIT_AS is not enforced), a wall-clock timeout, bounded
stderr capture, and OS isolation that denies network access, writes outside the scratch dir, escaping descendants,
and every file read except the Python runtime (prefixes, stdlib, site-packages, the kiln package), system libraries,
a few devices and the scratch dir (so nothing under $HOME, no keys, no .env files), plus the caller's `deny_read` dirs:
`bwrap` or `unshare` user, mount, net and pid namespaces over a root holding only those binds on Linux, a
`sandbox-exec` profile on macOS (which also denies fork, exec and Mach lookups). The isolation is probed once per
process; without it no program runs (E-SANDBOX-UNAVAILABLE). Program-controlled text that becomes LLM feedback is
redacted (`redact`) and capped. kiln itself only ever receives the design as JSON data.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import resource
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import kiln

RUNNER = Path(__file__).with_name("_sandbox_runner.py")
MOUNTS = Path(__file__).with_name("_sandbox_mounts.py")
FEEDBACK_MAX_CHARS = 2000
_STDERR_TAIL = 800
_STDERR_KEEP = 64 << 10
_REPORT_MAX = 64 << 20
_DRAIN_S = 0.5
# Not `-I`: it implies `-E`, which would ignore PYTHONHASHSEED (string-set order would vary between builds). The
# environment is scrubbed instead; `-s` drops the user site and `-P` (3.11+) the script dir (the runner strips it
# itself on older Pythons).
PY_FLAGS = ("-s", "-P") if sys.version_info >= (3, 11) else ("-s",)
_STAGES = ("load", "build", "serialize")
_MESSAGE_MAX = 1000
REDACTED = "[REDACTED]"
_SECRET_PATTERNS = [re.compile(p, f) for p, f in (
    (r"-----BEGIN [A-Z0-9 ]*-----.*?(?:-----END [A-Z0-9 ]*-----|\Z)", re.S),
    (r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}", 0),
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0),
    (r"\b(?:gh[pousr]_|github_pat_|glpat-|hf_|gsk_|xai-|npm_|pypi-|sk_live_|sk_test_)[A-Za-z0-9_\-]{16,}", 0),
    (r"\bxox[abprs]-[A-Za-z0-9\-]{10,}", 0),
    (r"\bAIza[0-9A-Za-z_\-]{30,}", 0),
    (r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", 0),
    (r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/\-]{12,}=*", 0),
    (r"(?i)(\b[a-z0-9_.\-]*(?:api[_\-]?key|secret|token(?!s)|passw(?:or)?d|private[_\-]?key|credential)[a-z0-9_\-]*"
     r"[\"']?\s*[:=]\s*[\"']?)(?=[^\s\"',;}]*[A-Za-z])[^\s\"',;}]{6,}", 0),
)]
_OPAQUE = re.compile(r"[A-Za-z0-9+/_\-]{32,}={0,2}")


def _opaque(m: re.Match) -> str:
    t = m.group(0)
    mixed = any(c.isupper() for c in t) and any(c.islower() for c in t) and any(c.isdigit() for c in t)
    hexy = len(t) >= 40 and all(c in "0123456789abcdefABCDEF" for c in t)
    return REDACTED if mixed or hexy else t


def redact(text: str, limit: int | None = None) -> str:
    """Strips anything that looks like a credential (API keys, tokens, PEM blocks, `password=...`, long opaque
    base64/hex strings) from text that is about to reach an LLM provider, then caps it at `limit` characters."""
    if not text:
        return text
    for p in _SECRET_PATTERNS:
        text = p.sub(lambda m: (m.group(1) if m.lastindex else "") + REDACTED, text)
    text = _OPAQUE.sub(_opaque, text)
    return text if limit is None or len(text) <= limit else text[:limit] + "...[truncated]"


@dataclass
class BuildOutcome:
    design: dict | str | None = None
    stage: str = "ok"
    error: dict | None = None
    stderr: str = ""
    wall_s: float = 0.0
    sandbox: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.design is not None


def _error(code: str, message: str, hint: str, **extra) -> dict:
    return {"code": code, "severity": "error", "message": message, "path": "program", "hint": hint,
            "section": "06 §6.6", **extra}


def _sb_str(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


_SYSTEM_READ = ("/usr/lib", "/usr/share", "/System", "/private/var/db/dyld", "/Library/Apple/System")
_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")


def _home() -> str | None:
    h = os.path.expanduser("~")
    return os.path.realpath(h) if h and h != "~" and os.path.isabs(h) else None


def _contains(parent: str, child: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


@functools.cache
def _runtime_roots() -> tuple[str, ...]:
    """Directories the sandboxed interpreter must read: its prefixes, the stdlib and site-packages of a
    `python -s -P` child, and the kiln package. A root that is $HOME (or contains it) or `/` is dropped, so a
    runtime that lives under $HOME only exposes its own subtree."""
    try:
        out = subprocess.run([sys.executable, *PY_FLAGS, "-c", "import json, sys; print(json.dumps(sys.path))"],
                             env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, capture_output=True, text=True,
                             timeout=30, check=True).stdout
        child_path = [p for p in json.loads(out) if isinstance(p, str)]
    except (OSError, ValueError, subprocess.SubprocessError):
        child_path = []
    cands = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))),
             os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))),
             os.path.dirname(os.path.realpath(kiln.__file__)), *child_path]
    home = _home()
    roots = set()
    for c in cands:
        if not c or not os.path.isabs(c):
            continue
        while c != "/" and not os.path.exists(c):  # python311.zip and the like: allow the would-be parent
            c = os.path.dirname(c)
        for p in {os.path.abspath(c), os.path.realpath(c)}:
            if p == "/" or (home and _contains(p, home)):
                continue
            roots.add(p)
    return tuple(sorted(r for r in roots if not any(o != r and _contains(o, r) for o in roots)))


def _ancestors(paths) -> list[str]:
    out = set()
    for p in paths:
        while p not in ("/", ""):
            p = os.path.dirname(p)
            out.add(p)
    return sorted(out)


def _seatbelt_profile(tmp: str, deny_read: list[str]) -> str:
    """Default-deny for reads: only the Python runtime, system libraries, a few devices and the scratch dir are
    readable, so a program cannot read secrets ($HOME, .env files, keys) and leak them through its error text.
    Only the ancestors of the readable roots may be stat'ed (path resolution needs it); their listings are denied."""
    exe = os.path.realpath(sys.executable)
    roots = [*_runtime_roots(), *(r for r in _SYSTEM_READ if os.path.exists(r)), tmp]
    sub = " ".join(f"(subpath {_sb_str(r)})" for r in roots)
    lit = " ".join(f"(literal {_sb_str(d)})" for d in _DEVICES)
    anc = " ".join(f"(literal {_sb_str(a)})" for a in _ancestors([*roots, exe, *_DEVICES]))
    prof = ("(version 1)(allow default)(deny network*)(deny process-fork)(deny mach-lookup)(deny process-exec*)"
            f"(allow process-exec* (literal {_sb_str(exe)}) (subpath {_sb_str(sys.base_prefix)}))"
            f'(deny file-read*)(allow file-read* (literal "/") {sub} {lit})(allow file-read-metadata {anc})'
            f'(deny file-write*)(allow file-write* (subpath {_sb_str(tmp)}) (literal "/dev/null"))')
    if deny_read:
        prof += "(deny file-read* " + " ".join(f"(subpath {_sb_str(d)})" for d in deny_read) + ")"
    return prof


def _wrap_seatbelt(tmp: str, deny_read: list[str]) -> list[str]:
    return ["sandbox-exec", "-p", _seatbelt_profile(tmp, deny_read)]


_LINUX_SYSTEM = ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc/ld.so.cache", "/etc/ld.so.conf",
                 "/etc/ld.so.conf.d", "/etc/localtime")


def _linux_binds() -> list[str]:
    """Read-only bind sources for the Linux sandbox: system library dirs and the Python runtime, never $HOME."""
    return [p for p in (*_LINUX_SYSTEM, *_runtime_roots()) if os.path.lexists(p)]


def _wrap_bwrap(tmp: str, deny_read: list[str]) -> list[str]:
    binds = []
    for p in _linux_binds():
        if os.path.islink(p) and os.path.dirname(p) == "/":  # merged /usr: /lib -> usr/lib
            binds += ["--symlink", os.readlink(p), p]
        else:
            binds += ["--ro-bind", p, p]
    visible = [*_linux_binds(), tmp]
    hide = [a for d in deny_read if any(_contains(v, d) for v in visible)
            for a in ("--perms", "0000", "--tmpfs", d, "--remount-ro", d)]
    return ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", *binds, "--dev", "/dev", "--proc",
            "/proc", "--bind", tmp, tmp, *hide, "--remount-ro", "/", "--chdir", tmp, "--"]


def _wrap_unshare(tmp: str, deny_read: list[str]) -> list[str]:
    return ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child",
            "--mount-proc", sys.executable, "-I", str(MOUNTS), tmp, json.dumps(_linux_binds()),
            json.dumps(deny_read), "--"]


_PROBE = r"""
import json, os, socket, sys
scratch, outside, secret, platform, port, home = sys.argv[1:7]
res = {}
def attempt(key, f):
    try:
        f()
        res[key] = True
    except OSError as e:
        res[key] = False
        res[key + "_errno"] = e.errno
attempt("scratch", lambda: open(os.path.join(scratch, "w"), "w").write("x"))
attempt("outside", lambda: open(outside, "w").write("x"))
attempt("hidden", lambda: open(secret).read())
if home:
    attempt("home", lambda: os.listdir(home))
def net():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("127.0.0.1", int(port)))
attempt("net", net)
attempt("chroot", lambda: os.chroot(scratch))
if platform != "darwin":
    st = dict(ln.split(":", 1) for ln in open("/proc/self/status").read().splitlines() if ":" in ln)
    cap = lambda k: int(st.get(k, "0").strip(), 16)
    uid = int(st["Uid"].split()[1])
    res["unprivileged"] = (cap("CapEff") == cap("CapPrm") == cap("CapAmb") == 0 and st["NoNewPrivs"].strip() == "1"
                           and (uid != 0 or cap("CapBnd") == 0))
if platform == "darwin":
    def fork():
        if os.fork() == 0:
            os._exit(0)
    attempt("fork", fork)
    res["contained"] = not res["fork"]
else:
    res["contained"] = os.readlink("/proc/self/ns/pid") != os.environ.get("HOST_PIDNS")
with open(os.path.join(scratch, "probe.json"), "w") as f:
    json.dump(res, f)
"""


def _probe(wrap) -> bool:
    """Runs a probe through `wrap` and checks every property the sandbox relies on, including (on Linux) that the
    program holds no capability it could use to chroot, mount or unmount its way out."""
    with tempfile.TemporaryDirectory(prefix="kiln-sbx-probe-", ignore_cleanup_errors=True) as root:
        root = os.path.realpath(root)
        scratch, hidden = os.path.join(root, "scratch"), os.path.join(root, "hidden")
        os.mkdir(scratch)
        os.mkdir(hidden)
        secret, outside = os.path.join(hidden, "secret"), os.path.join(root, "outside")
        Path(secret).write_text("secret")
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        if sys.platform.startswith("linux"):
            env["HOST_PIDNS"] = os.readlink("/proc/self/ns/pid")
        try:
            with socket.create_server(("127.0.0.1", 0)) as listener:
                port = str(listener.getsockname()[1])
                cmd = [*wrap(scratch, [hidden]), sys.executable, "-I", "-c", _PROBE, scratch, outside, secret,
                       sys.platform, port, _home() or ""]
                subprocess.run(cmd, cwd=scratch, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
            res = json.loads(Path(scratch, "probe.json").read_text())
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
        return (res.get("scratch") is True and res.get("outside") is False and res.get("hidden") is False
                and res.get("net") is False and res.get("contained") is True and res.get("chroot") is False
                and res.get("unprivileged", sys.platform == "darwin") is True
                and res.get("home", False) is False and not os.path.exists(outside))


@functools.cache
def _isolation() -> tuple[str, object] | None:
    """(name, argv-prefix builder) of the first OS isolation that passes the probe, or None."""
    if sys.platform == "darwin":
        cands = [("seatbelt", _wrap_seatbelt)] if shutil.which("sandbox-exec") else []
    elif sys.platform.startswith("linux"):
        cands = [(n, w) for n, w, exe in (("bwrap", _wrap_bwrap, "bwrap"), ("unshare-ns", _wrap_unshare, "unshare"))
                 if shutil.which(exe)]
    else:
        cands = []
    return next(((n, w) for n, w in cands if _probe(w)), None)


def _limits(timeout_s: float, memory_mb: int):
    mem = memory_mb << 20

    def apply():
        os.setsid()
        for res, value in (
            (resource.RLIMIT_CPU, math.ceil(timeout_s) + 1),
            (resource.RLIMIT_FSIZE, 64 << 20),
            (resource.RLIMIT_CORE, 0),
            (resource.RLIMIT_NOFILE, 256),
            (resource.RLIMIT_AS, mem),
            (resource.RLIMIT_DATA, mem),
        ):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass

    return apply


def _rss_bytes(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out.isdigit() else 0


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _supervise(p: subprocess.Popen, deadline: float, memory_bytes: int | None) -> tuple[bytes, str | None]:
    """Drains `p.stderr` (keeping the last `_STDERR_KEEP` bytes) until EOF, kills the process group at the deadline
    or the memory limit (when `memory_bytes` is given, by RSS polling), and stops draining `_DRAIN_S` after the
    process ends even if an escaped descendant holds the pipe open. Returns (stderr tail, kill reason)."""
    fd = p.stderr.fileno()
    os.set_blocking(fd, False)
    buf, killed, drain_until, next_rss = bytearray(), None, None, 0.0
    try:
        while True:
            now = time.monotonic()
            if killed is None and p.poll() is None:
                if now >= deadline:
                    killed = "timeout"
                elif memory_bytes is not None and now >= next_rss:
                    next_rss = now + 0.1
                    if _rss_bytes(p.pid) > memory_bytes:
                        killed = "memory"
                if killed:
                    _kill_group(p)
            if drain_until is None and (killed or p.poll() is not None):
                drain_until = now + _DRAIN_S
            limit = drain_until if drain_until is not None else deadline
            if now >= limit and drain_until is not None:
                break
            ready, _, _ = select.select([fd], [], [], max(0.0, min(0.1, limit - now)))
            if ready:
                try:
                    chunk = os.read(fd, 1 << 20)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 2 * _STDERR_KEEP:
                    del buf[:-_STDERR_KEEP]
    finally:
        _kill_group(p)
        p.stderr.close()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    return bytes(buf[-_STDERR_KEEP:]), killed


def _read_report(path: str):
    """The runner's report, read without following links or blocking on special files; ValueError if unusable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ValueError(f"cannot open the report: {e}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _REPORT_MAX:
            raise ValueError("the report is not a regular file of bounded size")
        text = os.read(fd, _REPORT_MAX + 1)
    finally:
        os.close(fd)
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"sandbox output is not JSON: {type(e).__name__}") from None


def _design_text(s: str) -> bool:
    """True when kiln parses `s` as design text, never as a file path or reference name (its `looks_like_text`
    holds whenever this does: it strips at least these whitespace characters)."""
    t = s.lstrip(" \t\r\n")
    return t.startswith("{") or (t.startswith(("//", "/*")) and "{" in t)


def _valid_report(r) -> bool:
    if not isinstance(r, dict) or not isinstance(r.get("ok"), bool):
        return False
    if r["ok"]:
        return isinstance(r.get("design"), (dict, str))
    frames = r.get("frames", [])
    return (r.get("stage") in _STAGES and isinstance(r.get("error"), str) and isinstance(r.get("message"), str)
            and isinstance(frames, list)
            and all(isinstance(f, dict) and isinstance(f.get("line"), int) and isinstance(f.get("code"), str)
                    for f in frames))


def run_build(program_path: str | os.PathLike, timeout_s: float = 10.0, memory_mb: int = 1024,
              deny_read: list[str | os.PathLike] = ()) -> BuildOutcome:
    """Runs `build()` from `program_path` in the sandbox and returns the design (as data) or a structured error.
    `deny_read` lists directories the program must not read (campaign state, caches). Program-controlled text in
    the outcome (stderr, error message, frames) is redacted and capped: it ends up in LLM feedback."""
    out = _run_build(program_path, timeout_s, memory_mb, deny_read)
    out.stderr = redact(out.stderr, _STDERR_TAIL)
    if out.error:
        out.error["message"] = redact(out.error["message"], _MESSAGE_MAX)
        for f in out.error.get("frames") or []:
            f["code"] = redact(f["code"], 200)
            f["function"] = redact(str(f.get("function", "")), 100)
    return out


def _run_build(program_path, timeout_s, memory_mb, deny_read) -> BuildOutcome:
    t0 = time.monotonic()
    src = Path(program_path)
    if not src.is_file():
        return BuildOutcome(stage="load", error=_error("E-SANDBOX-0001", f"program {src} does not exist",
                                                       "pass the path of a Python file defining build()"))
    iso = _isolation()
    if iso is None:
        return BuildOutcome(stage="sandbox", error=_error(
            "E-SANDBOX-UNAVAILABLE",
            f"no OS sandbox isolating network and file writes is available on {sys.platform}; the program was not run",
            "install bubblewrap (bwrap) or enable unprivileged user namespaces on Linux; use sandbox-exec on macOS"))
    name, wrap = iso
    sandbox = ["subprocess", "rlimits", "socket-patch", name]
    with tempfile.TemporaryDirectory(prefix="kiln-sbx-", ignore_cleanup_errors=True) as tmp:
        tmp = os.path.realpath(tmp)
        prog = os.path.join(tmp, "program.py")
        out = os.path.join(tmp, "out.json")
        shutil.copyfile(src, prog)
        hidden = sorted({os.path.realpath(d) for d in deny_read if os.path.isdir(d)})
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "LANG": "C.UTF-8",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [*wrap(tmp, hidden), sys.executable, *PY_FLAGS, str(RUNNER), prog, out]
        poll_rss = sys.platform == "darwin"
        if poll_rss:
            sandbox.append("rss-poll")
        p = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, preexec_fn=_limits(timeout_s, memory_mb))
        stderr, killed = _supervise(p, t0 + timeout_s, memory_mb << 20 if poll_rss else None)
        tail = stderr.decode(errors="replace")[-_STDERR_TAIL:]
        res = BuildOutcome(stderr=tail, wall_s=time.monotonic() - t0, sandbox=sandbox)
        if killed == "timeout":
            res.stage, res.error = "timeout", _error(
                "E-SANDBOX-TIMEOUT", f"build() did not finish within {timeout_s:g} s",
                "build() must return a design quickly; remove loops that search or simulate", limit=timeout_s,
                unit="s")
            return res
        if killed == "memory":
            res.stage, res.error = "memory", _error(
                "E-SANDBOX-MEMORY", f"build() exceeded the {memory_mb} MiB memory limit",
                "build() should only construct a design dict; avoid large arrays", limit=memory_mb, unit="MiB")
            return res
        try:
            report = _read_report(out)
        except ValueError as e:
            res.stage, res.error = "report", _error(
                "E-SANDBOX-REPORT", f"the sandbox report is unusable: {e}",
                "return a design dict from build(); do not write the runner's output file")
            return res
        if report is None:
            sig = -p.returncode if p.returncode and p.returncode < 0 else None
            what = f"signal {signal.Signals(sig).name}" if sig else f"exit code {p.returncode}"
            memory_hint = sig in (signal.SIGKILL, signal.SIGSEGV) or "MemoryError" in tail
            res.stage, res.error = "crash", _error(
                "E-SANDBOX-CRASH", f"the program process died ({what}) before returning a design",
                "the memory limit may have been hit; build() should only construct a dict" if memory_hint
                else "do not call sys.exit/os._exit or crash the interpreter; return a design dict from build()")
            return res
        if not _valid_report(report):
            res.stage, res.error = "report", _error(
                "E-SANDBOX-REPORT", "the sandbox report does not have the runner's structure",
                "return a design dict from build(); do not write the runner's output file")
            return res
        if not report["ok"]:
            frames = report.get("frames") or []
            where = f" at program.py line {frames[-1]['line']} ({frames[-1]['code']})" if frames else ""
            hints = {
                "load": "the program failed to import; fix the syntax or import error",
                "build": "build() raised; fix the exception",
                "serialize": "build() must return a JSON-serializable dict with finite numbers",
            }
            res.stage = report["stage"]
            res.error = _error(
                f"E-SANDBOX-{res.stage.upper()}", f"{report['error']}: {report['message']}{where}",
                hints[res.stage], frames=frames)
            return res
        if isinstance(report["design"], str) and not _design_text(report["design"]):
            res.stage, res.error = "serialize", _error(
                "E-SANDBOX-SERIALIZE", "build() returned a string that is not JSON/JSON5 design text",
                "return the design as a dict or as JSON/JSON5 text starting with '{'; file paths and reference "
                "names are not accepted from a program")
            return res
        res.design = report["design"]
        return res


_SESSION: kiln.Session | None = None


def _session() -> kiln.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = kiln.Session()
    return _SESSION


def _failure(stage: str, errors: list[dict], feedback: str, **extra) -> dict:
    return {"combined_score": 0.0, "valid": 0.0, "status": "invalid" if stage != "timeout" else "timeout",
            "stage": stage, "features": {}, "descriptors": {}, "errors": errors,
            "feedback": redact(feedback, FEEDBACK_MAX_CHARS),
            "result": None, **extra}


def _error_feedback(err: dict, stderr: str = "") -> str:
    lines = [f"FAILED ({err['code']}): {err['message']}", f"hint: {err['hint']}"]
    if stderr.strip():
        lines.append("stderr (tail):\n" + stderr.strip()[-600:])
    return "\n".join(lines)


def evaluate_program(
    program_path: str | os.PathLike,
    suite: str = "standard",
    baseline: str = "a100_40gb",
    fitness: str | dict | None = None,
    *,
    session=None,
    options: dict | None = None,
    timeout_s: float = 10.0,
    memory_mb: int = 1024,
    deny_read: list[str | os.PathLike] = (),
) -> dict:
    """OpenEvolve-style evaluation of a program defining `build()`.

    Returns `combined_score` (0 on any failure), `valid`, each standard MAP-Elites descriptor normalized to
    [0, 1] on its fixed range (flattened, plus under `descriptors`), raw `features`, `status`, `stage`,
    structured `errors`, an LLM-readable `feedback` string, and the full `kiln.result/1` dict as `result`.
    """
    build = run_build(program_path, timeout_s=timeout_s, memory_mb=memory_mb, deny_read=deny_read)
    if not build.ok:
        return _failure(build.stage, [build.error], _error_feedback(build.error, build.stderr),
                        sandbox=build.sandbox)
    opts = dict(options or {})
    fit = {"kind": fitness} if isinstance(fitness, str) else dict(fitness or {})
    fit.setdefault("baseline", baseline)
    opts["fitness"] = {**opts.get("fitness", {}), **fit}
    try:
        r = (session or _session()).evaluate(build.design, suite, opts)
    except (ValueError, TypeError) as e:
        err = _error("E-SANDBOX-OPTIONS", f"kiln rejected the evaluation request: {e}",
                     "check the suite name, fitness and options")
        return _failure("options", [err], _error_feedback(err), sandbox=build.sandbox)
    d = r.to_dict()
    ok = d["status"] == "ok"
    score = float(d["score"]) if ok and math.isfinite(d["score"]) else 0.0
    desc = kiln.normalize_features(d.get("features") or {})
    feedback = r.explain()
    if any(e["code"] == "E-NOT-IMPLEMENTED" for e in d["errors"]):
        feedback += "note: the design passed validation but this kiln build cannot score it yet.\n"
    return {
        "combined_score": score,
        "valid": 1.0 if ok else 0.0,
        **desc,
        "descriptors": desc,
        "features": d.get("features") or {},
        "status": d["status"],
        "stage": d["stage_reached"],
        "errors": d["violations"] + d["errors"],
        "feedback": redact(feedback, FEEDBACK_MAX_CHARS),
        "result": d,
        "sandbox": build.sandbox,
    }
