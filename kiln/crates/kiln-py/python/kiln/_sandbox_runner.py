"""Runs an untrusted `build()` program; writes {"ok", "design" | "stage", "error", ...} as JSON to argv[2].

Started by kiln.evolve with `python -s -P`, resource limits, a scrubbed environment (PYTHONHASHSEED=0) and an OS sandbox
(network, file writes, descendants: no process may start on Linux). The socket patch below is defense in depth only.
"""

import ctypes
import errno
import importlib.util
import json
import os
import socket
import struct
import sys
import traceback

PR_SET_SECCOMP, SECCOMP_MODE_FILTER, PR_SET_NO_NEW_PRIVS = 22, 2, 38
CLONE_THREAD = 0x10000
CLONE3 = 435
# (AUDIT_ARCH, fork-like syscalls denied outright, clone; x86-64 also denies the x32 syscall range).
_ARCH = {"x86_64": (0xC000003E, (57, 58), 56), "aarch64": (0xC00000B7, (), 220)}
_ALLOW, _ERRNO, _KILL = 0x7FFF0000, 0x00050000, 0x80000000


def _no_network(*_args, **_kwargs):
    raise PermissionError("network access is disabled in the kiln sandbox")


def _block_network():
    for name in ("socket", "create_connection", "getaddrinfo", "socketpair", "fromfd"):
        if hasattr(socket, name):
            setattr(socket, name, _no_network)


def _seccomp_program(machine: str) -> bytes:
    """Classic BPF over `seccomp_data`: fork, vfork, clone without CLONE_THREAD fail with EPERM and clone3 with
    ENOSYS (libc then falls back to clone, so threads still start); other ABIs are killed."""
    arch, forks, clone = _ARCH[machine]
    ins = [("ld", 4), ("jeq", arch, None, "kill"), ("ld", 0)]
    if machine == "x86_64":
        ins.append(("jge", 0x40000000, "deny", None))
    ins += [("jeq", nr, "deny", None) for nr in forks]
    ins += [("jeq", CLONE3, "enosys", None), ("jeq", clone, None, "allow"), ("ld", 16),
            ("jset", CLONE_THREAD, "allow", "deny")]
    rets = {"allow": _ALLOW, "deny": _ERRNO | errno.EPERM, "enosys": _ERRNO | errno.ENOSYS, "kill": _KILL}
    at = {name: len(ins) + i for i, name in enumerate(rets)}
    ops = {"ld": 0x20, "jeq": 0x15, "jge": 0x35, "jset": 0x45}
    out = b""
    for i, (op, k, *jumps) in enumerate(ins):
        jt, jf = (0 if j is None else at[j] - i - 1 for j in (jumps or (None, None)))
        out += struct.pack("=HBBI", ops[op], jt, jf, k)
    return out + b"".join(struct.pack("=HBBI", 0x06, 0, 0, v) for v in rets.values())


def deny_processes():
    """Confines the program to this one process on Linux: RLIMIT_AS bounds a process, so any descendant would get a
    memory limit of its own. Raises OSError (the program must then not run) if the filter cannot be installed."""
    machine = os.uname().machine
    if machine not in _ARCH:
        raise OSError(errno.ENOTSUP, f"no process filter for {machine}")
    prog = ctypes.create_string_buffer(_seccomp_program(machine))
    fprog = struct.pack("HxxxxxxP", len(prog.raw) // 8, ctypes.addressof(prog))
    libc = ctypes.CDLL(None, use_errno=True)
    for args in ((PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), (PR_SET_SECCOMP, SECCOMP_MODE_FILTER, fprog, 0, 0)):
        if libc.prctl(*(ctypes.c_char_p(a) if isinstance(a, bytes) else ctypes.c_ulong(a) for a in args)) != 0:
            e = ctypes.get_errno()
            raise OSError(e, f"prctl: {os.strerror(e)}")


def _frames(exc, program):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == program]
    return [{"line": f.lineno, "function": f.name, "code": (f.line or "").strip()} for f in frames[-5:]]


def main():
    program, out_path = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or ".") != here]
    if sys.platform.startswith("linux"):
        deny_processes()
    _block_network()
    stage = "load"
    try:
        spec = importlib.util.spec_from_file_location("candidate", program)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stage = "build"
        build = getattr(module, "build", None)
        if not callable(build):
            raise TypeError("the program must define build() returning a kiln.hw/1.0 design dict")
        design = build()
        stage = "serialize"
        if hasattr(design, "to_dict"):
            design = design.to_dict()
        if not isinstance(design, (dict, str)):
            raise TypeError(f"build() must return a dict (or JSON/JSON5 text), got {type(design).__name__}")
        result = {"ok": True, "design": design}
        text = json.dumps(result, allow_nan=False)
    except BaseException as e:  # noqa: BLE001 - every failure becomes a structured report
        result = {
            "ok": False,
            "stage": stage,
            "error": type(e).__name__,
            "message": str(e)[:2000],
            "frames": _frames(e, program),
        }
        text = json.dumps(result, default=str)
    with open(out_path, "w") as f:
        f.write(text)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
