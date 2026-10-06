"""Runs an untrusted `build()` program; writes {"ok", "design" | "stage", "error", ...} as JSON to argv[2].

Started by kiln.evolve with `python -s -P`, resource limits, a scrubbed environment (PYTHONHASHSEED=0) and an OS sandbox
(network, file writes, descendants). The socket patch below is defense in depth only.
"""

import importlib.util
import json
import os
import socket
import sys
import traceback


def _no_network(*_args, **_kwargs):
    raise PermissionError("network access is disabled in the kiln sandbox")


def _block_network():
    for name in ("socket", "create_connection", "getaddrinfo", "socketpair", "fromfd"):
        if hasattr(socket, name):
            setattr(socket, name, _no_network)


def _frames(exc, program):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == program]
    return [{"line": f.lineno, "function": f.name, "code": (f.line or "").strip()} for f in frames[-5:]]


def main():
    program, out_path = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or ".") != here]
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
