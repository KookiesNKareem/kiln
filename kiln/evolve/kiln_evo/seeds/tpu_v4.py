"""Seed: Google TPU v4 single chip (designs/reference/tpu_v4.json5) as a parametric program.

A systolic, scratchpad-managed, statically scheduled design: a structurally different starting point from the
A100 baseline. BASE and PARAMS are materialized from REFERENCE when a campaign starts (programs.materialize_seed),
so the seed cannot drift from the reference design. PARAMS become the document's kiln `params` (expressions like
"=n_mxu" in BASE read them), plus two structural knobs applied in Python. Off-chip knobs (ENVELOPE_KNOBS) are
capped at the reference (E-ENV-0007/0008).
"""

import copy

REFERENCE = "tpu_v4"
NAME = "tpu-v4-seed"
ENVELOPE_KNOBS = ("n_hbm", "hbm_pin")
STRUCTURAL = ("n_tc", "cmem_mib")

PARAMS = {}

PARAM_SPACE = {
    "mxu_edge": {"min": 32, "max": 512, "log2": True},
    "n_mxu": {"min": 1, "max": 16, "log2": True},
    "vmem_cap": {"choices": ["4MiB", "8MiB", "16MiB", "32MiB", "64MiB"]},
    "vmem_rd_ports": {"min": 1, "max": 6, "step": 1},
    "hbm_pin": {"choices": ["1.6Gbps", "2.0Gbps", "2.34375Gbps", "2.8Gbps", "3.2Gbps"]},
    "n_hbm": {"min": 2, "max": 8, "step": 1},
    "n_tc": {"min": 1, "max": 8, "step": 1},
    "cmem_mib": {"min": 16, "max": 512, "log2": True},
}

BASE = {}


def reference_params(d):
    """The knob values of design `d` (the reference BASE)."""
    die = d["system"]["package"]["dies"][0]
    cap = die["memories"][0]["capacity"]
    assert cap.endswith("MiB"), cap
    return {**d["params"], "n_tc": die["clusters"][0]["count"], "cmem_mib": int(cap[:-3])}


def build():
    P = PARAMS
    d = copy.deepcopy(BASE)
    d["params"] = {**d.get("params", {}), **{k: v for k, v in P.items() if k not in STRUCTURAL}}
    die = d["system"]["package"]["dies"][0]
    die["clusters"][0]["count"] = P["n_tc"]
    if P["cmem_mib"] != reference_params(BASE)["cmem_mib"]:
        die["memories"][0]["capacity"] = f"{P['cmem_mib']}MiB"
    return d
