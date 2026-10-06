"""Seed: A100-SXM4-40GB (designs/reference/a100_sxm4_40gb.json5) as a parametric program.

build() returns a kiln.hw/1.0 design dict. BASE is the full reference structure and PARAMS the knobs read from it;
kiln_evo materializes both from REFERENCE when a campaign starts (programs.materialize_seed), so the seed cannot
drift from the reference design. The non-LLM parametric mutation turns PARAMS within PARAM_SPACE; LLM edits may
change BASE directly. build() rewrites only the parts of BASE whose knobs differ from the reference, so the default
PARAMS reproduce it exactly. Off-chip knobs (ENVELOPE_KNOBS) are capped at the reference: more HBM bandwidth or
capacity than the baseline is rejected (E-ENV-0007/0008).
"""

import copy
import math

REFERENCE = "a100_sxm4_40gb"
NAME = "a100-seed"
ENVELOPE_KNOBS = ("hbm_enabled", "hbm_stack_gib", "hbm_pin_gbps")

PARAMS = {}

PARAM_SPACE = {
    "gpc_rows": {"min": 1, "max": 4, "step": 1},
    "gpc_cols": {"min": 1, "max": 8, "step": 1},
    "tpc_per_gpc": {"min": 1, "max": 16, "log2": True},
    "sm_per_tpc": {"min": 1, "max": 4, "step": 1},
    "harvest": {"choices": [True, False]},
    "l1_kib": {"min": 64, "max": 1024, "log2": True},
    "l2_slices_per_partition": {"min": 8, "max": 128, "log2": True},
    "l2_slice_kib": {"min": 128, "max": 4096, "log2": True},
    "hbm_sites": {"min": 2, "max": 8, "step": 2},
    "hbm_enabled": {"min": 1, "max": 8, "step": 1},
    "hbm_stack_gib": {"choices": [8, 16, 24]},
    "hbm_pin_gbps": {"min": 1.6, "max": 3.6, "scale": 1.15},
    "fabric_edge_bits": {"min": 2048, "max": 40960, "log2": True},
    "die_w_mm": {"min": 15.0, "max": 26.0, "scale": 1.08},
    "die_h_mm": {"min": 15.0, "max": 33.0, "scale": 1.08},
}

BASE = {}

L1_SCRATCH_FRACTIONS = (0.0, 8 / 192, 16 / 192, 32 / 192, 64 / 192, 100 / 192, 132 / 192, 164 / 192)


def _l1_carveouts(l1_kib):
    out, seen = [], set()
    for f in L1_SCRATCH_FRACTIONS:
        s = int(round(l1_kib * f))
        if s not in seen:
            seen.add(s)
            out.append({"scratch": f"{s}KiB", "cache": f"{l1_kib - s}KiB"})
    return out


def _num(text, unit):
    assert text.endswith(unit), (text, unit)
    v = text[: -len(unit)]
    return int(v) if unit.endswith("iB") else float(v)


def reference_params(d):
    """The knob values of design `d` (the reference BASE)."""
    die = d["system"]["package"]["dies"][0]
    gpc, l2p, hbm_if = die["clusters"]
    stack = d["system"]["package"]["mem_stacks"][0]
    rows, cols = gpc["layout"]["grid"]
    return {
        "gpc_rows": rows,
        "gpc_cols": cols,
        "tpc_per_gpc": gpc["clusters"][0]["count"],
        "sm_per_tpc": gpc["clusters"][0]["clusters"][0]["count"],
        "harvest": bool(gpc.get("disabled")),
        "l1_kib": _num(d["templates"]["sm"]["body"]["memories"][0]["capacity"], "KiB"),
        "l2_slices_per_partition": l2p["memories"][0]["count"],
        "l2_slice_kib": _num(l2p["memories"][0]["capacity"], "KiB"),
        "hbm_sites": hbm_if["count"],
        "hbm_enabled": hbm_if["count"] - len(hbm_if.get("disabled", [])),
        "hbm_kind": stack["kind"],
        "hbm_stack_gib": _num(stack["capacity"], "GiB"),
        "hbm_pin_gbps": _num(stack["pin_rate_bits_per_s"], "Gbps"),
        "fabric_edge_bits": die["networks"][0]["topology"]["edges"][0]["link"]["width_bits"],
        "die_w_mm": _num(die["floorplan"]["outline"]["w"], "mm"),
        "die_h_mm": _num(die["floorplan"]["outline"]["h"], "mm"),
    }


GPC_KNOBS = ("gpc_rows", "gpc_cols", "tpc_per_gpc", "sm_per_tpc", "harvest")
HBM_KNOBS = ("hbm_sites", "hbm_enabled", "hbm_kind", "hbm_stack_gib", "hbm_pin_gbps")


def build():
    P = PARAMS
    d = copy.deepcopy(BASE)
    ref = reference_params(BASE)
    changed = {k for k in P if P[k] != ref.get(k)}
    die = d["system"]["package"]["dies"][0]
    gpc, l2p, hbm_if = die["clusters"]
    fabric = die["networks"][0]

    if "l1_kib" in changed:
        l1 = d["templates"]["sm"]["body"]["memories"][0]
        l1["capacity"] = f"{P['l1_kib']}KiB"
        l1["operands"]["options"] = _l1_carveouts(P["l1_kib"])
    if changed & {"die_w_mm", "die_h_mm"}:
        die["floorplan"]["outline"].update(w=f"{P['die_w_mm']}mm", h=f"{P['die_h_mm']}mm")
    if changed & set(GPC_KNOBS):
        gpc["count"] = P["gpc_rows"] * P["gpc_cols"]
        gpc["layout"] = {"grid": [P["gpc_rows"], P["gpc_cols"]]}
        gpc["clusters"][0]["count"] = P["tpc_per_gpc"]
        gpc["clusters"][0]["clusters"][0]["count"] = P["sm_per_tpc"]
        full_size = P["gpc_rows"] >= 2 and P["gpc_cols"] >= 4 and P["tpc_per_gpc"] >= 8
        gpc["disabled"] = list(BASE["system"]["package"]["dies"][0]["clusters"][0].get("disabled", [])) \
            if (P["harvest"] and full_size) else []
        rows, n_gpc = P["gpc_rows"], P["gpc_rows"] * P["gpc_cols"]
        if rows >= 2:
            sm_eps = [{"select": f"gpc{r}_*.tpc*.sm*.l1", "at": {"router": [0 if r < rows / 2 else 1]}}
                      for r in range(rows)]
        else:
            h = math.ceil(n_gpc / 2)
            sm_eps = [{"select": f"gpc[0..{h}].tpc*.sm*.l1", "at": {"router": [0]}}]
            if n_gpc > h:
                sm_eps.append({"select": f"gpc[{h}..{n_gpc}].tpc*.sm*.l1", "at": {"router": [1]}})
        fabric["endpoints"] = sm_eps + [e for e in fabric["endpoints"] if ".sm" not in e["select"]]
    if changed & {"l2_slices_per_partition", "l2_slice_kib"}:
        slice_ = l2p["memories"][0]
        slice_["count"] = P["l2_slices_per_partition"]
        slice_["capacity"] = f"{P['l2_slice_kib']}KiB"
        slice_["cache"]["pinnable"] = f"{P['l2_slice_kib'] * 3 // 4}KiB"
    if changed & set(HBM_KNOBS):
        sites, enabled = P["hbm_sites"], min(P["hbm_enabled"], P["hbm_sites"])
        west = math.ceil(sites / 2)
        hbm_if["count"] = sites
        hbm_if["disabled"] = [f"hbm_if{j}" for j in range(enabled, sites)]
        shore = die["floorplan"]["shoreline"]
        shore[0]["count"], shore[1]["count"] = west, sites - west
        if sites - west == 0:
            shore.pop(1)
        mc_eps = [{**e} for e in fabric["endpoints"] if e["select"].startswith("hbm_if")]
        mc_eps[0]["select"] = f"hbm_if[0..{west}].mc*"
        mc_eps[1]["select"] = f"hbm_if[{west}..{sites}].mc*"
        if sites - west == 0:
            mc_eps.pop(1)
        fabric["endpoints"] = [e for e in fabric["endpoints"] if not e["select"].startswith("hbm_if")] + mc_eps
        stack = d["system"]["package"]["mem_stacks"][0]
        stack["count"] = sites
        stack["disabled"] = [f"hbm{j}" for j in range(enabled, sites)]
        stack["kind"] = P["hbm_kind"]
        stack["capacity"] = f"{P['hbm_stack_gib']}GiB"
        stack["pin_rate_bits_per_s"] = f"{P['hbm_pin_gbps']}Gbps"
        for c in d["clocks"]:
            if c["id"] == "hbm_clk":
                c["freq"] = f"{P['hbm_pin_gbps'] * 500:g}MHz"
    if "fabric_edge_bits" in changed:
        fabric["topology"]["edges"][0]["link"]["width_bits"] = P["fabric_edge_bits"]
    return d
