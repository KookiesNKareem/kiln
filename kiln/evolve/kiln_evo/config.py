"""Campaign configuration (YAML or JSON). Every key has a default here except the LLM model and prices, which the
owner fills from the provider's live catalog (see campaigns/a100_decode.yaml)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

PKG = Path(__file__).resolve().parent
KILN_ROOT = PKG.parents[1]
REFERENCE_DESIGNS = KILN_ROOT / "designs" / "reference"
PLACEHOLDER = "FILL_ME"

DEFAULTS: dict = {
    "name": None,
    "mode": "evolve",
    "out_dir": None,
    "calibration": "generic-v1",
    "baseline": "a100_40gb",
    "fitness": {"kind": "matched_envelope", "interval_basis": "central", "aggregation": "geomean", "weights": None},
    "envelope": None,
    "options": {"profile": "search", "interval": "sensitivity", "stack": "kiln_ideal"},
    "workloads": {
        "train": ["llama3_8b:decode_b1", "llama3_8b:decode_b8", "llama3_8b:decode_b32"],
        "screen": ["llama3_8b:decode_b8"],
        "heldout": ["llama3_8b:decode_b1_kv32768", "gptj_6b:decode_b8", "llama3_8b:prefill_b1"],
    },
    "descriptors": [
        {"name": "machine_balance", "bins": 8},
        {"name": "onchip_bytes", "bins": 8},
        {"name": "flops_per_tile", "bins": 6},
    ],
    "cascade": {"prune_ratio": 0.5},
    "audit": {"suspicion_ratio": 1.15, "random_rate": 0.02, "seeds": [0, 1, 2], "on_new_elite": True,
              "max_seed_spread": 0.10, "interval": "corners"},
    "heldout": {"every_generations": 5, "top_k": 10, "overfit_gap": -0.25},
    "llm": {
        "backend": "mock",
        "model": None,
        "base_url": None,
        "api_key_env": None,
        "price_usd_per_mtok": {"input": None, "output": None},
        "max_tokens": 8192,
        "temperature": 0.8,
        "timeout_s": 180,
        "retries": 2,
        "anthropic_version": "2023-06-01",
        "mock": {"seed": 0, "failure_rate": 0.1, "full_rewrite_rate": 0.2, "tokens_per_char": 0.25},
    },
    "operators": {"parametric": 0.3, "llm_diff": 0.5, "llm_full": 0.2},
    "selection": {"exploit": 0.3, "tournament": 3, "inspirations": 3, "failures_shown": 3},
    "prompt": {"max_chars": 60000, "explain_chars": 1500, "full_program_inspirations": 1},
    "budget": {"max_evals": 1000, "max_usd": None, "max_wall_s": None, "max_generations": None},
    "islands": {"count": 2, "migration_interval": 5, "migrants": 2},
    "batch": {"children_per_island": 8, "workers": None, "max_build_workers": 8},
    "seeds": [
        {"name": "a100", "program": "kiln_evo:seeds/a100.py"},
        {"name": "a100_sram_heavy", "program": "kiln_evo:seeds/a100.py",
         "params": {"tpc_per_gpc": 4, "l2_slice_kib": 2048, "l1_kib": 256}},
        {"name": "a100_lean_wide_hbm", "program": "kiln_evo:seeds/a100.py",
         "params": {"sm_per_tpc": 1, "hbm_enabled": 6, "hbm_pin_gbps": 2.0}},
        {"name": "tpu_v4", "program": "kiln_evo:seeds/tpu_v4.py"},
    ],
    "sandbox": {"timeout_s": 10.0, "memory_mb": 1024},
    "redteam": {"objective": "floor_margin", "hole_threshold": None},
    "adversarial_dir": None,
    "cache_dir": None,
    "seed": 0,
}

FITNESS_KINDS = {"matched_envelope", "explicit_envelope", "perf_per_watt", "perf_per_area", "pareto",
                 "baseline_relative"}
REDTEAM_OBJECTIVES = {"floor_margin", "physics_bound", "extrapolation_leverage", "tier_disagreement"}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_text(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve_program_path(spec: str, base_dir: Path) -> Path:
    if spec.startswith("kiln_evo:"):
        return PKG / spec.split(":", 1)[1]
    p = Path(spec)
    return p if p.is_absolute() else (base_dir / p)


def load(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path).resolve()
    raw = _load_text(path)
    cfg = _merge(DEFAULTS, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg["_source"] = str(path)
    return finalize(cfg, path.parent)


def finalize(cfg: dict, base_dir: Path) -> dict:
    cfg = _merge(DEFAULTS, cfg)
    cfg["name"] = cfg["name"] or "campaign"
    out = Path(cfg["out_dir"] or f"runs/{cfg['name']}")
    cfg["out_dir"] = str(out if out.is_absolute() else (base_dir / out).resolve())
    adv = cfg["adversarial_dir"]
    cfg["adversarial_dir"] = str(Path(adv).resolve() if adv else KILN_ROOT / "corpus" / "adversarial")
    if cfg["cache_dir"]:
        cfg["cache_dir"] = str((base_dir / cfg["cache_dir"]).resolve())
    for s in cfg["seeds"]:
        s["path"] = str(resolve_program_path(s["program"], base_dir))
    validate(cfg)
    return cfg


def _usd(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def validate(cfg: dict) -> None:
    errs = []
    if cfg["mode"] not in ("evolve", "redteam"):
        errs.append(f"mode must be evolve or redteam, got {cfg['mode']!r}")
    if cfg["fitness"]["kind"] not in FITNESS_KINDS:
        errs.append(f"fitness.kind {cfg['fitness']['kind']!r} not in {sorted(FITNESS_KINDS)}")
    if cfg["fitness"]["aggregation"] not in ("geomean", "min", "weighted_harmonic"):
        errs.append("fitness.aggregation must be geomean, min or weighted_harmonic")
    w = cfg["workloads"]
    if not w["train"]:
        errs.append("workloads.train is empty")
    if not set(w["screen"]) <= set(w["train"]):
        errs.append("workloads.screen must be a subset of workloads.train")
    if set(w["heldout"]) & set(w["train"]):
        errs.append("workloads.heldout must not overlap workloads.train (held-out results never reach the loop)")
    if not cfg["descriptors"]:
        errs.append("descriptors is empty")
    if cfg["mode"] == "redteam" and cfg["redteam"]["objective"] not in REDTEAM_OBJECTIVES:
        errs.append(f"redteam.objective must be one of {sorted(REDTEAM_OBJECTIVES)}")
    weights = cfg["fitness"].get("weights") or {}
    bad_w = {k: v for k, v in weights.items()
             if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0}
    if bad_w:
        errs.append(f"fitness.weights must be finite and non-negative, got {bad_w}")
    else:
        for part in ("train", "heldout"):
            if w[part] and sum(float(weights.get(x, 1.0)) for x in w[part]) <= 0:
                errs.append(f"fitness.weights give workloads.{part} a zero total weight")
    max_usd = cfg["budget"].get("max_usd")
    if max_usd is not None and not _usd(max_usd):
        errs.append(f"budget.max_usd = {max_usd!r} must be a finite, non-negative number")
    ops = cfg["operators"]
    if sum(ops.values()) <= 0:
        errs.append("operators weights sum to zero")
    if not cfg["seeds"]:
        errs.append("seeds is empty")
    for s in cfg["seeds"]:
        if not Path(s["path"]).is_file():
            errs.append(f"seed {s.get('name')}: program {s['path']} not found")
    llm = cfg["llm"]
    uses_llm = ops.get("llm_diff", 0) + ops.get("llm_full", 0) > 0
    if llm["backend"] not in ("mock", "openai", "anthropic"):
        errs.append("llm.backend must be mock, openai (any OpenAI-compatible chat API) or anthropic")
    elif llm["backend"] != "mock" and uses_llm:
        model = llm.get("model")
        if not model or PLACEHOLDER in str(model):
            errs.append("llm.model is a placeholder: copy a model id from the provider's live catalog "
                        "(`python -m kiln_evo models <campaign>` lists it)")
        price = llm.get("price_usd_per_mtok") or {}
        for k in ("input", "output"):
            v = price.get(k)
            if not _usd(v):
                errs.append(f"llm.price_usd_per_mtok.{k} is unset or not a finite, non-negative number: read it "
                            "from the provider's pricing page for that exact model; the dollar budget cannot be "
                            "enforced without it")
        if cfg["budget"].get("max_usd") is None:
            errs.append("budget.max_usd must be set for a real LLM backend")
        if not llm.get("api_key_env"):
            errs.append("llm.api_key_env names the environment variable holding the API key")
        extra = llm.get("extra_body") or {}
        limits = [llm.get("max_tokens"), *(extra[k] for k in ("max_tokens", "max_completion_tokens",
                                                                  "max_output_tokens") if k in extra)]
        if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in limits):
            errs.append("llm.max_tokens (and any output limit in llm.extra_body) must be a positive integer: the "
                        "dollar budget reserves against it")
        if extra.get("n", 1) != 1:
            errs.append("llm.extra_body.n must be 1: one completion per call is used and reserved against the dollar "
                        "budget")
        if llm["backend"] == "openai" and not llm.get("base_url"):
            errs.append("llm.base_url is required for the openai backend (e.g. the provider's /v1 endpoint)")
    if errs:
        raise ConfigError("invalid campaign config:\n  - " + "\n  - ".join(errs))


def config_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("budget", "out_dir")}
    return "cfg-" + hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def kiln_options(cfg: dict, **extra) -> dict:
    fit = {"kind": cfg["fitness"]["kind"], "baseline": cfg["baseline"],
           "interval_basis": cfg["fitness"]["interval_basis"]}
    if cfg.get("envelope"):
        fit["envelope"] = cfg["envelope"]
    opts = {**cfg["options"], "fitness": fit}
    opts.update(extra)
    return opts
