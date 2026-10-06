import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kiln_evo  # noqa: E402,F401  (puts kiln on the path)
from kiln_evo import config as C  # noqa: E402

FAST = {
    "workloads": {"train": ["llama3_8b:decode_b1", "llama3_8b:decode_b8"], "screen": ["llama3_8b:decode_b8"],
                  "heldout": ["gptj_6b:decode_b8"]},
    "descriptors": [{"name": "machine_balance", "bins": 8}, {"name": "onchip_bytes", "bins": 8},
                    {"name": "flops_per_tile", "bins": 4}],
    "llm": {"backend": "mock", "price_usd_per_mtok": {"input": 1.0, "output": 4.0}},
    "operators": {"parametric": 0.4, "llm_diff": 0.45, "llm_full": 0.15},
    "islands": {"count": 2, "migration_interval": 2, "migrants": 1},
    "batch": {"children_per_island": 4},
    "audit": {"on_new_elite": False, "seeds": [0, 1]},
    "heldout": {"every_generations": 2, "top_k": 3},
    "seeds": [{"name": "a100", "program": "kiln_evo:seeds/a100.py"},
              {"name": "a100_lean", "program": "kiln_evo:seeds/a100.py", "params": {"sm_per_tpc": 1}},
              {"name": "tpu_v4", "program": "kiln_evo:seeds/tpu_v4.py"}],
}


@pytest.fixture(scope="session")
def kiln_cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("kiln_cache"))


@pytest.fixture
def make_cfg(tmp_path, kiln_cache):
    def make(name="t", **over):
        cfg = C._merge(FAST, {"name": name, "out_dir": str(tmp_path / name), "cache_dir": kiln_cache,
                              "adversarial_dir": str(tmp_path / "adversarial")})
        cfg = C._merge(cfg, over)
        return C.finalize(cfg, tmp_path)

    return make
