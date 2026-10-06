"""kiln_evo: LLM-driven MAP-Elites design search over kiln (spec 06 §6, §12.3)."""

import sys
from pathlib import Path

try:
    import kiln  # noqa: F401
except ImportError:  # in-repo use without an installed wheel: the package sits next to kiln-py's sources
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crates" / "kiln-py" / "python"))
    import kiln  # noqa: F401

from .config import ConfigError, load  # noqa: E402

__all__ = ["ConfigError", "load"]
