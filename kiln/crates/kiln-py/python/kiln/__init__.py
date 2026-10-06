"""kiln evaluation API (spec 06 §6): Session, evaluate, evaluate_batch, validate, explain, bench_export."""

from ._kiln import (
    GIT_HASH,
    KILN_VERSION,
    Result,
    Session,
    bench_export,
    descriptors,
    evaluate,
    explain,
    normalize_features,
    render,
    validate,
)

__version__ = KILN_VERSION

__all__ = [
    "GIT_HASH",
    "KILN_VERSION",
    "Result",
    "Session",
    "bench_export",
    "descriptors",
    "evaluate",
    "explain",
    "normalize_features",
    "render",
    "validate",
]
