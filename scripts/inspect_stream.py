import json
import sys
import tempfile

from stream.api import evaluate_mapping
from stream.ir import AllocationIR

with tempfile.TemporaryDirectory() as tmp:
    est = evaluate_mapping(sys.argv[1], sys.argv[2], tmp)
perf = AllocationIR.from_internal(est.context.get("scheduler")).model_dump()["performance"]
print("INSPECT", json.dumps({k: perf[k] for k in ("latency", "bottleneck", "aggregate", "nodes")}, indent=1))
