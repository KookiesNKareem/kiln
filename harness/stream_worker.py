"""Run one Stream evaluation and print a single JSON line; executed in a subprocess by harness.evaluate."""

import json
import logging
import sys
import tempfile
import traceback


def main(hw_yaml: str, onnx_path: str) -> dict:
    logging.disable(logging.CRITICAL)
    from stream.api import evaluate_mapping
    from stream.ir.infeasibility import InfeasibleAllocationError

    try:
        with tempfile.TemporaryDirectory() as tmp:
            est = evaluate_mapping(hw_yaml, onnx_path, tmp)
        return {"status": "ok", "cycles": float(est.cycles), "group_cycles": list(map(float, est.group_cycles))}
    except InfeasibleAllocationError as e:
        return {"status": "infeasible", "error": str(e)[:2000]}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:2000], "trace": traceback.format_exc()[-3000:]}


if __name__ == "__main__":
    print("HARNESS_RESULT " + json.dumps(main(sys.argv[1], sys.argv[2])), flush=True)
