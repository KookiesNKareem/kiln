import sys
import tempfile
import time

from stream.api import evaluate_mapping

CLOCK_HZ = {"a100": 1.41e9, "a100_12": 1.41e9}


def main(hw: str, workload: str, flops: float):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        est = evaluate_mapping(f"hw/{hw}/{hw}.yaml", workload, tmp)
    secs = est.cycles / CLOCK_HZ[hw]
    print(f"RESULT hw={hw} workload={workload} cycles={est.cycles:.0f} "
          f"time_ms={secs * 1e3:.3f} tflops={flops / secs / 1e12:.1f} wall_s={time.time() - t0:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
