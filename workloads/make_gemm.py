import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def make_gemm(m: int, n: int, k: int, out_dir: Path) -> Path:
    w = TensorProto(name="W", data_type=TensorProto.BFLOAT16, dims=[k, n])
    node = helper.make_node("Gemm", ["X", "W"], ["Y"], name="Gemm")
    graph = helper.make_graph(
        [node], f"gemm_{m}_{n}_{k}",
        [helper.make_tensor_value_info("X", TensorProto.BFLOAT16, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [m, n])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = out_dir / f"gemm_{m}_{n}_{k}.onnx"
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    m, n, k = map(int, sys.argv[1:4])
    print(make_gemm(m, n, k, Path(__file__).parent))
