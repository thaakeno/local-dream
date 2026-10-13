"""Cloud regression checks for the exact numerical-lowering boundary."""
import os

if os.environ.get("GITHUB_ACTIONS") != "true":
    raise SystemExit("Cloud-only dependency environment required")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from lowering import lower_nonpositive_exp


def check():
    shape = [8, 1]
    tensors = [numpy_helper.from_array(np.tril(np.ones((8, 8), dtype=np.float32)), "prefix"),
               numpy_helper.from_array(np.array([-0.25], dtype=np.float32), "log_rate")]
    graph = helper.make_graph([
        helper.make_node("Exp", ["log_rate"], ["rate"], name="constant_exp"),
        helper.make_node("Neg", ["rate"], ["negative_rate"]),
        helper.make_node("Softplus", ["a"], ["positive_gate"]),
        helper.make_node("Mul", ["negative_rate", "positive_gate"], ["g"]),
        helper.make_node("MatMul", ["prefix", "g"], ["cumulative"]),
        helper.make_node("Exp", ["cumulative"], ["decay"], name="decay_exp"),
        helper.make_node("Neg", ["cumulative"], ["positive_cumulative"]),
        helper.make_node("Exp", ["positive_cumulative"], ["growth"], name="positive_exp"),
        helper.make_node("Exp", ["unrestricted"], ["unknown"], name="unknown_exp"),
    ], "sign_proof",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ["a", "unrestricted"]],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ["decay", "growth", "unknown"]],
        tensors)
    original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    lowered = onnx.ModelProto()
    lowered.CopyFrom(original)
    changes = lower_nonpositive_exp(lowered)
    assert [c["node"] for c in changes] == ["decay_exp"], changes
    onnx.checker.check_model(lowered, full_check=True)
    # Signed/unknown exponents must not be rewritten. A second pass is a no-op.
    assert lower_nonpositive_exp(lowered) == []
    baseline = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261013)
    for a in [np.linspace(-20, 6, 8, dtype=np.float32).reshape(shape),
              np.full(shape, -1000, dtype=np.float32),
              rng.uniform(-12, 6, shape).astype(np.float32)]:
        feed = {"a": a, "unrestricted": rng.uniform(-5, 5, shape).astype(np.float32)}
        reference, actual = baseline.run(None, feed), candidate.run(None, feed)
        for expected, observed in zip(reference, actual):
            assert np.isfinite(observed).all()
            np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    print("Stable-decay sign proof, exclusions, idempotence and ONNX numerical parity passed", flush=True)


if __name__ == "__main__":
    check()
