"""Cloud-only, isolated test of the measured DeltaNet exponential boundary.

The ratio is an algebraic candidate, not an approved model modification:
exp(x) = sigmoid(x) / sigmoid(-x). For x <= 0 its denominator is >= 0.5.
No clipping, output replacement, changed weights, or CPU model fallback.
"""
import json


def run_exp_repro(values, root, client, device):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    if not np.isfinite(values).all() or np.max(values) > 0:
        raise RuntimeError("Decay fixture must be finite and nonpositive")
    shape = list(values.shape)
    nodes = [
        helper.make_node("Exp", ["x"], ["direct"], name="direct_exp"),
        helper.make_node("Sigmoid", ["x"], ["positive"], name="sigmoid_x"),
        helper.make_node("Neg", ["x"], ["negative_x"], name="negate_x"),
        helper.make_node("Sigmoid", ["negative_x"], ["negative"], name="sigmoid_neg_x"),
        helper.make_node("Div", ["positive", "negative"], ["ratio"], name="algebraic_exp"),
    ]
    graph = helper.make_graph(nodes, "decay_exp_repro",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape)
                               for n in ["direct", "ratio"]])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = root / "decay_exp_repro.onnx"
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path), full_check=True)
    result = {"fixture_shape": shape, "fixture_min": float(values.min()),
              "fixture_max": float(values.max()), "deployable": False}

    def save():
        (root / "exp-repro.json").write_text(json.dumps(result, indent=2))

    compile_job = client.submit_compile_job(str(path), device=device,
        name="Recall_D3Nano_decay_exp_repro",
        options="--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16")
    result["compile_url"] = compile_job.url
    save()
    print("Isolated Exp compilation: " + compile_job.url, flush=True)
    if not compile_job.wait().success:
        raise RuntimeError("Isolated Exp compilation failed")
    link = client.submit_link_job([compile_job.get_target_model()], device=device,
        name="Recall_D3Nano_decay_exp_repro_HTP",
        options="--qnn_options default_graph_htp_optimizations=O=3")
    result["link_url"] = link.url
    save()
    if not link.wait().success:
        raise RuntimeError("Isolated Exp linking failed")
    inference = client.submit_inference_job(link.get_target_model(), device=device,
        inputs={"x": [values]}, profile=True, name="Recall_D3Nano_decay_exp_repro_parity")
    result["inference_url"] = inference.url
    save()
    if not inference.wait().success:
        raise RuntimeError("Isolated Exp inference failed")
    reference = np.exp(values.astype(np.float64))
    result["outputs"] = {}
    for name, batches in inference.download_output_data().items():
        if len(batches) != 1 or batches[0].shape != values.shape:
            raise RuntimeError("Isolated Exp output contract changed")
        value = batches[0].astype(np.float64)
        finite = np.isfinite(value)
        error = np.abs(value[finite] - reference[finite])
        result["outputs"][name] = {
            "finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "finite_min": float(value[finite].min()) if finite.any() else None,
            "finite_max": float(value[finite].max()) if finite.any() else None,
            "max_abs_error_finite_only": float(error.max()) if error.size else None,
            "rmse_finite_only": float(np.sqrt(np.mean(error * error))) if error.size else None,
        }
    result["profile"] = inference.download_profile()
    save()
    print("Isolated Exp results: " + json.dumps({k: v for k, v in result.items() if k != "profile"}), flush=True)
