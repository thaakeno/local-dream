"""Observe the first invalid tensor in the NPU graph. Never patch model values."""
import json


def diagnose(model_factory, values, root, client, device):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import qai_hub as hub
    import torch
    from onnx import helper, TensorProto

    observations = {"phase": "float16_failure_localization", "cpu_is_reference_only": True,
                    "torch_fp16_modules": [], "graph_probes": []}
    model = model_factory().eval()

    def persist():
        (root / "diagnostic.json").write_text(json.dumps(observations, indent=2))

    def stats(value):
        a = value.detach().float().numpy() if isinstance(value, torch.Tensor) else value
        finite = np.isfinite(a)
        vals = a[finite]
        return {"shape": list(a.shape), "finite": bool(finite.all()),
                "nonfinite_count": int(a.size - finite.sum()),
                "finite_min": float(vals.min()) if vals.size else None,
                "finite_max": float(vals.max()) if vals.size else None}

    handles = []
    for name, module in model.named_modules():
        def observe(mod, args, output, module_name=name):
            if isinstance(output, torch.Tensor):
                observations["torch_fp16_modules"].append({"module": module_name, **stats(output)})
        handles.append(module.register_forward_hook(observe))
    model.half()
    with torch.inference_mode():
        model(*(v.half() if v.is_floating_point() else v for v in values.values()))
    for h in handles:
        h.remove()
    persist()
    print("Torch half reference module observations: " + json.dumps(observations["torch_fp16_modules"]), flush=True)
    del model
    # Reload original weights; half().float() would preserve FP16 rounding.
    model = model_factory().eval()
    directory = root / "decoder_probe.onnx"
    directory.mkdir(exist_ok=True)
    path = directory / "decoder_probe.onnx"
    torch.onnx.export(model.eval(), tuple(values.values()), str(path), opset_version=17,
                      input_names=list(values), output_names=["hidden_out"], dynamo=False)
    graph = onnx.load(str(path))
    original_nodes = list(graph.graph.node)
    selectors = []
    matmul_count = 0
    for node in original_nodes:
        if node.op_type == "MatMul":
            matmul_count += 1
        selected = node.op_type in {"Exp", "Softplus", "Conv", "Sigmoid", "Sqrt", "Reciprocal", "ReduceMean"}
        selected |= node.op_type == "MatMul" and (matmul_count <= 3 or matmul_count % 8 == 0)
        selected |= node.op_type == "Add" and node.output[0] == "hidden_out"
        if selected:
            selectors.append(node)
    if len(selectors) > 64:
        raise RuntimeError("Diagnostic probe budget exceeded")
    for i, node in enumerate(selectors):
        name = f"probe_{i:03d}"
        abs_name, max_name = name + "_abs", name + "_max"
        graph.graph.node.extend([
            helper.make_node("Abs", [node.output[0]], [abs_name], name=name + "_abs_op"),
            helper.make_node("ReduceMax", [abs_name], [max_name], keepdims=1, name=name + "_reduce_op"),
            helper.make_node("Reshape", [max_name, "probe_shape"], [name], name=name + "_reshape_op"),
        ])
        graph.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]))
        observations["graph_probes"].append({"probe": name, "node": node.name, "op": node.op_type, "tensor": node.output[0]})
    graph.graph.initializer.append(helper.make_tensor("probe_shape", TensorProto.INT64, [1], [1]))
    onnx.save_model(graph, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="decoder_probe.data", size_threshold=1024)
    del graph
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {i.name: values[i.name].numpy() for i in session.get_inputs()}
    ref = session.run(None, feed)
    outputs = [x.name for x in session.get_outputs()]
    observations["reference_outputs"] = {n: stats(v) for n, v in zip(outputs, ref)}
    if not all(np.isfinite(x).all() for x in ref):
        raise RuntimeError("Instrumented ONNX reference itself is invalid")
    del session, model, ref
    compile_job = client.submit_compile_job(
        str(directory), device=device, name="Recall_D3Nano_decoder00_probe",
        options="--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16",
    )
    observations["compile_url"] = compile_job.url
    persist()
    print("Diagnostic compile: " + compile_job.url, flush=True)
    if not compile_job.wait().success:
        raise RuntimeError("Diagnostic compilation failed")
    link = client.submit_link_job([compile_job.get_target_model()], device=device,
                                 name="Recall_D3Nano_decoder00_probe_HTP",
                                 options="--qnn_options default_graph_htp_optimizations=O=3")
    observations["link_url"] = link.url
    persist()
    if not link.wait().success:
        raise RuntimeError("Diagnostic context linking failed")
    inference = client.submit_inference_job(link.get_target_model(), device=device,
                                           inputs={k: [v] for k, v in feed.items()}, profile=True,
                                           name="Recall_D3Nano_decoder00_probe_parity")
    observations["inference_url"] = inference.url
    persist()
    if not inference.wait().success:
        raise RuntimeError("Diagnostic NPU inference failed")
    outputs = inference.download_output_data()
    observations["npu_outputs"] = {k: [stats(v) for v in batches] for k, batches in outputs.items()}
    observations["profile"] = inference.download_profile()
    persist()
    print("NPU diagnostic outputs: " + json.dumps(observations["npu_outputs"]), flush=True)
