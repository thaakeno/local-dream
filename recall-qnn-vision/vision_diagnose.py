"""Cloud-only layerwise observation of the measured vision accuracy failure."""
import gc
import json
import os


def diagnose_vision(model_factory, root, client, device):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn
    import torch.nn.functional as F

    original = client.get_job("jpy87xe8g")
    entries = original.inputs.download()
    if any(len(batches) != 1 for batches in entries.values()):
        raise RuntimeError("Expected the exact single-fixture vision test")
    names = ["patches", "pos_embeds", "cos", "sin", "mask"]
    values = {name: torch.from_numpy(entries[name][0]) for name in names}
    report = {"status": "observing", "deployable": False, "source_inference": original.url,
              "scope": "exact original synthetic fixture; CPU is reference only, not app fallback"}

    def save():
        (root / "vision-diagnostic.json").write_text(json.dumps(report, indent=2))

    def compare(expected, actual):
        a, b = np.asarray(expected, dtype=np.float64).reshape(-1), np.asarray(actual, dtype=np.float64).reshape(-1)
        finite = np.isfinite(b)
        result = {"finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
                  "reference_max_abs": float(np.abs(a).max())}
        if finite.all():
            error = a - b
            rmse = float(np.sqrt(np.mean(error * error)))
            rms = float(np.sqrt(np.mean(a * a)))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            result.update(max_abs_error=float(np.abs(error).max()), rmse=rmse,
                          relative_rmse=rmse / rms if rms else None,
                          cosine=float(a @ b / denominator) if denominator else None)
        return result

    class ObservedVision(nn.Module):
        def __init__(self, vision):
            super().__init__()
            self.vision = vision
            self.selected = {0, 3, 7, 11, 15, 19, 23}

        def forward(self, patches, pos_embeds, cos, sin, mask):
            v = self.vision
            x = v.patch_embed(patches) + pos_embeds
            stages = [x]
            for i, block in enumerate(v.blocks):
                x = block(x, cos, sin, mask)
                if i in self.selected:
                    stages.append(x)
            normalized = v.merger_norm(x).reshape(v.n_patches // v.cfg.merge**2, -1)
            pre_gelu = v.merger_fc1(normalized)
            result = v.merger_fc2(F.gelu(pre_gelu))
            return (result, *stages, normalized, pre_gelu)

    output_names = ["final", "patch_position", "block_00", "block_03", "block_07", "block_11",
                    "block_15", "block_19", "block_23", "merger_norm", "merger_fc1"]
    class ObservedFirstBlock(nn.Module):
        """Observe unchanged first-block arithmetic; score probes cover 16 queries."""
        def __init__(self, vision):
            super().__init__()
            self.vision = vision

        def forward(self, patches, pos_embeds, cos, sin, mask):
            v = self.vision
            b = v.blocks[0]
            x = v.patch_embed(patches) + pos_embeds
            h = b.norm1(x)
            packed = b.qkv(h)
            qkv = packed.reshape(x.shape[0], 3, b.cfg.heads, b.cfg.head_dim)
            q, k, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            def rotate(t):
                half = t.shape[-1] // 2
                return torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            c, s = cos[:, None, :], sin[:, None, :]
            q = q * c + rotate(q) * s
            k = k * c + rotate(k) * s
            qh = q.permute(1, 0, 2)[None]
            kh = k.permute(1, 0, 2)[None]
            vh = value.permute(1, 0, 2)[None]
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * b.scale
            probability = torch.softmax(scores + mask, dim=-1)
            attention = torch.matmul(probability, vh)[0].permute(1, 0, 2).reshape(x.shape[0], b.cfg.hidden)
            projected = b.proj(attention)
            residual = x + projected
            normalized = b.norm2(residual)
            pre_activation = b.fc1(normalized)
            activation = F.gelu(pre_activation, approximate="tanh" if b.cfg.act == "gelu_pytorch_tanh" else "none")
            delta = b.fc2(activation)
            result = residual + delta
            return (result, x, h, packed, q, k, scores[:, :, :16, :],
                    probability[:, :, :16, :], attention, projected, residual,
                    normalized, pre_activation, activation, delta)

    first_block = os.environ.get("D3_VISION_FIRST_BLOCK") == "true"
    observed = ObservedFirstBlock if first_block else ObservedVision
    if first_block:
        output_names = ["final", "patch_position", "norm1", "qkv", "rotated_q", "rotated_k",
                        "score_query_sample", "probability_query_sample", "attention", "projection",
                        "attention_residual", "norm2", "mlp_fc1", "mlp_activation", "mlp_fc2"]
        report["scope"] += "; first block, attention-score probes sample first 16 queries"
    model = observed(model_factory()).eval()
    with torch.inference_mode():
        ref = [x.numpy().copy() for x in model(*values.values())]
    if len(ref) != len(output_names) or any(not np.isfinite(x).all() for x in ref):
        raise RuntimeError("Vision reference output contract is invalid")
    if first_block:
        with torch.inference_mode():
            native = model.vision.blocks[0](torch.from_numpy(ref[1]), values["cos"], values["sin"], values["mask"]).numpy()
        np.testing.assert_allclose(ref[0], native, rtol=0.00001, atol=0.00001)
        report["manual_vs_original_block"] = compare(native, ref[0])
        del native
    else:
        device_original = original.download_output_data()
        if len(device_original) != 1:
            raise RuntimeError("Original vision output contract changed")
        original_value = next(iter(device_original.values()))[0]
        report["original_qnn"] = compare(ref[0], original_value)
        del device_original, original_value

    # Compare unchanged original weights in half precision before instrumenting
    # the device. Reload FP32 parameters afterward; never half().float() them.
    model.half()
    with torch.inference_mode():
        half = [x.float().numpy() for x in model(*(v.half() for v in values.values()))]
    report["cpu_half_stages"] = {name: compare(a, b) for name, a, b in zip(output_names, ref, half)}
    save()
    print("Vision CPU half stage comparisons: " + json.dumps(report["cpu_half_stages"]), flush=True)
    del model, half
    gc.collect()
    model = observed(model_factory()).eval()
    directory = root / "vision_observed.onnx"
    directory.mkdir(exist_ok=True)
    path = directory / "vision_observed.onnx"
    torch.onnx.export(model, tuple(values.values()), str(path), input_names=names,
                      output_names=output_names, opset_version=17, dynamo=False)
    graph = onnx.load(str(path))
    for extra in directory.iterdir():
        if extra != path:
            extra.unlink()
    onnx.save_model(graph, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="vision_observed.data", size_threshold=1024)
    del graph, model
    gc.collect()
    onnx.checker.check_model(str(path), full_check=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in values.items()}
    checked = session.run(None, feed)
    for a, b in zip(ref, checked):
        np.testing.assert_allclose(b, a, rtol=0.003, atol=0.003)
    del session, checked
    gc.collect()
    compile_job = client.submit_compile_job(str(directory), device=device,
        name="Recall_D3Nano_vision_observed",
        options="--target_runtime qnn_dlc --output_names " + ",".join(output_names) +
                " --qnn_options default_graph_htp_precision=FLOAT16")
    report["compile_url"] = compile_job.url
    save()
    print("Vision diagnostic compile: " + compile_job.url, flush=True)
    if not compile_job.wait().success:
        raise RuntimeError("Vision diagnostic compilation failed")
    link = client.submit_link_job([compile_job.get_target_model()], device=device,
        name="Recall_D3Nano_vision_observed_HTP", options="--qnn_options default_graph_htp_optimizations=O=3")
    report["link_url"] = link.url
    save()
    if not link.wait().success:
        raise RuntimeError("Vision diagnostic linking failed")
    inference = client.submit_inference_job(link.get_target_model(), device=device,
        inputs={k: [v] for k, v in feed.items()}, profile=True,
        name="Recall_D3Nano_vision_observed_parity")
    report["inference_url"] = inference.url
    save()
    if not inference.wait().success:
        raise RuntimeError("Vision diagnostic NPU execution failed")
    outputs = inference.download_output_data()
    if set(outputs) != set(output_names):
        raise RuntimeError("Vision diagnostic compiled output names changed")
    report["npu_stages"] = {}
    for name, reference in zip(output_names, ref):
        batches = outputs[name]
        if len(batches) != 1 or batches[0].shape != reference.shape:
            raise RuntimeError("Vision diagnostic compiled output shape changed")
        report["npu_stages"][name] = compare(reference, batches[0])
    report["profile"] = inference.download_profile()
    report["status"] = "observed_pending_root_cause_review"
    save()
    print("Vision NPU stage comparisons: " + json.dumps(report["npu_stages"]), flush=True)
