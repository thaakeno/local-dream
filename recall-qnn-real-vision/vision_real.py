"""Cloud-only real-image accuracy control and QAIRT version comparison.

Uses the model author's public example, never Recall gallery images. Original
Transformers and tensor-export references must agree before testing the NPU.
"""
import gc
import json


def run_real_vision(checkpoint, load_weights, root, client, device):
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, Qwen3_5VisionConfig, Qwen3_5VisionModel
    from vision_export import load_vision, prepare_inputs

    report = {"deployable": False, "status": "reference", "scope": "public model example; no private media",
              "model_revision": "9d9b937f5f117f78c030ea7ba677c0b3d328ec58"}

    def save():
        (root / "vision-real.json").write_text(json.dumps(report, indent=2))

    def metrics(reference, value):
        a, b = reference.astype(np.float64).reshape(-1), value.astype(np.float64).reshape(-1)
        finite = np.isfinite(b)
        result = {"finite": bool(finite.all()), "nonfinite_count": int((~finite).sum())}
        if finite.all():
            error = a - b
            rmse, rms = float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(a**2)))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            result.update(max_abs_error=float(np.abs(error).max()), rmse=rmse,
                          relative_rmse=rmse / rms if rms else None,
                          cosine=float(a @ b / denominator) if denominator else None)
        return result

    path = hf_hub_download("vllm-sr/d3-nano", "assets/example-receipt.png", revision=report["model_revision"])
    image = Image.open(path).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    processor = AutoImageProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processed = processor(images=image, return_tensors="pt",
                          size={"shortest_edge": 1024, "longest_edge": 512 * 512})
    pixels, grid = processed["pixel_values"], processed["image_grid_thw"]
    tower, cfg, position_table = load_vision(checkpoint, 1024)
    inputs = prepare_inputs(cfg, position_table, pixels, grid, 1024)
    real_patches = inputs[-1]
    real_tokens = real_patches // cfg.merge**2
    names = ["patches", "pos_embeds", "cos", "sin", "mask"]
    feed = {k: v.numpy() for k, v in zip(names, inputs[:-1])}
    with torch.inference_mode():
        expected = tower(*inputs[:-1]).numpy()[:real_tokens].copy()
    report["input_geometry"] = {"resized_image": list(image.size), "grid_thw": grid.tolist(),
                                "real_patches": real_patches, "bucket_patches": 1024,
                                "real_vision_tokens": real_tokens}

    configuration = json.loads((checkpoint / "config.json").read_text())
    hf_cfg = Qwen3_5VisionConfig(**configuration["vision_config"])
    hf_cfg._attn_implementation = "eager"
    official = Qwen3_5VisionModel(hf_cfg).eval()
    official.load_state_dict(load_weights("visual."), strict=True, assign=True)
    with torch.inference_mode():
        official_output = official(pixels.float(), grid_thw=grid).pooler_output.numpy().copy()
    if expected.shape != official_output.shape:
        raise RuntimeError("Original Transformers versus exported vision shape mismatch")
    report["export_vs_official_fp32"] = metrics(official_output, expected)
    save()
    print("Real vision reference: " + json.dumps(report), flush=True)
    np.testing.assert_allclose(expected, official_output, rtol=0.003, atol=0.003)
    del official, official_output
    tower.half()
    with torch.inference_mode():
        half = tower(*(v.half() for v in inputs[:-1])).float().numpy()[:real_tokens].copy()
    report["cpu_half_vs_fp32"] = metrics(expected, half)
    del tower, half, position_table
    gc.collect()
    save()
    original_inference = client.get_job("jpy87xe8g")

    def infer(model, version):
        job = client.submit_inference_job(model, device=device, inputs={k: [v] for k, v in feed.items()},
            options="--qairt_version " + version, profile=True,
            name="Recall_D3Nano_vision_real_" + version.replace(".", "") + "_parity")
        report["inference_" + version] = job.url
        save()
        if not job.wait().success:
            raise RuntimeError("Real vision NPU inference failed for QAIRT " + version)
        outputs = job.download_output_data()
        if len(outputs) != 1 or len(next(iter(outputs.values()))) != 1:
            raise RuntimeError("Real vision compiled output contract changed")
        value = next(iter(outputs.values()))[0]
        if value.shape != (256, cfg.out_hidden):
            raise RuntimeError("Real vision compiled output shape changed")
        report["qnn_" + version] = metrics(expected, value[:real_tokens])
        report["profile_" + version] = job.download_profile()
        save()
        print("Real vision QAIRT " + version + ": " + json.dumps(report["qnn_" + version]), flush=True)

    infer(original_inference.model, "2.50")
    # Controlled runtime/compiler A/B: identical ONNX source and image inputs.
    supported = {f.api_version for f in client.get_frameworks() if f.name.upper() == "QAIRT"}
    if "2.51" not in supported:
        raise RuntimeError("QAIRT 2.51 is no longer available for the planned comparison")
    compile_job = client.submit_compile_job(client.get_job("jp3ow99zp").get_source_model(), device=device,
        name="Recall_D3Nano_vision_qairt251",
        options="--target_runtime qnn_dlc --qairt_version 2.51 --qnn_options default_graph_htp_precision=FLOAT16")
    report["compile_2.51"] = compile_job.url
    save()
    if not compile_job.wait().success:
        raise RuntimeError("QAIRT 2.51 vision compilation failed")
    link = client.submit_link_job([compile_job.get_target_model()], device=device,
        name="Recall_D3Nano_vision_qairt251_HTP",
        options="--qairt_version 2.51 --qnn_options default_graph_htp_optimizations=O=3")
    report["link_2.51"] = link.url
    save()
    if not link.wait().success:
        raise RuntimeError("QAIRT 2.51 vision linking failed")
    infer(link.get_target_model(), "2.51")
    report["status"] = "real_image_runtime_comparison_pending_quality_review"
    save()
