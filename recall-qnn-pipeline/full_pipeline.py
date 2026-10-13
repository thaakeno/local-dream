"""Cloud-hosted complete QNN dataflow against the official D3 runtime.

Individual AI Hub calls validate dataflow, not Android end-to-end latency.
Never describe their network wall time or summed profiles as phone speed.
"""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def qualify_pipeline(checkpoint, root, client, device, model_revision):
    import math
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights
    from vision_export import load_vision, prepare_inputs

    report = {"deployable": False, "status": "waiting_for_components", "cases": [],
              "scope": "all neural stages executed on hosted HTP; official CPU runtime is reference only",
              "timing_scope": "separate hosted component profiles; not a phone pipeline latency or loading benchmark"}
    def save():
        (root / "full-pipeline.json").write_text(json.dumps(report, indent=2))
    def event(message):
        print(message, flush=True)
    def read_report(run, artifact, filename):
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            metadata = json.loads(subprocess.check_output([
                "gh", "run", "view", run, "--repo", "thaakeno/local-dream", "--json", "status,conclusion"
            ], text=True))
            if metadata["status"] == "completed":
                if metadata["conclusion"] != "success":
                    raise RuntimeError("Required qualification did not pass: " + run)
                target = root / run
                subprocess.run(["gh", "run", "download", run, "--repo", "thaakeno/local-dream",
                                "--name", artifact, "--dir", str(target)], check=True)
                return json.loads((target / filename).read_text())
            event("Waiting for component qualification " + run)
            time.sleep(30)
        raise RuntimeError("Component qualification wait expired")
    save()
    fused = read_report(os.environ["D3_FUSED_RUN"], "recall-d3-nano-qnn-fused-qualification", "report.json")
    native = read_report(os.environ["D3_NATIVE_RUN"], "recall-d3-nano-qnn-native-vision", "native-vision.json")
    components = {c["name"]: c for c in fused["components"]}
    block_names = [f"block_{i:02d}_{i+3:02d}_l384" for i in range(0, 24, 4)]
    if any(name not in components for name in block_names):
        raise RuntimeError("All six decoder blocks are required")
    if fused.get("sequence_bucket") != 384 or fused.get("model_revision") != model_revision:
        raise RuntimeError("Decoder model or bucket contract changed")
    native_id = native["fusion_off"]["inference_url"].rstrip("/").split("/")[-1]
    vision_model = client.get_job(native_id).model
    report["vision_qualification"] = native["fusion_off"]["reference_comparison"]
    save()

    # Use the upstream prompt renderer, tokenizer, M-RoPE and decision contract.
    # No custom inference class patches or substitute prompt are introduced.
    sys.path.insert(0, str(checkpoint))
    from d3_runtime import D3, normalize_question
    official = D3.from_pretrained(str(checkpoint), device="cpu", local_files_only=True)
    official.backbone.float()
    official.processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 512*512}
    image_path = hf_hub_download("vllm-sr/d3-nano", "assets/example-receipt.png", revision=model_revision)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    cases = []
    for instruction in ["Does this image show a receipt?", "Does this image show a dog?"]:
        question = normalize_question({"type": "noul", "instructions": instruction})
        text = official.image_text("Evaluate the attached image.", question.rendered, 1)
        encoded = official.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        n = encoded["input_ids"].shape[1]
        if not 1 <= n <= 384:
            raise RuntimeError(f"Real prompt has {n} tokens and does not fit the qualified decoder bucket; no truncation")
        with torch.inference_mode():
            embeds, mrope = official.image_embeddings(encoded)
            cos, sin = official.backbone.language_model.rotary_emb(embeds, mrope)
            hidden = official.encoder(embeds, encoded["attention_mask"], mrope)[:, -1]
            logits = (hidden.float() @ official.readout.T).numpy().copy()
        cases.append({"encoded": {k: v.numpy().copy() for k, v in encoded.items()},
                      "cos": cos[0].numpy().copy(), "sin": sin[0].numpy().copy(),
                      "reference_logits": logits, "temperature": official.temperature, "length": n})
    del official, embeds, mrope, cos, sin, hidden, logits, encoded
    gc.collect()

    def infer(label, model, feed, timings):
        job = client.submit_inference_job(model, device=device, inputs={k: [v] for k,v in feed.items()},
            profile=True, options="--qairt_version 2.50", name="Recall_D3Nano_full_" + label + "_parity")
        event(label + ": " + job.url)
        if not job.wait().success:
            raise RuntimeError("Full pipeline component failed: " + label)
        data = job.download_output_data()
        if len(data) != 1 or len(next(iter(data.values()))) != 1:
            raise RuntimeError("Full pipeline output contract changed: " + label)
        value = next(iter(data.values()))[0]
        if not np.isfinite(value).all():
            raise RuntimeError("Nonfinite full pipeline component: " + label)
        timings.append({"stage": label, "job": job.url,
                        "profile": job.download_profile().get("execution_summary", {})})
        return value

    # Geometry calculations on host; learned position lookup/interpolation on HTP.
    first = cases[0]["encoded"]
    tower, cfg, table = load_vision(checkpoint, 1024)
    geometric = prepare_inputs(cfg, table, torch.from_numpy(first["pixel_values"]),
                               torch.from_numpy(first["image_grid_thw"]), 1024)
    del tower, table
    gc.collect()
    grid = torch.from_numpy(first["image_grid_thw"])
    indices, weights = get_vision_interpolation_indices_and_weights(grid,
        num_grid_per_side=math.isqrt(cfg.num_pos), mode="bilinear", align_corners=True,
        spatial_merge_size=cfg.merge, kwargs={})
    pos_indices = np.zeros((1024,4), dtype=np.int32)
    pos_weights = np.zeros((1024,4), dtype=np.float32)
    patches = geometric[-1]
    pos_indices[:patches] = indices.numpy().astype(np.int32)
    pos_weights[:patches] = weights.numpy()
    common_timings = []
    position = infer("positions", client.get_job("jgod4lr45").model,
                     {"indices": pos_indices, "interpolation_weights": pos_weights}, common_timings)
    vision_inputs = {k: v.numpy() for k,v in zip(["patches","pos_embeds","cos","sin","mask"], geometric[:-1])}
    vision_inputs["pos_embeds"] = position
    vision = infer("vision", vision_model, vision_inputs, common_timings)
    if vision.shape != (256,2048):
        raise RuntimeError("Vision token contract changed")
    report["shared_vision_stages"] = common_timings
    report["geometry"] = {"grid_thw": first["image_grid_thw"].tolist(), "vision_tokens": patches//4}
    save()
    configuration = json.loads((checkpoint / "config.json").read_text())
    for index, case in enumerate(cases):
        n = case["length"]
        ids = np.full((1,640), configuration["text_config"].get("pad_token_id") or 0, dtype=np.int32)
        ids[0,:n] = case["encoded"]["input_ids"][0]
        image_slots = ids == configuration["image_token_id"]
        if int(image_slots.sum()) != patches//4:
            raise RuntimeError("Prompt image placeholders do not match vision output")
        positions = np.zeros((1,640), dtype=np.int32)
        positions[image_slots] = np.arange(patches//4, dtype=np.int32)
        timings = []
        hidden = infer(f"case{index}_embedding", client.get_job("jp8j4e6z5").model,
            {"input_ids": ids, "vision": vision, "image_positions": positions}, timings)[:,:384].copy()
        valid = np.zeros(384,dtype=np.float32)
        valid[:n] = 1
        cos = np.repeat(case["cos"][-1:,:],384,axis=0)
        sin = np.repeat(case["sin"][-1:,:],384,axis=0)
        cos[:n], sin[:n] = case["cos"], case["sin"]
        for name in block_names:
            hidden = infer(f"case{index}_"+name, client.get_job(components[name]["inference_id"]).model,
                {"hidden": hidden,"cos":cos,"sin":sin,"key_valid":valid}, timings)
        padded = np.zeros((1,640,2048),dtype=np.float32)
        padded[:,:384] = hidden
        last = np.zeros((1,640),dtype=np.float32)
        last[0,n-1] = 1
        logits = infer(f"case{index}_readout", client.get_job("jp8j4e3q5").model,
                       {"hidden":padded,"last_onehot":last},timings)
        reference = case["reference_logits"]
        if logits.shape != reference.shape:
            raise RuntimeError("Final decision logit contract changed")
        def probabilities(value):
            x = value[0,:2].astype(np.float64) / case["temperature"]
            x = np.exp(x-x.max())
            return x/x.sum()
        actual, expected = probabilities(logits), probabilities(reference)
        item = {"case":index,"prompt_tokens":n,"reference_probabilities":expected.tolist(),
                "qnn_probabilities":actual.tolist(), "probability_max_abs_error":float(np.max(np.abs(actual-expected))),
                "same_decision":bool(actual.argmax()==expected.argmax()),
                "logit_rmse":float(np.sqrt(np.mean((logits.astype(np.float64)-reference)**2))),
                "stages":timings}
        report["cases"].append(item)
        save()
        event("Full QNN pipeline case: " + json.dumps({k:v for k,v in item.items() if k!='stages'}))
    report["status"] = "full_dataflow_executed_pending_quality_review_and_phone_latency"
    save()
