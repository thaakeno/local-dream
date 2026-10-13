"""Cloud-only full-model QNN qualification. Never run on the user's workstation.

All weights/source downloads are pinned and stay on the GitHub runner. This does
not publish a Recall download or claim numeric equivalence from compilation.
Published FluidInference tensor graphs are fetched under their Apache-2.0 card;
original model LICENSE/NOTICE are retained with any resulting context bundle.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from collections import Counter

MODEL_ID = "vllm-sr/d3-nano"
MODEL_REV = "9d9b937f5f117f78c030ea7ba677c0b3d328ec58"
GRAPH_ID = "FluidInference/d3-coreml"
GRAPH_REV = "62b6a996e15342bdd3d608030d8f445f11a52367"
LENGTH = 640
PATCHES = 1024
CHUNK = 64


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Cloud runner required; local model downloads are prohibited.")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import qai_hub as hub
    import torch
    from torch import nn
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open

    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    root = Path(os.environ["D3_WORK"])
    root.mkdir(parents=True, exist_ok=True)
    contexts = root / "contexts"
    contexts.mkdir(exist_ok=True)
    report = {
        "status": "preflight", "deployable": False,
        "model": MODEL_ID, "model_revision": MODEL_REV,
        "graph_source": GRAPH_ID, "graph_revision": GRAPH_REV,
        "device": os.environ["QAI_DEVICE"],
        "sequence_bucket": LENGTH, "vision_patch_bucket": PATCHES,
        "delta_chunk_size": CHUNK,
        "vision_tokens": PATCHES // 4, "precision": "QNN HTP FLOAT16 request",
        "validation_scope": "PyTorch/ONNX graph parity and hosted QNN component inference; end-to-end official-model/ranking parity still required",
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "onnx", "onnxruntime", "qai-hub", "transformers"]},
        "components": [],
    }
    reuse = {}
    if os.environ.get("D3_REUSE_REPORT"):
        prior = json.loads(Path(os.environ["D3_REUSE_REPORT"]).read_text())
        contract = ["model", "model_revision", "graph_source", "graph_revision", "device",
                    "sequence_bucket", "vision_patch_bucket", "precision"]
        if any(prior.get(key) != report.get(key) for key in contract):
            raise RuntimeError("Previous cloud run has a different model/export contract")
        reuse = {c["name"]: c for c in prior["components"]}

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2))

    def event(message):
        print(message, flush=True)

    save()
    try:
        client = hub.Client(hub.ClientConfig(api_token=os.environ["QAI_HUB_API_TOKEN"]))
        matches = client.get_devices(name=os.environ["QAI_DEVICE"])
        if not matches:
            raise RuntimeError("AI Hub target device is unavailable")
        report["device_attributes"] = [str(d) for d in matches]
        event("AI Hub authenticated; target device found")
        save()
        # Only original model files and the three tensor-graph sources, never
        # CoreML weights. Downloads take place exclusively on this runner.
        checkpoint = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REV,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.jinja", "LICENSE", "NOTICE"],
        ))
        source = root / "upstream"
        source.mkdir(exist_ok=True)
        report["source_sha256"] = {}
        for filename in ["qwen35_export.py", "vision_export.py", "d3_graph.py"]:
            src = Path(hf_hub_download(GRAPH_ID, "nano/python/" + filename, revision=GRAPH_REV))
            payload = src.read_bytes()
            report["source_sha256"][filename] = hashlib.sha256(payload).hexdigest()
            (source / filename).write_bytes(payload)
        for filename in ["LICENSE", "NOTICE"]:
            (contexts / filename).write_bytes((checkpoint / filename).read_bytes())
        (contexts / "SOURCE.json").write_text(json.dumps({
            "model": MODEL_ID, "revision": MODEL_REV,
            "tensor_graph_source": GRAPH_ID, "tensor_graph_revision": GRAPH_REV,
            "tensor_graph_license": "Apache-2.0 (published repository metadata)",
        }, indent=2))
        sys.path.insert(0, str(source))
        from qwen35_export import DecoderLayer, RMSNorm, TextConfig
        from vision_export import load_vision

        config = json.loads((checkpoint / "config.json").read_text())
        decision = json.loads((checkpoint / "decision_config.json").read_text())
        if decision["attention_mode"] != "noncausal_full_attention" or decision["pooling"] != "last":
            raise RuntimeError("Unexpected D3 attention/pooling contract")
        cfg = TextConfig(config["text_config"])
        shards = sorted(checkpoint.glob("model*.safetensors"))

        def weights(prefix):
            state = {}
            for shard in shards:
                with safe_open(shard, framework="pt") as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            state[key[len(prefix):]] = f.get_tensor(key).float()
            if not state:
                raise RuntimeError("No weights for " + prefix)
            return state

        class Layer(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.layer = DecoderLayer(cfg, index, LENGTH, CHUNK)
                self.layer.load_state_dict(weights(f"language_model.layers.{index}."), strict=True, assign=True)

            def forward(self, hidden, cos, sin, key_valid):
                # Full-attention sees all real keys. DeltaNet remains causal.
                mask = ((1.0 - key_valid) * -1e4)[None, None, None, :]
                return self.layer(hidden, cos, sin, mask)

        class Readout(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
                self.norm.load_state_dict(weights("language_model.norm."), strict=True, assign=True)
                with safe_open(checkpoint / "readout.safetensors", framework="pt") as f:
                    self.register_buffer("readout", f.get_tensor("weight").float())

            def forward(self, hidden, last_onehot):
                # Keep all 255 answer codes. The host restricts the requested
                # options and applies the upstream temperature, never a vocab LM head.
                final = self.norm(torch.matmul(last_onehot, hidden[0]))
                return torch.matmul(final, self.readout.T)

        class Embedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding.from_pretrained(weights("language_model.embed_tokens.")["weight"], freeze=True)

            def forward(self, input_ids, vision, image_positions):
                # QNN performs both table gather and the visual-token splice.
                text = self.embed(input_ids)
                selected = torch.index_select(vision, 0, image_positions.reshape(-1)).reshape(1, LENGTH, cfg.hidden_size)
                is_image = (input_ids == config["image_token_id"]).unsqueeze(-1)
                return torch.where(is_image, selected, text)

        def qualify(name, model, values, output_name):
            """Fail closed on export, compilation or non-finite inference output.

            Component numeric errors are recorded, not misrepresented as a
            full-model quality certificate. No deployable flag is set here.
            """
            started = time.monotonic()
            component = {"name": name, "status": "exporting"}
            report["components"].append(component)
            save()
            event("Exporting " + name)
            model = model.eval()
            inputs = tuple(values.values())
            with torch.inference_mode():
                expected = model(*inputs).numpy()
            if not np.isfinite(expected).all():
                raise RuntimeError(name + ": non-finite PyTorch reference")
            directory = root / (name + ".onnx")
            directory.mkdir(exist_ok=True)
            target = directory / (name + ".onnx")
            torch.onnx.export(
                model, inputs, str(target), input_names=list(values), output_names=[output_name],
                opset_version=17, do_constant_folding=True, dynamo=False,
                external_data=True,
            )
            graph = onnx.load(str(target))
            component["onnx_node_count"] = len(graph.graph.node)
            component["onnx_ops"] = dict(Counter(node.op_type for node in graph.graph.node))
            component["input_shapes"] = {k: list(v.shape) for k, v in values.items()}
            component["output_shape"] = list(expected.shape)
            # AI Hub external-data format: exactly one .onnx plus one .data.
            for extra in directory.iterdir():
                if extra != target:
                    extra.unlink()
            onnx.save_model(graph, str(target), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=name + ".data", size_threshold=1024)
            del graph
            onnx.checker.check_model(str(target), full_check=True)
            # A prior job is reusable only for these exact graph bytes and inputs.
            # Revision/shape checks alone would incorrectly reuse a failed graph
            # after changing its numerical lowering or seeded test fixture.
            graph_hash = hashlib.sha256()
            for part in sorted(directory.iterdir()):
                graph_hash.update(part.name.encode())
                with part.open("rb") as stream:
                    graph_hash.update(hashlib.file_digest(stream, "sha256").digest())
            component["export_sha256"] = graph_hash.hexdigest()
            session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"],
                                           sess_options=ort.SessionOptions())
            # The exporter removes unused RoPE/mask inputs on DeltaNet layers.
            feed = {i.name: values[i.name].numpy() for i in session.get_inputs()}
            feed_hash = hashlib.sha256()
            for key, value in sorted(feed.items()):
                feed_hash.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
                feed_hash.update(value.tobytes(order="C"))
            component["fixture_sha256"] = feed_hash.hexdigest()
            actual = session.run(None, feed)[0]
            err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            component["onnx_max_abs_error"] = float(err.max())
            component["onnx_rmse"] = float(np.sqrt(np.mean(err * err)))
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0.003, atol=0.003):
                raise RuntimeError(name + ": ONNX/PyTorch parity failed")
            del session, model
            gc.collect()
            component["status"] = "submitting"
            save()
            specs = {k: (tuple(v.shape), "int32" if v.dtype == torch.int32 else "float32") for k, v in values.items() if k in feed}
            previous = reuse.get(name, {})
            if any(previous.get(key) != component[key] for key in ("export_sha256", "fixture_sha256")):
                previous = {}
            compile_job = (client.get_job(previous["compile_id"]) if previous.get("compile_id") else
                           client.submit_compile_job(
                               model=str(directory), device=hub.Device(os.environ["QAI_DEVICE"]),
                               name="Recall_D3Nano_" + name, input_specs=specs,
                               options="--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16",
                           ))
            component["compile_id"] = compile_job.job_id
            component["compile_url"] = compile_job.url
            event("AI Hub compile " + name + ": " + compile_job.url)
            save()
            status = compile_job.wait()
            component["compile_status"] = str(status)
            save()
            if not status.success:
                raise RuntimeError(name + ": AI Hub compilation failed; see " + compile_job.url)
            link_job = (client.get_job(previous["link_id"]) if previous.get("link_id") else
                        client.submit_link_job(
                            models=[compile_job.get_target_model()], device=hub.Device(os.environ["QAI_DEVICE"]),
                            name="Recall_D3Nano_" + name + "_HTP",
                            options="--qnn_options default_graph_htp_optimizations=O=3",
                        ))
            component["link_id"] = link_job.job_id
            component["link_url"] = link_job.url
            save()
            if not link_job.wait().success:
                raise RuntimeError(name + ": HTP context linking failed; see " + link_job.url)
            context = link_job.get_target_model()
            context_path = contexts / (name + ".bin")
            link_job.download_target_model(str(context_path))
            import zipfile
            if zipfile.is_zipfile(context_path):
                with zipfile.ZipFile(context_path) as archive:
                    candidates = [n for n in archive.namelist() if n.endswith(".bin") and not n.endswith("/")]
                    if len(candidates) != 1:
                        raise RuntimeError(name + ": ambiguous context archive")
                    payload = archive.read(candidates[0])
                context_path.write_bytes(payload)
            component["context_bytes"] = context_path.stat().st_size
            with context_path.open("rb") as context_file:
                component["context_sha256"] = hashlib.file_digest(context_file, "sha256").hexdigest()
            inference = (client.get_job(previous["inference_id"]) if previous.get("inference_id") else
                         client.submit_inference_job(
                             model=context, device=hub.Device(os.environ["QAI_DEVICE"]),
                             inputs={k: [v] for k, v in feed.items()}, profile=True,
                             name="Recall_D3Nano_" + name + "_parity",
                         ))
            component["inference_id"] = inference.job_id
            component["inference_url"] = inference.url
            save()
            if not inference.wait().success:
                raise RuntimeError(name + ": hosted NPU inference failed; see " + inference.url)
            outputs = inference.download_output_data()
            if outputs is None or len(outputs) != 1:
                raise RuntimeError(name + ": expected exactly one compiled output")
            compiled_output_name, batches = next(iter(outputs.items()))
            if len(batches) != 1:
                raise RuntimeError(name + ": unexpected compiled output batch count")
            device_out = batches[0]
            if device_out.shape != expected.shape:
                raise RuntimeError(name + ": compiled output shape differs from the ONNX contract")
            component["compiled_output_name"] = compiled_output_name
            device_err = np.abs(device_out.astype(np.float64) - expected.astype(np.float64))
            component["qnn_max_abs_error"] = float(device_err.max())
            component["qnn_rmse"] = float(np.sqrt(np.mean(device_err * device_err)))
            component["qnn_finite"] = bool(np.isfinite(device_out).all())
            # Download profile is JSON/metadata; stays on the cloud runner.
            if hasattr(inference, "download_profile"):
                profile = inference.download_profile()
                (root / (name + "-profile.json")).write_text(json.dumps(profile, indent=2, default=str))
            if not component["qnn_finite"]:
                raise RuntimeError(name + ": non-finite QNN output")
            component["status"] = "compiled_and_profiled_numeric_review_required"
            component["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save()
            event("Compiled and profiled " + name + "; numeric review required")
            # Keep only the small report and QNN context; don't accumulate ONNX
            # weights on the limited runner disk after successful conversion.
            import shutil
            shutil.rmtree(directory)

        hidden = torch.randn(1, LENGTH, cfg.hidden_size) * 0.1
        cos = torch.ones(LENGTH, cfg.rotary_dim)
        sin = torch.zeros_like(cos)
        valid = torch.zeros(LENGTH)
        valid[:384] = 1
        layer_inputs = {"hidden": hidden, "cos": cos, "sin": sin, "key_valid": valid}
        if os.environ.get("D3_DIAGNOSE_LAYER00") == "true":
            from diagnose import diagnose
            diagnose(lambda: Layer(0), layer_inputs, root, client, hub.Device(os.environ["QAI_DEVICE"]))
            report["status"] = "diagnostic_complete_not_deployable"
            save()
            return
        # Qualify both backbone layer types before spending cloud resources on
        # all remaining layers. This is a full-model feasibility gate, not a
        # vision-only product or a substitute model.
        qualify("decoder_00", Layer(0), layer_inputs, "hidden_out")
        qualify("decoder_03", Layer(3), layer_inputs, "hidden_out")
        vision, vcfg, _ = load_vision(checkpoint, PATCHES)
        vision_inputs = {
            "patches": torch.randn(PATCHES, vcfg.patch_dim) * 0.25,
            "pos_embeds": torch.randn(PATCHES, vcfg.hidden) * 0.1,
            "cos": torch.ones(PATCHES, vcfg.head_dim),
            "sin": torch.zeros(PATCHES, vcfg.head_dim),
            "mask": torch.zeros(1, 1, PATCHES, PATCHES),
        }
        qualify("vision_p1024", vision, vision_inputs, "tokens")
        del vision
        last = torch.zeros(1, LENGTH)
        last[0, 383] = 1
        qualify("decision_readout", Readout(), {"hidden": hidden, "last_onehot": last}, "code_logits")
        embed = Embedding()
        ids = torch.ones(1, LENGTH, dtype=torch.int32)
        ids[:, 10:266] = config["image_token_id"]
        positions = torch.zeros(1, LENGTH, dtype=torch.int32)
        positions[:, 10:266] = torch.arange(256, dtype=torch.int32)
        qualify("token_embedding", embed, {"input_ids": ids, "vision": torch.randn(256, cfg.hidden_size) * 0.1,
                                           "image_positions": positions}, "hidden")
        del embed
        for index in range(cfg.num_layers):
            if index not in (0, 3):
                qualify(f"decoder_{index:02d}", Layer(index), layer_inputs, "hidden_out")
        report["status"] = "complete_contexts_pending_end_to_end_validation"
        # Explicitly keep deployable false until real image prompt, M-RoPE,
        # probability and ranking parity plus device/runtime checks pass.
        save()
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = str(exc)
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
