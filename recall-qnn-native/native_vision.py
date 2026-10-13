"""Cloud-only native HTP fusion-control experiment using the original DLC."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile


def qualify_native_vision(model_factory, root, client, device):
    import numpy as np
    import torch

    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Native SDK and model downloads are cloud-only")
    report = {"deployable": False, "status": "preparing", "sdk": "2.50.0.260828",
              "source_compile": "jp3ow99zp", "source_fixture": "jpy87xe8g",
              "scope": "unchanged original vision DLC; native activation-fusion A/B; CPU reference only"}
    def save():
        (root / "native-vision.json").write_text(json.dumps(report, indent=2))
    def event(message):
        print(message, flush=True)
    def command(args, environment):
        result = subprocess.run([str(x) for x in args], env=environment, capture_output=True,
                                text=True, errors="replace", timeout=1800)
        (root / (Path(str(args[0])).name + ".log")).write_text(result.stdout + result.stderr)
        event((result.stdout + result.stderr)[-10000:])
        if result.returncode:
            raise RuntimeError(Path(str(args[0])).name + " failed: " + str(result.returncode))
    def download_model(model, destination, suffix):
        downloaded = model.download(str(destination))
        path = Path(downloaded) if downloaded else destination
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                files = [n for n in z.namelist() if n.endswith(suffix)]
                # DLC is itself a ZIP container, unlike an AI Hub download
                # wrapper. If it contains no nested DLC, feed it unchanged to
                # the SDK parser, which validates the actual DLC schema.
                if not files and suffix == ".dlc":
                    event("Direct DLC container entries: " + json.dumps(z.namelist()[:20]))
                    return path
                if len(files) != 1:
                    raise RuntimeError("Unexpected model archive contract")
                output = destination.with_name(destination.stem + "_unwrapped" + suffix)
                with z.open(files[0]) as source, output.open("wb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)
                return output
        return path
    def metrics(a, b):
        if a.shape != b.shape:
            raise RuntimeError("Compiled vision output shape changed")
        a, b = a.astype(np.float64).reshape(-1), b.astype(np.float64).reshape(-1)
        finite = bool(np.isfinite(b).all())
        result = {"finite": finite}
        if finite:
            rmse = float(np.sqrt(np.mean((a-b)**2)))
            result.update(max_abs_error=float(np.max(np.abs(a-b))), rmse=rmse,
                          relative_rmse=rmse / float(np.sqrt(np.mean(a*a))),
                          cosine=float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
        return result

    save()
    version = report["sdk"]
    url = f"https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{version}/v{version}.zip"
    archive = root / "sdk.zip"
    event("Downloading official QAIRT SDK on cloud runner only")
    urllib.request.urlretrieve(url, archive)
    report["sdk_archive_sha256"] = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    sdk_dir = root / "sdk"
    with zipfile.ZipFile(archive) as z:
        # Keep only host compilation tools/libraries and public HTP config docs.
        for item in z.infolist():
            if not any(s in item.filename for s in ["/bin/x86_64-linux-clang/", "/lib/x86_64-linux-clang/", "/include/QNN/HTP/"]):
                continue
            resolved = (sdk_dir / item.filename).resolve()
            if not resolved.is_relative_to(sdk_dir.resolve()):
                raise RuntimeError("Unsafe SDK archive path")
            z.extract(item, sdk_dir)
            if "/bin/" in item.filename and resolved.is_file():
                resolved.chmod(0o755)
    archive.unlink()
    tools = list(sdk_dir.rglob("qnn-context-binary-generator"))
    if len(tools) != 1:
        raise RuntimeError("SDK host tool contract changed")
    binary_dir = tools[0].parent
    sdk = binary_dir.parent.parent
    libraries = sdk / "lib/x86_64-linux-clang"
    environment = dict(os.environ, LD_LIBRARY_PATH=str(libraries), QNN_SDK_ROOT=str(sdk))
    baseline = download_model(client.get_job("jp1ovjm25").get_target_model(), root / "baseline.bin", ".bin")
    metadata = root / "baseline-metadata.json"
    command([binary_dir / "qnn-context-binary-utility", "--context_binary", baseline, "--json_file", metadata], environment)
    info = json.loads(metadata.read_text())
    graph_names = set()
    def find_names(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"graphName", "graph_name"} and isinstance(v, str):
                    graph_names.add(v)
                find_names(v)
        elif isinstance(value, list):
            for v in value:
                find_names(v)
    find_names(info)
    if len(graph_names) != 1:
        event("Context metadata: " + json.dumps(info)[:15000])
        raise RuntimeError("Could not establish the exact vision graph name")
    graph_name = next(iter(graph_names))
    report["graph_name"] = graph_name
    dlc = download_model(client.get_job("jp3ow99zp").get_target_model(), root / "vision.dlc", ".dlc")
    report["dlc_sha256"] = hashlib.file_digest(dlc.open("rb"), "sha256").hexdigest()
    fixture = client.get_job("jpy87xe8g").inputs.download()
    names = ["patches", "pos_embeds", "cos", "sin", "mask"]
    model = model_factory().eval()
    with torch.inference_mode():
        reference = model(*(torch.from_numpy(fixture[n][0]) for n in names)).numpy().copy()
    del model
    gc.collect()
    for enabled in [False, True]:
        label = "fusion_on" if enabled else "fusion_off"
        graph_config = root / (label + ".json")
        graph_config.write_text(json.dumps({"graphs": [{"graph_names": [graph_name],
            "fp16_relaxed_precision": 1, "advanced_activation_fusion": enabled, "O": 3}],
            "devices": [{"device_id": 0, "soc_id": 87, "dsp_arch": "v81"}]}))
        backend = root / (label + "_backend.json")
        backend.write_text(json.dumps({"backend_extensions": {
            "shared_library_path": str(libraries / "libQnnHtpNetRunExtensions.so"),
            "config_file_path": str(graph_config)}}))
        started = time.monotonic()
        command([tools[0], "--backend", libraries / "libQnnHtp.so", "--model", libraries / "libQnnModelDlc.so",
                 "--dlc_path", dlc, "--config_file", backend, "--binary_file", label,
                 "--output_dir", root, "--log_level", "info"], environment)
        context = root / (label + ".bin")
        entry = {"prepare_seconds": time.monotonic() - started, "advanced_activation_fusion": enabled,
                 "context_bytes": context.stat().st_size,
                 "context_sha256": hashlib.file_digest(context.open("rb"), "sha256").hexdigest()}
        report[label] = entry
        save()
        inference = client.submit_inference_job(str(context), device=device, inputs=fixture,
            options="--qairt_version 2.50", profile=True,
            name="Recall_D3Nano_vision_native_" + label + "_parity")
        entry["inference_url"] = inference.url
        save()
        event("Native vision " + label + ": " + inference.url)
        if not inference.wait().success:
            raise RuntimeError("Native vision hosted execution failed: " + label)
        outputs = inference.download_output_data()
        if len(outputs) != 1 or len(next(iter(outputs.values()))) != 1:
            raise RuntimeError("Native vision output contract changed")
        entry["reference_comparison"] = metrics(reference, next(iter(outputs.values()))[0])
        entry["profile"] = inference.download_profile()
        save()
        event("Native vision result " + label + ": " + json.dumps(entry["reference_comparison"]))
    report["status"] = "native_fusion_comparison_pending_quality_review"
    save()
