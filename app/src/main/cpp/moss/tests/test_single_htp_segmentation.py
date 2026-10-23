#!/usr/bin/env python3
"""Assert MOSS uses upstream, budgeted single-NPU segmented execution."""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: test_single_htp_segmentation.py STAGED_SOURCE")

stage = Path(sys.argv[1])
moss = Path(__file__).resolve().parents[1]
main = (moss.parent / "src/main.cpp").read_text()
bridge = (moss / "MossEngine.cpp").read_text()
svc = (moss.parent.parent / "java/io/github/xororz/localdream/service/BackendService.kt").read_text()
runner = (stage / "src/core/ggml_runner.cpp").read_text()
hexagon = (stage / "ggml/src/ggml-hexagon/ggml-hexagon.cpp").read_text()
partition = (stage / "src/core/layer_split_partition.cpp").read_text()
assert 'moss ? "diffusion=HTP0:0,te=HTP0:0,vae=HTP0:0"' in main
assert 'moss ? "all=disk"' in main
assert 'const std::string moss_htp = "HTP0:0&HTP0:1"' not in main
assert 'kMossSingleHtpBudget = "HTP0:0=2.60"' in bridge
assert 'std::strcmp(req->params_backend, "all=disk")' in bridge
assert 'max_vram=HTP0:0=2.60GiB params_backend=all=disk' in svc
# Scope accelerator checks to MOSS only. Other backends (notably YuE2)
# intentionally supply their own NHMX/fusion/batch/queue values, and must
# NEVER be mutated just to satisfy a MOSS regression.
moss_service = svc.split('if (backendType == MossImagePackages.DIT_KIND) {', 1)[1].split(
    'if (backendType == "sdxl")', 1
)[0]
assert 'env["GGML_HEXAGON_DEVICES"] = "HTP0:0,HTP0:1"' in moss_service
# The pinned MOSS ggml already enables HMX and fusion by default.
for setting in ("NHMX", "OPFUSION", "OPBATCH", "OPQUEUE"):
    assert f'env["GGML_HEXAGON_{setting}"] =' not in moss_service
# Guard against a false positive: these overrides DO exist, but only for
# the separate YuE2 engine in the rest of BackendService.
assert 'env["GGML_HEXAGON_NHMX"] = "1"' in svc
assert 'env["GGML_HEXAGON_OPFUSION"] = "1"' in svc
assert 'opt_nhmx    = 1' in hexagon
assert 'opt_opfusion = 1' in hexagon

assert '!is_multi_device() && !sd_backend_is_cpu(runtime_backend)' in runner
assert 'manager->segmented_compute_enabled()' in runner
assert 'ggml_graph_cut::build_segment_graph' in runner
assert '[MOSS_HTP_COPY]' not in hexagon
assert 'MOSS HTP split planner:' not in partition
# The build must use the already pinned Nightmare-compatible happyyzy/ggml,
# but never alter the shared DiT submodule in place.
build_script = (moss / "build.sh").read_text()
assert 'UPSTREAM="../3rdparty/stable-diffusion.cpp"' in build_script
assert 'EXPECTED=a63954dc794f6daade32c407da1a9c4bc2d73cf0' in build_script
assert 'BUILD_DIR=build/android-moss-fused-rope' in build_script
assert '035-nightmare-moss-fused-rope.sdcpp.patch' in build_script
assert '028-nightmare-moss-image.sdcpp.patch' in build_script
moss_model = (stage / "src/model/diffusion/moss_image.hpp").read_text()
assert 'class MOSSQKNorm' in moss_model
assert 'ggml_qknorm_rope' in moss_model
assert 'NM_MOSS_COMPOSED_ROPE' in moss_model
assert 'if (!ggml_backend_supports_op(ctx->backend, out))' in moss_model
assert 'GGML_OP_QKNORM_ROPE' in (stage / "ggml/include/ggml.h").read_text()
assert 'ggml_qknorm_rope' in hexagon
assert 'std::getenv("NM_MOSS_COMPOSED_ROPE")' in bridge
assert '[MOSS_BENCH]' in bridge
assert 'setenv("SD_QWEN35_TOKENIZER"' in bridge
assert 'p.tokenizer =' not in bridge
assert 'moss_composed_rope.flag' in svc
print("MOSS single HTP segmented graph cut and pristine ggml source: PASS")

# 8-step merged Turbo catalog and shared-asset safety regression.
pkg = (moss.parent.parent / "java/io/github/xororz/localdream/data/MossImagePackages.kt").read_text()
model = (moss.parent.parent / "java/io/github/xororz/localdream/data/Model.kt").read_text()
downloader = (moss.parent.parent / "java/io/github/xororz/localdream/service/ModelDownloadService.kt").read_text()
engine = (moss / "MossEngine.cpp").read_text()
assert 'molbal/MOSS-Image-GGUF/resolve/main/moss_image_sft_turbo-Q4_0.gguf|dit.gguf' in pkg
assert 'molbal/MOSS-Image-GGUF/resolve/main/moss_image_sft_turbo-Q4_K_M.gguf|dit.gguf' in pkg
assert 'add(createMossMergedTurboModel(fast = true))' in model
assert 'add(createMossMergedTurboModel(fast = false))' in model
assert 'steps = 8f' in model
assert 'cfg = 1f' in model
assert 'android.system.Os.link(source.absolutePath, dest.absolutePath)' in downloader
assert 'source.length() != expected' in downloader
assert 'File(donorDir, MossImagePackages.MODEL_MARKER).isFile' in downloader
assert '[MOSS_TURBO] merged DMD checkpoint; no runtime LoRA' in engine
print("MOSS pre-merged Turbo catalog and common-runtime file hard links: PASS")
