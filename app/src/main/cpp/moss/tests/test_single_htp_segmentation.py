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
assert 'NM_MOSS_MAX_VRAM' in bridge
assert 'HTP0:0=2.84' in bridge and 'HTP0:0=2.88' in bridge
assert 'if (!ctx->model && budget != kMossSingleHtpBudget)' in bridge
assert 'std::strcmp(req->params_backend, "all=disk")' in bridge
assert '"max_vram=$mossBudget params_backend=all=disk "' in svc
assert 'moss_fast_residency' in svc
assert 'NM_MOSS_MAX_VRAM' in svc
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
assert '035-nightmare-moss-fused-rope.sdcpp.patch' in build_script
assert '[MOSS_ROPE_DISPATCH] candidate=fused' in (stage / "src/model/diffusion/moss_image.hpp").read_text()
assert '[MOSS_ROPE_DISPATCH] fallback=' in (stage / "src/model/diffusion/moss_image.hpp").read_text()
assert 'moss_htp_profile.flag' in moss_service
assert 'env["GGML_HEXAGON_PROFILE"] = "1"' in moss_service
assert 'env.remove("GGML_HEXAGON_PROFILE")' in moss_service
assert '036-moss-fast-tensor-loader-poll.patch' in build_script
staged_loader = (stage / "src/model_loader.cpp").read_text()
assert 'std::this_thread::sleep_for(std::chrono::milliseconds(5));' in staged_loader
assert 'milliseconds(total_tensors_to_process <= 4 ? 10 : 200)' not in staged_loader
assert '028-nightmare-moss-image.sdcpp.patch' in build_script
moss_model = (stage / "src/model/diffusion/moss_image.hpp").read_text()
assert 'class MOSSQKNorm' in moss_model
assert 'ggml_qknorm_rope' in moss_model
assert 'NM_MOSS_COMPOSED_ROPE' in moss_model
assert 'if (!ggml_backend_supports_op(ctx->backend, out))' in moss_model
assert 'GGML_OP_QKNORM_ROPE' in (stage / "ggml/include/ggml.h").read_text()
assert 'GGML_OP_QKNORM_ROPE' in hexagon
assert 'std::getenv("NM_MOSS_COMPOSED_ROPE")' in bridge
assert 'params.sample_params.scheduler = SIMPLE_SCHEDULER' in bridge
assert 'params.sample_params.scheduler = SGM_UNIFORM_SCHEDULER' in bridge
assert 'params.sample_params.scheduler = DISCRETE_SCHEDULER' in bridge
assert 'params.sample_params.scheduler = KARRAS_SCHEDULER' in bridge
dit_bridge = (moss.parent / "src/PipelineDit.hpp").read_text()
assert 'const std::string moss_sampling = kind_ == DIT_MODEL_MOSS_IMAGE' in dit_bridge
assert 'params.sample_method = kind_ == DIT_MODEL_MOSS_IMAGE' in dit_bridge
model_source = (moss.parent.parent / "java/io/github/xororz/localdream/data/Model.kt").read_text()
assert 'euler|simple' in model_source and 'euler|sgm_uniform' in model_source
ui_settings = (moss.parent.parent / "java/io/github/xororz/localdream/ui/screens/AdvancedSettingsDialog.kt").read_text()
assert 'Noise schedule' in ui_settings and 'Sampler' in ui_settings
assert 'onSchedulerChange("euler|$id")' in ui_settings
assert 'Native FlowMatch' in ui_settings
assert 'Recommended: Simple · 8 steps' in ui_settings
assert 'AppHaptics.Kind.Interaction' in ui_settings
assert 'AppHaptics.Kind.Progress' in ui_settings
assert 'FilterChip' not in ui_settings
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

# The user reported 8/8 appearing before a single MOSS graph pass finished.
# For this engine only, progress must come from real Qwen and DiT segments.
moss_src = (moss / "MossEngine.cpp").read_text()
core_src = (moss.parent / "src/PipelineDit.hpp").read_text()
ui_src = (moss.parent.parent / "java/io/github/xororz/localdream/ui/screens/GenerationProgressCard.kt").read_text()
assert 'void on_progress(int, int, float, void*)' in moss_src
assert 'on_log(sd_log_level_t level, const char* message, void*)' in moss_src
assert 'std::strstr(message, " executing segment ")' in moss_src
assert '(active.moss_pass - 1) * total + n - 1' in moss_src
assert 'sampling completed' in moss_src and 'DIT_PHASE_DECODING' in moss_src
assert 'callbacks.moss = kind_ == DIT_MODEL_MOSS_IMAGE' in core_src
assert 'moss_sampling_layers:' in core_src
assert 'moss_sampling_layers:' in ui_src
assert 'Step $shownStep/$steps' in ui_src
print("MOSS real graph progress, phased decoding, and UI regression: PASS")

# Haptic updates MUST never fire per MOSS segment/encoder/decoder callback.
screen = (moss.parent.parent / "java/io/github/xororz/localdream/ui/screens/ModelRunScreen.kt").read_text()
settings = (moss.parent.parent / "java/io/github/xororz/localdream/ui/screens/ModelListScreen.kt").read_text()
assert 'lastMossCompletedHapticStep' in screen
assert 'val completed = (state.step.coerceAtLeast(0) / layersPerStep)' in screen
assert 'completed > lastMossCompletedHapticStep' in screen
assert 'MossPerformanceDialog' in settings
assert 'Progress pulses' in settings
print("MOSS once-per-sampler-step haptics and opt-in compatibility switch: PASS")

# MOSS-only poll no longer imposes 200ms of idle time per tensor group.
# Vibration is based on completed Euler passes, including final decode.
assert 'priorPhase.startsWith("moss_sampling_layers:")' in screen
assert 'lastMossCompletedHapticStep = finalStep' in screen
assert 'TextOverflow.Ellipsis' in screen
print("MOSS low-latency tensor streaming and final-step haptic: PASS")

# All new precision paths are opt-in and use existing model directory contracts.
moss_package_source = (moss.parent.parent / "java/io/github/xororz/localdream/data/MossImagePackages.kt").read_text()
assert 'TURBO_NPU_Q8_ID' in moss_package_source
assert 'moss_image_sft-Q4_0.gguf|dit.gguf' in moss_package_source
assert 'Qwen3.5-9B-Q8_0.gguf|llm.gguf' in moss_package_source
assert 'lora.safetensors|moss_turbo_lora.safetensors' in moss_package_source
assert 'cpu_graph_islands=possible' in moss_service
