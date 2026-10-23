// MOSS-Image runtime: independent official upstream; no patch to the Qwen engine.
#include "DitEngine.h"
#include "stable-diffusion.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>

struct dit_ctx {
    sd_ctx_t *model = nullptr;
    std::string error;
    int preview_interval = 0;
    std::string turbo_lora_path;
};

namespace {
std::mutex generation_mutex;
std::string creation_error;
dit_log_cb logger = nullptr;
void* logger_user = nullptr;
struct ActiveSession {
    dit_ctx *context = nullptr;
    dit_progress_cb progress = nullptr;
    dit_phase_cb phase = nullptr;
    dit_preview_cb preview = nullptr;
    void *user = nullptr;
    bool cancelled = false;
    int expected_sampling_steps = 0;
    int text_segments = 0;
    int last_text_reported = -1;
    int moss_segments = 0;
    int last_moss_segment = 0;
    int moss_pass = 1;
    int last_moss_reported = -1;
    bool encoding_prompt = false;
    bool diffusion_started = false;
    bool decoding_started = false;
} active;

// ggml's progress callback is also invoked for auxiliary operations. The
// old bridge confused those values with real sampler steps (8/8 before MOSS
// even finished pass 1). Graph-cut runner logs are the authoritative source:
// entering segment N confirms N-1 segments completed. Report a monotonic,
// conservative stage-local fraction, never speculative image completion.
void emit_graph_progress(int completed, int total) {
    if (!active.progress || active.cancelled || total <= 0) return;
    if (!active.progress(DIT_PROGRESS_MODEL_LOADING,
                         std::clamp(completed, 0, total), total, 0.f,
                         active.user)) {
        active.cancelled = true;
        if (active.context && active.context->model)
            sd_cancel_generation(active.context->model, SD_CANCEL_ALL);
    }
}
void on_log(sd_log_level_t level, const char* message, void*) {
    if (logger) logger(static_cast<int>(level), message ? message : "", logger_user);
    if (!message || !active.context || active.cancelled) return;
    int n = 0;
    int total = 0;
    // The pinned ggml engine logs the real NPU graph segment execution.
    // Qwen variants name their runner qwen3/qwen3.5; do not depend on
    // fabricated progress estimates or tensor upload denominators.
    const char *segment = std::strstr(message, " executing segment ");
    if (segment && std::sscanf(segment, " executing segment %d/%d", &n, &total) == 2 &&
        n >= 1 && total > 0 && n <= total && total <= 256) {
        if (std::strstr(message, "MOSS-Image")) {
            if (!active.diffusion_started) {
                active.diffusion_started = true;
                active.encoding_prompt = false;
                if (active.phase) active.phase(DIT_PHASE_DENOISING, active.user);
            }
            active.moss_segments = total;
            if (n == 1 && active.last_moss_segment > 1)
                ++active.moss_pass;
            active.last_moss_segment = n;
            const int all = total * active.expected_sampling_steps;
            const int finished = std::min(all,
                (active.moss_pass - 1) * total + n - 1);
            if (finished > active.last_moss_reported) {
                active.last_moss_reported = finished;
                emit_graph_progress(finished, all);
            }
        } else if (!active.diffusion_started &&
                   (std::strstr(message, "qwen3") ||
                    std::strstr(message, "Qwen3"))) {
            if (!active.encoding_prompt) {
                active.encoding_prompt = true;
                if (active.phase) active.phase(DIT_PHASE_ENCODING_PROMPT, active.user);
            }
            active.text_segments = total;
            if (n - 1 > active.last_text_reported) {
                active.last_text_reported = n - 1;
                emit_graph_progress(n - 1, total);
            }
        }
        return;
    }
    if (!active.diffusion_started &&
        std::strstr(message, "get_learned_condition completed") &&
        active.encoding_prompt) {
        emit_graph_progress(active.text_segments, active.text_segments);
        active.encoding_prompt = false;
    } else if (active.diffusion_started &&
               std::strstr(message, "sampling completed")) {
        const int all = active.moss_segments * active.expected_sampling_steps;
        emit_graph_progress(all, all);
        active.decoding_started = true;
        if (active.phase) active.phase(DIT_PHASE_DECODING, active.user);
    } else if (active.diffusion_started && !active.decoding_started &&
               std::strstr(message, "decoding ") &&
               std::strstr(message, " latents")) {
        active.decoding_started = true;
        if (active.phase) active.phase(DIT_PHASE_DECODING, active.user);
    }
}
void on_progress(int, int, float, void*) {
    // Never use upstream's untyped callback to advance MOSS sampling. In the
    // user's alpha.79 log, it reported 8/8 while graph execution continued.
    // The on_log graph-segment observer provides real progress instead.
}
void on_preview(int step, int count, sd_image_t* images, bool, void*) {
    if (!active.preview || count < 1 || !images || !images[0].data) return;
    auto &im = images[0];
    if (im.channel != 3 || im.width > INT32_MAX || im.height > INT32_MAX) return;
    active.preview(step, im.data, static_cast<int>(im.width), static_cast<int>(im.height), active.user);
}
dit_ctx *create(const dit_ctx_params* req) {
    creation_error.clear();
    if (!req || req->kind != DIT_MODEL_MOSS_IMAGE || !req->diffusion_model_path ||
        !req->llm_path || !req->vae_path) {
        creation_error = "invalid MOSS model package or engine kind";
        return nullptr;
    }
    std::filesystem::path folder(req->diffusion_model_path);
    const auto tokenizer = folder.parent_path() / "tokenizer.json";
    if (!std::filesystem::is_regular_file(tokenizer)) {
        creation_error = "MOSS requires its separate Qwen3.5 tokenizer.json";
        return nullptr;
    }
    auto *ctx = new dit_ctx();
    const auto adapter = folder.parent_path() / "moss_turbo_lora.safetensors";
    // The new GGUF files already contain DMD weights. Never double-apply
    // a stale LoRA copied into their model folder by a user or migration.
    const auto model_id = folder.parent_path().filename().string();
    const bool merged_turbo = model_id == "moss_image_sft_turbo_merged_q4_0" ||
                              model_id == "moss_image_sft_turbo_merged_q4_k_m";
    const bool expects_unmerged_adapter =
        model_id == "moss_image_sft_turbo_q4" ||
        model_id == "moss_image_base_turbo_q4" ||
        model_id == "moss_image_sft_turbo_npu_q8";
    if (merged_turbo) {
        std::fprintf(stderr, "[MOSS_TURBO] merged DMD checkpoint; no runtime LoRA\n");
    } else if (std::filesystem::is_regular_file(adapter)) {
        ctx->turbo_lora_path = adapter.string();
        std::fprintf(stderr, "[MOSS_TURBO] separate DMD adapter active (not merged)\n");
    } else if (expects_unmerged_adapter) {
        creation_error = "MOSS Turbo requires separate FP32 DMD LoRA; adapter file missing";
        delete ctx;
        return nullptr;
    }
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    const std::string token_path = tokenizer.string();
    p.diffusion_model_path = req->diffusion_model_path;
    p.llm_path = req->llm_path;
    p.vae_path = req->vae_path;
    // Nightmare's upstream MOSS patch (dit/028) resolves Qwen3.5 tokenizer
    // via SD_QWEN35_TOKENIZER, not via the newer leejet-only p.tokenizer API.
    // Local Dream starts MOSS in its own isolated backend process.
    if (setenv("SD_QWEN35_TOKENIZER", token_path.c_str(), 1) != 0) {
        creation_error = "unable to configure Qwen3.5 tokenizer path";
        delete ctx;
        return nullptr;
    }
    p.n_threads = req->n_threads > 0 ? req->n_threads : 4;
    p.flash_attn = req->flash_attn;
    p.diffusion_flash_attn = req->flash_attn;
    p.vae_conv_direct = req->vae_conv_direct;
    p.enable_mmap = true;
    // Never silently move unsupported/oversized MOSS inference to CPU.
    // A genuine NPU mapping failure must surface rather than downgrade.
    p.auto_fit = false;
    if (req->backend && req->backend[0]) p.backend = req->backend;
    if (req->params_backend && req->params_backend[0]) p.params_backend = req->params_backend;
    // Budgeted single-backend segments stream both Qwen and DiT weights
    // from disk; the two-session layer split previously disabled segmentation.
    // The managed budget includes compute buffers and retained layer weights;
    // it is NOT physical RAM or a promise about NPU map limits.
    // Only accept application-owned, prevalidated profiles. Never accept an
    // arbitrary budget from an untrusted environment or another model engine.
    constexpr const char *kMossSingleHtpBudget = "HTP0:0=2.60";
    const char* budget = kMossSingleHtpBudget;
    const char* requested_budget = std::getenv("NM_MOSS_MAX_VRAM");
    if (requested_budget &&
        (std::strcmp(requested_budget, "HTP0:0=2.84") == 0 ||
         std::strcmp(requested_budget, "HTP0:0=2.88") == 0)) {
        budget = requested_budget;
    }
    if (!req->backend || std::strstr(req->backend, "HTP0:0&") ||
        !std::strstr(req->backend, "diffusion=HTP0:0,te=HTP0:0,vae=HTP0:0") ||
        !req->params_backend || std::strcmp(req->params_backend, "all=disk") != 0) {
        creation_error = "MOSS requires single HTP0:0 compute and all=disk segmentation";
        delete ctx;
        return nullptr;
    }
    p.max_vram = budget;
    std::fprintf(stderr, "[MOSS_SEGMENTED] max_vram=%s runtime=%s params=%s\n",
                 budget, req->backend, req->params_backend);
    ctx->model = new_sd_ctx(&p);
    if (!ctx->model && budget != kMossSingleHtpBudget) {
        std::fprintf(stderr, "[MOSS_MEMORY] fast residency init failed; retrying 2.60GiB compatibility\n");
        p.max_vram = kMossSingleHtpBudget;
        ctx->model = new_sd_ctx(&p);
    }
    if (!ctx->model) {
        creation_error = "MOSS upstream new_sd_ctx failed; check tokenizer, weights and backend";
        delete ctx;
        return nullptr;
    }
    return ctx;
}
void destroy(dit_ctx *ctx) {
    if (!ctx) return;
    if (ctx->model) free_sd_ctx(ctx->model);
    delete ctx;
}
bool generate(dit_ctx* ctx, const dit_gen_params* req,
              dit_progress_cb progress, dit_phase_cb phase,
              dit_preview_cb preview, void* user,
              uint8_t** pixels, int* width, int* height, int* channels) {
    if (!ctx || !ctx->model || !req || !pixels || !width || !height || !channels) return false;
    ctx->error.clear();
    if ((req->width & 15) || (req->height & 15) || req->width <= 0 || req->height <= 0) {
        ctx->error = "MOSS width and height must be positive multiples of 16";
        return false;
    }
    if (req->reference_image_count || req->init_image_rgb || req->mask_image) {
        ctx->error = "MOSS currently supports text-to-image only";
        return false;
    }
    std::lock_guard<std::mutex> hold(generation_mutex);
    active = {};
    active.context = ctx;
    active.progress = progress;
    active.phase = phase;
    active.preview = preview;
    active.user = user;
    active.expected_sampling_steps = std::max(1, req->steps);
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = req->prompt ? req->prompt : "";
    // Stock Euler sampler + LoRA is the published 8-step recommendation.
    // No hidden 4-step data-space algorithm, and never modify Base/SFT.
    sd_lora_t turbo_lora{};
    if (!ctx->turbo_lora_path.empty()) {
        turbo_lora.path = ctx->turbo_lora_path.c_str();
        turbo_lora.multiplier = 1.0f;
        params.loras = &turbo_lora;
        params.lora_count = 1;
    }
    params.negative_prompt = req->negative_prompt ? req->negative_prompt : "";
    params.width = req->width;
    params.height = req->height;
    params.seed = req->seed;
    params.batch_count = 1;
    params.sample_params.sample_steps = std::max(1, req->steps);
    params.sample_params.guidance.txt_cfg = std::isfinite(req->cfg_scale) && req->cfg_scale > 0 ? req->cfg_scale : 3.f;
    // "euler|simple" encodes sampler + noise schedule in the existing ABI
    // field. No ABI change; all other DiT engines retain their own behavior.
    const std::string sampling_spec = req->sample_method && req->sample_method[0]
        ? req->sample_method : "euler|simple";
    const auto separator = sampling_spec.find('|');
    const std::string sampler_name = sampling_spec.substr(0, separator);
    const std::string schedule_name = separator == std::string::npos
        ? "simple" : sampling_spec.substr(separator + 1);
    if (sampler_name != "euler") {
        ctx->error = "Unsupported MOSS sampler: " + sampler_name;
        return false;
    }
    if (schedule_name == "simple") {
        params.sample_params.scheduler = SIMPLE_SCHEDULER;
    } else if (schedule_name == "sgm_uniform") {
        params.sample_params.scheduler = SGM_UNIFORM_SCHEDULER;
    } else if (schedule_name == "discrete") {
        params.sample_params.scheduler = DISCRETE_SCHEDULER;
    } else if (schedule_name == "karras") {
        params.sample_params.scheduler = KARRAS_SCHEDULER;
    } else {
        ctx->error = "Unsupported MOSS noise schedule: " + schedule_name;
        return false;
    }
    params.sample_params.flow_shift = 5.f;
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    std::fprintf(stderr, "[MOSS_SAMPLING] sampler=euler schedule=%s shift=5 steps=%d\n",
                 schedule_name.c_str(), params.sample_params.sample_steps);
    if (req->vae_tile_size > 0) {
        params.vae_tiling_params.enabled = true;
        params.vae_tiling_params.tile_size_x = req->vae_tile_size;
        params.vae_tiling_params.tile_size_y = req->vae_tile_size;
        params.vae_tiling_params.target_overlap = req->vae_tile_overlap;
    }
    if (phase) phase(DIT_PHASE_PREPARING, user);
    sd_set_progress_callback(progress || phase ? on_progress : nullptr, nullptr);
    sd_set_preview_callback(preview && ctx->preview_interval > 0 ? on_preview : nullptr,
                            PREVIEW_PROJ, ctx->preview_interval, true, false, nullptr);
    sd_image_t *out = nullptr;
    int count = 0;
    const auto run_start = std::chrono::steady_clock::now();
    bool ok = generate_image(ctx->model, &params, &out, &count);
    const auto run_end = std::chrono::steady_clock::now();
    const bool composed_rope = std::getenv("NM_MOSS_COMPOSED_ROPE") != nullptr;
    const double duration_s =
        std::chrono::duration<double>(run_end - run_start).count();
    // Comparable metrics for internal A/B runs: keep prompt private and
    // reuse the SAME seed, dimensions, steps and model for both paths.
    std::fprintf(stderr,
                 "[MOSS_BENCH] rope=%s status=%s resolution=%dx%d steps=%d seed=%lld total_s=%.3f\n",
                 composed_rope ? "composed" : "fused_or_backend_fallback",
                 ok ? "ok" : "failed", req->width, req->height, req->steps,
                 static_cast<long long>(req->seed), duration_s);
    sd_set_progress_callback(nullptr, nullptr);
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 0, false, false, nullptr);
    const bool cancelled = active.cancelled;
    active = {};
    if (cancelled) sd_cancel_generation(ctx->model, SD_CANCEL_RESET);
    if (!ok || count < 1 || !out || !out[0].data) {
        ctx->error = cancelled ? "generation cancelled" : "MOSS generation failed";
        if (out) free_sd_images(out, std::max(0, count));
        return false;
    }
    const auto &img = out[0];
    if (!img.width || !img.height || (img.channel != 3 && img.channel != 4) ||
        img.width > INT32_MAX || img.height > INT32_MAX ||
        static_cast<uint64_t>(img.width) * img.height * img.channel > SIZE_MAX) {
        ctx->error = "MOSS returned invalid output dimensions";
        free_sd_images(out, count);
        return false;
    }
    const size_t size = static_cast<size_t>(img.width) * img.height * img.channel;
    uint8_t *copy = static_cast<uint8_t*>(std::malloc(size));
    if (!copy) {
        ctx->error = "unable to allocate output image";
        free_sd_images(out, count);
        return false;
    }
    std::memcpy(copy, img.data, size);
    *pixels = copy;
    *width = static_cast<int>(img.width);
    *height = static_cast<int>(img.height);
    *channels = static_cast<int>(img.channel);
    free_sd_images(out, count);
    if (phase) phase(DIT_PHASE_FINALIZING, user);
    return true;
}
void free_image(uint8_t *p) { std::free(p); }
bool tokenize(dit_ctx *, const char *, dit_tokenize_result *) { return false; }
const char *last_error(const dit_ctx *ctx) { return ctx ? ctx->error.c_str() : creation_error.c_str(); }
void set_logger(dit_log_cb fn, void *user) {
    logger = fn;
    logger_user = user;
    sd_set_log_callback(fn ? on_log : nullptr, nullptr);
}
void set_preview_interval(dit_ctx *ctx, int interval) {
    if (ctx) ctx->preview_interval = std::max(0, interval);
}
const dit_engine_api api = {DIT_ENGINE_ABI_VERSION, create, destroy, generate,
                            free_image, tokenize, last_error, set_logger, set_preview_interval};
} // namespace
extern "C" __attribute__((visibility("default"))) const dit_engine_api *dit_engine_get_api(int abi) {
    return abi == DIT_ENGINE_ABI_VERSION ? &api : nullptr;
}
