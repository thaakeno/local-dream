// MOSS-Image runtime: independent official upstream; no patch to the Qwen engine.
#include "DitEngine.h"
#include "stable-diffusion.h"
#include <algorithm>
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
    bool started = false;
} active;

void on_log(sd_log_level_t level, const char* message, void*) {
    if (logger) logger(static_cast<int>(level), message ? message : "", logger_user);
}
void on_progress(int step, int total, float elapsed, void*) {
    if (active.phase && !active.started) {
        active.started = true;
        active.phase(DIT_PHASE_DENOISING, active.user);
    }
    if (active.progress && !active.progress(DIT_PROGRESS_SAMPLING, step, total, elapsed, active.user)) {
        active.cancelled = true;
        if (active.context && active.context->model)
            sd_cancel_generation(active.context->model, SD_CANCEL_ALL);
    }
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
    if (std::filesystem::is_regular_file(adapter))
        ctx->turbo_lora_path = adapter.string();
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    const std::string token_path = tokenizer.string();
    p.diffusion_model_path = req->diffusion_model_path;
    p.llm_path = req->llm_path;
    p.vae_path = req->vae_path;
    p.tokenizer = token_path.c_str();
    p.n_threads = req->n_threads > 0 ? req->n_threads : 4;
    p.flash_attn = req->flash_attn;
    p.diffusion_flash_attn = req->flash_attn;
    p.vae_conv_direct = req->vae_conv_direct;
    p.enable_mmap = true;
    p.auto_fit = true;
    if (req->backend && req->backend[0]) p.backend = req->backend;
    if (req->params_backend && req->params_backend[0]) p.params_backend = req->params_backend;
    ctx->model = new_sd_ctx(&p);
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
    active = {ctx, progress, phase, preview, user, false, false};
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
    params.sample_params.scheduler = DISCRETE_SCHEDULER;
    params.sample_params.flow_shift = 5.f;
    params.sample_params.sample_method = req->sample_method && req->sample_method[0]
        ? str_to_sample_method(req->sample_method)
        : EULER_SAMPLE_METHOD;
    if (req->vae_tile_size > 0) {
        params.vae_tiling_params.enabled = true;
        params.vae_tiling_params.tile_size_w = req->vae_tile_size;
        params.vae_tiling_params.tile_size_h = req->vae_tile_size;
        params.vae_tiling_params.target_overlap = req->vae_tile_overlap;
    }
    if (phase) phase(DIT_PHASE_PREPARING, user);
    sd_set_progress_callback(progress || phase ? on_progress : nullptr, nullptr);
    sd_set_preview_callback(preview && ctx->preview_interval > 0 ? on_preview : nullptr,
                            PREVIEW_PROJ, ctx->preview_interval, true, false, nullptr);
    sd_image_t *out = nullptr;
    int count = 0;
    bool ok = generate_image(ctx->model, &params, &out, &count);
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
