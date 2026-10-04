#pragma once

// Functional AR LoRA support for the Serveurperso YuE2 runtime.
//
// The adapter stays separate from the quantized base weights and evaluates
// W*x + scale*B*(A*x), matching the native yuey.cpp implementation.  This is
// important for Q8/Q5/Q6 bases: no destructive weight merge and no CPU path.

#include "gguf-weights.h"

#include <string>

struct Yue2LoraPair {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    float scale = 0.0f;
};

static inline bool yue2_lora_pair_ready(const Yue2LoraPair & p) {
    return p.a != nullptr && p.b != nullptr && p.scale != 0.0f;
}

static inline ggml_tensor * yue2_lora_apply(
    ggml_context * ctx,
    ggml_tensor * base,
    const Yue2LoraPair & pair,
    ggml_tensor * input,
    bool enabled) {
    if (!enabled || !yue2_lora_pair_ready(pair)) {
        return base;
    }
    ggml_tensor * ax = ggml_mul_mat(ctx, pair.a, input);
    ggml_tensor * bax = ggml_mul_mat(ctx, pair.b, ax);
    if (pair.scale != 1.0f) {
        bax = ggml_scale(ctx, bax, pair.scale);
    }
    return ggml_add(ctx, base, bax);
}

static inline bool yue2_lora_load_pair(
    WeightCtx * wctx,
    const GGUFModel & gf,
    const std::string & target,
    Yue2LoraPair * out,
    float scale) {
    const std::string an = target + ".lora_A";
    const std::string bn = target + ".lora_B";
    if (gguf_find_tensor(gf.gguf, an.c_str()) < 0 ||
        gguf_find_tensor(gf.gguf, bn.c_str()) < 0) {
        return false;
    }

    out->a = gf_load_tensor(wctx, gf, an);
    out->b = gf_load_tensor(wctx, gf, bn);
    out->scale = scale;

    if ((out->a->type != GGML_TYPE_F16 && out->a->type != GGML_TYPE_F32) ||
        (out->b->type != GGML_TYPE_F16 && out->b->type != GGML_TYPE_F32)) {
        fprintf(stderr, "[AR-LoRA] FATAL: unsupported factor type for %s\n", target.c_str());
        return false;
    }
    if (out->a->ne[1] != out->b->ne[0]) {
        fprintf(stderr, "[AR-LoRA] FATAL: rank mismatch for %s\n", target.c_str());
        return false;
    }
    return true;
}
