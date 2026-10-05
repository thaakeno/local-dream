#include "breeze/codec.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace breeze {
namespace codec_detail {

static ggml_tensor * debug_probe_transformer(
    CodecDebugProbes * probes, const std::string & name, ggml_tensor * tensor
) {
    if (probes && tensor) {
        ggml_set_output(tensor);
        probes->push_back({ name, tensor });
    }
    return tensor;
}

ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int T) {
    const CodecConfig & c = m.cfg.codec;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const float eps = c.layer_norm_eps;

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v = build_causal_mask(T, T, 0, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, T, T);

    ggml_tensor * h = x;
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = prefix + ".blk." + std::to_string(il);
        ggml_tensor * res = h;
        ggml_tensor * cur = layer_norm(ctx, h, m.w(p + ".attn_norm.weight"), m.w(p + ".attn_norm.bias"), eps);
        ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), cur), c.head_dim, c.n_head, T);
        q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(ctx, q, k, v, mask, scale, c.n_head, c.n_head);
        a = linear(ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(ctx, a, m.w(p + ".attn_scale"));
        h = ggml_add(ctx, res, a);

        res = h;
        cur = layer_norm(ctx, h, m.w(p + ".ffn_norm.weight"), m.w(p + ".ffn_norm.bias"), eps);
        cur = ggml_gelu_erf(ctx, linear(ctx, m.w(p + ".ffn_up.weight"), cur));
        cur = linear(ctx, m.w(p + ".ffn_down.weight"), cur);
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        h = ggml_add(ctx, res, cur);
    }
    return h;
}

ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x, int T) {
    const VocoderConfig & c = m.cfg.voc;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v = build_causal_mask(T, T, 0, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, T, T);

    ggml_tensor * h = ggml_add(ctx, linear(ctx, m.w("codec.dtf.in_proj.weight"), x),
                               m.w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = h;
        ggml_tensor * cur = rms_norm(ctx, h, m.w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), cur), c.head_dim, c.n_kv_head, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), cur), c.head_dim, c.n_kv_head, T);
        q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(ctx, q, k, v, mask, scale, c.n_head, c.n_kv_head);
        a = linear(ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(ctx, a, m.w(p + ".attn_scale"));
        h = ggml_add(ctx, res, a);

        res = h;
        cur = rms_norm(ctx, h, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
                         m.w(p + ".ffn_down.weight"));
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        h = ggml_add(ctx, res, cur);
    }
    h = rms_norm(ctx, h, m.w("codec.dtf.norm.weight"), c.rms_eps);
    return ggml_add(ctx, linear(ctx, m.w("codec.dtf.out_proj.weight"), h), m.w("codec.dtf.out_proj.bias"));
}

static std::vector<float> build_stream_window_mask(
    int n_q, int kv_start, int n_kv, int q_start, int sliding_window
) {
    std::vector<float> mask((size_t) n_q * (size_t) n_kv, 0.0f);
    for (int q = 0; q < n_q; ++q) {
        const int qpos = q_start + q;
        for (int kk = 0; kk < n_kv; ++kk) {
            const int kpos = kv_start + kk;
            bool ok = kpos <= qpos;
            if (ok && sliding_window > 0 && qpos - kpos >= sliding_window) {
                ok = false;
            }
            mask[(size_t) q * (size_t) n_kv + (size_t) kk] =
                ok ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

static ggml_tensor * kv_history_view(
    ggml_context * ctx, ggml_tensor * cache, int start, int count
) {
    return ggml_view_3d(
        ctx,
        cache,
        cache->ne[0],
        cache->ne[1],
        count,
        cache->nb[1],
        cache->nb[2],
        (size_t) start * cache->nb[2]
    );
}

static void kv_store_future(
    ggml_context * ctx, Graph & g, ggml_tensor * cache,
    ggml_tensor * cur, int pos
) {
    ggml_tensor * dst = ggml_view_3d(
        ctx,
        cache,
        cache->ne[0],
        cache->ne[1],
        cur->ne[2],
        cache->nb[1],
        cache->nb[2],
        (size_t) pos * cache->nb[2]
    );
    g.write(ggml_cpy(ctx, cur, dst));
}

ggml_tensor * vocoder_transformer_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    ggml_tensor * x, int T, CodecDebugProbes * probes
) {
    const VocoderConfig & c = m.cfg.voc;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const int pos0 = state.position;
    if (pos0 + T > state.kv.max_seq) {
        throw std::runtime_error("Breeze vocoder KV cache capacity exceeded");
    }

    const int kv_start = c.sliding_window > 0
        ? std::max(0, pos0 - c.sliding_window + 1)
        : 0;
    const int past_len = pos0 - kv_start;
    const int kv_len = past_len + T;

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = pos0 + i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v =
        build_stream_window_mask(T, kv_start, kv_len, pos0, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, kv_len, T);

    ggml_tensor * h = ggml_add(
        ctx, linear(ctx, m.w("codec.dtf.in_proj.weight"), x),
        m.w("codec.dtf.in_proj.bias")
    );
    debug_probe_transformer(probes, "dtf.in_proj", h);

    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        const std::string d = "dtf.blk." + std::to_string(il);

        ggml_tensor * res = h;
        ggml_tensor * cur = rms_norm(ctx, h, m.w(p + ".attn_norm.weight"), c.rms_eps);
        debug_probe_transformer(probes, d + ".attn_norm", cur);

        ggml_tensor * q0 = linear(ctx, m.w(p + ".attn_q.weight"), cur);
        ggml_tensor * k0 = linear(ctx, m.w(p + ".attn_k.weight"), cur);
        ggml_tensor * v0 = linear(ctx, m.w(p + ".attn_v.weight"), cur);
        debug_probe_transformer(probes, d + ".q_linear", q0);
        debug_probe_transformer(probes, d + ".k_linear", k0);
        debug_probe_transformer(probes, d + ".v_linear", v0);

        ggml_tensor * q = ggml_reshape_3d(ctx, q0, c.head_dim, c.n_head, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, k0, c.head_dim, c.n_kv_head, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, v0, c.head_dim, c.n_kv_head, T);
        q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        debug_probe_transformer(probes, d + ".q_rope", q);
        debug_probe_transformer(probes, d + ".k_rope", k);

        ggml_tensor * kfull = k;
        ggml_tensor * vfull = v;
        if (past_len > 0) {
            ggml_tensor * kp = kv_history_view(ctx, state.kv.k[il], kv_start, past_len);
            ggml_tensor * vp = kv_history_view(ctx, state.kv.v[il], kv_start, past_len);
            kfull = ggml_concat(ctx, kp, k, 2);
            vfull = ggml_concat(ctx, vp, v, 2);
        }
        debug_probe_transformer(probes, d + ".kfull", kfull);
        debug_probe_transformer(probes, d + ".vfull", vfull);

        kv_store_future(ctx, g, state.kv.k[il], k, pos0);
        kv_store_future(ctx, g, state.kv.v[il], v, pos0);

        ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
        debug_probe_transformer(probes, d + ".attention", a);
        a = linear(ctx, m.w(p + ".attn_output.weight"), a);
        debug_probe_transformer(probes, d + ".attn_output", a);
        a = ggml_mul(ctx, a, m.w(p + ".attn_scale"));
        debug_probe_transformer(probes, d + ".attn_scaled", a);
        h = ggml_add(ctx, res, a);
        debug_probe_transformer(probes, d + ".attn_residual", h);

        res = h;
        cur = rms_norm(ctx, h, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        debug_probe_transformer(probes, d + ".ffn_norm", cur);
        cur = swiglu_ffn(
            ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
            m.w(p + ".ffn_down.weight")
        );
        debug_probe_transformer(probes, d + ".ffn", cur);
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        debug_probe_transformer(probes, d + ".ffn_scaled", cur);
        h = ggml_add(ctx, res, cur);
        debug_probe_transformer(probes, d + ".ffn_residual", h);
    }

    h = rms_norm(ctx, h, m.w("codec.dtf.norm.weight"), c.rms_eps);
    debug_probe_transformer(probes, "dtf.final_norm", h);
    h = ggml_add(
        ctx, linear(ctx, m.w("codec.dtf.out_proj.weight"), h),
        m.w("codec.dtf.out_proj.bias")
    );
    return debug_probe_transformer(probes, "dtf.out_proj", h);
}

}
}
