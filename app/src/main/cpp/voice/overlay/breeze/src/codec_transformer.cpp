#include "breeze/codec.h"

#include <cmath>
#include <string>

namespace breeze {
namespace codec_detail {

static ggml_tensor * vocoder_diag_probe(
    ggml_context * ctx, Graph & g, std::vector<VocoderDiagProbe> * probes,
    const std::string & name, ggml_tensor * x
) {
    if (!probes) return x;
    ggml_tensor * scalar = ggml_sum(ctx, x);
    ggml_set_output(scalar);
    g.write(scalar);
    probes->push_back({ name, scalar });
    return x;
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
        h = vocoder_diag_probe(
            ctx, g, probes, "dtf.blk." + std::to_string(il) + ".attn_residual", h
        );

        res = h;
        cur = layer_norm(ctx, h, m.w(p + ".ffn_norm.weight"), m.w(p + ".ffn_norm.bias"), eps);
        cur = ggml_gelu_erf(ctx, linear(ctx, m.w(p + ".ffn_up.weight"), cur));
        cur = linear(ctx, m.w(p + ".ffn_down.weight"), cur);
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        h = ggml_add(ctx, res, cur);
    }
    return h;
}

ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                                  int T, std::vector<VocoderDiagProbe> * probes) {
    const VocoderConfig & c = m.cfg.voc;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v = build_causal_mask(T, T, 0, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, T, T);

    ggml_tensor * h = ggml_add(ctx, linear(ctx, m.w("codec.dtf.in_proj.weight"), x),
                               m.w("codec.dtf.in_proj.bias"));
    h = vocoder_diag_probe(ctx, g, probes, "dtf.in_proj", h);
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
        h = vocoder_diag_probe(
            ctx, g, probes, "dtf.blk." + std::to_string(il) + ".ffn_residual", h
        );
    }
    h = rms_norm(ctx, h, m.w("codec.dtf.norm.weight"), c.rms_eps);
    h = vocoder_diag_probe(ctx, g, probes, "dtf.norm", h);
    h = ggml_add(ctx, linear(ctx, m.w("codec.dtf.out_proj.weight"), h), m.w("codec.dtf.out_proj.bias"));
    return vocoder_diag_probe(ctx, g, probes, "dtf.out_proj", h);
}

ggml_tensor * vocoder_transformer_stream(ggml_context * ctx, BreezeModel & m, Graph & g,
                                        VocoderStreamState & state, ggml_tensor * x, int T) {
    const VocoderConfig & c = m.cfg.voc;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const int pos0 = state.position;
    const int kv_len = pos0 + T;

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = pos0 + i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v = build_causal_mask(T, kv_len, pos0, c.sliding_window);
    // GGML softmax expects mask ne[0] to match the attention KV axis.
    // build_causal_mask stores [query][kv], so expose kv_len as ne0.
    ggml_tensor * mask = g.input_f32(mask_v, kv_len, T);

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

        ggml_tensor * kfull = cache_append(ctx, g, state.kv.k[il], k, pos0);
        ggml_tensor * vfull = cache_append(ctx, g, state.kv.v[il], v, pos0);
        ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
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

}
}
