#include "breeze/codec.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {
namespace codec_detail {

static ggml_tensor * snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb) {
    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * inv_beta = ggml_reshape_2d(ctx, ggml_exp(ctx, ggml_neg(ctx, lb)), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), inv_beta));
}

static ggml_tensor * convnext(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = depthwise1d_causal(ctx, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    return ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
}

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, const std::string & p,
                                   ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

static ggml_tensor * quantizer_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                                      const std::vector<int> & codes, int n_cb, int T) {
    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) idx[t] = codes[(size_t) t * n_cb + cb];
        ggml_tensor * ids = g.input_i32(idx, T);
        return ggml_get_rows(ctx, m.w(name), ids);
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    if (rest) {
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        first = ggml_add(ctx, first, rest);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, first));
}

static CodecStreamCacheBlock & ensure_cache(
    VocoderStreamState & state,
    std::unordered_map<std::string, CodecStreamCacheBlock> & map,
    const std::string & name, int left, int channels
) {
    CodecStreamCacheBlock & b = map[name];
    if (b.left == left && b.channels == channels && left > 0) return b;

    if (!state.conv_storage) {
        throw std::runtime_error("Breeze vocoder state arena is not initialized");
    }

    const size_t align_f32 = 64u / sizeof(float);
    const size_t aligned =
        (state.conv_used_f32 + align_f32 - 1) & ~(align_f32 - 1);
    const size_t need = (size_t) left * (size_t) channels;
    if (aligned + need > state.conv_capacity_f32) {
        throw std::runtime_error("Breeze vocoder persistent state arena exhausted");
    }

    b.offset_f32 = aligned;
    b.left = left;
    b.channels = channels;
    state.conv_used_f32 = aligned + need;

    std::vector<float> zeros(need, 0.0f);
    for (int bank = 0; bank < 2; ++bank) {
        ggml_backend_tensor_set(
            state.conv_storage,
            zeros.data(),
            ((size_t) bank * state.conv_capacity_f32 + b.offset_f32) * sizeof(float),
            zeros.size() * sizeof(float)
        );
    }
    return b;
}

static ggml_tensor * cache_view(
    ggml_context * ctx, VocoderStreamState & state,
    const CodecStreamCacheBlock & block, int bank
) {
    if (bank < 0 || bank > 1) {
        throw std::runtime_error("invalid Breeze vocoder state bank");
    }
    const size_t base_f32 =
        (size_t) bank * state.conv_capacity_f32 + block.offset_f32;
    return ggml_view_2d(
        ctx,
        state.conv_storage,
        block.left,
        block.channels,
        (size_t) block.left * sizeof(float),
        base_f32 * sizeof(float)
    );
}

static void write_state(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    ggml_tensor * value, CodecStreamCacheBlock & block
) {
    if (block.left <= 0) return;
    if (!ggml_is_contiguous(value)) value = ggml_cont(ctx, value);
    ggml_tensor * dst = cache_view(ctx, state, block, state.conv_bank ^ 1);
    g.write(ggml_cpy(ctx, value, dst));
}

static void keep_input_tail(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    ggml_tensor * joined, CodecStreamCacheBlock & block
) {
    if (block.left <= 0) return;
    const int start = (int) joined->ne[0] - block.left;
    ggml_tensor * tail = ggml_view_2d(
        ctx, joined, block.left, block.channels, joined->nb[1],
        (size_t) start * joined->nb[0]
    );
    write_state(ctx, g, state, tail, block);
}

static ggml_tensor * stream_conv1d(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int dilation
) {
    const int K = (int) w->ne[0];
    const int left = (K - 1) * dilation;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return conv1d_causal(ctx, w, b, x, 1, dilation);

    CodecStreamCacheBlock & block = ensure_cache(state, state.conv1d, name, left, C);
    ggml_tensor * cache = cache_view(ctx, state, block, state.conv_bank);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);

    // Persistent state already supplies the exact causal left context.
    ggml_tensor * y = ggml_conv_1d(ctx, w, joined, 1, 0, dilation);
    if (y->ne[0] != N) {
        throw std::runtime_error("Breeze streaming Conv1d produced an unexpected length");
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    keep_input_tail(ctx, g, state, joined, block);
    return y;
}

static ggml_tensor * stream_depthwise(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int K
) {
    const int left = K - 1;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return depthwise1d_causal(ctx, w, b, x, K);

    CodecStreamCacheBlock & block = ensure_cache(state, state.conv1d, name, left, C);
    ggml_tensor * cache = cache_view(ctx, state, block, state.conv_bank);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    ggml_tensor * all = depthwise1d_causal(ctx, w, b, joined, K);
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, N, C, all->nb[1], (size_t) left * all->nb[0]
    ));
    keep_input_tail(ctx, g, state, joined, block);
    return y;
}

static ggml_tensor * stream_tconv(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int stride
) {
    const int K = (int) w->ne[0];
    const int OC = (int) w->ne[1];
    const int IC = (int) w->ne[2];
    const int N = (int) x->ne[0];
    const int trim = K - stride;
    const int emit = N * stride;

    if (trim <= 0) return convtr1d_causal(ctx, w, b, x, stride);
    if (emit < trim) {
        throw std::runtime_error("unsupported Breeze transposed-conv streaming geometry");
    }

    // Exact Qwen causal ConvTranspose1d streaming: carry the raw output
    // overlap [K-stride, OC] forward. This avoids recomputing cached inputs.
    CodecStreamCacheBlock & block =
        ensure_cache(state, state.tconv1d, name, trim, OC);
    ggml_tensor * carry = cache_view(ctx, state, block, state.conv_bank);

    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, (int64_t) K * OC, IC);
    w2 = ggml_cont(ctx, ggml_transpose(ctx, w2));
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * projected = ggml_cont(ctx, ggml_mul_mat(ctx, w2, xt));
    ggml_tensor * raw = ggml_col2im_1d(ctx, projected, stride, OC, 0);
    if (raw->ne[0] != emit + trim) {
        throw std::runtime_error("Breeze streaming ConvTranspose1d produced an unexpected length");
    }

    ggml_tensor * head = ggml_view_2d(
        ctx, raw, trim, OC, raw->nb[1], 0
    );
    ggml_tensor * y = ggml_add(ctx, head, carry);
    if (emit > trim) {
        ggml_tensor * mid = ggml_view_2d(
            ctx, raw, emit - trim, OC, raw->nb[1],
            (size_t) trim * raw->nb[0]
        );
        y = ggml_concat(ctx, y, mid, 0);
    }

    ggml_tensor * tail = ggml_view_2d(
        ctx, raw, trim, OC, raw->nb[1],
        (size_t) emit * raw->nb[0]
    );
    write_state(ctx, g, state, tail, block);

    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

static ggml_tensor * convnext_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::string & name, const std::string & p, ggml_tensor * x
) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = stream_depthwise(
        ctx, g, state, name, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]
    );
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    return ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
}

static ggml_tensor * residual_unit_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::string & name, const std::string & p, ggml_tensor * x, int dilation
) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = stream_conv1d(
        ctx, m, g, state, name,
        m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, dilation
    );
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    // conv2 is kernel 1 in the Qwen decoder and carries no temporal state.
    h = conv1d_causal(
        ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1
    );
    return ggml_add(ctx, x, h);
}

ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_cb, int T) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T);
    h = conv1d_causal(ctx, m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1, 1);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer(ctx, m, g, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsampling_ratios[i]);
        h = convnext(ctx, m, p, h);
    }

    h = conv1d_causal(ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(ctx, m, p + ".res." + std::to_string(j), h, dilations[j]);
        }
    }

    h = snake_beta(ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = conv1d_causal(ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

ggml_tensor * vocoder_decode_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::vector<int> & codes, int n_cb, int T
) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T);
    h = stream_conv1d(
        ctx, m, g, state, "pre_conv",
        m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1
    );

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer_stream(ctx, m, g, state, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = stream_tconv(
            ctx, g, state, "upsample_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
            c.upsampling_ratios[i]
        );
        h = convnext_stream(
            ctx, m, g, state, "upsample_" + std::to_string(i) + "_dwconv", p, h
        );
    }

    h = stream_conv1d(
        ctx, m, g, state, "decoder_pre_conv",
        m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1
    );

    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = stream_tconv(
            ctx, g, state, "decoder_block_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
            c.upsample_rates[i]
        );
        for (int j = 0; j < 3; j++) {
            h = residual_unit_stream(
                ctx, m, g, state,
                "decoder_block_" + std::to_string(i) +
                    "_residual_" + std::to_string(j) + "_conv1",
                p + ".res." + std::to_string(j), h, dilations[j]
            );
        }
    }

    h = snake_beta(ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = stream_conv1d(
        ctx, m, g, state, "final_conv",
        m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1
    );
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}


}
}
