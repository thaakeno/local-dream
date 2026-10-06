#include "breeze/codec.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {
namespace codec_detail {

static ggml_tensor * debug_probe(
    CodecDebugProbes * probes, const std::string & name, ggml_tensor * tensor
) {
    if (probes && tensor) {
        ggml_set_output(tensor);
        probes->push_back({ name, tensor });
    }
    return tensor;
}

static ggml_tensor * snake_beta(
    ggml_context * ctx, Graph & g, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb
) {
    // Exact Breeze TTS 2 tokenizer semantics:
    // x + sin(x * exp(alpha))^2 / (exp(beta) + 1e-9)
    const int C = (int) la->ne[0];
    std::vector<float> ones((size_t) C, 1.0f);
    std::vector<float> eps((size_t) C, 1.0e-9f);

    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, C);
    ggml_tensor * beta  = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, C);
    ggml_tensor * one   = g.input_f32(ones, 1, C);
    ggml_tensor * tiny  = g.input_f32(eps, 1, C);
    ggml_tensor * inv_beta = ggml_div(ctx, one, ggml_add(ctx, beta, tiny));

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

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, Graph & g,
                                   const std::string & p, ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, g, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = snake_beta(ctx, g, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

static ggml_tensor * quantizer_decode(
    ggml_context * ctx, BreezeModel & m, Graph & g,
    const std::vector<int> & codes, int n_cb, int T, CodecDebugProbes * probes
) {
    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) idx[t] = codes[(size_t) t * n_cb + cb];
        ggml_tensor * ids = g.input_i32(idx, T);
        return ggml_get_rows(ctx, m.w(name), ids);
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    debug_probe(probes, "dq.first_embed", first);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);
    debug_probe(probes, "dq.first_proj", first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        debug_probe(probes, "dq.rest_embed." + std::to_string(cb), e);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    if (rest) {
        debug_probe(probes, "dq.rest_sum", rest);
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        debug_probe(probes, "dq.rest_proj", rest);
        first = ggml_add(ctx, first, rest);
        debug_probe(probes, "dq.combined", first);
    }
    return debug_probe(probes, "dq.transpose", ggml_cont(ctx, ggml_transpose(ctx, first)));
}

static CodecStreamCacheBlock & ensure_cache(
    std::unordered_map<std::string, CodecStreamCacheBlock> & map,
    const std::string & name, int left, int channels
) {
    CodecStreamCacheBlock & block = map[name];
    if (block.left != left || block.channels != channels ||
        (int) block.data.size() != left * channels) {
        block.left = left;
        block.channels = channels;
        block.data.assign((size_t) left * (size_t) channels, 0.0f);
    }
    return block;
}

static void keep_tail(
    ggml_context * ctx, Graph & g, ggml_tensor * joined,
    CodecStreamCacheBlock & block, std::vector<StreamCacheUpdate> & updates
) {
    if (block.left <= 0) return;
    const int start = (int) joined->ne[0] - block.left;
    ggml_tensor * tail = ggml_view_2d(
        ctx, joined, block.left, block.channels, joined->nb[1],
        (size_t) start * joined->nb[0]
    );
    tail = ggml_cont(ctx, tail);
    ggml_set_output(tail);
    g.write(tail);
    updates.push_back({ &block, tail });
}

static ggml_tensor * stream_conv1d(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int dilation,
    CodecDebugProbes * probes
) {
    const int K = (int) w->ne[0];
    const int left = (K - 1) * dilation;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return conv1d_causal(ctx, w, b, x, 1, dilation);

    CodecStreamCacheBlock & block = ensure_cache(state.conv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    debug_probe(probes, "conv." + name + ".joined", joined);

    // Keep the corrected reference math, but use the v153-safe ownership model:
    // the previous cache is an immutable graph input and the next cache is only
    // copied back to host memory after the whole HTP graph has completed.
    ggml_tensor * all = conv1d_causal(ctx, w, b, joined, 1, dilation);
    debug_probe(probes, "conv." + name + ".all", all);
    if (all->ne[0] < N) {
        throw std::runtime_error("Breeze streaming Conv1d returned a short output");
    }
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, N, all->ne[1], all->nb[1],
        (size_t) (all->ne[0] - N) * all->nb[0]
    ));
    debug_probe(probes, "conv." + name + ".out", y);
    keep_tail(ctx, g, joined, block, updates);
    return y;
}

static ggml_tensor * stream_depthwise(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int K,
    CodecDebugProbes * probes
) {
    const int left = K - 1;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return depthwise1d_causal(ctx, w, b, x, K);

    CodecStreamCacheBlock & block = ensure_cache(state.conv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    debug_probe(probes, "depthwise." + name + ".joined", joined);
    ggml_tensor * all = depthwise1d_causal(ctx, w, b, joined, K);
    debug_probe(probes, "depthwise." + name + ".all", all);
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, N, C, all->nb[1],
        (size_t) (all->ne[0] - N) * all->nb[0]
    ));
    debug_probe(probes, "depthwise." + name + ".out", y);
    keep_tail(ctx, g, joined, block, updates);
    return y;
}

static ggml_tensor * stream_tconv(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int stride,
    CodecDebugProbes * probes
) {
    const int K = (int) w->ne[0];
    const int left = (K - 1) / stride;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return convtr1d_causal(ctx, w, b, x, stride);

    CodecStreamCacheBlock & block = ensure_cache(state.tconv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    debug_probe(probes, "tconv." + name + ".joined", joined);

    ggml_tensor * all = convtr1d_causal(ctx, w, b, joined, stride);
    debug_probe(probes, "tconv." + name + ".all", all);
    const int prefix = left * stride;
    const int new_len = N * stride;
    if (all->ne[0] < prefix + new_len) {
        throw std::runtime_error("Breeze streaming ConvTranspose1d returned a short output");
    }
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, new_len, all->ne[1], all->nb[1],
        (size_t) prefix * all->nb[0]
    ));
    debug_probe(probes, "tconv." + name + ".out", y);
    keep_tail(ctx, g, joined, block, updates);
    return y;
}

static ggml_tensor * convnext_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates,
    const std::string & name, const std::string & p, ggml_tensor * x,
    CodecDebugProbes * probes
) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = stream_depthwise(
        ctx, g, state, updates, name, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1], probes
    );
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    debug_probe(probes, p + ".convnext.transpose", h);
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    debug_probe(probes, p + ".convnext.norm", h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    debug_probe(probes, p + ".convnext.pw1", h);
    h = ggml_gelu_erf(ctx, h);
    debug_probe(probes, p + ".convnext.gelu", h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    debug_probe(probes, p + ".convnext.pw2", h);
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    debug_probe(probes, p + ".convnext.gamma", h);
    return debug_probe(probes, p + ".convnext.residual",
                       ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h))));
}

static ggml_tensor * residual_unit_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates,
    const std::string & name, const std::string & p, ggml_tensor * x, int dilation,
    CodecDebugProbes * probes
) {
    ggml_tensor * h = snake_beta(ctx, g, x, m.w(p + ".a1"), m.w(p + ".b1"));
    debug_probe(probes, p + ".snake1", h);
    h = stream_conv1d(
        ctx, m, g, state, updates, name,
        m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, dilation, probes
    );
    debug_probe(probes, p + ".conv1", h);
    h = snake_beta(ctx, g, h, m.w(p + ".a2"), m.w(p + ".b2"));
    debug_probe(probes, p + ".snake2", h);
    h = conv1d_causal(
        ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1
    );
    debug_probe(probes, p + ".conv2", h);
    return debug_probe(probes, p + ".residual_out", ggml_add(ctx, x, h));
}

ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_cb, int T) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T, nullptr);
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
        h = snake_beta(ctx, g, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(ctx, m, g, p + ".res." + std::to_string(j), h, dilations[j]);
        }
    }

    h = snake_beta(ctx, g, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = conv1d_causal(ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

ggml_tensor * vocoder_decode_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::vector<int> & codes, int n_cb, int T,
    std::vector<StreamCacheUpdate> & updates, CodecDebugProbes * probes
) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T, probes);
    debug_probe(probes, "stage.quantizer", h);

    h = stream_conv1d(
        ctx, m, g, state, updates, "pre_conv",
        m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1, probes
    );
    debug_probe(probes, "stage.pre_conv", h);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    debug_probe(probes, "stage.dtf_input", h);
    h = vocoder_transformer_stream(ctx, m, g, state, h, T, probes);
    debug_probe(probes, "stage.dtf_output", h);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    debug_probe(probes, "stage.dtf_transpose", h);

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = stream_tconv(
            ctx, g, state, updates, "upsample_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
            c.upsampling_ratios[i], probes
        );
        debug_probe(probes, "stage.dup." + std::to_string(i) + ".tconv", h);
        h = convnext_stream(
            ctx, m, g, state, updates,
            "upsample_" + std::to_string(i) + "_dwconv", p, h, probes
        );
        debug_probe(probes, "stage.dup." + std::to_string(i) + ".convnext", h);
    }

    h = stream_conv1d(
        ctx, m, g, state, updates, "decoder_pre_conv",
        m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, probes
    );
    debug_probe(probes, "stage.decoder_pre_conv", h);

    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, g, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        debug_probe(probes, "stage.dblk." + std::to_string(i) + ".snake", h);
        h = stream_tconv(
            ctx, g, state, updates, "decoder_block_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
            c.upsample_rates[i], probes
        );
        debug_probe(probes, "stage.dblk." + std::to_string(i) + ".tconv", h);
        for (int j = 0; j < 3; j++) {
            h = residual_unit_stream(
                ctx, m, g, state, updates,
                "decoder_block_" + std::to_string(i) +
                    "_residual_" + std::to_string(j) + "_conv1",
                p + ".res." + std::to_string(j), h, dilations[j], probes
            );
            debug_probe(probes, "stage.dblk." + std::to_string(i) +
                ".res." + std::to_string(j), h);
        }
    }

    h = snake_beta(ctx, g, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    debug_probe(probes, "stage.final_snake", h);
    h = stream_conv1d(
        ctx, m, g, state, updates, "final_conv",
        m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, probes
    );
    debug_probe(probes, "stage.final_conv", h);
    return debug_probe(probes, "stage.final_clamp", ggml_clamp(ctx, h, -1.0f, 1.0f));
}


}
}
