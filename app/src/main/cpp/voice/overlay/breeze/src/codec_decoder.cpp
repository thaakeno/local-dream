#include "breeze/codec.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

static ggml_tensor * snake_beta(
    ggml_context * ctx, Graph & g, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb
) {
    // Match the tokenizer/reference implementation exactly:
    // x + sin(x * exp(alpha))^2 / (exp(beta) + 1e-9).
    // Do not rewrite this as exp(-beta): real Breeze decoder beta values can
    // drive that form to +Inf on HTP before the final clamp.
    const int C = (int) la->ne[0];
    std::vector<float> ones((size_t) C, 1.0f);
    std::vector<float> eps((size_t) C, 1.0e-9f);

    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, C);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, C);
    ggml_tensor * one = g.input_f32(ones, 1, C);
    ggml_tensor * tiny = g.input_f32(eps, 1, C);
    ggml_tensor * inv_beta = ggml_div(ctx, one, ggml_add(ctx, beta, tiny));

    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), inv_beta));
}

static ggml_tensor * convnext(
    ggml_context * ctx, BreezeModel & m, Graph & g, std::vector<VocoderDiagProbe> * probes,
    const std::string & p, ggml_tensor * x
) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = depthwise1d_causal(ctx, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]);
    h = vocoder_diag_probe(ctx, g, probes, p + ".dw", h);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = vocoder_diag_probe(ctx, g, probes, p + ".norm", h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = vocoder_diag_probe(ctx, g, probes, p + ".pw1", h);
    h = ggml_gelu_erf(ctx, h);
    h = vocoder_diag_probe(ctx, g, probes, p + ".gelu", h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = vocoder_diag_probe(ctx, g, probes, p + ".pw2", h);
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    h = vocoder_diag_probe(ctx, g, probes, p + ".gamma", h);
    ggml_tensor * out = ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
    return vocoder_diag_probe(ctx, g, probes, p + ".residual", out);
}

static ggml_tensor * residual_unit(
    ggml_context * ctx, BreezeModel & m, Graph & g, std::vector<VocoderDiagProbe> * probes,
    const std::string & p, ggml_tensor * x, int dilation
) {
    ggml_tensor * h = snake_beta(ctx, g, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = vocoder_diag_probe(ctx, g, probes, p + ".snake1", h);
    h = conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = vocoder_diag_probe(ctx, g, probes, p + ".conv1", h);
    h = snake_beta(ctx, g, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = vocoder_diag_probe(ctx, g, probes, p + ".snake2", h);
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    h = vocoder_diag_probe(ctx, g, probes, p + ".conv2", h);
    ggml_tensor * out = ggml_add(ctx, x, h);
    return vocoder_diag_probe(ctx, g, probes, p + ".residual", out);
}

static ggml_tensor * quantizer_decode(
    ggml_context * ctx, BreezeModel & m, Graph & g,
    const std::vector<int> & codes, int n_cb, int T,
    std::vector<VocoderDiagProbe> * probes
) {
    const int book = m.cfg.codec_codebook_size;
    if (n_cb <= 0 || T <= 0 || codes.size() != (size_t) n_cb * (size_t) T) {
        throw std::runtime_error("Breeze vocoder received malformed codec indices");
    }

    int total_invalid = 0;
    for (int cb = 0; cb < n_cb; ++cb) {
        int lo = std::numeric_limits<int>::max();
        int hi = std::numeric_limits<int>::min();
        int invalid = 0;
        for (int t = 0; t < T; ++t) {
            const int v = codes[(size_t) t * (size_t) n_cb + (size_t) cb];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            if (v < 0 || v >= book) invalid++;
        }
        total_invalid += invalid;
        std::fprintf(
            stderr,
            "[BREEZE_VOCODER_CODES] cb=%d min=%d max=%d invalid=%d book=%d pad=%d eos=%d\n",
            cb, lo, hi, invalid, book,
            m.cfg.codebook_pad_token_id,
            m.cfg.codebook_eos_token_id
        );
    }
    if (total_invalid != 0) {
        std::fprintf(
            stderr,
            "[BREEZE_VOCODER_CODES] INVALID_TOTAL=%d frames=%d codebooks=%d\n",
            total_invalid, T, n_cb
        );
        throw std::runtime_error(
            "Breeze vocoder received out-of-range codec index before dq lookup"
        );
    }

    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) {
            idx[t] = codes[(size_t) t * (size_t) n_cb + (size_t) cb];
        }
        ggml_tensor * ids = g.input_i32(idx, T);
        ggml_tensor * rows = ggml_get_rows(ctx, m.w(name), ids);
        return vocoder_diag_probe(
            ctx, g, probes, "dq.cb." + std::to_string(cb) + ".lookup", rows
        );
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);
    first = vocoder_diag_probe(ctx, g, probes, "dq.first.proj", first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        if (rest) {
            rest = ggml_add(ctx, rest, e);
            rest = vocoder_diag_probe(
                ctx, g, probes, "dq.rest.sum." + std::to_string(cb), rest
            );
        } else {
            rest = e;
        }
    }
    if (rest) {
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        rest = vocoder_diag_probe(ctx, g, probes, "dq.rest.proj", rest);
        first = ggml_add(ctx, first, rest);
        first = vocoder_diag_probe(ctx, g, probes, "dq.merge", first);
    }
    ggml_tensor * out = ggml_cont(ctx, ggml_transpose(ctx, first));
    return vocoder_diag_probe(ctx, g, probes, "dq.out", out);
}

static CodecStreamCacheBlock & ensure_cache(
    std::unordered_map<std::string, CodecStreamCacheBlock> & map,
    const std::string & name, int left, int channels
) {
    CodecStreamCacheBlock & b = map[name];
    if (b.left != left || b.channels != channels ||
        (int) b.data.size() != left * channels) {
        b.left = left;
        b.channels = channels;
        b.data.assign((size_t) left * channels, 0.0f);
    }
    return b;
}

static void keep_tail(
    ggml_context * ctx, Graph & g, ggml_tensor * joined,
    CodecStreamCacheBlock & block, int new_len,
    std::vector<StreamCacheUpdate> & updates
) {
    if (block.left <= 0) return;
    ggml_tensor * tail = ggml_view_2d(
        ctx, joined, block.left, block.channels, joined->nb[1],
        (size_t) new_len * joined->nb[0]
    );
    tail = ggml_cont(ctx, tail);
    ggml_set_output(tail);
    g.write(tail);
    updates.push_back({ &block, tail });
}

static ggml_tensor * stream_conv1d(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int dilation
) {
    const int K = (int) w->ne[0];
    const int left = (K - 1) * dilation;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return conv1d_causal(ctx, w, b, x, 1, dilation);

    CodecStreamCacheBlock & block = ensure_cache(state.conv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);

    // Cache already supplies the exact causal left padding, so use valid convolution.
    ggml_tensor * y = ggml_conv_1d(ctx, w, joined, 1, 0, dilation);
    if (y->ne[0] != N) {
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, N, y->ne[1], y->nb[1],
                                        (size_t) (y->ne[0] - N) * y->nb[0]));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    keep_tail(ctx, g, joined, block, N, updates);
    return y;
}

static ggml_tensor * stream_depthwise(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int K
) {
    const int left = K - 1;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return depthwise1d_causal(ctx, w, b, x, K);

    CodecStreamCacheBlock & block = ensure_cache(state.conv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    ggml_tensor * all = depthwise1d_causal(ctx, w, b, joined, K);
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, N, C, all->nb[1], (size_t) left * all->nb[0]
    ));
    keep_tail(ctx, g, joined, block, N, updates);
    return y;
}

static ggml_tensor * stream_tconv(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int stride
) {
    const int K = (int) w->ne[0];
    const int left = (K - 1) / stride;
    const int C = (int) x->ne[1];
    const int N = (int) x->ne[0];
    if (left <= 0) return convtr1d_causal(ctx, w, b, x, stride);

    CodecStreamCacheBlock & block = ensure_cache(state.tconv1d, name, left, C);
    ggml_tensor * cache = g.input_f32(block.data, left, C);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    ggml_tensor * all = convtr1d_causal(ctx, w, b, joined, stride);
    const int prefix = left * stride;
    const int new_len = N * stride;
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, new_len, all->ne[1], all->nb[1], (size_t) prefix * all->nb[0]
    ));
    keep_tail(ctx, g, joined, block, N, updates);
    return y;
}

static ggml_tensor * convnext_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    const std::string & p, ggml_tensor * x
) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = stream_depthwise(
        ctx, g, state, updates, name, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]
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
    std::vector<StreamCacheUpdate> & updates, const std::string & name,
    const std::string & p, ggml_tensor * x, int dilation
) {
    ggml_tensor * h = snake_beta(ctx, g, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = stream_conv1d(
        ctx, m, g, state, updates, name,
        m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, dilation
    );
    h = snake_beta(ctx, g, h, m.w(p + ".a2"), m.w(p + ".b2"));
    // The second convolution has no causal carry in the reference fast runtime.
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

ggml_tensor * vocoder_decode(
    ggml_context * ctx, BreezeModel & m, Graph & g,
    const std::vector<int> & codes, int n_cb, int T,
    std::vector<VocoderDiagProbe> * probes
) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T, probes);

    h = conv1d_causal(ctx, m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1, 1);
    h = vocoder_diag_probe(ctx, g, probes, "dpre.conv", h);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer(ctx, m, g, h, T, probes);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_diag_probe(ctx, g, probes, "dtf.transpose_out", h);

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsampling_ratios[i]);
        h = vocoder_diag_probe(ctx, g, probes, p + ".up", h);
        h = convnext(ctx, m, g, probes, p, h);
    }

    h = conv1d_causal(ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    h = vocoder_diag_probe(ctx, g, probes, "dhead.conv", h);

    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, g, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = vocoder_diag_probe(ctx, g, probes, p + ".snake", h);
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        h = vocoder_diag_probe(ctx, g, probes, p + ".up", h);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(
                ctx, m, g, probes, p + ".res." + std::to_string(j), h, dilations[j]
            );
        }
    }

    h = snake_beta(ctx, g, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = vocoder_diag_probe(ctx, g, probes, "dfin.snake", h);
    h = conv1d_causal(ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    h = vocoder_diag_probe(ctx, g, probes, "dfin.conv", h);
    h = ggml_clamp(ctx, h, -1.0f, 1.0f);
    return vocoder_diag_probe(ctx, g, probes, "output.clamp", h);
}

ggml_tensor * vocoder_decode_stream(
    ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStreamState & state,
    const std::vector<int> & codes, int n_cb, int T,
    std::vector<StreamCacheUpdate> & updates
) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T, nullptr);
    h = stream_conv1d(
        ctx, m, g, state, updates, "pre_conv",
        m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1
    );

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer_stream(ctx, m, g, state, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = stream_tconv(
            ctx, g, state, updates, "upsample_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h, c.upsampling_ratios[i]
        );
        h = convnext_stream(
            ctx, m, g, state, updates, "upsample_" + std::to_string(i) + "_dwconv", p, h
        );
    }

    h = stream_conv1d(
        ctx, m, g, state, updates, "decoder_pre_conv",
        m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1
    );

    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, g, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = stream_tconv(
            ctx, g, state, updates, "decoder_block_" + std::to_string(i) + "_tconv",
            m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h, c.upsample_rates[i]
        );
        for (int j = 0; j < 3; j++) {
            h = residual_unit_stream(
                ctx, m, g, state, updates,
                "decoder_block_" + std::to_string(i) + "_residual_" + std::to_string(j) + "_conv1",
                p + ".res." + std::to_string(j), h, dilations[j]
            );
        }
    }

    h = snake_beta(ctx, g, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = stream_conv1d(
        ctx, m, g, state, updates, "final_conv",
        m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1
    );
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

}
}
