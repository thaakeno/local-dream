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
        if (probes || invalid != 0) {
            std::fprintf(
                stderr,
                "[BREEZE_VOCODER_CODES] cb=%d min=%d max=%d invalid=%d book=%d pad=%d eos=%d\n",
                cb, lo, hi, invalid, book,
                m.cfg.codebook_pad_token_id,
                m.cfg.codebook_eos_token_id
            );
        }
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
    VocoderStreamState & state,
    std::unordered_map<std::string, CodecStreamCacheBlock> & map,
    const std::string & name, int left, int channels
) {
    CodecStreamCacheBlock & block = map[name];
    if (block.left == left && block.channels == channels && left > 0) return block;

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

    block.offset_f32 = aligned;
    block.left = left;
    block.channels = channels;
    state.conv_used_f32 = aligned + need;

    std::vector<float> zeros(need, 0.0f);
    for (int bank = 0; bank < 2; ++bank) {
        ggml_backend_tensor_set(
            state.conv_storage,
            zeros.data(),
            ((size_t) bank * state.conv_capacity_f32 + block.offset_f32) * sizeof(float),
            zeros.size() * sizeof(float)
        );
    }
    return block;
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

    CodecStreamCacheBlock & block =
        ensure_cache(state, state.conv1d, name, left, C);
    ggml_tensor * cache = cache_view(ctx, state, block, state.conv_bank);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);

    // The persistent input tail supplies the exact causal left context, so the
    // convolution can run valid and compute only the fresh output rows.
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

    CodecStreamCacheBlock & block =
        ensure_cache(state, state.conv1d, name, left, C);
    ggml_tensor * cache = cache_view(ctx, state, block, state.conv_bank);
    ggml_tensor * joined = ggml_concat(ctx, cache, x, 0);
    ggml_tensor * all = depthwise1d_causal(ctx, w, b, joined, K);
    if (all->ne[0] < N) {
        throw std::runtime_error("Breeze streaming depthwise convolution returned a short output");
    }
    ggml_tensor * y = ggml_cont(ctx, ggml_view_2d(
        ctx, all, N, C, all->nb[1],
        (size_t) (all->ne[0] - N) * all->nb[0]
    ));
    keep_input_tail(ctx, g, state, joined, block);
    return y;
}

// Strict-HTP raw ConvTranspose1d lowering. This returns the uncropped output so
// incremental decoding can carry the exact K-stride output overlap forward.
static ggml_tensor * stream_convtr1d_raw(
    ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int stride
) {
    const int64_t K = w->ne[0];
    const int64_t OC = w->ne[1];
    const int64_t IC = w->ne[2];
    if (x->ne[1] != IC) {
        throw std::runtime_error("Breeze streaming ConvTranspose1d channel mismatch");
    }

    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, K * OC, IC);
    w2 = ggml_cont(ctx, ggml_transpose(ctx, w2));
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * projected = ggml_cont(ctx, ggml_mul_mat(ctx, w2, xt));
    return ggml_col2im_1d(ctx, projected, stride, (int) OC, 0);
}

static ggml_tensor * stream_tconv(
    ggml_context * ctx, Graph & g, VocoderStreamState & state,
    const std::string & name,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int stride
) {
    const int K = (int) w->ne[0];
    const int N = (int) x->ne[0];
    const int trim = K - stride;
    const int emit = N * stride;
    if (trim <= 0) return convtr1d_causal(ctx, w, b, x, stride);
    if (emit < trim) {
        throw std::runtime_error("unsupported Breeze transposed-conv streaming geometry");
    }

    // Fresh inputs are evaluated once. ConvTranspose1d overlap is output state:
    // add the previous raw tail to this chunk's head and carry the new raw tail.
    ggml_tensor * raw = stream_convtr1d_raw(ctx, w, x, stride);
    const int OC = (int) raw->ne[1];
    if (raw->ne[0] != emit + trim) {
        throw std::runtime_error("Breeze streaming ConvTranspose1d produced an unexpected length");
    }

    CodecStreamCacheBlock & block =
        ensure_cache(state, state.tconv1d, name, trim, OC);
    ggml_tensor * carry = cache_view(ctx, state, block, state.conv_bank);

    ggml_tensor * head = ggml_view_2d(ctx, raw, trim, OC, raw->nb[1], 0);
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
    ggml_tensor * h = snake_beta(ctx, g, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = stream_conv1d(
        ctx, m, g, state, name,
        m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, dilation
    );
    h = snake_beta(ctx, g, h, m.w(p + ".a2"), m.w(p + ".b2"));
    // In Breeze's decoder conv2 is temporal kernel 1, so it needs no carry.
    h = conv1d_causal(
        ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1
    );
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
    const std::vector<int> & codes, int n_cb, int T
) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T, nullptr);
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
        h = snake_beta(ctx, g, h, m.w(p + ".alpha"), m.w(p + ".beta"));
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

    h = snake_beta(ctx, g, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = stream_conv1d(
        ctx, m, g, state, "final_conv",
        m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1
    );
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}
}
}
