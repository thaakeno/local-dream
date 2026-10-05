#include "breeze/codec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace breeze {

using namespace codec_detail;

static ggml_tensor * transpose_cont(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

void VocoderStreamState::init(BreezeModel & model) {
    if (initialized) free();

    const VocoderConfig & c = model.cfg.voc;
    const int max_seq = std::max(model.cfg.max_new_tokens + 32, c.sliding_window + 64);
    kv.init(model.backend, c.n_layer, c.head_dim, c.n_kv_head, max_seq);

    // Match Breeze's reference fast-streaming runtime: every state block is a
    // real persistent device tensor with its own shape. Do not manufacture
    // cross-context views into a monolithic arena.
    ggml_init_params p{
        ggml_tensor_overhead() * 256 + 65536,
        nullptr,
        true,
    };
    conv_ctx = ggml_init(p);
    if (!conv_ctx) {
        kv.free();
        throw std::runtime_error("failed to create Breeze vocoder state context");
    }

    auto add_state = [&](std::unordered_map<std::string, CodecStreamCacheBlock> & map,
                         const std::string & name, int left, int channels) {
        if (left <= 0) return;
        CodecStreamCacheBlock block;
        block.left = left;
        block.channels = channels;
        block.tensor = ggml_new_tensor_2d(conv_ctx, GGML_TYPE_F32, left, channels);
        ggml_set_name(block.tensor, name.c_str());
        map.emplace(name, block);
    };

    auto add_conv = [&](const std::string & name, const std::string & weight, int dilation) {
        ggml_tensor * w = model.w(weight);
        add_state(conv1d, name, ((int) w->ne[0] - 1) * dilation, (int) w->ne[1]);
    };
    auto add_tconv = [&](const std::string & name, const std::string & weight, int stride) {
        ggml_tensor * w = model.w(weight);
        add_state(tconv1d, name, ((int) w->ne[0] - 1) / stride, (int) w->ne[2]);
    };

    add_conv("pre_conv", "codec.dpre.conv.weight", 1);

    for (size_t i = 0; i < c.upsampling_ratios.size(); ++i) {
        const std::string p = "codec.dup." + std::to_string(i);
        add_tconv(
            "upsample_" + std::to_string(i) + "_tconv",
            p + ".up.conv.weight",
            c.upsampling_ratios[i]
        );
        ggml_tensor * dw = model.w(p + ".dw.weight");
        add_state(
            conv1d,
            "upsample_" + std::to_string(i) + "_dwconv",
            (int) dw->ne[1] - 1,
            (int) dw->ne[0]
        );
    }

    add_conv("decoder_pre_conv", "codec.dhead.conv.weight", 1);

    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); ++i) {
        const std::string p = "codec.dblk." + std::to_string(i);
        add_tconv(
            "decoder_block_" + std::to_string(i) + "_tconv",
            p + ".up.conv.weight",
            c.upsample_rates[i]
        );
        for (int j = 0; j < 3; ++j) {
            add_conv(
                "decoder_block_" + std::to_string(i) +
                    "_residual_" + std::to_string(j) + "_conv1",
                p + ".res." + std::to_string(j) + ".conv1.conv.weight",
                dilations[j]
            );
        }
    }

    add_conv("final_conv", "codec.dfin.conv.weight", 1);

    conv_buffer = ggml_backend_alloc_ctx_tensors(conv_ctx, model.backend.backend);
    if (!conv_buffer) {
        ggml_free(conv_ctx);
        conv_ctx = nullptr;
        conv1d.clear();
        tconv1d.clear();
        kv.free();
        throw std::runtime_error("failed to allocate persistent Breeze vocoder state on HTP");
    }

    initialized = true;
    reset();
}

void VocoderStreamState::reset() {
    position = 0;
    if (initialized) kv.reset();

    auto clear_map = [](auto & map) {
        for (auto & it : map) {
            CodecStreamCacheBlock & block = it.second;
            const size_t count = (size_t) block.left * (size_t) block.channels;
            std::vector<float> zeros(count, 0.0f);
            ggml_backend_tensor_set(
                block.tensor, zeros.data(), 0, zeros.size() * sizeof(float)
            );
        }
    };
    clear_map(conv1d);
    clear_map(tconv1d);
}

void VocoderStreamState::free() {
    if (initialized) kv.free();
    if (conv_buffer) ggml_backend_buffer_free(conv_buffer);
    if (conv_ctx) ggml_free(conv_ctx);
    conv_buffer = nullptr;
    conv_ctx = nullptr;
    initialized = false;
    position = 0;
    conv1d.clear();
    tconv1d.clear();
}

void MimiCodec::init(BreezeModel & model) {
    m = &model;
    stream.init(model);
}

void MimiCodec::stream_reset() {
    if (!m) return;
    if (!stream.initialized) stream.init(*m);
    else stream.reset();
}

std::vector<float> MimiCodec::decode_stream(const std::vector<int> & codes, int T, int n_cb) {
    if (!m || T <= 0) return {};
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    if (!stream.initialized) stream.init(*m);

    const size_t code_count = (size_t) T * (size_t) n_cb;
    if (codes.size() != code_count) {
        throw std::runtime_error("Breeze streaming codec received a malformed code chunk");
    }

    Graph g(32768);
    ggml_tensor * x = vocoder_decode_stream(g.ctx, *m, g, stream, codes, n_cb, T);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);

    std::vector<float> out = tensor_to_f32(audio);
    const size_t want = (size_t) T * (size_t) m->cfg.samples_per_frame;
    if (out.size() < want) {
        throw std::runtime_error("Breeze streaming vocoder returned a short PCM chunk");
    }

    bool any_signal = false;
    for (size_t i = 0; i < want; ++i) {
        const float v = out[i];
        if (!std::isfinite(v)) {
            throw std::runtime_error("Breeze streaming vocoder produced non-finite PCM");
        }
        any_signal = any_signal || v != 0.0f;
    }
    if (!any_signal) {
        throw std::runtime_error("Breeze streaming vocoder produced an all-zero PCM chunk");
    }

    // State advances only after a numerically valid chunk completed.
    stream.position += T;
    stream.kv.len = stream.position;
    return out;
}

std::vector<float> MimiCodec::decode(const std::vector<int> & codes, int T, int n_cb) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    Graph g(32768);
    ggml_tensor * x = vocoder_decode(g.ctx, *m, g, codes, n_cb, T);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);
    return tensor_to_f32(audio);
}

static int nearest(const std::vector<float> & book, const float * v, int dim, int n) {
    int best = 0;
    float best_d = FLT_MAX;
    for (int c = 0; c < n; c++) {
        const float * b = &book[(size_t) c * dim];
        float d = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = v[k] - b[k];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

static std::vector<float> read_book(BreezeModel & m, const std::string & name) {
    return tensor_to_f32(m.w(name));
}

std::vector<int> MimiCodec::encode(const std::vector<float> & audio, int & n_frames) {
    const int nc = m->cfg.num_codebooks;
    const int n_sem = m->cfg.codec.num_semantic;
    const int n_ac = nc - n_sem;
    const int dim = m->cfg.codec.codebook_dim;
    const int book = m->cfg.codec_codebook_size;

    Graph g(16384);
    ggml_tensor * x = g.input_f32(audio, (int) audio.size(), 1);
    x = seanet_encoder(g.ctx, *m, x);
    x = transpose_cont(g.ctx, x);
    x = mimi_transformer(g.ctx, *m, g, x, "codec.enct", (int) x->ne[1]);
    x = transpose_cont(g.ctx, x);
    x = conv1d_causal(g.ctx, m->w("codec.downsample.conv.weight"), nullptr, x, 2, 1);
    ggml_tensor * emb = transpose_cont(g.ctx, x);
    ggml_tensor * rs = linear(g.ctx, m->w("codec.sq.in_proj.weight"), emb);
    ggml_tensor * ra = linear(g.ctx, m->w("codec.aq.in_proj.weight"), emb);
    ggml_set_output(rs);
    g.write(rs);
    g.compute(m->backend, ra);

    std::vector<float> rs_h = tensor_to_f32(rs);
    std::vector<float> ra_h = tensor_to_f32(ra);
    const int T = (int) ra->ne[1];
    n_frames = T;

    std::vector<float> sbook = read_book(*m, "codec.sq.0.embed");
    std::vector<std::vector<float>> abook(n_ac);
    for (int i = 0; i < n_ac; i++) abook[i] = read_book(*m, "codec.aq." + std::to_string(i) + ".embed");

    std::vector<int> codes((size_t) T * nc);
    std::vector<float> res(dim);
    for (int f = 0; f < T; f++) {
        codes[(size_t) f * nc + 0] = nearest(sbook, &rs_h[(size_t) f * dim], dim, book);
        for (int k = 0; k < dim; k++) res[k] = ra_h[(size_t) f * dim + k];
        for (int i = 0; i < n_ac; i++) {
            int c = nearest(abook[i], res.data(), dim, book);
            codes[(size_t) f * nc + (n_sem + i)] = c;
            const float * b = &abook[i][(size_t) c * dim];
            for (int k = 0; k < dim; k++) res[k] -= b[k];
        }
    }
    return codes;
}

}
