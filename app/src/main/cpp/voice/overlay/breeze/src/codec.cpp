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

    // Two equally sized banks live in one HTP allocation. A streaming graph
    // only reads bank A and writes bank B (then swaps them after a successful
    // compute), so backend scheduling can never turn a state refresh into a
    // read/write alias hazard.
    conv_capacity_f32 = (16u * 1024u * 1024u) / sizeof(float);
    ggml_init_params p{
        ggml_tensor_overhead() * 8 + 4096,
        nullptr,
        true,
    };
    conv_ctx = ggml_init(p);
    if (!conv_ctx) throw std::runtime_error("failed to create Breeze vocoder state context");
    conv_storage = ggml_new_tensor_1d(
        conv_ctx, GGML_TYPE_F32, (int64_t) conv_capacity_f32 * 2
    );
    conv_buffer = ggml_backend_alloc_ctx_tensors(conv_ctx, model.backend.backend);
    if (!conv_buffer) {
        ggml_free(conv_ctx);
        conv_ctx = nullptr;
        conv_storage = nullptr;
        kv.free();
        throw std::runtime_error("failed to allocate persistent Breeze vocoder state on HTP");
    }

    conv_used_f32 = 0;
    conv_bank = 0;
    initialized = true;
    reset();
}

void VocoderStreamState::reset() {
    position = 0;
    conv_bank = 0;
    if (initialized) kv.reset();

    if (conv_storage && conv_used_f32 > 0) {
        std::vector<float> zeros(conv_used_f32, 0.0f);
        for (int bank = 0; bank < 2; ++bank) {
            ggml_backend_tensor_set(
                conv_storage,
                zeros.data(),
                ((size_t) bank * conv_capacity_f32) * sizeof(float),
                zeros.size() * sizeof(float)
            );
        }
    }
}

void VocoderStreamState::free() {
    if (initialized) kv.free();
    if (conv_buffer) ggml_backend_buffer_free(conv_buffer);
    if (conv_ctx) ggml_free(conv_ctx);
    conv_buffer = nullptr;
    conv_ctx = nullptr;
    conv_storage = nullptr;
    conv_capacity_f32 = 0;
    conv_used_f32 = 0;
    conv_bank = 0;
    initialized = false;
    position = 0;
    conv1d.clear();
    tconv1d.clear();
}

void MimiCodec::init(BreezeModel & model) {
    m = &model;
    stream.init(model);
    stream_safe_full = false;
    stream_history_frames = 0;
    stream_history_n_cb = 0;
    stream_history_codes.clear();
}

void MimiCodec::stream_reset() {
    if (!m) return;
    if (!stream.initialized) stream.init(*m);
    else stream.reset();
    stream_safe_full = false;
    stream_history_frames = 0;
    stream_history_n_cb = 0;
    stream_history_codes.clear();
}

static bool valid_stream_pcm(const std::vector<float> & audio, size_t want) {
    if (audio.size() < want || want == 0) return false;
    bool any_signal = false;
    for (size_t i = 0; i < want; ++i) {
        const float v = audio[i];
        if (!std::isfinite(v)) return false;
        any_signal = any_signal || v != 0.0f;
    }
    return any_signal;
}

std::vector<float> MimiCodec::decode_stream(const std::vector<int> & codes, int T, int n_cb) {
    if (!m || T <= 0) return {};
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    if (!stream.initialized) stream.init(*m);

    const size_t code_count = (size_t) T * (size_t) n_cb;
    if (codes.size() != code_count) {
        throw std::runtime_error("Breeze streaming codec received a malformed code chunk");
    }
    if (stream_history_n_cb == 0) stream_history_n_cb = n_cb;
    if (stream_history_n_cb != n_cb) {
        throw std::runtime_error("Breeze streaming codec codebook count changed mid-stream");
    }

    const int previous_frames = stream_history_frames;
    stream_history_codes.insert(stream_history_codes.end(), codes.begin(), codes.end());
    stream_history_frames += T;

    const size_t chunk_samples = (size_t) T * (size_t) m->cfg.samples_per_frame;
    auto exact_full_tail = [&]() -> std::vector<float> {
        std::vector<float> full = decode(
            stream_history_codes, stream_history_frames, stream_history_n_cb
        );
        const size_t total_samples =
            (size_t) stream_history_frames * (size_t) m->cfg.samples_per_frame;
        const size_t start =
            (size_t) previous_frames * (size_t) m->cfg.samples_per_frame;
        if (!valid_stream_pcm(full, total_samples) ||
            start + chunk_samples > full.size()) {
            return {};
        }
        return std::vector<float>(
            full.begin() + (ptrdiff_t) start,
            full.begin() + (ptrdiff_t) (start + chunk_samples)
        );
    };

    if (stream_safe_full) {
        return exact_full_tail();
    }

    std::vector<float> out;
    bool fast_ok = false;
    try {
        Graph g(32768);
        ggml_tensor * x = vocoder_decode_stream(g.ctx, *m, g, stream, codes, n_cb, T);
        ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));

        g.compute(m->backend, audio);
        out = tensor_to_f32(audio);
        fast_ok = valid_stream_pcm(out, chunk_samples);
    } catch (const std::exception & e) {
        std::fprintf(
            stderr,
            "[BREEZE_VOCODER_FALLBACK] incremental decode failed: %s\n",
            e.what()
        );
    }

    if (fast_ok) {
        stream.position += T;
        stream.kv.len = stream.position;
        stream.conv_bank ^= 1;
        return out;
    }

    stream_safe_full = true;
    std::fprintf(
        stderr,
        "[BREEZE_VOCODER_FALLBACK] invalid incremental PCM; switching to exact full-history decode at frame %d\n",
        stream_history_frames
    );
    return exact_full_tail();
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
