#include "breeze/codec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace breeze {

using namespace codec_detail;

static ggml_tensor * transpose_cont(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

struct DiagStats {
    size_t n = 0;
    size_t finite = 0;
    size_t nan = 0;
    size_t pinf = 0;
    size_t ninf = 0;
    size_t zero = 0;
    size_t tiny = 0;
    size_t first_bad = (size_t) -1;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double rms = 0.0;
    double max_abs = 0.0;
};

static DiagStats diag_stats(const std::vector<float> & v) {
    DiagStats s;
    s.n = v.size();
    double sum = 0.0;
    double sum_sq = 0.0;
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = v[i];
        if (std::isnan(x)) {
            s.nan++;
            if (s.first_bad == (size_t) -1) s.first_bad = i;
            continue;
        }
        if (std::isinf(x)) {
            if (x > 0) s.pinf++; else s.ninf++;
            if (s.first_bad == (size_t) -1) s.first_bad = i;
            continue;
        }
        s.finite++;
        if (x == 0.0f) s.zero++;
        if (std::fabs(x) < 1.0e-12f) s.tiny++;
        mn = std::min(mn, (double) x);
        mx = std::max(mx, (double) x);
        s.max_abs = std::max(s.max_abs, std::fabs((double) x));
        sum += x;
        sum_sq += (double) x * (double) x;
    }
    if (s.finite > 0) {
        s.min = mn;
        s.max = mx;
        s.mean = sum / (double) s.finite;
        s.rms = std::sqrt(sum_sq / (double) s.finite);
    }
    return s;
}

static std::string diag_head(const std::vector<float> & v, size_t count = 6) {
    std::ostringstream os;
    os << "[";
    const size_t n = std::min(count, v.size());
    for (size_t i = 0; i < n; ++i) {
        if (i) os << ",";
        if (std::isnan(v[i])) os << "nan";
        else if (std::isinf(v[i])) os << (v[i] > 0 ? "+inf" : "-inf");
        else os << std::setprecision(6) << v[i];
    }
    if (v.size() > n) os << ",...";
    os << "]";
    return os.str();
}

static DiagStats log_values(
    const char * tag, const std::string & name, const std::vector<float> & v,
    const std::string & shape
) {
    const DiagStats s = diag_stats(v);
    std::fprintf(
        stderr,
        "[%s] name=%s shape=%s n=%zu finite=%zu nan=%zu +inf=%zu -inf=%zu "
        "zero=%zu tiny=%zu min=%.8g max=%.8g mean=%.8g rms=%.8g maxabs=%.8g "
        "first_bad=%lld head=%s\n",
        tag, name.c_str(), shape.c_str(), s.n, s.finite, s.nan, s.pinf, s.ninf,
        s.zero, s.tiny, s.min, s.max, s.mean, s.rms, s.max_abs,
        s.first_bad == (size_t) -1 ? -1LL : (long long) s.first_bad,
        diag_head(v).c_str()
    );
    return s;
}

static DiagStats log_tensor(const char * tag, const std::string & name, ggml_tensor * t) {
    if (!t) {
        std::fprintf(stderr, "[%s] name=%s tensor=null\n", tag, name.c_str());
        DiagStats s;
        s.nan = 1;
        return s;
    }
    std::ostringstream shape;
    shape << t->ne[0] << "x" << t->ne[1] << "x" << t->ne[2] << "x" << t->ne[3];
    if (t->type != GGML_TYPE_F32) {
        std::fprintf(
            stderr,
            "[%s] name=%s shape=%s type=%s bytes=%zu stats=skipped_non_f32\n",
            tag, name.c_str(), shape.str().c_str(), ggml_type_name(t->type), ggml_nbytes(t)
        );
        DiagStats s;
        s.n = (size_t) ggml_nelements(t);
        return s;
    }
    return log_values(tag, name, tensor_to_f32(t), shape.str());
}

static void log_weight(BreezeModel & m, const std::string & name) {
    ggml_tensor * t = m.w(name);
    std::fprintf(
        stderr,
        "[BREEZE_DIAG_WEIGHT] name=%s type=%s shape=%lldx%lldx%lldx%lld bytes=%zu\n",
        name.c_str(), ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1],
        (long long) t->ne[2], (long long) t->ne[3], ggml_nbytes(t)
    );
}

static void log_codec_weight_map(BreezeModel & m) {
    const VocoderConfig & c = m.cfg.voc;
    log_weight(m, "codec.dq.first.0.embed");
    log_weight(m, "codec.dq.first.out_proj.weight");
    log_weight(m, "codec.dq.rest.out_proj.weight");
    log_weight(m, "codec.dpre.conv.weight");
    log_weight(m, "codec.dtf.in_proj.weight");
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        log_weight(m, p + ".attn_q.weight");
        log_weight(m, p + ".attn_k.weight");
        log_weight(m, p + ".attn_v.weight");
        log_weight(m, p + ".attn_output.weight");
        log_weight(m, p + ".ffn_gate.weight");
        log_weight(m, p + ".ffn_up.weight");
        log_weight(m, p + ".ffn_down.weight");
    }
    log_weight(m, "codec.dtf.out_proj.weight");
    for (size_t i = 0; i < c.upsampling_ratios.size(); ++i) {
        const std::string p = "codec.dup." + std::to_string(i);
        log_weight(m, p + ".up.conv.weight");
        log_weight(m, p + ".dw.weight");
        log_weight(m, p + ".pw1.weight");
        log_weight(m, p + ".pw2.weight");
    }
    log_weight(m, "codec.dhead.conv.weight");
    for (size_t i = 0; i < c.upsample_rates.size(); ++i) {
        const std::string p = "codec.dblk." + std::to_string(i);
        log_weight(m, p + ".up.conv.weight");
        for (int j = 0; j < 3; ++j) {
            const std::string r = p + ".res." + std::to_string(j);
            log_weight(m, r + ".conv1.conv.weight");
            log_weight(m, r + ".conv2.conv.weight");
        }
    }
    log_weight(m, "codec.dfin.conv.weight");
}

static void log_cache_map(
    const char * tag,
    const std::unordered_map<std::string, CodecStreamCacheBlock> & map
) {
    for (const auto & it : map) {
        log_tensor(tag, it.first, it.second.tensor);
    }
}

static void log_kv_first_slot(const char * tag, const KVCache & kv) {
    for (size_t il = 0; il < kv.k.size(); ++il) {
        const size_t count = (size_t) kv.head_dim * (size_t) kv.n_kv_head;
        std::vector<float> k(count), v(count);
        ggml_backend_tensor_get(kv.k[il], k.data(), 0, count * sizeof(float));
        ggml_backend_tensor_get(kv.v[il], v.data(), 0, count * sizeof(float));
        log_values(tag, "kv." + std::to_string(il) + ".k0", k,
                   std::to_string(kv.head_dim) + "x" + std::to_string(kv.n_kv_head));
        log_values(tag, "kv." + std::to_string(il) + ".v0", v,
                   std::to_string(kv.head_dim) + "x" + std::to_string(kv.n_kv_head));
    }
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

    const bool first_frame = stream.position == 0;
    if (first_frame) {
        int invalid = 0;
        int cmin = codes.empty() ? 0 : codes[0];
        int cmax = codes.empty() ? 0 : codes[0];
        std::ostringstream values;
        values << "[";
        for (size_t i = 0; i < codes.size(); ++i) {
            if (i) values << ",";
            values << codes[i];
            cmin = std::min(cmin, codes[i]);
            cmax = std::max(cmax, codes[i]);
            if (codes[i] < 0 || codes[i] >= m->cfg.codec_codebook_size) invalid++;
        }
        values << "]";
        std::fprintf(
            stderr,
            "[BREEZE_DIAG_BEGIN] position=%d T=%d n_cb=%d code_count=%zu codec_book=%d "
            "sample_rate=%d samples_per_frame=%d codes_min=%d codes_max=%d "
            "invalid_codes=%d codes=%s\n",
            stream.position, T, n_cb, codes.size(), m->cfg.codec_codebook_size,
            m->cfg.sample_rate, m->cfg.samples_per_frame, cmin, cmax, invalid,
            values.str().c_str()
        );
        std::fprintf(
            stderr,
            "[BREEZE_DIAG_CONFIG] voc_layers=%d hidden=%d heads=%d kv_heads=%d "
            "head_dim=%d sliding_window=%d upsampling_stages=%zu decoder_stages=%zu\n",
            m->cfg.voc.n_layer, m->cfg.voc.hidden, m->cfg.voc.n_head,
            m->cfg.voc.n_kv_head, m->cfg.voc.head_dim, m->cfg.voc.sliding_window,
            m->cfg.voc.upsampling_ratios.size(), m->cfg.voc.upsample_rates.size()
        );
        log_codec_weight_map(*m);
        log_cache_map("BREEZE_DIAG_CACHE_PRE", stream.conv1d);
        log_cache_map("BREEZE_DIAG_CACHE_PRE", stream.tconv1d);
    }

    Graph g(32768);
    ggml_tensor * x = vocoder_decode_stream(g.ctx, *m, g, stream, codes, n_cb, T, nullptr);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);

    std::vector<float> out = tensor_to_f32(audio);
    const size_t want = (size_t) T * (size_t) m->cfg.samples_per_frame;
    const DiagStats production_stats = first_frame
        ? log_values("BREEZE_DIAG_PRODUCTION_PCM", "stream.production", out,
                     std::to_string(out.size()))
        : diag_stats(out);

    bool any_signal = false;
    bool invalid_pcm = out.size() < want;
    for (size_t i = 0; i < std::min(want, out.size()); ++i) {
        const float v = out[i];
        if (!std::isfinite(v)) invalid_pcm = true;
        any_signal = any_signal || v != 0.0f;
    }
    if (!any_signal) invalid_pcm = true;

    if (first_frame && invalid_pcm) {
        std::fprintf(
            stderr,
            "[BREEZE_DIAG_REPLAY_BEGIN] reason=invalid_first_stream_pcm production_nan=%zu "
            "production_pinf=%zu production_ninf=%zu\n",
            production_stats.nan, production_stats.pinf, production_stats.ninf
        );

        VocoderStreamState diag_state;
        try {
            diag_state.init(*m);
            log_cache_map("BREEZE_DIAG_REPLAY_CACHE_PRE", diag_state.conv1d);
            log_cache_map("BREEZE_DIAG_REPLAY_CACHE_PRE", diag_state.tconv1d);

            CodecDebugProbes probes;
            Graph dg(65536);
            ggml_tensor * dx = vocoder_decode_stream(
                dg.ctx, *m, dg, diag_state, codes, n_cb, T, &probes
            );
            ggml_tensor * daudio =
                ggml_cont(dg.ctx, ggml_reshape_1d(dg.ctx, dx, dx->ne[0]));
            ggml_set_output(daudio);
            dg.compute(m->backend, daudio);

            std::string first_bad;
            std::string previous_good = "none";
            for (const CodecDebugProbe & p : probes) {
                const DiagStats ps = log_tensor("BREEZE_DIAG_STAGE", p.name, p.tensor);
                const bool bad = ps.nan + ps.pinf + ps.ninf > 0;
                if (bad && first_bad.empty()) {
                    first_bad = p.name;
                    std::fprintf(
                        stderr,
                        "[BREEZE_DIAG_FIRST_BAD] stage=%s previous_good=%s nan=%zu "
                        "+inf=%zu -inf=%zu maxabs=%.8g\n",
                        p.name.c_str(), previous_good.c_str(),
                        ps.nan, ps.pinf, ps.ninf, ps.max_abs
                    );
                }
                if (!bad) previous_good = p.name;
            }

            const std::vector<float> diagnostic_pcm = tensor_to_f32(daudio);
            const DiagStats ds = log_values(
                "BREEZE_DIAG_REPLAY_PCM", "stream.instrumented",
                diagnostic_pcm, std::to_string(diagnostic_pcm.size())
            );
            log_cache_map("BREEZE_DIAG_REPLAY_CACHE_POST", diag_state.conv1d);
            log_cache_map("BREEZE_DIAG_REPLAY_CACHE_POST", diag_state.tconv1d);
            log_kv_first_slot("BREEZE_DIAG_REPLAY_KV_POST", diag_state.kv);

            std::fprintf(
                stderr,
                "[BREEZE_DIAG_REPLAY_SUMMARY] probes=%zu first_bad=%s instrumented_nonfinite=%zu "
                "production_nonfinite=%zu\n",
                probes.size(), first_bad.empty() ? "none" : first_bad.c_str(),
                ds.nan + ds.pinf + ds.ninf,
                production_stats.nan + production_stats.pinf + production_stats.ninf
            );
            diag_state.free();
        } catch (const std::exception & e) {
            diag_state.free();
            std::fprintf(stderr, "[BREEZE_DIAG_REPLAY_ERROR] %s\n", e.what());
        }

        try {
            Graph rg(32768);
            ggml_tensor * rx = vocoder_decode(rg.ctx, *m, rg, codes, n_cb, T);
            ggml_tensor * raudio =
                ggml_cont(rg.ctx, ggml_reshape_1d(rg.ctx, rx, rx->ne[0]));
            rg.compute(m->backend, raudio);
            const std::vector<float> reference_pcm = tensor_to_f32(raudio);
            const DiagStats rs = log_values(
                "BREEZE_DIAG_OFFLINE_PCM", "offline.one_frame",
                reference_pcm, std::to_string(reference_pcm.size())
            );
            std::fprintf(
                stderr,
                "[BREEZE_DIAG_OFFLINE_SUMMARY] nonfinite=%zu classification=%s\n",
                rs.nan + rs.pinf + rs.ninf,
                (rs.nan + rs.pinf + rs.ninf) == 0
                    ? "streaming_specific"
                    : "common_codec_or_htp"
            );
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[BREEZE_DIAG_OFFLINE_ERROR] %s\n", e.what());
        }

        std::fprintf(stderr, "[BREEZE_DIAG_END] first_frame_failure=1\n");
    }

    if (out.size() < want) {
        throw std::runtime_error("Breeze streaming vocoder returned a short PCM chunk");
    }
    for (size_t i = 0; i < want; ++i) {
        if (!std::isfinite(out[i])) {
            throw std::runtime_error("Breeze streaming vocoder produced non-finite PCM");
        }
    }
    if (!any_signal) {
        throw std::runtime_error("Breeze streaming vocoder produced an all-zero PCM chunk");
    }

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
