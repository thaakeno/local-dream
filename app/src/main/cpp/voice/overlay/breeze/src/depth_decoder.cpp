#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <string>

namespace breeze {

static std::vector<float> llama3_freq_factors(const DepthConfig & c) {
    const int half = c.head_dim / 2;
    std::vector<float> ff(half, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float low_wl = c.rope_orig_ctx / c.rope_low_freq;
    const float high_wl = c.rope_orig_ctx / c.rope_high_freq;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(c.rope_theta, -2.0f * i / c.head_dim);
        float wavelen = 2.0f * pi / freq;
        if (wavelen > low_wl) {
            ff[i] = c.rope_factor;
        } else if (wavelen < high_wl) {
            ff[i] = 1.0f;
        } else {
            float smooth = (c.rope_orig_ctx / wavelen - c.rope_low_freq) / (c.rope_high_freq - c.rope_low_freq);
            ff[i] = 1.0f / ((1.0f - smooth) / c.rope_factor + smooth);
        }
    }
    return ff;
}

static void check_shape(ggml_tensor * t, const std::string & name,
                        int64_t n0, int64_t n1, int64_t n2 = -1) {
    if (t->ne[0] == n0 && t->ne[1] == n1 && (n2 < 0 || t->ne[2] == n2)) return;
    throw std::runtime_error(
        name + " is [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) + ", " +
        std::to_string(t->ne[2]) + "], expected [" + std::to_string(n0) + ", " + std::to_string(n1) +
        ", " + (n2 < 0 ? std::string("any") : std::to_string(n2)) + "]");
}

void DepthRunner::init(BreezeModel & m, int n_branches) {
    free();
    n_branch = n_branches;
    const DepthConfig & c = m.cfg.dd;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;

    // codebook slices require the checkpoint shapes to match the config
    check_shape(m.w("dd.in_proj.weight"), "dd.in_proj.weight", m.cfg.hidden_size, c.hidden);
    check_shape(m.w("audio_embd.weight"), "audio_embd.weight", m.cfg.hidden_size, (int64_t) nc * vs);
    check_shape(m.w("dd.codebooks_head.weight"), "dd.codebooks_head.weight", c.hidden, vs, nc - 1);

    kv.init(m.backend, c.n_layer, c.head_dim, c.n_kv_head, nc + 1, n_branches);
    freq_factors = llama3_freq_factors(c);
    steps.resize(nc - 1);
    profiled_frames = 0;
    profiled_set_ms = profiled_htp_ms = profiled_read_ms = profiled_sample_ms = 0.0;
}

void DepthRunner::begin_request() {
    // Keep graph allocations and tensor mirrors alive across requests.
    // run() resets KV positions for each codec frame before replay.
    kv.reset();
    profiled_frames = 0;
    profiled_set_ms = profiled_htp_ms = profiled_read_ms = profiled_sample_ms = 0.0;
}

void DepthRunner::free() {
    steps.clear();
    kv.free();
}

static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, Graph & g, KVCache & kv,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * ff,
                              ggml_tensor * mask, int start, int n) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = ggml_rope_ext(ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = cache_append(ctx, g, kv.k[il], k, start);
    ggml_tensor * vfull = cache_append(ctx, g, kv.v[il], v, start);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

static std::unique_ptr<DepthStep> build_depth_step(BreezeModel & m, DepthRunner & r, int head_idx) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const int start = head_idx == 0 ? 0 : head_idx + 1;
    const int n_pos = head_idx == 0 ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;
    auto step = std::make_unique<DepthStep>();
    Graph & g = step->graph;

    step->audio = g.input_i32(std::vector<int32_t>(nb), nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), step->audio);
    if (head_idx == 0) {
        step->hidden = g.input_f32(std::vector<float>((size_t) nb * m.cfg.hidden_size), m.cfg.hidden_size, nb);
        embed = ggml_concat(g.ctx, step->hidden, embed, 1);
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    std::vector<int32_t> pos_i(n_tok);
    for (int i = 0; i < n_tok; i++) pos_i[i] = start + i / nb;
    ggml_tensor * pos = g.input_i32(pos_i, n_tok);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    std::vector<float> mask_v = build_branch_causal_mask(n_tok, total, start, nb);
    ggml_tensor * mask = g.input_f32(mask_v, total, n_tok);

    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(g.ctx, m, g, r.kv, x, il, pos, ff, mask, start * nb, n_tok);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) head_idx * head->nb[2]);
    step->logits = ggml_mul_mat(g.ctx, hw, last);
    g.prepare(m.backend, step->logits);
    return step;
}



std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;
    if (sp_in) sp = *sp_in;

    std::vector<int32_t> idx(n_branch);
    std::vector<int> codes;
    codes.reserve((size_t) nc);
    codes.push_back(cb0);
    // Profile the true cost of the 15 serial steps. Large graph fusion proved
    // unsafe on SM8850, so optimize allocations and record separate host,
    // HTP execution and readback time without changing sampling semantics.
    const char * depth_profile = std::getenv("BREEZE_DEPTH_PROFILE");
    const bool profile = depth_profile && depth_profile[0] == '1';
    using Clock = std::chrono::steady_clock;
    double set_ms = 0.0, execute_ms = 0.0, read_ms = 0.0, sample_ms = 0.0;
    auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    std::vector<float> guided_logits;
    if (n_branch > 1) guided_logits.resize((size_t) vs);
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        if (!steps[head_idx]) steps[head_idx] = build_depth_step(m, *this, head_idx);
        DepthStep & step = *steps[head_idx];
        const auto ts = Clock::now();
        for (int b = 0; b < n_branch; b++) idx[b] = codes[head_idx] + head_idx * vs;
        ggml_backend_tensor_set(step.audio, idx.data(), 0, idx.size() * sizeof(int32_t));
        if (step.hidden) {
            for (int b = 0; b < n_branch; b++)
                ggml_backend_tensor_set(step.hidden, hiddens[b].data(),
                                        (size_t) b * m.cfg.hidden_size * sizeof(float),
                                        (size_t) m.cfg.hidden_size * sizeof(float));
        }
        if (profile) set_ms += elapsed(ts);
        const auto te = Clock::now();
        step.graph.replay(m.backend);
        if (profile) execute_ms += elapsed(te);
        const auto tr = Clock::now();
        std::vector<float> out = tensor_to_f32(step.logits);
        if (profile) read_ms += elapsed(tr);

        const auto tp = Clock::now();
        int next_code;
        if (j <= n_force) {
            next_code = force[j - 1];
        } else if (n_branch == 1) {
            // Exact same sampling distribution, but avoid allocating/copying
            // a second full logits vector for every residual codebook.
            next_code = sample_token(out, sp, rng);
        } else {
            const int vocab = (int) out.size() / n_branch;
            if ((int) guided_logits.size() != vocab) guided_logits.resize((size_t) vocab);
            for (int i = 0; i < vocab; i++)
                guided_logits[(size_t) i] = out[vocab + i] + cfg_scale * (out[i] - out[vocab + i]);
            next_code = sample_token(guided_logits, sp, rng);
        }
        codes.push_back(next_code);
        if (profile) sample_ms += elapsed(tp);
    }
    if (profile) {
        ++profiled_frames;
        profiled_set_ms += set_ms;
        profiled_htp_ms += execute_ms;
        profiled_read_ms += read_ms;
        profiled_sample_ms += sample_ms;
        if (profiled_frames == 1 || profiled_frames % 32 == 0) {
            const double n = (double) profiled_frames;
            std::fprintf(
                stderr,
                "[BREEZE_DEPTH_BREAKDOWN] frames=%d steps=%d branches=%d "
                "set_ms=%.2f htp_ms=%.2f read_ms=%.2f sample_ms=%.2f total_ms=%.2f\n",
                profiled_frames, nc - 1, n_branch,
                profiled_set_ms / n, profiled_htp_ms / n,
                profiled_read_ms / n, profiled_sample_ms / n,
                (profiled_set_ms + profiled_htp_ms +
                 profiled_read_ms + profiled_sample_ms) / n
            );
        }
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

}
