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
    frame_fast.reset();
    frame_fast_disabled = false;
    profiled_frames = 0;
    profiled_set_ms = profiled_htp_ms = profiled_read_ms = profiled_sample_ms = 0.0;
}

void DepthRunner::free() {
    frame_fast.reset();
    frame_fast_disabled = false;
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


static ggml_tensor * depth_audio_book_view(
    ggml_context * ctx, BreezeModel & m, int book
) {
    const int vs = m.cfg.audio_vocab_size;
    ggml_tensor * all = m.w("audio_embd.weight");
    if (book < 0 || book >= m.cfg.num_codebooks) {
        throw std::runtime_error("depth audio book index out of range");
    }
    return ggml_view_2d(
        ctx,
        all,
        all->ne[0],
        vs,
        all->nb[1],
        (size_t) book * (size_t) vs * all->nb[1]
    );
}

static ggml_tensor * depth_layer_chained(
    ggml_context * ctx,
    BreezeModel & m,
    Graph & g,
    ggml_tensor * x,
    int il,
    ggml_tensor * pos,
    ggml_tensor * ff,
    ggml_tensor * mask,
    int n,
    std::vector<ggml_tensor *> & keys,
    std::vector<ggml_tensor *> & values
) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(
        ctx,
        linear(ctx, m.w(p + ".attn_q.weight"), h),
        c.head_dim,
        c.n_head,
        n
    );
    ggml_tensor * k = ggml_reshape_3d(
        ctx,
        linear(ctx, m.w(p + ".attn_k.weight"), h),
        c.head_dim,
        c.n_kv_head,
        n
    );
    ggml_tensor * v = ggml_reshape_3d(
        ctx,
        linear(ctx, m.w(p + ".attn_v.weight"), h),
        c.head_dim,
        c.n_kv_head,
        n
    );
    q = ggml_rope_ext(
        ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX,
        0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );
    k = ggml_rope_ext(
        ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX,
        0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );

    ggml_tensor * kfull = keys[il] ? ggml_concat(ctx, keys[il], k, 2) : k;
    ggml_tensor * vfull = values[il] ? ggml_concat(ctx, values[il], v, 2) : v;
    keys[il] = kfull;
    values[il] = vfull;

    ggml_tensor * a = attention(
        ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head
    );
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(
        ctx,
        h,
        m.w(p + ".ffn_gate.weight"),
        m.w(p + ".ffn_up.weight"),
        m.w(p + ".ffn_down.weight")
    );
    return ggml_add(ctx, res, h);
}

static ggml_tensor * depth_sample_in_graph(
    Graph & g,
    ggml_tensor * logits,
    ggml_tensor * uniform,
    int top_k,
    int sample_width,
    float temperature,
    ggml_tensor * pad
) {
    // argsort_top_k is descending, matching the host sampler's top-k set/order.
    // Reshape logits to [1, vocab] so GET_ROWS can gather arbitrary vocabulary
    // indices into one compact candidate row.
    ggml_tensor * top_ids = ggml_argsort_top_k(g.ctx, logits, top_k);
    ggml_tensor * rows = ggml_reshape_2d(g.ctx, logits, 1, logits->ne[0]);
    ggml_tensor * top_values = ggml_get_rows(g.ctx, rows, top_ids);
    top_values = ggml_reshape_1d(g.ctx, top_values, top_k);
    top_values = ggml_scale(
        g.ctx,
        top_values,
        1.0f / std::max(temperature, 1.0e-5f)
    );

    ggml_tensor * softmax_input = top_values;
    if (sample_width > top_k) {
        softmax_input = ggml_concat(g.ctx, top_values, pad, 0);
    }
    softmax_input = ggml_reshape_2d(g.ctx, softmax_input, sample_width, 1);

    // Inverse-CDF categorical sampling with one host RNG draw per codebook.
    // sample_width is padded to an HTP-friendly multiple of 32 when needed;
    // padding logits are -1e30 so their probability is effectively zero.
    ggml_tensor * probs = ggml_soft_max(g.ctx, softmax_input);
    ggml_tensor * cdf = ggml_cumsum(g.ctx, probs);
    ggml_tensor * u = ggml_repeat(g.ctx, uniform, cdf);
    ggml_tensor * before = ggml_step(g.ctx, ggml_sub(g.ctx, u, cdf));
    ggml_tensor * pos_f = ggml_sum(g.ctx, before);
    pos_f = ggml_clamp(g.ctx, pos_f, 0.0f, (float) (top_k - 1));
    ggml_tensor * pos_i = ggml_cast(g.ctx, pos_f, GGML_TYPE_I32);

    // top_ids is I32. Cast its exact small integer values to F32 only so
    // GET_ROWS can select the chosen id, then cast that scalar back to I32.
    ggml_tensor * ids_f = ggml_cast(g.ctx, top_ids, GGML_TYPE_F32);
    ids_f = ggml_reshape_2d(g.ctx, ids_f, 1, top_k);
    ggml_tensor * selected_f = ggml_get_rows(g.ctx, ids_f, pos_i);
    return ggml_cast(g.ctx, selected_f, GGML_TYPE_I32);
}

static std::unique_ptr<DepthFrameFast> build_depth_frame_fast(
    BreezeModel & m,
    DepthRunner & r
) {
    const DepthConfig & c = m.cfg.dd;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const int top_k = std::min(std::max(1, m.cfg.depth_top_k), vs);
    const int sample_width = top_k <= 32 ? top_k : ((top_k + 31) / 32) * 32;

    auto fast = std::make_unique<DepthFrameFast>();
    fast->top_k = top_k;
    fast->sample_width = sample_width;
    Graph & g = fast->graph;

    fast->cb0 = g.input_i32(std::vector<int32_t>(1, 0), 1);
    fast->hidden = g.input_f32(
        std::vector<float>((size_t) m.cfg.hidden_size, 0.0f),
        m.cfg.hidden_size,
        1
    );
    fast->uniforms = g.input_f32(
        std::vector<float>((size_t) (nc - 1), 0.5f),
        nc - 1
    );
    ggml_tensor * ff = g.input_f32(
        r.freq_factors,
        (int) r.freq_factors.size()
    );

    ggml_tensor * pad = nullptr;
    if (sample_width > top_k) {
        pad = g.input_f32(
            std::vector<float>((size_t) (sample_width - top_k), -1.0e30f),
            sample_width - top_k
        );
    }

    std::vector<ggml_tensor *> keys((size_t) c.n_layer, nullptr);
    std::vector<ggml_tensor *> values((size_t) c.n_layer, nullptr);
    std::vector<ggml_tensor *> sampled;
    sampled.reserve((size_t) nc - 1);

    ggml_tensor * current_token = fast->cb0;
    for (int head_idx = 0; head_idx < nc - 1; ++head_idx) {
        const int n = head_idx == 0 ? 2 : 1;
        const int start = head_idx == 0 ? 0 : head_idx + 1;

        ggml_tensor * book = depth_audio_book_view(g.ctx, m, head_idx);
        ggml_tensor * embed = ggml_get_rows(g.ctx, book, current_token);
        ggml_tensor * x = embed;
        if (head_idx == 0) {
            x = ggml_concat(g.ctx, fast->hidden, embed, 1);
        }
        x = linear(g.ctx, m.w("dd.in_proj.weight"), x);

        std::vector<int32_t> pos_i((size_t) n);
        for (int i = 0; i < n; ++i) pos_i[(size_t) i] = start + i;
        ggml_tensor * pos = g.input_i32(pos_i, n);
        const int total = start + n;
        std::vector<float> mask_v = build_branch_causal_mask(n, total, start, 1);
        ggml_tensor * mask = g.input_f32(mask_v, total, n);

        for (int il = 0; il < c.n_layer; ++il) {
            x = depth_layer_chained(
                g.ctx, m, g, x, il, pos, ff, mask, n, keys, values
            );
        }
        x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);
        ggml_tensor * last = ggml_cont(
            g.ctx,
            ggml_view_2d(
                g.ctx,
                x,
                c.hidden,
                1,
                x->nb[1],
                (size_t) (n - 1) * x->nb[1]
            )
        );

        ggml_tensor * head = m.w("dd.codebooks_head.weight");
        ggml_tensor * hw = ggml_view_2d(
            g.ctx,
            head,
            head->ne[0],
            head->ne[1],
            head->nb[1],
            (size_t) head_idx * head->nb[2]
        );
        ggml_tensor * logits = ggml_mul_mat(g.ctx, hw, last);
        ggml_tensor * uniform = ggml_view_1d(
            g.ctx,
            fast->uniforms,
            1,
            (size_t) head_idx * sizeof(float)
        );
        current_token = depth_sample_in_graph(
            g,
            logits,
            uniform,
            top_k,
            sample_width,
            m.cfg.depth_temperature,
            pad
        );
        sampled.push_back(current_token);
    }

    ggml_tensor * output = sampled.front();
    for (size_t i = 1; i < sampled.size(); ++i) {
        output = ggml_concat(g.ctx, output, sampled[i], 0);
    }
    fast->output = ggml_cont(g.ctx, output);
    g.prepare(m.backend, fast->output);

    std::fprintf(
        stderr,
        "[BREEZE_DEPTH_FAST] prepared whole-frame graph steps=%d top_k=%d sample_width=%d branches=1\n",
        nc - 1,
        top_k,
        sample_width
    );
    return fast;
}

static bool run_depth_frame_fast(
    BreezeModel & m,
    DepthRunner & r,
    const std::vector<std::vector<float>> & hiddens,
    int cb0,
    std::mt19937 & rng,
    std::vector<int> & out
) {
    if (!r.frame_fast) {
        r.frame_fast = build_depth_frame_fast(m, r);
    }
    DepthFrameFast & fast = *r.frame_fast;
    const int steps = m.cfg.num_codebooks - 1;

    const int32_t first = (int32_t) cb0;
    ggml_backend_tensor_set(fast.cb0, &first, 0, sizeof(first));
    ggml_backend_tensor_set(
        fast.hidden,
        hiddens[0].data(),
        0,
        (size_t) m.cfg.hidden_size * sizeof(float)
    );

    std::uniform_real_distribution<float> uniform(
        std::nextafter(0.0f, 1.0f),
        std::nextafter(1.0f, 0.0f)
    );
    std::vector<float> draws((size_t) steps);
    for (float & v : draws) v = uniform(rng);
    ggml_backend_tensor_set(
        fast.uniforms,
        draws.data(),
        0,
        draws.size() * sizeof(float)
    );

    fast.graph.replay(m.backend);
    std::vector<int32_t> raw((size_t) steps);
    ggml_backend_tensor_get(
        fast.output,
        raw.data(),
        0,
        raw.size() * sizeof(int32_t)
    );

    out.resize((size_t) steps);
    for (int i = 0; i < steps; ++i) {
        const int v = (int) raw[(size_t) i];
        if (v < 0 || v >= m.cfg.audio_vocab_size) {
            throw std::runtime_error("whole-frame depth graph produced invalid token");
        }
        out[(size_t) i] = v;
    }
    return true;
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    // Normal CFG=1 TTS uses the exact same depth model but keeps all 15
    // autoregressive residual-codebook steps inside one HTP graph. This removes
    // 14 AP<->HTP synchronization boundaries per audio frame. The legacy cached
    // per-step path remains the fallback for CFG, forced codes, custom depth
    // sampling, or any device/runtime that rejects the fused graph.
    const bool whole_frame_eligible =
        n_branch == 1 &&
        sp_in == nullptr &&
        force == nullptr &&
        n_force == 0 &&
        m.cfg.depth_top_k > 0 &&
        m.cfg.depth_top_p >= 0.9999f;
    // IMPORTANT: the monolithic graph crashed on SM8850; never use it in a
    // distributable build. Only a separate explicitly controlled lab run may
    // opt in after verifying HTP numerical equivalence on the actual device.
    const char * fusion_trial = std::getenv("BREEZE_DEPTH_FUSION_TRIAL");
    const bool fusion_opt_in =
        fusion_trial && fusion_trial[0] == '1' && fusion_trial[1] == '\0';
    if (fusion_opt_in && whole_frame_eligible && !frame_fast_disabled) {
        try {
            std::vector<int> fast_codes;
            if (run_depth_frame_fast(m, *this, hiddens, cb0, rng, fast_codes)) {
                return fast_codes;
            }
        } catch (const std::exception & e) {
            frame_fast.reset();
            frame_fast_disabled = true;
            std::fprintf(
                stderr,
                "[BREEZE_DEPTH_FAST] disabled after runtime failure: %s; falling back to cached per-step graph\n",
                e.what()
            );
        }
    }

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
