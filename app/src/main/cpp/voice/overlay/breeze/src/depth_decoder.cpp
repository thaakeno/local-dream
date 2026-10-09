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
    verification.resize((size_t) nc * 4);
    speculation_unavailable = false;
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
    verification.clear();
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



// Optional lossless-distribution RVQ speculation. Both CFG branches still run;
// only the number of SERIAL transformer evaluations is reduced when proposals
// are accepted. Use the official position-specific heads as cheap draft heads.
// Never skip an autoregressive target probability: all proposed tokens are
// verified against the original model before being committed.
static std::unique_ptr<DepthStep> build_rvq_verifier(
    BreezeModel & m, DepthRunner & r, int first_head, int width
) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const bool first = first_head == 0;
    if (first && width != 1) throw std::runtime_error("RVQ first verifier shape");
    const int start = first ? 0 : first_head + 1;
    const int positions = first ? 2 : width;
    const int tokens = positions * nb;
    const int total = (start + positions) * nb;
    const int last_head = first_head + width - 1;
    auto step = std::make_unique<DepthStep>();
    Graph & g = step->graph;
    step->audio = g.input_i32(std::vector<int32_t>((size_t) width * nb), width * nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), step->audio);
    if (first) {
        step->hidden = g.input_f32(
            std::vector<float>((size_t) nb * m.cfg.hidden_size), m.cfg.hidden_size, nb);
        embed = ggml_concat(g.ctx, step->hidden, embed, 1);
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed);
    std::vector<int32_t> positions_tensor((size_t) tokens);
    for (int i = 0; i < tokens; ++i) positions_tensor[(size_t) i] = start + i / nb;
    ggml_tensor * pos = g.input_i32(positions_tensor, tokens);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    ggml_tensor * mask = g.input_f32(
        build_branch_causal_mask(tokens, total, start, nb), total, tokens);
    for (int layer = 0; layer < c.n_layer; ++layer)
        x = dd_layer(g.ctx, m, g, r.kv, x, layer, pos, ff, mask, start * nb, tokens);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);
    ggml_tensor * all_heads = m.w("dd.codebooks_head.weight");
    auto project = [&](int row, int head_idx) {
        ggml_tensor * hidden = ggml_cont(g.ctx, ggml_view_2d(
            g.ctx, x, c.hidden, nb, x->nb[1], (size_t) row * nb * x->nb[1]));
        ggml_tensor * head = ggml_view_2d(
            g.ctx, all_heads, all_heads->ne[0], all_heads->ne[1], all_heads->nb[1],
            (size_t) head_idx * all_heads->nb[2]);
        return ggml_mul_mat(g.ctx, head, hidden);
    };
    ggml_tensor * result = nullptr;
    for (int i = 0; i < width; ++i) {
        ggml_tensor * target = project(first ? 1 : i, first_head + i);
        result = result ? ggml_concat(g.ctx, result, target, 0) : target;
    }
    // Only draft up to two more RVQ heads from the *same* hidden state.
    // If the target distribution rejects a proposal, return to serial decode
    // for the rest of this frame; this guarantees correct KV-cache ownership.
    const int future = std::min(2, m.cfg.num_codebooks - 2 - last_head);
    for (int i = 1; i <= future; ++i)
        result = ggml_concat(g.ctx, result, project(positions - 1, last_head + i), 0);
    step->logits = result;
    g.prepare(m.backend, step->logits);
    return step;
}

struct RvqProbabilities {
    std::vector<float> values;
    std::vector<int> tokens;
};

// Match Breeze's temperature, partial-sort top-k and renormalized top-p
// semantics. Acceptance below is exact up to floating-point rounding.
static RvqProbabilities rvq_probs(const std::vector<float> & logits, const SampleParams & sp) {
    const int count = (int) logits.size();
    if (!count) throw std::runtime_error("RVQ no logits");
    const int keep = sp.top_k > 0 && sp.top_k < count ? sp.top_k : count;
    RvqProbabilities d;
    d.tokens.resize((size_t) count);
    for (int i = 0; i < count; ++i) d.tokens[(size_t) i] = i;
    std::partial_sort(d.tokens.begin(), d.tokens.begin() + keep, d.tokens.end(),
        [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    d.tokens.resize((size_t) keep);
    const float highest = logits[(size_t) d.tokens[0]];
    const float temperature = sp.temperature > 0 ? sp.temperature : 1.0f;
    d.values.resize((size_t) keep);
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) {
        const float v = std::exp((logits[(size_t) d.tokens[(size_t) i]] - highest) / temperature);
        if (!std::isfinite(v)) throw std::runtime_error("RVQ nonfinite probability");
        d.values[(size_t) i] = v;
        sum += v;
    }
    if (!(sum > 0) || !std::isfinite(sum)) throw std::runtime_error("RVQ invalid normalization");
    for (float & v : d.values) v /= sum;
    if (sp.top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t cut = d.values.size();
        for (size_t i = 0; i < d.values.size(); ++i) {
            cumulative += d.values[i];
            if (cumulative >= sp.top_p) { cut = i + 1; break; }
        }
        d.values.resize(cut);
        d.tokens.resize(cut);
        float remainder = 0.0f;
        for (float v : d.values) remainder += v;
        if (!(remainder > 0)) throw std::runtime_error("RVQ empty nucleus");
        for (float & v : d.values) v /= remainder;
    }
    return d;
}

static float rvq_prob(const RvqProbabilities & d, int token) {
    for (size_t i = 0; i < d.tokens.size(); ++i)
        if (d.tokens[i] == token) return d.values[i];
    return 0.0f;
}

static int rvq_sample(const RvqProbabilities & d, std::mt19937 & rng) {
    std::discrete_distribution<int> pick(d.values.begin(), d.values.end());
    return d.tokens[(size_t) pick(rng)];
}

static int rvq_rejection_sample(
    const RvqProbabilities & target, const RvqProbabilities & draft, std::mt19937 & rng
) {
    RvqProbabilities residual = target;
    float mass = 0;
    for (size_t i = 0; i < residual.tokens.size(); ++i) {
        const float weight = std::max(
            0.0f, residual.values[i] - rvq_prob(draft, residual.tokens[i]));
        residual.values[i] = weight;
        mass += weight;
    }
    if (!std::isfinite(mass) || mass <= 0) return rvq_sample(target, rng);
    for (float & v : residual.values) v /= mass;
    return rvq_sample(residual, rng);
}

std::vector<int> DepthRunner::run_speculative(
    BreezeModel & m, const std::vector<std::vector<float>> & hiddens, int cb0,
    float cfg_scale, std::mt19937 & rng, const SampleParams & sp
) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const int target_count = nc - 1;
    if (target_count < 3 || verification.size() < (size_t) nc * 4)
        throw std::runtime_error("RVQ verifier unavailable");
    kv.reset();
    std::vector<int> codes;
    codes.reserve((size_t) nc);
    codes.push_back(cb0);
    std::vector<RvqProbabilities> future_distributions;
    std::vector<int> future_proposals;
    int next_head = 0;
    int rounds = 0, proposals = 0, accepted = 0;
    bool serial_remainder = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (next_head < target_count) {
        const int remaining = target_count - next_head;
        const int width = next_head == 0 || serial_remainder ? 1 :
            std::min(remaining, 1 + (int) future_proposals.size());
        const int slot = next_head * 4 + width;
        if (!verification[(size_t) slot])
            verification[(size_t) slot] = build_rvq_verifier(m, *this, next_head, width);
        DepthStep & step = *verification[(size_t) slot];
        std::vector<int32_t> inputs((size_t) width * n_branch);
        for (int i = 0; i < width; ++i) {
            // target head i predicts codebook i+1 from the PREVIOUS codebook.
            const int previous = i == 0 ? codes[(size_t) next_head] :
                future_proposals[(size_t) i - 1];
            for (int branch = 0; branch < n_branch; ++branch)
                inputs[(size_t) i * n_branch + branch] = previous + (next_head + i) * vs;
        }
        ggml_backend_tensor_set(step.audio, inputs.data(), 0, inputs.size() * sizeof(int32_t));
        if (next_head == 0) {
            for (int branch = 0; branch < n_branch; ++branch)
                ggml_backend_tensor_set(
                    step.hidden, hiddens[(size_t) branch].data(),
                    (size_t) branch * m.cfg.hidden_size * sizeof(float),
                    (size_t) m.cfg.hidden_size * sizeof(float));
        }
        step.graph.replay(m.backend);
        const std::vector<float> raw = tensor_to_f32(step.logits);
        const int future_count = std::min(2, target_count - next_head - width);
        const int heads = width + future_count;
        if (raw.size() != (size_t) n_branch * heads * vs)
            throw std::runtime_error("RVQ verifier logits mismatch");
        auto guided = [&](int offset) {
            std::vector<float> logits((size_t) vs);
            const size_t head_start = (size_t) offset * vs;
            const size_t branch_size = (size_t) heads * vs;
            for (int k = 0; k < vs; ++k) {
                const float c = raw[head_start + (size_t) k];
                const float u = n_branch == 2 ? raw[branch_size + head_start + (size_t) k] : c;
                logits[(size_t) k] = u + cfg_scale * (c - u);
            }
            return logits;
        };
        ++rounds;
        bool rejected = false;
        for (int i = 0; i < width; ++i) {
            RvqProbabilities target = rvq_probs(guided(i), sp);
            if (i + 1 < width) {
                const int proposed = future_proposals[(size_t) i];
                const RvqProbabilities & draft = future_distributions[(size_t) i];
                const float denominator = rvq_prob(draft, proposed);
                const float numerator = rvq_prob(target, proposed);
                const float probability = denominator > 0.0f ?
                    std::min(1.0f, numerator / denominator) : 0.0f;
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                ++proposals;
                if (uniform(rng) <= probability) {
                    ++accepted;
                    codes.push_back(proposed);
                } else {
                    codes.push_back(rvq_rejection_sample(target, draft, rng));
                    rejected = true;
                    serial_remainder = true;
                    break;
                }
            } else {
                codes.push_back(rvq_sample(target, rng));
            }
        }
        next_head = (int) codes.size() - 1;
        future_distributions.clear();
        future_proposals.clear();
        if (!rejected && !serial_remainder && next_head < target_count) {
            for (int i = 0; i < future_count; ++i) {
                future_distributions.push_back(rvq_probs(guided(width + i), sp));
                future_proposals.push_back(rvq_sample(future_distributions.back(), rng));
            }
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    static int frame_index = 0;
    if (++frame_index == 1 || frame_index % 32 == 0)
        std::fprintf(stderr,
            "[BREEZE_RVQ_SPEC] frames=%d cfg=%.2f heads=%d rounds=%d accepted=%d proposed=%d total_ms=%.2f serial_tail=%d\n",
            frame_index, cfg_scale, target_count, rounds, accepted, proposals,
            ms, serial_remainder ? 1 : 0);
    return std::vector<int>(codes.begin() + 1, codes.end());
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

    const char * opt = std::getenv("BREEZE_RVQ_SPECULATIVE");
    if (opt && opt[0] == '1' && n_force == 0 && !force && !speculation_unavailable) {
        const std::mt19937 previous_rng = rng;
        try {
            return run_speculative(m, hiddens, cb0, cfg_scale, rng, sp);
        } catch (const std::exception & e) {
            speculation_unavailable = true;
            rng = previous_rng;
            kv.reset();
            std::fprintf(stderr, "[BREEZE_RVQ_SPEC] fallback=stable reason=%s\n", e.what());
        }
    }

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
