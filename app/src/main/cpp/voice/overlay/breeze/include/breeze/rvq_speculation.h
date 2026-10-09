#pragma once

// Host-testable speculative-sampling math, independent of QNN and ggml.
// Keep this logic shared between Android's DepthRunner and the C++ unit test.
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace breeze {

struct RvqSampleConfig {
    float temperature;
    int top_k;
    float top_p;
};

struct RvqProbabilities {
    std::vector<float> values;
    std::vector<int> tokens;
};

// Match Breeze's temperature, partial-sort top-k and renormalized top-p
// semantics. Acceptance below is exact up to floating-point rounding.
inline RvqProbabilities rvq_probs(const std::vector<float> & logits, const RvqSampleConfig & sp) {
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

inline float rvq_prob(const RvqProbabilities & d, int token) {
    for (size_t i = 0; i < d.tokens.size(); ++i)
        if (d.tokens[i] == token) return d.values[i];
    return 0.0f;
}

inline int rvq_sample(const RvqProbabilities & d, std::mt19937 & rng) {
    std::discrete_distribution<int> pick(d.values.begin(), d.values.end());
    return d.tokens[(size_t) pick(rng)];
}

inline int rvq_rejection_sample(
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

 } // namespace breeze
