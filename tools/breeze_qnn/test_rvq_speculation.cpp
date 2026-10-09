// Standalone host-side exact speculative-sampling distribution tests.
#include "breeze/rvq_speculation.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

int main() {
    using namespace breeze;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> draws(-4.0f, 4.0f);
    const RvqSampleConfig full{0.9f, 0, 1.0f};
    const RvqSampleConfig clipped{0.7f, 11, 0.82f};
    for (auto cfg : {full, clipped}) {
        for (int trial = 0; trial < 1000; ++trial) {
            std::vector<float> logits_p(64), logits_q(64);
            for (float & x : logits_p) x = draws(rng);
            for (float & x : logits_q) x = draws(rng);
            const auto p = rvq_probs(logits_p, cfg);
            const auto q = rvq_probs(logits_q, cfg);
            float rejection_probability = 0.0f;
            for (size_t j = 0; j < q.tokens.size(); ++j) {
                const float qj = q.values[j];
                const float pj = rvq_prob(p, q.tokens[j]);
                rejection_probability += qj * (1.0f - std::min(1.0f, pj / qj));
            }
            float positive_mass = 0.0f;
            for (size_t j = 0; j < p.tokens.size(); ++j)
                positive_mass += std::max(0.0f, p.values[j] - rvq_prob(q, p.tokens[j]));
            assert(std::fabs(rejection_probability - positive_mass) < 0.00001f);
            for (size_t j = 0; j < p.tokens.size(); ++j) {
                const int token = p.tokens[j];
                const float target = p.values[j];
                const float proposal = rvq_prob(q, token);
                const float accept_contribution = std::min(proposal, target);
                const float residual = positive_mass > 0.0f ?
                    rejection_probability * std::max(0.0f, target - proposal) / positive_mass : 0.0f;
                assert(std::fabs(accept_contribution + residual - target) < 0.00001f);
            }
            const int sampled = rvq_sample(q, rng);
            assert(rvq_prob(q, sampled) > 0.0f);
            const int rejected = rvq_rejection_sample(p, q, rng);
            assert(rvq_prob(p, rejected) > 0.0f);
        }
    }
    std::cout << "RVQ acceptance, rejection and sampling: 2000 distribution tests passed" << std::endl;
    return 0;
}
