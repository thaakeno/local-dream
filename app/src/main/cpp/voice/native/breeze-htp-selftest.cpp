#include "breeze/common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace breeze;

static void require_close(
    const char * name,
    const std::vector<float> & got,
    const std::vector<float> & expected,
    float atol = 2e-3f
) {
    if (got.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + ": size mismatch");
    }
    float worst = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float err = std::fabs(got[i] - expected[i]);
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    if (!(worst <= atol)) {
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "%s mismatch at %zu: got %.7f expected %.7f err %.7f",
            name,
            worst_i,
            got[worst_i],
            expected[worst_i],
            worst
        );
        throw std::runtime_error(buf);
    }
    std::fprintf(stderr, "[BREEZE_SELFTEST] %s ok worst=%.7f\n", name, worst);
}

static void test_snake(Backend & be) {
    constexpr int T = 32;
    constexpr int C = 4;
    std::vector<float> x(T * C);
    std::vector<float> alpha(C);
    std::vector<float> inv_beta(C);
    for (int c = 0; c < C; ++c) {
        alpha[c] = 0.55f + 0.12f * c;
        inv_beta[c] = 0.7f + 0.09f * c;
        for (int t = 0; t < T; ++t) {
            x[t + T * c] = std::sin(0.17f * t + 0.31f * c) * 1.4f;
        }
    }

    Graph g(128);
    auto * tx = g.input_f32(x, T, C);
    auto * ta = g.input_f32(alpha, 1, C);
    auto * tb = g.input_f32(inv_beta, 1, C);
    auto * s = ggml_sin(g.ctx, ggml_mul(g.ctx, tx, ta));
    auto * out = ggml_add(g.ctx, tx, ggml_mul(g.ctx, ggml_sqr(g.ctx, s), tb));
    g.compute(be, out);

    std::vector<float> expected(x.size());
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            const size_t i = (size_t) t + (size_t) T * c;
            const float s0 = std::sin(x[i] * alpha[c]);
            expected[i] = x[i] + s0 * s0 * inv_beta[c];
        }
    }
    require_close("snake-channel-broadcast", tensor_to_f32(out), expected, 3e-3f);
}

static void test_col2im_bias_case(
    Backend & be,
    const char * name,
    int K,
    int OC,
    int TIN,
    int STRIDE,
    int PAD
) {
    const int TOUT = (TIN - 1) * STRIDE + K - 2 * PAD;

    std::vector<float> cols((size_t) K * OC * TIN);
    for (int t = 0; t < TIN; ++t) {
        for (int k = 0; k < K; ++k) {
            for (int ch = 0; ch < OC; ++ch) {
                // Exact ggml [K * OC, T] layout: flattened index = k * OC + ch.
                cols[(size_t) (k * OC + ch) + (size_t) (K * OC) * t] =
                    0.013f * (1 + t * 101 + k * 11 + ch * 3);
            }
        }
    }

    std::vector<float> bias(OC);
    for (int ch = 0; ch < OC; ++ch) {
        bias[ch] = -0.31f + 0.19f * ch;
    }

    Graph g(192);
    auto * tc = g.input_f32(cols, K * OC, TIN);
    auto * tb = g.input_f32(bias, 1, OC);
    auto * col = ggml_col2im_1d(g.ctx, tc, STRIDE, OC, PAD);
    auto * out = ggml_add(g.ctx, col, tb);
    g.compute(be, out);

    std::vector<float> expected((size_t) TOUT * OC, 0.0f);
    for (int t = 0; t < TIN; ++t) {
        for (int k = 0; k < K; ++k) {
            const int dst_t = t * STRIDE + k - PAD;
            if (dst_t < 0 || dst_t >= TOUT) continue;
            for (int ch = 0; ch < OC; ++ch) {
                expected[(size_t) dst_t + (size_t) TOUT * ch] +=
                    cols[(size_t) (k * OC + ch) + (size_t) (K * OC) * t];
            }
        }
    }
    for (int ch = 0; ch < OC; ++ch) {
        for (int t = 0; t < TOUT; ++t) {
            expected[(size_t) t + (size_t) TOUT * ch] += bias[ch];
        }
    }

    require_close(name, tensor_to_f32(out), expected, 4e-3f);
}

static void test_col2im_bias(Backend & be) {
    // Generic padded case catches indexing/range bugs.
    test_col2im_bias_case(be, "col2im1d-bias-generic", 3, 3, 5, 2, 1);

    // Real Breeze/Oobleck transpose-conv family: kernel = 2 * stride.
    // Cover every stride used by the decoder and multiple channel counts.
    test_col2im_bias_case(be, "col2im1d-bias-s2",  4, 5, 5, 2, 0);
    test_col2im_bias_case(be, "col2im1d-bias-s4",  8, 4, 4, 4, 0);
    test_col2im_bias_case(be, "col2im1d-bias-s5", 10, 3, 4, 5, 0);
    test_col2im_bias_case(be, "col2im1d-bias-s6", 12, 3, 4, 6, 0);
}

static void test_sin(Backend & be) {
    std::vector<float> x(97);
    for (size_t i = 0; i < x.size(); ++i) x[i] = -2.0f + 4.0f * (float) i / (float) (x.size() - 1);
    Graph g(64);
    auto * tx = g.input_f32(x, (int) x.size());
    auto * out = ggml_sin(g.ctx, tx);
    g.compute(be, out);
    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) expected[i] = std::sin(x[i]);
    require_close("sin-f32", tensor_to_f32(out), expected, 3e-3f);
}

int main() {
    try {
        Backend be;
        be.init(true);
        std::fprintf(stderr, "[BREEZE_SELFTEST] backend=%s\n", be.name());
        test_sin(be);
        test_snake(be);
        test_col2im_bias(be);
        be.free();
        std::fprintf(stderr, "[BREEZE_SELFTEST] all-ok\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[BREEZE_SELFTEST] failed: %s\n", e.what());
        return 2;
    }
}
