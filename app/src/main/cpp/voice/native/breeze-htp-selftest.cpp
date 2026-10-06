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
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
            char buf[256];
            std::snprintf(
                buf,
                sizeof(buf),
                "%s non-finite at %zu: got %.7g expected %.7g",
                name,
                i,
                got[i],
                expected[i]
            );
            throw std::runtime_error(buf);
        }
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

static void test_get_rows_f32(Backend & be) {
    constexpr int D = 256;
    constexpr int ROWS = 64;

    std::vector<float> table((size_t) D * ROWS);
    for (int r = 0; r < ROWS; ++r) {
        for (int d = 0; d < D; ++d) {
            table[(size_t) d + (size_t) D * r] =
                -3.0f + 0.25f * (float) r + 0.001f * (float) d;
        }
    }

    auto run = [&](const char * name, const std::vector<int32_t> & ids) {
        Graph g(96);
        auto * values = g.input_f32(table, D, ROWS);
        auto * indices = g.input_i32(ids, (int) ids.size());
        auto * out = ggml_get_rows(g.ctx, values, indices);
        g.compute(be, out);

        std::vector<float> expected((size_t) D * ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const int r = ids[i];
            for (int d = 0; d < D; ++d) {
                expected[(size_t) d + (size_t) D * i] =
                    table[(size_t) d + (size_t) D * r];
            }
        }
        require_close(name, tensor_to_f32(out), expected, 1e-6f);
    };

    run("get-rows-f32-single", { 37 });
    run("get-rows-f32-multi", { 0, 7, 31, 63 });
}

static void test_get_rows_f32_ordinary_buffer(Backend & be) {
    constexpr int D = 256;
    constexpr int ROWS = 2048;
    const std::vector<int32_t> row_ids = { 0, 31, 219, 1221, 1938, 2047 };

    ggml_init_params params{
        ggml_tensor_overhead() * 8 + 4096,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ordinary GET_ROWS context allocation failed");

    ggml_backend_buffer_t buffer = nullptr;
    try {
        ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, ROWS);
        ggml_set_name(table, "selftest.decoder_codebook.ordinary");

        buffer = ggml_backend_alloc_ctx_tensors(ctx, be.backend);
        if (!buffer) {
            throw std::runtime_error("ordinary GET_ROWS HTP buffer allocation failed");
        }
        std::vector<float> values((size_t) D * ROWS);
        for (int r = 0; r < ROWS; ++r) {
            for (int d = 0; d < D; ++d) {
                values[(size_t) d + (size_t) D * r] =
                    -1.75f + 0.0025f * (float) r + 0.0005f * (float) d;
            }
        }
        ggml_backend_tensor_set(table, values.data(), 0, values.size() * sizeof(float));

        std::vector<float> expected((size_t) D * row_ids.size());
        for (size_t r = 0; r < row_ids.size(); ++r) {
            for (int d = 0; d < D; ++d) {
                expected[(size_t) D * r + (size_t) d] =
                    values[(size_t) D * (size_t) row_ids[r] + (size_t) d];
            }
        }

        Graph g(128);
        auto * ids = g.input_i32(row_ids, (int) row_ids.size());
        auto * out = ggml_get_rows(g.ctx, table, ids);
        g.compute(be, out);
        require_close(
            "get-rows-f32-ordinary-2048-highrows",
            tensor_to_f32(out),
            expected,
            1e-6f
        );

        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ctx);
    } catch (...) {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
}

static void test_snake(Backend & be) {
    constexpr int T = 32;
    constexpr int C = 4;
    std::vector<float> x(T * C);
    std::vector<float> log_alpha = { -0.60f, -0.10f, 0.35f, 0.80f };

    for (int ch = 0; ch < C; ++ch) {
        for (int t = 0; t < T; ++t) {
            x[t + T * ch] = std::sin(0.17f * t + 0.31f * ch) * 1.4f;
        }
    }

    // First validate normal Breeze SnakeBeta numerics against a scalar
    // reference with realistic magnitudes.
    {
        std::vector<float> log_beta = { -2.0f, -0.5f, 0.0f, 4.0f };
        std::vector<float> ones(C, 1.0f);
        std::vector<float> eps(C, 1.0e-9f);

        Graph g(160);
        auto * tx = g.input_f32(x, T, C);
        auto * tla = g.input_f32(log_alpha, 1, C);
        auto * tlb = g.input_f32(log_beta, 1, C);
        auto * one = g.input_f32(ones, 1, C);
        auto * tiny = g.input_f32(eps, 1, C);

        auto * alpha = ggml_exp(g.ctx, tla);
        auto * beta = ggml_exp(g.ctx, tlb);
        auto * inv_beta = ggml_div(g.ctx, one, ggml_add(g.ctx, beta, tiny));
        auto * s = ggml_sin(g.ctx, ggml_mul(g.ctx, tx, alpha));
        auto * out = ggml_add(
            g.ctx, tx, ggml_mul(g.ctx, ggml_sqr(g.ctx, s), inv_beta)
        );
        g.compute(be, out);

        std::vector<float> expected(x.size());
        for (int ch = 0; ch < C; ++ch) {
            const float a = std::exp(log_alpha[ch]);
            const float b = std::exp(log_beta[ch]);
            const float inv_b = 1.0f / (b + 1.0e-9f);
            for (int t = 0; t < T; ++t) {
                const size_t i = (size_t) t + (size_t) T * ch;
                const float s0 = std::sin(x[i] * a);
                expected[i] = x[i] + s0 * s0 * inv_b;
            }
        }
        require_close("snake-beta-reference", tensor_to_f32(out), expected, 4e-3f);
    }

    // Then explicitly cover the v156 overflow scenario. This test only asks
    // for finiteness because values near the epsilon floor can be ~1e9 and an
    // absolute close check would be meaningless.
    {
        std::vector<float> log_beta = { -100.0f, -20.0f, 0.0f, 4.0f };
        std::vector<float> ones(C, 1.0f);
        std::vector<float> eps(C, 1.0e-9f);

        Graph g(160);
        auto * tx = g.input_f32(x, T, C);
        auto * tla = g.input_f32(log_alpha, 1, C);
        auto * tlb = g.input_f32(log_beta, 1, C);
        auto * one = g.input_f32(ones, 1, C);
        auto * tiny = g.input_f32(eps, 1, C);

        auto * alpha = ggml_exp(g.ctx, tla);
        auto * beta = ggml_exp(g.ctx, tlb);
        auto * inv_beta = ggml_div(g.ctx, one, ggml_add(g.ctx, beta, tiny));
        auto * s = ggml_sin(g.ctx, ggml_mul(g.ctx, tx, alpha));
        auto * out = ggml_add(
            g.ctx, tx, ggml_mul(g.ctx, ggml_sqr(g.ctx, s), inv_beta)
        );
        g.compute(be, out);

        const std::vector<float> got = tensor_to_f32(out);
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) {
                throw std::runtime_error(
                    "snake-beta-extreme produced non-finite output at " + std::to_string(i)
                );
            }
        }
        std::fprintf(stderr, "[BREEZE_SELFTEST] snake-beta-extreme finite\n");
    }
}

static void test_col2im_case(
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
                // Exact ggml [K * OC, T] layout: flattened index = ch * K + k.
                cols[(size_t) (ch * K + k) + (size_t) (K * OC) * t] =
                    0.013f * (1 + t * 101 + k * 11 + ch * 3);
            }
        }
    }

    std::vector<float> expected((size_t) TOUT * OC, 0.0f);
    for (int t = 0; t < TIN; ++t) {
        for (int k = 0; k < K; ++k) {
            const int dst_t = t * STRIDE + k - PAD;
            if (dst_t < 0 || dst_t >= TOUT) continue;
            for (int ch = 0; ch < OC; ++ch) {
                expected[(size_t) dst_t + (size_t) TOUT * ch] +=
                    cols[(size_t) (ch * K + k) + (size_t) (K * OC) * t];
            }
        }
    }

    // Raw COL2IM is the production Breeze path before its causal crop.
    {
        Graph g(192);
        auto * tc = g.input_f32(cols, K * OC, TIN);
        auto * out = ggml_col2im_1d(g.ctx, tc, STRIDE, OC, PAD);
        g.compute(be, out);
        require_close((std::string(name) + "-raw").c_str(), tensor_to_f32(out), expected, 4e-3f);
    }

    // Also validate the fused COL2IM+bias opcode independently.
    std::vector<float> bias(OC);
    for (int ch = 0; ch < OC; ++ch) {
        bias[ch] = -0.31f + 0.19f * ch;
    }
    std::vector<float> expected_bias = expected;
    for (int ch = 0; ch < OC; ++ch) {
        for (int t = 0; t < TOUT; ++t) {
            expected_bias[(size_t) t + (size_t) TOUT * ch] += bias[ch];
        }
    }

    Graph g(192);
    auto * tc = g.input_f32(cols, K * OC, TIN);
    auto * tb = g.input_f32(bias, 1, OC);
    auto * col = ggml_col2im_1d(g.ctx, tc, STRIDE, OC, PAD);
    auto * out = ggml_add(g.ctx, col, tb);
    g.compute(be, out);
    require_close((std::string(name) + "-bias").c_str(), tensor_to_f32(out), expected_bias, 4e-3f);
}

static void test_col2im_bias(Backend & be) {
    // Generic padded case catches indexing/range bugs.
    test_col2im_case(be, "col2im1d-bias-generic", 3, 3, 5, 2, 1);

    // Real Breeze/Oobleck transpose-conv family: kernel = 2 * stride.
    // Cover every stride used by the decoder and multiple channel counts.
    test_col2im_case(be, "col2im1d-bias-s2",  4, 5, 5, 2, 0);
    test_col2im_case(be, "col2im1d-bias-s3",  6, 7, 5, 3, 0);
    test_col2im_case(be, "col2im1d-bias-s4",  8, 8, 5, 4, 0);
    test_col2im_case(be, "col2im1d-bias-s5", 10, 9, 5, 5, 0);
    test_col2im_case(be, "col2im1d-bias-s8", 16, 16, 5, 8, 0);
    // Multi-thread stress case: more channels than the v81 HTP thread count.
    test_col2im_case(be, "col2im1d-bias-s8-c64", 16, 64, 8, 8, 0);
}

static void test_v81_hmx_visibility_chain(Backend & be) {
    // Evidence-backed SM8850/v81 regression test: HMX MUL_MAT output consumed
    // by a dependent ADD. This exact class of producer/consumer corruption is
    // reported publicly on v81 and directly exercises the FIFO packet-boundary
    // visibility workaround used by Breeze.
    constexpr int K = 256;
    constexpr int M = 64;
    constexpr int N = 32;

    std::vector<float> w((size_t) K * M);
    std::vector<float> x((size_t) K * N);
    std::vector<float> bias(M);
    for (int m = 0; m < M; ++m) {
        bias[m] = -0.04f + 0.001f * (float) m;
        for (int k = 0; k < K; ++k) {
            w[(size_t) k + (size_t) K * m] =
                0.018f * std::sin(0.013f * (float) (1 + k + 3 * m));
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            x[(size_t) k + (size_t) K * n] =
                0.021f * std::cos(0.017f * (float) (1 + 2 * k + n));
        }
    }

    std::vector<float> expected((size_t) M * N);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double acc = bias[m];
            for (int k = 0; k < K; ++k) {
                acc += (double) w[(size_t) k + (size_t) K * m] *
                       (double) x[(size_t) k + (size_t) K * n];
            }
            expected[(size_t) m + (size_t) M * n] = (float) acc;
        }
    }

    Graph g(192);
    auto * tw = g.input_f32(w, K, M);
    auto * tx = g.input_f32(x, K, N);
    auto * tb = g.input_f32(bias, M);
    auto * mm = ggml_mul_mat(g.ctx, tw, tx);
    auto * out = ggml_add(g.ctx, mm, tb);
    g.compute(be, out);
    require_close(
        "v81-hmx-mulmat-add-visibility",
        tensor_to_f32(out),
        expected,
        8e-3f
    );
}

static void test_v81_direct_residual_add(Backend & be) {
    // Mirrors the streamed ConvNeXt/residual geometry that previously reached
    // the generic chunked binary DMA/VTCM kernel and stalled the v81 DSP.
    constexpr int T = 8;
    constexpr int C = 512;

    std::vector<float> a((size_t) T * C);
    std::vector<float> b((size_t) T * C);
    std::vector<float> expected_add(a.size());

    for (int ch = 0; ch < C; ++ch) {
        for (int t = 0; t < T; ++t) {
            const size_t i = (size_t) t + (size_t) T * ch;
            a[i] = -0.75f + 0.013f * (float) t + 0.001f * (float) ch;
            b[i] =  0.25f - 0.007f * (float) t + 0.0005f * (float) ch;
            expected_add[i] = a[i] + b[i];
        }
    }

    {
        Graph g(96);
        auto * ta = g.input_f32(a, T, C);
        auto * tb = g.input_f32(b, T, C);
        auto * out = ggml_add(g.ctx, ta, tb);
        g.compute(be, out);
        require_close("v81-direct-residual-add", tensor_to_f32(out), expected_add, 1e-6f);
    }

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
        test_get_rows_f32(be);
        test_get_rows_f32_ordinary_buffer(be);
        test_snake(be);
        test_col2im_bias(be);
        test_v81_direct_residual_add(be);
        test_v81_hmx_visibility_chain(be);
        be.free();
        std::fprintf(stderr, "[BREEZE_SELFTEST] all-ok\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[BREEZE_SELFTEST] failed: %s\n", e.what());
        return 2;
    }
}
