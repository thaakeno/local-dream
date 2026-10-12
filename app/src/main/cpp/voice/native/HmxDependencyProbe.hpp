#pragma once

#include "breeze/common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Device-only SM8850/v81 validation of the dependency sequence that previously
// produced garbage on HMX. Independent from Breeze's production decoder.
// The same executable is run once with GGML_HEXAGON_NHMX=0 (HVX) and once
// with GGML_HEXAGON_NHMX=1 (HMX). This only gates correctness; it does not
// enable an inference backend or modify CFG.
class HmxDependencyProbe final {
public:
    void run(breeze::Backend & backend) const {
        for (int rows : {1, 2, 6, 16}) {
            for (int variant = 0; variant < 4; ++variant) {
                run_case(backend, rows, variant);
            }
        }
        std::fprintf(stderr, "[BREEZE_HMX_CHAIN] complete 16 cases; "
            "HMX is confirmed only when GGML_HEXAGON_NHMX=1 "
            "and the backend reports actual HMX matmul coverage.\n");
    }

private:
    static constexpr int kInput = 256;
    static constexpr int kHidden = 128;
    static constexpr int kOutput = 64;

    static void check(const char * stage, const std::vector<float> & got,
                      const std::vector<float> & expected, int rows, int variant) {
        if (got.size() != expected.size()) {
            throw std::runtime_error(std::string(stage) + ": element count differs");
        }
        float worst = 0.0f;
        double squared = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
                throw std::runtime_error(std::string(stage) + ": non-finite output");
            }
            const float err = std::fabs(got[i] - expected[i]);
            worst = std::max(worst, err);
            squared += (double)err * (double)err;
        }
        const double rmse = std::sqrt(squared / (double)got.size());
        const float tolerance = 1.2e-2f; // Same existing v81 F32 DSP parity tolerance.
        if (worst > tolerance) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "v81-hmx-chain %s rows=%d variant=%d max_abs=%.7f rmse=%.7f",
                stage, rows, variant, worst, rmse);
            throw std::runtime_error(buf);
        }
        std::fprintf(stderr,
            "[BREEZE_HMX_CHAIN] stage=%s rows=%d variant=%d max_abs=%.7f rmse=%.7f\n",
            stage, rows, variant, worst, rmse);
    }

    static void run_case(breeze::Backend & backend, int rows, int variant) {
        std::vector<float> x((size_t)kInput * rows);
        std::vector<float> first((size_t)kInput * kHidden);
        std::vector<float> bias(kHidden);
        std::vector<float> last((size_t)kHidden * kOutput);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.75f * std::sin((float)i * 0.019f + (float)variant * 0.31f);
        }
        for (size_t i = 0; i < first.size(); ++i) {
            first[i] = 0.051f * std::cos((float)i * 0.013f + 0.017f * variant);
        }
        for (int i = 0; i < kHidden; ++i) {
            bias[i] = 0.02f * std::sin((float)i * 0.15f + 0.2f * variant);
        }
        for (size_t i = 0; i < last.size(); ++i) {
            last[i] = 0.071f * std::cos((float)i * 0.009f + (float)variant);
        }
        std::vector<float> expected_add((size_t)kHidden * rows);
        std::vector<float> expected_gelu(expected_add.size());
        std::vector<float> expected_output((size_t)kOutput * rows);
        for (int r = 0; r < rows; ++r) {
            for (int h = 0; h < kHidden; ++h) {
                double mm = 0.0;
                for (int k = 0; k < kInput; ++k) {
                    mm += (double)first[(size_t)h * kInput + k] *
                          (double)x[(size_t)r * kInput + k];
                }
                const size_t idx = (size_t)r * kHidden + h;
                const float add = (float)mm + bias[h];
                expected_add[idx] = add;
                expected_gelu[idx] = 0.5f * add *
                    (1.0f + std::erf(add * 0.7071067811865475f));
            }
            for (int o = 0; o < kOutput; ++o) {
                double sum = 0.0;
                for (int h = 0; h < kHidden; ++h) {
                    sum += (double)last[(size_t)o * kHidden + h] *
                           expected_gelu[(size_t)r * kHidden + h];
                }
                expected_output[(size_t)r * kOutput + o] = (float)sum;
            }
        }

        breeze::Graph graph(192);
        auto * w0 = graph.input_f32(first, kInput, kHidden);
        auto * v = graph.input_f32(x, kInput, rows);
        auto * b = graph.input_f32(bias, kHidden, 1);
        auto * w1 = graph.input_f32(last, kHidden, kOutput);
        auto * mat0 = ggml_mul_mat(graph.ctx, w0, v);
        // Alternate direct and strided producer/consumer boundaries. The
        // transposed copy cases exercise the DMA/CONT path independently.
        auto * producer = mat0;
        if ((variant & 1) != 0) {
            auto * first_copy = ggml_cont(graph.ctx, ggml_transpose(graph.ctx, mat0));
            producer = ggml_cont(graph.ctx, ggml_transpose(graph.ctx, first_copy));
        }
        auto * add = ggml_add(graph.ctx, producer, b);
        auto * gelu = ggml_gelu_erf(graph.ctx, add);
        auto * out = ggml_mul_mat(graph.ctx, w1, gelu);
        // Intermediate tensors must retain their own storage after execution.
        // Otherwise the graph allocator can reuse their backing memory and
        // turn a correct kernel into a false numerical mismatch on readback.
        ggml_set_output(add);
        ggml_set_output(gelu);
        graph.prepare(backend, out);
        graph.replay(backend); // exclude DSP mapping/warmup from the timing

        constexpr int timed_replays = 4;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < timed_replays; ++i) graph.replay(backend);
        const auto end = std::chrono::steady_clock::now();

        // Check every retained intermediate, not only the final matmul. This detects
        // stale data across HMX -> HVX -> HMX boundaries, not just bad GEMMs.
        check("ADD", breeze::tensor_to_f32(add), expected_add, rows, variant);
        check("GELU_ERF", breeze::tensor_to_f32(gelu), expected_gelu, rows, variant);
        check("FINAL_MUL_MAT", breeze::tensor_to_f32(out), expected_output, rows, variant);
        const double duration_ms =
            std::chrono::duration<double, std::milli>(end - begin).count() /
            timed_replays;
        std::fprintf(stderr,
            "[BREEZE_HMX_CHAIN_BENCH] rows=%d variant=%d replay_ms=%.3f samples=%d\n",
            rows, variant, duration_ms, timed_replays);
    }
};
