#pragma once

#include "breeze/depth_decoder.h"
#include "breeze/model.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Runs Breeze's actual 15-step DepthRunner with real GGUF weights and two
// CFG branches. Caller launches separate HVX and HMX processes and compares
// every F32 logit. No backend monkey-patching, no reduced CFG, no skipped
// codebooks, no production-path modifications.
class DepthParityProbe final : public breeze::DepthLogitsObserver {
public:
    void on_logits(int codebook, const std::vector<float> & logits) override {
        if (codebook != static_cast<int>(trace_.size()) + 1) {
            throw std::runtime_error("Depth diagnostic received codebooks out of order");
        }
        for (float value : logits) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Depth diagnostic produced non-finite logits");
            }
        }
        trace_.push_back(logits);
    }

    void run(const std::string & model_path, const std::string & output_file) {
        breeze::BreezeModel model;
        if (!model.load(model_path, true)) {
            throw std::runtime_error("Could not load Breeze GGUF for Depth diagnostic");
        }
        constexpr float cfg_scale = 3.2f;
        const int codebooks = model.cfg.num_codebooks;
        if (codebooks != 16 || model.cfg.hidden_size <= 0 ||
            model.cfg.audio_vocab_size <= 0) {
            throw std::runtime_error("Depth diagnostic expects 16 audio codebooks");
        }

        std::vector<std::vector<float>> hidden(
            2, std::vector<float>(static_cast<size_t>(model.cfg.hidden_size))
        );
        for (int branch = 0; branch < 2; ++branch) {
            for (size_t i = 0; i < hidden[branch].size(); ++i) {
                // Deterministic hidden activations, not synthetic weights:
                // the model itself executes every Depth transformer layer.
                hidden[branch][i] = 0.075f * std::sin(
                    static_cast<float>(i) * 0.021f + branch * 0.37f
                );
            }
        }

        std::vector<int> forced(static_cast<size_t>(codebooks - 1));
        for (size_t i = 0; i < forced.size(); ++i) {
            forced[i] = static_cast<int>((41 + i * 173) %
                                         model.cfg.audio_vocab_size);
        }

        {
            breeze::DepthRunner runner;
            runner.init(model, 2);
            std::mt19937 rng(42);
            // Build/preheat the graph once, keeping the same forced
            // inter-codebook dependencies for HVX/HMX comparison.
            runner.begin_request();
            runner.run(model, hidden, 42 % model.cfg.audio_vocab_size,
                       cfg_scale, rng, nullptr, forced.data(),
                       static_cast<int>(forced.size()));
            runner.begin_request();
            trace_.clear();
            runner.logits_observer = this;
            const auto start = std::chrono::steady_clock::now();
            const auto result = runner.run(
                model, hidden, 42 % model.cfg.audio_vocab_size, cfg_scale,
                rng, nullptr, forced.data(), static_cast<int>(forced.size())
            );
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start
            ).count();
            if (result != forced || trace_.size() != forced.size()) {
                throw std::runtime_error("Depth diagnostic incomplete 15-step output");
            }
            save(output_file);
            std::fprintf(stderr,
                "[BREEZE_DEPTH_MODEL_TEST] steps=%zu branches=2 cfg=3.20 "
                "duration_ms=%.3f trace=%s all-ok\n",
                trace_.size(), elapsed_ms, output_file.c_str());
            runner.logits_observer = nullptr;
        }
        model.free();
    }

private:
    std::vector<std::vector<float>> trace_;

    void save(const std::string & output_file) const {
        std::ofstream out(output_file, std::ios::out | std::ios::binary |
                                              std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot create Depth logits trace");
        const uint32_t magic = 0x42445054u; // BDPT, little-endian on arm64
        const uint32_t version = 1;
        const uint32_t count = static_cast<uint32_t>(trace_.size());
        out.write(reinterpret_cast<const char *>(&magic), 4);
        out.write(reinterpret_cast<const char *>(&version), 4);
        out.write(reinterpret_cast<const char *>(&count), 4);
        for (const auto & logits : trace_) {
            const uint32_t size = static_cast<uint32_t>(logits.size());
            out.write(reinterpret_cast<const char *>(&size), 4);
            out.write(reinterpret_cast<const char *>(logits.data()),
                      static_cast<std::streamsize>(logits.size() * sizeof(float)));
        }
        if (!out.good()) throw std::runtime_error("Incomplete Depth logits trace");
    }
};
