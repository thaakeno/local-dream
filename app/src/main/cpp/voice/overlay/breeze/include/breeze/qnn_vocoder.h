#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace breeze {

// QAIRT/QNN HTP vocoder. The context binary contains the official
// Qwen3-TTS/Breeze decoder at a fixed 64-frame causal window. Runtime calls pad
// only future frames, so valid prefix samples are bitwise independent of the
// padding by construction.
class BreezeQnnVocoder {
public:
    BreezeQnnVocoder();
    ~BreezeQnnVocoder();

    BreezeQnnVocoder(const BreezeQnnVocoder &) = delete;
    BreezeQnnVocoder & operator=(const BreezeQnnVocoder &) = delete;

    bool init_from_environment();
    bool ready() const;
    void reset();

    // Input is frame-major [T][n_cb], output is exactly T*samples_per_frame.
    std::vector<float> decode_stream(
        const std::vector<int> & codes,
        int n_frames,
        int n_codebooks,
        int samples_per_frame
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace breeze
