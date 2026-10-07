#pragma once

#include "breeze/backbone.h"

#include <memory>
#include <vector>

namespace breeze {

// Optional SM8850/V81 QNN fast path for the complete autoregressive Qwen3
// backbone. Prefill is bucketed (64/128/256/512); single-token decode keeps a
// persistent host-visible KV tensor and only copies the newly produced K/V row
// into the next slot after each graph execution.
class QnnBackboneRunner {
public:
    QnnBackboneRunner();
    ~QnnBackboneRunner();

    QnnBackboneRunner(const QnnBackboneRunner &) = delete;
    QnnBackboneRunner & operator=(const QnnBackboneRunner &) = delete;

    bool init(BreezeModel & m, int branches);
    bool ready() const;
    int max_seq() const;

    bool prefill(
        BreezeModel & m,
        const std::vector<float> & cond_embeddings,
        int cond_tokens,
        const std::vector<float> * uncond_embeddings,
        int uncond_tokens,
        StepOut & cond_out,
        StepOut & uncond_out
    );

    bool step(
        BreezeModel & m,
        const std::vector<float> & audio_embedding,
        StepOut & cond_out,
        StepOut & uncond_out
    );

    void disable();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace breeze
