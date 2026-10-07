#pragma once

#include <memory>
#include <random>
#include <vector>

namespace breeze {

struct BreezeModel;

// Optional SM8850 QNN fast path for Breeze's 15-step residual codebook decoder.
// V2 supports batch-1 and CFG batch-2 graphs. Sampling stays on the host so
// temperature/top-k/top-p and CFG semantics remain identical to DepthRunner.
class QnnDepthRunner {
public:
    QnnDepthRunner();
    ~QnnDepthRunner();

    QnnDepthRunner(const QnnDepthRunner &) = delete;
    QnnDepthRunner & operator=(const QnnDepthRunner &) = delete;

    bool init(BreezeModel & m, int branches);
    bool ready() const;
    bool run(
        BreezeModel & m,
        const std::vector<std::vector<float>> & backbone_hiddens,
        int first_codebook,
        float cfg_scale,
        std::mt19937 & rng,
        std::vector<int> & residual_codebooks
    );
    void disable();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace breeze
