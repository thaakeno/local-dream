#pragma once

#include <memory>
#include <random>
#include <vector>

namespace breeze {

struct BreezeModel;

// Optional SM8850 QNN fast path for Breeze's 15-step residual codebook decoder.
// Sampling stays on the host so the model's temperature/top-k/top-p behavior
// remains identical to the normal DepthRunner path.
class QnnDepthRunner {
public:
    QnnDepthRunner();
    ~QnnDepthRunner();

    QnnDepthRunner(const QnnDepthRunner &) = delete;
    QnnDepthRunner & operator=(const QnnDepthRunner &) = delete;

    bool init(BreezeModel & m);
    bool ready() const;
    bool run(
        BreezeModel & m,
        const std::vector<float> & backbone_hidden,
        int first_codebook,
        std::mt19937 & rng,
        std::vector<int> & residual_codebooks
    );
    void disable();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace breeze
