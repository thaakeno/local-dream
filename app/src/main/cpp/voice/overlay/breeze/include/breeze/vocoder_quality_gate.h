#pragma once

// App-owned streaming-vocoder acceptance gate. Never confuse a graph returning
// the expected sample count with producing audible or equivalent PCM.
// Pure C++: deterministic host tests, no Qualcomm/ggml runtime dependency.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace breeze {

struct VocoderQualityReport {
    bool comparable = false;
    bool passed = false;
    size_t samples = 0;
    double reference_rms = 0.0;
    double candidate_rms = 0.0;
    double rms_ratio = 0.0;
    double correlation = 0.0;
    double normalized_error = 0.0;
};

class VocoderQualityGate final {
public:
    static VocoderQualityReport evaluate(
        const std::vector<float> & reference,
        const std::vector<float> & candidate
    ) {
        VocoderQualityReport report;
        report.samples = reference.size();
        if (reference.empty() || reference.size() != candidate.size())
            return report;
        double ref2 = 0, candidate2 = 0, dot = 0, error2 = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double a = reference[i], b = candidate[i];
            if (!std::isfinite(a) || !std::isfinite(b))
                return report;
            ref2 += a * a;
            candidate2 += b * b;
            dot += a * b;
            const double delta = a - b;
            error2 += delta * delta;
        }
        const double n = (double) reference.size();
        report.reference_rms = std::sqrt(ref2 / n);
        report.candidate_rms = std::sqrt(candidate2 / n);
        // Silence cannot establish whether a vocoder path is equivalent.
        if (report.reference_rms < 0.0001 || !std::isfinite(ref2) ||
            !std::isfinite(candidate2) || candidate2 == 0)
            return report;
        report.comparable = true;
        report.rms_ratio = std::sqrt(candidate2 / ref2);
        report.correlation = dot / std::sqrt(ref2 * candidate2);
        report.normalized_error = std::sqrt(error2 / ref2);
        report.passed =
            report.rms_ratio >= 0.75 && report.rms_ratio <= 1.33 &&
            report.correlation >= 0.90 && report.normalized_error <= 0.45;
        return report;
    }
};
} // namespace breeze
