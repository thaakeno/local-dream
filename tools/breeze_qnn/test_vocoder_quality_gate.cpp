#include "breeze/vocoder_quality_gate.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

int main() {
    using breeze::VocoderQualityGate;
    std::vector<float> audio(1920);
    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.25f * std::sin(6.283185307179586 * (double) i / 64.0);
    auto report = VocoderQualityGate::evaluate(audio, audio);
    assert(report.comparable && report.passed);
    assert(report.normalized_error == 0.0);
    assert(std::abs(report.rms_ratio - 1.0) < 1e-6);
    auto candidate = audio;
    for (float & x : candidate) x *= 0.01f;
    report = VocoderQualityGate::evaluate(audio, candidate);
    assert(report.comparable && !report.passed && report.rms_ratio < 0.02);
    for (float & x : candidate) x *= -100.0f;
    report = VocoderQualityGate::evaluate(audio, candidate);
    assert(report.comparable && !report.passed && report.correlation < -0.95);
    candidate = audio;
    candidate[200] = std::numeric_limits<float>::quiet_NaN();
    report = VocoderQualityGate::evaluate(audio, candidate);
    assert(!report.comparable && !report.passed);
    candidate.clear();
    report = VocoderQualityGate::evaluate(audio, candidate);
    assert(!report.comparable && !report.passed);
    const std::vector<float> silence(audio.size(),0.0f);
    report = VocoderQualityGate::evaluate(silence,silence);
    assert(!report.comparable && !report.passed);
    std::cout << "Vocoder quality gate: 6/6 cases passed\n";
}
