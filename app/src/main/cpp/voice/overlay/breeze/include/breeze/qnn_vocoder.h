#pragma once
#include <memory>
#include <vector>
namespace breeze {
class BreezeQnnVocoder {
public:
    BreezeQnnVocoder();
    ~BreezeQnnVocoder();
    BreezeQnnVocoder(const BreezeQnnVocoder &) = delete;
    BreezeQnnVocoder & operator=(const BreezeQnnVocoder &) = delete;
    bool init_from_environment();
    bool ready() const;
    void reset();
    std::vector<float> decode_stream(const std::vector<int> & codes,int n_frames,int n_codebooks,int samples_per_frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
