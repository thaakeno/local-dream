#pragma once
#include "breeze/model.h"
#include "breeze/depth_decoder.h"
#include <memory>
#include <random>
#include <vector>
namespace breeze {
struct GpuDepthEngine {
    BreezeModel gpu;
    DepthRunner depth;
    const BreezeModel * source = nullptr;
    bool active = false;
    int branches = 0;
    std::string loaded_path;

    ~GpuDepthEngine() { reset(); }
    void reset();
    bool init(BreezeModel & original, int n_branches, const char * gguf_path);
    std::vector<int> run(
        const std::vector<std::vector<float>> & hiddens,
        int first, float cfg_scale, std::mt19937 & rng
    );
};
}
