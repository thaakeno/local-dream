#include "breeze/gpu_depth.h"
#include "ggml-vulkan.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace breeze {
void GpuDepthEngine::reset() {
    depth.free();
    gpu.gg.free();
    gpu.backend.free();
    active = false;
    source = nullptr;
    branches = 0;
}
bool GpuDepthEngine::init(BreezeModel & original, int n_branches, const char * gguf_path) {
    if (active && source == &original && branches == n_branches) return true;
    reset();
    if (!gguf_path || !*gguf_path) {
        std::fprintf(stderr, "[BREEZE_GPU_DEPTH] missing GGUF path\n");
        return false;
    }
    try {
        const int devices = ggml_backend_vk_get_device_count();
        if (devices < 1) {
            std::fprintf(stderr, "[BREEZE_GPU_DEPTH] no Vulkan compute device\n");
            return false;
        }
        int gpu_index = -1;
        char name[256]{};
        for (int i = 0; i < devices; ++i) {
            char candidate[256]{};
            ggml_backend_vk_get_device_description(i, candidate, sizeof(candidate));
            if (std::strstr(candidate, "Adreno")) {
                gpu_index = i;
                std::strncpy(name, candidate, sizeof(name) - 1);
                break;
            }
        }
        if (gpu_index < 0) {
            std::fprintf(stderr, "[BREEZE_GPU_DEPTH] Adreno Vulkan device not found\n");
            return false;
        }
        gpu.backend.backend = ggml_backend_vk_init((size_t) gpu_index);
        if (!gpu.backend.backend) throw std::runtime_error("Adreno Vulkan init failed");
        gpu.backend.is_gpu = true;
        gpu.backend.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(gpu.backend.backend)
        );
        if (!gpu.backend.alloc) throw std::runtime_error("GPU graph allocator init failed");
        gpu.cfg = original.cfg;
        if (!gpu.gg.loadDepthSubset(gguf_path, gpu.backend))
            throw std::runtime_error("GPU depth weights loading failed");
        depth.init(gpu, n_branches);
        source = &original;
        branches = n_branches;
        active = true;
        std::fprintf(stderr,
            "[BREEZE_DEPTH_BACKEND] selected=adreno-vulkan device=%s branches=%d\n",
            name, branches);
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[BREEZE_GPU_DEPTH] initialization failed: %s\n", e.what());
        reset();
        return false;
    }
}
std::vector<int> GpuDepthEngine::run(
    const std::vector<std::vector<float>> & hiddens,
    int first, float cfg_scale, std::mt19937 & rng
) {
    if (!active) throw std::logic_error("Adreno depth backend inactive");
    const std::vector<int> result = depth.run(gpu, hiddens, first, cfg_scale, rng);
    for (int value : result) {
        if (value < 0 || value >= gpu.cfg.audio_vocab_size)
            throw std::runtime_error("GPU depth produced invalid codebook token");
    }
    return result;
}
}
