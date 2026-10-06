#pragma once

#include "breeze/common.h"
#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

// GGUF model storage. The primary model allocation intentionally follows the
// original v153 loader semantics. Decoder F32 codebooks are mirrored into one
// small ordinary HTP allocation so their GET_ROWS path does not depend on the
// large aggregate model mapping.
struct GGUFModel {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_context * codebook_meta = nullptr;
    ggml_backend_buffer_t codebook_buffer = nullptr;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    bool load(const std::string & path, Backend & be);
    void free();

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * find(const std::string & name) const;
    bool has(const std::string & name) const;

    int   kv_u32(const char * key, int def = 0) const;
    float kv_f32(const char * key, float def = 0.0f) const;
    bool  kv_bool(const char * key, bool def = false) const;
    std::string kv_str(const char * key, const std::string & def = "") const;
    std::vector<int> kv_i32_array(const char * key) const;
    std::vector<std::string> kv_str_array(const char * key) const;
};

}
