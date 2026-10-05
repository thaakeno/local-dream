#pragma once

#include "breeze/model.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

struct CodecStreamCacheBlock {
    size_t offset_f32 = 0;
    int left = 0;
    int channels = 0;
};

struct VocoderStreamState {
    bool initialized = false;
    int position = 0;
    KVCache kv;

    // All causal conv / transposed-conv carry state lives in one persistent
    // HTP buffer. Previous builds copied every cache block HTP -> CPU -> HTP
    // on each flush, which was both slow and vulnerable to stale/invalid state.
    ggml_context * conv_ctx = nullptr;
    ggml_backend_buffer_t conv_buffer = nullptr;
    ggml_tensor * conv_storage = nullptr;
    // Per-bank capacity. Two banks are allocated so a flush never overwrites
    // carry state that is still being consumed by the same HTP graph.
    size_t conv_capacity_f32 = 0;
    size_t conv_used_f32 = 0;
    int conv_bank = 0;

    std::unordered_map<std::string, CodecStreamCacheBlock> conv1d;
    std::unordered_map<std::string, CodecStreamCacheBlock> tconv1d;

    void init(BreezeModel & model);
    void reset();
    void free();
};

struct MimiCodec {
    BreezeModel * m = nullptr;
    VocoderStreamState stream;

    void init(BreezeModel & model);
    void stream_reset();

    // Exact stateful decoder: only the newly generated frames are evaluated.
    // Transformer KV plus causal conv / transposed-conv carry state are preserved
    // between calls, matching Breeze's reference fast streaming runtime.
    std::vector<float> decode_stream(const std::vector<int> & codes, int n_frames, int n_cb = 0);

    // Full-clip reference path kept for voice conversion and validation.
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);
};

namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation);
ggml_tensor * convtr1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride);
ggml_tensor * depthwise1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int kernel);
ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & prefix, ggml_tensor * x);
ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x);
ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int seq_len);
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_transformer_stream(ggml_context * ctx, BreezeModel & m, Graph & g,
                                        VocoderStreamState & state, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_codebooks, int seq_len);
ggml_tensor * vocoder_decode_stream(ggml_context * ctx, BreezeModel & m, Graph & g,
                                    VocoderStreamState & state,
                                    const std::vector<int> & codes, int n_codebooks, int seq_len);

}

}
