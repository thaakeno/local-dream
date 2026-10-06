#pragma once

#include "breeze/model.h"
#include "breeze/qnn_vocoder.h"

#include <memory>
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

    // Two ping-pong banks in one persistent HTP allocation. A streaming graph
    // reads only the current bank and writes only the next bank, so there is no
    // same-graph read/write alias and no HTP -> CPU -> HTP cache round trip.
    ggml_context * conv_ctx = nullptr;
    ggml_backend_buffer_t conv_buffer = nullptr;
    ggml_tensor * conv_storage = nullptr;
    size_t conv_capacity_f32 = 0; // capacity per bank
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
    std::unique_ptr<BreezeQnnVocoder> qnn_vocoder;

    void init(BreezeModel & model);
    void stream_reset();
    bool uses_qnn_vocoder() const;

    // Exact stateful decoder: only the newly generated frames are evaluated.
    // Transformer KV plus causal conv / transposed-conv carry state are preserved
    // between calls, matching Breeze's reference fast streaming runtime.
    std::vector<float> decode_stream(const std::vector<int> & codes, int n_frames, int n_cb = 0);

    // Full-clip reference path kept for voice conversion and validation.
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);
};

namespace codec_detail {

struct VocoderDiagProbe {
    std::string name;
    ggml_tensor * scalar = nullptr;
};

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
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                                  int seq_len, std::vector<VocoderDiagProbe> * probes = nullptr);
ggml_tensor * vocoder_transformer_stream(ggml_context * ctx, BreezeModel & m, Graph & g,
                                        VocoderStreamState & state, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_codebooks, int seq_len,
                             std::vector<VocoderDiagProbe> * probes = nullptr);
ggml_tensor * vocoder_decode_stream(ggml_context * ctx, BreezeModel & m, Graph & g,
                                    VocoderStreamState & state,
                                    const std::vector<int> & codes, int n_codebooks, int seq_len);

}

}
