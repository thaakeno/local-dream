#!/usr/bin/env bash
# Breeze TTS 2: proven Hexagon HTP path plus optional Adreno Vulkan depth.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT}"

cd "$(dirname "$0")"

BREEZE_REPO="https://github.com/HoppouAI/Breeze-TTS-2.cpp.git"
BREEZE_COMMIT="a0e177f91242ebbd2ea8d61548d742c89e4b9e06"
HEXAGON_REPO="https://github.com/kan-linux/ggml-hexagon.git"
HEXAGON_COMMIT="ab9acc4476cfe3add369379a3783fea190505a48"

DEPS_DIR="$(pwd)/build/deps"
BREEZE_DIR="$DEPS_DIR/Breeze-TTS-2.cpp"
HEXAGON_DIR="$DEPS_DIR/ggml-hexagon"
GGML_DIR="$HEXAGON_DIR/ggml"
BUILD_DIR="$(pwd)/build/android"

fetch_pinned_repo() {
    local dir="$1"
    local url="$2"
    local commit="$3"
    if [[ ! -d "$dir/.git" ]]; then
        rm -rf "$dir"
        mkdir -p "$(dirname "$dir")"
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
    fi
    if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
        git -C "$dir" fetch -q --depth 1 origin "$commit"
        git -C "$dir" checkout -q --detach FETCH_HEAD
    fi
    test "$(git -C "$dir" rev-parse HEAD)" = "$commit"
    git -C "$dir" reset --hard "$commit"
    git -C "$dir" clean -fdx
}

fetch_pinned_repo "$BREEZE_DIR" "$BREEZE_REPO" "$BREEZE_COMMIT"
fetch_pinned_repo "$HEXAGON_DIR" "$HEXAGON_REPO" "$HEXAGON_COMMIT"

test -f "$BREEZE_DIR/include/breeze/breeze.h"
test -f "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'GGML_TYPE_Q6_K' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'GGML_TYPE_Q4_K' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'GGML_TYPE_Q2_K' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"

# Integrate the LocalDream Hexagon extension as complete pinned source overlays.
# The upstream SHAs above are immutable; no fuzzy patch hunks or runtime rewriting.
HEXAGON_OVERLAY="$(pwd)/overlay/ggml-hexagon"
for rel in \
    ggml/src/ggml-hexagon/htp/CMakeLists.txt \
    ggml/src/ggml-hexagon/htp/htp-ops.h \
    ggml/src/ggml-hexagon/htp/htp-ctx.h \
    ggml/src/ggml-hexagon/htp/hex-utils.h \
    ggml/src/ggml-hexagon/htp/hvx-erf.h \
    ggml/src/ggml-hexagon/htp/main.c \
    ggml/src/ggml-hexagon/ggml-hexagon.cpp; do
    test -s "$HEXAGON_OVERLAY/$rel"
    cp "$HEXAGON_OVERLAY/$rel" "$HEXAGON_DIR/$rel"
done

for src in \
    breeze-sin-ops.c \
    breeze-col2im-ops.c \
    breeze-channel-bcast-ops.c \
    breeze-snake-ops.c; do
    cp "$(pwd)/native/$src" "$GGML_DIR/src/ggml-hexagon/htp/$src"
done

test -s "$(pwd)/overlay/breeze/include/breeze/backbone.h"
test -s "$(pwd)/overlay/breeze/src/backbone.cpp"
test -s "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
test -s "$(pwd)/overlay/breeze/include/breeze/gguf_loader.h"
test -s "$(pwd)/overlay/breeze/include/breeze/codec.h"
test -s "$(pwd)/overlay/breeze/src/codec.cpp"
test -s "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
test -s "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
test -s "$(pwd)/overlay/breeze/src/generation.cpp"

# v174 is intentionally the exact v153 codec/transformer graph that completed
# end-to-end on the target SM8850. Only backend/model-loader correctness fixes
# that do not rewrite the graph are allowed here.
grep -q 'HTP_OP_SIN' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_COL2IM_1D_BIAS' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_SNAKE' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_ADD' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_MUL' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"

# Preserve the proven v153 queue capacity/fusion defaults exactly. The v181-v185
# producer/consumer visibility scheduler rewrites were experimental and caused
# major regressions (startup SIGSEGVs and vocoder stalls), so they are forbidden.
grep -q 'static int opt_opbatch  = 1280' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'static int opt_opqueue  = 32' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'static int opt_opfusion = 1' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
if grep -q 'ggml_hexagon_v81_needs_visibility_split\|v81 visibility batch split before\|v81 visibility sync before' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"; then
    echo "Speculative v81 scheduler rewrite returned" >&2
    exit 1
fi

# Apply upstream llama.cpp PR #29977 exactly: Hexagon SDK 6.6 performs one
# dccleaninva every 64 bytes. A 128-byte step can leave half-lines stale after
# DMA reuse on SM8850/v81. Alignment remains 128 bytes; only instruction stride
# changes to 64 bytes.
grep -q '#define HEX_DCACHE_OP_SIZE         64' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"
grep -q 'j += HEX_DCACHE_OP_SIZE' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"
grep -q 'i += HEX_DCACHE_OP_SIZE' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"

# Keep v153 GET_ROWS, but fix the quantized-weight contract. Hexagon MUL_MAT
# consumes Q4_K/Q2_K/etc. from tiled REPACK storage. The loader briefly marks
# the aggregate buffer as WEIGHTS while uploading (to repack), then switches it
# back to ANY before first compute so SM8850 keeps ordinary delayed mapping
# instead of the v161-v163 delayed-extended mapping that stalled the vocoder.
if grep -q 'src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"; then
    echo "Non-v153 F32 GET_ROWS override returned" >&2
    exit 1
fi
grep -q 'get-rows-f32-single' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'get-rows-f32-multi' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'get-rows-f32-ordinary-2048-highrows' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'quant-matmul-%s-repack-ordinary-map-n%d' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -Fq 'GGML_TYPE_Q4_K, "q4k", 1' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -Fq 'GGML_TYPE_Q4_K, "q4k", 40' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -Fq 'GGML_TYPE_Q2_K, "q2k", 1' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -Fq 'GGML_TYPE_Q2_K, "q2k", 40' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'v81-dsp-gelu-erf-reference' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'v81-hvx-gelu-matmul-chain' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'Hexagon libm erff() per element' "$GGML_DIR/src/ggml-hexagon/htp/hvx-erf.h"
grep -q 'ordinary HTP codebook mirrors verified' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'quantized GGUF weights repacked for HTP' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'first decoder projection verified finite' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'GGML_BACKEND_BUFFER_USAGE_WEIGHTS' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'GGML_BACKEND_BUFFER_USAGE_ANY' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'codebook_buffer' "$(pwd)/overlay/breeze/include/breeze/gguf_loader.h"
grep -q 'refusing %s with raw quantized HTP weight' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -Fq '"${BREEZE_OVERLAY_DIR}/src/gguf_loader.cpp"' "$(pwd)/CMakeLists.txt"

# v193 production streaming invariants. v192 proved that the old silent
# stateful runs were contaminated by raw quantized weights. Keep the v192
# REPACK correctness fix, but restore an exact incremental vocoder that leaves
# causal state resident on HTP and never re-decodes overlapping prefixes.
grep -q 'conv_capacity_f32' "$(pwd)/overlay/breeze/include/breeze/codec.h"
grep -q 'conv_bank' "$(pwd)/overlay/breeze/include/breeze/codec.h"
if grep -q 'struct StreamCacheUpdate\|std::vector<float> data' "$(pwd)/overlay/breeze/include/breeze/codec.h"; then
    echo "Host-roundtrip vocoder cache regression returned" >&2
    exit 1
fi
grep -q 'cache_view(ctx, state, block, state.conv_bank)' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'state.conv_bank ^ 1' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'stream_convtr1d_raw' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'K - stride' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'kv_store_future' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
grep -q 'kv_history_view' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
if grep -q 'cache_append(ctx, g, state.kv' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"; then
    echo "Streaming KV write-read alias regression returned" >&2
    exit 1
fi

# SnakeBeta keeps the exact v192-safe reference math, but invariant alpha
# and inv-beta vectors are precomputed once while the GGUF is loaded. The hot
# waveform graph must contain only the five ops recognized by HTP_OP_SNAKE.
grep -q 'SnakeBeta reference parameters precomputed' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -Fq '1.0f / (std::exp(v) + 1.0e-9f)' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'alpha_param' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'inv_beta_param' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
if grep -q 'ggml_exp(ctx, lb)\|std::vector<float> ones' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"; then
    echo "Per-flush SnakeBeta parameter recomputation returned" >&2
    exit 1
fi

# Normal TTS must decode each generated frame once. The old reference window
# path decoded 40, then 80, then 94 frames for a 94-frame clip and caused the
# 167-second run on SM8850.
grep -q 'codec.decode_stream(sub, count)' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q '[BREEZE_AUDIO].*path=stateful-stream' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q '[BREEZE_VOCODER_STREAM]' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'fallback_chunk = chunk_max' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'audio_embed.run(m, frame)' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'BREEZE_GENERATION_DONE' "$(pwd)/overlay/breeze/src/generation.cpp"
if grep -q 'std::async\|submit_qnn\|collect_qnn' "$(pwd)/overlay/breeze/src/generation.cpp"; then
    echo "Concurrent HTP/QNN scheduling regression returned" >&2
    exit 1
fi

# QNN vocoder fast path must remain optional and fall back to the proven ggml
# stateful decoder when its downloaded context binary is absent.
grep -q 'BREEZE_QNN_VOCODER_PATH' "$(pwd)/native/breeze-qnn-vocoder.cpp"
grep -q 'graph%d_ms' "$(pwd)/native/breeze-qnn-vocoder.cpp"
grep -q 'qnn_vocoder->decode_stream' "$(pwd)/overlay/breeze/src/codec.cpp"
grep -q 'BREEZE_QNN_FIRST_NEW' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'BREEZE_QNN_STEADY_NEW' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'anchor_chars = 200' "$(pwd)/overlay/breeze/src/generation.cpp"

# Full QNN generator v5 keeps the proven batch-1 depth contract while fixing
# the v4 backbone architecture: linked shared-weight contexts, true batch-1
# AR1 for CFG=1, smaller prompt buckets, and an explicit legacy toggle.
grep -q 'BREEZE_QNN_DEPTH_PATH' "$(pwd)/native/breeze-qnn-depth-decoder.cpp"
grep -q 'physical_batch=1' "$(pwd)/native/breeze-qnn-depth-decoder.cpp"
grep -q 'cfg_serial=' "$(pwd)/native/breeze-qnn-depth-decoder.cpp"
grep -q 'profile=adaptive-performance' "$(pwd)/native/breeze-qnn-depth-decoder.cpp"
grep -q 'rpc_poll=0' "$(pwd)/native/breeze-qnn-vocoder.cpp"
grep -q 'dcvsEnable = 1' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'BREEZE_QNN_BACKBONE_LINKED_PATH' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'BREEZE_QNN_BACKBONE_PREFILL_PATH' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'BREEZE_QNN_BACKBONE_STEP_B1_PATH' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'shared_context=%d' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'prompt_context_released=%d' "$(pwd)/native/breeze-qnn-backbone.cpp"
grep -q 'persistent_backbone' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'persistent_depth' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'BREEZE_QNN_CACHE' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'depth.run(m, hiddens, cb0' "$(pwd)/overlay/breeze/src/generation.cpp"

# Full-clip decode remains available for voice conversion/reference work, but
# production decode must not attach the old full-tensor SUM diagnostic probes.
grep -Fq 'vocoder_decode(g.ctx, *m, g, codes, n_cb, T, nullptr)' "$(pwd)/overlay/breeze/src/codec.cpp"
if grep -q 'u.block->data = tensor_to_f32' "$(pwd)/overlay/breeze/src/codec.cpp"; then
    echo "Per-flush HTP-to-host cache round trips returned" >&2
    exit 1
fi

# glslc must execute on the Linux host while targeting Android Vulkan.
if ! dpkg-query -W -f='\x24{Status}' spirv-headers 2>/dev/null | grep -q "install ok installed"; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends spirv-headers
fi
GLSLC="$ANDROID_NDK_ROOT/shader-tools/linux-x86_64/glslc"
test -x "$GLSLC" || { echo "Android NDK host glslc is missing" >&2; exit 1; }

rm -rf "$BUILD_DIR"

cmake -S "$(pwd)" -B "$BUILD_DIR" -G Ninja     -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-28     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DBREEZE_SOURCE_DIR="$BREEZE_DIR"     -DGGML_SOURCE_DIR="$GGML_DIR"     -DBUILD_SHARED_LIBS=OFF     -DGGML_STATIC=ON     -DGGML_HEXAGON=ON     -DGGML_HEXAGON_USE_MEMPOOL=OFF     -DGGML_OPENMP=OFF     -DGGML_CPU=OFF     -DGGML_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE="$GLSLC"     -DGGML_CUDA=OFF     -DGGML_LLAMAFILE=OFF     -DGGML_BACKEND_DL=OFF     -DPREBUILT_LIB_DIR=android_aarch64     -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"     -DQNN_SDK_ROOT="$QNN_SDK_ROOT"     -DCMAKE_POLICY_VERSION_MINIMUM=3.10

cmake --build "$BUILD_DIR" --target breeze-server breeze-htp-selftest htp-v73 htp-v75 htp-v79 htp-v81 -j "$(nproc)"

READELF="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
test -x "$READELF"
"$READELF" -d "$BUILD_DIR/breeze-server" | tee "$BUILD_DIR/breeze-needed.txt"
"$READELF" -d "$BUILD_DIR/breeze-htp-selftest" | tee "$BUILD_DIR/breeze-selftest-needed.txt"
if grep -Eq 'Shared library: \[libggml(-base|-hexagon)?\.so\]' "$BUILD_DIR/breeze-needed.txt"; then
    echo "Breeze server still depends on private ggml shared libraries" >&2
    exit 1
fi

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/breezelibs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"

cp "$BUILD_DIR/breeze-server" "$JNI_DIR/libbreeze_server.so"
cp "$BUILD_DIR/breeze-htp-selftest" "$JNI_DIR/libbreeze_selftest.so"
chmod +x "$JNI_DIR/libbreeze_server.so" "$JNI_DIR/libbreeze_selftest.so"

for arch in v73 v75 v79 v81; do
    skel="$(find "$BUILD_DIR" -type f -name "libggml-htp-${arch}.so" -print -quit)"
    if [[ -z "$skel" || ! -s "$skel" ]]; then
        echo "Missing Breeze HTP skel for $arch" >&2
        exit 1
    fi
    cp "$skel" "$ASSET_DIR/libggml-htp-${arch}.so"
done

cat > "$ASSET_DIR/backend-version.txt" <<EOF
runtime=HoppouAI/Breeze-TTS-2.cpp
runtime_commit=$BREEZE_COMMIT
backend=kan-linux/ggml-hexagon
backend_commit=$HEXAGON_COMMIT
mode=strict-htp-dspqueue
fallback=disabled
integration=pinned-source-overlay
queue=v202-opbatch1280x32-oppoll0
extensions=sin-hvx,col2im1d-htp,col2im-bias-fused,col2im-layout-ocxk-ggml-reference,channel-bcast-addmul-hvx,snake-hvx-fused,adaptive-binary-vtcm,exact-elu-lowering,transpose-conv-gemm-col2im,quant-weight-repack-ordinary-map,decoder-codebook-ordinary-htp-mirror,raw-quant-matmul-guard,stateful-vocoder-htp-resident,pingpong-causal-state,exact-tconv-output-overlap,bounded-stream-kv,precomputed-snake-params,snake-hotgraph-fused,upstream-dcache-64b-pr29977
v81_visibility=none-v153-scheduler
v81_execution=hvx-only-no-hmx
gelu_erf=dsp-libm-reference-v81
generator=qnn-v5-linked-fastpath-or-ggml-htp,optional-adreno-vulkan-depth
vocoder=qnn-htp-feature64-serialized-or-ggml-fallback-v198
qnn_vocoder=sm8850-v3-v81,feature64,host-lut-fp32,cached-reference-selftest,left-context25,serialized64x39,eos-first
formats=f16,q8_0,q6_k,q4_k,q8_0-dd4,q8_0-dd2,q4_k-dd2
EOF

ls -lh "$JNI_DIR/libbreeze_server.so" "$JNI_DIR/libbreeze_selftest.so" "$ASSET_DIR"/libggml-htp-v*.so
