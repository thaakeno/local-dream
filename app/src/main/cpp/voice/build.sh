#!/usr/bin/env bash
# Breeze TTS 2 strict Qualcomm Hexagon HTP runtime for Local Dream.
# No CPU or Vulkan backend is compiled into the Breeze execution path.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"

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
    ggml/src/ggml-hexagon/htp/main.c \
    ggml/src/ggml-hexagon/htp/hex-utils.h \
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

# Preserve the original v153 DSPQueue scheduler: 1280-op packets, 32 pending,
# fusion on, with no dependency-barrier rewrite.
grep -q 'static int opt_opbatch  = 1280' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'static int opt_opqueue  = 32' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'static int opt_opfusion = 1' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
if grep -q 'GGML_HEXAGON_DEPBARRIER\|has_data_hazard\|V81_LEGACY_BATCH' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"; then
    echo "Non-v153 DSPQueue scheduler code returned" >&2
    exit 1
fi

# Preserve v153's proven GET_ROWS/scheduler path exactly. The previous
# F32 VTCM-staging override changed the real packet/kernels and caused the
# immediate 0x2e abort in v175. Instead apply the upstream v81 cache
# coherency fix from llama.cpp PR #29977: dccleaninva must step by 64 bytes.
grep -q 'HEX_DCACHE_OP_SIZE[[:space:]]*64' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"
grep -q 'j += HEX_DCACHE_OP_SIZE' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"
grep -q 'i += HEX_DCACHE_OP_SIZE' "$GGML_DIR/src/ggml-hexagon/htp/hex-utils.h"
if grep -q 'src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"; then
    echo "Non-v153 F32 GET_ROWS override returned" >&2
    exit 1
fi
grep -q 'get-rows-f32-single' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'get-rows-f32-multi' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'get-rows-f32-weight-2048-highrows' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'GGML_BACKEND_BUFFER_USAGE_WEIGHTS' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'HTP decoder codebook rows verified' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -q 'verified %zu finite decoder codebooks' "$(pwd)/overlay/breeze/src/gguf_loader.cpp"
grep -Fq '"${BREEZE_OVERLAY_DIR}/src/gguf_loader.cpp"' "$(pwd)/CMakeLists.txt"

# Exact v153 streaming graph invariants.
grep -q 'std::vector<float> data' "$(pwd)/overlay/breeze/include/breeze/codec.h"
grep -q 'struct StreamCacheUpdate' "$(pwd)/overlay/breeze/include/breeze/codec.h"
grep -q 'g.input_f32(block.data' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -Fq 'ggml_conv_1d(ctx, w, joined, 1, 0, dilation)' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'keep_tail(ctx, g, joined, block, N, updates)' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'ggml_exp(ctx, ggml_neg(ctx, lb))' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -Fq 'g.input_f32(mask_v, kv_len, T)' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
grep -q 'u.block->data = tensor_to_f32(u.tensor)' "$(pwd)/overlay/breeze/src/codec.cpp"
if grep -q 'CodecDebugProbes\|convtr1d_raw\|kv_store_future\|block.tensor\|conv_ctx\|conv_buffer' \
    "$(pwd)/overlay/breeze/include/breeze/codec.h" \
    "$(pwd)/overlay/breeze/src/codec.cpp" \
    "$(pwd)/overlay/breeze/src/codec_decoder.cpp" \
    "$(pwd)/overlay/breeze/src/codec_transformer.cpp"; then
    echo "Post-v153 codec graph rewrite returned" >&2
    exit 1
fi

# Keep the exact v153 generation/streaming behavior too.
grep -q '\[BREEZE_VOCODER_STREAM\]' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'codec.decode_stream' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q 'struct AudioEmbedRunner' "$(pwd)/overlay/breeze/include/breeze/backbone.h"

rm -rf "$BUILD_DIR"

cmake -S "$(pwd)" -B "$BUILD_DIR" -G Ninja     -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-28     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DBREEZE_SOURCE_DIR="$BREEZE_DIR"     -DGGML_SOURCE_DIR="$GGML_DIR"     -DBUILD_SHARED_LIBS=OFF     -DGGML_STATIC=ON     -DGGML_HEXAGON=ON     -DGGML_HEXAGON_USE_MEMPOOL=OFF     -DGGML_OPENMP=OFF     -DGGML_CPU=OFF     -DGGML_VULKAN=OFF     -DGGML_CUDA=OFF     -DGGML_LLAMAFILE=OFF     -DGGML_BACKEND_DL=OFF     -DPREBUILT_LIB_DIR=android_aarch64     -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"     -DCMAKE_POLICY_VERSION_MINIMUM=3.10

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
queue=v153-default-opbatch1280x32
extensions=sin-hvx,col2im1d-htp,col2im-bias-fused,col2im-layout-ocxk-ggml-reference,channel-bcast-addmul-hvx,snake-hvx-fused,adaptive-binary-vtcm,exact-elu-lowering,transpose-conv-gemm-col2im,model-weights-extended-map,exact-v153-hexagon-kernels,exact-v153-codec-graph,v81-dcache-cleaninva-64b
streaming_vocoder=exact-v153-host-cache-chunk4
formats=f16,q8_0,q6_k,q4_k,q8_0-dd4,q8_0-dd2,q4_k-dd2
EOF

ls -lh "$JNI_DIR/libbreeze_server.so" "$JNI_DIR/libbreeze_selftest.so" "$ASSET_DIR"/libggml-htp-v*.so
