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
test -s "$(pwd)/overlay/breeze/include/breeze/codec.h"
test -s "$(pwd)/overlay/breeze/src/codec.cpp"
test -s "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
test -s "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
test -s "$(pwd)/overlay/breeze/src/generation.cpp"

grep -q 'HTP_OP_SIN' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_COL2IM_1D_BIAS' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_SNAKE' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_ADD' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_MUL' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'ggml_hexagon_is_breeze_channel_binary' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'Fit binary staging to the available VTCM' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'BREEZE_SNAKE' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'BREEZE_COL2IM_BIAS' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'hmx_rows > 2' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'if (opt_arch >= 81)' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'vocoder_decode_stream' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'decode_stream' "$(pwd)/overlay/breeze/src/codec.cpp"
grep -q 'vocoder_transformer_stream' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
# Stateful vocoder attention must expose the KV axis as GGML ne[0]. This
# catches the flush-2 regression where T == KV only on the first chunk.
grep -Fq 'g.input_f32(mask_v, kv_len, T)' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
grep -q 'ggml_tensor \* tensor' "$(pwd)/overlay/breeze/include/breeze/codec.h"
grep -q 'ggml_new_tensor_2d(conv_ctx, GGML_TYPE_F32' "$(pwd)/overlay/breeze/src/codec.cpp"
grep -q 'ggml_concat(ctx, block.tensor, x, 0)' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q '1.0e-9f' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'ggml_div(ctx, one, ggml_add(ctx, beta, tiny))' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
if grep -q 'ggml_exp(ctx, ggml_neg(ctx, lb))' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"; then
    echo "Unsafe exp(-beta) SnakeBeta rewrite returned" >&2
    exit 1
fi
grep -q 'snake-beta-reference' "$(pwd)/native/breeze-htp-selftest.cpp"
grep -q 'const int left = (K - 1) / stride' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
grep -q 'const int prefix = left \* stride' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"
if grep -q 'conv_storage\|conv_bank\|offset_f32\|conv_capacity_f32\|conv_used_f32' "$(pwd)/overlay/breeze/include/breeze/codec.h" "$(pwd)/overlay/breeze/src/codec.cpp" "$(pwd)/overlay/breeze/src/codec_decoder.cpp"; then
    echo "Legacy monolithic/ping-pong Breeze state returned" >&2
    exit 1
fi
grep -q 'const int kv_start' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
grep -q 'kv_store_future' "$(pwd)/overlay/breeze/src/codec_transformer.cpp"
if grep -q 'stream_safe_full\|stream_history_\|VOCODER_FALLBACK' "$(pwd)/overlay/breeze/include/breeze/codec.h" "$(pwd)/overlay/breeze/src/codec.cpp"; then
    echo "Breeze streaming must not contain a hidden full-decode fallback" >&2
    exit 1
fi
if grep -q 'StreamCacheUpdate\|block.data' "$(pwd)/overlay/breeze/src/codec_decoder.cpp"; then
    echo "Legacy host-roundtrip vocoder cache path returned" >&2
    exit 1
fi
grep -q 'step + 1 >= max_new' "$(pwd)/overlay/breeze/src/generation.cpp"
grep -q '\[BREEZE_AUDIO\]' "$(pwd)/overlay/breeze/src/generation.cpp"
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
queue=backend-default
extensions=sin-hvx,col2im1d-htp,col2im-bias-fused,col2im-layout-ocxk-ggml-reference,channel-bcast-addmul-hvx,snake-hvx-fused,adaptive-binary-vtcm,exact-elu-lowering,transpose-conv-gemm-col2im
streaming_vocoder=reference-state-tensors-input-cache-tconv-windowed-kv-chunk1-exact-snakebeta
formats=f16,q8_0,q6_k,q4_k,q8_0-dd4,q8_0-dd2,q4_k-dd2
EOF

ls -lh "$JNI_DIR/libbreeze_server.so" "$JNI_DIR/libbreeze_selftest.so" "$ASSET_DIR"/libggml-htp-v*.so
