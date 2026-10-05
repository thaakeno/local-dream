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

# Apply versioned source patches against the exact pinned upstream commits.
# This is normal git patching: no runtime monkey-patching, no regex/source
# rewriting, and git apply --check fails immediately if either upstream moves.
for src in     breeze-sin-ops.c     breeze-col2im-ops.c     breeze-channel-bcast-ops.c     breeze-snake-ops.c; do
    cp "$(pwd)/native/$src" "$GGML_DIR/src/ggml-hexagon/htp/$src"
done

git -C "$HEXAGON_DIR" apply --check "$(pwd)/hexagon-breeze-v148.patch"
git -C "$HEXAGON_DIR" apply "$(pwd)/hexagon-breeze-v148.patch"
git -C "$HEXAGON_DIR" apply --check "$(pwd)/hexagon-fastrpc-breeze-v149.patch"
git -C "$HEXAGON_DIR" apply "$(pwd)/hexagon-fastrpc-breeze-v149.patch"
git -C "$BREEZE_DIR" apply --check "$(pwd)/breeze-core-v148.patch"
git -C "$BREEZE_DIR" apply "$(pwd)/breeze-core-v148.patch"

grep -q 'HTP_OP_SIN' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_COL2IM_1D_BIAS' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_SNAKE' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_ADD' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_MUL' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'ggml_hexagon_is_breeze_channel_binary' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'Fit binary staging to the available VTCM' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'BREEZE_SNAKE' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'BREEZE_COL2IM_BIAS' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'breeze_is_channel_binary' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"
grep -q 'HTP_OP_SNAKE.*op_snake' "$GGML_DIR/src/ggml-hexagon/htp/entry.c"
grep -q '1 / exp(lb) == exp(-lb)' "$BREEZE_DIR/src/codec_decoder.cpp"
grep -q 'struct AudioEmbedRunner' "$BREEZE_DIR/include/breeze/backbone.h"

rm -rf "$BUILD_DIR"

cmake -S "$(pwd)" -B "$BUILD_DIR" -G Ninja     -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-28     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DBREEZE_SOURCE_DIR="$BREEZE_DIR"     -DGGML_SOURCE_DIR="$GGML_DIR"     -DBUILD_SHARED_LIBS=OFF     -DGGML_STATIC=ON     -DGGML_HEXAGON=ON     -DGGML_HEXAGON_USE_MEMPOOL=ON     -DGGML_OPENMP=OFF     -DGGML_CPU=OFF     -DGGML_VULKAN=OFF     -DGGML_CUDA=OFF     -DGGML_LLAMAFILE=OFF     -DGGML_BACKEND_DL=OFF     -DPREBUILT_LIB_DIR=android_aarch64     -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"     -DCMAKE_POLICY_VERSION_MINIMUM=3.10

cmake --build "$BUILD_DIR" --target breeze-server htp-mempool-v73 htp-mempool-v75 htp-mempool-v79 htp-mempool-v81 -j "$(nproc)"

READELF="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
test -x "$READELF"
"$READELF" -d "$BUILD_DIR/breeze-server" | tee "$BUILD_DIR/breeze-needed.txt"
if grep -Eq 'Shared library: \[libggml(-base|-hexagon)?\.so\]' "$BUILD_DIR/breeze-needed.txt"; then
    echo "Breeze server still depends on private ggml shared libraries" >&2
    exit 1
fi

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/breezelibs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"

cp "$BUILD_DIR/breeze-server" "$JNI_DIR/libbreeze_server.so"
chmod +x "$JNI_DIR/libbreeze_server.so"

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
mode=strict-htp-fastrpc-mempool
fallback=disabled
transport=fastrpc-single-pool
extensions=sin-hvx,col2im1d-htp,col2im-bias-fused,channel-bcast-addmul-hvx,snake-hvx-fused,adaptive-binary-vtcm,exact-elu-lowering,transpose-conv-gemm-col2im
formats=f16,q8_0,q6_k,q4_k,q8_0-dd4,q8_0-dd2,q4_k-dd2
EOF

cat > "$ASSET_DIR/ggml-hexagon.cfg" <<'EOF'
[general]
version=0.5.7
dump_debug_info=0

[cdsp]
thread_counts=6
dump_diag_info=0
ndev=1
rpc_mmap_mode=0
enable_opfusion=1
fa_select=2
dsp_cache_mode=5
dsp_cache_trace_bit0=0
dsp_cache_trace_bit1=0
enable_graph_optimize=1
enable_graph_cache=1
mirror_threshold=0.88
enabled_ops=all
EOF

ls -lh "$JNI_DIR/libbreeze_server.so" "$ASSET_DIR"/libggml-htp-v*.so
