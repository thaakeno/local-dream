#!/usr/bin/env bash
# Native YuE2 text-to-music runtime for Local Dream.
#
# YuE2 uses a dedicated, pinned Hexagon DSPQueue backend. The previous JZ
# single-mempool transport could reserve ~4 GiB on the Android side but then
# fail when the SM8850 DSP process tried to map that same region. DSPQueue uses
# normal per-buffer mappings instead of one giant shared VA reservation, so it
# avoids that failure mode without source patches or CPU compatibility paths.
#
# The backend is built directly from one reviewed upstream commit. Local Dream
# does not rewrite ggml-hexagon source and ships the matching AP/DSP binaries.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"

cd "$(dirname "$0")"
BUILD_DIR=build/android
FAST_BUILD_DIR=build/android-fastrpc
YUE2_DIR="$(cd ../3rdparty/yue2.cpp && pwd)"

JZ_REPO="https://github.com/kan-linux/ggml-hexagon.git"
JZ_COMMIT="37f752b77fc4d661e2552128f42cb811d7bdb628"
JZ_ROOT="$(pwd)/build/deps/ggml-hexagon"
GGML_DIR="$JZ_ROOT/ggml"

# Fetch exactly one reviewed backend revision. This is a normal pinned
# dependency, not a source rewrite/monkey patch.
if [[ ! -d "$JZ_ROOT/.git" ]]; then
    rm -rf "$JZ_ROOT"
    mkdir -p "$(dirname "$JZ_ROOT")"
    git init -q "$JZ_ROOT"
    git -C "$JZ_ROOT" remote add origin "$JZ_REPO"
fi

if [[ "$(git -C "$JZ_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$JZ_COMMIT" ]]; then
    git -C "$JZ_ROOT" fetch -q --depth 1 origin "$JZ_COMMIT"
    git -C "$JZ_ROOT" checkout -q --detach FETCH_HEAD
fi

test "$(git -C "$JZ_ROOT" rev-parse HEAD)" = "$JZ_COMMIT"
test -f "$GGML_DIR/CMakeLists.txt"

# Start from the exact pinned tree, then compile Local Dream's maintained YuE2
# HTP integration into both the AP backend and matched DSP skel. The integration
# is fail-on-drift and there is no runtime rewrite or CPU compatibility path.
git -C "$JZ_ROOT" reset --hard "$JZ_COMMIT"
git -C "$JZ_ROOT" clean -fdx
test -f "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q '#include <dspqueue.h>' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
cp "$(pwd)/native/yue-transport-bench.cpp" "$YUE2_DIR/tools/yue-transport-bench.cpp"
python3 "$(pwd)/integrate_htp_yue2.py" "$JZ_ROOT" "$YUE2_DIR" "$(pwd)/native"
grep -q 'HTP_OP_COL2IM_1D' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_COL2IM_1D_BIAS' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_SNAKE' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_ADD' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_CHANNEL_BCAST_MUL' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'op_snake' "$GGML_DIR/src/ggml-hexagon/htp/main.c"
grep -q 'op_col2im_1d_bias' "$GGML_DIR/src/ggml-hexagon/htp/main.c"
grep -q 'op_channel_bcast_add' "$GGML_DIR/src/ggml-hexagon/htp/main.c"
grep -q 'op_channel_bcast_mul' "$GGML_DIR/src/ggml-hexagon/htp/main.c"
grep -q 'YUE2_CHANNEL_ADD_CHUNK_ELEMS' "$GGML_DIR/src/ggml-hexagon/htp/channel-bcast-add-ops.c"
grep -q 'YUE2_CHANNEL_BINARY_MUL' "$GGML_DIR/src/ggml-hexagon/htp/channel-bcast-add-ops.c"
grep -q 'ggml_hexagon_is_yue2_channel_bcast' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'GGML_OP_SIN' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'YUE2_STRICT_ACCELERATOR' "$YUE2_DIR/src/backend.h"
grep -q 'Adaptive generic binary VTCM thread fit' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'col2im_fast_channel' "$(pwd)/native/col2im-ops.c"
grep -q 'op_col2im_1d_bias' "$(pwd)/native/col2im-ops.c"
grep -q 'YUE2_COL2IM_BIAS' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'YUE2_SNAKE' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'op_snake' "$(pwd)/native/snake-ops.c"
grep -q 'ode_method' "$YUE2_DIR/src/request.h"
grep -q 'dpmpp_2m' "$YUE2_DIR/src/nar.h"
grep -q 'COMFY_DPM_PLUS_PLUS_2M_SGM_UNIFORM' "$YUE2_DIR/src/nar.h"
grep -q '"sgm_uniform"' "$YUE2_DIR/src/nar.h"
grep -q 'GGML_PAD(max_kv_len, 64)' "$YUE2_DIR/src/qwen3-lm.h"
grep -q 'GGML_PAD(kv_len, 64)' "$YUE2_DIR/src/qwen3-lm.h"
grep -q 'r.ode_method.c_str()' "$YUE2_DIR/src/pipeline.h"

rm -rf "$BUILD_DIR" "$FAST_BUILD_DIR"

cmake -S "$YUE2_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PROJECT_INCLUDE="$(pwd)/android-yue2-deps.cmake" \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DGGML_SOURCE_DIR="$GGML_DIR" \
    -DGGML_HEXAGON=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10

# ExternalProject skels are not guaranteed to be pulled in by the host
# executable target, so build the two Android architectures Local Dream ships
# explicitly. This is still one matched source/backend revision.
cmake --build "$BUILD_DIR" --target yue-server yue-transport-bench htp-v79 htp-v81 -j "$(nproc)"

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/yue2libs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"

cp "$BUILD_DIR/yue-server" "$JNI_DIR/libyue2_server.so"
cp "$BUILD_DIR/yue-transport-bench" "$JNI_DIR/libyue2_bench_dspqueue.so"

# Keep YuE2's DSPQueue skels in their own asset directory so image-generation
# runtimes can continue using their independently-built Hexagon runtime without
# binary collisions.
for arch in v79 v81; do
    skel="$(find "$BUILD_DIR" -type f -name "libggml-htp-${arch}.so" -print -quit)"
    if [[ -z "$skel" || ! -s "$skel" ]]; then
        echo "Missing YuE2 mempool HTP skel for $arch" >&2
        exit 1
    fi
    cp "$skel" "$ASSET_DIR/libggml-htp-${arch}.so"
    cp "$skel" "$ASSET_DIR/libggml-htp-${arch}-dspqueue.so"
done

# Build the exact same benchmark graph against the coherent FastRPC/mempool
# AP+DSP pair from the same pinned source. It is staged separately and is never
# mixed with the DSPQueue generation runtime.
cmake -S "$YUE2_DIR" -B "$FAST_BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PROJECT_INCLUDE="$(pwd)/android-yue2-deps.cmake" \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DGGML_SOURCE_DIR="$GGML_DIR" \
    -DGGML_HEXAGON=ON \
    -DGGML_HEXAGON_USE_MEMPOOL=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10

cmake --build "$FAST_BUILD_DIR" --target yue-server yue-transport-bench htp-mempool-v79 htp-mempool-v81 -j "$(nproc)"
cp "$FAST_BUILD_DIR/yue-server" "$JNI_DIR/libyue2_server_fastrpc.so"
cp "$FAST_BUILD_DIR/yue-transport-bench" "$JNI_DIR/libyue2_bench_fastrpc.so"

for arch in v79 v81; do
    fast_skel="$(find "$FAST_BUILD_DIR" -type f -name "libggml-htp-${arch}.so" -print -quit)"
    if [[ -z "$fast_skel" || ! -s "$fast_skel" ]]; then
        echo "Missing YuE2 FastRPC HTP skel for $arch" >&2
        exit 1
    fi
    cp "$fast_skel" "$ASSET_DIR/libggml-htp-${arch}-fastrpc.so"
done

# DSPQueue is configured via GGML_HEXAGON_* environment variables. Remove any
# stale mempool config generated by an older native build.
rm -f "$ASSET_DIR/ggml-hexagon.cfg"

cat > "$ASSET_DIR/backend-version.txt" <<EOF
backend=kan-linux/ggml-hexagon
commit=$JZ_COMMIT
variant=dual-transport-yue2-native-0.7.0
integration=snake-hvx-fused,sin-hvx,col2im1d-htp,col2im-bias-fused,col2im-channel-blocked,channel-bcast-addmul-hvx,adaptive-binary-vtcm,dpmpp2m-sgm-uniform,htp-kv-window64,vae-cpu-parity-gate,dual-transport-server,dual-transport-bench,strict-accelerator
cpu_fallback=disabled
EOF

chmod +x "$JNI_DIR/libyue2_server.so" "$JNI_DIR/libyue2_server_fastrpc.so" "$JNI_DIR/libyue2_bench_dspqueue.so" "$JNI_DIR/libyue2_bench_fastrpc.so"
ls -lh "$JNI_DIR/libyue2_server.so" "$JNI_DIR/libyue2_server_fastrpc.so" "$JNI_DIR"/libyue2_bench_*.so \
    "$ASSET_DIR"/libggml-htp-v{79,81}.so "$ASSET_DIR"/libggml-htp-v{79,81}-fastrpc.so
