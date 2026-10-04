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

# Breeze's vocoder needs SIN and transpose-convolution semantics that current
# upstream Hexagon does not expose. Extend the pinned backend with two reviewed
# native HTP primitives. This is a normal git patch with a hard drift check,
# not a runtime source rewrite and not a CPU/GPU fallback.
cp "$(pwd)/native/breeze-sin-ops.c"     "$GGML_DIR/src/ggml-hexagon/htp/breeze-sin-ops.c"
cp "$(pwd)/native/breeze-col2im-ops.c"     "$GGML_DIR/src/ggml-hexagon/htp/breeze-col2im-ops.c"
git -C "$HEXAGON_DIR" apply --check "$(pwd)/hexagon-breeze.patch"
git -C "$HEXAGON_DIR" apply "$(pwd)/hexagon-breeze.patch"

grep -q 'HTP_OP_SIN' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'HTP_OP_COL2IM_1D' "$GGML_DIR/src/ggml-hexagon/htp/htp-ops.h"
grep -q 'GGML_OP_SIN' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"
grep -q 'GGML_OP_COL2IM_1D' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon.cpp"

rm -rf "$BUILD_DIR"

cmake -S "$(pwd)" -B "$BUILD_DIR" -G Ninja     -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-28     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE"     -DBREEZE_SOURCE_DIR="$BREEZE_DIR"     -DGGML_SOURCE_DIR="$GGML_DIR"     -DGGML_HEXAGON=ON     -DGGML_HEXAGON_USE_MEMPOOL=OFF     -DGGML_OPENMP=OFF     -DGGML_LLAMAFILE=OFF     -DGGML_BACKEND_DL=OFF     -DPREBUILT_LIB_DIR=android_aarch64     -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"     -DCMAKE_POLICY_VERSION_MINIMUM=3.10

cmake --build "$BUILD_DIR" --target breeze-server htp-v73 htp-v75 htp-v79 htp-v81 -j "$(nproc)"

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
mode=strict-htp-only
fallback=disabled
extensions=sin-hvx,col2im1d-htp,exact-elu-lowering,transpose-conv-gemm-col2im
formats=f16,q8_0,q6_k,q4_k,q8_0-dd4,q8_0-dd2,q4_k-dd2
EOF

ls -lh "$JNI_DIR/libbreeze_server.so" "$ASSET_DIR"/libggml-htp-v*.so
