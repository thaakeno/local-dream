#!/usr/bin/env bash
# Native YuE2 text-to-music runtime for Local Dream.
# Configure yue2.cpp as the top-level project (its build intentionally uses
# CMAKE_SOURCE_DIR), while pointing its documented GGML_SOURCE_DIR hook at the
# exact Local Dream GGML/Hexagon tree already used by Qwen.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"

cd "$(dirname "$0")"
BUILD_DIR=build/android
YUE2_DIR="$(cd ../3rdparty/yue2.cpp && pwd)"
SDCPP_DIR="$(cd ../3rdparty/stable-diffusion.cpp && pwd)"
GGML_DIR="$SDCPP_DIR/ggml"

cmake -S "$YUE2_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DGGML_SOURCE_DIR="$GGML_DIR" \
    -DGGML_HEXAGON=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10

# yue-server links the host Hexagon backend. Build the two DSP skels used by
# modern Snapdragon parts in the same graph, then stage everything into the
# locations Local Dream already puts on LD/ADSP_LIBRARY_PATH.
cmake --build "$BUILD_DIR" --target yue-server htp-v79 htp-v81 -j "$(nproc)"

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/ditlibs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"
cp "$BUILD_DIR/yue-server" "$JNI_DIR/libyue2_server.so"
cp "$BUILD_DIR"/ggml/src/ggml-hexagon/libggml-htp-v79.so \
   "$BUILD_DIR"/ggml/src/ggml-hexagon/libggml-htp-v81.so \
   "$ASSET_DIR/"

chmod +x "$JNI_DIR/libyue2_server.so"
ls -lh "$JNI_DIR/libyue2_server.so" "$ASSET_DIR"/libggml-htp-v{79,81}.so
