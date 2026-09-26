#!/usr/bin/env bash
# Native YuE2 text-to-music runtime for Local Dream.
# Uses yue2.cpp's public GGML_SOURCE_DIR integration point and the same
# Hexagon backend/skels as the image DiT runtime. No source monkey patches.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"

cd "$(dirname "$0")"
BUILD_DIR=build/android

cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10

cmake --build "$BUILD_DIR" --target yue-server -j "$(nproc)"

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/ditlibs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"
cp "$BUILD_DIR/bin/arm64-v8a/libyue2_server.so" "$JNI_DIR/"

# yue2.cpp is built against the same Local Dream GGML tree as libdit_engine.
# Copy its freshly built skels too so this target is independently reproducible.
cp "$BUILD_DIR"/yue2/ggml/src/ggml-hexagon/libggml-htp-v79.so \
   "$BUILD_DIR"/yue2/ggml/src/ggml-hexagon/libggml-htp-v81.so \
   "$ASSET_DIR/"

ls -lh "$JNI_DIR/libyue2_server.so" "$ASSET_DIR"/libggml-htp-v{79,81}.so
