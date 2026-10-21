#!/usr/bin/env bash
# Standalone Android MOSS engine. Never modifies the pinned working DiT fork.
set -euo pipefail
: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
cd "$(dirname "$0")"
UPSTREAM="../3rdparty/moss-stable-diffusion.cpp"
EXPECTED=95b3090664d019fcf8fa032a44d1d4793ba244b8
ACTUAL="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "Unexpected MOSS upstream commit $ACTUAL (required $EXPECTED)" >&2
    exit 1
fi
test -f "$UPSTREAM/ggml/CMakeLists.txt" || {
    echo "Missing nested ggml submodule. git submodule update --init --recursive" >&2
    exit 1
}
cmake -S . -B build/android -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10 "$@"
cmake --build build/android --target moss_engine -j "$(nproc)"
JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
mkdir -p "$JNI_DIR"
cp build/android/lib/arm64-v8a/libmoss_engine.so "$JNI_DIR/"
# DSP skels are optional source outputs but mandatory for HTP deployment.
# Reuse the established ditlibs packaging path; never silently replace a
# working v79/v81 skeleton with an unverified new one.
# Match the MOSS ggml DSP skels to their own AP library; old DiT skels
# remain untouched and continue to serve Qwen/Krea/FLUX.
MOSS_LIBS="$(cd ../.. && pwd)/assets/mosslibs"
mkdir -p "$MOSS_LIBS"
for arch in v79 v81; do
    source_so="$BUILD_DIR/moss-sdcpp/ggml/src/ggml-hexagon/libggml-htp-$arch.so"
    test -s "$source_so"
    cp "$source_so" "$MOSS_LIBS/"
done
printf '%s\n' "$EXPECTED" > "$MOSS_LIBS/upstream-revision.txt"
echo "MOSS matching v79/v81 Hexagon skels staged separately."
