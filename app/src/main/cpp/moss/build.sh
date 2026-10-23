#!/usr/bin/env bash
# Standalone Android MOSS engine. Never modifies the pinned working DiT fork.
set -euo pipefail
: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
cd "$(dirname "$0")"
# Identical Ninja directory for build and DSP skel staging.
BUILD_DIR=build/android-moss-fused-rope
# Same EXACT pinned happyyzy source and qknorm-capable ggml as the existing
# classic DiT engine. The checkout is NEVER modified: MOSS has its own copy,
# model patch stack, CMake cache, .so and Hexagon skels.
UPSTREAM="../3rdparty/stable-diffusion.cpp"
EXPECTED=a63954dc794f6daade32c407da1a9c4bc2d73cf0
ACTUAL="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "Unexpected MOSS upstream commit $ACTUAL (required $EXPECTED)" >&2
    exit 1
fi
test -f "$UPSTREAM/ggml/CMakeLists.txt" || {
    echo "Missing nested ggml submodule. git submodule update --init --recursive" >&2
    exit 1
}
# Use an immutable archive of the exact pinned upstream revision. Never alter
# the checked-out upstream submodule, shared DiT fork, or either ggml source.
# Stage exact pinned upstream and its ggml, then MOSS-only vetted source patches.
# No monkeypatching the checked-out submodule or other engines.
STAGED_SOURCE="$(pwd)/build/moss-sdcpp-staged"
rm -rf "$STAGED_SOURCE"
mkdir -p "$STAGED_SOURCE"
git -C "$UPSTREAM" archive "$EXPECTED" | tar -x -C "$STAGED_SOURCE"
# Git archive may emit an empty directory for the ggml gitlink; remove it
# before copying, otherwise cp could create ggml/ggml and break CMake.
rm -rf "$STAGED_SOURCE/ggml"
cp -a "$UPSTREAM/ggml" "$STAGED_SOURCE/ggml"
rm -f "$STAGED_SOURCE/ggml/.git"
test -f "$STAGED_SOURCE/ggml/CMakeLists.txt"
# Patches are mirrored verbatim with Nightmare attribution and applied ONLY
# inside the staged MOSS tree. 028 adds MOSS/Qwen3.5 to happyyzy's pinned
# DiT source; 035 builds MOSS attention from ggml's existing fused QKNORM_ROPE.
# Fuzz disabled; malformed or outdated patches fail the build.
for name in 028-nightmare-moss-image.sdcpp.patch 035-nightmare-moss-fused-rope.sdcpp.patch 036-moss-fast-tensor-loader-poll.patch; do
    echo "Applying MOSS-only upstream patch $name"
    patch --batch --forward --fuzz=0 -d "$STAGED_SOURCE" -p1 < "patches/$name"
done
python3 tests/test_single_htp_segmentation.py "$STAGED_SOURCE"
test -z "$(git -C "$UPSTREAM" status --porcelain --untracked-files=no)" || {
    echo "Pinned MOSS upstream checkout changed unexpectedly" >&2
    exit 1
}
cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DMOSS_SDCPP="$STAGED_SOURCE" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10 "$@"
cmake --build "$BUILD_DIR" --target moss_engine -j "$(nproc)"
JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
mkdir -p "$JNI_DIR"
test -s "$BUILD_DIR/lib/arm64-v8a/libmoss_engine.so"
cp "$BUILD_DIR/lib/arm64-v8a/libmoss_engine.so" "$JNI_DIR/"
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
