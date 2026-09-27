#!/usr/bin/env bash
# Native YuE2 text-to-music runtime for Local Dream.
#
# YuE2 uses a dedicated, pinned FastRPC/mempool Hexagon backend instead of the
# image runtime's dspqueue backend. The reason is architectural: YuE2's score
# decoder has a ~151k-row LM head and batch-1 autoregressive decode. The
# per-buffer backend deliberately refuses very large quantized LM heads, which
# leaves the hottest projection on CPU. The mempool backend keeps repacked
# weights resident in one FastRPC pool and can execute that head on HTP.
#
# Both dependencies are pinned. Local Dream carries one narrow, version-locked
# backend fix for Android HTP: FastRPC must retry the *complete* mempool
# registration transaction when a large AP allocation does not fit in the DSP
# process VA window. The patch is checked against the exact pinned revision and
# the build fails on source drift; there is no runtime fallback or source
# rewriting.
set -euo pipefail

: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT}"
: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"

cd "$(dirname "$0")"
BUILD_DIR=build/android
YUE2_DIR="$(cd ../3rdparty/yue2.cpp && pwd)"

JZ_REPO="https://github.com/kan-linux/ggml-hexagon.git"
JZ_COMMIT="6485ca781502e57975053b6b82c09a3e9492731d"
JZ_ROOT="$(pwd)/build/deps/ggml-hexagon"
GGML_DIR="$JZ_ROOT/ggml"
PATCH_DIR="$(pwd)/patches"
MEMPOOL_PATCH="$PATCH_DIR/ggml-hexagon-adaptive-mempool.patch"

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
test -s "$MEMPOOL_PATCH"

# Always start from the pinned dependency tree, then apply the reviewed patch
# as a normal Git patch. --check makes dependency drift a hard build failure
# instead of silently producing a different backend.
git -C "$JZ_ROOT" reset --hard "$JZ_COMMIT"
git -C "$JZ_ROOT" clean -fdx
git -C "$JZ_ROOT" apply --check "$MEMPOOL_PATCH"
git -C "$JZ_ROOT" apply "$MEMPOOL_PATCH"
grep -q 'rpc mempool selected:' "$GGML_DIR/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp"

rm -rf "$BUILD_DIR"

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
    -DGGML_HEXAGON_USE_MEMPOOL=ON \
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
cmake --build "$BUILD_DIR" --target yue-server htp-mempool-v79 htp-mempool-v81 -j "$(nproc)"

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/yue2libs"
mkdir -p "$JNI_DIR" "$ASSET_DIR"

cp "$BUILD_DIR/yue-server" "$JNI_DIR/libyue2_server.so"

# The mempool AP backend and DSP skel are a matched pair. Keep YuE2's skel in
# its own asset directory so image-generation runtimes can continue using their
# independently-built Hexagon backend without binary collisions.
for arch in v79 v81; do
    skel="$(find "$BUILD_DIR" -type f -name "libggml-htp-${arch}.so" -print -quit)"
    if [[ -z "$skel" || ! -s "$skel" ]]; then
        echo "Missing YuE2 mempool HTP skel for $arch" >&2
        exit 1
    fi
    cp "$skel" "$ASSET_DIR/libggml-htp-${arch}.so"
done

# Ship the backend's production defaults next to its DSP skels. Local Dream
# launches yue-server with this directory as cwd so the backend loads this file
# directly; no runtime source mutation or magic environment translation.
cp "$JZ_ROOT/scripts/ggml-hexagon.cfg" "$ASSET_DIR/ggml-hexagon.cfg"

cat > "$ASSET_DIR/backend-version.txt" <<EOF
backend=kan-linux/ggml-hexagon
commit=$JZ_COMMIT
variant=fastrpc-mempool
patchset=localdream-adaptive-mempool-v1
EOF

chmod +x "$JNI_DIR/libyue2_server.so"
ls -lh "$JNI_DIR/libyue2_server.so" "$ASSET_DIR"/libggml-htp-v{79,81}.so "$ASSET_DIR/ggml-hexagon.cfg"
