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

# Android 12+ can hide vendor libraries from an app's default linker namespace
# even when the device physically ships them. HyperOS 3 on SM8850 currently
# exhibits exactly that for libcdsprpc.so, which makes ggml-hexagon fail before
# it can create HTP0. Keep the normal dlopen first, then fall back to Android's
# exported vendor/sphal namespace. This preserves the device's own FastRPC/HAL
# stack instead of bundling a foreign Qualcomm blob.
GGML_LIBDL="$GGML_DIR/src/ggml-hexagon/libdl.h"
python3 - "$GGML_LIBDL" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
if "LOCAL_DREAM_ANDROID_VENDOR_NAMESPACE_FALLBACK" not in s:
    s = s.replace(
        "#    include <dlfcn.h>\n#    include <unistd.h>\n",
        "#    include <dlfcn.h>\n#    include <unistd.h>\n#    include <cstdio>\n"
        "#    ifdef __ANDROID__\n"
        "#        include <android/dlext.h>\n"
        "#    endif\n",
    )

    old = """static inline dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}
"""
    new = r"""// LOCAL_DREAM_ANDROID_VENDOR_NAMESPACE_FALLBACK
#ifdef __ANDROID__
using android_get_exported_namespace_fn = android_namespace_t * (*)(const char *);
using android_load_sphal_library_fn = void * (*)(const char *, int);

static inline dl_handle * dl_load_android_vendor_library(const char * soname) {
    dl_handle * handle = nullptr;

    // First use Android's own vendor-loader helper when it is reachable. AOSP
    // uses android_load_sphal_library() specifically to load vendor HAL-side
    // libraries from a non-vendor process while keeping their transitive
    // dependencies in the correct namespace.
    void * vndk_support = dlopen("libvndksupport.so", RTLD_NOW | RTLD_LOCAL);
    if (vndk_support != nullptr) {
        auto load_sphal = reinterpret_cast<android_load_sphal_library_fn>(
            dlsym(vndk_support, "android_load_sphal_library"));
        if (load_sphal != nullptr) {
            handle = reinterpret_cast<dl_handle *>(
                load_sphal(soname, RTLD_NOW | RTLD_LOCAL));
        }
        dlclose(vndk_support);
        if (handle != nullptr) {
            return handle;
        }
    }

    // Fallback for Android builds where libvndksupport is not app-visible:
    // resolve the exported namespace API dynamically, then ask the linker to
    // load libcdsprpc in sphal/vendor instead of the isolated default namespace.
    auto get_ns = reinterpret_cast<android_get_exported_namespace_fn>(
        dlsym(RTLD_DEFAULT, "android_get_exported_namespace"));

    void * libdl_android = nullptr;
    if (get_ns == nullptr) {
        libdl_android = dlopen("libdl_android.so", RTLD_NOW | RTLD_LOCAL);
        if (libdl_android != nullptr) {
            get_ns = reinterpret_cast<android_get_exported_namespace_fn>(
                dlsym(libdl_android, "android_get_exported_namespace"));
        }
    }

    if (get_ns != nullptr) {
        for (const char * ns_name : { "sphal", "vendor" }) {
            android_namespace_t * ns = get_ns(ns_name);
            if (ns == nullptr) {
                continue;
            }

            android_dlextinfo info{};
            info.flags = ANDROID_DLEXT_USE_NAMESPACE;
            info.library_namespace = ns;
            handle = reinterpret_cast<dl_handle *>(
                android_dlopen_ext(soname, RTLD_NOW | RTLD_LOCAL, &info));
            if (handle != nullptr) {
                break;
            }
        }
    }

    if (libdl_android != nullptr) {
        dlclose(libdl_android);
    }
    return handle;
}
#endif

static inline dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#ifdef __ANDROID__
    if (handle == nullptr && path.filename() == "libcdsprpc.so") {
        handle = dl_load_android_vendor_library("libcdsprpc.so");
        if (handle != nullptr) {
            fprintf(stderr, "ggml-hex: loaded libcdsprpc.so through Android vendor namespace fallback\n");
        }
    }
#endif
    return handle;
}
"""
    if old not in s:
        raise SystemExit("ggml-hexagon libdl.h shape changed; refusing silent patch")
    s = s.replace(old, new)
    p.write_text(s)
PY

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
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10

# yue-server links the Android host-side Hexagon backend. The workflow builds
# Local Dream's DiT engine immediately before this target, which already builds
# and stages the exact same v79/v81 DSP skels from the same GGML tree. Reusing
# those files avoids compiling the Hexagon ExternalProjects twice.
cmake --build "$BUILD_DIR" --target yue-server -j "$(nproc)"

JNI_DIR="$(cd ../.. && pwd)/jniLibs/arm64-v8a"
ASSET_DIR="$(cd ../.. && pwd)/assets/ditlibs"
mkdir -p "$JNI_DIR"
cp "$BUILD_DIR/yue-server" "$JNI_DIR/libyue2_server.so"

# Fail loudly if the shared HTP runtime was not staged by dit/build.sh.
test -s "$ASSET_DIR/libggml-htp-v79.so"
test -s "$ASSET_DIR/libggml-htp-v81.so"

chmod +x "$JNI_DIR/libyue2_server.so"
ls -lh "$JNI_DIR/libyue2_server.so" "$ASSET_DIR"/libggml-htp-v{79,81}.so
