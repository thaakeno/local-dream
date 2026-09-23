#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
NDK_ROOT="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}"

if [[ -z "$NDK_ROOT" ]]; then
  echo "ANDROID_NDK_ROOT is required" >&2
  exit 1
fi

HOST_TAG="linux-x86_64"
if [[ "$(uname -s)" == "Darwin" ]]; then
  HOST_TAG="darwin-x86_64"
fi

TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/$HOST_TAG/bin"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$TOOLCHAIN/aarch64-linux-android28-clang"
export CC_aarch64_linux_android="$TOOLCHAIN/aarch64-linux-android28-clang"
export CXX_aarch64_linux_android="$TOOLCHAIN/aarch64-linux-android28-clang++"
export AR_aarch64_linux_android="$TOOLCHAIN/llvm-ar"

rustup target add aarch64-linux-android
cd "$SCRIPT_DIR"
cargo build --release --target aarch64-linux-android

OUT="$APP_DIR/src/main/jniLibs/arm64-v8a"
mkdir -p "$OUT"
cp "$SCRIPT_DIR/target/aarch64-linux-android/release/liblocaldream_xet.so" "$OUT/"
echo "Staged $OUT/liblocaldream_xet.so"
