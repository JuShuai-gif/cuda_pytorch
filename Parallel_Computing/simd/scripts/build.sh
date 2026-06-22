#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

ARCH=$(uname -m)
NPROC=$(nproc)

echo "============================================"
echo " SIMD Tutorial - Build Script"
echo "============================================"
echo " Platform:       $(uname -s) $(uname -r)"
echo " Architecture:   $ARCH"
echo " CPUs:           $NPROC"
echo " Build dir:      $BUILD_DIR"
echo "============================================"

BUILD_ARM="OFF"
BUILD_ARM_SVE="OFF"
BUILD_X86="OFF"
BUILD_X86_AVX512="OFF"
CROSS_FLAGS=""

case "$ARCH" in
    aarch64|arm64|armv8*|armv9*)
        BUILD_ARM="ON"
        echo "Detected ARM platform -> enabling ARM builds"

        # Check for SVE support via /proc/cpuinfo or lscpu
        if grep -q -E "^Features.*\bsve\b" /proc/cpuinfo 2>/dev/null || \
           lscpu 2>/dev/null | grep -qi "sve"; then
            BUILD_ARM_SVE="ON"
            echo "SVE detected -> enabling SVE builds"
        else
            echo "SVE not detected -> disabling SVE builds (cross-compile: set BUILD_ARM_SVE=ON manually)"
        fi
        ;;

    x86_64|amd64)
        BUILD_X86="ON"
        echo "Detected x86_64 platform -> enabling x86 builds"

        # Check for AVX-512 support
        if grep -q -E "^flags.*\bavx512f\b" /proc/cpuinfo 2>/dev/null; then
            BUILD_X86_AVX512="ON"
            echo "AVX-512F detected -> enabling AVX-512 builds"
        else
            echo "AVX-512F not detected -> disabling AVX-512 builds (cross-compile: set BUILD_X86_AVX512=ON manually)"
        fi
        ;;

    *)
        echo "WARNING: Unrecognized architecture '$ARCH'."
        echo "  Set BUILD_ARM=ON / BUILD_X86=ON manually if cross-compiling."
        ;;
esac

mkdir -p "$BUILD_DIR"

echo ""
echo "Running cmake..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_ARM="$BUILD_ARM" \
    -DBUILD_ARM_SVE="$BUILD_ARM_SVE" \
    -DBUILD_X86="$BUILD_X86" \
    -DBUILD_X86_AVX512="$BUILD_X86_AVX512" \
    ${CROSS_FLAGS}

echo ""
echo "Running make -j$NPROC ..."
make -C "$BUILD_DIR" -j"$NPROC"

echo ""
echo "============================================"
echo " Build Summary"
echo "============================================"
echo " ARM (NEON):     $BUILD_ARM"
echo " ARM (SVE):      $BUILD_ARM_SVE"
echo " x86 (AVX2):     $BUILD_X86"
echo " x86 (AVX-512):  $BUILD_X86_AVX512"

if [ "$BUILD_ARM" = "ON" ]; then
    echo "--- ARM binaries ---"
    find "$BUILD_DIR/arm" -maxdepth 1 -type f -executable 2>/dev/null | sort || echo "  (none)"
fi

if [ "$BUILD_X86" = "ON" ]; then
    echo "--- x86 binaries ---"
    find "$BUILD_DIR/x86" -maxdepth 1 -type f -executable 2>/dev/null | sort || echo "  (none)"
fi

echo "============================================"
echo "Build complete."
