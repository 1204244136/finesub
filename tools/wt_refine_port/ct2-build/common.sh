#!/bin/bash
# Shared by the per-platform scripts: a static, trimmed oneDNN for the CPU GEMM.
#
# Same version and flags as the Windows build in ../ct2-patches/README.md, for
# the same reasons recorded there (Ruy deadlocks in the model destructor, MKL
# costs 3.4 GB and does not run on AMD CPUs). Sourced from the CTranslate2
# checkout root; installs into $ONEDNN_PREFIX.

ONEDNN_VERSION=3.12.3
ONEDNN_PREFIX="${ONEDNN_PREFIX:-$PWD/onednn-install}"

build_onednn() {
    local runtime="$1"   # OMP or SEQ
    shift
    git clone --depth 1 --branch "v${ONEDNN_VERSION}" https://github.com/oneapi-src/oneDNN onednn-src
    cmake -S onednn-src -B onednn-build "$@" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$ONEDNN_PREFIX" \
        -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_CPU_RUNTIME="$runtime" -DDNNL_GPU_RUNTIME=NONE \
        -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_DOC=OFF \
        -DONEDNN_BUILD_GRAPH=OFF -DDNNL_ENABLE_WORKLOAD=INFERENCE \
        -DDNNL_ENABLE_PRIMITIVE="CONVOLUTION;MATMUL;INNER_PRODUCT;REORDER" \
        -DDNNL_ENABLE_PRIMITIVE_CPU_ISA=ALL
    cmake --build onednn-build --config Release --parallel 3
    cmake --install onednn-build --config Release
    rm -rf onednn-src onednn-build
}

# How many GPU-kernel compile jobs run at once. nvcc / amdclang on these kernels
# take several GB each; at $(nproc) the 16 GB hosted runner was starved until it
# "lost communication with the server" (2026-09-17, both Linux variants, no log).
CT2_GPU_BUILD_JOBS="${CT2_GPU_BUILD_JOBS:-2}"

# The CUDA architecture list of every CUDA variant; bump together with
# device.CT2_MIN_COMPUTE_CAPABILITY and ../ct2-patches/README.md.
CT2_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# Flash attention in the CUDA variants (patch 0012 routes it to the Whisper
# encoder, 0013 compiles only the head dimension 64 kernels). The HIP variants
# cannot have it (CTranslate2's CMake refuses WITH_HIP with WITH_FLASH_ATTN).
# OFF is for bisecting: WITH_FLASH_ATTN also adds --expt-relaxed-constexpr and
# --expt-extended-lambda to every CUDA compile, so a build with it is not
# guaranteed to generate the other kernels identically.
CT2_WITH_FLASH_ATTN="${CT2_WITH_FLASH_ATTN:-ON}"

# The AMD targets of every HIP variant; must equal device.CT2_HIP_GFX_TARGETS
# (test_ct2_build_targets_match_the_device_layer holds the two together).
CT2_HIP_GFX_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
