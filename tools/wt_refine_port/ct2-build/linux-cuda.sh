#!/bin/bash
# Patched CTranslate2 for Linux x86_64 + CUDA 12.8 (cibuildwheel before-all,
# manylinux_2_28, run from the patched CTranslate2 checkout root).
#
# The Linux twin of the Windows build: CUDA loaded at run time, the same
# architecture list, static oneDNN instead of MKL/Ruy, flash attention
# (CT2_WITH_FLASH_ATTN). OpenMP is the compiler's
# (GNU) here, where Windows uses Intel's; both are what the stock wheels ship.
set -euxo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/common.sh"

pip install "cmake==3.31.*"
dnf install -y dnf-plugins-core git
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-8 cuda-cudart-devel-12-8 libcublas-devel-12-8 libcurand-devel-12-8
ln -sfn cuda-12.8 /usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

ONEDNN_PREFIX=/usr/local build_onednn OMP

cmake -S . -B build-release \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="-msse4.1" \
    -DBUILD_CLI=OFF -DOPENMP_RUNTIME=COMP \
    -DWITH_MKL=OFF -DWITH_RUY=OFF -DWITH_DNNL=ON \
    -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_DYNAMIC_LOADING=ON \
    -DWITH_FLASH_ATTN="$CT2_WITH_FLASH_ATTN" \
    -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" \
    -DCUDA_ARCH_LIST="$CT2_CUDA_ARCH_LIST"
cmake --build build-release --parallel "$CT2_GPU_BUILD_JOBS" --target install
rm -rf build-release
cp README.md python/
