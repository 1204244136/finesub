#!/bin/bash
# Patched CTranslate2 for macOS arm64, CPU only (cibuildwheel before-all, run
# from the patched CTranslate2 checkout root).
#
# Upstream's arm64 recipe is Accelerate + Ruy with no OpenMP. Ruy is the
# backend whose model destructor never returns on Windows
# (../ct2-patches/README.md), and this pipeline decodes on the CPU in float32
# only, which Accelerate's SGEMM covers -- so Ruy is left out. Whether that
# holds is exactly what the workflow's destructor smoke test checks: decode
# once, delete the model, and it must come back within a time limit.
set -euxo pipefail

cmake -S . -B build-release \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="${CTRANSLATE2_ROOT:-/usr/local}" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DBUILD_CLI=OFF -DOPENMP_RUNTIME=NONE \
    -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DWITH_RUY=OFF
cmake --build build-release --parallel "$(sysctl -n hw.physicalcpu_max)"
sudo cmake --install build-release
sudo rm -rf build-release
cp README.md python/
