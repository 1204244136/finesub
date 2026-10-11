#!/bin/bash
# Patched CTranslate2 for Linux x86_64 + AMD ROCm/HIP (cibuildwheel before-all,
# manylinux_2_28, run from the patched CTranslate2 checkout root).
#
# Follows upstream's own ROCm wheel recipe (python/tools/prepare_build_environment_linux_rocm.sh
# at v4.8.1): ROCm 7.2 from repo.radeon.com, amdclang, WITH_HIP, which defines
# CT2_WITH_CUDA too -- so patch 0006's `#ifdef CT2_WITH_CUDA` applies and 0003's
# median_filter_gpu.cu is compiled as HIP. The CPU side is ours (static oneDNN,
# no MKL/Ruy), and the target list is CT2_HIP_GFX_TARGETS.
#
# The ROCm libraries are not bundled (auditwheel excludes /opt/rocm): the wheel
# uses the system ROCm, which is the double-runtime question the plan records
# against torch's own bundled ROCm.
set -euxo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/common.sh"

# The runner image is too small for ROCm plus a build without this.
rm -rf /host/usr/local/lib/android /host/usr/local/.ghcup /host/usr/share/dotnet \
    /host/usr/share/swift /host/opt/hostedtoolcache/CodeQL /host/usr/local/share/powershell || true
df -h

export LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14:${LIBRARY_PATH:-}"
tee /etc/yum.repos.d/rocm.repo <<'EOF'
[rocm]
name=ROCm 7.2.0 repository
baseurl=https://repo.radeon.com/rocm/el8/7.2/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[amdgraphics]
name=AMD Graphics 7.2.0 repository
baseurl=https://repo.radeon.com/graphics/7.2/el/8/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
dnf clean all
dnf install -y rocm-hip-sdk git

export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/amdgcn/bitcode
export HIP_CLANG_ROOT=$ROCM_PATH/lib/llvm

pip install "cmake==3.31.*"
ONEDNN_PREFIX=/usr/local build_onednn OMP

cmake -S . -B build-release \
    -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="-msse4.1 -Wno-deprecated-literal-operator" \
    -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator" \
    -DBUILD_CLI=OFF -DOPENMP_RUNTIME=COMP \
    -DWITH_MKL=OFF -DWITH_RUY=OFF -DWITH_DNNL=ON \
    -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$CT2_HIP_GFX_TARGETS"
cmake --build build-release --parallel "$CT2_GPU_BUILD_JOBS" --target install
rm -rf build-release
cp README.md python/
