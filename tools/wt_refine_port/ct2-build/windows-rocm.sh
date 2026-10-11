#!/bin/bash
# Patched CTranslate2 for Windows x64 + AMD ROCm/HIP (cibuildwheel before-all,
# run with bash from the patched CTranslate2 checkout root).
#
# Follows upstream's Windows ROCm recipe (python/tools/prepare_build_environment_windows_rocm.sh
# at v4.8.1): the ROCm SDK comes as pip wheels, the build is Ninja + ROCm's
# clang with WITH_HIP. The CPU side is ours (static trimmed oneDNN, Intel
# OpenMP from pip, no MKL/Ruy). PyTorch's ROCm build on Windows is a preview
# that covers RX 7000/9000 on Windows 11; this wheel's target list is wider,
# and a card only works when both sides have it.
set -euxo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/common.sh"

python -m pip install --no-cache-dir ninja \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl \
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
rocm-sdk init

python -m pip download intel-openmp --only-binary :all: --no-deps -d iomp-wheel
python - <<'PY'
import pathlib, shutil, zipfile
wheel = next(pathlib.Path("iomp-wheel").glob("*.whl"))
target = pathlib.Path("iomp")
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        for kind in ("lib", "bin"):
            if f"/Library/{kind}/" in name and name.endswith((".lib", ".dll")):
                out = target / kind / pathlib.PurePosixPath(name).name
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, open(out, "wb") as sink:
                    shutil.copyfileobj(source, sink)
PY
IOMP="$(pwd -W 2>/dev/null || pwd)/iomp"
ONEDNN_PREFIX="$(pwd -W 2>/dev/null || pwd)/onednn-install" build_onednn OMP -G "Visual Studio 17 2022"
ONEDNN_PREFIX="$(pwd -W 2>/dev/null || pwd)/onednn-install"

export ROCM_PATH=$(python -c "from rocm_sdk._devel import get_devel_root;print(get_devel_root().as_posix())")
export PATH="$ROCM_PATH:$PATH"
export CC="$ROCM_PATH/lib/llvm/bin/clang.exe"
export CXX="$ROCM_PATH/lib/llvm/bin/clang++.exe"
export HIP_PLATFORM="amd"
export HIP_PATH="$ROCM_PATH"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
export HIP_CLANG_ROOT="$ROCM_PATH/lib/llvm"

cmake -GNinja -S . -B build-hip \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
    -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator" \
    -DCMAKE_INSTALL_PREFIX="$CTRANSLATE2_ROOT" \
    -DOPENMP_RUNTIME=INTEL -DINTEL_ROOT="$IOMP" \
    -DWITH_MKL=OFF -DWITH_RUY=OFF -DWITH_DNNL=ON \
    -DDNNL_INCLUDE_DIR="$ONEDNN_PREFIX/include" \
    -DDNNL_LIBRARY="$ONEDNN_PREFIX/lib/dnnl.lib" \
    -DBUILD_CLI=OFF \
    -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$CT2_HIP_GFX_TARGETS"
cmake --build build-hip --config Release --target install --parallel
rm -rf build-hip

cp README.md python/
cp "$CTRANSLATE2_ROOT/bin/ctranslate2.dll" python/ctranslate2/
cp "$IOMP/bin/libiomp5md.dll" python/ctranslate2/
