#!/bin/bash
# Patched CTranslate2 for Windows x64 + CUDA 12.8 (cibuildwheel before-all,
# run with bash from the patched CTranslate2 checkout root).
#
# The CI form of the hand build in ../ct2-patches/README.md, flag for flag:
# Intel OpenMP from pip, static trimmed oneDNN, CUDA loaded at run time, the
# same architecture list, Visual Studio 2022, flash attention (CT2_WITH_FLASH_ATTN). It exists so the wheel can be
# rebuilt without the maintainer's machine; before a CI-built wheel replaces
# the hand-built one, its output is compared on the two parity fixtures
# (byte-exact first, else characterised as an execution-form change --
# README_DEV.md, development principles).
set -euxo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/common.sh"

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
curl -fsSL --retry 3 -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
./cuda.exe -s nvcc_12.8 cudart_12.8 cublas_dev_12.8 curand_dev_12.8
rm cuda.exe

# Intel OpenMP from pip, unpacked into the layout CT2's INTEL_ROOT lookup reads.
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
test -f "$IOMP/lib/libiomp5md.lib" && test -f "$IOMP/bin/libiomp5md.dll"

ONEDNN_PREFIX="$(pwd -W 2>/dev/null || pwd)/onednn-install" build_onednn OMP -G "Visual Studio 17 2022"
ONEDNN_PREFIX="$(pwd -W 2>/dev/null || pwd)/onednn-install"

cmake -S . -B build-cu -G "Visual Studio 17 2022" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$CTRANSLATE2_ROOT" \
    -DOPENMP_RUNTIME=INTEL -DINTEL_ROOT="$IOMP" \
    -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" -DCUDA_DYNAMIC_LOADING=ON \
    -DWITH_FLASH_ATTN="$CT2_WITH_FLASH_ATTN" \
    -DWITH_MKL=OFF -DWITH_RUY=OFF \
    -DWITH_DNNL=ON \
    -DDNNL_INCLUDE_DIR="$ONEDNN_PREFIX/include" \
    -DDNNL_LIBRARY="$ONEDNN_PREFIX/lib/dnnl.lib" \
    -DBUILD_CLI=OFF \
    -DCUDA_ARCH_LIST="$CT2_CUDA_ARCH_LIST"
cmake --build build-cu --config Release --parallel
cmake --install build-cu --config Release
rm -rf build-cu

cp README.md python/
cp "$CTRANSLATE2_ROOT/bin/ctranslate2.dll" python/ctranslate2/
cp "$IOMP/bin/libiomp5md.dll" python/ctranslate2/
