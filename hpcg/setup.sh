#!/bin/bash

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export HPCG_SRC_DIR="${SCRIPT_DIR}"
export BUILD_DIR="build"


spack env create hpcg || true
spack env activate hpcg
spack add cmake gcc ninja 
spack install

mkdir -p "${BUILD_DIR}"; cd "${BUILD_DIR}";
cmake -G Ninja ${HPCG_SRC_DIR}/

ninja -v

cp "_deps/hpcg-reference-src/bin/hpcg.dat" ./

echo "to run : 'cd ${BUILD_DIR}; ./_deps/hpcg-reference-build/xhpcg' "


