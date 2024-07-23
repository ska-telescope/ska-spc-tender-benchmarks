#!/bin/bash
set -e
source ${SPACK_ROOT}/share/spack/setup-env.sh

# Setting up a few path variables and preparing the askap repo dir.
ORGPATH=`pwd`
RELPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $RELPATH
SCRIPT_DIR=`pwd`
REPO_DIR=$SCRIPT_DIR/repo
ASKAP_DIR=$REPO_DIR/askap-benchmarks
cd $ORGPATH

# Searching locally installed dependencies
spack external find nvhpc openmpi fftw cmake cuda

# Managing build dependencies through Spack.
spack env activate $SCRIPT_DIR/spack
spack concretize -f
spack install

# Getting the ASKAP benchmarks source files and applying a few modifications for compatibility with the latest NVHPC compiler.
if ! [ -d $ASKAP_DIR ]; then
    git clone --branch v1.0 https://github.com/ATNF/askap-benchmarks.git $ASKAP_DIR
    cd $ASKAP_DIR
    git apply $SCRIPT_DIR/compile_errors_correction.patch
fi

# Setting up the build files hierarchy.
if ! [ -d $SCRIPT_DIR/build ]; then
    mkdir $SCRIPT_DIR/build
fi
cd $SCRIPT_DIR/build

# Build each benchmark.
cd ${REPO_DIR}/askap-benchmarks/current/tHogbomCleanACC/
make clean && make
cp ${REPO_DIR}/askap-benchmarks/current/tHogbomCleanACC/tHogbomCleanACC $SCRIPT_DIR/build/

cd ${REPO_DIR}/askap-benchmarks/current/tConvolveACC/
make clean && make
cp ${REPO_DIR}/askap-benchmarks/current/tConvolveACC/tConvolveACC $SCRIPT_DIR/build/

cd ${REPO_DIR}/askap-benchmarks/current/tHogbomCleanOMP/
make clean && make
cp ${REPO_DIR}/askap-benchmarks/current/tHogbomCleanOMP/tHogbomCleanOMP $SCRIPT_DIR/build/

cd ${REPO_DIR}/askap-benchmarks/current/tMajorACC/
make clean && make
cp ${REPO_DIR}/askap-benchmarks/current/tMajorACC/tMajorACC $SCRIPT_DIR/build/

cd ${REPO_DIR}/askap-benchmarks/current/tConvolveMPI/
make clean && make
cp ${REPO_DIR}/askap-benchmarks/current/tConvolveMPI/tConvolveMPI $SCRIPT_DIR/build/
