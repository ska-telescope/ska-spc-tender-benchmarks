#!/bin/bash
source ${SPACK_ROOT}/share/spack/setup-env.sh
spack env deactivate

# Setting up a few path variables and preparing the askap repo dir.
ORGPATH=`pwd`
RELPATH=$(dirname "$0")
cd $RELPATH
SCRIPT_DIR=`pwd`
REPO_DIR=$SCRIPT_DIR/repo
ASKAP_DIR=$REPO_DIR/askap-benchmarks
cd $ORGPATH

# Getting the ASKAP benchmarks source files and applying a few modifications for compatibility with the latest NVHPC compiler.
if ! [ -d $ASKAP_DIR ]; then
    git clone --branch v1.0 https://github.com/ATNF/askap-benchmarks.git $ASKAP_DIR
    cd $ASKAP_DIR
    git apply $SCRIPT_DIR/compile_errors_correction.patch
fi

# Managing build dependencies through Spack.
spack install gcc@11
spack env activate $SCRIPT_DIR/spack
spack concretize -f
spack install

# Setting up the build files hierarchy.
cp -r $SCRIPT_DIR/cmake_files/* $ASKAP_DIR/current
if ! [ -d $SCRIPT_DIR/build ]; then
    mkdir $SCRIPT_DIR/build
fi
cd $SCRIPT_DIR/build

# Configuration and build.
cmake $ASKAP_DIR/current
cmake --build .