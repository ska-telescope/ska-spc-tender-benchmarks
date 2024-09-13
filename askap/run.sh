#!/bin/bash
source ${SPACK_ROOT}/share/spack/setup-env.sh

ORGPATH=`pwd`
RELPATH=$(dirname "$0")
cd $RELPATH
ASKAP_DIR=`pwd`/repo/askap-benchmarks
SCRIPT_DIR=`pwd`
BUILD_DIR=`pwd`/build
cd $ORGPATH

cp $ASKAP_DIR/data/dirty_1024.img dirty.img
cp $ASKAP_DIR/data/psf_1024.img psf.img

spack env activate ${SCRIPT_DIR}/spack
$BUILD_DIR/tConvolveACC >> results_tConvolveACC_$(date +"%d-%m-%y_%H:%M:%S")_$(hostname).log
$BUILD_DIR/tConvolveMPI >> results_tConvolveMPI_$(date +"%d-%m-%y_%H:%M:%S")_$(hostname).log
$BUILD_DIR/tHogbomCleanACC >> results_tHogbomCleanACC_$(date +"%d-%m-%y_%H:%M:%S")_$(hostname).log
$BUILD_DIR/tHogbomCleanOMP >> results_tHogbomCleanOMP_$(date +"%d-%m-%y_%H:%M:%S")_$(hostname).log
$BUILD_DIR/tMajorACC >> results_tMajorACC_$(date +"%d-%m-%y_%H:%M:%S")_$(hostname).log
