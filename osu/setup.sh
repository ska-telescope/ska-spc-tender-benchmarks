#!/bin/bash

set -e
#set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OSU_SRC_DIR="${SCRIPT_DIR}"
BUILD_DIR="build"
SRC_DIR="sources"
OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz"
OSU_DIR="osu-micro-benchmarks-7.4"
OSU_CHECKSUM="1edd0c2efa61999409bfb28740a7f39689a5b42b1a1b4c66d1656e5637f7cefc"
OSU_SHA256="test"
NPROC="$(cat /proc/cpuinfo | grep processor | wc -l)"



source ${OSU_SRC_DIR}/../config/dispatcher.sh

set +e
load_compiler
load_mpi
set -e

mkdir -p "${SRC_DIR}"
pushd "${SRC_DIR}"

curl "${OSU_URL}" -o osu_benches.tar.gz
echo  "${OSU_CHECKSUM} osu_benches.tar.gz" > osu.chksum
sha256sum -c osu.chksum

tar xvf osu_benches.tar.gz

mkdir -p "${BUILD_DIR}"; cd "${BUILD_DIR}";


echo " ** Configure and build"

#make -C ../${OSU_DIR}  distclean
export CC=mpicc
export CXX=mpic++
../${OSU_DIR}/configure

make -C  c/mpi/collective -j ${NPROC}
make -C c/mpi/pt2pt -j ${NPROC}

