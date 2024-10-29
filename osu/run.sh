#!/bin/bash

set -e
#set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OSU_SRC_DIR="${SCRIPT_DIR}"
BUILD_DIR="build"
RESULT_DIR="result"
SRC_DIR="sources"
MESSAGE_SIZE="4194304"

source ${OSU_SRC_DIR}/../config/dispatcher.sh

set +e
load_compiler
load_mpi
set -e

pushd "${SRC_DIR}"

pushd "${BUILD_DIR}";

mkdir -p "${RESULT_DIR}"

echo "** Benches"

srun ./c/mpi/pt2pt/standard/osu_latency -m "${MESSAGE_SIZE}" | tee -a ${RESULT_DIR}/latency.result
srun ./c/mpi/pt2pt/standard/osu_bw -m "${MESSAGE_SIZE}" | tee -a ${RESULT_DIR}/bw.result 
srun ./c/mpi/collective/blocking/osu_barrier | tee -a ${RESULT_DIR}/barrier.result 
srun ./sources/build/c/mpi/collective/blocking/osu_bcast -m "${MESSAGE_SIZE}" | tee -a ${RESULT_DIR}/barrier.result
