cmake_minimum_required(VERSION 3.22)

project(ska-spc-tender-benchmarks LANGUAGES CXX CUDA)


# Hardware options
option(ENABLE_CPU "Enable CPU benchmarks" ON)
option(ENABLE_GPU "Enable GPU benchmarks" ON)

# bench options
option(ENABLE_BENCH_FFT "Enable the benchmarks related to FFT" ON)
option(ENABLE_BENCH_GRIDDING "Enable the benchmarks related to Gridding" ON)


# parallelization options
#
# Parallelization management Adds the right dependencies and defines depending
# on the parallelization library.
#
set(PARALLELIZATION_LIBRARIES "OMP" "TBB")
set(PARALLELIZATION_LIBRARY
    "OMP"
    CACHE
      STRING
      "Parallelization library to use for CPU benchmarks. May be TBB, OMP or NONE."
)
set_property(CACHE PARALLELIZATION_LIBRARY PROPERTY STRINGS ${PARALLELIZATION_LIBRARIES})

if(NOT PARALLELIZATION_LIBRARY MATCHES "^(OMP|TBB|NONE)$")
  message(
    FATAL_ERROR
      "The PARALLELIZATION_LIBRARY parameter must be set to either OMP, TBB or NONE."
  )
endif()


set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VTUNE_PROFILE "Use Vtune annotations" ON)
if(VTUNE_PROFILE)
  include(FetchContent)
  FetchContent_Declare(
    ittnotify
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/intel/ittapi/archive/v3.23.0.zip
    URL_HASH
      SHA256=9456a76fcf9f30d77674b751e206b89a00cc0367ef3fadfe802b93519105793b)
  FetchContent_MakeAvailable(ittnotify)
endif()


find_package(MPI REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_subdirectory(benchmarks-common)

if(ENABLE_BENCH_FFT)
  add_subdirectory(fft)
endif()

if(ENABLE_BENCH_GRIDDING)
add_subdirectory(gridding)
endif()