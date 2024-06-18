cmake_minimum_required(VERSION 3.25)

project(ska-spc-tender-benchmarks LANGUAGES CXX CUDA)

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

# Hardware options
option(ENABLE_CPU "Enable CPU benchmarks." ON)
option(ENABLE_GPU "Enable GPU benchmarks." ON)

find_package(MPI REQUIRED)
find_package(TBB REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_subdirectory(benchmarks-common)
add_subdirectory(fft)
add_subdirectory(gridding)
