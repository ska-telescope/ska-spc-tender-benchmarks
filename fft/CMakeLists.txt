cmake_minimum_required(VERSION 3.23)

project(fft-benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RelWithDbgInfo)

add_executable(fft-benchmarks)

# Internal files and includes
target_include_directories(fft-benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_sources(fft-benchmarks PRIVATE inc/fft_configuration.h
                                      src/fft_configuration.cpp)
target_sources(fft-benchmarks PRIVATE inc/fft_benchmark.h src/fft_benchmark.cpp)
target_sources(fft-benchmarks PRIVATE src/main.cpp)

# Commonly required packages
find_package(Heffte REQUIRED)
find_package(MPI REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
# Linking external dependencies.
target_link_libraries(fft-benchmarks PUBLIC Heffte::Heffte)
target_link_libraries(fft-benchmarks PUBLIC MPI::MPI_CXX)
target_link_libraries(fft-benchmarks PUBLIC yaml-cpp)

# Hardware options
option(ENABLE_CPU "Enable CPU benchmarks." ON)
if(${ENABLE_CPU})
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_CPU)
endif()
option(ENABLE_GPU "Enable GPU benchmarks." ON)
if(${ENABLE_GPU})
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_GPU)
endif()

#
# Parallelization management Adds the right dependencies and defines depending
# on the parallelization library.
#
set(PARALLELIZATION_LIBRARY
    "OMP"
    CACHE
      STRING
      "Parallelization library to use for CPU benchmarks. May be TBB, OMP or NONE."
)
if(NOT (${PARALLELIZATION_LIBRARY} MATCHES "^(OMP|TBB|NONE)$"))
  message(
    FATAL_ERROR
      "The PARALLELIZATION_LIBRARY parameter must be set to either OMP, TBB or NONE."
  )
endif()

if(${PARALLELIZATION_LIBRARY} STREQUAL "TBB")
  find_package(TBB REQUIRED)
  target_link_libraries(fft-benchmarks PUBLIC TBB::tbb)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_TBB)
endif()

if(${PARALLELIZATION_LIBRARY} STREQUAL "OMP")
  find_package(OpenMP REQUIRED)
  target_link_libraries(fft-benchmarks PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_OMP)
endif()

# Optional annotations to restrict profiling to computation parts.
option(VTUNE_PROFILE "Use Vtune annotations" ON)
if(VTUNE_PROFILE)
  target_compile_definitions(fft-benchmarks PRIVATE VTUNE_PROFILE=1)
  # External ITT API library for VTune annotations.
  include(FetchContent)
  FetchContent_Declare(
    ittnotify
    DOWNLOAD_EXTRACT_TIMESTAMP true
    URL https://github.com/intel/ittapi/archive/v3.23.0.zip
    URL_HASH
      SHA256=9456a76fcf9f30d77674b751e206b89a00cc0367ef3fadfe802b93519105793b)
  FetchContent_MakeAvailable(ittnotify)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()
