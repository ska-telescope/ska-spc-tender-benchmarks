cmake_minimum_required(VERSION 3.23)

project(fft-benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(fft-benchmarks)

find_package(MPI REQUIRED)
find_package(TBB REQUIRED)

find_package(yaml-cpp CONFIG REQUIRED)
target_include_directories(fft-benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_sources(fft-benchmarks PRIVATE inc/fft_configuration.h src/fft_configuration.cpp)
target_sources(fft-benchmarks PRIVATE inc/fft_benchmark.h src/fft_benchmark.cpp)
target_sources(fft-benchmarks PRIVATE src/main.cpp)
target_link_libraries(fft-benchmarks PUBLIC yaml-cpp)
target_link_libraries(fft-benchmarks PUBLIC MPI::MPI_CXX)
target_link_libraries(fft-benchmarks PUBLIC TBB::tbb)

find_package(Heffte REQUIRED)
target_compile_definitions(fft-benchmarks PRIVATE FFT_BENCHMARK_ENABLE_HEFFTE=1)
target_link_libraries(fft-benchmarks PUBLIC Heffte::Heffte)

if(FFT_BENCHMARK_ENABLE_PROFILING)
  target_compile_definitions(fft-benchmarks PRIVATE FFT_BENCHMARK_ENABLE_PROFILING=1)

  include(FetchContent)
  FetchContent_Declare(
    ittnotify
    URL https://github.com/intel/ittapi/archive/v3.23.0.zip
    URL_HASH
      SHA256=9456a76fcf9f30d77674b751e206b89a00cc0367ef3fadfe802b93519105793b)
  FetchContent_MakeAvailable(ittnotify)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()

if(${FFT_BENCHMARK_ENABLE_PROFILING})
  target_compile_definitions(fft-benchmarks PRIVATE ENABLE_PROFILING)
endif()
