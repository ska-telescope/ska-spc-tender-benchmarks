cmake_minimum_required(VERSION 3.23)

project(fft-benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FFT_BENCHMARK_ENABLE_CPU    "Enable benchmarking on CPU")
option(FFT_BENCHMARK_ENABLE_NVIDIA "Enable benchmarking on Nvidia GPU")
option(FFT_BENCHMARK_ENABLE_AMD    "Enable benchmarking on AMD GPU")

add_executable(fft-mkl)

find_package(yaml-cpp CONFIG REQUIRED)
target_include_directories(fft-mkl PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_sources(fft-mkl PRIVATE inc/fft_configuration.h src/fft_configuration.cpp)
target_sources(fft-mkl PRIVATE src/main.cpp)
target_link_libraries(fft-mkl PUBLIC yaml-cpp)

if(${FFT_BENCHMARK_ENABLE_AMD})
    find_package(rocfft REQUIRED)
endif()

if(${FFT_BENCHMARK_ENABLE_NVIDIA})
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(fft-mkl PRIVATE CUDA::cufft CUDA::cudart)
    target_sources(fft-mkl PRIVATE inc/benchmark_nvidia.h src/benchmark_nvidia.cpp)
endif()

if(${FFT_BENCHMARK_ENABLE_CPU})
    find_package(TBB REQUIRED)
    find_package(MKL CONFIG REQUIRED)
    target_sources(fft-mkl PRIVATE inc/benchmark_cpu.h src/benchmark_cpu.cpp)
    target_include_directories(fft-mkl PUBLIC $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(fft-mkl PUBLIC inc/)
    target_compile_options(fft-mkl PUBLIC $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_COMPILE_OPTIONS>)
    target_link_libraries(fft-mkl PUBLIC MKL::MKL_DPCPP)
    target_link_libraries(fft-mkl PUBLIC TBB::tbb)
endif()
