add_executable(fft-benchmarks)

if(VTUNE_PROFILE)
  target_compile_definitions(fft-benchmarks PRIVATE VTUNE_PROFILE=1)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()

find_package(Heffte REQUIRED)

option(FFT_ENABLE_PYTHON_CHECK "Enable FFT results check using Python." ON)
if(FFT_ENABLE_PYTHON_CHECK)
  find_package(
    Python
    COMPONENTS Interpreter NumPy
    REQUIRED)
  target_compile_definitions(fft-benchmarks
                             PRIVATE PYTHON_EXECUTABLE=${Python_EXECUTABLE})

  message(${Python_NumPy_INCLUDE_DIRS})
endif()

target_include_directories(fft-benchmarks PRIVATE inc)

target_sources(
  fft-benchmarks PRIVATE inc/fft_configuration.h src/fft_configuration.cpp
                         inc/fft_benchmark.h src/fft_benchmark.cpp src/main.cpp)

target_link_libraries(fft-benchmarks PUBLIC Heffte::Heffte)
target_link_libraries(fft-benchmarks PUBLIC yaml-cpp)
target_link_libraries(fft-benchmarks PUBLIC MPI::MPI_CXX)
target_link_libraries(fft-benchmarks PUBLIC benchmarks-common)

# Hardware options
if(ENABLE_CPU)
  target_compile_definitions(fft-benchmarks PRIVATE ENABLE_CPU)
  target_sources(fft-benchmarks PRIVATE src/fft_benchmark_heffte_cpu.cpp)
endif()
if(ENABLE_GPU)
  target_compile_definitions(fft-benchmarks PRIVATE ENABLE_GPU)
  target_sources(fft-benchmarks PRIVATE src/fft_benchmark_heffte_gpu.cpp)
endif()
if(ENABLE_MKL)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_MKL)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(fft-benchmarks PUBLIC MKL::MKL)
  target_sources(fft-benchmarks PRIVATE src/fft_benchmark_mkl.cpp)
endif()

if(ENABLE_FFTW)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_FFTW)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
  target_include_directories(fft-benchmarks PUBLIC PkgConfig::FFTW)
  target_link_libraries(fft-benchmarks PUBLIC PkgConfig::FFTW)
  target_sources(fft-benchmarks PRIVATE src/fft_benchmark_fftw.cpp)
endif()

if(PARALLELIZATION_LIBRARY STREQUAL "TBB")
  find_package(TBB REQUIRED)
  target_link_libraries(fft-benchmarks PUBLIC TBB::tbb)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_TBB)
endif()

if(PARALLELIZATION_LIBRARY STREQUAL "OMP")
  find_package(OpenMP REQUIRED)
  target_link_libraries(fft-benchmarks PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_OMP)
endif()

# Optional annotations to restrict profiling to computation parts.
if(VTUNE_PROFILE)
  target_compile_definitions(fft-benchmarks PRIVATE VTUNE_PROFILE=1)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()
