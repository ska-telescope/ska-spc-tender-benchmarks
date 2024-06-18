add_executable(fft-benchmarks)

if(VTUNE_PROFILE)
  target_compile_definitions(fft-benchmarks PRIVATE VTUNE_PROFILE=1)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()

find_package(Heffte REQUIRED)

target_include_directories(fft-benchmarks PRIVATE inc)

target_sources(
  fft-benchmarks PRIVATE inc/fft_configuration.h src/fft_configuration.cpp
                         inc/fft_benchmark.h src/fft_benchmark.cpp src/main.cpp)

target_link_libraries(fft-benchmarks PUBLIC Heffte::Heffte)
target_link_libraries(fft-benchmarks PUBLIC yaml-cpp)
target_link_libraries(fft-benchmarks PUBLIC MPI::MPI_CXX)
target_link_libraries(fft-benchmarks PUBLIC benchmarks-common)

# Hardware options
if(${ENABLE_CPU})
  target_compile_definitions(fft-benchmarks PUBLIC ENABLE_CPU)
endif()
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
if(VTUNE_PROFILE)
  target_compile_definitions(fft-benchmarks PRIVATE VTUNE_PROFILE=1)
  target_link_libraries(fft-benchmarks PRIVATE ittnotify)
endif()
