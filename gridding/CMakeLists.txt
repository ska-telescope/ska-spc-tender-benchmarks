add_executable(gridding-benchmarks)

if(VTUNE_PROFILE)
  target_compile_definitions(gridding-benchmarks PRIVATE VTUNE_PROFILE=1)
  target_link_libraries(gridding-benchmarks PRIVATE ittnotify)
endif()

if(PARALLELIZATION_LIBRARY STREQUAL "TBB")
  find_package(TBB REQUIRED)
  target_link_libraries(gridding-benchmarks PUBLIC TBB::tbb)
  target_compile_definitions(gridding-benchmarks PUBLIC ENABLE_TBB)
endif()

if(PARALLELIZATION_LIBRARY STREQUAL "OMP")
  find_package(OpenMP REQUIRED)
  target_link_libraries(gridding-benchmarks PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(gridding-benchmarks PUBLIC ENABLE_OMP)
endif()

# Hardware options
if(ENABLE_CPU)
  target_compile_definitions(gridding-benchmarks PUBLIC ENABLE_CPU=1)
endif()
if(ENABLE_GPU)
  target_compile_definitions(gridding-benchmarks PUBLIC ENABLE_GPU=1)
endif()

target_include_directories(gridding-benchmarks PRIVATE inc)

target_sources(
  gridding-benchmarks
  PRIVATE inc/common_gridding.h
          inc/common_gridding.cuh
          inc/degridding_benchmark.h
          inc/gridding_configuration.h
          inc/gridding_benchmark.h
          inc/math.hpp
          inc/types.hpp
          src/degridding_benchmark.cpp
          src/degridding_benchmark.cu
          src/gridding_configuration.cpp
          src/gridding_benchmark.cpp
          src/gridding_benchmark.cu
          inc/gridding_initialization.h
          src/gridding_initialization.cpp
          src/main.cpp)

target_link_libraries(gridding-benchmarks PUBLIC yaml-cpp)
target_link_libraries(gridding-benchmarks PUBLIC MPI::MPI_CXX)
target_link_libraries(gridding-benchmarks PUBLIC benchmarks-common)
