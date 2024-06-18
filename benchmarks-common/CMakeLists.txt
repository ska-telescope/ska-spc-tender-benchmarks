add_library(benchmarks-common)

target_include_directories(benchmarks-common PUBLIC inc)
target_sources(benchmarks-common PRIVATE inc/benchmarks_common.h)
target_sources(benchmarks-common PRIVATE src/benchmarks_common.cpp)
