find_package(MPI REQUIRED)

add_executable(tConvolveMPI)
target_link_libraries(tConvolveMPI PRIVATE MPI::MPI_CXX)
target_include_directories(tConvolveMPI PRIVATE inc)
target_sources(tConvolveMPI PRIVATE Benchmark.h Benchmark.cc Stopwatch.h
                                    Stopwatch.cc tConvolveMPI.h tConvolveMPI.cc)
