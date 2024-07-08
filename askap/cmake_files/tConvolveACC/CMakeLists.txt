find_package(OpenACC REQUIRED)

add_executable(tConvolveACC)

target_compile_options(tConvolveACC PRIVATE "-acc=gpu")
target_link_libraries(tConvolveACC PRIVATE OpenACC::OpenACC_CXX)
target_include_directories(tConvolveACC PRIVATE inc)
target_sources(tConvolveACC PRIVATE Benchmark.h Benchmark.cc Stopwatch.h
                                    Stopwatch.cc tConvolveACC.h tConvolveACC.cc)
