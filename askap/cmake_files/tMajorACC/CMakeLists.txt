find_package(OpenACC REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 fftw3f IMPORTED_TARGET)

add_executable(tMajorACC)
target_link_libraries(tMajorACC PRIVATE OpenACC::OpenACC_CXX)
target_link_libraries(tMajorACC PRIVATE PkgConfig::FFTW)
target_include_directories(tMajorACC PRIVATE inc)
target_sources(tMajorACC PRIVATE Stopwatch.h Stopwatch.cc tMajorACC.cc)
