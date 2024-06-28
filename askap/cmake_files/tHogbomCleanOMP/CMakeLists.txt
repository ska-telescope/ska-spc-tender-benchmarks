find_package(OpenMP REQUIRED)

add_executable(tHogbomCleanOMP)
target_link_libraries(tHogbomCleanOMP PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(tHogbomCleanOMP PRIVATE inc)
target_sources(
  tHogbomCleanOMP
  PRIVATE HogbomOMP.cc
          HogbomGolden.cc
          Parameters.h
          Stopwatch.h
          tHogbomCleanOMP.h
          HogbomOMP.h
          HogbomGolden.h
          Stopwatch.cc
          tHogbomCleanOMP.cc)
