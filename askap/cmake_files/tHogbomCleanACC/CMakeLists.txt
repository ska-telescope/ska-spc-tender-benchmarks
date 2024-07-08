find_package(OpenACC REQUIRED)

add_executable(tHogbomCleanACC)

target_compile_options(tHogbomCleanACC PRIVATE "-acc=gpu")
target_link_libraries(tHogbomCleanACC PRIVATE OpenACC::OpenACC_CXX)
target_include_directories(tHogbomCleanACC PRIVATE inc)
target_sources(
  tHogbomCleanACC
  PRIVATE HogbomACC.cc
          HogbomGolden.cc
          Parameters.h
          Stopwatch.h
          tHogbomCleanACC.h
          HogbomACC.h
          HogbomGolden.h
          Stopwatch.cc
          tHogbomCleanACC.cc)
