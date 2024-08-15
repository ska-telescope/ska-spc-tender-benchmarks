cmake_minimum_required(VERSION 3.22)

project(hpcg-wrapper LANGUAGES CXX)


include(FetchContent)

FetchContent_Declare(
    hpcg-reference
    GIT_REPOSITORY https://github.com/hpcg-benchmark/hpcg.git
    GIT_TAG 114602d458d1034faa52b71e4c15aba9b3a17698
)

FetchContent_MakeAvailable(hpcg-reference)


