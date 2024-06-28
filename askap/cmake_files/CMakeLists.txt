cmake_minimum_required(VERSION 3.20)

project(askap-benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(tConvolveACC)
add_subdirectory(tHogbomCleanACC)
add_subdirectory(tHogbomCleanOMP)
add_subdirectory(tMajorACC)
add_subdirectory(tConvolveMPI)
