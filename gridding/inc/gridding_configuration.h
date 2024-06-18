#pragma once
#include <cstddef>
#include <filesystem>
#include <vector>

#include "benchmarks_common.h"

namespace gridding_benchmark
{
    struct configuration
    {
        benchmarks_common::hardware_type htype;
        size_t grid_size;
        size_t subgrid_size;
        size_t nstations;
        size_t nchannels;
        size_t ntimeslots;
        size_t ntimesteps_per_subgrid;
    };

    std::vector<configuration> read_configuration(std::filesystem::path path);
} // namespace gridding_benchmark
