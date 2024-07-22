#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "benchmarks_common.h"

namespace gridding_benchmark
{
    enum class operation_type
    {
        gridding,
        degridding,
        err
    };

    static const std::array<std::string, 3> operation_type_strings = {"gridding", "degridding", "err"};

    inline bool is_operation_type(const std::string &str)
    {
        return std::find(operation_type_strings.cbegin(), operation_type_strings.cend(), str) !=
               operation_type_strings.cend();
    }

    inline operation_type to_operation_type(const std::string &str)
    {
        if (str == operation_type_strings[0])
        {
            return operation_type::gridding;
        }
        if (str == operation_type_strings[1])
        {
            return operation_type::degridding;
        }
        return operation_type::err;
    }

    inline std::string to_string(const operation_type operation)
    {
        switch (operation)
        {
        case operation_type::gridding:
            return operation_type_strings[0];
        case operation_type::degridding:
            return operation_type_strings[1];
        default:
            return operation_type_strings[2];
        }
    }

    struct configuration
    {
        benchmarks_common::hardware_type htype;
        operation_type operation;
        size_t niterations;
        size_t grid_size;
        size_t subgrid_size;
        size_t nstations;
        size_t nchannels;
        size_t ntimeslots;
        size_t ntimesteps_per_subgrid;
    };

    std::vector<configuration> read_configuration(std::filesystem::path path);
} // namespace gridding_benchmark
