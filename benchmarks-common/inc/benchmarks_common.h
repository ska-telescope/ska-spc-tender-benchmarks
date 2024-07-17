#pragma once
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace benchmarks_common
{
    enum class hardware_type
    {
        cpu,
        gpu,
        err
    };

    inline std::string hardware_type_string(const benchmarks_common::hardware_type htype)
    {
        switch (htype)
        {
        case benchmarks_common::hardware_type::cpu:
            return "cpu";
        case benchmarks_common::hardware_type::gpu:
            return "gpu";
        default:
            return "";
        }
    }

    inline bool is_hardware_type_enabled(const benchmarks_common::hardware_type htype){
    #ifdef ENABLE_CPU
        if(htype == benchmarks_common::hardware_type::cpu){
            return true;
        }
    #endif
    #ifdef ENABLE_GPU
        if(htype == benchmarks_common::hardware_type::gpu){
            return true;
        }
    #endif
        return false;
    }

    template <size_t N>
    void print_columns(const std::vector<std::array<std::string, N>> &data)
    {
        std::array<size_t, N> column_width;
        column_width.fill(0);
        for (const auto &row : data)
        {
            for (size_t i = 0; i < N; ++i)
            {
                column_width[i] = std::max(column_width[i], row[i].size());
            }
        }

        for (const auto &row : data)
        {
            for (size_t i = 0; i < N; ++i)
            {
                std::cout << std::right << std::setfill(' ') << std::setw(column_width[i]) << row[i];
                std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout.flush();
    }

    template <size_t N>
    void write_csv(const std::filesystem::path &path, const std::vector<std::array<std::string, N>> &data)
    {
        std::ofstream file(path.filename());
        for (const auto &row : data)
        {
            for (size_t i = 0; i < N; ++i)
            {
                file << row[i];
                file << ";";
            }
            file << "\n";
        }
        std::cout.flush();
    }

    size_t log2_size_t(size_t index);

    std::string bytes_to_memory_size(const size_t n);

    inline void log_and_abort(std::string_view message, int status_code = -1){
        std::cerr << "Error: " << message << "\n";
        std::exit(status_code);
    }
    
} // namespace benchmarks_common