#pragma once
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fft_benchmark
{
    enum class float_type
    {
        single_precision,
        double_precision
    };

    enum class transform_type
    {
        forward = -1,
        inverse = 1
    };

    inline transform_type invert(const transform_type t)
    {
        return t == transform_type::forward ? transform_type::inverse : transform_type::forward;
    }

    enum class hardware_type
    {
#ifdef ENABLE_CPU
        cpu,
#endif
#ifdef ENABLE_GPU
        gpu,
#endif
        err
    };

    struct configuration
    {
        float_type ftype;
        transform_type ttype;
        hardware_type htype;
        size_t nx, ny;
        size_t niterations;
        size_t memorysize;
        bool in_place;
    };

    template <hardware_type htype, float_type ftype>
    struct fftw_type_helper;

    class invalid_float_type_error : std::runtime_error
    {
      public:
        invalid_float_type_error() : std::runtime_error("Invalid floating type specifier used.")
        {
        }
    };

    std::vector<fft_benchmark::configuration> read_configurations(std::filesystem::path path);
} // namespace fft_benchmark