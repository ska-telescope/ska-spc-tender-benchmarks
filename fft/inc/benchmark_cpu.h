#pragma once
#include <fftw/fftw3.h>

#include "fft_configuration.h"

namespace fft_benchmark
{
    template <>
    struct fftw_type_helper<hardware_type::cpu, float_type::single_precision>
    {
        using complex = fftwf_complex;
        using real = float;
        using plan = fftwf_plan;
        static plan create_plan(const configuration &configuration);
        static void destroy_plan(plan plan);
        static void run_fft(plan plan, transform_type ttype, complex *in, complex *out);
        static void *malloc(size_t size);
    };

    template <>
    struct fftw_type_helper<hardware_type::cpu, float_type::double_precision>
    {
        using complex = fftw_complex;
        using real = double;
        using plan = fftw_plan;
        static plan create_plan(const configuration &configuration);
        static void destroy_plan(plan plan);
        static void run_fft(plan plan, transform_type ttype, complex *in, complex *out);
        static void *malloc(size_t size);
    };

    namespace cpu
    {
        double run(const configuration &configuration);
    }
} // namespace fft_benchmark