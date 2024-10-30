#include "fft_benchmark.h"

namespace fft_benchmark
{
    benchmark_result launch_benchmark(const fft_benchmark::configuration &configuration)
    {
        using namespace fft_benchmark;
        switch (configuration.htype)
        {
#ifdef ENABLE_CPU
        case benchmarks_common::backend_type::cpu:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<benchmarks_common::backend_type::cpu>::launch<float_type::single_precision>(
                    configuration);
            }
            else
            {
                return benchmark_launcher<benchmarks_common::backend_type::cpu>::launch<float_type::double_precision>(
                    configuration);
            }
            break;
#endif
#ifdef ENABLE_GPU
        case benchmarks_common::backend_type::gpu:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<benchmarks_common::backend_type::gpu>::launch<float_type::single_precision>(
                    configuration);
            }
            else
            {
                return benchmark_launcher<benchmarks_common::backend_type::gpu>::launch<float_type::double_precision>(
                    configuration);
            }
            break;
#endif
#ifdef ENABLE_FFTW
        case benchmarks_common::backend_type::fftw:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<benchmarks_common::backend_type::fftw>::launch<float_type::single_precision>(
                    configuration);
            }
            else
            {
                return benchmark_launcher<benchmarks_common::backend_type::fftw>::launch<float_type::double_precision>(
                    configuration);
            }
            break;
#endif
#ifdef ENABLE_MKL
        case benchmarks_common::backend_type::mkl:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<benchmarks_common::backend_type::mkl>::launch<float_type::single_precision>(
                    configuration);
            }
            else
            {
                return benchmark_launcher<benchmarks_common::backend_type::mkl>::launch<float_type::double_precision>(
                    configuration);
            }
            break;
#endif
        default:
            return {};
        }
    }
} // namespace fft_benchmark
