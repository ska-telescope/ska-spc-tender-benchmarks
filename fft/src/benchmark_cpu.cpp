#include <tbb/parallel_for.h>

#include "benchmark_cpu.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    fftw_type_helper<hardware_type::cpu, float_type::single_precision>::plan fftw_type_helper<
        hardware_type::cpu, float_type::single_precision>::create_plan(const configuration &configuration)
    {
        return fftwf_plan_dft_2d(configuration.nx, configuration.ny, nullptr, nullptr,
                                 static_cast<int>(configuration.ftype), FFTW_MEASURE);
    }

    void fftw_type_helper<hardware_type::cpu, float_type::single_precision>::destroy_plan(plan plan)
    {
        fftwf_destroy_plan(plan);
    }

    void fftw_type_helper<hardware_type::cpu, float_type::single_precision>::run_fft(plan plan, transform_type ttype,
                                                                                     complex *in, complex *out)
    {
        fftwf_execute_dft(plan, in, out);
    }

    void *fftw_type_helper<hardware_type::cpu, float_type::single_precision>::malloc(size_t size)
    {
        return fftw_malloc(size);
    }

    fftw_type_helper<hardware_type::cpu, float_type::double_precision>::plan fftw_type_helper<
        hardware_type::cpu, float_type::double_precision>::create_plan(const configuration &configuration)
    {
        return fftw_plan_dft_2d(configuration.nx, configuration.ny, nullptr, nullptr,
                                static_cast<int>(configuration.ftype), FFTW_MEASURE);
    }

    void fftw_type_helper<hardware_type::cpu, float_type::double_precision>::destroy_plan(plan plan)
    {
        fftw_destroy_plan(plan);
    }

    void fftw_type_helper<hardware_type::cpu, float_type::double_precision>::run_fft(plan plan, transform_type ttype,
                                                                                     complex *in, complex *out)
    {
        fftw_execute_dft(plan, in, out);
    }

    void *fftw_type_helper<hardware_type::cpu, float_type::double_precision>::malloc(size_t size)
    {
        return fftw_malloc(size);
    }

    namespace cpu
    {
        template <float_type ftype>
        double run_per_float(const configuration &configuration)
        {
            const auto nx = configuration.nx;
            const auto ny = configuration.ny;
            const auto n_samples = nx * ny;
            const auto nbatches = configuration.nbatches;

            using complex = typename fftw_type_helper<hardware_type::cpu, ftype>::complex;

            const auto signal_bytes = configuration.nx * configuration.ny * configuration.nbatches * sizeof(complex);
            complex *const input_ptr = reinterpret_cast<complex *>(fftw_malloc(signal_bytes));
            complex *const output_ptr =
                configuration.in_place ? input_ptr : reinterpret_cast<complex *>(fftw_malloc(signal_bytes));

            auto plan = fftw_type_helper<hardware_type::cpu, ftype>::create_plan(configuration);

            const auto begin_parallel = std::chrono::high_resolution_clock::now();
            tbb::parallel_for(
                tbb::blocked_range<int>(0, configuration.nbatches), [&](const tbb::blocked_range<int> &range) {
                    for (int i_batch = range.begin(); i_batch < range.end(); ++i_batch)
                    {
                        fftw_type_helper<hardware_type::cpu, ftype>::run_fft(plan, configuration.ttype,
                                                                             input_ptr + i_batch * n_samples,
                                                                             output_ptr + i_batch * n_samples);
                    }
                });
            const auto end_parallel = std::chrono::high_resolution_clock::now();

            fftw_type_helper<hardware_type::cpu, ftype>::destroy_plan(plan);

            return std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - begin_parallel).count();
        }

        double run(const configuration &configuration)
        {
            switch (configuration.ftype)
            {
            case float_type::single_precision:
                return run_per_float<float_type::single_precision>(configuration);
            case float_type::double_precision:
                return run_per_float<float_type::double_precision>(configuration);
            default:
                throw(std::runtime_error{"Invalid floating point error."});
            }
        }
    } // namespace cpu
} // namespace fft_benchmark
