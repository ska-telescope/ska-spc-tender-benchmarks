#include "benchmarks_common.h"
#include "benchmarks_timer.h"
#include "fft_benchmark.h"
#include "fft_check.h"
#include "fft_configuration.h"

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

#include <fftw3.h>
#ifdef ENABLE_TBB
#include <tbb/parallel_for.h>
#endif

#include <iostream>
#include <memory>

namespace fft_benchmark
{
    template <typename fft_benchmark::float_type ftype>
    struct fftw_helper;

    template <>
    struct fftw_helper<fft_benchmark::float_type::single_precision>
    {
        using plan_type = fftwf_plan;
        using complex_type = fftwf_complex;

        static auto make_plan()
        {
            return fftwf_plan_many_dft;
        }

        static auto execute()
        {
            return fftwf_execute;
        }

        static auto destroy_plan()
        {
            return fftwf_destroy_plan;
        }
    };

    template <>
    struct fftw_helper<fft_benchmark::float_type::double_precision>
    {
        using plan_type = fftw_plan;
        using complex_type = fftw_complex;

        static auto make_plan()
        {
            return fftw_plan_many_dft;
        }

        static auto execute()
        {
            return fftw_execute;
        }

        static auto destroy_plan()
        {
            return fftw_destroy_plan;
        }
    };

    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<benchmarks_common::backend_type::fftw>::launch(
        const fft_benchmark::configuration &configuration)
    {
        using fftw_complex = typename fftw_helper<ftype>::complex_type;
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
        using real = typename fft_benchmark::float_type_helper<ftype>::real;
        using plan_type = fftw_helper<ftype>::plan_type;

        fftw_init_threads();
        benchmarks_common::timer timer;

        size_t batch_size = configuration.memorysize / (2 * configuration.nx * configuration.ny * sizeof(complex));
        if (batch_size == 0)
        {
            return {};
        }

        std::array<int, 2> sizes = {static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        const size_t in_size = configuration.nx * configuration.ny * batch_size;
        const size_t out_size = in_size;

        std::unique_ptr<fftw_complex> in;
        std::unique_ptr<fftw_complex> out;
        try
        {
        in.reset(new fftw_complex[in_size]);
        out.reset(new fftw_complex[out_size]);
        }
        catch(std::bad_alloc err)
        {
            std::cout << in << std::endl;
            std::cout << out << std::endl;
            std::cout << err.what() << std::endl;
            return {};
        }

        timer.reset();
        // Plan creation.
        auto plan =
            fftw_helper<ftype>::make_plan()(2, sizes.data(), batch_size, in.get(), sizes.data(), 1, sizes[0] * sizes[1],
                                            out.get(), sizes.data(), 1, sizes[0] * sizes[1], 1, FFTW_MEASURE);
        const auto init_us = timer.time();

        // Warmup run.
        fftw_helper<ftype>::execute()(plan);

#ifdef VTUNE_PROFILE
        // Starting VTune data collection through ITT API.
        __itt_resume();
        __itt_domain *domain = __itt_domain_create("FFT.Benchmark");
        __itt_string_handle *handle_main = __itt_string_handle_create("run");
        __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
#endif

        timer.reset();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            fftw_helper<ftype>::execute()(plan);
        }
        const auto compute_us = timer.time();
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

#ifdef VTUNE_PROFILE
        // Stopping VTune data collection.
        __itt_task_end(domain);
        __itt_pause();
#endif

        // Compare results with a reference Python execution.
        const auto max_error =
            compute_fft_error(configuration, std::span{reinterpret_cast<complex *>(in.get()), in_size},
                              std::span{reinterpret_cast<complex *>(out.get()), out_size});

        fftw_helper<ftype>::destroy_plan()(plan);

        // No data for transfer times.
        benchmark_result result;
        result.status = benchmark_result::status_t::sucess;
        result.niterations = configuration.niterations;
        result.batch_size = batch_size;
        result.init_time = init_us;
        result.compute_time = average_compute_us;
        result.max_error = max_error;

        return result;
    }

    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::fftw>::launch<
        float_type::single_precision>(const fft_benchmark::configuration &configuration);
    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::fftw>::launch<
        float_type::double_precision>(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark
