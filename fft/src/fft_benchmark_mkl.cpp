#include "benchmarks_common.h"
#include "benchmarks_timer.h"
#include "fft_benchmark.h"
#include "fft_check.h"
#include "fft_configuration.h"
#include "hpallocator.h"

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif
#include <iostream>
#include <mkl_dfti.h>
#include <thread>

namespace fft_benchmark
{
    inline MKL_LONG check_mkl_status(const MKL_LONG status)
    {
        if (status && !DftiErrorClass(status, DFTI_NO_ERROR))
        {
            std::cerr << "Error: " << DftiErrorMessage(status) << std::endl;
        }
        return status;
    }

    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<benchmarks_common::backend_type::mkl>::launch(
        const fft_benchmark::configuration &configuration)
    {
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
        using real = typename fft_benchmark::float_type_helper<ftype>::real;

        const size_t batch_size = configuration.memorysize / (2 * configuration.nx * configuration.ny * sizeof(complex));
        if (batch_size == 0)
            return {};

        // Plan creation + warmup for plan initialization.
        std::array<MKL_LONG, 2> sizes{static_cast<MKL_LONG>(configuration.nx), static_cast<MKL_LONG>(configuration.ny)};
        DFTI_DESCRIPTOR_HANDLE mkl_plan_handle = NULL;

        MKL_LONG mkl_status;
        const auto begin_init = std::chrono::high_resolution_clock::now();
        mkl_status = check_mkl_status(DftiCreateDescriptor(
            &mkl_plan_handle, ftype == fft_benchmark::float_type::single_precision ? DFTI_SINGLE : DFTI_DOUBLE,
            DFTI_COMPLEX, 2, sizes.data()));
        mkl_status = check_mkl_status(DftiSetValue(mkl_plan_handle, DFTI_NUMBER_OF_TRANSFORMS, batch_size));

        const auto processor_count = std::thread::hardware_concurrency();

        mkl_status = check_mkl_status(
            DftiSetValue(mkl_plan_handle, DFTI_THREAD_LIMIT, processor_count > 0 ? processor_count : 1));
        mkl_status = check_mkl_status(DftiSetValue(mkl_plan_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
        mkl_status = check_mkl_status(
            DftiSetValue(mkl_plan_handle, DFTI_FORWARD_SCALE, 1. / (configuration.nx * configuration.ny)));
        mkl_status = check_mkl_status(DftiCommitDescriptor(mkl_plan_handle));
        const auto after_init = std::chrono::high_resolution_clock::now();
        const auto init_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(after_init - begin_init).count() / 1000.;

        const auto in_size = sizes[0] * sizes[1] * batch_size;
        const auto out_size = in_size;

        std::unique_ptr<complex> in;
        std::unique_ptr<complex> out;
        try
        {
            in.reset(new complex[in_size]);
            out.reset(new complex[out_size]);
        }
        catch(std::bad_alloc err)
        {
            std::cout << err.what() << std::endl;
            return {};
        }

#ifdef VTUNE_PROFILE
        __itt_resume();
        __itt_domain *domain = __itt_domain_create("FFT.Benchmark");
        __itt_string_handle *handle_main = __itt_string_handle_create("run");
        __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
#endif
        const auto begin_compute = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            mkl_status = check_mkl_status(DftiComputeForward(mkl_plan_handle, in.get(), out.get()));
        }
        const auto after_compute = std::chrono::high_resolution_clock::now();
#ifdef VTUNE_PROFILE
        __itt_task_end(domain);
        __itt_pause();
#endif
        const auto compute_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(after_compute - begin_compute).count() / 1000.;
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

        const auto max_error = compute_fft_error(configuration, std::span{in.get(), in_size}, std::span{out.get(), out_size});

        mkl_status = check_mkl_status(DftiFreeDescriptor(&mkl_plan_handle));
        //free(in);
        //free(out);

        // No data for transfer times.
        benchmark_result result;
        result.status =
            mkl_status == DFTI_NO_ERROR ? benchmark_result::status_t::sucess : benchmark_result::status_t::failure;
        result.niterations = configuration.niterations;
        result.batch_size = batch_size;
        result.init_time = init_us;
        result.compute_time = average_compute_us;
        result.max_error = max_error;

        return result;
    }

    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::mkl>::launch<
        float_type::single_precision>(const fft_benchmark::configuration &configuration);
    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::mkl>::launch<
        float_type::double_precision>(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark
