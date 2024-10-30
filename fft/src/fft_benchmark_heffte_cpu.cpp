#include "benchmarks_common.h"
#include "benchmarks_timer.h"
#include "fft_benchmark.h"
#include "fft_check.h"
#include "fft_configuration.h"

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

namespace fft_benchmark
{
    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<benchmarks_common::backend_type::cpu>::launch(
        const fft_benchmark::configuration &configuration)
    {
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
        using real = typename fft_benchmark::float_type_helper<ftype>::real;

        // Adaptive batch size depending on the user memory limit. If there is not enough memory for a single run,
        // returns.
        size_t batch_size;
        try
        {
            batch_size = compute_batch_size(configuration);
        }
        catch (std::bad_alloc)
        {
            batch_size = 0;
        }
        if (batch_size == 0)
        {
            return benchmark_result::invalid_result();
        }

        benchmarks_common::timer timer;
        // Plan creation.
        auto plan =
            fft_benchmark::fft_helper<benchmarks_common::backend_type::cpu>::create_plan(configuration, MPI_COMM_WORLD);
        const auto init_us = timer.time();

        std::unique_ptr<complex> in(new complex[batch_size * plan.size_inbox()]);
        std::unique_ptr<complex> out(new complex[batch_size * plan.size_inbox()]);
        std::unique_ptr<complex> workspace(new complex[batch_size * plan.size_workspace()]);

        // Warmup run.
        fft_benchmark::fft_helper<benchmarks_common::backend_type::cpu>::run(plan, 1, configuration.ttype, in.get(),
                                                                             out.get(), workspace.get());

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
            fft_benchmark::fft_helper<benchmarks_common::backend_type::cpu>::run(plan, batch_size, configuration.ttype,
                                                                                 in.get(), out.get(), workspace.get());
        }
        const auto compute_us = timer.time();
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

#ifdef VTUNE_PROFILE
        // Stopping VTune data collection.
        __itt_task_end(domain);
        __itt_pause();
#endif

        // Compare results with a reference Python execution.
        const auto max_error = compute_fft_error(configuration, std::span{in.get(), batch_size * plan.size_inbox()},
                                                 std::span{out.get(), batch_size * plan.size_outbox()});

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

    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::cpu>::launch<
        float_type::single_precision>(const fft_benchmark::configuration &configuration);
    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::cpu>::launch<
        float_type::double_precision>(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark