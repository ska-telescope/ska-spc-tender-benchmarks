#include "benchmarks_common.h"
#include "benchmarks_timer.h"
#include "fft_benchmark.h"
#include "fft_check.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<benchmarks_common::backend_type::gpu>::launch(
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

        // Plan creation + warmup for plan initialization.
        auto plan =
            fft_benchmark::fft_helper<benchmarks_common::backend_type::gpu>::create_plan(configuration, MPI_COMM_WORLD);
        heffte::fft3d<fft_helper<benchmarks_common::backend_type::gpu>::backend_tag>::buffer_container<complex>
            workspace(plan.size_workspace());
        {
            heffte::gpu::vector<complex> gpu_warmup_input(plan.size_inbox());
            heffte::gpu::vector<complex> gpu_warmup_output(plan.size_outbox());
            fft_benchmark::fft_helper<benchmarks_common::backend_type::gpu>::run(
                plan, 1, configuration.ttype, gpu_warmup_input.data(), gpu_warmup_output.data(), workspace.data());
        }
        const auto init_us = timer.time();

        // Input data initialization.
        std::vector<complex> in(batch_size * plan.size_inbox());
        // fill_signal<complex>(in.begin(), in.begin() + plan.size_inbox());
        heffte::gpu::vector<complex> gpu_input(batch_size * plan.size_inbox());

        // Moving input data to the device.
        const auto begin_in_transfer = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::load(in, gpu_input.data());
        }
        const auto after_in_transfer = std::chrono::high_resolution_clock::now();
        const auto in_transfer_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(after_in_transfer - begin_in_transfer).count() / 1000;
        const auto average_in_transfer_us =
            in_transfer_us / (static_cast<double>(batch_size * configuration.niterations));
        const auto in_bandwith_mibps = static_cast<double>(configuration.niterations * in.size() * sizeof(in[0])) /
                                       static_cast<double>(in_transfer_us);

        heffte::gpu::vector<complex> gpu_output(batch_size * plan.size_outbox());
        timer.reset();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            fft_benchmark::fft_helper<benchmarks_common::backend_type::gpu>::run(
                plan, batch_size, configuration.ttype, gpu_input.data(), gpu_output.data(), workspace.data());
        }
        heffte::gpu::synchronize_default_stream();
        const auto compute_us = timer.time();
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

        std::vector<complex> out(batch_size * plan.size_outbox());
        // Recovering output data.
        timer.reset();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::unload(gpu_output, out.data());
        }
        const auto out_transfer_us = timer.time();
        const auto average_out_transfer_us =
            out_transfer_us / (static_cast<double>(batch_size * configuration.niterations));
        const auto out_bandwith_mibps = static_cast<double>(configuration.niterations * out.size() * sizeof(out[0])) /
                                        static_cast<double>(out_transfer_us);

        // Reverting the FFT operation for validation purposes.
        fft_benchmark::fft_helper<benchmarks_common::backend_type::gpu>::run(
            plan, batch_size, invert(configuration.ttype), gpu_output.data(), gpu_input.data(), workspace.data());
        in = heffte::gpu::transfer::unload(gpu_input);
        const auto max_error = compute_fft_error(configuration, std::span{in}, std::span{out});

        benchmark_result result;
        result.status = benchmark_result::status_t::sucess;
        result.batch_size = batch_size;
        result.niterations = configuration.niterations;
        result.init_time = init_us;
        result.compute_time = average_compute_us;
        result.in_transfer_time = average_in_transfer_us;
        result.out_transfer_time = average_out_transfer_us;
        result.max_error = max_error;
        result.in_bandwidth = in_bandwith_mibps;
        result.out_bandwidth = out_bandwith_mibps;

        return result;
    }

    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::gpu>::launch<
        float_type::single_precision>(const fft_benchmark::configuration &configuration);
    template benchmark_result benchmark_launcher<benchmarks_common::backend_type::gpu>::launch<
        float_type::double_precision>(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark