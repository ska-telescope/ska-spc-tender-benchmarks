#include <cmath>
#include <complex>
#include <limits>
#include <tbb/parallel_for.h>
#include <vector>

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

#include "fft_benchmark.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    template <hardware_type htype, float_type ftype>
    size_t batch_size_compute_helper(const configuration &configuration)
    {
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
        using real = typename fft_benchmark::float_type_helper<ftype>::real;
        const auto plan = fft_helper<htype>::create_plan(configuration, MPI_COMM_WORLD);
        const size_t plan_bytes = plan.size_comm_buffers() + plan.size_workspace() * sizeof(complex);
        const size_t remaining_bytes = (plan_bytes > configuration.memorysize) ? 0 : configuration.memorysize;
        const size_t bytes_per_batch = (plan.size_inbox() + plan.size_outbox()) * sizeof(complex);
        return remaining_bytes / bytes_per_batch;
    }

    size_t compute_batch_size(const fft_benchmark::configuration &configuration)
    {
        try
        {
            switch (configuration.htype)
            {
            case fft_benchmark::hardware_type::cpu: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<hardware_type::cpu, float_type::single_precision>(configuration);
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<hardware_type::cpu, float_type::double_precision>(configuration);
                }
                break;
            }
            case fft_benchmark::hardware_type::nvidia: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<hardware_type::nvidia, float_type::single_precision>(
                        configuration);
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<hardware_type::nvidia, float_type::single_precision>(
                        configuration);
                }
                break;
            }
            default:
                return 0;
            }
        }
        catch (std::bad_alloc)
        {
            std::cerr << "Unable to allocate plan for " << configuration.nx * configuration.ny << " FFT." << std::endl;
            return 0;
        }
        return 0;
    }

    template <typename complex>
    void fill_signal(typename std::vector<complex>::iterator begin, typename std::vector<complex>::iterator end)
    {
        int k = 0;
        for (auto it = begin; it < end; ++it)
        {
            *it = complex(k);
            k = (k + 1) % 128;
        }
    }

    template <typename float_t>
    bool check_result(typename std::vector<std::complex<float_t>>::iterator begin,
                      typename std::vector<std::complex<float_t>>::iterator end, const float_t scaling)
    {
        bool correct = true;
        int i = 0;
        for (auto it = begin; it < end; ++it)
        {
            auto &x = *it;
            const std::complex<float_t> ref = i;
            i = (i + 1) % 128;
            const std::complex<float_t> scaled_x = x * scaling;
            const auto err = std::abs(ref - scaled_x);
            const auto threshold = std::numeric_limits<float_t>::epsilon() * 256;
            correct = correct and err < threshold;
            if (!correct)
            {
                break;
            }
        }
        return correct;
    }

    template <fft_benchmark::hardware_type htype>
    struct benchmark_launcher
    {
        template <fft_benchmark::float_type ftype>
        static benchmark_result launch(const fft_benchmark::configuration &configuration);
    };

    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<fft_benchmark::hardware_type::cpu>::launch(
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

        // Plan creation + warmup for plan initialization.
        const auto before_init = std::chrono::high_resolution_clock::now();
        auto plan = fft_benchmark::fft_helper<hardware_type::cpu>::create_plan(configuration, MPI_COMM_WORLD);
        std::vector<complex> warmup_in(plan.size_inbox());
        std::vector<complex> warmup_out(plan.size_inbox());
        auto warmup_configuration = configuration;
        std::vector<complex> workspace(plan.size_workspace());
        fft_benchmark::fft_helper<hardware_type::cpu>::run(plan, 1, configuration.ttype, warmup_in.data(),
                                                           warmup_out.data(), workspace.data());
        const auto after_init = std::chrono::high_resolution_clock::now();
        const auto init_us = std::chrono::duration_cast<std::chrono::microseconds>(after_init - before_init).count();

        // Input data initialization.
        std::vector<complex> in(batch_size * plan.size_inbox());
        fill_signal<complex>(in.begin(), in.begin() + plan.size_inbox());
        std::vector<complex> out(batch_size * plan.size_outbox());

#ifdef VTUNE_PROFILE
        __itt_resume();
        __itt_domain *domain = __itt_domain_create("FFT.Benchmark.MKL");
        __itt_string_handle *handle_main = __itt_string_handle_create("run");
        __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
#endif
        const auto before_compute = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, batch_size), [&](const tbb::blocked_range<size_t> &range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                {
                    fft_benchmark::fft_helper<hardware_type::cpu>::run(
                        plan, 1, configuration.ttype, in.data() + i * plan.size_inbox(),
                        out.data() + i * plan.size_outbox(), workspace.data());
                }
            });
        }
        const auto after_compute = std::chrono::high_resolution_clock::now();
#ifdef VTUNE_PROFILE
        __itt_task_end(domain);
        __itt_pause();
#endif
        const auto compute_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_compute - before_compute).count();
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

        // Reverting the FFT operation for validation purposes.
        fft_benchmark::fft_helper<hardware_type::cpu>::run(plan, batch_size, invert(configuration.ttype), out.data(),
                                                           in.data(), workspace.data());
        const auto status = check_result(in.begin(), in.begin() + plan.size_outbox(),
                                         real(1) / static_cast<real>(configuration.nx * configuration.ny * 2))
                                ? benchmark_result::status_t::correct
                                : benchmark_result::status_t::error;

        // No data for transfer times.
        benchmark_result result;
        result.status = status;
        result.niterations = configuration.niterations;
        result.batch_size = batch_size;
        result.init_time = init_us;
        result.compute_time = average_compute_us;

        return result;
    }

    template <>
    template <fft_benchmark::float_type ftype>
    benchmark_result benchmark_launcher<fft_benchmark::hardware_type::nvidia>::launch(
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

        // Plan creation + warmup for plan initialization.
        const auto before_init = std::chrono::high_resolution_clock::now();
        auto plan = fft_benchmark::fft_helper<hardware_type::nvidia>::create_plan(configuration, MPI_COMM_WORLD);
        heffte::fft3d<fft_helper<fft_benchmark::hardware_type::nvidia>::backend_tag>::buffer_container<complex>
            workspace(plan.size_workspace());
        {
            heffte::gpu::vector<complex> gpu_warmup_input(plan.size_inbox());
            heffte::gpu::vector<complex> gpu_warmup_output(plan.size_outbox());
            fft_benchmark::fft_helper<hardware_type::nvidia>::run(plan, 1, configuration.ttype, gpu_warmup_input.data(),
                                                                  gpu_warmup_output.data(), workspace.data());
        }
        const auto after_init = std::chrono::high_resolution_clock::now();
        const auto init_us = std::chrono::duration_cast<std::chrono::microseconds>(after_init - before_init).count();

        // Input data initialization.
        std::vector<complex> in(batch_size * plan.size_inbox());
        fill_signal<complex>(in.begin(), in.begin() + plan.size_inbox());
        heffte::gpu::vector<complex> gpu_input(batch_size * plan.size_inbox());

        // Moving input data to the device.
        const auto before_in_transfer = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::load(in, gpu_input.data());
        }
        const auto after_in_transfer = std::chrono::high_resolution_clock::now();
        const auto in_transfer_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_in_transfer - before_in_transfer).count();
        const auto average_in_transfer_us =
            in_transfer_us / (static_cast<double>(batch_size * configuration.niterations));

        heffte::gpu::vector<complex> gpu_output(batch_size * plan.size_outbox());
        const auto before_compute = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            fft_benchmark::fft_helper<hardware_type::nvidia>::run(
                plan, batch_size, configuration.ttype, gpu_input.data(), gpu_output.data(), workspace.data());
        }
        heffte::gpu::synchronize_default_stream();
        const auto after_compute = std::chrono::high_resolution_clock::now();
        const auto compute_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_compute - before_compute).count();
        const auto average_compute_us = compute_us / (static_cast<double>(batch_size * configuration.niterations));

        std::vector<complex> out(batch_size * plan.size_outbox());
        // Recovering output data.
        const auto before_out_transfer = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::unload(gpu_output, out.data());
        }
        const auto after_out_transfer = std::chrono::high_resolution_clock::now();
        const auto out_transfer_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_out_transfer - before_out_transfer).count();
        const auto average_out_transfer_us =
            out_transfer_us / (static_cast<double>(batch_size * configuration.niterations));

        // Reverting the FFT operation for validation purposes.
        fft_benchmark::fft_helper<hardware_type::nvidia>::run(plan, batch_size, invert(configuration.ttype),
                                                              gpu_output.data(), gpu_input.data(), workspace.data());
        in = heffte::gpu::transfer::unload(gpu_input);
        const auto status = check_result(in.begin(), in.begin() + plan.size_outbox(),
                                         real(1) / static_cast<real>(configuration.nx * configuration.ny * 2))
                                ? benchmark_result::status_t::correct
                                : benchmark_result::status_t::error;

        benchmark_result result;
        result.status = status;
        result.batch_size = batch_size;
        result.niterations = configuration.niterations;
        result.init_time = std::chrono::duration_cast<std::chrono::microseconds>(after_init - before_init).count();
        result.compute_time = average_compute_us;
        result.in_transfer_time = average_in_transfer_us;
        result.out_transfer_time = average_out_transfer_us;

        return result;
    }

    benchmark_result launch_benchmark(const fft_benchmark::configuration &configuration)
    {
        using namespace fft_benchmark;
        switch (configuration.htype)
        {
        case fft_benchmark::hardware_type::cpu:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<hardware_type::cpu>::launch<float_type::single_precision>(configuration);
            }
            else
            {
                return benchmark_launcher<hardware_type::cpu>::launch<float_type::double_precision>(configuration);
            }
            break;
        case fft_benchmark::hardware_type::nvidia:
            if (configuration.ftype == float_type::single_precision)
            {
                return benchmark_launcher<hardware_type::nvidia>::launch<float_type::single_precision>(configuration);
            }
            else
            {
                return benchmark_launcher<hardware_type::nvidia>::launch<float_type::double_precision>(configuration);
            }
            break;
        default:
            return {};
        }
    }
} // namespace fft_benchmark