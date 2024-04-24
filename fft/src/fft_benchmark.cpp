#include "fft_benchmark.h"

namespace fft_benchmark
{
    size_t compute_batch_size(const fft_benchmark::configuration &configuration)
    {
        try
        {
            switch (configuration.htype)
            {
            case fft_benchmark::hardware_type::cpu: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    using complex =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::single_precision>::complex;
                    using real =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::single_precision>::complex;
                    const auto plan =
                        fft_benchmark::fft_helper<heffte::tag::cpu>::create_plan<real>(configuration, MPI_COMM_WORLD);
                    const size_t plan_bytes = plan.size_comm_buffers() + plan.size_workspace() * sizeof(complex);
                    const size_t remaining_bytes =
                        (plan_bytes > configuration.max_data_bytes) ? 0 : configuration.max_data_bytes;
                    const size_t bytes_per_batch = (plan.size_inbox() + plan.size_outbox()) * sizeof(complex);
                    return remaining_bytes / bytes_per_batch;
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    using complex =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::double_precision>::complex;
                    using real =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::double_precision>::complex;
                    const auto plan =
                        fft_benchmark::fft_helper<heffte::tag::cpu>::create_plan<real>(configuration, MPI_COMM_WORLD);
                    const size_t plan_bytes = plan.size_comm_buffers() + plan.size_workspace() * sizeof(complex);
                    const size_t remaining_bytes =
                        (plan_bytes > configuration.max_data_bytes) ? 0 : configuration.max_data_bytes;
                    const size_t bytes_per_batch = (plan.size_inbox() + plan.size_outbox()) * sizeof(complex);
                    return remaining_bytes / bytes_per_batch;
                }
                break;
            }
            case fft_benchmark::hardware_type::nvidia: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    using complex =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::single_precision>::complex;
                    using real =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::single_precision>::complex;
                    const auto plan =
                        fft_benchmark::fft_helper<heffte::tag::gpu>::create_plan<real>(configuration, MPI_COMM_WORLD);
                    const size_t plan_bytes = plan.size_comm_buffers() + plan.size_workspace() * sizeof(complex);
                    const size_t remaining_bytes =
                        (plan_bytes > configuration.max_data_bytes) ? 0 : configuration.max_data_bytes;
                    const size_t bytes_per_batch = (plan.size_inbox() + plan.size_outbox()) * sizeof(complex);

                    return remaining_bytes / bytes_per_batch;
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    using complex =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::double_precision>::complex;
                    using real =
                        typename fft_benchmark::float_type_helper<fft_benchmark::float_type::double_precision>::complex;
                    const auto plan =
                        fft_benchmark::fft_helper<heffte::tag::gpu>::create_plan<real>(configuration, MPI_COMM_WORLD);
                    const size_t plan_bytes = plan.size_comm_buffers() + plan.size_workspace() * sizeof(complex);
                    const size_t remaining_bytes =
                        (plan_bytes > configuration.max_data_bytes) ? 0 : configuration.max_data_bytes;
                    const size_t bytes_per_batch = (plan.size_inbox() + plan.size_outbox()) * sizeof(complex);
                    return remaining_bytes / bytes_per_batch;
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

    template <fft_benchmark::hardware_type htype>
    struct benchmark_launcher
    {
        template <fft_benchmark::float_type ftype>
        static time_result launch(const fft_benchmark::configuration &configuration);
    };

    template <>
    template <fft_benchmark::float_type ftype>
    time_result benchmark_launcher<fft_benchmark::hardware_type::cpu>::launch(
        const fft_benchmark::configuration &configuration)
    {
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;

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
            return time_result::invalid_result();
        }

        const auto before_init = std::chrono::high_resolution_clock::now();
        auto plan = fft_benchmark::fft_helper<heffte::tag::cpu>::create_plan<float>(configuration, MPI_COMM_WORLD);
        std::vector<complex> warmup_in(plan.size_inbox());
        std::vector<complex> warmup_out(plan.size_inbox());
        auto warmup_configuration = configuration;
        fft_benchmark::fft_helper<heffte::tag::cpu>::run(plan, 1, configuration.ttype, warmup_in.data(),
                                                         warmup_out.data());
        const auto after_init = std::chrono::high_resolution_clock::now();

        std::vector<complex> in(batch_size * plan.size_inbox());
        std::vector<complex> out(batch_size * plan.size_outbox());

        const auto before_compute = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            fft_benchmark::fft_helper<heffte::tag::cpu>::run(plan, batch_size, configuration.ttype, in.data(),
                                                             out.data());
        }
        const auto after_compute = std::chrono::high_resolution_clock::now();

        time_result result;
        result.niterations = configuration.niterations;
        result.batch_size = batch_size;
        result.init_time = std::chrono::duration_cast<std::chrono::microseconds>(after_init - before_init).count();
        result.compute_time =
            std::chrono::duration_cast<std::chrono::microseconds>(after_compute - before_compute).count() /
            (static_cast<double>(batch_size) * static_cast<double>(configuration.niterations));

        return result;
    }

    template <>
    template <fft_benchmark::float_type ftype>
    time_result benchmark_launcher<fft_benchmark::hardware_type::nvidia>::launch(
        const fft_benchmark::configuration &configuration)
    {
        using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
        using real = typename fft_benchmark::float_type_helper<ftype>::real;

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
            return time_result::invalid_result();
        }

        const auto before_init = std::chrono::high_resolution_clock::now();
        auto plan = fft_benchmark::fft_helper<heffte::tag::gpu>::create_plan<real>(configuration, MPI_COMM_WORLD);

        {
            heffte::gpu::vector<complex> gpu_warmup_input(plan.size_inbox());
            heffte::gpu::vector<complex> gpu_warmup_output(plan.size_outbox());
            auto warmup_configuration = configuration;
            fft_benchmark::fft_helper<heffte::tag::gpu>::run(plan, 1, configuration.ttype, gpu_warmup_input.data(),
                                                             gpu_warmup_output.data());
        }
        const auto after_init = std::chrono::high_resolution_clock::now();

        std::vector<complex> in(batch_size * plan.size_inbox());
        std::iota(in.begin(), in.end(), 0);
        heffte::gpu::vector<complex> gpu_input(batch_size * plan.size_inbox());

        const auto before_in_transfer = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::load(in, gpu_input.data());
        }
        const auto after_in_transfer = std::chrono::high_resolution_clock::now();

        heffte::gpu::vector<complex> gpu_output(batch_size * plan.size_outbox());

        const auto before_compute = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            fft_benchmark::fft_helper<heffte::tag::gpu>::run(plan, batch_size, configuration.ttype, gpu_input.data(),
                                                             gpu_output.data());
        }
        const auto after_compute = std::chrono::high_resolution_clock::now();
        std::vector<complex> out(batch_size * plan.size_outbox());
        const auto before_out_transfer = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            heffte::gpu::transfer::unload(gpu_output, out.data());
        }
        const auto after_out_transfer = std::chrono::high_resolution_clock::now();

        time_result result;
        result.batch_size = batch_size;
        result.niterations = configuration.niterations;
        result.init_time = std::chrono::duration_cast<std::chrono::microseconds>(after_init - before_init).count();
        result.compute_time =
            std::chrono::duration_cast<std::chrono::microseconds>(after_compute - before_compute).count() /
            (static_cast<double>(batch_size) * static_cast<double>(configuration.niterations));
        result.in_transfer_time =
            std::chrono::duration_cast<std::chrono::microseconds>(after_in_transfer - before_in_transfer).count() /
            static_cast<double>(configuration.niterations);
        result.out_transfer_time =
            std::chrono::duration_cast<std::chrono::microseconds>(after_out_transfer - before_out_transfer).count() /
            static_cast<double>(configuration.niterations);

        return result;
    }

    time_result launch_benchmark(const fft_benchmark::configuration &configuration)
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