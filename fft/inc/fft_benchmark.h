#pragma once
#include <array>
#include <complex>
#include <cstddef>
#include <heffte.h>

#include "fft_configuration.h"

namespace fft_benchmark
{
    template <fft_benchmark::float_type ftype>
    struct float_type_helper;

    template <>
    struct float_type_helper<fft_benchmark::float_type::single_precision>
    {
        using complex = std::complex<float>;
        using real = float;
    };

    template <>
    struct float_type_helper<fft_benchmark::float_type::double_precision>
    {
        using complex = std::complex<double>;
        using real = double;
    };

    template <benchmarks_common::backend_type htype>
    struct backend_type_helper;

#ifdef ENABLE_CPU
    template <>
    struct backend_type_helper<benchmarks_common::backend_type::cpu>
    {
        using target_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
    };
#endif

#ifdef ENABLE_GPU
    template <>
    struct backend_type_helper<benchmarks_common::backend_type::gpu>
    {
        using target_tag = heffte::backend::default_backend<heffte::tag::gpu>::type;
    };
#endif

    inline std::string float_type_string(const fft_benchmark::float_type ftype)
    {
        switch (ftype)
        {
        case fft_benchmark::float_type::single_precision:
            return "single";
        case fft_benchmark::float_type::double_precision:
            return "double";
        default:
            return "";
        }
    }

    template <benchmarks_common::backend_type htype>
    struct fft_helper
    {
        using backend_tag = typename backend_type_helper<htype>::target_tag;

        static auto create_plan(const configuration &configuration, MPI_Comm comm)
        {
            heffte::box3d<> in_indexes(std::array<int, 3>({0, 0, 0}),
                                       std::array<int, 3>({static_cast<int>(configuration.nx - 1),
                                                           static_cast<int>(configuration.ny - 1), 0}));
            heffte::box3d<> out_indexes(std::array<int, 3>({0, 0, 0}),
                                        std::array<int, 3>({static_cast<int>(configuration.nx - 1),
                                                            static_cast<int>(configuration.ny - 1), 0}));

            std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(in_indexes, 1);

            std::vector<heffte::box2d<>> in_boxes = heffte::split_world(in_indexes, proc_grid);
            std::vector<heffte::box2d<>> out_boxes = heffte::split_world(out_indexes, proc_grid);

            heffte::box2d<> const inbox = in_boxes[0];
            heffte::box2d<> const outbox = out_boxes[0];

            auto plan = heffte::fft2d<backend_tag>{inbox, outbox, comm};
            return plan;
        }

        template <typename plan_t, typename float_type>
        static void run(const plan_t &plan, const size_t batch_size, const transform_type ttype,
                        const std::complex<float_type> *in, std::complex<float_type> *out,
                        std::complex<float_type> *workspace)
        {
            if (ttype == transform_type::forward)
            {
                plan.forward(batch_size, in, out, workspace, heffte::scale::full);
            }
            else
            {
                plan.backward(batch_size, in, out, workspace, heffte::scale::full);
            }
        }
    };

    struct benchmark_result
    {
        enum class status_t
        {
            failure,
            sucess
        };

        status_t status = status_t::failure;
        size_t batch_size = 0;
        size_t niterations = 0;
        double init_time{-1.};
        double in_transfer_time{-1.};
        double in_bandwidth{-1};
        double out_bandwidth{-1.};
        double compute_time{-1.};
        double out_transfer_time{-1.};
        double max_error{-1.};

        static benchmark_result invalid_result()
        {
            return benchmark_result{};
        }
    };

    benchmark_result launch_benchmark(const fft_benchmark::configuration &configuration);

    template <benchmarks_common::backend_type htype>
    struct benchmark_launcher
    {
        template <fft_benchmark::float_type ftype>
        static benchmark_result launch(const fft_benchmark::configuration &configuration);
    };

    template <benchmarks_common::backend_type htype, float_type ftype>
    size_t batch_size_compute_helper(const configuration &configuration)
    {
        if constexpr (htype == benchmarks_common::backend_type::mkl)
        {
            return configuration.memorysize / (configuration.nx * configuration.ny);
        }
        else
        {
            using complex = typename fft_benchmark::float_type_helper<ftype>::complex;
            using real = typename fft_benchmark::float_type_helper<ftype>::real;
            const auto plan = fft_helper<htype>::create_plan(configuration, MPI_COMM_WORLD);
            const size_t plan_bytes = plan.size_comm_buffers();
            const size_t remaining_bytes =
                (plan_bytes > configuration.memorysize) ? 0 : configuration.memorysize - plan_bytes;
            const size_t bytes_per_batch =
                (plan.size_inbox() + plan.size_outbox() + plan.size_workspace()) * sizeof(complex);
            return remaining_bytes / bytes_per_batch;
        }
    }

    inline size_t compute_batch_size(const fft_benchmark::configuration &configuration)
    {
        try
        {
            switch (configuration.htype)
            {
#ifdef ENABLE_CPU
            case benchmarks_common::backend_type::cpu: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::cpu,
                                                     float_type::single_precision>(configuration);
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::cpu,
                                                     float_type::double_precision>(configuration);
                }
                break;
            }
#endif
#ifdef ENABLE_GPU
            case benchmarks_common::backend_type::gpu: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::gpu,
                                                     float_type::single_precision>(configuration);
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::gpu,
                                                     float_type::single_precision>(configuration);
                }
                break;
            }
#endif
#ifdef ENABLE_MKL
            case benchmarks_common::backend_type::mkl: {
                if (fft_benchmark::float_type::single_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::mkl,
                                                     float_type::single_precision>(configuration);
                }
                else if (fft_benchmark::float_type::double_precision == configuration.ftype)
                {
                    return batch_size_compute_helper<benchmarks_common::backend_type::mkl,
                                                     float_type::single_precision>(configuration);
                }
                break;
            }
#endif
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

} // namespace fft_benchmark
