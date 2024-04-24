#pragma once
#include <complex>
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

    template <typename target_tag>
    struct fft_helper
    {
        using backend_tag = typename heffte::backend::default_backend<target_tag>::type;

        template <typename float_type>
        static auto create_plan(const configuration &configuration, MPI_Comm comm)
        {
            heffte::box3d<> in_indexes({0, 0, 0},
                                       {static_cast<int>(configuration.nx), static_cast<int>(configuration.ny), 1});
            heffte::box3d<> out_indexes({0, 0, 0},
                                        {static_cast<int>(configuration.nx), static_cast<int>(configuration.ny), 1});

            std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(in_indexes, 1);

            std::vector<heffte::box3d<>> in_boxes = heffte::split_world(in_indexes, proc_grid);
            std::vector<heffte::box3d<>> out_boxes = heffte::split_world(out_indexes, proc_grid);

            heffte::box3d<> const inbox = in_boxes[0];
            heffte::box3d<> const outbox = out_boxes[0];

            auto plan = heffte::fft3d<backend_tag>{inbox, outbox, comm};
            std::vector<std::complex<float_type>> in(plan.size_inbox());

            return plan;
        }

        template <typename plan_t, typename float_type>
        static void run(const plan_t &plan, const size_t batch_size, const transform_type ttype,
                        const std::complex<float_type> *in, std::complex<float_type> *out)
        {
            if (ttype == transform_type::forward)
            {
                plan.forward(batch_size, in, out);
            }
            else
            {
                plan.backward(batch_size, in, out);
            }
        }
    };

    struct time_result
    {
        size_t batch_size = 0;
        size_t niterations = 0;
        double init_time{0.};
        double in_transfer_time{0.};
        double compute_time{0.};
        double out_transfer_time{0.};

        static time_result invalid_result()
        {
            return time_result{0, 0, -1., -1., -1., -1.};
        }
    };

    time_result launch_benchmark(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark