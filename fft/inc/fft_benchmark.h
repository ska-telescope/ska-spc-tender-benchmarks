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

    template <hardware_type htype>
    struct hardware_type_helper;

    template <>
    struct hardware_type_helper<hardware_type::cpu>
    {
        using target_tag = heffte::backend::mkl;
    };

    template <>
    struct hardware_type_helper<hardware_type::nvidia>
    {
        using target_tag = heffte::backend::cufft;
    };

    template <>
    struct hardware_type_helper<hardware_type::amd>
    {
        using target_tag = heffte::backend::rocfft;
    };

    template <hardware_type htype>
    struct fft_helper
    {
        using backend_tag = typename hardware_type_helper<htype>::target_tag;

        static auto create_plan(const configuration &configuration, MPI_Comm comm)
        {
            heffte::box3d<> in_indexes(
                {0, 0, 0}, {static_cast<int>(configuration.nx - 1), static_cast<int>(configuration.ny - 1), 1});
            heffte::box3d<> out_indexes(
                {0, 0, 0}, {static_cast<int>(configuration.nx - 1), static_cast<int>(configuration.ny - 1), 1});

            std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(in_indexes, 1);

            std::vector<heffte::box3d<>> in_boxes = heffte::split_world(in_indexes, proc_grid);
            std::vector<heffte::box3d<>> out_boxes = heffte::split_world(out_indexes, proc_grid);

            heffte::box3d<> const inbox = in_boxes[0];
            heffte::box3d<> const outbox = out_boxes[0];

            auto plan = heffte::fft3d<backend_tag>{inbox, outbox, comm};
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
    struct benchmark_result
    {
        enum class status_t
        {
            failed,
            error,
            correct
        };

        status_t status = status_t::failed;
        size_t batch_size = 0;
        size_t niterations = 0;
        double init_time{-1.};
        double in_transfer_time{-1.};
        double compute_time{-1.};
        double out_transfer_time{-1.};

        static benchmark_result invalid_result()
        {
            return benchmark_result{status_t::failed, 0, 0, -1., -1., -1., -1.};
        }
    };

    benchmark_result launch_benchmark(const fft_benchmark::configuration &configuration);
} // namespace fft_benchmark