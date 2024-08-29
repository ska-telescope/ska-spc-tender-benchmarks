#include "gridding_benchmark.h"
#include "gridding_configuration.h"
#include "gridding_initialization.h"
#include "math.hpp"
#include "types.hpp"

#include <CL/sycl.hpp>
#include <chrono>
#include <complex>
#include <fstream>

namespace gridding_benchmark
{
    template <typename T>
    auto make_device(sycl::queue &queue, T &vector, size_t &total_size)
    {
        total_size += vector.size() * sizeof(vector[0]);
        return sycl::malloc_device<typeof(vector[0])>(vector.size(), queue);
    }

    template <>
    benchmark_result gridding_benchmark_launcher<benchmarks_common::hardware_type::cpu>::launch(
        const gridding_benchmark::configuration &configuration, Array2D<UVWCoordinate<float>> &uvw,
        Array3D<Visibility<std::complex<float>>> &visibilities, Array1D<Baseline> &baselines,
        Array4D<Matrix2x2<std::complex<float>>> &aterms, Array1D<float> &frequencies, Array1D<float> &wavenumbers,
        Array2D<float> &spheroidal, Array4D<std::complex<float>> &subgrids, Array1D<Metadata> &metadata)
    {
        const auto n_baselines = (configuration.nstations * (configuration.nstations - 1)) / 2;
        const auto n_subgrids = n_baselines * configuration.ntimeslots;
        const auto total_ntimesteps = configuration.ntimesteps_per_subgrid * n_subgrids;

        const auto w_step_in_lambda = 0;
        const auto image_size = 0.1F;

        // Find offset of first subgrid
        const Metadata m = metadata.data()[0];
        const int baseline_offset_1 = m.baseline_offset;

        sycl::queue queue;

        std::cout << "Using " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

        size_t total_in_size = 0;
        size_t total_out_size = 0;
        auto *uvw_buffer = make_device(queue, uvw, total_in_size);
        auto *wavenumbers_buffer = make_device(queue, wavenumbers, total_in_size);
        auto *visibilities_buffer = make_device(queue, visibilities, total_in_size);
        auto *spheroidal_buffer = make_device(queue, spheroidal, total_in_size);
        auto *aterms_buffer = make_device(queue, aterms, total_in_size);
        auto *metadata_buffer = make_device(queue, metadata, total_in_size);
        auto *subgrids_buffer = make_device(queue, subgrids, total_out_size);

        const auto begin_in_copy = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            queue.memcpy(uvw_buffer, uvw.data(), uvw.size() * sizeof(uvw[0]));
            queue.memcpy(wavenumbers_buffer, wavenumbers.data(), wavenumbers.size() * sizeof(wavenumbers[0]));
            queue.memcpy(visibilities_buffer, visibilities.data(), visibilities.size() * sizeof(visibilities[0]));
            queue.memcpy(spheroidal_buffer, spheroidal.data(), spheroidal.size() * sizeof(spheroidal[0]));
            queue.memcpy(aterms_buffer, aterms.data(), aterms.size() * sizeof(aterms[0]));
            queue.memcpy(metadata_buffer, metadata.data(), metadata.size() * sizeof(metadata[0]));
        }
        queue.wait();
        const auto end_in_copy = std::chrono::high_resolution_clock::now();

        const auto in_copy_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_in_copy - begin_in_copy).count() /
            (1000. * configuration.niterations);

        const auto begin = std::chrono::high_resolution_clock::now();
        const auto subgrid_size = configuration.subgrid_size;
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            queue
                .submit([&](sycl::handler &h) {
                    auto *uvw = uvw_buffer;
                    auto *wavenumbers = wavenumbers_buffer;
                    auto *visibilities = visibilities_buffer;
                    auto *spheroidal = spheroidal_buffer;
                    auto *aterms = aterms_buffer;
                    auto *metadata = metadata_buffer;
                    auto *subgrids = subgrids_buffer;

                    sycl::range<3> range(n_subgrids, subgrid_size, subgrid_size);
                    h.parallel_for(range, [=](const sycl::id<3> i) {
                        const auto s = i[0];
                        const auto y = i[1];
                        const auto x = i[2];

                        // Load metadata
                        const Metadata md = metadata[s];
                        const int time_offset = (md.baseline_offset - baseline_offset_1) + md.time_offset;
                        const int nr_timesteps = md.nr_timesteps;
                        const int aterm_index = md.aterm_index;
                        const int station1 = md.baseline.station1;
                        const int station2 = md.baseline.station2;
                        const int x_coordinate = md.coordinate.x;
                        const int y_coordinate = md.coordinate.y;
                        const float w_offset_in_lambda = w_step_in_lambda * (md.coordinate.z + 0.5);

                        const auto grid_size = configuration.grid_size;
                        const auto nchannels = configuration.nchannels;
                        const auto nstations = configuration.nstations;

                        const auto half_subgrid_size = static_cast<float>(subgrid_size) * 0.5F;
                        const auto half_grid_size = static_cast<float>(grid_size) * 0.5F;
                        // Compute u and v offset in wavelenghts
                        const float u_offset =
                            (x_coordinate + half_subgrid_size - half_grid_size) * (2.F * pi() / image_size);
                        const float v_offset =
                            (y_coordinate + half_subgrid_size - half_grid_size) * (2.F * pi() / image_size);
                        const float w_offset = 2 * pi() * w_offset_in_lambda;

                        {
                            // Initialize pixel for every polarization
                            std::array<std::complex<float>, n_correlations> pixels;
                            pixels.fill(0);

                            // Compute l,m,n
                            const float l = compute_l(x, subgrid_size, image_size);
                            const float m = compute_m(y, subgrid_size, image_size);
                            const float n = compute_n(l, m);

                            // Iterate all timesteps
                            for (size_t time = 0; time < nr_timesteps; time++)
                            {
                                // Load UVW coordinates
                                const float u = uvw[time_offset + time].u;
                                const float v = uvw[time_offset + time].v;
                                const float w = uvw[time_offset + time].w;

                                // Compute phase index
                                const float phase_index = u * l + v * m + w * n;

                                // Compute phase offset
                                const float phase_offset = u_offset * l + v_offset * m + w_offset * n;

                                // Update pixel for every channel
                                for (size_t chan = 0; chan < nchannels; chan++)
                                {
                                    // Compute phase
                                    const float phase = phase_offset - (phase_index * wavenumbers[chan]);

                                    // Compute phasor
                                    const std::complex<float> phasor = {cosf(phase), sinf(phase)};

                                    // Update pixel for every polarization
                                    const size_t index = (time_offset + time) * nchannels + chan;
                                    const auto *visibilities_cpx_ptr =
                                        reinterpret_cast<const std::complex<float> *>(&visibilities[index]);
                                    for (size_t pol = 0; pol < n_correlations; pol++)
                                    {
                                        pixels[pol] += visibilities_cpx_ptr[pol] * phasor;
                                    }
                                } // end for chan
                            }     // end for time

                            // Load a term for station1
                            const size_t station1_index =
                                (aterm_index * nstations + station1) * subgrid_size * subgrid_size * n_correlations +
                                y * subgrid_size * n_correlations + x * n_correlations;
                            const std::complex<float> *aterm1_ptr =
                                &reinterpret_cast<const std::complex<float> *>(&aterms[0])[station1_index];

                            // Load aterm for station2
                            const size_t station2_index =
                                (aterm_index * nstations + station2) * subgrid_size * subgrid_size * n_correlations +
                                y * subgrid_size * n_correlations + x * n_correlations;
                            const std::complex<float> *aterm2_ptr =
                                &reinterpret_cast<const std::complex<float> *>(&aterms[0])[station2_index];

                            // Apply aterm
                            apply_aterm_gridder(&pixels[0], aterm1_ptr, aterm2_ptr);

                            // Load spheroidal
                            const float sph = spheroidal[y * subgrid_size + x];

                            // Set subgrid value
                            for (size_t pol = 0; pol < n_correlations; pol++)
                            {
                                const size_t idx_subgrid = s * n_correlations * subgrid_size * subgrid_size +
                                                           pol * subgrid_size * subgrid_size + y * subgrid_size + x;
                                subgrids[idx_subgrid] = pixels[pol] * sph;
                            }
                        }
                    });
                })
                .wait();
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto compute_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000.;

        const auto begin_out_copy = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            queue.memcpy(subgrids.data(), subgrids_buffer, subgrids.size() * sizeof(subgrids[0]));
        }
        queue.wait();
        const auto end_out_copy = std::chrono::high_resolution_clock::now();

        const auto out_copy_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_out_copy - begin_out_copy).count() /
            (1000. * configuration.niterations);

        benchmark_result result;
        result.in_transfer_time = in_copy_us;
        result.out_transfer_time = out_copy_us;
        result.in_bandwidth = static_cast<float>(total_in_size) / (in_copy_us);
        result.out_bandwidth = static_cast<float>(total_out_size) / (out_copy_us);
        result.compute_time = compute_us / static_cast<double>(configuration.niterations);
        return result;
    }
} // namespace gridding_benchmark
