#include "degridding_benchmark.h"
#include "gridding_configuration.h"
//#include "gridding_initialization.h"
#include "math.hpp"
#include "types.hpp"

#include <CL/sycl.hpp>
#include <chrono>
#include <complex>

namespace gridding_benchmark
{
    template <typename T>
    auto make_device(sycl::queue &queue, T &vector)
    {
        return sycl::malloc_device<typeof(vector[0])>(vector.size(), queue);
    }

    template <typename T>
    auto make_device(sycl::queue &queue, T &vector, size_t &total_size)
    {
        total_size += vector.size() * sizeof(vector[0]);
        return make_device(queue, vector);
    }

    template <>
    benchmark_result degridding_benchmark_launcher<benchmarks_common::hardware_type::cpu>::launch(
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

        const auto subgrid_size = configuration.subgrid_size;
        std::vector<std::complex<float>> pixels_vec(subgrid_size * subgrid_size * n_correlations * 8);
        auto *pixels_buffer = make_device(queue, pixels_vec);

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
                    auto *pixels = pixels_buffer;

                    sycl::range<3> range(n_subgrids, subgrid_size, subgrid_size);
                    h.parallel_for(range, [=](const sycl::id<3> i) {
                        const auto s = i[0];
                        const auto y = i[1];
                        const auto x = i[2];

                        // Load metadata
                        const Metadata m = metadata[s];
                        const int time_offset = (m.baseline_offset - baseline_offset_1) + m.time_offset;
                        const int nr_timesteps = m.nr_timesteps;
                        const int aterm_index = m.aterm_index;
                        const int station1 = m.baseline.station1;
                        const int station2 = m.baseline.station2;
                        const int x_coordinate = m.coordinate.x;
                        const int y_coordinate = m.coordinate.y;
                        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

                        //   Apply aterm to subgrid
                        //   Load aterm for station1
                        const size_t station1_index = (aterm_index * configuration.nstations + station1) *
                                                          subgrid_size * subgrid_size * n_correlations +
                                                      y * subgrid_size * n_correlations + x * n_correlations;
                        const std::complex<float> *aterm1_ptr =
                            &reinterpret_cast<std::complex<float> *>(aterms)[station1_index];

                        // Load aterm for station2
                        const size_t station2_index = (aterm_index * configuration.nstations + station2) *
                                                          subgrid_size * subgrid_size * n_correlations +
                                                      y * subgrid_size * n_correlations + x * n_correlations;
                        const std::complex<float> *aterm2_ptr =
                            &reinterpret_cast<std::complex<float> *>(aterms)[station2_index];

                        // Load spheroidal
                        const float sph = spheroidal[y * subgrid_size + x];

                        // Load uv values
                        std::complex<float> pixels_[n_correlations];
                        for (size_t pol = 0; pol < n_correlations; pol++)
                        {
                            const size_t idx_subgrid = s * n_correlations * subgrid_size * subgrid_size +
                                                       pol * subgrid_size * subgrid_size + y * subgrid_size + x;
                            pixels_[pol] = sph * subgrids[idx_subgrid];
                        }

                        // Apply aterm
                        gridding_benchmark::apply_aterm_gridder(pixels_, aterm1_ptr, aterm2_ptr);

                        // Store pixels
                        for (size_t pol = 0; pol < n_correlations; pol++)
                        {
                            pixels[(y * subgrid_size + x) * subgrid_size + pol] = pixels_[pol];
                        }

                        // Compute u and v offset in wavelenghts
                        const float u_offset = (x_coordinate + static_cast<float>(subgrid_size) / 2 -
                                                static_cast<float>(configuration.grid_size) / 2) *
                                               (2 * M_PI / image_size);
                        const float v_offset = (y_coordinate + static_cast<float>(subgrid_size) / 2 -
                                                static_cast<float>(configuration.grid_size) / 2) *
                                               (2 * M_PI / image_size);
                        const float w_offset = 2 * M_PI * w_offset_in_lambda;

                        // Iterate all timesteps and channels

                        for (size_t time = 0; time < nr_timesteps; time++)
                            for (size_t chan = 0; chan < configuration.nchannels; chan++)
                            {
                                // Load UVW coordinates
                                const float u = uvw[time_offset + time].u;
                                const float v = uvw[time_offset + time].v;
                                const float w = uvw[time_offset + time].w;

                                // Update all polarizations
                                std::array<std::complex<float>, n_correlations> sum;
                                sum.fill(0);

                                // Compute l,m,n
                                const float l = compute_l(x, subgrid_size, image_size);
                                const float m = compute_m(y, subgrid_size, image_size);
                                const float n = compute_n(l, m);
                                // Compute phase index
                                const float phase_index = u * l + v * m + w * n;
                                // Compute phase offset
                                const float phase_offset = u_offset * l + v_offset * m + w_offset * n;
                                // Compute phase
                                const float phase = (phase_index * wavenumbers[chan]) - phase_offset;
                                // Compute phasor
                                std::complex<float> phasor = {cosf(phase), sinf(phase)};
                                for (int pol = 0; pol < n_correlations; pol++)
                                {
                                    sum[pol] += pixels[(y * subgrid_size + x) * subgrid_size + pol] * phasor;
                                }

                                const size_t index = (time_offset + time) * configuration.nchannels + chan;
                                for (size_t pol = 0; pol < n_correlations; pol++)
                                {
                                    reinterpret_cast<std::complex<float> *>(
                                        visibilities)[index * n_correlations + pol] = sum[pol];
                                }
                            } // end for time and channel
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
