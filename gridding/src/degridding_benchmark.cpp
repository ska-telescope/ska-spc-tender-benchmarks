#include "degridding_benchmark.h"
#include "gridding_benchmark.h"
#include "gridding_configuration.h"
#include "gridding_initialization.h"
#include "math.hpp"

#include <chrono>
#include <complex>
#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif
#include <oneapi/tbb/blocked_range3d.h>
#ifdef ENABLE_OMP
#include <omp.h>
#endif
#ifdef ENABLE_TBB
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#endif

namespace gridding_benchmark
{
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

#ifdef VTUNE_PROFILE
        __itt_resume();
        __itt_domain *domain = __itt_domain_create("Degridding.Benchmark");
        __itt_string_handle *handle_main = __itt_string_handle_create("run");
        __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
#endif

        const auto begin = std::chrono::high_resolution_clock::now();
        const auto subgrid_size = configuration.subgrid_size;

#ifdef ENABLE_TBB
        const tbb::blocked_range3d<size_t> range(0, n_subgrids, 0, subgrid_size, 0, subgrid_size);
        tbb::parallel_for(range, [&](const tbb::blocked_range3d<size_t> &range) {
            for (size_t s = range.pages().begin(); s < range.pages().end(); ++s)
            {
#else
        for (size_t s = 0; s < n_subgrids; ++s)
        {
#endif
                // Load metadata
                const Metadata m = metadata.data()[s];
                const int time_offset = (m.baseline_offset - baseline_offset_1) + m.time_offset;
                const int nr_timesteps = m.nr_timesteps;
                const int aterm_index = m.aterm_index;
                const int station1 = m.baseline.station1;
                const int station2 = m.baseline.station2;
                const int x_coordinate = m.coordinate.x;
                const int y_coordinate = m.coordinate.y;
                const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

                // Storage
                std::complex<float> pixels[subgrid_size][subgrid_size][n_correlations];

#ifdef ENABLE_TBB
                for (size_t y = range.rows().begin(); y < range.rows().end(); ++y)
                {
                    for (size_t x = range.cols().begin(); x < range.cols().end(); ++x)
                    {
#elif defined(ENABLE_OMP)
#pragma omp parallel for collapse(2)
            for (int y = 0; y < subgrid_size; y++)
                for (int x = 0; x < subgrid_size; x++)
                {
#endif
                        // Apply aterm to subgrid
                        // Load aterm for station1
                        int station1_index = (aterm_index * configuration.nstations + station1) * subgrid_size *
                                                 subgrid_size * n_correlations +
                                             y * subgrid_size * n_correlations + x * n_correlations;
                        const std::complex<float> *aterm1_ptr =
                            &reinterpret_cast<std::complex<float> *>(aterms.data())[station1_index];

                        // Load aterm for station2
                        int station2_index = (aterm_index * configuration.nstations + station2) * subgrid_size *
                                                 subgrid_size * n_correlations +
                                             y * subgrid_size * n_correlations + x * n_correlations;
                        const std::complex<float> *aterm2_ptr =
                            &reinterpret_cast<std::complex<float> *>(aterms.data())[station2_index];

                        // Load spheroidal
                        float sph = spheroidal[y * subgrid_size + x];

                        // Load uv values
                        std::complex<float> pixels_[n_correlations];
                        for (int pol = 0; pol < n_correlations; pol++)
                        {
                            unsigned idx_subgrid = s * n_correlations * subgrid_size * subgrid_size +
                                                   pol * subgrid_size * subgrid_size + y * subgrid_size + x;
                            pixels_[pol] = sph * subgrids[idx_subgrid];
                        }

                        // Apply aterm
                        gridding_benchmark::apply_aterm_gridder(pixels_, aterm1_ptr, aterm2_ptr);

                        // Store pixels
                        for (int pol = 0; pol < n_correlations; pol++)
                        {
                            pixels[y][x][pol] = pixels_[pol];
                        }
                    }

                    // Compute u and v offset in wavelenghts
                    const float u_offset = (x_coordinate + static_cast<float>(subgrid_size) / 2 -
                                            static_cast<float>(configuration.grid_size) / 2) *
                                           (2 * M_PI / image_size);
                    const float v_offset = (y_coordinate + static_cast<float>(subgrid_size) / 2 -
                                            static_cast<float>(configuration.grid_size) / 2) *
                                           (2 * M_PI / image_size);
                    const float w_offset = 2 * M_PI * w_offset_in_lambda;

                    // Iterate all timesteps
                    for (int time = 0; time < nr_timesteps; time++)
                    {
                        // Load UVW coordinates
                        float u = uvw[time_offset + time].u;
                        float v = uvw[time_offset + time].v;
                        float w = uvw[time_offset + time].w;

                        // Iterate all channels
                        for (int chan = 0; chan < configuration.nchannels; chan++)
                        {
                            // Update all polarizations
                            std::array<std::complex<float>, n_correlations> sum;
                            sum.fill(0);

                            // Iterate all pixels in subgrid
                            for (int y = 0; y < subgrid_size; y++)
                            {
                                for (int x = 0; x < subgrid_size; x++)
                                {
                                    // Compute l,m,n
                                    const float l = compute_l(x, subgrid_size, image_size);
                                    const float m = compute_m(y, subgrid_size, image_size);
                                    const float n = compute_n(l, m);

                                    // Compute phase index
                                    float phase_index = u * l + v * m + w * n;

                                    // Compute phase offset
                                    float phase_offset = u_offset * l + v_offset * m + w_offset * n;

                                    // Compute phase
                                    float phase = (phase_index * wavenumbers[chan]) - phase_offset;

                                    // Compute phasor
                                    std::complex<float> phasor = {cosf(phase), sinf(phase)};

                                    for (int pol = 0; pol < n_correlations; pol++)
                                    {
                                        sum[pol] += pixels[y][x][pol] * phasor;
                                    }
                                } // end for x
                            }     // end for y

                            size_t index = (time_offset + time) * configuration.nchannels + chan;
                            for (int pol = 0; pol < n_correlations; pol++)
                            {
                                reinterpret_cast<std::complex<float> *>(
                                    visibilities.data())[index * n_correlations + pol] = sum[pol];
                            }
                        } // end for channel
                    }     // end for time
                }
#ifdef ENABLE_TBB
                }
            });
#endif
            const auto end = std::chrono::high_resolution_clock::now();
            const auto compute_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000.;

#ifdef VTUNE_PROFILE
            __itt_task_end(domain);
            __itt_pause();
#endif

            benchmark_result result;
            result.compute_time = compute_us;
            return result;
    }

    benchmark_result launch_degridding(const configuration &configuration)
    {
            const auto n_baselines = (configuration.nstations * (configuration.nstations - 1)) / 2;
            const auto n_subgrids = n_baselines * configuration.ntimeslots;
            const auto image_size = 0.1F;

            Array2D<UVWCoordinate<float>> uvw(n_subgrids, configuration.ntimesteps_per_subgrid);
            Array3D<Visibility<std::complex<float>>> visibilities(n_subgrids, configuration.ntimesteps_per_subgrid,
                                                                  configuration.nchannels);
            Array1D<Baseline> baselines(n_baselines);
            Array4D<Matrix2x2<std::complex<float>>> aterms(configuration.ntimeslots, configuration.nstations,
                                                           configuration.subgrid_size, configuration.subgrid_size);
            Array1D<float> frequencies(configuration.nchannels);
            Array1D<float> wavenumbers(configuration.nchannels);
            Array2D<float> spheroidal(configuration.subgrid_size, configuration.subgrid_size);
            Array4D<std::complex<float>> subgrids(n_subgrids, n_correlations, configuration.subgrid_size,
                                                  configuration.subgrid_size);
            Array1D<Metadata> metadata(n_subgrids);

            // Initialize data structures
            srand(0);
            initialize_uvw(configuration.grid_size, uvw);
            initialize_frequencies(frequencies);
            initialize_wavenumbers(frequencies, wavenumbers);
            initialize_visibilities(configuration.grid_size, image_size, frequencies, uvw, visibilities);
            initialize_baselines(configuration.nstations, baselines);
            initialize_spheroidal(spheroidal);
            initialize_aterms(spheroidal, aterms);
            initialize_metadata(configuration.grid_size, configuration.ntimeslots, configuration.ntimesteps_per_subgrid,
                                baselines, metadata);

            if (configuration.htype == benchmarks_common::hardware_type::cpu)
            {
                return degridding_benchmark_launcher<benchmarks_common::hardware_type::cpu>::launch(
                    configuration, uvw, visibilities, baselines, aterms, frequencies, wavenumbers, spheroidal, subgrids,
                    metadata);
            }
            else if (configuration.htype == benchmarks_common::hardware_type::gpu)
            {
                return degridding_benchmark_launcher<benchmarks_common::hardware_type::gpu>::launch(
                    configuration, uvw, visibilities, baselines, aterms, frequencies, wavenumbers, spheroidal, subgrids,
                    metadata);
            }
            return {};
    }
    } // namespace gridding_benchmark
