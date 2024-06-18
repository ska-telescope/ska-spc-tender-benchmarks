#include "gridding_benchmark.h"
#include "gridding_configuration.h"
#include "gridding_initialization.h"
#include "math.hpp"

#include <chrono>
#include <complex>
#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif
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

#ifdef VTUNE_PROFILE
        __itt_resume();
        __itt_domain *domain = __itt_domain_create("Gridding.Benchmark");
        __itt_string_handle *handle_main = __itt_string_handle_create("run");
        __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
#endif

        const auto begin = std::chrono::high_resolution_clock::now();
        const auto subgrid_size = configuration.subgrid_size;
#if defined(ENABLE_TBB)
        tbb::parallel_for(
            tbb::blocked_range3d<size_t>(0, n_subgrids, 0, subgrid_size, 0, subgrid_size),
            [&](const tbb::blocked_range3d<size_t> &range) {
                for (auto s = range.pages().begin(); s < range.pages().end(); ++s)
#else
        for (size_t s = 0; s < n_subgrids; ++s)
#endif
                {
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
#ifdef ENABLE_TBB
                    for (auto y = range.pages().begin(); y < range.pages().end(); ++y)
                    {
                        for (auto x = range.pages().begin(); x < range.pages().end(); ++x)
#else
#ifdef ENABLE_OMP
#pragma omp parallel for collapse(2)
#endif
            for (size_t y = 0; y < subgrid_size; ++y)
            {
                for (size_t x = 0; x < subgrid_size; ++x)
#endif
                        {
                            // Initialize pixel for every polarization
                            std::array<std::complex<float>, n_correlations> pixels;
                            pixels.fill(0);

                            // Compute l,m,n
                            float l = compute_l(x, subgrid_size, image_size);
                            float m = compute_m(y, subgrid_size, image_size);
                            float n = compute_n(l, m);

                            // Iterate all timesteps
                            for (int time = 0; time < nr_timesteps; time++)
                            {
                                // Load UVW coordinates
                                float u = uvw[time_offset + time].u;
                                float v = uvw[time_offset + time].v;
                                float w = uvw[time_offset + time].w;

                                // Compute phase index
                                float phase_index = u * l + v * m + w * n;

                                // Compute phase offset
                                float phase_offset = u_offset * l + v_offset * m + w_offset * n;

                                // Update pixel for every channel
                                for (int chan = 0; chan < nchannels; chan++)
                                {
                                    // Compute phase
                                    float phase = phase_offset - (phase_index * wavenumbers[chan]);

                                    // Compute phasor
                                    std::complex<float> phasor = {cosf(phase), sinf(phase)};

                                    // Update pixel for every polarization
                                    size_t index = (time_offset + time) * nchannels + chan;
                                    const auto *visibilities_cpx_ptr =
                                        reinterpret_cast<const std::complex<float> *>(visibilities.data() + index);
                                    for (int pol = 0; pol < n_correlations; pol++)
                                    {
                                        pixels[pol] += visibilities_cpx_ptr[pol] * phasor;
                                    }
                                } // end for chan
                            }     // end for time

                            // Load a term for station1
                            int station1_index =
                                (aterm_index * nstations + station1) * subgrid_size * subgrid_size * n_correlations +
                                y * subgrid_size * n_correlations + x * n_correlations;
                            const std::complex<float> *aterm1_ptr =
                                &reinterpret_cast<const std::complex<float> *>(aterms.data())[station1_index];

                            // Load aterm for station2
                            int station2_index =
                                (aterm_index * nstations + station2) * subgrid_size * subgrid_size * n_correlations +
                                y * subgrid_size * n_correlations + x * n_correlations;
                            const std::complex<float> *aterm2_ptr =
                                &reinterpret_cast<const std::complex<float> *>(aterms.data())[station2_index];

                            // Apply aterm
                            apply_aterm_gridder(&pixels[0], aterm1_ptr, aterm2_ptr);

                            // Load spheroidal
                            float sph = spheroidal[y * subgrid_size + x];

                            // Set subgrid value
                            for (int pol = 0; pol < n_correlations; pol++)
                            {
                                unsigned idx_subgrid = s * n_correlations * subgrid_size * subgrid_size +
                                                       pol * subgrid_size * subgrid_size + y * subgrid_size + x;
                                subgrids[idx_subgrid] = pixels[pol] * sph;
                            }
                        }
                    }
                }
#ifdef ENABLE_TBB
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

    benchmark_result launch_gridding(const configuration &configuration)
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
            return gridding_benchmark_launcher<benchmarks_common::hardware_type::cpu>::launch(configuration, uvw, visibilities, baselines,
                                                                           aterms, frequencies, wavenumbers, spheroidal,
                                                                           subgrids, metadata);
        }
        else if (configuration.htype == benchmarks_common::hardware_type::gpu)
        {
            return gridding_benchmark_launcher<benchmarks_common::hardware_type::gpu>::launch(configuration, uvw, visibilities, baselines,
                                                                           aterms, frequencies, wavenumbers, spheroidal,
                                                                           subgrids, metadata);
        }
        return {};
    }
} // namespace gridding_benchmark
