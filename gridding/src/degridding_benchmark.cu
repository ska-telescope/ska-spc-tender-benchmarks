#include "common_gridding.cuh"
#include "degridding_benchmark.h"
#include "math.cuh"
#include "math.hpp"
#include "util.cuh"

#define ALIGN(N, A) (((N) + (A)-1) / (A) * (A))
#define NUM_THREADS 128

namespace gridding_benchmark
{
    // Storage
    __shared__ float4 shared_v6[3][NUM_THREADS];

    __device__ void prepare_shared(const int current_nr_pixels, const int pixel_offset, const int nr_polarizations,
                                   const int grid_size, const int subgrid_size, const float image_size,
                                   const int nr_stations, const int aterm_idx, const Metadata &metadata,
                                   const float *spheroidal, const float2 *aterms, const float2 *subgrid)
    {
        int s = blockIdx.x;
        int num_threads = blockDim.x;
        int tid = threadIdx.x;

        // Load metadata for current subgrid
        const int x_coordinate = metadata.coordinate.x;
        const int y_coordinate = metadata.coordinate.y;
        const int station1 = metadata.baseline.station1;
        const int station2 = metadata.baseline.station2;

        // Compute u,v,w offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
        const float w_offset = (metadata.coordinate.z + 0.5) * 2 * M_PI;

        for (int j = tid; j < current_nr_pixels; j += num_threads)
        {
            int y = (pixel_offset + j) / subgrid_size;
            int x = (pixel_offset + j) % subgrid_size;

            // Load spheroidal
            const float spheroidal_ = spheroidal[y * subgrid_size + x];

            // Load pixels
            float2 pixel[4];
            for (unsigned pol = 0; pol < nr_polarizations; pol++)
            {
                unsigned int pixel_idx = index_subgrid(subgrid_size, s, pol, y, x);
                pixel[pol] = subgrid[pixel_idx] * spheroidal_;
            }

            // Apply aterm
            int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station1, y, x);
            int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station2, y, x);
            float2 *aterm1 = (float2 *)&aterms[station1_idx];
            float2 *aterm2 = (float2 *)&aterms[station2_idx];
            apply_aterm_degridder(pixel, aterm1, aterm2);

            // Store pixels in shared memory
            shared_v6[0][j] = make_float4(pixel[0].x, pixel[0].y, pixel[1].x, pixel[1].y);
            shared_v6[1][j] = make_float4(pixel[2].x, pixel[2].y, pixel[3].x, pixel[3].y);

            // Compute l,m,n for phase offset and phase index
            const float l = compute_l(x, subgrid_size, image_size);
            const float m = compute_m(y, subgrid_size, image_size);
            const float n = compute_n(l, m);
            const float phase_offset = -(u_offset * l + v_offset * m + w_offset * n);

            // Store l_index,m_index,n and phase offset in shared memory
            shared_v6[2][j] = make_float4(l, m, n, phase_offset);
        } // end for j (pixels)
    }

    __device__ void cmac(float2 &a, float2 b, float2 c)
    {
        a.x = fma(b.x, c.x, a.x);
        a.y = fma(b.x, c.y, a.y);
        a.x = fma(-b.y, c.y, a.x);
        a.y = fma(b.y, c.x, a.y);
    }

    __device__ void compute_visibility(const int nr_polarizations, const int current_nr_pixels, const int channel,
                                       const float u, const float v, const float w, const float *wavenumbers,
                                       float2 *visibility)
    {
        for (int i = 0; i < current_nr_pixels; i++)
        {
            // Compute phase_offset and phase index
            float l = shared_v6[2][i].x;
            float m = shared_v6[2][i].y;
            float n = shared_v6[2][i].z;
            float phase_offset = shared_v6[2][i].w;
            float phase_index = u * l + v * m + w * n;

            // Compute visibility
            const float4 a = shared_v6[0][i];
            const float4 b = shared_v6[1][i];
            float phase = wavenumbers[channel] * phase_index + phase_offset;
            float2 phasor = make_float2(__cosf(phase), __sinf(phase));
            cmac(visibility[0], phasor, make_float2(a.x, a.y));
            cmac(visibility[1], phasor, make_float2(a.z, a.w));
            cmac(visibility[2], phasor, make_float2(b.x, b.y));
            cmac(visibility[3], phasor, make_float2(b.z, b.w));
        } // end for k (batch)
    }

    __global__ void kernel_degridder_v6(int grid_size, int subgrid_size, float image_size, float w_step_in_lambda,
                                        int nr_channels, int nr_stations, UVWCoordinate<float> *uvw, float *wavenumbers,
                                        float2 *visibilities, float *spheroidal, float2 *aterms, Metadata *metadata,
                                        float2 *subgrids)
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int tid = tidx + tidy * blockDim.x;
        int s = blockIdx.x;

        // Find offset of first subgrid
        const Metadata m_0 = metadata[0];
        const int baseline_offset_1 = m_0.baseline_offset;

        // Load metadata for current subgrid
        const Metadata &m = metadata[s];
        const int time_offset = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index = m.aterm_index;

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time += NUM_THREADS)
        {
            int index_time = time + tid;

            // Iterate all channels
            for (int chan = 0; chan < nr_channels; chan++)
            {
                size_t index = (time_offset + index_time) * nr_channels + chan;
                for (int pol = 0; pol < n_correlations; pol++)
                {
                    visibilities[index * n_correlations + pol] = make_float2(0, 0);
                }
            } // end for chan
        }     // end for time

        // Iterate pixels
        const int nr_pixels = subgrid_size * subgrid_size;
        const int batch_size = NUM_THREADS;
        int current_nr_pixels = batch_size;
        for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels)
        {
            current_nr_pixels = nr_pixels - pixel_offset < min(NUM_THREADS, batch_size) ? nr_pixels - pixel_offset
                                                                                        : min(NUM_THREADS, batch_size);

            // Iterate timesteps
            for (int time = 0; time < nr_timesteps; time += NUM_THREADS)
            {

                __syncthreads();

                // Prepare data
                prepare_shared(current_nr_pixels, pixel_offset, 4, grid_size, subgrid_size, image_size, nr_stations,
                               aterm_index, m, spheroidal, aterms, subgrids);

                __syncthreads();

                // Determine the first and last timestep to process
                int time_start = time_offset + time;
                int time_end = time_start + NUM_THREADS;

                for (int i = tid; i < ALIGN(NUM_THREADS * nr_channels, NUM_THREADS); i += NUM_THREADS)
                {
                    int time = time_start + (i / nr_channels);
                    int channel = (i % nr_channels);

                    float2 visibility[4];

                    for (int pol = 0; pol < 4; pol++)
                    {
                        visibility[pol] = make_float2(0, 0);
                    }

                    float u, v, w;

                    if (time < time_end)
                    {
                        u = uvw[time].u;
                        v = uvw[time].v;
                        w = uvw[time].w;
                    }

                    // Compute visibility
                    compute_visibility(4, current_nr_pixels, channel, u, v, w, wavenumbers, visibility);

                    // Update visibility
                    if (time < time_end)
                    {
                        size_t index = time * nr_channels + channel;
                        for (int pol = 0; pol < n_correlations; pol++)
                        {
                            visibilities[index * n_correlations + pol] += visibility[pol];
                        }
                    }
                } // end for time
            }     // end for time_offset_local
        }         // end for pixel_offset
    }

    template <>
    benchmark_result degridding_benchmark_launcher<benchmarks_common::hardware_type::gpu>::launch(
        const gridding_benchmark::configuration &configuration, Array2D<UVWCoordinate<float>> &uvw,
        Array3D<Visibility<std::complex<float>>> &visibilities, Array1D<Baseline> &baselines,
        Array4D<Matrix2x2<std::complex<float>>> &aterms, Array1D<float> &frequencies, Array1D<float> &wavenumbers,
        Array2D<float> &spheroidal, Array4D<std::complex<float>> &subgrids, Array1D<Metadata> &metadata)
    {
        size_t total_in_size = 0;

        UVWCoordinate<float> *d_uvw;
        const auto d_uvw_size = uvw.size() * sizeof(uvw[0]);
        total_in_size += d_uvw_size;
        cudaCheck(cudaMalloc(&d_uvw, d_uvw_size));

        float *d_wavenumbers;
        const auto d_wavenumbers_size = wavenumbers.size() * sizeof(wavenumbers[0]);
        total_in_size += d_wavenumbers_size;
        cudaCheck(cudaMalloc(&d_wavenumbers, d_wavenumbers_size));

        float2 *d_visibilities;
        const auto d_visibilities_size = visibilities.size() * sizeof(visibilities[0]);
        total_in_size += d_visibilities_size;
        cudaCheck(cudaMalloc(&d_visibilities, d_visibilities_size));

        float *d_spheroidal;
        const auto d_spheroidal_size = spheroidal.size() * sizeof(spheroidal[0]);
        total_in_size += d_spheroidal_size;
        cudaCheck(cudaMalloc(&d_spheroidal, d_spheroidal_size));

        float2 *d_aterms;
        const auto d_aterms_size = aterms.size() * sizeof(aterms[0]);
        total_in_size += d_aterms_size;
        cudaCheck(cudaMalloc(&d_aterms, d_aterms_size));

        Metadata *d_metadata;
        const auto d_metadata_size = metadata.size() * sizeof(metadata[0]);
        total_in_size += d_metadata_size;
        cudaCheck(cudaMalloc(&d_metadata, d_metadata_size));

        float2 *d_subgrids;
        const auto d_subgrids_size = subgrids.size() * sizeof(subgrids[0]);
        cudaCheck(cudaMalloc(&d_subgrids, d_subgrids_size));
        const auto total_out_size = d_subgrids_size;

        cudaEvent_t begin_in_memcpy;
        cudaCheck(cudaEventCreate(&begin_in_memcpy));
        cudaEvent_t end_in_memcpy;
        cudaCheck(cudaEventCreate(&end_in_memcpy));

        cudaCheck(cudaEventRecord(begin_in_memcpy));
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            cudaCheck(cudaMemcpy(d_uvw, uvw.data(), d_uvw_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_wavenumbers, wavenumbers.data(), d_wavenumbers_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_visibilities, visibilities.data(), d_visibilities_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_spheroidal, spheroidal.data(), d_spheroidal_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_aterms, aterms.data(), d_aterms_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_metadata, metadata.data(), d_metadata_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaEventRecord(end_in_memcpy));
        }
        cudaCheck(cudaEventSynchronize(end_in_memcpy));

        float in_copy_time_ms;
        cudaCheck(cudaEventElapsedTime(&in_copy_time_ms, begin_in_memcpy, end_in_memcpy));

        const auto n_baselines = (configuration.nstations * (configuration.nstations - 1)) / 2;
        const auto n_subgrids = n_baselines * configuration.ntimeslots;
        const auto w_step_in_lambda = 0;
        const auto image_size = 0.1F;

        cudaEvent_t begin_compute;
        cudaCheck(cudaEventCreate(&begin_compute));
        cudaEvent_t end_compute;
        cudaCheck(cudaEventCreate(&end_compute));

        cudaCheck(cudaEventRecord(begin_compute));
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            kernel_degridder_v6<<<n_subgrids, NUM_THREADS>>>(
                configuration.grid_size, configuration.subgrid_size, image_size, w_step_in_lambda,
                configuration.nchannels, configuration.nstations, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                d_aterms, d_metadata, d_subgrids);
        }
        cudaCheck(cudaEventRecord(end_compute));
        cudaCheck(cudaEventSynchronize(end_compute));

        float compute_time_ms;
        cudaCheck(cudaEventElapsedTime(&compute_time_ms, begin_compute, end_compute));

        cudaEvent_t begin_out_memcpy;
        cudaCheck(cudaEventCreate(&begin_out_memcpy));
        cudaEvent_t end_out_memcpy;
        cudaCheck(cudaEventCreate(&end_out_memcpy));

        cudaCheck(cudaEventRecord(begin_out_memcpy));
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            cudaCheck(cudaMemcpy(subgrids.data(), d_subgrids, d_subgrids_size, cudaMemcpyDeviceToHost));
        }
        cudaCheck(cudaEventRecord(end_out_memcpy));
        cudaCheck(cudaEventSynchronize(end_out_memcpy));

        float out_copy_time_ms;
        cudaCheck(cudaEventElapsedTime(&out_copy_time_ms, begin_out_memcpy, end_out_memcpy));

        benchmark_result result;
        result.in_transfer_time = in_copy_time_ms * 1000.f / static_cast<float>(configuration.niterations);
        result.out_transfer_time = out_copy_time_ms * 1000.f / static_cast<float>(configuration.niterations);
        result.in_bandwidth = static_cast<float>(total_in_size) / (1000.f * in_copy_time_ms);
        result.out_bandwidth = static_cast<float>(total_out_size) / (1000.f * out_copy_time_ms);
        result.compute_time = compute_time_ms * 1000.f / static_cast<float>(configuration.niterations);

        return result;
    }
} // namespace gridding_benchmark
