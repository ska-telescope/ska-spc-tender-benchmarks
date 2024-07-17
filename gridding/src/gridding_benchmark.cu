#include "common_gridding.cuh"
#include "gridding_benchmark.h"
#include "math.cuh"
#include "math.hpp"
#include "util.cuh"

// #define ALIGN(N, A) (((N) + (A)-1) / (A) * (A))
#define NUM_THREADS 128
#define UNROLL_PIXELS 4
#define BATCH_SIZE 128
#define MAX_NR_CHANNELS 8

namespace gridding_benchmark
{
    __shared__ float4 visibilities_v7_[BATCH_SIZE][2];
    __shared__ float4 uvw_v7_[BATCH_SIZE];
    __shared__ float wavenumbers_v7_[MAX_NR_CHANNELS];

    template <int current_nr_channels>
    __device__ void kernel_gridder_v7_(const int grid_size, int subgrid_size, float image_size, float w_step_in_lambda,
                                       int nchannels,
                                       int channel_offset, // channel_offset? for the macro?
                                       int nstations, const UVWCoordinate<float> *uvw, const float *wavenumbers,
                                       const float2 *visibilities, const float *spheroidal, const float2 *aterms,
                                       const Metadata *metadata, float2 *subgrids)
    {
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int tid = tidx + tidy * blockDim.x;
        int nr_threads = blockDim.x * blockDim.y;
        int s = blockIdx.x;

        // Find offset of first subgrid
        const Metadata m_0 = metadata[0];
        const int baseline_offset_1 = m_0.baseline_offset;

        // Load metadata
        const Metadata m = metadata[s];
        const int time_offset_global = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index = m.aterm_index;
        const int station1 = m.baseline.station1;
        const int station2 = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Set subgrid to zero
        if (channel_offset == 0)
        {
            for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads)
            {
                int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
                int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
                int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
                int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
                subgrids[idx_xx] = make_float2(0, 0);
                subgrids[idx_xy] = make_float2(0, 0);
                subgrids[idx_yx] = make_float2(0, 0);
                subgrids[idx_yy] = make_float2(0, 0);
            }
        }

        for (int i = tid; i < current_nr_channels; i += nr_threads)
        {
            wavenumbers_v7_[i] = wavenumbers[i + channel_offset];
        }

        __syncthreads();

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
        const float w_offset = 2 * M_PI * w_offset_in_lambda;

        // Iterate all pixels in subgrid
        for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads * UNROLL_PIXELS)
        {
            float2 pixelXX[UNROLL_PIXELS];
            float2 pixelXY[UNROLL_PIXELS];
            float2 pixelYX[UNROLL_PIXELS];
            float2 pixelYY[UNROLL_PIXELS];

            for (int p = 0; p < UNROLL_PIXELS; p++)
            {
                pixelXX[p] = make_float2(0, 0);
                pixelXY[p] = make_float2(0, 0);
                pixelYX[p] = make_float2(0, 0);
                pixelYY[p] = make_float2(0, 0);
            }

            float l[UNROLL_PIXELS];
            float m[UNROLL_PIXELS];
            float n[UNROLL_PIXELS];
            float phase_offset[UNROLL_PIXELS];
            for (int p = 0; p < UNROLL_PIXELS; p++)
            {

                int x = (i + p * nr_threads) % subgrid_size;
                int y = (i + p * nr_threads) / subgrid_size;
                l[p] = compute_l(x, subgrid_size, image_size);
                m[p] = compute_m(y, subgrid_size, image_size);
                n[p] = compute_n(l[p], m[p]);
                phase_offset[p] = u_offset * l[p] + v_offset * m[p] + w_offset * n[p];
            }

            int current_nr_timesteps = BATCH_SIZE / MAX_NR_CHANNELS;
            // Iterate all timesteps
            for (int time_offset_local = 0; time_offset_local < nr_timesteps; time_offset_local += current_nr_timesteps)
            {
                current_nr_timesteps = nr_timesteps - time_offset_local < current_nr_timesteps
                                           ? nr_timesteps - time_offset_local
                                           : current_nr_timesteps;
                __syncthreads();
                for (int time = tid; time < current_nr_timesteps; time += nr_threads)
                {
                    UVWCoordinate<float> a = uvw[time_offset_global + time_offset_local + time];
                    uvw_v7_[time] = make_float4(a.u, a.v, a.w, 0);
                }

                // Load visibilities
                for (int ii = tid; ii < current_nr_timesteps * current_nr_channels * 2; ii += nr_threads)
                {
                    int j = ii % 2; // one thread loads either upper or lower float4 part of visibility
                    int k = ii / 2;
                    int idx_time = time_offset_global + time_offset_local + (k / current_nr_channels);
                    int idx_chan = channel_offset + (k % current_nr_channels);
                    int idx_vis = index_visibility(nchannels, idx_time, idx_chan, 0);
                    float4 *vis_ptr = (float4 *)&visibilities[idx_vis];
                    visibilities_v7_[k][j] = vis_ptr[j];
                }

                __syncthreads();

                for (int time = 0; time < current_nr_timesteps; time++)
                {

                    // Load UVW coordinates
                    float u = uvw_v7_[time].x;
                    float v = uvw_v7_[time].y;
                    float w = uvw_v7_[time].z;
                    for (int p = 0; p < UNROLL_PIXELS; p++)
                    {
                        // Compute phase index
                        float phase_index = u * l[p] + v * m[p] + w * n[p];

                        for (int chan = 0; chan < current_nr_channels; chan++)
                        {
                            // Compute phase
                            float phase = phase_offset[p] - (phase_index * wavenumbers_v7_[chan]);

                            // Compute phasor
                            float2 phasor = make_float2(__cosf(phase), __sinf(phase));

                            // Update pixel for every polarization

                            // Load visibilities from shared memory
                            float4 a = visibilities_v7_[time * current_nr_channels + chan][0];
                            float4 b = visibilities_v7_[time * current_nr_channels + chan][1];
                            float2 visXX = make_float2(a.x, a.y);
                            float2 visXY = make_float2(a.z, a.w);
                            float2 visYX = make_float2(b.x, b.y);
                            float2 visYY = make_float2(b.z, b.w);

                            pixelXX[p].x += phasor.x * visXX.x;
                            pixelXX[p].y += phasor.x * visXX.y;
                            pixelXX[p].x -= phasor.y * visXX.y;
                            pixelXX[p].y += phasor.y * visXX.x;

                            pixelXY[p].x += phasor.x * visXY.x;
                            pixelXY[p].y += phasor.x * visXY.y;
                            pixelXY[p].x -= phasor.y * visXY.y;
                            pixelXY[p].y += phasor.y * visXY.x;

                            pixelYX[p].x += phasor.x * visYX.x;
                            pixelYX[p].y += phasor.x * visYX.y;
                            pixelYX[p].x -= phasor.y * visYX.y;
                            pixelYX[p].y += phasor.y * visYX.x;

                            pixelYY[p].x += phasor.x * visYY.x;
                            pixelYY[p].y += phasor.x * visYY.y;
                            pixelYY[p].x -= phasor.y * visYY.y;
                            pixelYY[p].y += phasor.y * visYY.x;
                        }
                    }
                }
            }

            for (int p = 0; p < UNROLL_PIXELS; p++)
            {
                int x = (i + p * nr_threads) % subgrid_size;
                int y = (i + p * nr_threads) / subgrid_size;

                // Load aterm for station1
                float2 aXX1, aXY1, aYX1, aYY1;
                read_aterm(subgrid_size, nstations, aterm_index, station1, y, x, aterms, &aXX1, &aXY1, &aYX1, &aYY1);

                // Load aterm for station2
                float2 aXX2, aXY2, aYX2, aYY2;
                read_aterm(subgrid_size, nstations, aterm_index, station2, y, x, aterms, &aXX2, &aXY2, &aYX2, &aYY2);

                // Apply the conjugate transpose of the A-term
                apply_aterm(conj(aXX1), conj(aYX1), conj(aXY1), conj(aYY1), conj(aXX2), conj(aYX2), conj(aXY2),
                            conj(aYY2), pixelXX[p], pixelXY[p], pixelYX[p], pixelYY[p]);

                // Load a term for station1
                // Load spheroidal
                float sph = spheroidal[y * subgrid_size + x];

                int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i + p * nr_threads);
                int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i + p * nr_threads);
                int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i + p * nr_threads);
                int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i + p * nr_threads);

                subgrids[idx_xx] += pixelXX[p] * sph;
                subgrids[idx_xy] += pixelXY[p] * sph;
                subgrids[idx_yx] += pixelYX[p] * sph;
                subgrids[idx_yy] += pixelYY[p] * sph;
            }
        }
    }

#define KERNEL_GRIDDER_TEMPLATE(current_nr_channels)                                                                   \
    for (; (channel_offset + current_nr_channels) <= nr_channels; channel_offset += current_nr_channels)               \
    {                                                                                                                  \
        kernel_gridder_v7_<current_nr_channels>(grid_size, subgrid_size, image_size, w_step_in_lambda, nr_channels,    \
                                                channel_offset, nr_stations, uvw, wavenumbers, visibilities,           \
                                                spheroidal, aterms, metadata, subgrids);                               \
    }

    __global__ void kernel_gridder_v7(const int grid_size, int subgrid_size, float image_size, float w_step_in_lambda,
                                      int nr_channels, // channel_offset? for the macro?
                                      int nr_stations, UVWCoordinate<float> *uvw, float *wavenumbers,
                                      float2 *visibilities, float *spheroidal, float2 *aterms, Metadata *metadata,
                                      float2 *subgrids)
    {
        int channel_offset = 0;
        KERNEL_GRIDDER_TEMPLATE(8);
        KERNEL_GRIDDER_TEMPLATE(7);
        KERNEL_GRIDDER_TEMPLATE(6);
        KERNEL_GRIDDER_TEMPLATE(5);
        KERNEL_GRIDDER_TEMPLATE(4);
        KERNEL_GRIDDER_TEMPLATE(3);
        KERNEL_GRIDDER_TEMPLATE(2);
        KERNEL_GRIDDER_TEMPLATE(1);
    }

    template <>
    benchmark_result gridding_benchmark_launcher<benchmarks_common::hardware_type::gpu>::launch(
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

        float in_copy_time;
        cudaCheck(cudaEventElapsedTime(&in_copy_time, begin_in_memcpy, end_in_memcpy));

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
            kernel_gridder_v7<<<n_subgrids, NUM_THREADS>>>(
                configuration.grid_size, configuration.subgrid_size, image_size, w_step_in_lambda,
                configuration.nchannels, configuration.nstations, d_uvw, d_wavenumbers, d_visibilities, d_spheroidal,
                d_aterms, d_metadata, d_subgrids);
        }
        cudaCheck(cudaEventRecord(end_compute));
        cudaCheck(cudaEventSynchronize(end_compute));

        float compute_time;
        cudaCheck(cudaEventElapsedTime(&compute_time, begin_compute, end_compute));

        cudaEvent_t begin_out_memcpy;
        cudaCheck(cudaEventCreate(&begin_out_memcpy));
        cudaEvent_t end_out_memcpy;
        cudaCheck(cudaEventCreate(&end_out_memcpy));

        cudaCheck(cudaEventRecord(begin_out_memcpy));
        for (size_t i = 0; i < configuration.niterations; ++i)
        {
            cudaCheck(cudaMemcpy(d_subgrids, subgrids.data(), d_subgrids_size, cudaMemcpyHostToDevice));
        }
        cudaCheck(cudaEventRecord(end_out_memcpy));

        float out_copy_time;
        cudaCheck(cudaEventElapsedTime(&out_copy_time, begin_out_memcpy, end_out_memcpy));

        benchmark_result result;
        result.in_transfer_time = in_copy_time / static_cast<float>(configuration.niterations);
        result.out_transfer_time = out_copy_time / static_cast<float>(configuration.niterations);
        result.in_bandwidth = static_cast<float>(total_in_size) / (1000.f * in_copy_time);
        result.out_bandwidth = static_cast<float>(total_out_size) / (1000.f * out_copy_time);
        result.compute_time = compute_time / static_cast<float>(configuration.niterations);

        return result;
    }
} // namespace gridding_benchmark
