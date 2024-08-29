#pragma once
#include "math.cuh"
#include "math.hpp"

namespace gridding_benchmark
{
    inline __device__ long index_subgrid_(int subgrid_size, int s, int pol, int y, int x)
    {
        // subgrid: [nr_subgrids][NR_POLARIZATIONS][subgrid_size][subgrid_size]
        return s * n_correlations * subgrid_size * subgrid_size + pol * subgrid_size * subgrid_size + y * subgrid_size +
               x;
    }

    inline __device__ int index_visibility(int nr_channels, int time, int chan, int pol)
    {
        // visibilities: [nr_time][nr_channels][nr_polarizations]
        return time * nr_channels * n_correlations + chan * n_correlations + pol;
    }

    inline __device__ int index_aterm(int subgrid_size, int nr_stations, int aterm_index, int station, int y, int x)
    {
        // aterm: [nr_aterms][subgrid_size][subgrid_size][n_correlations]
        int aterm_nr = (aterm_index * nr_stations + station);
        return aterm_nr * subgrid_size * subgrid_size * n_correlations + y * subgrid_size * n_correlations +
               x * n_correlations;
    }

    inline __device__ void read_aterm(int subgrid_size, int nr_stations, int aterm_index, int station, int y, int x,
                                      const float2 *aterms_ptr, float2 *aXX, float2 *aXY, float2 *aYX, float2 *aYY)
    {
        int station_idx = index_aterm(subgrid_size, nr_stations, aterm_index, station, y, x);
        float4 *aterm_ptr = (float4 *)&aterms_ptr[station_idx];
        float4 atermA = aterm_ptr[0];
        float4 atermB = aterm_ptr[1];
        *aXX = make_float2(atermA.x, atermA.y);
        *aXY = make_float2(atermA.z, atermA.w);
        *aYX = make_float2(atermB.x, atermB.y);
        *aYY = make_float2(atermB.z, atermB.w);
    }

    inline __device__ void apply_aterm(const float2 aXX1, const float2 aXY1, const float2 aYX1, const float2 aYY1,
                                       const float2 aXX2, const float2 aXY2, const float2 aYX2, const float2 aYY2,
                                       float2 pixels[n_correlations])
    {
        float2 pixelsXX = pixels[0];
        float2 pixelsXY = pixels[1];
        float2 pixelsYX = pixels[2];
        float2 pixelsYY = pixels[3];

        // Apply aterm to subgrid: P = A1 * P
        // [ pixels[0], pixels[1];  = [ aXX1, aXY1;  [ pixelsXX, pixelsXY;
        //   pixels[2], pixels[3] ]     aYX1, aYY1 ]   pixelsYX], pixelsYY ] *
        pixels[0] = (pixelsXX * aXX1);
        pixels[0] += (pixelsYX * aXY1);
        pixels[1] = (pixelsXY * aXX1);
        pixels[1] += (pixelsYY * aXY1);
        pixels[2] = (pixelsXX * aYX1);
        pixels[2] += (pixelsYX * aYY1);
        pixels[3] = (pixelsXY * aYX1);
        pixels[3] += (pixelsYY * aYY1);

        pixelsXX = pixels[0];
        pixelsXY = pixels[1];
        pixelsYX = pixels[2];
        pixelsYY = pixels[3];

        // Apply aterm to subgrid: P = P * A2^H
        //    [ pixels[0], pixels[1];  =   [ pixelsXX, pixelsXY;  *  [ conj(aXX2),
        //    conj(aYX2);
        //      pixels[2], pixels[3] ]       pixelsYX, pixelsYY ]      conj(aXY2),
        //      conj(aYY2) ]
        pixels[0] = (pixelsXX * conj(aXX2));
        pixels[0] += (pixelsXY * conj(aXY2));
        pixels[1] = (pixelsXX * conj(aYX2));
        pixels[1] += (pixelsXY * conj(aYY2));
        pixels[2] = (pixelsYX * conj(aXX2));
        pixels[2] += (pixelsYY * conj(aXY2));
        pixels[3] = (pixelsYX * conj(aYX2));
        pixels[3] += (pixelsYY * conj(aYY2));
    }

    inline __device__ void apply_aterm_degridder(float2 *pixels, const float2 *aterm1, const float2 *aterm2)
    {
        // Apply aterm: P = A1 * P
        float2 temp[4];
        matmul(aterm1, pixels, temp);

        // Aterm 2 hermitian
        float2 aterm2_h[4];
        hermitian(aterm2, aterm2_h);

        // Apply aterm: P = P * A2^H
        matmul(temp, aterm2_h, pixels);
    }

    inline __device__ void apply_aterm(const float2 aXX1, const float2 aXY1, const float2 aYX1, const float2 aYY1,
                                       const float2 aXX2, const float2 aXY2, const float2 aYX2, const float2 aYY2,
                                       float2 &uvXX, float2 &uvXY, float2 &uvYX, float2 &uvYY)
    {
        float2 uv[n_correlations] = {uvXX, uvXY, uvYX, uvYY};

        apply_aterm(aXX1, aXY1, aYX1, aYY1, aXX2, aXY2, aYX2, aYY2, uv);

        uvXX = uv[0];
        uvXY = uv[1];
        uvYX = uv[2];
        uvYY = uv[3];
    }
} // namespace gridding_benchmark
