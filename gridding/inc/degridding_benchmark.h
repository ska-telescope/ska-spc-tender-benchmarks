#pragma once
#include <complex>

#include "common_gridding.h"
#include "gridding_configuration.h"
#include "types.hpp"

#ifdef __CUDA_ARCH__
#define FUNCTION_ATTRIBUTES __device__
#endif

#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

namespace gridding_benchmark
{
    template <benchmarks_common::hardware_type htype>
    struct degridding_benchmark_launcher
    {
        static benchmark_result launch(const gridding_benchmark::configuration &configuration,
                                       Array2D<UVWCoordinate<float>> &uvw,
                                       Array3D<Visibility<std::complex<float>>> &visibilities,
                                       Array1D<Baseline> &baselines, Array4D<Matrix2x2<std::complex<float>>> &aterms,
                                       Array1D<float> &frequencies, Array1D<float> &wavenumbers,
                                       Array2D<float> &spheroidal, Array4D<std::complex<float>> &subgrids,
                                       Array1D<Metadata> &metadata);
    };

    benchmark_result launch_degridding(const configuration &configuration);
} // namespace gridding_benchmark
