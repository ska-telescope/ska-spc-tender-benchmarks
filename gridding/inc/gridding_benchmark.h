#pragma once
#include <complex>

#include "common_gridding.h"
#include "gridding_configuration.h"
#include "types.hpp"

namespace gridding_benchmark
{
    template <benchmarks_common::hardware_type htype>
    struct gridding_benchmark_launcher
    {
        // template <gridding_benchmark::float_type ftype>
        static benchmark_result launch(const gridding_benchmark::configuration &configuration,
                                       Array2D<UVWCoordinate<float>> &uvw,
                                       Array3D<Visibility<std::complex<float>>> &visibilities,
                                       Array1D<Baseline> &baselines, Array4D<Matrix2x2<std::complex<float>>> &aterms,
                                       Array1D<float> &frequencies, Array1D<float> &wavenumbers,
                                       Array2D<float> &spheroidal, Array4D<std::complex<float>> &subgrids,
                                       Array1D<Metadata> &metadata);
    };

    benchmark_result launch_gridding(const configuration &configuration);
} // namespace gridding_benchmark
