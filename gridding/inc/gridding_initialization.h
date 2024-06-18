#pragma once
#include "types.hpp"

namespace gridding_benchmark
{
    void initialize_uvw(unsigned int grid_size, Array2D<UVWCoordinate<float>> &uvw);

    void initialize_frequencies(Array1D<float> &frequencies);

    void initialize_wavenumbers(const Array1D<float> &frequencies, Array1D<float> &wavenumbers);

    void initialize_visibilities(unsigned int grid_size, float image_size, const Array1D<float> &frequencies,
                                 const Array2D<UVWCoordinate<float>> &uvw,
                                 Array3D<Visibility<std::complex<float>>> &visibilities);

    void initialize_baselines(unsigned int nr_stations, Array1D<Baseline> &baselines);

    void initialize_spheroidal(Array2D<float> &spheroidal);

    void initialize_aterms(const Array2D<float> &spheroidal, Array4D<Matrix2x2<std::complex<float>>> &aterms);

    void initialize_metadata(unsigned int grid_size, unsigned int nr_timeslots, unsigned int nr_timesteps_subgrid,
                             const Array1D<Baseline> &baselines, Array1D<Metadata> &metadata);

    void initialize_subgrids(Array4D<std::complex<float>> &subgrids);

    void initialize_uvw_offsets(unsigned int subgrid_size, unsigned int grid_size, float image_size, float w_step,
                                const Array1D<Metadata> &metadata, Array2D<float> &uvw_offsets);

    void initialize_lmn(float image_size, Array3D<float> &lmn);
} // namespace gridding_benchmark
