#include "gridding_initialization.h"
#include "math.hpp"
#include "types.hpp"

namespace gridding_benchmark
{
    float rand_normalized_float()
    {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    void initialize_uvw(unsigned int grid_size, Array2D<UVWCoordinate<float>> &uvw)
    {
        size_t n_baselines = uvw.get_y_dim();
        size_t ntimesteps = uvw.get_x_dim();

        for (unsigned bl = 0; bl < n_baselines; bl++)
        {
            const auto half_grid_size = 0.5F * static_cast<float>(grid_size);

            // Get random radius
            float radius_u = half_grid_size + rand_normalized_float() * half_grid_size;
            float radius_v = half_grid_size + rand_normalized_float() * half_grid_size;

            // Evaluate elipsoid
            const auto angle_step = 1.F / static_cast<float>(ntimesteps);
            for (size_t itime = 0; itime < ntimesteps; itime++)
            {
                float angle = (static_cast<float>(itime) + 0.5) * angle_step;
                float u = radius_u * std::cos(angle * M_PI);
                float v = radius_v * std::sin(angle * M_PI);
                float w = 0;
                uvw(bl, itime) = {u, v, w};
            }
        }
    }

    void initialize_frequencies(Array1D<float> &frequencies)
    {
        unsigned int nr_channels = frequencies.get_x_dim();

        const unsigned int start_frequency = 150e6;
        const float frequency_increment = 0.7e6;
        for (unsigned i = 0; i < nr_channels; i++)
        {
            double frequency = start_frequency + frequency_increment * i;
            frequencies(i) = frequency;
        }
    }

    void initialize_wavenumbers(const Array1D<float> &frequencies, Array1D<float> &wavenumbers)
    {
        unsigned int nr_channels = frequencies.get_x_dim();

        const double speed_of_light = 299792458.0;
        for (unsigned i = 0; i < nr_channels; i++)
        {
            wavenumbers(i) = 2 * M_PI * frequencies(i) / speed_of_light;
        }
    }

    void initialize_visibilities(unsigned int grid_size, float image_size, const Array1D<float> &frequencies,
                                 const Array2D<UVWCoordinate<float>> &uvw,
                                 Array3D<Visibility<std::complex<float>>> &visibilities)
    {
        unsigned int nr_baselines = visibilities.get_z_dim();
        unsigned int nr_timesteps = visibilities.get_y_dim();
        unsigned int nr_channels = visibilities.get_x_dim();

        float x_offset = 0.6 * grid_size;
        float y_offset = 0.7 * grid_size;
        float amplitude = 1.0f;
        float l = x_offset * image_size / grid_size;
        float m = y_offset * image_size / grid_size;

        for (unsigned bl = 0; bl < nr_baselines; bl++)
        {
            for (unsigned time = 0; time < nr_timesteps; time++)
            {
                for (unsigned chan = 0; chan < nr_channels; chan++)
                {
                    const double speed_of_light = 299792458.0;
                    float u = (frequencies(chan) / speed_of_light) * uvw(bl, time).u;
                    float v = (frequencies(chan) / speed_of_light) * uvw(bl, time).v;
                    std::complex<float> value = amplitude * exp(std::complex<float>(0, -2 * M_PI * (u * l + v * m)));
                    visibilities(bl, time, chan).xx = value * 1.01f;
                    visibilities(bl, time, chan).xy = value * 1.02f;
                    visibilities(bl, time, chan).yx = value * 1.03f;
                    visibilities(bl, time, chan).yy = value * 1.04f;
                }
            }
        }
    }

    void initialize_baselines(unsigned int nr_stations, Array1D<Baseline> &baselines)
    {
        unsigned int nr_baselines = baselines.get_x_dim();

        unsigned bl = 0;
        for (unsigned station1 = 0; station1 < nr_stations; station1++)
        {
            for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++)
            {
                if (bl >= nr_baselines)
                {
                    break;
                }
                baselines(bl) = {station1, station2};
                bl++;
            }
        }
    }

    void initialize_spheroidal(Array2D<float> &spheroidal)
    {
        unsigned int subgrid_size = spheroidal.get_x_dim();

        for (unsigned y = 0; y < subgrid_size; y++)
        {
            float tmp_y = fabs(-1 + y * 2.0f / float(subgrid_size));
            for (unsigned x = 0; x < subgrid_size; x++)
            {
                float tmp_x = fabs(-1 + x * 2.0f / float(subgrid_size));
                spheroidal(y, x) = tmp_y * tmp_x;
            }
        }
    }

    void initialize_aterms(const Array2D<float> &spheroidal, Array4D<Matrix2x2<std::complex<float>>> &aterms)
    {
        unsigned int nr_timeslots = aterms.get_w_dim();
        unsigned int nr_stations = aterms.get_z_dim();
        unsigned int subgrid_size = aterms.get_y_dim();

        for (unsigned ts = 0; ts < nr_timeslots; ts++)
        {
            for (unsigned station = 0; station < nr_stations; station++)
            {
                for (unsigned y = 0; y < subgrid_size; y++)
                {
                    for (unsigned x = 0; x < subgrid_size; x++)
                    {
                        float scale = 0.8 + ((double)rand() / (double)(RAND_MAX) * 0.4);
                        float value = spheroidal(y, x) * scale;
                        Matrix2x2<std::complex<float>> aterm;
                        aterm.xx = std::complex<float>(value + 0.1, -0.1);
                        aterm.xy = std::complex<float>(value - 0.2, 0.1);
                        aterm.yx = std::complex<float>(value - 0.2, 0.1);
                        aterm.yy = std::complex<float>(value + 0.1, -0.1);
                        aterms(ts, station, y, x) = aterm;
                    }
                }
            }
        }
    }

    void initialize_metadata(unsigned int grid_size, unsigned int nr_timeslots, unsigned int nr_timesteps_subgrid,
                             const Array1D<Baseline> &baselines, Array1D<Metadata> &metadata)
    {
        unsigned int nr_baselines = baselines.get_x_dim();

        for (unsigned int bl = 0; bl < nr_baselines; bl++)
        {
            for (unsigned int ts = 0; ts < nr_timeslots; ts++)
            {
                // Metadata settings
                int baseline_offset = 0;
                int time_offset = bl * nr_timeslots * nr_timesteps_subgrid + ts * nr_timesteps_subgrid;
                int aterm_index = 0; // use the same aterm for every timeslot
                Baseline baseline = baselines(bl);
                int x = (double)rand() / (double)(RAND_MAX)*grid_size;
                int y = (double)rand() / (double)(RAND_MAX)*grid_size;
                Coordinate coordinate = {x, y};

                // Set metadata for current subgrid
                Metadata m = {baseline_offset, time_offset, (int)nr_timesteps_subgrid,
                              aterm_index,     baseline,    coordinate};
                metadata(bl * nr_timeslots + ts) = m;
            }
        }
    }

    void initialize_subgrids(Array4D<std::complex<float>> &subgrids)
    {
        unsigned int nr_subgrids = subgrids.get_w_dim();
        unsigned int nr_correlations = subgrids.get_z_dim();
        unsigned int subgrid_size = subgrids.get_y_dim();

        // Initialize subgrids
        for (unsigned s = 0; s < nr_subgrids; s++)
        {
            for (unsigned c = 0; c < nr_correlations; c++)
            {
                for (unsigned y = 0; y < subgrid_size; y++)
                {
                    for (unsigned x = 0; x < subgrid_size; x++)
                    {
                        std::complex<float> pixel_value(
                            ((y * subgrid_size + x + 1) / ((float)100 * subgrid_size * subgrid_size)), (c / 10.0f));
                        subgrids(s, c, y, x) = pixel_value;
                    }
                }
            }
        }
    }

    void initialize_uvw_offsets(unsigned int subgrid_size, unsigned int grid_size, float image_size, float w_step,
                                const Array1D<Metadata> &metadata, Array2D<float> &uvw_offsets)
    {
        unsigned int nr_subgrids = metadata.get_x_dim();

        for (unsigned int i = 0; i < nr_subgrids; i++)
        {
            Metadata m = metadata(i);
            Coordinate c = m.coordinate;

            float w_offset_in_lambda = w_step * (c.z + 0.5);
            uvw_offsets(i, 0) = ((float)c.x + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            uvw_offsets(i, 1) = ((float)c.y + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            uvw_offsets(i, 2) = 2 * M_PI * w_offset_in_lambda;
        }
    }

    void initialize_lmn(float image_size, Array3D<float> &lmn)
    {
        unsigned int height = lmn.get_z_dim();

#if defined(DEBUG)
        unsigned int width = lmn.get_y_dim();
        assert(height == width);
        assert(lmn.get_x_dim() == 3);
#endif

        unsigned int subgrid_size = height;

        for (unsigned y = 0; y < subgrid_size; y++)
        {
            for (unsigned x = 0; x < subgrid_size; x++)
            {
                float l = compute_l(x, subgrid_size, image_size);
                float m = compute_m(y, subgrid_size, image_size);
                float n = compute_n(l, m);
                lmn(y, x, 0) = l;
                lmn(y, x, 1) = m;
                lmn(y, x, 2) = n;
            }
        }
    }
} // namespace gridding_benchmark
