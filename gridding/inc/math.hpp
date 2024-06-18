#pragma once
#include <complex>

#if defined(__CUDA_ARCH__)
#include "math.cuh"
#define DEVICE_FUNCTION_ANNOTATION __device__
#else
#define DEVICE_FUNCTION_ANNOTATION
#endif

namespace gridding_benchmark
{
    static constexpr float pi()
    {
        return 3.141592653589793;
    }

    static constexpr size_t n_correlations = 4;

    inline DEVICE_FUNCTION_ANNOTATION float compute_l(const int x, const int subgrid_size, const float image_size)
    {
        return (static_cast<float>(x) + 0.5F - (static_cast<float>(subgrid_size) / 2.F)) * image_size /
               static_cast<float>(subgrid_size);
    }

    inline DEVICE_FUNCTION_ANNOTATION float compute_m(const int y, const int subgrid_size, const float image_size)
    {
        return compute_l(y, subgrid_size, image_size);
    }

    inline DEVICE_FUNCTION_ANNOTATION float compute_n(const float l, const float m)
    {
        // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
        // accurately for small values of l and m
        const float tmp = (l * l) + (m * m);
        return tmp > 1.F ? 1.F : tmp / (1.f + sqrtf(1.F - tmp));
    }

    template <typename T>
    inline DEVICE_FUNCTION_ANNOTATION void matmul(const T *a, const T *b, T *c)
    {
        c[0] = a[0] * b[0];
        c[1] = a[0] * b[1];
        c[2] = a[2] * b[0];
        c[3] = a[2] * b[1];
        c[0] += a[1] * b[2];
        c[1] += a[1] * b[3];
        c[2] += a[3] * b[2];
        c[3] += a[3] * b[3];
    }

#ifdef __CUDA_ARCH__
    inline __device__ void conjugate(const float2 *a, float2 *b)
    {
        for (unsigned i = 0; i < 4; ++i)
        {
            b[i] = conj(a[i]);
        }
    }
#else
    inline void conjugate(const std::complex<float> *a, std::complex<float> *b)
    {
        for (unsigned i = 0; i < 4; ++i)
        {
            b[i] = std::conj(a[i]);
        }
    }
#endif

    template <typename T>
    inline DEVICE_FUNCTION_ANNOTATION void transpose(const T *a, T *b)
    {
        b[0] = a[0];
        b[1] = a[2];
        b[2] = a[1];
        b[3] = a[3];
    }

    template <typename T>
    inline DEVICE_FUNCTION_ANNOTATION void hermitian(const T *a, T *b)
    {
        T temp[4];
        conjugate(a, temp);
        transpose(temp, b);
    }

    template <typename T>
    inline void apply_aterm_gridder(T *pixels, const T *aterm1, const T *aterm2)
    {
        // Aterm 1 hermitian
        T aterm1_h[4];
        hermitian(aterm1, aterm1_h);

        // Apply aterm: P = A1^H * P
        T temp[4];
        matmul(aterm1_h, pixels, temp);

        // Apply aterm: P = P * A2
        matmul(temp, aterm2, pixels);
    }
} // namespace gridding_benchmark
