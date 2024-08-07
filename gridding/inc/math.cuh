#pragma once
#include <cuComplex.h>

inline __device__ float2 conj(float2 a)
{
    return cuConjf(a);
}

inline __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator*(float a, float2 b)
{
    return make_float2(a * b.x, a * b.y);
}

inline __device__ float2 operator*(const float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float a, float4 b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
