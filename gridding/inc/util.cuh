#pragma once
#include <cuda.h>
#include <iostream>

#define cudaCheck(ans)                                                                                                 \
    {                                                                                                                  \
        cudaAssert((ans), __FILE__, __LINE__);                                                                         \
    }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
