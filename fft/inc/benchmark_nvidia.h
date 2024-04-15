#pragma once
#include <cuda_runtime.h>
#include <cufftXt.h>

#include "cufft_utils.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    namespace nvidia
    {
        struct cuda_event
        {
            cudaEvent_t event;

            cuda_event()
            {
                CUDA_RT_CALL(cudaEventCreate(&event));
            }

            ~cuda_event()
            {
                CUDA_RT_CALL(cudaEventDestroy(event));
            }

            cuda_event(const cuda_event&) = delete;
            cuda_event& operator=(const cuda_event&) = delete;
            cuda_event(cuda_event&&) = default;
            cuda_event& operator=(cuda_event&&) = default;

            void record(cudaStream_t stream = 0)
            {
                CUDA_RT_CALL(cudaEventRecord(event, stream));
            }
        };

        struct cufft_plan
        {
            cufftHandle handle = 0;

            cufft_plan()
            {
                CUFFT_CALL(cufftCreate(&handle));
            }

            cufft_plan(const cufft_plan&) = delete;
            cufft_plan& operator=(const cufft_plan&) = delete;
            cufft_plan(cufft_plan&&) = default;
            cufft_plan& operator=(cufft_plan&&) = default;

            ~cufft_plan()
            {
                CUFFT_CALL(cufftDestroy(handle));
            }
        };

        struct cuda_stream
        {
            cudaStream_t stream;

            cuda_stream()
            {
                CUDA_RT_CALL(cudaStreamCreate(&stream));
            }

            ~cuda_stream()
            {
                CUDA_RT_CALL(cudaStreamDestroy(stream));
            }

            void synchronize()
            {
                CUDA_RT_CALL(cudaStreamSynchronize(stream));
            }

            void query()
            {
                CUDA_RT_CALL(cudaStreamQuery(stream));
            }

            cuda_stream(const cuda_stream&) = delete;
            cuda_stream& operator=(const cuda_stream&) = delete;
            cuda_stream(cuda_stream&&) = default;
            cuda_stream& operator=(cuda_stream&&) = default;
        };

        double run(const configuration &configuration);
    } // namespace nvidia

    template <>
    struct fftw_type_helper<hardware_type::nvidia, float_type::single_precision>
    {
        using complex = cufftComplex;
        using real = float;
        using plan = nvidia::cufft_plan;
        static plan create_plan(const configuration &configuration);
        static double run_fft(const plan& plan, transform_type ttype, complex *in, complex *out);
    };

    template <>
    struct fftw_type_helper<hardware_type::nvidia, float_type::double_precision>
    {
        using complex = cufftDoubleComplex;
        using real = double;
        using plan = nvidia::cufft_plan;
        static plan create_plan(const configuration &configuration);
        static double run_fft(const plan& plan, transform_type ttype, complex *in, complex *out);
    };
} // namespace fft_benchmark