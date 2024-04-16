#include <array>
#include <chrono>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <vector>

#include "benchmark_nvidia.h"
#include "cufft_utils.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    void fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::create_plan(
        const configuration &configuration, plan &plan)
    {
        std::array<int, 2> dimensions{static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        CUFFT_CALL(cufftPlanMany(&plan.handle, 2, dimensions.data(), nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C,
                                 configuration.nbatches));
    }

    void fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::run_fft(const plan &plan,
                                                                                        transform_type ttype,
                                                                                        complex *in, complex *out)
    {
        CUFFT_CALL(
            cufftExecC2C(plan.handle, in, out, ttype == transform_type::forward ? CUFFT_FORWARD : CUFFT_INVERSE));
    }

    void fftw_type_helper<hardware_type::nvidia, float_type::double_precision>::create_plan(
        const configuration &configuration, plan &plan)
    {
        std::array<int, 2> dimensions{static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        CUFFT_CALL(cufftPlanMany(&plan.handle, 2, dimensions.data(), nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2Z,
                                 configuration.nbatches));
    }

    void fftw_type_helper<hardware_type::nvidia, float_type::double_precision>::run_fft(const plan &plan,
                                                                                        transform_type ttype,
                                                                                        complex *in, complex *out)
    {
        CUFFT_CALL(
            cufftExecZ2Z(plan.handle, in, out, ttype == transform_type::forward ? CUFFT_FORWARD : CUFFT_INVERSE));
    }

    namespace nvidia
    {
        template <fft_benchmark::float_type ftype>
        double run_per_float(const configuration &configuration)
        {
#ifdef ENABLE_PROFILING
            cudaProfilerStop();
#endif
            using complex = typename fftw_type_helper<hardware_type::nvidia, ftype>::complex;

            const auto n_samples = configuration.nx * configuration.ny * configuration.nbatches;

            std::vector<complex> input_host(n_samples);
            std::vector<complex> output_host;
            if (!configuration.in_place)
            {
                output_host.resize(n_samples);
            }

            for (auto &x : input_host)
            {
                x.x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                x.y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            CUDA_RT_CALL(
                cudaHostRegister(input_host.data(), input_host.size() * sizeof(complex), cudaHostRegisterPortable));
            if (!configuration.in_place)
            {
                CUDA_RT_CALL(cudaHostRegister(output_host.data(), input_host.size() * sizeof(complex),
                                              cudaHostRegisterPortable));
            }

            static constexpr size_t n_streams = 1;

            complex *input;
            complex *output;
            CUDA_RT_CALL(cudaMalloc(&input, n_streams * sizeof(complex) * n_samples));
            if (!configuration.in_place)
            {
                CUDA_RT_CALL(cudaMalloc(&output, n_streams * sizeof(complex) * n_samples));
            }
            else
            {
                output = input;
            }

            using plan_t = typename fftw_type_helper<hardware_type::nvidia, ftype>::plan;
            std::array<plan_t, n_streams> plans;
            std::array<cuda_stream, n_streams> streams;
            std::array<void *, n_streams> wk_areas;

            for (size_t i_plan = 0; i_plan < n_streams; ++i_plan)
            {
                fftw_type_helper<hardware_type::nvidia, ftype>::create_plan(configuration, plans[i_plan]);

                size_t ws;
                CUFFT_CALL(cufftGetSize(plans[i_plan].handle, &ws));
                CUDA_RT_CALL(cudaMalloc(&(wk_areas[i_plan]), ws));
                CUFFT_CALL(cufftSetAutoAllocation(plans[i_plan].handle, 0));
                CUFFT_CALL(cufftSetWorkArea(plans[i_plan].handle, wk_areas[i_plan]));
                fftw_type_helper<hardware_type::nvidia, ftype>::run_fft(plans[i_plan], configuration.ttype, input,
                                                                        input);

                CUDA_RT_CALL(cufftSetStream(plans[i_plan].handle, streams[i_plan].stream));
            }

            double time = 0.;

            cuda_event start;
            cuda_event stop;

#ifdef ENABLE_PROFILING
            cudaProfilerStart();
#endif
            const auto begin_parallel = std::chrono::high_resolution_clock::now();
            size_t i_stream = 0;
            for (size_t i_iteration = 0; i_iteration < configuration.niterations; ++i_iteration)
            {
                CUDA_RT_CALL(cudaMemcpyAsync(input + i_stream * n_samples, input_host.data(),
                                             sizeof(complex) * n_samples, cudaMemcpyHostToDevice,
                                             streams[i_stream].stream));
                fftw_type_helper<hardware_type::nvidia, ftype>::run_fft(plans[i_stream], configuration.ttype, input,
                                                                        output);
                CUDA_RT_CALL(cudaMemcpyAsync(input_host.data(), output + i_stream * n_samples,
                                             sizeof(complex) * n_samples, cudaMemcpyDeviceToHost,
                                             streams[i_stream].stream));
                i_stream = (i_stream + 1) % n_streams;
            }
            cudaDeviceSynchronize();
            const auto end_parallel = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_PROFILING
            cudaProfilerStop();
#endif

            CUDA_RT_CALL(cudaHostUnregister(input_host.data()));
            if (!configuration.in_place)
            {
                CUDA_RT_CALL(cudaHostUnregister(output_host.data()));
            }

            for (size_t i_plan = 0; i_plan < n_streams; ++i_plan)
            {
                CUDA_RT_CALL(cudaFree((wk_areas[i_plan])));
            }
            CUDA_RT_CALL(cudaFree(input));
            if (!configuration.in_place)
            {
                CUDA_RT_CALL(cudaFree(output));
            }

            return std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - begin_parallel).count() /
                   static_cast<double>(configuration.niterations);
        }

        double run(const configuration &configuration)
        {
            switch (configuration.ftype)
            {
            case float_type::single_precision:
                return run_per_float<float_type::single_precision>(configuration);
            case float_type::double_precision:
                return run_per_float<float_type::double_precision>(configuration);
            default:
                throw(std::runtime_error{"Invalid floating point error."});
            }
        }
    } // namespace nvidia
} // namespace fft_benchmark
