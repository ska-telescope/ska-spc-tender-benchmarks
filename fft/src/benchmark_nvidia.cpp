#include <array>
#include <complex>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>

#include "benchmark_nvidia.h"
#include "cufft_utils.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::plan fftw_type_helper<
        hardware_type::nvidia, float_type::single_precision>::create_plan(const configuration &configuration)
    {
        fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::plan plan;
        std::array<int, 2> dimensions{static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        CUFFT_CALL(cufftPlanMany(&plan.handle, 2, dimensions.data(), nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C,
                                 configuration.nbatches));
        return plan;
    }

    double fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::run_fft(const plan &plan,
                                                                                          transform_type ttype,
                                                                                          complex *in, complex *out)
    {
        CUFFT_CALL(
            cufftExecC2C(plan.handle, in, out, ttype == transform_type::forward ? CUFFT_FORWARD : CUFFT_INVERSE));
    }

    fftw_type_helper<hardware_type::nvidia, float_type::double_precision>::plan fftw_type_helper<
        hardware_type::nvidia, float_type::double_precision>::create_plan(const configuration &configuration)
    {
        plan plan;

        int n_gpus = 0;
        CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
        std::vector<int> gpus(n_gpus);
        std::iota(gpus.begin(), gpus.end(), 0);
        CUFFT_CALL(cufftXtSetGPUs(plan.handle, n_gpus, gpus.data()));

        std::vector<size_t> work_sizes(n_gpus);
        std::array<long long int, 2> dimensions{static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        CUFFT_CALL(cufftXtMakePlanMany(plan.handle, 2, dimensions.data(), nullptr, 0, 0, CUDA_C_64F, nullptr, 0, 0,
                                       CUDA_C_64F, configuration.nbatches, work_sizes.data(), CUDA_C_64F));
        return plan;
    }

    double fftw_type_helper<hardware_type::nvidia, float_type::double_precision>::run_fft(const plan &plan,
                                                                                          transform_type ttype,
                                                                                          complex *in, complex *out)
    {
        cudaLibXtDesc *input_desc;
        CUFFT_CALL(cufftXtMalloc(plan.handle, &input_desc, CUFFT_XT_FORMAT_INPLACE));
        cudaLibXtDesc *output_desc;
        CUFFT_CALL(cufftXtMalloc(plan.handle, &output_desc, CUFFT_XT_FORMAT_INPLACE));

        CUFFT_CALL(cufftXtMemcpy(plan.handle, input_desc, in, CUFFT_COPY_HOST_TO_DEVICE));

        nvidia::cuda_event start;
        nvidia::cuda_event stop;

        start.record();
        CUFFT_CALL(cufftXtExec(plan.handle, in, out, ttype == transform_type::forward ? CUFFT_FORWARD : CUFFT_INVERSE));
        stop.record();

        CUFFT_CALL(cufftXtMemcpy(plan.handle, out, output_desc, CUFFT_COPY_DEVICE_TO_HOST));

        CUDA_RT_CALL(cudaEventSynchronize(stop.event));

        CUFFT_CALL(cufftXtFree(input_desc));
        CUFFT_CALL(cufftXtFree(output_desc));

        float time = 0.F;
        cudaEventElapsedTime(&time, start.event, stop.event);

        return static_cast<double>(time);
    }

    namespace nvidia
    {
        template <fft_benchmark::float_type ftype>
        double run_per_float(const configuration &configuration)
        {
            using complex = typename fftw_type_helper<hardware_type::nvidia, ftype>::complex;

            const auto n_samples = configuration.nx * configuration.ny * configuration.nbatches;
            std::cout << sizeof(complex) * n_samples << std::endl;

            std::vector<complex> input_host(configuration.niterations * n_samples);
            std::vector<complex> output_host(configuration.niterations * n_samples);

            for(size_t ix = 0; ix < configuration.nx; ++ix)
            {
                for(size_t iy = 0; iy < configuration.ny; ++iy)
                {
                    input_host[ix * configuration.ny + iy].x = std::cos(static_cast<float>(10.f * ix)) * std::sin(static_cast<float>(0.1f * iy));
                    input_host[ix * configuration.ny + iy].x = std::sin(static_cast<float>(500.f * ix)) * std::cos(static_cast<float>(10.f * iy));
                }
            }

            CUDA_RT_CALL(cudaHostRegister(input_host.data(), input_host.size() * sizeof(complex), cudaHostRegisterPortable));
            CUDA_RT_CALL(cudaHostRegister(output_host.data(), input_host.size() * sizeof(complex), cudaHostRegisterPortable));
            
            static constexpr size_t n_streams = 4;

            complex *input;
            complex *output;
            CUDA_RT_CALL(cudaMalloc(&input, n_streams * sizeof(complex) * n_samples));
            CUDA_RT_CALL(cudaMalloc(&output, n_streams * sizeof(complex) * n_samples));

            using plan_t = typename fftw_type_helper<hardware_type::nvidia, ftype>::plan;
            std::array<plan_t, n_streams> plans;
            std::array<cuda_stream, n_streams> streams;
            std::array<void *, n_streams> wk_areas;

            for (size_t i_plan = 0; i_plan < n_streams; ++i_plan)
            {
                plans[i_plan] = fftw_type_helper<hardware_type::nvidia, ftype>::create_plan(configuration);

                size_t ws;
                CUFFT_CALL(cufftGetSize(plans[i_plan].handle, &ws));
                CUDA_RT_CALL(cudaMalloc(&(wk_areas[i_plan]), ws));
                CUFFT_CALL(cufftSetAutoAllocation(plans[i_plan].handle, 0));
                CUFFT_CALL(cufftSetWorkArea(plans[i_plan].handle, wk_areas[i_plan]));

                fftw_type_helper<hardware_type::nvidia, ftype>::run_fft(plans[i_plan], configuration.ttype, input,
                                                                        output);

                CUDA_RT_CALL(cufftSetStream(plans[i_plan].handle, streams[i_plan].stream));
            }

            double time = 0.;

            cuda_event start;
            cuda_event stop;

            const auto begin_parallel = std::chrono::high_resolution_clock::now();
            size_t i_stream = 0;
            for (size_t i_iteration = 0; i_iteration < configuration.niterations; ++i_iteration)
            {
                CUDA_RT_CALL(cudaMemcpyAsync(input + i_stream * n_samples, &input_host[i_iteration * n_samples],
                                             sizeof(complex) * n_samples, cudaMemcpyHostToDevice,
                                             streams[i_stream].stream));
                fftw_type_helper<hardware_type::nvidia, ftype>::run_fft(plans[i_stream], configuration.ttype, input, output);
                CUDA_RT_CALL(cudaMemcpyAsync(&output_host[i_iteration * n_samples], output + i_stream * n_samples,
                                             sizeof(complex) * n_samples, cudaMemcpyDeviceToHost,
                                             streams[i_stream].stream));
                i_stream = (i_stream + 1) % n_streams;
            }
            cudaDeviceSynchronize();
            const auto end_parallel = std::chrono::high_resolution_clock::now();

            return std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - begin_parallel).count() / static_cast<double>(configuration.niterations);
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
