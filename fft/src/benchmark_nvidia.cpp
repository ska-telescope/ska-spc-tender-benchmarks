#include <array>
#include <complex>
#include <cufftXt.h>
#include <numeric>
#include <vector>

#include "benchmark_nvidia.h"
#include "cufft_utils.h"
#include "fft_configuration.h"

namespace fft_benchmark
{
    fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::plan fftw_type_helper<
        hardware_type::nvidia, float_type::single_precision>::create_plan(const configuration &configuration)
    {
        plan plan;

        int n_gpus = 0;
        CUDA_RT_CALL(cudaGetDeviceCount(&n_gpus));
        std::vector<int> gpus(n_gpus);
        std::iota(gpus.begin(), gpus.end(), 0);
        CUFFT_CALL(cufftXtSetGPUs(plan.handle, n_gpus, gpus.data()));

        std::vector<size_t> work_sizes(n_gpus);
        std::array<long long int, 2> dimensions{static_cast<int>(configuration.nx), static_cast<int>(configuration.ny)};
        CUFFT_CALL(cufftXtMakePlanMany(plan.handle, 2, dimensions.data(), nullptr, 0, 0, CUDA_C_32F, nullptr, 0, 0,
                                       CUDA_C_32F, configuration.nbatches, work_sizes.data(), CUDA_C_32F));
        return plan;
    }

    double fftw_type_helper<hardware_type::nvidia, float_type::single_precision>::run_fft(plan plan,
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
printf("yooooo\n");
        return static_cast<double>(time);
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

    double fftw_type_helper<hardware_type::nvidia, float_type::double_precision>::run_fft(plan plan,
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
            const auto plan = fftw_type_helper<hardware_type::nvidia, ftype>::create_plan(configuration);

            std::vector<typename fftw_type_helper<hardware_type::nvidia, ftype>::complex> input(
                configuration.nx * configuration.ny * configuration.nbatches);
            std::vector<typename fftw_type_helper<hardware_type::nvidia, ftype>::complex> output(
                configuration.nx * configuration.ny * configuration.nbatches);

            const auto time = fftw_type_helper<hardware_type::nvidia, ftype>::run_fft(plan, configuration.ttype,
                                                                                      input.data(), output.data());
            return static_cast<double>(time);
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
