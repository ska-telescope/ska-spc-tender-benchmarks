#include <iostream>
#include <vector>

#ifdef FFT_BENCHMARK_ENABLE_CPU
#include "benchmark_cpu.h"
#endif
#ifdef FFT_BENCHMARK_ENABLE_NVIDIA
#include "benchmark_nvidia.h"
#endif
#include "fft_configuration.h"

int main(int argc, char **argv)
{
    std::vector<fft_benchmark::configuration> configurations;
    try
    {
        configurations = fft_benchmark::read_configurations(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error while reading configurations file:\n";
        std::cerr << e.what() << std::endl;
        exit(-1);
    }

    double time;
    for (const auto &configuration : configurations)
    {
        switch (configuration.htype)
        {
#ifdef FFT_BENCHMARK_ENABLE_CPU
        case fft_benchmark::hardware_type::cpu:
            time = fft_benchmark::cpu::run(configuration);
            break;
#endif
#ifdef FFT_BENCHMARK_ENABLE_NVIDIA
        case fft_benchmark::hardware_type::nvidia:
            time = fft_benchmark::nvidia::run(configuration);
            break;
#endif
        default:
            std::cerr << "Invalid floating point type." << std::endl;
            return -1;
        }
        std::cout << configuration.nx << ", " << configuration.ny << " => " << time << " µs" << std::endl;
    }
}