#include "benchmark_cpu.h"
#include "benchmark_nvidia.h"

#include <iostream>

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
        case fft_benchmark::hardware_type::cpu:
            time = fft_benchmark::cpu::run(configuration);
            break;
        case fft_benchmark::hardware_type::nvidia:
            time = fft_benchmark::nvidia::run(configuration);
            break;
        default:
            std::cerr << "Invalid floating point type." << std::endl;
            return -1;
        }
        std::cout << configuration.nx << ", " << configuration.ny << " => " << time << " µs" << std::endl;
    }
}