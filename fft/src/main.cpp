#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "fft_benchmark.h"
#include "fft_configuration.h"

template <size_t N>
void print_columns(const std::vector<std::array<std::string, N>> &data)
{
    std::array<size_t, N> column_width;
    column_width.fill(0);
    for (const auto &row : data)
    {
        for (size_t i = 0; i < N; ++i)
        {
            column_width[i] = std::max(column_width[i], row[i].size());
        }
    }

    for (const auto &row : data)
    {
        for (size_t i = 0; i < N; ++i)
        {
            std::cout << std::right << std::setfill(' ') << std::setw(column_width[i]) << row[i];
            std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout.flush();
}

void run(const std::vector<fft_benchmark::configuration> &configurations)
{
    std::vector<std::string> titles;
    std::vector<fft_benchmark::time_result> results;
    for (const auto &configuration : configurations)
    {
        results.emplace_back(fft_benchmark::launch_benchmark(configuration));
        std::string title =
            std::to_string(configuration.nx) + " * " + std::to_string(configuration.ny) +
            (configuration.ftype == fft_benchmark::float_type::single_precision ? " / single" : " / double");
        titles.emplace_back(title);
    }

    std::vector<std::array<std::string, 7>> table;

    std::array<std::string, 7> labels{"configuration",
                                      "number of iterations",
                                      "batch size",
                                      "init time (us)",
                                      "mean input transfer time (us)",
                                      "mean compute time (us)",
                                      "mean output transfer time (us)"};

    const auto to_result_string = [](const double x) {
        return x < 0. ? "N/A" : std::to_string(static_cast<size_t>(x));
    };

    table.emplace_back(labels);
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &result = results[i];
        const auto &title = titles[i];
        std::array<std::string, 7> row{title,
                                       std::to_string(result.niterations),
                                       std::to_string(result.batch_size),
                                       to_result_string(result.init_time),
                                       to_result_string(result.in_transfer_time),
                                       to_result_string(result.compute_time),
                                       to_result_string(result.out_transfer_time)};
        table.emplace_back(row);
    }

    std::cout << "Benchmark results:\n\n\nb";
    print_columns(table);
}

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

    MPI_Init(&argc, &argv);
    run(configurations);
    MPI_Finalize();
}