#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <sys/ioctl.h>
#include <vector>

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

#include "benchmarks_common.h"
#include "fft_benchmark.h"
#include "fft_configuration.h"

void run(const std::vector<fft_benchmark::configuration> &configurations, const std::filesystem::path &output)
{
#ifdef VTUNE_PROFILE
    __itt_pause();
#endif

    struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);
    const auto columns = w.ws_col;

    std::cout << "##################################\n";
    std::cout << "#### FFT benchmarking utility ####\n";
    std::cout << "##################################\n\n";
    std::cout << "##############################\n";
    std::cout << "## Benchmarking configurations\n\n";

    std::vector<std::array<std::string, 6>> config_table;
    config_table.emplace_back(std::array<std::string, 6>{"Configuration ID", "  Iterations", "  Dimensions",
                                                         "  Max memory size", "  Precision", "  Backend"});
    int i_configuration = 0;
    for (const auto configuration : configurations)
    {
        std::array<std::string, 6> line;
        line[0] = std::to_string(i_configuration++);
        line[1] = std::to_string(configuration.niterations);
        line[2] = std::to_string(configuration.nx) + " * " + std::to_string(configuration.ny);
        line[3] = benchmarks_common::bytes_to_memory_size(configuration.memorysize);
        line[4] = (configuration.ftype == fft_benchmark::float_type::single_precision ? "single" : "double");
        line[5] = backend_type_string(configuration.htype);
        config_table.emplace_back(line);
    }
    benchmarks_common::print_columns(config_table);
    std::cout << std::endl;

    std::vector<std::string> titles;
    std::vector<fft_benchmark::benchmark_result> results;

    std::cout << "###############\n";
    std::cout << "## Benchmarking\n\n";
    int i = 0;
    const auto begin_bench = std::chrono::high_resolution_clock::now();
    for (const auto &configuration : configurations)
    {
        std::cout << "Running configuration " << i << "… ";
        std::cout.flush();

        const auto begin = std::chrono::high_resolution_clock::now();
        results.emplace_back(fft_benchmark::launch_benchmark(configuration));
        const auto end = std::chrono::high_resolution_clock::now();

        const auto time_s = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
        std::cout << "Done in " << time_s << " seconds." << std::endl;

        std::string title = std::to_string(i++);
        titles.emplace_back(title);
    }
    const auto end_bench = std::chrono::high_resolution_clock::now();
    const auto bench_time_s = std::chrono::duration_cast<std::chrono::seconds>(end_bench - begin_bench).count();
    std::cout << "==> Total benchmarking time: " << bench_time_s << " seconds." << std::endl << std::endl;

    std::vector<std::array<std::string, 12>> table;

    std::array<std::string, 12> labels{
        "  Configuration ID",
        "  Dimensions",
        "  Status",
        "  Max error",
        "  Number of iterations",
        "  Batch size",
        "  Init time (us)",
        "  Mean input transfer time (us)",
        "  Achieved input bandwidth (MiB/s)",
        "  Mean compute time (us)",
        "  Mean output transfer time (us)",
        "  Achieved output bandwidth (MiB/s)",
    };

    const auto to_result_string = [](const double x) { return x < 0. ? "N/A" : std::to_string(x); };

    const auto to_correctness_string = [](const fft_benchmark::benchmark_result::status_t status) {
        switch (status)
        {
        case fft_benchmark::benchmark_result::status_t::sucess:
            return std::string{"OK"};
        case fft_benchmark::benchmark_result::status_t::failure:
            return std::string{"ERROR"};
        }
        return std::string{};
    };

    table.emplace_back(labels);
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &result = results[i];
        const auto &title = titles[i];
        std::array<std::string, 12> row{title,
                                        std::to_string(configurations[i].nx) + " * " +
                                            std::to_string(configurations[i].ny),
                                        to_correctness_string(result.status),
                                        to_result_string(result.max_error),
                                        std::to_string(result.niterations),
                                        std::to_string(result.batch_size),
                                        to_result_string(result.init_time),
                                        to_result_string(result.in_transfer_time),
                                        to_result_string(result.in_bandwidth),
                                        to_result_string(result.compute_time),
                                        to_result_string(result.out_transfer_time),
                                        to_result_string(result.out_bandwidth)};
        table.emplace_back(row);
    }

    const auto title_max_length =
        std::max_element(titles.cbegin(), titles.cend(), [](const std::string &a, const std::string &b) {
            return a.size() < b.size();
        })->size();

    benchmarks_common::print_columns(table);
    benchmarks_common::write_csv(output, table);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "fft-benchmarks <configuration file> <output file>" << std::endl;
        exit(-1);
    }

    const std::filesystem::path configuration_path = argv[1];
    const std::filesystem::path output_path = argv[2];

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
    run(configurations, output_path);
    MPI_Finalize();
}
