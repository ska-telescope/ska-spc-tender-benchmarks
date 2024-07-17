#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

#include "benchmarks_common.h"
#include "degridding_benchmark.h"
#include "gridding_benchmark.h"
#include "gridding_configuration.h"

void run(const std::vector<gridding_benchmark::configuration> &configurations, const std::filesystem::path &output)
{
#ifdef VTUNE_PROFILE
    __itt_pause();
#endif

    std::cout << "##################################################\n";
    std::cout << "#### Gridding/degridding benchmarking utility ####\n";
    std::cout << "##################################################\n\n";
    std::cout << "##################################################\n";
    std::cout << "## Benchmarking configurations\n\n";

    std::vector<std::array<std::string, 7>> config_table;
    config_table.emplace_back(std::array<std::string, 7>{"Configuration ID", "  Operation", "  Grid size",
                                                         "  Subgrid size", "  Number of channels",
                                                         "  Number of stations", "  Hardware"});

    int i_configuration = 0;
    for (const auto configuration : configurations)
    {
        std::array<std::string, 7> line;
        line[0] = std::to_string(i_configuration++);
        line[1] = to_string(configuration.operation);
        line[2] = std::to_string(configuration.grid_size);
        line[3] = std::to_string(configuration.subgrid_size);
        line[4] = std::to_string(configuration.nchannels);
        line[5] = std::to_string(configuration.nstations);
        line[6] = benchmarks_common::hardware_type_string(configuration.htype);
        config_table.emplace_back(line);
    }
    benchmarks_common::print_columns(config_table);
    std::cout << std::endl;

    std::cout << "##############\n";
    std::cout << "## Warmup step\n\n";
    std::cout << "Warming up…";

    const auto begin_warmup = std::chrono::high_resolution_clock::now();
    gridding_benchmark::launch_gridding(configurations.front());
    const auto end_warmup = std::chrono::high_resolution_clock::now();
    const auto warmup_s = std::chrono::duration_cast<std::chrono::seconds>(end_warmup - begin_warmup).count();

    std::cout << " Done in " << warmup_s << " seconds.\n";
    std::cout << "==> Total benchmark should take a bit more than "
              << std::round(static_cast<double>(configurations.size()) * warmup_s) << " seconds." << std::endl;
    std::cout << std::endl;

    std::vector<std::string> titles;
    std::vector<gridding_benchmark::benchmark_result> results;

    std::cout << "###############\n";
    std::cout << "## Benchmarking\n\n";
    int i = 0;
    const auto begin_bench = std::chrono::high_resolution_clock::now();
    for (const auto &configuration : configurations)
    {
        std::cout << "Running configuration " << i << "… ";
        std::cout.flush();

        const auto begin = std::chrono::high_resolution_clock::now();
        const auto result = configuration.operation == gridding_benchmark::operation_type::gridding
                                ? gridding_benchmark::launch_gridding(configuration)
                                : gridding_benchmark::launch_degridding(configuration);
        results.emplace_back(result);
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
        "  Operation",
        "  Iterations",
        "  Grid size",
        "  Subgrid size",
        "  Number of channels",
        "  Number of stations",
        "  Mean input transfer time (us)",
        "  Achieved input bandwidth (MiB/s)",
        "  Mean compute time (us)",
        "  Mean output transfer time (us)",
        "  Achieved output bandwidth (MiB/s)",
    };

    const auto to_result_string = [](const double x) { return x < 0. ? "N/A" : std::to_string(x); };

    table.emplace_back(labels);
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto &result = results[i];
        const auto &title = titles[i];
        std::array<std::string, 12> row{title,
                                        to_string(configurations[i].operation),
                                        std::to_string(configurations[i].niterations),
                                        std::to_string(configurations[i].grid_size),
                                        std::to_string(configurations[i].subgrid_size),
                                        std::to_string(configurations[i].nchannels),
                                        std::to_string(configurations[i].nstations),
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
        std::cerr << "gridding-benchmarks <configuration file> <output file>" << std::endl;
        exit(-1);
    }

    const std::filesystem::path configuration_path = argv[1];
    const std::filesystem::path output_path = argv[2];

    std::vector<gridding_benchmark::configuration> configurations;
    try
    {
        configurations = gridding_benchmark::read_configuration(configuration_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error while reading configurations file:\n";
        std::cerr << e.what() << std::endl;
        exit(-1);
    }

    run(configurations, output_path);
}
