#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <sys/ioctl.h>
#include <vector>

#ifdef VTUNE_PROFILE
#include <ittnotify.h>
#endif

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

size_t log2_size_t(size_t index)
{
    size_t targetlevel;
    while (index >>= 1)
        ++targetlevel;
    return targetlevel;
}

std::string bytes_to_memory_size(const size_t n)
{
    const std::array<std::string, 9> strs = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

    const auto log2n = (log2_size_t(n) - 1);
    const auto logrank = log2n / 10;
    const auto base = 1 << (logrank * 10);
    const auto rounded = std::round(static_cast<double>(10. * n) / static_cast<double>(base)) / 10.;
    std::stringstream sstr;
    sstr << rounded << " " << strs[logrank];
    return sstr.str();
}

std::string hardware_type_string(const fft_benchmark::hardware_type htype)
{
    switch (htype)
    {
    case fft_benchmark::hardware_type::amd:
        return "amd";
    case fft_benchmark::hardware_type::cpu:
        return "cpu";
    case fft_benchmark::hardware_type::nvidia:
        return "nvidia";
    default:
        return "";
    }
}

std::string float_type_string(const fft_benchmark::float_type ftype)
{
    switch (ftype)
    {
    case fft_benchmark::float_type::single_precision:
        return "single";
    case fft_benchmark::float_type::double_precision:
        return "double";
    default:
        return "";
    }
}

void run(const std::vector<fft_benchmark::configuration> &configurations)
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
                                                         "  Max memory size", "  Precision", "  Hardware"});
    int i_configuration = 0;
    for (const auto configuration : configurations)
    {
        std::array<std::string, 6> line;
        line[0] = std::to_string(i_configuration++);
        line[1] = std::to_string(configuration.niterations);
        line[2] = std::to_string(configuration.nx) + " * " + std::to_string(configuration.ny);
        line[3] = bytes_to_memory_size(configuration.memorysize);
        line[4] = (configuration.ftype == fft_benchmark::float_type::single_precision ? "single" : "double");
        line[5] = hardware_type_string(configuration.htype);
        config_table.emplace_back(line);
    }
    print_columns(config_table);
    std::cout << std::endl;

    std::cout << "##############\n";
    std::cout << "## Warmup step\n\n";
    std::cout << "Warming up…";

    const auto begin_warmup = std::chrono::high_resolution_clock::now();
    fft_benchmark::launch_benchmark(configurations.front());
    const auto end_warmup = std::chrono::high_resolution_clock::now();
    const auto warmup_s = std::chrono::duration_cast<std::chrono::seconds>(end_warmup - begin_warmup).count();

    std::cout << " Done in " << warmup_s << " seconds.\n";
    std::cout << "==> Total benchmark should take a bit more than "
              << std::round(static_cast<double>(configurations.size()) * warmup_s) << " seconds." << std::endl;
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

    std::vector<std::array<std::string, 9>> table;

    std::array<std::string, 9> labels{"  Configuration ID",
                                      "  Status",
                                      "  Max error",
                                      "  Number of iterations",
                                      "  Batch size",
                                      "  Init time (us)",
                                      "  Mean input transfer time (us)",
                                      "  Mean compute time (us)",
                                      "  Mean output transfer time (us)"};

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
        std::array<std::string, 9> row{title,
                                       to_correctness_string(result.status),
                                       std::to_string(result.max_error),
                                       std::to_string(result.niterations),
                                       std::to_string(result.batch_size),
                                       to_result_string(result.init_time),
                                       to_result_string(result.in_transfer_time),
                                       to_result_string(result.compute_time),
                                       to_result_string(result.out_transfer_time)};
        table.emplace_back(row);
    }

    const auto title_max_length =
        std::max_element(titles.cbegin(), titles.cend(), [](const std::string &a, const std::string &b) {
            return a.size() < b.size();
        })->size();

    print_columns(table);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "No configuration file provided." << std::endl;
        exit(-1);
    }

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