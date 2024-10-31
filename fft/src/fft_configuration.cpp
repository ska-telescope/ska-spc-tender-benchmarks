#include <iostream>
#include <yaml-cpp/yaml.h>

#include "fft_configuration.h"

namespace fft_benchmark
{
    std::vector<fft_benchmark::configuration> read_configurations(std::filesystem::path path)
    {
        if (!std::filesystem::exists(path))
        {
            throw std::runtime_error{"File " + path.string() + " not found."};
        }

        YAML::Node yaml_config = YAML::LoadFile(path);

        const auto &dimensions_config = yaml_config["dimensions"];
        if (dimensions_config.size() == 0)
        {
            std::cerr << "dimensions section empty." << std::endl;
            exit(-1);
        }

        std::vector<std::pair<size_t, size_t>> dimensions;
        dimensions.reserve(dimensions_config.size());
        for (const auto &dimension_config : dimensions_config)
        {
            if (dimension_config.size() != 2)
            {
                std::cerr << "Invalid dimension provided. Dimensions section should look like this:" << std::endl;
                std::cerr << "dimensions:" << std::endl;
                std::cerr << "- nx0, ny0" << std::endl;
                std::cerr << "- nx1, ny1" << std::endl;
                std::cerr << "- nx2, ny2" << std::endl;
                std::cerr << "..." << std::endl;
                exit(-1);
            }
            dimensions.emplace_back(dimension_config[0].as<size_t>(), dimension_config[1].as<size_t>());
        }

        const auto &float_types_config = yaml_config["float_types"];
        if (float_types_config.size() == 0)
        {
            std::cerr << "float_types section empty." << std::endl;
            exit(-1);
        }

        std::vector<fft_benchmark::float_type> float_types;
        float_types.reserve(float_types_config.size());
        for (const auto &float_type_config : float_types_config)
        {
            const auto float_type_string = float_type_config.as<std::string>();
            if (float_type_string == "float")
            {
                float_types.push_back(fft_benchmark::float_type::single_precision);
            }
            else if (float_type_string == "double")
            {
                float_types.push_back(fft_benchmark::float_type::double_precision);
            }
            else
            {
                std::cerr << "Invalid float type token " << float_type_string
                          << " was found. Must be either float or double." << std::endl;
                exit(-1);
            }
        }

        const auto &transform_types_config = yaml_config["transform_types"];
        if (transform_types_config.size() == 0)
        {
            std::cerr << "transform_types section empty." << std::endl;
            exit(-1);
        }

        std::vector<fft_benchmark::transform_type> transform_types;
        transform_types.reserve(transform_types_config.size());
        for (const auto &transform_type_config : transform_types_config)
        {
            const auto transform_type_string = transform_type_config.as<std::string>();
            if (transform_type_string == "forward")
            {
                transform_types.push_back(fft_benchmark::transform_type::forward);
            }
            else if (transform_type_string == "inverse")
            {
                transform_types.push_back(fft_benchmark::transform_type::inverse);
            }
            else
            {
                std::cerr << "Invalid transform_type token " << transform_type_string
                          << " was found. Must be either forward or inverse." << std::endl;
                exit(-1);
            }
        }

        const auto &backend_types_config = yaml_config["backend_types"];
        if (backend_types_config.size() == 0)
        {
            std::cerr << "backend_types section empty." << std::endl;
            exit(-1);
        }

        std::vector<benchmarks_common::backend_type> backend_types;
        backend_types.reserve(backend_types_config.size());
        for (const auto &backend_type_config : backend_types_config)
        {
            const auto backend_type_string = backend_type_config.as<std::string>();
#ifdef ENABLE_CPU
            if (backend_type_string == "cpu")
            {
                backend_types.push_back(benchmarks_common::backend_type::cpu);
                continue;
            }
#endif
#ifdef ENABLE_GPU
            if (backend_type_string == "gpu")
            {
                backend_types.push_back(benchmarks_common::backend_type::gpu);
                continue;
            }
#endif
#ifdef ENABLE_MKL
            if (backend_type_string == "mkl")
            {
                backend_types.push_back(benchmarks_common::backend_type::mkl);
                continue;
            }
#endif
#ifdef ENABLE_FFTW
            if (backend_type_string == "fftw")
            {
                backend_types.push_back(benchmarks_common::backend_type::fftw);
                continue;
            }
#endif
            std::cerr << "Invalid backend_type token " << backend_type_string
                      << " was found. Must be either cpu, gpu, fftw or mkl." << std::endl;
            exit(-1);
        }

        if (!yaml_config["niterations"])
        {
            std::cerr << "Invalid niterations section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <number of batch iterations to run>" << std::endl;
            exit(-1);
        }
        const size_t niterations = yaml_config["niterations"].as<size_t>();

        if (!yaml_config["memorysize"])
        {
            std::cerr << "Invalid memorysize section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <memory size to use in bytes>" << std::endl;
            exit(-1);
        }
        const size_t memorysize = yaml_config["memorysize"].as<size_t>();

        std::vector<fft_benchmark::configuration> configurations;
        configurations.reserve(dimensions.size() * float_types.size());
        for (const auto backend_type : backend_types)
        {
            for (const auto float_type : float_types)
            {
                for (const auto transform_type : transform_types)
                {
                    for (const auto dimension : dimensions)
                    {
                        fft_benchmark::configuration configuration;
                        configuration.ftype = float_type;
                        configuration.ttype = transform_type;
                        configuration.nx = dimension.first;
                        configuration.ny = dimension.second;
                        configuration.niterations = niterations;
                        configuration.memorysize = memorysize;
                        configuration.htype = backend_type;
                        configurations.emplace_back(configuration);
                    }
                }
            }
        }

        return configurations;
    }
} // namespace fft_benchmark
