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

        const auto &hardware_types_config = yaml_config["hardware_types"];
        if (hardware_types_config.size() == 0)
        {
            std::cerr << "hardware_types section empty." << std::endl;
            exit(-1);
        }

        std::vector<fft_benchmark::hardware_type> hardware_types;
        hardware_types.reserve(hardware_types_config.size());
        for (const auto &hardware_type_config : hardware_types_config)
        {
            const auto hardware_type_string = hardware_type_config.as<std::string>();
            if (hardware_type_string == "cpu")
            {
                hardware_types.push_back(fft_benchmark::hardware_type::cpu);
            }
            else if (hardware_type_string == "nvidia")
            {
                hardware_types.push_back(fft_benchmark::hardware_type::nvidia);
            }
            else if (hardware_type_string == "amd")
            {
                hardware_types.push_back(fft_benchmark::hardware_type::amd);
            }
            else if (hardware_type_string == "heffte")
            {
                hardware_types.push_back(fft_benchmark::hardware_type::heffte);
            }
            else
            {
                std::cerr << "Invalid hardware_type token " << hardware_type_string
                          << " was found. Must be either cpu, nvidia, amd or heffte." << std::endl;
                exit(-1);
            }
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

        const auto inplace_err_func = []() {
            std::cerr << "Invalid inplace section. Should look like this:" << std::endl;
            std::cerr << "inplace: <true if the transforms should be run in place, false otherwise>" << std::endl;
            exit(-1);
        };

        if (!yaml_config["in_place"])
        {
            inplace_err_func();
        }
        const auto in_place_string = yaml_config["in_place"].as<std::string>();
        if (in_place_string != "true" && in_place_string != "false")
        {
            inplace_err_func();
        }
        const bool in_place = in_place_string == "true";

        std::vector<fft_benchmark::configuration> configurations;
        configurations.reserve(dimensions.size() * float_types.size());
        for (const auto hardware_type : hardware_types)
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
                        configuration.in_place = in_place;
                        configuration.htype = hardware_type;
                        configurations.emplace_back(configuration);
                    }
                }
            }
        }

        return configurations;
    }
} // namespace fft_benchmark