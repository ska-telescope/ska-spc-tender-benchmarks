#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "gridding_configuration.h"

namespace gridding_benchmark
{
    std::vector<configuration> read_configuration(std::filesystem::path path)
    {
        if (!std::filesystem::exists(path))
        {
            throw std::runtime_error{"File " + path.string() + " not found."};
        }

        YAML::Node yaml_config = YAML::LoadFile(path);

        const auto &hardware_types_config = yaml_config["hardware_types"];
        if (hardware_types_config.size() == 0)
        {
            std::cerr << "hardware_types section empty." << std::endl;
            exit(-1);
        }

        std::vector<benchmarks_common::hardware_type> hardware_types;
        hardware_types.reserve(hardware_types_config.size());
        for (const auto &hardware_type_config : hardware_types_config)
        {
            const auto hardware_type_string = hardware_type_config.as<std::string>();
#ifdef ENABLE_CPU
            if (hardware_type_string == "cpu")
            {
                hardware_types.push_back(benchmarks_common::hardware_type::cpu);
                continue;
            }
#endif
#ifdef ENABLE_GPU
            if (hardware_type_string == "gpu")
            {
                hardware_types.push_back(benchmarks_common::hardware_type::gpu);
                continue;
            }
#endif
            std::cerr << "Invalid hardware_type token " << hardware_type_string
                      << " was found. Must be either cpu or gpu." << std::endl;
            exit(-1);
        }

        if (!yaml_config["grid_size"])
        {
            std::cerr << "Invalid nitgrid_sizerations section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <grid size>" << std::endl;
            exit(-1);
        }
        const size_t grid_size = yaml_config["grid_size"].as<size_t>();

        if (!yaml_config["subgrid_size"])
        {
            std::cerr << "Invalid subgrid_size section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <subgrid size>" << std::endl;
            exit(-1);
        }
        const size_t subgrid_size = yaml_config["subgrid_size"].as<size_t>();

        if (!yaml_config["nstations"])
        {
            std::cerr << "Invalid nstations section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <number of stations>" << std::endl;
            exit(-1);
        }
        const size_t nstations = yaml_config["nstations"].as<size_t>();

        if (!yaml_config["nchannels"])
        {
            std::cerr << "Invalid nchannels section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <number of channels>" << std::endl;
            exit(-1);
        }
        const size_t nchannels = yaml_config["nchannels"].as<size_t>();

        if (!yaml_config["ntimeslots"])
        {
            std::cerr << "Invalid ntimeslots section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <number of time slots>" << std::endl;
            exit(-1);
        }
        const size_t ntimeslots = yaml_config["ntimeslots"].as<size_t>();

        if (!yaml_config["ntimesteps_per_subgrid"])
        {
            std::cerr << "Invalid ntimesteps_per_subgrid section. Should look like this:" << std::endl;
            std::cerr << "nbatches: <number of time steps per subgrid>" << std::endl;
            exit(-1);
        }
        const size_t ntimesteps_per_subgrid = yaml_config["ntimesteps_per_subgrid"].as<size_t>();

        std::vector<gridding_benchmark::configuration> configurations;
        configurations.reserve(hardware_types.size());
        for (const auto hardware_type : hardware_types)
        {
            gridding_benchmark::configuration configuration;
            configuration.htype = hardware_type;
            configuration.grid_size = grid_size;
            configuration.subgrid_size = subgrid_size;
            configuration.nstations = nstations;
            configuration.nchannels = nchannels;
            configuration.ntimeslots = ntimeslots;
            configuration.ntimesteps_per_subgrid = ntimesteps_per_subgrid;
            configurations.emplace_back(configuration);
        }

        return configurations;
    }
} // namespace gridding_benchmark
