#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "benchmarks_common.h"
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
            benchmarks_common::log_and_abort("hardware_types section empty.");
        }

        std::vector<benchmarks_common::hardware_type> hardware_types;
        hardware_types.reserve(hardware_types_config.size());
        for (const auto &hardware_type_config : hardware_types_config)
        {
            const auto hardware_type_string = hardware_type_config.as<std::string>();
            if (hardware_type_string == "cpu")
            {
                if (is_hardware_type_enabled(benchmarks_common::hardware_type::cpu))
                {
                    hardware_types.push_back(benchmarks_common::hardware_type::cpu);
                    continue;
                }
                benchmarks_common::log_and_abort("CPU support is disabled");
            }
            if (hardware_type_string == "gpu")
            {
                if (is_hardware_type_enabled(benchmarks_common::hardware_type::gpu))
                {
                    hardware_types.push_back(benchmarks_common::hardware_type::gpu);
                    continue;
                }
                benchmarks_common::log_and_abort("GPU support is disabled");
            }
            benchmarks_common::log_and_abort("Invalid hardware_type token " + hardware_type_string +
                                             " was found. Must be either cpu or gpu.");
            exit(-1);
        }

        const auto &operation_types_config = yaml_config["operation_types"];
        if (operation_types_config.size() == 0)
        {
            benchmarks_common::log_and_abort("operation_types section empty.");
        }

        std::vector<operation_type> operation_types;
        operation_types.reserve(operation_types_config.size());
        for (const auto &operation_type_config : operation_types_config)
        {
            const auto operation_type_string = operation_type_config.as<std::string>();
            const auto operation_type = to_operation_type(operation_type_string);

            if (operation_type == operation_type::err)
            {
                benchmarks_common::log_and_abort("Invalid operation type token " + operation_type_string +
                                                 " was found. Must be either gridding or degridding.");
            }
            operation_types.push_back(operation_type);
        }

        if (!yaml_config["grid_size"])
        {
            benchmarks_common::log_and_abort(
                "Invalid grid_size section. Should look like this:\n  nbatches: <grid size>");
        }
        const size_t grid_size = yaml_config["grid_size"].as<size_t>();

        if (!yaml_config["subgrid_size"])
        {
            benchmarks_common::log_and_abort(
                "Invalid subgrid_size section. Should look like this:\n  nbatches: <subgrid size>");
        }
        const size_t subgrid_size = yaml_config["subgrid_size"].as<size_t>();

        if (!yaml_config["niterations"])
        {
            benchmarks_common::log_and_abort(
                "Invalid niterations section. Should look like this:\n  nbatches: <niterations>");
        }
        const size_t niterations = yaml_config["niterations"].as<size_t>();

        if (!yaml_config["nstations"])
        {
            benchmarks_common::log_and_abort(
                "Invalid nstations section. Should look like this:\n  nbatches: <number of stations>");
        }
        const size_t nstations = yaml_config["nstations"].as<size_t>();

        if (!yaml_config["nchannels"])
        {
            benchmarks_common::log_and_abort(
                "Invalid nchannels section. Should look like this:\n  nbatches: <number of channels>");
        }
        const size_t nchannels = yaml_config["nchannels"].as<size_t>();

        if (!yaml_config["ntimeslots"])
        {
            benchmarks_common::log_and_abort(
                "Invalid ntimeslots section. Should look like this:\n  nbatches: <number of time slots>");
        }
        const size_t ntimeslots = yaml_config["ntimeslots"].as<size_t>();

        if (!yaml_config["ntimesteps_per_subgrid"])
        {
            benchmarks_common::log_and_abort("Invalid ntimesteps_per_subgrid section. Should look like this:\n  "
                                             "nbatches: <number of time steps per subgrid>");
        }
        const size_t ntimesteps_per_subgrid = yaml_config["ntimesteps_per_subgrid"].as<size_t>();

        std::vector<gridding_benchmark::configuration> configurations;
        configurations.reserve(hardware_types.size());
        for (const auto hardware_type : hardware_types)
        {
            for (const auto operation_type : operation_types)
            {
                gridding_benchmark::configuration configuration;
                configuration.operation = operation_type;
                configuration.htype = hardware_type;
                configuration.grid_size = grid_size;
                configuration.subgrid_size = subgrid_size;
                configuration.niterations = niterations;
                configuration.nstations = nstations;
                configuration.nchannels = nchannels;
                configuration.ntimeslots = ntimeslots;
                configuration.ntimesteps_per_subgrid = ntimesteps_per_subgrid;
                configurations.emplace_back(configuration);
            }
        }

        return configurations;
    }
} // namespace gridding_benchmark
