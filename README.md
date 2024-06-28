# SKA SPC Tender Benchmarks

## Contents of the repository
This repository contains the following benchmarks:
- fft: Multi target Fast Fourier transform benchmarks
- gridding: gridding/degridding benchmarks based on https://gitlab.com/ska-telescope/sdp/ska-sdp-idg-bench

Each of these benchmarks is configurable using YAML files and provides formatted output data in CSV files.
A Spack environment is provided to build the benchmark suite.

## Repository architecture
Each directory contains sources to build a benchmark executables, organized as follows:
- the inc and src directories for the project source files
- a spack directory that contains the Spack environment used to build and run benchmarks
- an examples directory that contains examples of yaml configuration files for the benchmarks
- a CMakeLists.txt file that describes the structure of the project and its dependencies

## Setting up the environment
Source your Spack environment setup script:
```shell
$ source <path to spack root>/share/spack/setup-env.sh
```

The *heffte* spec in spack/spack.yaml can be configured. By default it uses fftw as a CPU backend and CUDA as a GPU backend. Several other options are available. Use the command *spack info heffte* to see the possible options.

Activate the benchmarking environment:
```shell
$ spack env activate -d spack/
```

Concretize and install the environement:
```shell
$ spack concretize -f
$ spack install -j <max number of jobs desired>
```

## Build configuration
Once the Spack environment is activated, all the dependencies are available and CMake can be launched. Simply create a build directory and run CMake:
```shell
$ mkdir build && cd build
$ cmake ..
```

## CMake configuration options
Depending on the hardware you plan on using, you might need to tweak some of the project's CMake options:
- ENABLE_CPU: set to ON to be able to run the benchmarks on CPU.
- ENABLE_GPU: set to ON to be able to run the benchmarks on GPU.
- PARALLELIZATION_LIBRARY: choose between OMP, TBB and NONE for the parallelization library.
- VTUNE_PROFILE: set to ON to enable API calls to the ittnotify library. It restricts the profiled sections to computation parts when profiling with Intel oneAPI VTune.

# Building
The build can then be launched:
```shell
$ make
```

## Running the benchmarking tool

The benchmarking tools use a yaml configuration file system. Examples are provided in the examples directory.
To run a benchmark, use the executable as follows after having activated the Spack environment to have the dependencies loaded:
```shell
$ xxxx-benchmarks path/to/config.yaml output.csv
```
