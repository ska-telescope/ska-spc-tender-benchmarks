# FFT benchmarking tool

## Repository architecture
The FFT benchmarking tool is located in the **fft** directory. This directory contains:
- the inc and src directories for the project source files
- a spack directory that contains the Spack environment used to build and run benchmarks
- an examples directory that contains examples of yaml configuration files for the benchmarks
- a CMakeLists.txt file that describes the structure of the project and its dependencies

## Setting up the environment
To build the benckmarking tool, the following dependencies are required:
- CMake
- [heffte](https://github.com/icl-utk-edu/heffte/) (to configure based on the targetted hardware)
- Intel oneAPI TBB if the PARALLELIZATION_LIBRARY CMake option is set to TBB
- yaml-cpp

Source your Spack environment setup script:
```shell
$ source <path to spack root>/share/spack/setup-env.sh
```

The *heffte* spec in spack/spack.yaml can be configured. By default it uses fftw as a CPU backend and CUDA as a GPU backend. Several other options are available. Use the command *spack info heffte* to see the possible options.

Activate the benchmarking environment:
```shell
$ spack env activate -d fft/spack
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
- ENABLE_CPU: set to ON to be able to run FFT benchmarks on CPU.
- ENABLE_GPU: set to ON to be able to run FFT benchmarks on GPU.
- PARALLELIZATION_LIBRARY: choose between OMP, TBB and NONE for the batch-level parallelization library to use during FFT benchmarks.
- VTUNE_PROFILE: set to ON to enable API calls to the ittnotify library. It restricts the profiled sections to FFT computation parts when profiling with Intel oneAPI VTune.

# Building
The build can then be launched:
```shell
$ make
```

## Running the benchmarking tool

The benchmarking tool uses a yaml configuration file system. Examples are provided in the examples directory. Here is one that runs FFTs on Nvidia GPUs using single precision floating point numbers:
```yaml
nbatches: 100
niterations: 100
dimensions: [[128, 128], [256, 256], [512, 512], [1024, 1024], [2048, 2048]]
float_types: [float]
transform_types: [forward]
hardware_types: [gpu]
```

It runs in place forward FFTs with 5 differents sizes in batches of 100 FFTs. It repeats this process 100 times. The results are currently outputted in the standard output.