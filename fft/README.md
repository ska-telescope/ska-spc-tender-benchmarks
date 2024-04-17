# FFT benchmarking tool

## Repository architecture
The FFT benchmarking tool is located in the **fft** directory. This directory contains:
- the inc and src directories for the project source files
- a spack directory that contains the Spack environment used to build and run benchmarks
- an examples directory that contains examples of yaml configuration files for the benchmarks
- a CMakeLists.txt file that describes the structure of the project and its dependencies

## Setting up the environment
To build the benckmarking tool, the following dependencies are required:
- CMake >= 3.23
- GCC runtimes (we used version 11)
- Intel oneAPI compilers
- Intel oneAPI TBB
- Intel oneAPI MKL
- yaml-cpp

We created a spack environment file to easily install these dependencies. To reproduce this environment, first clone the repository:
```shell
$ git clone https://gitlab.com/ska-telescope/ska-spc-tender-benchmarks.git
$ cd ska-spc-tender-benchmarks
$ git checkout -b rac-48-fft-benchmark
```

Then, source your Spack environment setup script:
```shell
$ source <path to spack root>/share/spack/setup-env.sh
```

It may be useful on some systems to clean up the environment before activating the spack environment. For instance, on CSD3, the **CPATH** environment variables needs to be cleared as it provides a system version of Intel TBB that conflicts with the one we are using.
```shell
$ unset CPATH
```

You can now activate the benchmarking environment:
```shell
$ spack env activate -d fft/spack
```

This will resolve and install its dependencies:
```shell
$ spack concretize -f
$ spack install -j <max number of jobs desired>
```

As the Spack install command will consume a lot CPU resources, you may want to favour compute nodes if on a cluster.
Please note that these dependencies do not include CUDA. As the right version to load depends on the hardware you plan on using, you should load the CUDA version installed on the targeted system.

You can check that the installation went well by trying to use the **icpx** command.

## Building
Once the Spack environment is activated, all the dependencies are available and the project can be built. First, configure the CMake project by specifying the right compilers. Here, we use **icpx** as a C++ compiler and **gcc** as a C compiler. They are both provided by the **intel-oneapi-compilers** Spack package. From the source directory, do:
```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_ARCHITECTURES=native -DFFT_BENCHMARK_ENABLE_CPU=1 -DFFT_BENCHMARK_ENABLE_NVIDIA=1 -DCUDAToolkit_ROOT=<path to the system CUDA toolkit> ..
```

Then, build the project:
```shell
$ make
```

## Running the benchmarking tool

The benchmarking tool uses a yaml configuration file system. Examples are provided in the examples directory. Here is one that runs FFTs on Nvidia GPUs using single precision floating point numbers:
```yaml
nbatches: 100
niterations: 100
in_place: true
dimensions: [[128, 128], [256, 256], [512, 512], [1024, 1024], [2048, 2048]]
float_types: [float]
transform_types: [forward]
hardware_types: [nvidia]
```

It runs in place forward FFTs with 5 differents sizes in batches of 100 FFTs. It repeats this process 100 times. The results are currently outputted in the standard output.