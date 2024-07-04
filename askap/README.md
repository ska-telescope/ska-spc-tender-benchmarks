# Description
This part of the benchmark suite imports a set of benchmarks from the Australian SKA Pathfinder. A description of these benchmarks can be found [here](https://github.com/ATNF/askap-benchmarks).

# How to build
This set of benchmark uses OpenACC and relies on the NVHPC compiler to target Nvidia GPUs. It also requires MPI and FFTW. All these dependencies are automatically managed using Spack. The only thing required before building is to set the *SPACK_ROOT* environment variable to point to a Spack repo clone.

If none is present on the system, just clone one as follows:
```
git clone https://github.com/spack/spack.git spack-repo
```

Then build the benchmarks:
```
export SPACK_ROOT=<path to spack repo>
./build.sh
```

Each benchmark is then available under the *build* directory and can be executed within the Spack environment. For example:
```
source <path to spack repo>/share/spack/setup-env.sh
spack env activate -d ./spack
./build/tConvolveACC/tConvolveACC
```

Once the benchmarks are built, *the repo/askap-benchmarks* directory contains the instructions on how the run each benchmark.