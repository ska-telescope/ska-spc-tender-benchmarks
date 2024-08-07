# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-jammy:latest as builder


# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir -p /opt/spack-environment && \
set -o noclobber \
&&  (echo spack: \
&&   echo '  specs:' \
&&   echo '  - cmake arch=x86_64' \
&&   echo '  - intel-oneapi-compilers arch=x86_64' \
&&   echo '  - intel-oneapi-tbb arch=x86_64' \
&&   echo '  - intel-oneapi-mkl arch=x86_64' \
&&   echo '  - intel-oneapi-mpi arch=x86_64' \
&&   echo '  - yaml-cpp arch=x86_64' \
&&   echo '  - heffte+mkl+rocm amdgpu_target=gfx90a arch=x86_64' \
&&   echo '  view: /opt/views/view' \
&&   echo '  concretizer:' \
&&   echo '    unify: when_possible' \
&&   echo '  mirrors:' \
&&   echo '    E4S: https://cache.e4s.io' \
&&   echo '  config:' \
&&   echo '    install_tree: /opt/software') > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && spack env activate . && spack concretize -f && spack install -y --fail-fast && spack gc -y

# Strip all the binaries
#RUN find -L /opt/views/view/* -type f -exec readlink -f '{}' \; | \
#    xargs file -i | \
#    grep 'charset=binary' | \
#    grep 'x-executable\|x-archive\|x-sharedlib' | \
#    awk -F: '{print $1}' | xargs strip

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . > activate.sh


# Bare OS image to run the installed executables
FROM ubuntu:22.04

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software

# paths.view is a symlink, so copy the parent to avoid dereferencing and duplicating it
COPY --from=builder /opt/views /opt/views

RUN { \
      echo '#!/bin/sh' \
      && echo '.' /opt/spack-environment/activate.sh \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh \
&& ln -s /opt/views/view /opt/view


ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" ]

