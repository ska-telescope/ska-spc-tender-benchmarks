#!/bin/bash
#SBATCH --output=compile_log-%x.%j.out
#SBATCH --error=compile_log-%x.%j.err
#SBATCH --time=02:00:00


srun -u ./setup.sh 2>&1 | tee -a osu_compile.log
