#!/bin/bash

#SBATCH -A project02441
#SBATCH --job-name=ad-gpu-sycl-clean
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=special_tp1
#SBATCH --gres=gpu:pvc128g:4
#SBATCH --time=30:00
#SBATCH --mem-per-cpu=1000MB
#SBATCH --mail-type=END,FAIL

module load intel/2024.1

# Executing the SYCL version
WORK_DIR=${PWD}
cd ${WORK_DIR} && echo "${WORK_DIR}" && make DEVICE=XeGPU PLATFORM=PVC TESTLS=ad test && make DEVICE=XeGPU PLATFORM=PVC TESTLS=ad XMX=ON test

exit 0

