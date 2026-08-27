#! /bin/bash
#PBS -q lecture-g
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -W group_list=gt00
#PBS -j oe

cd $PBS_O_WORKDIR
cd "$(pwd -P)"
module load apptainer

apptainer exec --nv pytorch.sif python3 mnist.py

