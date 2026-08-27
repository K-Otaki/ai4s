#! /bin/bash
#PBS -q lecture-g
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -W group_list=gt00
#PBS -j oe

module load apptainer

cd $PBS_O_WORKDIR
mkdir -p apptainer-cache
mkdir -p apptainer-tmp

export APPTAINER_CACHEDIR=$PBS_O_WORKDIR/apptainer-cache
export APPTAINER_TMPDIR=$PBS_O_WORKDIR/apptainer-tmp

apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:25.04-py3
