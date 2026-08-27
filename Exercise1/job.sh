#!/bin/bash
#PBS -q lecture-g
#PBS -l select=1
#PBS -l walltime=00:01:00
#PBS -W group_list=gt00
#PBS -j oe

cd $PBS_O_WORKDIR

# =========================================
# 通常の方法
# =========================================
module load python
python3 sample.py


# Step2: a way to use apptainer

# =========================================
# コンテナを用いる方法
# 事前に以下を実行し、 python.sif を作成すること
# apptainer pull python.sif docker://python:3.12
# =========================================
module load apptainer
apptainer exec python.sif \ 
python3 sample.py
