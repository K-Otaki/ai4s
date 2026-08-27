#! /bin/bash
#PBS -q lecture-g
#PBS -l select=2
#PBS -l walltime=00:05:00
#PBS -W group_list=gt00
#PBS -j oe

cd $PBS_O_WORKDIR
cd "$(pwd -P)"
module load apptainer

MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

mpirun bash -c "
  echo host=\$(hostname), MPI_rank=\${OMPI_COMM_WORLD_RANK}
  apptainer exec --nv pytorch.sif \
    torchrun --nnodes=2 --nproc-per-node=1 \
      --node-rank=\${OMPI_COMM_WORLD_RANK} \
      --master-addr=${MASTER_ADDR} \
      --master-port=${MASTER_PORT} \
      mnist.py
"
