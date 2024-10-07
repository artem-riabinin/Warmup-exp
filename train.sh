#!/bin/bash
#SBATCH --job-name=pytorch_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output_%j.log

TOTAL_PROCESSES=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

MASTER_PORT=$(( ( RANDOM % 64512 ) + 1024 ))

python data/openwebtext/prepare.py

torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE \
         --nnodes=$SLURM_NNODES \
         --node_rank=$SLURM_PROCID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py