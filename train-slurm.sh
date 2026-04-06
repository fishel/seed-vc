#!/bin/bash
#
#SBATCH --job-name=vc-train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6  
#SBATCH --time=48:00:00
#SBATCH --partition=standard-g
#SBATCH --mem=480G

source ./init.sh

export MASTER_HOST=$( scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$( host $MASTER_HOST | cut -d " " -f 4 )
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 10000))

cmd="$@"

date

# Launch one Python per task/GPU.
#srun --label bash -lc "$cmd" 

$cmd
