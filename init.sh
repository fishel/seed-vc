#!/bin/bash

module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ./venv/bin/activate
