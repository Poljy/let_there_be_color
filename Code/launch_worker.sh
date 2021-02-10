#!/bin/bash

export MASTER_PORT=$1
export MASTER_ADDR=$2
export WORLD_SIZE=$3
export NODE_RANK=$4
export LOCAL_RANK=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
python main.py --n_nodes $WORLD_SIZE
