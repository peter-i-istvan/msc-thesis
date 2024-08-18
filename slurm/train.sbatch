#!/bin/bash

#SBATCH -A c_gnn42
#SBATCH --partition=ai
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --mem-per-cpu=8000

singularity exec --nv -B /project/c_gnn42 train.sif python3 train_experiments.py