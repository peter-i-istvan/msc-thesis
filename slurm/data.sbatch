#!/bin/bash

#SBATCH -A c_gnn42
#SBATCH --partition=cpu
#SBATCH --job-name=dhcp-prepare-data
#SBATCH --mem-per-cpu=8000

singularity exec -B /project/c_gnn42 train.sif python3 data.py