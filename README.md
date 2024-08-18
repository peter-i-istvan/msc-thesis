# MSC Thesis - Péter István

To use the code, aquire the data required into folders `connectomes-csv`, `dHCP-Dani` and `SLCN`.

## First steps

If you clone the repository for the first time, some setup steps are needed before starting the model training process.

If you are working in a HPC environment with slurm, the corresponding `.sbatch` file to be executed will be mentioned.

### Dataloader creation

To create the Torch `Dataloader`s, run the [`data.py`](data.py) script after uncommenting the relevant lines in the main body of the script. The paths at the beginning (denoted by uppercase variable names) should be set according to the new environment.

### Running under SLURM

Assuming project name is `c_gnn42`:
```
module load singularity
singularity build --fakeroot --fix-perms train.sif singularity/train.def
```
Then either run
```
srun --partition=cpu --mem-per-cpu=8000 --account=c_gnn42 singularity exec -B /project/c_gnn42 train.sif python3 data.py
```
or
```
sbatch slurm/data.sbatch
```
(either way it is important to have singularity loaded before running the SLURM job)

When commands are run with `sbatch`, the output usually gets saved in the current working directory as `slurm-{jobid}.out`. Otherwise, if it is run with `srun`, that output is written to the stdout.