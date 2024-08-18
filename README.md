# MSC Thesis - Péter István

To use the code, aquire the data required into folders `connectomes-csv`, `dHCP-Dani` and `SLCN`.

## First steps

If you clone the repository for the first time, some setup steps are needed before starting the model training process.

If you are working in a HPC environment with slurm, the corresponding `.sbatch` file to be executed will be mentioned.

If the dataloaders are not saved to files yet, you should run `data.py` first, before `train_experiments.py`.

## Running under SLURM

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

After this, and before running the train script, you should insert your WandB key in the corresponding line in `train_experiments.py` (`WANDB_KEY=""`). This is important, because the SLURM job does not get a TTY by default, so you cannot interactively provide the key via the command line.

Then to run the training:
```
srun --partition=ai --cpus-per-gpu=8 --mem-per-cpu=8000 --gres=gpu:1 singularity exec --nv -B /project/c_gnn42 train.sif python3 train_experiments.py
```
or
```
sbatch slurm/train.sbatch
```

> TODO: API key handling

When commands are run with `sbatch`, the output usually gets saved in the current working directory as `slurm-{jobid}.out`. Otherwise, if it is run with `srun`, that output is written to the stdout.