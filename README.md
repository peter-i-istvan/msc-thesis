# Decoding Neonatal Brain Development: Graph Neural Network-based Analysis of Multimodal MRI Scans

- The preprocessing pipeline for the connectome generation can be found in a separate [repository](https://github.com/peter-i-istvan/msc-thesis-preprocess/tree/main). The connectomes in CSV format should be copied to a convenient location and the paths set in `data.py`.

- The brain surface mesh data is taken as-is from the authors of [this paper](https://proceedings.mlr.press/v193/unyi22a/unyi22a.pdf) (Unyi et al.), and should be copied to a convenient location similarly to the connectomes.

- This code is the basis of a Student's Research Conference [paper](https://tdk.bme.hu/conference/VIK/2024/sessions/orvosi/paper/Az-ujszulott-agy-fejlodesenek-modellezese) (2nd prize in the Medical Applications category) and my Master's [thesis](thesis.pdf). The latter is longer and more comprehensive, consult for background information and methodology.

- The goal of this project was to establish baseline predictive model performance from connectome data (4 different types) when it comes to age regression (2 age targets), as well as exploiting the potential in combining the data and model architecture from Unyi et al. with connectome data and the GNN-based architecture built upon them. 

- The challenge was designing and fine-tuning a model architecture that can work with sparse (surface mesh) and dense (connectome) graph inputs simultaneously while keeping the trainable parameter count low enough to avoid overfit on the limited amount of data available.

- The code was mainly run in a High-Performance Computing (HPC) setting on the [Komondor Supercomputer](https://ncc.dkf.hu/en/komondor.html), therefore it uses the Singularity container runtime instead of Docker, with SLURM job scheduling. The set-up procedure can be found below. To run on Docker instead, convert the Singularity [`.def` file](singularity/train.def) to a Dockerfile then build and run accordingly.

## Repository structure

| Path | Description |
|------|--------------|
| `mesh/` | Sample GIfTI files for the mesh data. |
| `singularity/` | Build files for the Singularity container. |
| `slurm/` | SLURM batch files describing entry points and resource allocation for the data preparation and training workflows. |
| `splits/` | Subject and measurement IDs for the fixed data splits by task (BA, SA). |
| `src/` | Core source code modules used by the main script. |
| `data.py` | Data preparation script to create and serialize data loaders ahead of training for reusability. Set the proper paths to input data in the global variables before running. |
| `train.py` | A simple train script to establish baseline model performance, using vanilla PyTorch and PyTorch Geometric (PyG). |
| `train_experiments.py` | More complex train script to compare multiple model architectures and sizes as individual experiments, using PyTorch Lightning and Weights and Biases (WandB) on top of Torch and PyG. |
| `thesis.pdf` | My final Master's thesis on the topic. |

## First steps

If you clone the repository for the first time, some setup steps are needed before starting the model training process.

If you are working in a HPC environment with SLURM, the corresponding `.sbatch` file to be executed will be mentioned.

If the dataloaders are not saved to files yet, you should run `data.py` first, before `train_experiments.py`.

## Running under SLURM

Assuming the project account in the HPC cluster is `$PROJECT_ACCOUNT`, residing under `$PROJECT_PATH`:
```
module load singularity
singularity build --fakeroot --fix-perms train.sif singularity/train.def
```
builds the `train.sif` image file. Then either run:
```
srun --partition=cpu --mem-per-cpu=8000 --account=$PROJECT_ACCOUNT singularity exec -B $PROJECT_PATH train.sif python3 data.py
```
or:
```
sbatch slurm/data.sbatch
```
(Either way, it is important to have Singularity loaded before running the SLURM job)

After this, and before running the training script, you should insert your WandB key in the corresponding line in `train_experiments.py` (`WANDB_KEY=""`). This is important, because the SLURM job does not get a TTY by default, so you cannot interactively provide the key via the command line.

Then to run the training:
```
srun --partition=ai --cpus-per-gpu=8 --mem-per-cpu=8000 --gres=gpu:1 singularity exec --nv -B $PROJECT_PATH train.sif python3 train_experiments.py
```
or
```
sbatch slurm/train.sbatch
```

When commands are run with `sbatch`, the output usually gets saved in the current working directory as `slurm-{jobid}.out`. Otherwise, if it is run with `srun`, that output is written to the stdout.