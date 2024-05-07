import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from src.lightning_modules import GNNModule, FusionData
from src.models import FusionGNN, MeshGNN, ConnectomeGNN, MeshConf, ConnectomeConf, HeadConf


def experiment():
    """Initialize the components needed for the experiment"""
    # Define model and train configuration
    
    # setup 1.
    task = "scan_age"
    kind = "fusion"
    mesh = MeshConf()
    connectome = ConnectomeConf()
    head = HeadConf(hidden=32)
    seed = 42

    # setup 2.
    # task = "scan_age"
    # kind = "mesh"
    # mesh = MeshConf()
    # connectome = None
    # head = HeadConf(hidden=32)
    # seed = 42

    # setup 3.
    # task = "scan_age"
    # kind = "connectome"
    # mesh = None
    # connectome = ConnectomeConf()
    # head = HeadConf(hidden=10)
    # seed = 42

    # lr_reduction = "" # reduce_once, reduce_on_plateau

    seed_everything(seed, workers=True)

    # Define model and data
    if kind == "fusion":
        model = FusionGNN(connectome, mesh, head)
        name = f"{task}-{kind}-mesh-{mesh.hidden}-conn-{connectome.conv_channel}-{connectome.conv_out}-head-{head.hidden}-seed-{seed}"
    elif kind == "mesh":
        model = MeshGNN(mesh, head)
        name = f"{task}-{kind}-mesh-{mesh.hidden}-head-{head.hidden}-seed-{seed}"
    else: # connectome
        model = ConnectomeGNN(connectome, head)
        name = f"{task}-{kind}-conn-{connectome.conv_channel}-{connectome.conv_out}-head-{head.hidden}-seed-{seed}"

    model = GNNModule(model=model)
    data = FusionData(task=task, data_dir=f"data/{task}/")

    # # Set up components for trainer
    wandb_logger = WandbLogger(
        project="Diplomaterv",
        name=name
    )

    checkpoint_callback = ModelCheckpoint(monitor='Val/MAE')
    summary_callback = ModelSummary(max_depth=2) # more detailed #param breakdown

    # # TODO: LR reduction (None vs One Time vs Reduce on Plateau)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=2,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, summary_callback]
        # deterministic=True,
        # accelerator='cpu'
    )

    # # Train
    trainer.fit(model, datamodule=data)

    # Test
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=data)


if __name__ == "__main__":
    wandb.login()
    experiment()

# Baseline with 
# Mesh only, hidden dim 32, 16, 8 (log model parameters as well)
