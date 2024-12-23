import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from src.lightning_modules import GNNModule, FusionData
from src.models import FusionGNN, MeshGNN, ConnectomeGNN, MeshConf, ConnectomeConf, HeadConf

# INSERT WANDB KEY HERE
WANDB_KEY = ""
wandb.login(key=WANDB_KEY)

# Define model and train configuration

# setup 1.
# task = "scan_age"
# kind = "fusion"
# mesh = MeshConf()
# connectome = ConnectomeConf()
# head = HeadConf(hidden=32)

# setup 2.
# task = "scan_age"
# kind = "mesh"
# mesh = MeshConf()
# connectome = None
# head = HeadConf(hidden=32)

# setup 3.
# task = "scan_age"
# kind = "connectome"
# mesh = None
# connectome = ConnectomeConf()
# head = HeadConf(hidden=10)

# setup 4.
# task = "scan_age"
# kind = "mesh"
# mesh = MeshConf(hidden=64)
# connectome = None
# head = HeadConf(hidden=64)

# setup 5.
# task = "scan_age"
# kind = "fusion"
# mesh = MeshConf()
# connectome = ConnectomeConf(conv_channel=32, conv_out=32)
# head = HeadConf(hidden=32)

# setup 6.
# task = "scan_age"
# kind = "fusion"
# mesh = MeshConf()
# connectome = ConnectomeConf(agg="mean")
# head = HeadConf(hidden=32)

# setup 7.
# task = "scan_age"
# kind = "fusion"
# mesh = MeshConf()
# connectome = ConnectomeConf(conv_channel=32, conv_out=32, agg="mean")
# head = HeadConf(hidden=32)

# setup 8.
# task = "scan_age"
# kind = "connectome"
# mesh = None
# connectome = ConnectomeConf(conv_channel=16, conv_out=16, agg="mean")
# head = HeadConf(hidden=16)

# lr = 0.05
# bs = 512
# seed = 42

experiments = [
    {
        "task": "birth_age",
        "kind": "mesh",
        "mesh": MeshConf(),
        "connectome": None,
        "head": HeadConf(hidden=32),
        "optimizer": "sgd",
        "lr": 0.001,
        "bs": 64,
        "seed": 42
    },
    {
        "task": "birth_age",
        "kind": "mesh",
        "mesh": MeshConf(),
        "connectome": None,
        "head": HeadConf(hidden=32),
        "optimizer": "adam",
        "lr": 0.001,
        "bs": 64,
        "seed": 42
    },
    {
        "task": "scan_age",
        "kind": "mesh",
        "mesh": MeshConf(),
        "connectome": None,
        "head": HeadConf(hidden=32),
        "optimizer": "sgd",
        "lr": 0.001,
        "bs": 64,
        "seed": 42
    },
    {
        "task": "scan_age",
        "kind": "mesh",
        "mesh": MeshConf(),
        "connectome": None,
        "head": HeadConf(hidden=32),
        "optimizer": "adam",
        "lr": 0.001,
        "bs": 64,
        "seed": 42
    }
]


def experiment(task, kind, mesh, connectome, head, optimizer, lr, bs, seed, debug=False):
    """Initialize the components needed for the experiment"""
    # lr_reduction = "" # reduce_once, reduce_on_plateau

    seed_everything(seed, workers=True)

    # Define model and data
    if kind == "fusion":
        model = FusionGNN(connectome, mesh, head)
        name = f"{task}-{kind}-mesh-{mesh.hidden}-conn-{connectome.conv_channel}-{connectome.conv_out}-{connectome.agg}-head-{head.hidden}"
    elif kind == "mesh":
        model = MeshGNN(mesh, head)
        name = f"{task}-{kind}-mesh-{mesh.hidden}-head-{head.hidden}"
    else: # connectome
        model = ConnectomeGNN(connectome, head)
        name = f"{task}-{kind}-conn-{connectome.conv_channel}-{connectome.conv_out}-{connectome.agg}-head-{head.hidden}"

    name += f"-{optimizer}-lr-{lr:.4f}-bs-{bs}-seed-{seed}"

    model = GNNModule(model=model, lr=lr, optimizer=optimizer)
    data = FusionData(task=task, data_dir=f"data/{task}/", batch_size=bs)

    # Set up components for trainer
    if not debug:
        wandb_logger = WandbLogger(
            project="Diplomaterv",
            name=name
        )

    checkpoint_callback = ModelCheckpoint(monitor='Val/MAE', verbose=True)
    summary_callback = ModelSummary(max_depth=2)  # more detailed #param breakdown

    # # TODO: LR reduction (None vs One Time vs Reduce on Plateau)
    trainer = Trainer(
        logger=wandb_logger if not debug else None,
        max_epochs=200,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, summary_callback]
        # deterministic=True,
        # accelerator='cpu'
    )

    # # Train
    trainer.fit(model, datamodule=data)

    # Test
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=data)

    if not debug:
        wandb.finish()


if __name__ == "__main__":
    wandb.login()
    for e in experiments:
        experiment(**e)

# Baseline with 
# Mesh only, hidden dim 32, 16, 8 (log model parameters as well)
