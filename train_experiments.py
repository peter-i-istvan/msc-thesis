import wandb
from modules import FusionModel, FusionData
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger


if __name__ == "__main__":
    wandb.login()
    # Do per experiment
    seed_everything(42, workers=True)
    wandb_logger = WandbLogger(project="Diplomaterv")
    
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=10,
        default_root_dir="checkpoint/",
        log_every_n_steps=10,
        deterministic=True
    )

    params = {}

    model = FusionModel(**params)
    data = FusionData()

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data) 

# Baseline with 
# Mesh only, hidden dim 32, 16, 8 (log model parameters as well)
