import os
import torch
import pandas as pd
import lightning as L
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics import MeanAbsoluteError
from torch_geometric.loader import DataLoader
import wandb

# Initialize Weights & Biases
wandb.init(project='your-project-name', entity='your-username')

from baseline_model_fusion import BaselineFusionGNN

torch.set_float32_matmul_precision('high') # high|medium
L.seed_everything(42)


class FusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BaselineFusionGNN(10, 10, 32)
        # self.model = BaselineFusionGNN(
        #     mesh_in_channels=10,
        #     connecome_in_channels=10,
        #     mesh_hidden_channels=64,
        #     connectome_hidden_channels=40,
        #     fc_hidden_channels=64
        # )
        self.loss_fn = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00004)
        # self.validation_step_losses = []

    def forward(self, mesh, connectome):
        return self.model(mesh, connectome).squeeze()

    def training_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, batch_size=mesh.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        val_loss = self.loss_fn(y_pred, y)
        # self.validation_step_losses.append(val_loss)
        self.log('val_loss', val_loss, batch_size=mesh.num_graphs)
        self.mae.update(y_pred, y)

    def test_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        test_loss = self.loss_fn(y_pred, y)
        self.log('test_loss', test_loss, batch_size=mesh.num_graphs)
        self.mae.update(y_pred, y)

    # def on_validation_epoch_start(self):
    #     self.validation_step_losses = []

    def on_validation_epoch_end(self):
        self.log('val_mae', self.mae.compute())
        # taken from baseline repo
        # mean_val_loss = torch.mean(torch.tensor(self.validation_step_losses))
        # if mean_val_loss < 1.0:
        #     print("Reduced")
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = 0.0001
        self.mae.reset()


    def on_test_epoch_end(self):
        self.log('test_mae', self.mae.compute())
        self.mae.reset()

    def configure_optimizers(self):
        return self.optimizer


def rebatch_dataloader(dataloader, batch_size, shuffle, num_workers):
    dataset = dataloader.dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def plot(train_mses, val_mses, test_loss, fname="train.png", max_y=5):
    plt.plot(train_mses, label="Train MSE")
    plt.plot(val_mses, label="Val. MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.axhline(test_loss, ls="--", color="r", label="Test MSE")
    plt.annotate(f"Test MSE: {test_loss:.4f}", (0, test_loss + 0.2))
    plt.legend()
    plt.ylim(0, max_y)
    plt.savefig(fname)


def train(task):
    train_dataloader = rebatch_dataloader(torch.load(f"{task}_train_dataloader.pt"), batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = rebatch_dataloader(torch.load(f"{task}_val_dataloader.pt"), batch_size=8, shuffle=False, num_workers=8)
    test_dataloader = rebatch_dataloader(torch.load(f"{task}_test_dataloader.pt"), batch_size=8, shuffle=False, num_workers=8)
    
    model = FusionModel()

    trainer = L.Trainer(max_epochs=100, log_every_n_steps=1, logger=L.pytorch.loggers.WandbLogger())
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # TEST
    # test_results = trainer.test(dataloaders=test_dataloader)
    # test_loss = test_results[0]['test_loss']
    # test_mae = test_results[0]['test_mae']

    # # Plotting
    # df = pd.read_csv(os.path.join(model.logger.log_dir, 'metrics.csv'))
    # train_mses = df.loc[df["train_loss"].notna(), ["epoch", "train_loss"]].groupby("epoch").agg('mean').values
    # val_mses = df.loc[df["val_loss"].notna(), ["epoch", "val_loss"]].groupby("epoch").agg('mean').values
    # plot(train_mses, val_mses, test_loss)


if __name__ == "__main__":
    train("birth_age")
