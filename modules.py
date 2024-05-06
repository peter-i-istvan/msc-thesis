import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from lightning import LightningModule, LightningDataModule
from baseline_model_fusion import BaselineFusionGNN
from torchmetrics import MeanAbsoluteError, R2Score, PearsonCorrCoef


class FusionModel(LightningModule):
    """
    Wraps the FusionGNN model
    Logs the train, validation and test metrics
    TODO: Checkpoint the best models so far
    """
    def __init__(self):
        super().__init__()
        self.model = BaselineFusionGNN(10, 10, 32)

        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.corr = PearsonCorrCoef()

    def forward(self, mesh, connectome):
        return self.model(mesh, connectome).squeeze()

    def training_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        loss = self.loss_fn(y_pred, y)
        self.log('Train/MSE', loss, batch_size=mesh.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        val_loss = self.loss_fn(y_pred, y)
        self.log('Val/MSE', val_loss, batch_size=mesh.num_graphs)
        self.log('Val/MAE', self.mae(y_pred, y), batch_size=mesh.num_graphs)
        self.log('Val/R2', self.r2(y_pred, y), batch_size=mesh.num_graphs)
        self.log('Val/Corr', self.corr(y_pred, y), batch_size=mesh.num_graphs)

    def test_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        test_loss = self.loss_fn(y_pred, y)
        self.log('Test/MSE', test_loss, batch_size=mesh.num_graphs)
        self.log('Test/MAE', self.mae(y_pred, y), batch_size=mesh.num_graphs)
        self.log('Test/R2', self.r2(y_pred, y), batch_size=mesh.num_graphs)
        self.log('Test/Corr', self.corr(y_pred, y), batch_size=mesh.num_graphs)

    def configure_optimizers(self):
        return self.optimizer
    
class FusionData(LightningDataModule):
    def __init__(
            self,
            task: str = "scan_age",
            data_dir: str = "."
    ):
        super().__init__()
        self.task = task
        self.data_dir = data_dir

    def setup(self, stage: str):
        self.train_data = torch.load(
            os.path.join(self.data_dir, f"{self.task}_train_dataloader.pt")
        )
        self.val_data = torch.load(
            os.path.join(self.data_dir, f"{self.task}_val_dataloader.pt")
        )
        self.test_data = torch.load(
            os.path.join(self.data_dir, f"{self.task}_test_dataloader.pt")
        )


    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data
    
    def test_dataloader(self):
        return self.test_data
    