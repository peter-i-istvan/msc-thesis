import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader
from lightning import LightningModule, LightningDataModule
from torchmetrics import MeanAbsoluteError, R2Score, PearsonCorrCoef


class GNNModule(LightningModule):
    """
    Wraps the given Model and sets the loss and metric attributes.
    Logs the train, validation and test metrics.
    """
    def __init__(self, model, lr=0.001, optimizer="sgd"):
        super().__init__()
        self.model = model

        self.loss_fn = MSELoss()
        if optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.corr = PearsonCorrCoef()

    def forward(self, mesh, connectome):
        y_pred = self.model(mesh, connectome).squeeze()
        # If the batch has only one sample, the prediction will have Size([]) instead of Size([1])
        # This may cause a problem with TorchMetrics
        if y_pred.ndim == 0:
            y_pred = y_pred.unsqueeze(0)
        
        return y_pred

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
        
        batch_size = mesh.num_graphs    

        self.log('Val/MSE', val_loss, batch_size=batch_size)
        self._update_metrics(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('Val/MAE', self.mae.compute(), prog_bar=True)
        self.log('Val/R2', self.r2.compute(), prog_bar=True)
        self.log('Val/Corr', self.corr.compute(), prog_bar=True)
        self._reset_metrics()

    def test_step(self, batch, batch_idx):
        mesh, connectome, y = batch
        y_pred = self(mesh, connectome)
        test_loss = self.loss_fn(y_pred, y)

        batch_size = mesh.num_graphs

        self.log('Test/MSE', test_loss, batch_size=batch_size)
        self._update_metrics(y_pred, y)

    def on_test_epoch_end(self):
        self.log('Test/MAE', self.mae.compute(), prog_bar=True)
        self.log('Test/R2', self.r2.compute(), prog_bar=True)
        self.log('Test/Corr', self.corr.compute(), prog_bar=True)
        self._reset_metrics()

    def _update_metrics(self, y_pred, y):
        self.mae.update(y_pred, y)
        self.r2.update(y_pred, y)
        self.corr.update(y_pred, y)

    def _reset_metrics(self):
        self.mae.reset()
        self.r2.reset()
        self.corr.reset()

    def configure_optimizers(self):
        return self.optimizer


class FusionData(LightningDataModule):
    def __init__(
            self,
            task: str = "scan_age",
            data_dir: str = ".",
            batch_size: int = 4
    ):
        super().__init__()
        self.task = task
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        train_dataset = torch.load(os.path.join(self.data_dir, "train", f"{self.task}_train_dataloader.pt")).dataset
        self.train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)
        
        val_dataset = torch.load(os.path.join(self.data_dir, "val", f"{self.task}_val_dataloader.pt")).dataset
        self.val_data = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=15)

        test_dataset = torch.load(os.path.join(self.data_dir, "test", f"{self.task}_test_dataloader.pt")).dataset
        self.test_data = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=15)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data
    
    def test_dataloader(self):
        return self.test_data
    