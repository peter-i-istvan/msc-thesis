import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from lightning import LightningModule, LightningDataModule
from torchmetrics import MeanAbsoluteError, R2Score, PearsonCorrCoef


BATCH_SIZE = 4

class GNNModule(LightningModule):
    """
    Wraps the given Model and sets the 
    Logs the train, validation and test metrics
    TODO: Checkpoint the best models so far
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

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
        train_dataset = torch.load(os.path.join(self.data_dir, "train", f"{self.task}_train_dataloader.pt")).dataset
        self.train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)
        
        val_dataset = torch.load(os.path.join(self.data_dir, "val", f"{self.task}_val_dataloader.pt")).dataset
        self.val_data = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=15)

        test_dataset = torch.load(os.path.join(self.data_dir, "test", f"{self.task}_test_dataloader.pt")).dataset
        self.test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=15)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data
    
    def test_dataloader(self):
        return self.test_data
    