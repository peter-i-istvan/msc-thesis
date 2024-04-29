import torch
import numpy as np
from tqdm import tqdm
# from torch_geometric.nn import summary

from baseline_model_fusion import BaselineFusionGNN, GCN

torch.manual_seed(42)
np.random.seed(42)


# TODO: cleaner code, save to directory structure
# TODO: to(device), logging current device
def baseline_train():
    task = "scan_age"

    train_dataloader = torch.load(f"{task}_train_dataloader.pt")
    val_dataloader = torch.load(f"{task}_val_dataloader.pt")
    test_dataloader = torch.load(f"{task}_test_dataloader.pt")
    model = GCN(10, 64, 1)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    print(model)

    for epoch in range(10):
        # TRAIN
        train_loss_total = 0.0
        train_samples_total = 0.0

        model.train()
        for mesh, connectome, y in (pbar := tqdm(train_dataloader)):
            opt.zero_grad()
            y_pred = model(mesh).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

            train_loss_total += loss.detach().item()
            train_samples_total += mesh.num_graphs # 'Batch'
            # pbar.update(f"Loss: {loss.detach().item():.4f}")

        # EVAL
        val_loss_total = 0.0
        val_samples_total = 0.0

        model.eval()
        with torch.no_grad():
            for mesh, connectome, y in (pbar := tqdm(val_dataloader)):
                val_loss = loss_fn(model(mesh).squeeze(), y)
                val_loss_total += val_loss.item()
                val_samples_total += mesh.num_graphs
            
        print(f"Epoch {epoch}:\tTrain. MSE: {train_loss_total / train_samples_total:.4f}\tVal. MSE.: {val_loss_total / val_samples_total:.4f}")

    torch.save(model, f"{task}_model.pt")

    # TEST
    print("----------------TEST----------------")
    test_loss_total = 0.0
    test_samples_total = 0.0
    model.eval()
    with torch.no_grad():
        for mesh, connectome, y in (pbar := tqdm(test_dataloader)):
            test_loss = loss_fn(model(mesh).squeeze(), y)
            test_loss_total += test_loss.item()
            test_samples_total += mesh.num_graphs

    # "RMSE gives more weight to larger errors, while MAE is more robust to outliers"
    print(f"Test MSE: {test_loss_total / test_samples_total:.4f}")

def train():
    task = "scan_age"

    train_dataloader = torch.load(f"{task}_train_dataloader.pt")
    val_dataloader = torch.load(f"{task}_val_dataloader.pt")
    test_dataloader = torch.load(f"{task}_test_dataloader.pt")
    model = BaselineFusionGNN(10, 10, 32)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss(reduction='sum')
    val_loss_fn = torch.nn.L1Loss(reduction='sum')

    print(model)

    for epoch in range(10):
        # TRAIN
        train_loss_total = 0.0
        train_samples_total = 0.0

        model.train()
        for mesh, connectome, y in (pbar := tqdm(train_dataloader)):
            opt.zero_grad()
            y_pred = model(mesh, connectome).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

            train_loss_total += loss.detach().item()
            train_samples_total += mesh.num_graphs # 'Batch'
            # pbar.update(f"Loss: {loss.detach().item():.4f}")

        # EVAL
        val_loss_total = 0.0
        val_mae_total = 0.0
        val_samples_total = 0.0

        model.eval()
        with torch.no_grad():
            for mesh, connectome, y in (pbar := tqdm(val_dataloader)):
                y_pred = model(mesh, connectome).squeeze()
                val_loss = loss_fn(y_pred, y)
                val_loss_total += val_loss.item()
                val_mae_total += val_loss_fn(y_pred, y).item()
                val_samples_total += mesh.num_graphs
            
        print(f"Epoch {epoch}:\tTrain. MSE: {train_loss_total / train_samples_total:.4f}\tVal. MSE.: {val_loss_total / val_samples_total:.4f}\tVal. MAE.: {val_mae_total / val_samples_total:.4f}")

    torch.save(model, f"{task}_model.pt")

    # TEST
    print("----------------TEST----------------")
    test_loss_total = 0.0
    test_mae_total = 0.0
    test_samples_total = 0.0
    model.eval()
    with torch.no_grad():
        for mesh, connectome, y in (pbar := tqdm(test_dataloader)):
            y_pred = model(mesh, connectome).squeeze()
            test_loss = loss_fn(y_pred, y)
            test_mae_total += val_loss_fn(y_pred, y)
            test_loss_total += test_loss.item()
            test_samples_total += mesh.num_graphs

    # "RMSE gives more weight to larger errors, while MAE is more robust to outliers"
    print(f"Test MSE: {test_loss_total / test_samples_total:.4f} Test MAE: {test_mae_total / test_samples_total:.4f}")


if __name__ == "__main__":
    train()
    # baseline_train()

        