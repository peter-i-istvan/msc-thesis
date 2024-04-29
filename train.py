import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import MeanAbsoluteError
from torch_geometric.loader import DataLoader

from baseline_model_fusion import BaselineFusionGNN, GCN

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device: ", device)
print("------------------------")


# TODO: cleaner code, save to directory structure
# TODO: to(device), logging current device
def baseline_train():
    task = "scan_age"

    train_dataloader = torch.load(f"{task}_train_dataloader.pt")
    val_dataloader = torch.load(f"{task}_val_dataloader.pt")
    test_dataloader = torch.load(f"{task}_test_dataloader.pt")

    train_dataset = train_dataloader.dataset
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_dataset = val_dataloader.dataset
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    test_dataset = test_dataloader.dataset
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = GCN(10, 64, 1).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    print(model)
    train_mses = []
    val_mses = []

    for epoch in range(30):
        # TRAIN
        train_loss_total = 0.0
        train_samples_total = 0.0

        model.train()
        for mesh, _, y in (pbar := tqdm(train_dataloader)):
            opt.zero_grad()
            y_pred = model(mesh.to(device)).squeeze()
            loss = loss_fn(y_pred, y.to(device))
            loss.backward()
            opt.step()

            train_loss_total += loss.detach().item()
            train_samples_total += mesh.num_graphs # 'Batch'
            # pbar.update(f"Loss: {loss.detach().item():.4f}")

        # EVAL
        val_loss_total = 0.0
        val_samples_total = 0.0

        model.eval()
        mae = MeanAbsoluteError().to(device)
        with torch.no_grad():
            for mesh, _, y in (pbar := tqdm(val_dataloader)):
                y_pred = model(mesh.to(device)).squeeze()
                val_loss = loss_fn(y_pred, y.to(device))
                mae.update(y_pred, y.to(device))
                val_loss_total += val_loss.item()
                val_samples_total += mesh.num_graphs
            
        train_mses.append(train_loss_total / train_samples_total)
        val_mses.append(val_loss_total / val_samples_total)

        print(f"Epoch {epoch}:\tTrain. MSE: {train_loss_total / train_samples_total:.4f}\tVal. MSE.: {val_loss_total / val_samples_total:.4f}\tVal. MAE.: {mae.compute().item():.4f}")

    # torch.save(model, f"{task}_model.pt")

    # TEST
    print("----------------TEST----------------")
    test_loss_total = 0.0
    test_samples_total = 0.0
    model.eval()
    mae = MeanAbsoluteError().to(device)
    with torch.no_grad():
        for mesh, _, y in (pbar := tqdm(test_dataloader)):
            y_pred = model(mesh.to(device)).squeeze()
            test_loss = loss_fn(y_pred, y.to(device))
            mae.update(y_pred, y.to(device))
            test_loss_total += test_loss.item()
            test_samples_total += mesh.num_graphs

    test_mse = test_loss_total / test_samples_total

    # "RMSE gives more weight to larger errors, while MAE is more robust to outliers"
    print(f"Test MSE: {test_mse:.4f}\tTest. MAE.: {mae.compute().item():.4f}")

    plt.plot(train_mses, label="Train MSE")
    plt.plot(val_mses, label="Val. MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error [${week}^{2}$]")
    plt.axhline(test_mse, ls="--", color="r", label="Test MSE")
    plt.annotate(f"Test MSE: {test_mse:.4f}", (0, test_mse+0.2))
    plt.legend()
    plt.ylim(0, 10)
    plt.savefig("train.png")

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

        