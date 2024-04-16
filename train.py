import torch
from tqdm import tqdm

from baseline_model_fusion import BaselineFusionGNN

torch.manual_seed(42)


if __name__ == "__main__":
    task = "scan_age"
    split = "train"

    model = BaselineFusionGNN(10, 10, 32)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    dataloader = torch.load(f"{task}_{split}_dataloader.pt")
    for mesh, connectome, y in (pbar := tqdm(dataloader)):
        opt.zero_grad()
        y_pred = model(mesh, connectome).squeeze()
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        pbar.update(f"Loss: {loss.detach().item()}")
        