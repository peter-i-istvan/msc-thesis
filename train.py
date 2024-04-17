import torch
from tqdm import tqdm
# from torch_geometric.nn import summary

from baseline_model_fusion import BaselineFusionGNN

torch.manual_seed(42)


if __name__ == "__main__":
    task = "scan_age"
    split = "train"

    model = BaselineFusionGNN(10, 10, 32)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    print(model)

    dataloader = torch.load(f"{task}_{split}_dataloader.pt")
    for epoch in range(10):
        loss_total = 0.0
        samples_total = 0.0

        for mesh, connectome, y in (pbar := tqdm(dataloader)):
            opt.zero_grad()
            y_pred = model(mesh, connectome).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

            loss_total += loss.detach().item()
            samples_total += mesh.num_graphs # 'Batch'
            # pbar.update(f"Loss: {loss.detach().item():.4f}")

        print(f"Avg. MSE: {loss_total / samples_total: .4f}")
        