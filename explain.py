import torch
import random
from tqdm import tqdm
from torch.optim import Adam, SGD
from lightning import seed_everything
from torch.nn import Module, BatchNorm1d, ReLU, Linear, Sequential, MSELoss
from src.models import MeshGNN, MeshConf, HeadConf, concrete_sample, apply_mask
import matplotlib.pyplot as plt


seed_everything(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class PGExplainer(Module):
    def __init__(self, head_in=32, head_hidden=32, size_coef: float = 0.00005, entr_coef: float = 1.0) -> None:

        super().__init__()

        self.size_coef = size_coef
        self.entr_coef = entr_coef

        self.target_loss = torch.nn.MSELoss()

        self.head = Sequential(
            Linear(head_in, head_hidden),
            ReLU(),
            BatchNorm1d(head_hidden),
            Linear(head_hidden, 1)
        ).to(device)

    def forward(self, features):
        return self.head(features)

    def fit(self, epochs, model, dataloader):
        optimizer = Adam(model.parameters(), lr=1e-2)

        batch_mses, batch_ents, batch_sizes = [], [], []

        for _ in tqdm(range(epochs), desc='Fitting explainer'):
            acc_loss, acc_mse, acc_size, acc_ent = 0, 0, 0, 0

            for mesh, _, y in dataloader:
                optimizer.zero_grad()

                z = model.feature_extractor(mesh.to(device), pool=False)
                z = self(z)
                mask = concrete_sample(z, temperature=2.0)
                masked_mesh = apply_mask(mesh, mask)
                y_pred = model(masked_mesh, None).squeeze()

                mse, size, ent = self._loss(y_pred, y.to(device), mask)
                loss = mse + size + ent
                loss.backward()
                optimizer.step()

                acc_loss += loss.cpu().detach().item()
                acc_mse += mse.cpu().detach().item()
                acc_size += size.cpu().detach().item()
                acc_ent += ent.cpu().detach().item()

            mean_batch_loss = acc_loss / len(dataloader)
            print(f"{mean_batch_loss=:.4f}")

            batch_mses.append(acc_mse / len(dataloader))
            batch_sizes.append(acc_size / len(dataloader))
            batch_ents.append(acc_ent / len(dataloader))

        plt.plot(batch_mses, label="mse")
        plt.plot(batch_ents, label="ent")
        plt.plot(batch_sizes, label="size")
        plt.legend()
        plt.savefig("explainer.png")

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor]:
        loss = self.target_loss(y_hat, y)

        # Regularization loss:
        mask = node_mask.sigmoid()
        size_loss = mask.sum() * self.size_coef
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.entr_coef

        return loss, size_loss, mask_ent_loss


def evaluate(model, dataloader, loss_fn, split):
    loss_acc, n_graphs = 0, 0

    model.eval()
    with torch.no_grad():
        for mesh, _, y in dataloader:
            random_bit = random.random() > 0.5
            y_pred = model(mesh.to(device), None, mask=random_bit).squeeze()
            loss = loss_fn(y.to(device), y_pred)
            loss_acc += loss.cpu().detach().item()
            n_graphs += mesh.num_graphs

    avg_loss = loss_acc / n_graphs
    print(f"Avg {split} MSE: {avg_loss}")
    return avg_loss


def train_with_random() -> MeshGNN:
    # train on birth age with random masking:
    model = MeshGNN(mesh_conf=MeshConf(), head_conf=HeadConf(hidden=32)).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    loss_fn = MSELoss()

    train_dataloader = torch.load("data/birth_age/train/birth_age_train_dataloader.pt")
    val_dataloader = torch.load("data/birth_age/val/birth_age_val_dataloader.pt")
    test_dataloader = torch.load("data/birth_age/test/birth_age_test_dataloader.pt")

    for _ in tqdm(range(50)):
        loss_acc, n_graphs = 0, 0

        model.train()
        for batch in train_dataloader:
            mesh, _, y = batch
            optimizer.zero_grad()

            random_bit = random.random() > 0.75
            y_pred = model(mesh.to(device), None, mask=random_bit).squeeze()
            loss = loss_fn(y.to(device), y_pred)

            loss.backward()
            optimizer.step()

            loss_acc += loss.cpu().detach().item()
            n_graphs += mesh.num_graphs

        print(f"Avg Train MSE: {loss_acc / n_graphs}")

        evaluate(model, val_dataloader, loss_fn, "Val")

    evaluate(model, test_dataloader, loss_fn, "Test")
    torch.save(model, "model_random.pt")

    return model


def explain(model: MeshGNN):
    # dataloader = torch.load("birth_age_train_dataloader.pt")
    dataloader = torch.load("data/birth_age/test/birth_age_test_dataloader.pt")

    # freeze feature extractor
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    explainer = PGExplainer()
    explainer.fit(epochs=50, model=model, dataloader=dataloader)

    return explainer


def main():
    # model = train_with_random()
    model = torch.load("models/model.pt").model.to(device)
    explainer = explain(model)
    torch.save(explainer, "explainer.pt")


if __name__ == '__main__':
    main()
