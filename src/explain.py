import torch
import random
from torch.optim import Adam
from lightning import seed_everything
from torch.nn import Module, BatchNorm1d, ReLU, Linear, Sequential, MSELoss
from models import MeshGNN, MeshConf, HeadConf, concrete_sample, apply_mask


random.seed(0)
seed_everything(0)


class PGExplainer(Module):
    def __init__(self, head_in=32, head_hidden=32, size_coef: float = 0.05, entr_coef: float = 1.0) -> None:

        super().__init__()

        self.size_coef = size_coef
        self.entr_coef = entr_coef

        self.target_loss = torch.nn.MSELoss()

        self.head = Sequential(
            Linear(head_in, head_hidden),
            ReLU(),
            BatchNorm1d(head_hidden),
            Linear(head_hidden, 1)
        )

    def forward(self, features):
        return self.head(features)

    def fit(self, epochs, model, dataloader):
        optimizer = Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            for mesh in dataloader:
                optimizer.zero_grad()

                z = model.feature_extractor(mesh, pool=False)
                z = self(z)
                mask = concrete_sample(z, temperature=2.0)
                data = apply_mask(mesh, mask)
                y_pred = model(data)

                loss = self._loss(y_pred, mesh.y, mask)
                loss.backward()
                optimizer.step()

                print(loss)

    def _loss(self, y_hat: torch.Tensor, y: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        loss = self.target_loss(y_hat, y)

        # Regularization loss:
        mask = node_mask.sigmoid()
        size_loss = mask.sum() * self.size_coef
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.entr_coef

        return loss + size_loss + mask_ent_loss


def evaluate(model, dataloader, loss_fn, split):
    loss_acc, n_graphs = 0, 0

    model.eval()
    with torch.no_grad():
        for mesh, _, y in dataloader:
            random_bit = random.random() > 0.5
            y_pred = model(mesh, None, mask=random_bit)
            loss = loss_fn(y, y_pred)
            loss_acc += loss.cpu().detach().item()
            n_graphs += mesh.num_graphs

    print(f"Avg {split} MSE: {loss_acc / n_graphs}")


def train_with_random() -> MeshGNN:
    # train on birth age with random masking:
    model = MeshGNN(mesh_conf=MeshConf(), head_conf=HeadConf(hidden=32))
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()

    train_dataloader = torch.load("birth_age_train_dataloader.pt")
    # val_dataloader = torch.load("birth_age_val_dataloader.pt")
    # test_dataloader = torch.load("birth_age_test_dataloader.pt")

    for epoch in range(200):
        loss_acc, n_graphs = 0, 0

        model.train()
        for mesh, _, y in train_dataloader:
            optimizer.zero_grad()

            random_bit = random.random() > 0.5
            y_pred = model(mesh, None, mask=random_bit)
            loss = loss_fn(y, y_pred)

            loss.backward()
            optimizer.step()

            loss_acc += loss.cpu().detach().item()
            n_graphs += mesh.num_graphs

        print(f"Avg Train MSE: {loss_acc / n_graphs}")

        # evaluate(model, val_dataloader, loss_fn, "Val")

    # evaluate(model, test_dataloader, loss_fn, "Test")
    torch.save(model, "model_random.pt")

    return model


def explain(model: MeshGNN):
    train_dataloader = torch.load("birth_age_train_dataloader.pt")
    test_dataloader = torch.load("birth_age_test_dataloader.pt")

    # freeze feature extractor
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    explainer = PGExplainer()
    explainer.fit(epochs=50, model=model, dataloader=test_dataloader)


def main():
    model = train_with_random()
    explain(model)


if __name__ == '__main__':
    main()
