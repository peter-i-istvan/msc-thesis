import torch
from dataclasses import dataclass
from torch_geometric.data import Data
from torch_geometric.nn.dense import DenseGraphConv
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Module, BatchNorm1d, ReLU, PReLU, Flatten, Linear, Sequential


def concrete_mask(logits: torch.Tensor, temperature: float = 1.0, bias=0.01) -> torch.Tensor:
    """Samples random tensor with 'logits' shape."""
    eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
    return (eps.log() - (1 - eps).log() + logits) / temperature


def apply_mask(data: Data, mask: torch.Tensor) -> Data:
    """Applies binary 'mask' to nodes of 'data' and removes edges where either node was masked."""
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = (x * mask).float()
    mask = mask.squeeze().bool()
    edge_mask = (mask[edge_index[0]]) & (mask[edge_index[1]])
    edge_index = edge_index[:, edge_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch)


def random_mask(data, return_probs=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    probs = torch.rand((data.num_nodes, 1), device=data.x.device)
    mask = (probs > 0.5).detach().float()
    if return_probs:
        return mask, probs
    return mask


@dataclass(frozen=True)
class ConnectomeConf:
    NODES: int = 87 # assumed to be fixed
    in_channel: int = 10
    conv_channel: int = 20
    conv_out: int = 5
    agg: str = "flatten" # "flatten" | "mean"

    @property
    def out(self):
        if self.agg == "flatten":
            return self.NODES * self.conv_out
        else: # add
            return self.conv_out


@dataclass(frozen=True)
class MeshConf:
    in_channel: int = 13
    hidden: int = 32
    
    @property
    def out(self):
        return self.hidden


@dataclass(frozen=True)
class HeadConf:
    in_agg: int = "cat" # "cat" | "add" | None -> only relevant for FusionGNN
    hidden: int = 10 # conn: 10, mesh: 32


class MeshFeatureExtractor(Module):
    """Returns features after mean pool."""

    def __init__(self, conf: MeshConf):
        super().__init__()
        self.bn0 = BatchNorm1d(conf.in_channel)
        self.fc1 = GCNConv(conf.in_channel, conf.hidden)
        self.bn1 = BatchNorm1d(conf.hidden)
        self.fc2 = GCNConv(conf.hidden, conf.hidden)
        self.bn2 = BatchNorm1d(conf.hidden)
        self.fc3 = GCNConv(conf.hidden, conf.hidden)
        self.bn3 = BatchNorm1d(conf.hidden)
        self.fc4 = GCNConv(conf.hidden, conf.hidden)
        self.bn4 = BatchNorm1d(conf.hidden)
        self.act = ReLU()

    def forward(self, mesh, edge_weights=None, pool=True):
        # assemble inputs based on available features
        x = mesh.pos
        if mesh.norm is not None:
            x = torch.cat([x, mesh.norm], dim=1)
        if mesh.dha is not None:
            x = torch.cat([x, mesh.dha], dim=1)
        if mesh.x is not None:
            x = torch.cat([x, mesh.x], dim=1)
        
        # run feature extractor
        x = self.bn0(x)
        x = self.bn1(self.act(self.fc1(x, mesh.edge_index, edge_weight=edge_weights)))
        x = self.bn2(self.act(self.fc2(x, mesh.edge_index, edge_weight=edge_weights)))
        x = self.bn3(self.act(self.fc3(x, mesh.edge_index, edge_weight=edge_weights)))
        x = self.bn4(self.act(self.fc4(x, mesh.edge_index, edge_weight=edge_weights)))

        if pool:
            x = global_mean_pool(x, mesh.batch)

        return x
    

class ConnectomeFeatureExtractor(Module):
    """Returns features after aggregation (sum or concat)."""
    def __init__(self, conf: ConnectomeConf):
        super().__init__()
        self.conv1 = DenseGraphConv(conf.in_channel, conf.conv_channel)
        self.prelu1 = PReLU()
        self.conv2 = DenseGraphConv(conf.conv_channel, conf.conv_channel)
        self.prelu2 = PReLU()
        self.conv3 = DenseGraphConv(conf.conv_channel, conf.conv_out)
        self.prelu3 = PReLU()
        
        if conf.agg == "flatten":
            self.flatten = Flatten()
        else: # mean
            self.flatten = None

    def forward(self, connectome):
        x, adj = connectome.x, connectome.adj
        x = self.prelu1(self.conv1(x, adj))
        x = self.prelu2(self.conv2(x, adj))
        x = self.prelu3(self.conv3(x, adj))
        # final aggregation
        if self.flatten is not None:
            x = self.flatten(x)
        else:
            x = torch.mean(x, dim=-2)

        return x
    

class MeshGNN(Module):
    def __init__(self, mesh_conf: MeshConf, head_conf: HeadConf) -> None:
        super().__init__()
        self.feature_extractor = MeshFeatureExtractor(mesh_conf)

        self.head = Sequential(
            Linear(mesh_conf.out, head_conf.hidden),
            ReLU(),
            BatchNorm1d(head_conf.hidden),
            Linear(head_conf.hidden, 1)
        )

    def forward(self, mesh, connectome, edge_weights=None, mask=False):
        """Ignore connectome, but keep it for uniform interface."""
        if mask:
            mask = random_mask(mesh)
            mesh = apply_mask(mesh, mask)
        features = self.feature_extractor(mesh, edge_weights)
        return self.head(features)


class ConnectomeGNN(Module):
    def __init__(self, connectome_conf: ConnectomeConf, head_conf: HeadConf) -> None:
        super().__init__()
        self.feature_extractor = ConnectomeFeatureExtractor(connectome_conf)

        self.head = Sequential(
            Linear(connectome_conf.out, head_conf.hidden),
            ReLU(),
            BatchNorm1d(head_conf.hidden),
            Linear(head_conf.hidden, 1)
        )

    def forward(self, mesh, connectome):
        """Ignore connectome, but keep it for uniform interface."""
        features = self.feature_extractor(connectome)
        return self.head(features)
    

class FusionGNN(Module):
    def __init__(
        self,
        connectome_conf: ConnectomeConf,
        mesh_conf: MeshConf,
        head_conf: HeadConf
    ) -> None:
        
        super().__init__()

        self.mesh = MeshFeatureExtractor(mesh_conf)
        self.connectome = ConnectomeFeatureExtractor(connectome_conf)

        # Handle aggregation
        assert head_conf.in_agg in {"add", "cat"}
        self.agg = head_conf.in_agg

        if head_conf.in_agg == "add":
            msg = "Mesh and Connectome feature outputs must match if we aggregate by addition"
            assert mesh_conf.out == connectome_conf.out, msg
            head_in = mesh_conf.out
        else:
            head_in = mesh_conf.out + connectome_conf.out

        self.head = Sequential(
            Linear(head_in, head_conf.hidden),
            ReLU(),
            BatchNorm1d(head_conf.hidden),
            Linear(head_conf.hidden, 1)
        )

    def forward(self, mesh, connectome):
        connectome_features = self.connectome(connectome)
        mesh_features = self.mesh(mesh)
        
        if self.agg == "add":
            features = connectome_features + mesh_features
        else:
            features = torch.cat([mesh_features, connectome_features], dim=1)

        return self.head(features)