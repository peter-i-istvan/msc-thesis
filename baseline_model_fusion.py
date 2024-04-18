import torch
import torch.nn as nn
import torch_geometric.nn as gnn


# Dani baseline
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.fc1 = gnn.GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc3 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc4 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.fc5 = nn.Linear(hidden_channels, hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.fc6 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, data, selection=None):
        x = data.pos
        if data.norm is not None:
            x = torch.cat([x, data.norm], dim=1)
        # if data.dha is not None:
        #     x = torch.cat([x, data.dha], dim=1)
        if data.x is not None:
            x = torch.cat([x, data.x], dim=1)
        if selection is not None:
            x = (x * selection).float()
        x = self.bn0(x)
        x = self.bn1(self.act(self.fc1(x, data.edge_index)))
        x = self.bn2(self.act(self.fc2(x, data.edge_index)))
        x = self.bn3(self.act(self.fc3(x, data.edge_index)))
        x = self.bn4(self.act(self.fc4(x, data.edge_index)))
        x = gnn.global_mean_pool(x, data.batch)
        # x = torch.cat([x, data.confound], dim=1)
        x = self.bn5(self.act(self.fc5(x)))
        x = self.fc6(x)
        return x


class MeshGNN(nn.Module):
    """Returns features after mean pool."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_channels)
        self.fc1 = gnn.GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc3 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc4 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.act = nn.ReLU()

    def forward(self, data):
        # assemble inputs based on available features
        x = data.pos
        if data.norm is not None:
            x = torch.cat([x, data.norm], dim=1)
        # if data.dha is not None:
        #     x = torch.cat([x, data.dha], dim=1)
        if data.x is not None:
            x = torch.cat([x, data.x], dim=1)
        
        x = self.bn0(x)
        x = self.bn1(self.act(self.fc1(x, data.edge_index)))
        x = self.bn2(self.act(self.fc2(x, data.edge_index)))
        x = self.bn3(self.act(self.fc3(x, data.edge_index)))
        x = self.bn4(self.act(self.fc4(x, data.edge_index)))
        x = gnn.global_mean_pool(x, data.batch)

        return x


class ConnectomeGNN(nn.Module):
    """Returns features after flatten."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = gnn.dense.DenseGraphConv(in_channels, hidden_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = gnn.dense.DenseGraphConv(hidden_channels, hidden_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = gnn.dense.DenseGraphConv(hidden_channels, hidden_channels)
        self.prelu3 = nn.PReLU()
        # Try mean pool for as-expected output shape
        self.flatten = nn.Flatten()

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.prelu1(self.conv1(x, adj))
        x = self.prelu2(self.conv2(x, adj))
        x = self.prelu3(self.conv3(x, adj))
        x = self.flatten(x)

        return x


class BaselineFusionGNN(nn.Module):
    """Returns the single feature to be predicted."""

    def __init__(self, mesh_in_channels, connecome_in_channels, hidden_channels):
        super().__init__()
        self.mesh_gnn = MeshGNN(mesh_in_channels, hidden_channels)
        self.connectome_gnn = ConnectomeGNN(connecome_in_channels, hidden_channels)
        feature_length = hidden_channels + 87*hidden_channels
        self.linear1 = nn.Linear(feature_length, hidden_channels)
        self.prelu1 = nn.PReLU()
        self.linear2 = nn.Linear(hidden_channels, 1)

    def forward(self, mesh_data, connectome_data):
        mesh_features = self.mesh_gnn(mesh_data)
        connectome_features = self.connectome_gnn(connectome_data)
        x = torch.cat([mesh_features, connectome_features], dim=1)
        x = self.prelu1(self.linear1(x))
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    model = BaselineFusionGNN(13, 87, 32)
    print(model)