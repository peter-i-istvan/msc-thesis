from torch_geometric.data import Data

# https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
class MyData(Data):
    """Subclassing because of the need to batch the adjacency matrices differently.
    Default batching with PyG DataLoader causes (B*87, 87) shaped adjacency matrix batches,
    whih causes a dimension error when applying DenseGraphConv:

        `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 87] but got: [1, 2784].`

    Overriding __cat_dim__ as below causes the 'adj' field to be batched as (B, 87, 87) instead.
    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        # TODO: x as well for connectome - respecify as connectome
        if key == 'adj':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)