from torch_geometric.data import Data

# https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
# Apparently, adjacency matrix ('adj') batching only works as expected (diagonal concat into a large matrix)
# if the provided 'adj' matrix is sparse (see: SITE_PACKAGES/torch_geometric/data:646, PyG version 2.5.2)
# I managed to circumvent this by overriding __cat_dim__, so both 'x' and 'adj' recieve a batch dimension,
# rather than naively being concatenated ('x' batch will look like (B,87,F) rather than (B*87, F), 'adj': (B,87,87) rather than (B*87, 87))
# (for reference, "ideal" batch 'adj' would look like (B*87,B*87) as a sparse matrix, buth the above workaround is also adequate)
# In this case, 'num_nodes MUST be set after instantiation, because overriding for key 'x' causes 'num_nodes' to be None
# and AddLaplacianEigenvectorPE transform failing (among others).
class ConnectomeData(Data):
    """Subclassing because of the need to batch the adjacency matrices differently.
    Default batching with PyG DataLoader causes (B*87, 87) shaped adjacency matrix batches,
    whih causes a dimension error when applying DenseGraphConv:

        `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 87] but got: [1, 2784].`

    Overriding __cat_dim__ as below causes the 'adj' field to be batched as (B, 87, 87) instead.
    Furthermore, in the case of the connectomes, 'x' should be similarly handled to keep multiplication dimensions in sync.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # TODO: x as well for connectome - respecify as connectome
        if key in {'adj', 'x'}:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)