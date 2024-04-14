import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch_geometric.data import Data
import torch_geometric.transforms as T

from baseline_model_fusion import BaselineFusionGNN


def get_connectome_data(w: np.array) -> Data:
    transform = T.AddLaplacianEigenvectorPE(10)
    
    # create fully connected
    x = torch.eye(87, dtype=torch.float32)
    e = [[i, j] for i in range(87) for j in range(87)]
    e = torch.tensor(e, dtype=torch.long).t().contiguous()
    
    # build data object with edge index, weights
    data = Data(x=x, edge_index=e)
    data.adj = torch.from_numpy(w).to(torch.float32)

    return transform(data)


def get_mesh_data(pos: np.array, face: np.array, features: np.array) -> Data:
    transform = T.Compose([T.NormalizeScale(), T.GenerateMeshNormals(), T.FaceToEdge()])
    
    # set up mesh properties
    x = torch.from_numpy(features).to(torch.float32)
    pos = torch.from_numpy(pos).to(torch.float32)
    face = torch.from_numpy(face.T).to(torch.long).contiguous()


    # build data object
    data = Data()
    data.x = x
    data.pos = pos
    data.face = face
    
    return transform(data)


if __name__ == "__main__":
    # access subject and metadata
    meta = pd.read_csv("combined.tsv", sep="\t")
    sub, ses = "CC00818XX18", 4020
    mask = (meta["participant_id"]==sub) & (meta["session_id"]==ses)
    age = meta.loc[mask, "scan_age"].item()
    
    # CONNECTOME
    connectome = pd.read_csv(f"connectomes-csv/sub-{sub}-ses-{ses}-nws.csv", header=None).to_numpy()
    connectome = get_connectome_data(connectome)

    # MESH
    pos, face = nib.load(f"mesh/sub-{sub}_ses-{ses}_left.wm.surf.gii").agg_data()
    features = np.stack(nib.load(f"mesh/sub-{sub}_ses-{ses}_left.shape.gii").agg_data(), axis=1)
    mesh = get_mesh_data(pos, face, features)
    
    # MODEL
    model = BaselineFusionGNN(10, 87, 32)
    
    with torch.no_grad():
        print(model(mesh, connectome))
