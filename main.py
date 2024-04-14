import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch_geometric.data import Data
import torch_geometric.transforms as T

from baseline_model_fusion import BaselineFusionGNN


sub, ses = "CC00818XX18", 4020


def get_connectome_data(w: np.array) -> Data:
    transform = T.AddLaplacianEigenvectorPE(10)
    
    # create fully connected
    x = torch.eye(87, dtype=torch.float32)
    e = [[i, j] for i in range(87) for j in range(87)]
    e = torch.tensor(e, dtype=torch.long).t().contiguous()
    
    # build data object with edge index, weights
    data = Data(x=x, edge_index=e)
    data.adj = torch.from_numpy(w).to(torch.float32)

    data = transform(data)
    data.x = data.laplacian_eigenvector_pe

    return data


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


def merge_data(connectome: Data, mesh: Data) -> Data:
    labels = nib.load(f"mesh/sub-{sub}_ses-{ses}_hemi-left_desc-drawem_dseg.label.gii").agg_data()

    n_mesh = mesh.x.shape[0]
    
    x = torch.cat([
        torch.cat([mesh.pos, mesh.x, mesh.norm], dim=1),
        connectome.x + n_mesh # connectome node ids get shifted with n_mesh
    ])
    
    # edges that connect the connectome with the mesh are called "inter-edges"
    valid_region_id_mask = torch.from_numpy(labels != 0)
    mesh_node = torch.where(valid_region_id_mask)[0]
    connectome_node = torch.from_numpy(labels[valid_region_id_mask])
    
    # connectome node ids get shifted with n_mesh
    connectome_node += n_mesh
    
    # adding edges both ways
    new_edges_index = torch.stack([
        torch.cat([mesh_node, connectome_node]),
        torch.cat([connectome_node, mesh_node])
    ])

    merged_data = Data(
        x = x,
        edge_index = torch.cat([mesh.edge_index, connectome.edge_index + n_mesh, new_edges_index], dim=1) # connectome node ids get shifted with n_mesh
    )
    
    return merged_data


def main():
    # access subject and metadata
    meta = pd.read_csv("combined.tsv", sep="\t")
    mask = (meta["participant_id"]==sub) & (meta["session_id"]==ses)
    age = meta.loc[mask, "scan_age"].item()
    
    # CONNECTOME
    connectome = pd.read_csv(f"connectomes-csv/sub-{sub}-ses-{ses}-nws.csv", header=None).to_numpy()
    connectome = get_connectome_data(connectome)

    # MESH
    pos, face = nib.load(f"mesh/sub-{sub}_ses-{ses}_left.wm.surf.gii").agg_data()
    features = np.stack(nib.load(f"mesh/sub-{sub}_ses-{ses}_left.shape.gii").agg_data(), axis=1)
    mesh = get_mesh_data(pos, face, features)

    merged_data = merge_data(connectome, mesh)
    print(merged_data)
    
    # MODEL
    model = BaselineFusionGNN(10, 10, 32)
    
    with torch.no_grad():
        print(model(mesh, connectome))


if __name__ == "__main__":
    main()
