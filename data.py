import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from custom_data import MyData


PREPROCESSED_FEATURES_ROOT = "/run/media/i/ADATA HV620S/dHCP"
ANAT_PIPELINE_ROOT = "/run/media/i/ADATA HV620S/rel3_dhcp_anat_pipeline"
CONNECTOMES_ROOT = "connectomes-csv"
SPLIT_DF_ROOT = "."


def get_available_data(split_df):
    """Returns the DataFrame containing all paths and their availability."""

    sub_ses = split_df[0].str.split("_", expand=True)
    split_df["sub"] = sub_ses[0].str.split("-", expand=True)[1]
    split_df["ses"] = sub_ses[1].str.split("-", expand=True)[1].astype(int)
    
    # TODO: integrate right hemishpere as well
    split_df["surface_path"] = split_df.apply(
        lambda df: f"{PREPROCESSED_FEATURES_ROOT}/surfaces/sub-{df['sub']}_ses-{df['ses']}_left.wm.surf.gii",
        axis=1
    )
    split_df["features_path"] = split_df.apply(
        lambda df: f"{PREPROCESSED_FEATURES_ROOT}/features/sub-{df['sub']}_ses-{df['ses']}_left.shape.gii",
        axis=1
    )
    split_df["labels_path"] = split_df.apply(
        lambda df: f"{ANAT_PIPELINE_ROOT}/sub-{df['sub']}/ses-{df['ses']}/anat/sub-{df['sub']}_ses-{df['ses']}_hemi-left_desc-drawem_dseg.label.gii",
        axis=1
    )
    split_df["connectome_path"] = split_df.apply(
        lambda df: f"{CONNECTOMES_ROOT}/sub-{df['sub']}-ses-{df['ses']}-nws.csv",
        axis=1
    )

    # check the availability of each file
    path_cols = ["surface_path", "features_path", "labels_path", "connectome_path"]
    for col in path_cols:
        split_df[f"has_{col}"] = split_df[col].apply(lambda s: os.path.isfile(s))

    split_df["has_all_files"] = split_df["has_surface_path"] & split_df["has_features_path"] & split_df["has_labels_path"] & split_df["has_connectome_path"]
    
    split_df.drop(columns=[0])
    split_df = split_df.loc[split_df["has_all_files"], ["sub", "ses"] + path_cols]

    return split_df.reset_index(drop=True) 


def set_up_dfs(task: str):
    """Saves the DF containing file paths associated with the task, and the 3 splits."""

    for split in ["train", "val", "test"]:
        ids = pd.read_csv(os.path.join(SPLIT_DF_ROOT, task + f"_{split}.txt"), header=None)
        df = get_available_data(ids)
        df.to_csv(f"{task}_{split}_files.tsv", sep="\t")


def get_connectome_data(w: np.array) -> MyData:
    transform = T.AddLaplacianEigenvectorPE(10)
    
    # create fully connected
    x = torch.eye(87, dtype=torch.float32)
    e = [[i, j] for i in range(87) for j in range(87)]
    e = torch.tensor(e, dtype=torch.long).t().contiguous()
    
    # build data object with edge index, weights
    data = MyData(x=x, edge_index=e)
    data.adj = torch.from_numpy(w).to(torch.float32)

    data = transform(data)
    data.x = data.laplacian_eigenvector_pe

    return data


def get_mesh_data(pos: np.array, face: np.array, features: np.array) -> MyData:
    transform = T.Compose([T.NormalizeScale(), T.GenerateMeshNormals(), T.FaceToEdge()])
    
    # set up mesh properties
    x = torch.from_numpy(features).to(torch.float32)
    pos = torch.from_numpy(pos).to(torch.float32)
    face = torch.from_numpy(face.T).to(torch.long).contiguous()


    # build data object
    data = MyData()
    data.x = x
    data.pos = pos
    data.face = face

    return transform(data)


def save_dataloader(task: str, split: str):
    """Reads the '{task}_{split}_files.tsv' DF for file paths.
    Joins with 'combined.tsv' for task-related target variables
    Reads the files themselves and creates a [torch_geometric.loader.]DataLoader.
    Saves the DataLoader to '{task}_{split}_dataloader.pt'."""
    
    df = pd.read_csv(f"{task}_{split}_files.tsv", sep="\t")
    dataset = []

    combined_df = pd.read_csv("combined.tsv", sep="\t", usecols=["participant_id", "session_id", task])
    df = df.merge(
        combined_df, how="left", left_on=["sub", "ses"], right_on=["participant_id", "session_id"]
    )

    print(f"Loading {split} files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # MESH
        pos, face = nib.load(row["surface_path"]).agg_data()
        features = np.stack(nib.load(row["features_path"]).agg_data(), axis=1)
        mesh = get_mesh_data(pos, face, features)
        # CONNECTOME
        connectome = pd.read_csv(row["connectome_path"], header=None).to_numpy()
        connectome = get_connectome_data(connectome)
        # TODO: LABELS
        # labels = nib.load(row["labels_path"]).agg_data()

        y = row[task]

        dataset.append((mesh, connectome, y))

    dataloader = DataLoader(dataset, batch_size=32)
    torch.save(dataloader, f"{task}_{split}_dataloader.pt")


if __name__ == "__main__":
    
    task = "scan_age"
    # Run only once - no need to run after {task}_{split(s)}_files.tsv were created:
    # set_up_dfs(task)

    split = "train"
    # Run only once - no need to run after {task}_{split}_dataloader.pt was created:
    # save_dataloader(task, split)
    
    dataloader = torch.load(f"{task}_{split}_dataloader.pt")
    for mesh, connectome, y in dataloader:
        print(mesh)
        print(connectome)
        print(y)
        break
