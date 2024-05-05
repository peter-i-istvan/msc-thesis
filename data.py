import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from custom_data import ConnectomeData


PREPROCESSED_FEATURES_ROOT = "/run/media/i/ADATA HV620S/dHCP"
ANAT_PIPELINE_ROOT = "/run/media/i/ADATA HV620S/rel3_dhcp_anat_pipeline"
CONNECTOMES_ROOT = "connectomes-csv"
SPLIT_DF_ROOT = "."
BATCH_SIZE = 16

# the script will create the dataloaders and additional related files under this subdirectory
DATA_ROOT = "data"


def get_available_data(split_df: pd.DataFrame) -> pd.DataFrame:
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
    split_df["dha_path"] = split_df.apply(
        lambda df: f"{PREPROCESSED_FEATURES_ROOT}/preprocess/V_dihedral_angles/sub-{df['sub']}_ses-{df['ses']}_left.wm.surf_V_dihedralAngles.npy",
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
    path_cols = ["surface_path", "features_path", "labels_path", "connectome_path", "dha_path"]
    split_df["has_all_files"] = True

    for col in path_cols:
        split_df[f"has_{col}"] = split_df[col].apply(lambda s: os.path.isfile(s))
        split_df["has_all_files"] &= split_df[f"has_{col}"]
   
    split_df.drop(columns=[0])
    split_df = split_df.loc[split_df["has_all_files"], ["sub", "ses"] + path_cols]

    return split_df.reset_index(drop=True) 


def set_up_dfs(task: str):
    """Saves the DF containing file paths associated with the task, and the 3 splits.
    Creates the data directory and its subdirectories if not already present.
    If the directory is present, it is likely its contents will be overwritten."""

    for split in ["train", "val", "test"]:
        ids = pd.read_csv(os.path.join(SPLIT_DF_ROOT, task + f"_{split}.txt"), header=None)
        df = get_available_data(ids)
        # creates the intermediate directories if it does not exist yet
        # it will overwrite the files
        task_split_data_root = os.path.join(DATA_ROOT, task, split)
        os.makedirs(task_split_data_root, exist_ok=True)

        tsv_root = os.path.join(task_split_data_root, f"{task}_{split}_files.tsv")
        df.to_csv(tsv_root, sep="\t")


def get_connectome_data(w: np.array) -> ConnectomeData:
    transform = T.AddLaplacianEigenvectorPE(10)
    
    # create fully connected
    x = torch.eye(87, dtype=torch.float32)
    e = [[i, j] for i in range(87) for j in range(87)]
    e = torch.tensor(e, dtype=torch.long).t().contiguous()
    
    # build data object with edge index, weights
    data = ConnectomeData(x=x, edge_index=e)
    data.adj = torch.from_numpy(w).to(torch.float32)
    # ! 'num_nodes' needs to be set explicitly
    # My custom ConnectomeData's __cat_dim__, if returns None for key 'x'
    # results in None for data.num_nodes
    data.num_nodes = 87

    data = transform(data)
    data.x = data.laplacian_eigenvector_pe

    return data


def get_mesh_data(pos: np.array, face: np.array, features: np.array, dha: np.array) -> Data:
    transform = T.Compose([T.NormalizeScale(), T.GenerateMeshNormals(), T.FaceToEdge()])
    
    # set up mesh properties
    x = torch.from_numpy(features).to(torch.float32)
    pos = torch.from_numpy(pos).to(torch.float32)
    face = torch.from_numpy(face.T).to(torch.long).contiguous()
    dha = torch.from_numpy(dha).to(torch.float32)


    # build data object
    data = Data()
    data.x = x
    data.pos = pos
    data.face = face
    data.dha = dha

    return transform(data)


def save_dataloader(task: str, split: str):
    """Reads the '{task}_{split}_files.tsv' (under DATA_ROOT/task/split/) DF for file paths.
    Joins with 'combined.tsv' for task-related target variables
    Reads the files themselves and creates a [torch_geometric.loader.]DataLoader.
    Saves the DataLoader to '{task}_{split}_dataloader.pt' (under DATA_ROOT/task/split/)."""
    
    df = pd.read_csv(os.path.join(DATA_ROOT, task, split, f"{task}_{split}_files.tsv"), sep="\t")
    dataset = []

    combined_df = pd.read_csv("combined.tsv", sep="\t", usecols=["participant_id", "session_id", task])
    df = df.merge(
        combined_df, how="left", left_on=["sub", "ses"], right_on=["participant_id", "session_id"]
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{task}-{split}] Loading files..."):
        # MESH
        pos, face = nib.load(row["surface_path"]).agg_data()
        features = np.stack(nib.load(row["features_path"]).agg_data(), axis=1)
        dha = np.load(row["dha_path"])
        mesh = get_mesh_data(pos, face, features, dha)
        # CONNECTOME
        connectome = pd.read_csv(row["connectome_path"], header=None).to_numpy()
        connectome = get_connectome_data(connectome)
        # TODO: LABELS
        # labels = nib.load(row["labels_path"]).agg_data()

        y = row[task]

        dataset.append((mesh, connectome, y))

    if split == "train":
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    torch.save(dataloader, os.path.join(DATA_ROOT, task, split, f"{task}_{split}_dataloader.pt"))


if __name__ == "__main__":
    # for task in ["scan_age", "birth_age"]:
        # Run only once - no need to run after {task}_{split(s)}_files.tsv were created:
        # set_up_dfs(task)

        # Run only once - no need to run after {task}_{split}_dataloader.pt was created:
        # for split in ["train", "val", "test"]:
        #     save_dataloader(task, split)
        
    # Try out a random dataloader for sanity check
    dataloader = torch.load(os.path.join(DATA_ROOT, "scan_age", "train", f"scan_age_train_dataloader.pt"))

    print("Dataloader sanity check...")
    for mesh, connectome, y in dataloader:
        print(mesh)
        print(connectome)
        print(y)
        break
