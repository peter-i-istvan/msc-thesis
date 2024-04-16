import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

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


def set_up_dfs():
    task = "scan_age"

    for split in ["train", "val", "test"]:
        ids = pd.read_csv(os.path.join(SPLIT_DF_ROOT, task + f"_{split}.txt"), header=None)
        df = get_available_data(ids)
        df.to_csv(f"{task}_{split}_files.tsv", sep="\t")


if __name__ == "__main__":
    # Run only once:
    # set_up_dfs()

    task = "scan_age"
    split = "train"

    df = pd.read_csv(f"{task}_{split}_files.tsv", sep="\t")

    print(f"Loading {split} files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pos, face = nib.load(row["surface_path"]).agg_data()
        features = np.stack(nib.load(row["features_path"]).agg_data(), axis=1)
        connectome = pd.read_csv(row["connectome_path"], header=None).to_numpy()
        labels = nib.load(row["labels_path"]).agg_data()
