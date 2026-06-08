import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(
    "results/tables/crc_val_7k_integrated_features.tsv",
    sep="\t"
)

df = df[df["label"].isin(["TUM", "STR"])].copy()

meta_cols = ["image_id", "filename", "label", "path"]

feature_cols = [
    c for c in df.columns
    if c not in meta_cols
]

X = df[feature_cols]

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=10)
X = pca.fit_transform(X)

tum_mask = df["label"] == "TUM"
str_mask = df["label"] == "STR"

tum_centroid = X[tum_mask].mean(axis=0)
str_centroid = X[str_mask].mean(axis=0)

dist_tum = np.linalg.norm(X - tum_centroid, axis=1)
dist_str = np.linalg.norm(X - str_centroid, axis=1)

interface_score = np.abs(dist_tum - dist_str)

df["interface_score"] = interface_score
df["dist_tum"] = dist_tum
df["dist_str"] = dist_str

interface = df.sort_values(
    "interface_score",
    ascending=True
)

interface.head(200).to_csv(
    "results/tables/tumour_stroma_interface_patches.tsv",
    sep="\t",
    index=False
)

print(interface[
    ["label",
     "interface_score",
     "dist_tum",
     "dist_str"]
].head())

print(
    "\nSaved:",
    "results/tables/tumour_stroma_interface_patches.tsv"
)
