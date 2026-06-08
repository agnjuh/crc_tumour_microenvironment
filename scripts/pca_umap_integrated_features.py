from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_PCA = Path("results/figures/crc_val_7k_pca_integrated_features.png")
OUT_UMAP = Path("results/figures/crc_val_7k_umap_integrated_features.png")
OUT_COORDS = Path("results/tables/crc_val_7k_integrated_pca_umap_coordinates.tsv")

FEATURE_COLUMNS = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_intensity",
    "std_intensity",
    "entropy",
    "edge_density",
    "nuclei_count",
    "nuclei_density",
    "nuclear_area_fraction",
    "mean_nucleus_area",
    "median_nucleus_area",
    "std_nucleus_area",
    "mean_nucleus_eccentricity",
    "mean_nucleus_solidity",
    "mean_nucleus_perimeter",
]

def plot_embedding(df, x, y, out_file, title):
    plt.figure(figsize=(9, 7))

    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        plt.scatter(
            sub[x],
            sub[y],
            s=8,
            alpha=0.7,
            label=label,
        )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def main():
    OUT_PCA.parent.mkdir(parents=True, exist_ok=True)
    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_FILE, sep="\t")

    X = df[FEATURE_COLUMNS].copy()
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )

    umap_coords = reducer.fit_transform(X_scaled)

    df["PC1"] = pca_coords[:, 0]
    df["PC2"] = pca_coords[:, 1]
    df["UMAP1"] = umap_coords[:, 0]
    df["UMAP2"] = umap_coords[:, 1]

    df.to_csv(OUT_COORDS, sep="\t", index=False)

    plot_embedding(
        df,
        "PC1",
        "PC2",
        OUT_PCA,
        "CRC-VAL-HE-7K PCA: integrated morphology and nuclear features",
    )

    plot_embedding(
        df,
        "UMAP1",
        "UMAP2",
        OUT_UMAP,
        "CRC-VAL-HE-7K histology map: integrated morphology and nuclear features",
    )

    print("Saved:")
    print(OUT_COORDS)
    print(OUT_PCA)
    print(OUT_UMAP)
    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
