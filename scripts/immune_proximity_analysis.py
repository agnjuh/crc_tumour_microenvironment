from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_TABLE = Path("results/tables/tumour_immune_proximity_scores.tsv")
OUT_UMAP = Path("results/figures/tumour_immune_proximity_umap.png")
OUT_MONTAGE = Path("results/figures/tumour_immune_proximity_montage.png")

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

KEEP_LABELS = ["TUM", "LYM", "STR"]
N_MONTAGE = 48
N_COLS = 8


def make_montage(df, out_file):
    selected = pd.concat([
        df.sort_values("immune_proximity_score", ascending=False).head(24),
        df.sort_values("immune_proximity_score", ascending=True).head(24),
    ])

    n_rows = int(np.ceil(len(selected) / N_COLS))

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(14, 9))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, selected.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f'{row["label"]}\nimmune={row["immune_proximity_score"]:.2f}',
            fontsize=7,
        )
        ax.axis("off")

    plt.suptitle(
        "Tumour patches ranked by immune-proximity score\nTop: immune-like tumour regions; bottom: immune-distant tumour regions",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")
    df = df[df["label"].isin(KEEP_LABELS)].copy()

    X = df[FEATURE_COLUMNS]
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    lym_centroid = X_pca[df["label"] == "LYM"].mean(axis=0)
    tum_centroid = X_pca[df["label"] == "TUM"].mean(axis=0)
    str_centroid = X_pca[df["label"] == "STR"].mean(axis=0)

    dist_lym = np.linalg.norm(X_pca - lym_centroid, axis=1)
    dist_tum = np.linalg.norm(X_pca - tum_centroid, axis=1)
    dist_str = np.linalg.norm(X_pca - str_centroid, axis=1)

    df["dist_to_lym"] = dist_lym
    df["dist_to_tum"] = dist_tum
    df["dist_to_str"] = dist_str

    df["immune_proximity_score"] = 1 - (
        df["dist_to_lym"] / (df["dist_to_lym"] + df["dist_to_tum"])
    )

    df["stromal_proximity_score"] = 1 - (
        df["dist_to_str"] / (df["dist_to_str"] + df["dist_to_tum"])
    )

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )

    coords = reducer.fit_transform(X_scaled)
    df["UMAP1"] = coords[:, 0]
    df["UMAP2"] = coords[:, 1]

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_UMAP.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_TABLE, sep="\t", index=False)

    tum = df[df["label"] == "TUM"].copy()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        tum["UMAP1"],
        tum["UMAP2"],
        c=tum["immune_proximity_score"],
        s=14,
        alpha=0.85,
    )
    plt.colorbar(sc, label="Immune-proximity score")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Tumour morphology space coloured by immune-proximity score")
    plt.tight_layout()
    plt.savefig(OUT_UMAP, dpi=300)
    plt.close()

    make_montage(tum, OUT_MONTAGE)

    print("Saved:")
    print(OUT_TABLE)
    print(OUT_UMAP)
    print(OUT_MONTAGE)

    print("\nTumour immune-proximity score:")
    print(tum["immune_proximity_score"].describe())

    print("\nTop immune-proximal tumour patches:")
    print(
        tum.sort_values("immune_proximity_score", ascending=False)[
            ["image_id", "label", "immune_proximity_score", "dist_to_lym", "dist_to_tum"]
        ].head(10)
    )


if __name__ == "__main__":
    main()
