from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_COORDS = Path("results/tables/normal_tumour_progression_axis.tsv")
OUT_UMAP = Path("results/figures/normal_tumour_progression_umap.png")
OUT_ATLAS = Path("results/figures/normal_to_tumour_progression_atlas.png")

FEATURE_COLUMNS = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_intensity", "std_intensity",
    "entropy", "edge_density",
    "nuclei_count", "nuclei_density",
    "nuclear_area_fraction",
    "mean_nucleus_area",
    "median_nucleus_area",
    "std_nucleus_area",
    "mean_nucleus_eccentricity",
    "mean_nucleus_solidity",
    "mean_nucleus_perimeter",
]

KEEP_LABELS = ["NORM", "TUM"]
N_BINS = 8
PATCHES_PER_BIN = 6


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")
    df = df[df["label"].isin(KEEP_LABELS)].copy()

    X = df[FEATURE_COLUMNS].copy()
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    norm_centroid = X_pca[df["label"] == "NORM"].mean(axis=0)
    tum_centroid = X_pca[df["label"] == "TUM"].mean(axis=0)

    dist_norm = np.linalg.norm(X_pca - norm_centroid, axis=1)
    dist_tum = np.linalg.norm(X_pca - tum_centroid, axis=1)

    progression_score = dist_norm / (dist_norm + dist_tum)

    df["dist_norm"] = dist_norm
    df["dist_tum"] = dist_tum
    df["progression_score"] = progression_score

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )

    umap_coords = reducer.fit_transform(X_scaled)

    df["UMAP1"] = umap_coords[:, 0]
    df["UMAP2"] = umap_coords[:, 1]

    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    OUT_UMAP.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_COORDS, sep="\t", index=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["UMAP1"],
        df["UMAP2"],
        c=df["progression_score"],
        s=12,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="Normal → Tumour progression score")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Normal–Tumour morphology progression axis")
    plt.tight_layout()
    plt.savefig(OUT_UMAP, dpi=300)
    plt.close()

    selected_rows = []

    bins = np.linspace(0, 1, N_BINS + 1)

    for i in range(N_BINS):
        low = bins[i]
        high = bins[i + 1]

        sub = df[
            (df["progression_score"] >= low)
            & (df["progression_score"] < high)
        ].copy()

        if sub.empty:
            continue

        centre = (low + high) / 2
        sub["bin_distance"] = np.abs(sub["progression_score"] - centre)

        selected = (
            sub.sort_values("bin_distance")
            .head(PATCHES_PER_BIN)
            .copy()
        )

        selected["progression_bin"] = i + 1
        selected_rows.append(selected)

    selected_df = pd.concat(selected_rows, axis=0)

    n_rows = N_BINS
    n_cols = PATCHES_PER_BIN

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 18)
    )

    for ax in axes.flatten():
        ax.axis("off")

    for _, row in selected_df.iterrows():
        r = int(row["progression_bin"]) - 1

        bin_df = selected_df[
            selected_df["progression_bin"] == row["progression_bin"]
        ].reset_index(drop=True)

    for r in range(N_BINS):
        bin_subset = selected_df[
            selected_df["progression_bin"] == r + 1
        ].reset_index(drop=True)

        for c, (_, row) in enumerate(bin_subset.iterrows()):
            if c >= n_cols:
                continue

            ax = axes[r, c]
            img = Image.open(row["path"]).convert("RGB")
            ax.imshow(img)
            ax.set_title(
                f'{row["label"]}\nscore={row["progression_score"]:.2f}',
                fontsize=8
            )
            ax.axis("off")

        axes[r, 0].set_ylabel(
            f"Bin {r+1}",
            fontsize=10
        )

    plt.suptitle(
        "Normal → Tumour morphology progression atlas",
        fontsize=18
    )
    plt.tight_layout()
    plt.savefig(OUT_ATLAS, dpi=300)
    plt.close()

    print(f"Saved: {OUT_COORDS}")
    print(f"Saved: {OUT_UMAP}")
    print(f"Saved: {OUT_ATLAS}")
    print("\nProgression score by label:")
    print(df.groupby("label")["progression_score"].describe())


if __name__ == "__main__":
    main()
