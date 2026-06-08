from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_TABLE = Path("results/tables/tumour_interface_feature_summary.tsv")
OUT_FIG = Path("results/figures/tumour_interface_feature_comparison.png")
OUT_SELECTED = Path("results/tables/tumour_core_vs_interface_selected_patches.tsv")

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

PLOT_FEATURES = [
    "nuclei_count",
    "mean_nucleus_area",
    "nuclear_area_fraction",
    "entropy",
    "mean_intensity",
]

N_PER_GROUP = 200


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")

    ts = df[df["label"].isin(["TUM", "STR"])].copy()

    X = ts[FEATURE_COLUMNS]
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    tum_mask = ts["label"] == "TUM"
    str_mask = ts["label"] == "STR"

    tum_centroid = X_pca[tum_mask].mean(axis=0)
    str_centroid = X_pca[str_mask].mean(axis=0)

    dist_to_tum = np.linalg.norm(X_pca - tum_centroid, axis=1)
    dist_to_str = np.linalg.norm(X_pca - str_centroid, axis=1)

    ts["dist_to_tum"] = dist_to_tum
    ts["dist_to_str"] = dist_to_str
    ts["tumour_stroma_balance"] = np.abs(dist_to_tum - dist_to_str)

    tum = ts[ts["label"] == "TUM"].copy()

    interface_tum = (
        tum.sort_values("tumour_stroma_balance", ascending=True)
        .head(N_PER_GROUP)
        .copy()
    )
    interface_tum["group"] = "interface"

    core_tum = (
        tum.sort_values("dist_to_tum", ascending=True)
        .head(N_PER_GROUP)
        .copy()
    )
    core_tum = core_tum[~core_tum["image_id"].isin(interface_tum["image_id"])]

    if len(core_tum) < N_PER_GROUP:
        extra_core = (
            tum[~tum["image_id"].isin(interface_tum["image_id"])]
            .sort_values("dist_to_tum", ascending=True)
            .head(N_PER_GROUP)
            .copy()
        )
        core_tum = extra_core

    core_tum["group"] = "core"

    selected = pd.concat([interface_tum, core_tum], axis=0)

    print("\nSelected patch counts:")
    print(selected["group"].value_counts())

    print("\nMean distance values:")
    print(
        selected.groupby("group")[
            ["dist_to_tum", "dist_to_str", "tumour_stroma_balance"]
        ].mean()
    )

    summary = selected.groupby("group")[PLOT_FEATURES].mean()

    print("\nFeature means:")
    print(summary)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(OUT_TABLE, sep="\t")
    selected.to_csv(OUT_SELECTED, sep="\t", index=False)

    fig, axes = plt.subplots(
        1,
        len(PLOT_FEATURES),
        figsize=(20, 5)
    )

    for ax, feature in zip(axes, PLOT_FEATURES):
        interface_values = selected[selected["group"] == "interface"][feature]
        core_values = selected[selected["group"] == "core"][feature]

        ax.boxplot(
            [interface_values, core_values],
            tick_labels=["Interface", "Core"]
        )
        ax.set_title(feature)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"\nSaved: {OUT_TABLE}")
    print(f"Saved: {OUT_SELECTED}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()