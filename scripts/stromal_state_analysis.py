from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import umap

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_TABLE = Path("results/tables/stromal_state_analysis.tsv")
OUT_SUMMARY = Path("results/tables/stromal_state_summary.tsv")
OUT_FEATURES = Path("results/tables/stromal_state_feature_importance.tsv")
OUT_UMAP = Path("results/figures/stromal_state_umap.png")
OUT_MONTAGE = Path("results/figures/stromal_state_montage.png")
OUT_FEATURE_FIG = Path("results/figures/stromal_state_feature_importance.png")

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

N_STATES = 3
PATCHES_PER_STATE = 8


def make_montage(df, out_file):
    states = sorted(df["stromal_state"].unique())

    fig, axes = plt.subplots(
        len(states),
        PATCHES_PER_STATE,
        figsize=(16, 7),
    )

    if len(states) == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax in axes.flatten():
        ax.axis("off")

    for r, state in enumerate(states):
        sub = df[df["stromal_state"] == state].copy()
        sub = sub.sort_values("state_distance").head(PATCHES_PER_STATE)

        for c, (_, row) in enumerate(sub.iterrows()):
            img = Image.open(row["path"]).convert("RGB")
            axes[r, c].imshow(img)
            axes[r, c].set_title(
                f"{state}\ncell={row['nuclei_count']:.0f}",
                fontsize=7,
            )
            axes[r, c].axis("off")

        axes[r, 0].set_ylabel(state, fontsize=9)

    plt.suptitle("Representative stromal morphology states", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")
    stroma = df[df["label"] == "STR"].copy()

    X = stroma[FEATURE_COLUMNS]
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=N_STATES,
        random_state=42,
        n_init=50,
    )

    clusters = kmeans.fit_predict(X_pca)

    stroma["stromal_cluster"] = clusters

    state_names = {}

    summary_tmp = []

    for cluster in sorted(stroma["stromal_cluster"].unique()):
        sub = stroma[stroma["stromal_cluster"] == cluster]
        summary_tmp.append(
            {
                "cluster": cluster,
                "n": len(sub),
                "mean_nuclei_count": sub["nuclei_count"].mean(),
                "mean_nucleus_area": sub["mean_nucleus_area"].mean(),
                "mean_entropy": sub["entropy"].mean(),
                "mean_edge_density": sub["edge_density"].mean(),
                "mean_intensity": sub["mean_intensity"].mean(),
            }
        )

    tmp = pd.DataFrame(summary_tmp)

    high_cell_cluster = tmp.sort_values("mean_nuclei_count", ascending=False).iloc[0]["cluster"]
    large_nucleus_cluster = tmp.sort_values("mean_nucleus_area", ascending=False).iloc[0]["cluster"]

    for cluster in sorted(stroma["stromal_cluster"].unique()):
        if cluster == high_cell_cluster:
            state_names[cluster] = "cellular stroma"
        elif cluster == large_nucleus_cluster:
            state_names[cluster] = "large-nucleus stroma"
        else:
            state_names[cluster] = "matrix-like stroma"

    stroma["stromal_state"] = stroma["stromal_cluster"].map(state_names)

    centres = kmeans.cluster_centers_
    distances = np.linalg.norm(
        X_pca - centres[clusters],
        axis=1,
    )
    stroma["state_distance"] = distances

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
    )

    coords = reducer.fit_transform(X_scaled)
    stroma["UMAP1"] = coords[:, 0]
    stroma["UMAP2"] = coords[:, 1]

    summary = (
        stroma.groupby("stromal_state")
        .agg(
            n=("image_id", "count"),
            mean_nuclei_count=("nuclei_count", "mean"),
            mean_nuclei_density=("nuclei_density", "mean"),
            mean_nuclear_area_fraction=("nuclear_area_fraction", "mean"),
            mean_nucleus_area=("mean_nucleus_area", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_edge_density=("edge_density", "mean"),
            mean_intensity=("mean_intensity", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(stroma[FEATURE_COLUMNS], stroma["stromal_state"])

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_UMAP.parent.mkdir(parents=True, exist_ok=True)

    stroma.to_csv(OUT_TABLE, sep="\t", index=False)
    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
    importance.to_csv(OUT_FEATURES, sep="\t", index=False)

    plt.figure(figsize=(8, 6))
    for state in sorted(stroma["stromal_state"].unique()):
        sub = stroma[stroma["stromal_state"] == state]
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=18,
            alpha=0.8,
            label=f"{state} (n={len(sub)})",
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Stromal morphology states")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_UMAP, dpi=300)
    plt.close()

    make_montage(stroma, OUT_MONTAGE)

    plt.figure(figsize=(8, 6))
    top = importance.head(15)
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Random Forest importance")
    plt.ylabel("Feature")
    plt.title("Features separating stromal morphology states")
    plt.tight_layout()
    plt.savefig(OUT_FEATURE_FIG, dpi=300)
    plt.close()

    print("Saved:")
    print(OUT_TABLE)
    print(OUT_SUMMARY)
    print(OUT_FEATURES)
    print(OUT_UMAP)
    print(OUT_MONTAGE)
    print(OUT_FEATURE_FIG)

    print("\nStromal state summary:")
    print(summary.to_string(index=False))

    print("\nTop stromal-state features:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
