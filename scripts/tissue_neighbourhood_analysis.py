from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_COUNTS = Path("results/tables/tissue_neighbourhood_counts.tsv")
OUT_ENRICHMENT = Path("results/tables/tissue_neighbourhood_enrichment.tsv")
OUT_HEATMAP = Path("results/figures/tissue_neighbourhood_enrichment_heatmap.png")

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

K_NEIGHBOURS = 15


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t").copy()

    labels = sorted(df["label"].unique())

    X = df[FEATURE_COLUMNS]
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBOURS + 1,
        metric="euclidean",
    )

    nn.fit(X_pca)

    distances, indices = nn.kneighbors(X_pca)

    rows = []

    for i, neighbours in enumerate(indices):
        centre_label = df.iloc[i]["label"]

        # skip first neighbour because it is the patch itself
        for neighbour_index in neighbours[1:]:
            neighbour_label = df.iloc[neighbour_index]["label"]

            rows.append(
                {
                    "centre_label": centre_label,
                    "neighbour_label": neighbour_label,
                }
            )

    edges = pd.DataFrame(rows)

    counts = (
        edges.groupby(["centre_label", "neighbour_label"])
        .size()
        .reset_index(name="observed_count")
    )

    count_matrix = (
        counts.pivot(
            index="centre_label",
            columns="neighbour_label",
            values="observed_count",
        )
        .reindex(index=labels, columns=labels)
        .fillna(0)
    )

    label_freq = df["label"].value_counts(normalize=True).reindex(labels)

    expected_matrix = pd.DataFrame(
        index=labels,
        columns=labels,
        dtype=float,
    )

    for centre in labels:
        total_neighbours = count_matrix.loc[centre].sum()
        expected_matrix.loc[centre] = total_neighbours * label_freq

    enrichment_matrix = np.log2(
        (count_matrix + 1) / (expected_matrix + 1)
    )

    OUT_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)

    count_matrix.to_csv(OUT_COUNTS, sep="\t")
    enrichment_matrix.to_csv(OUT_ENRICHMENT, sep="\t")

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(
        enrichment_matrix.values.astype(float),
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2 enrichment")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Neighbour tissue class")
    ax.set_ylabel("Centre tissue class")
    ax.set_title("Morphology-space tissue neighbourhood enrichment")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = enrichment_matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(OUT_HEATMAP, dpi=300)
    plt.close()

    print("Saved:")
    print(OUT_COUNTS)
    print(OUT_ENRICHMENT)
    print(OUT_HEATMAP)

    print("\nTop enriched neighbourhoods:")
    long = (
        enrichment_matrix.stack()
        .reset_index()
    )
    long.columns = ["centre_label", "neighbour_label", "log2_enrichment"]
    long = long[long["centre_label"] != long["neighbour_label"]]
    print(
        long.sort_values("log2_enrichment", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
