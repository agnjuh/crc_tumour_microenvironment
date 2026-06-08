from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
import umap

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_COORDS = Path("results/tables/tme_subset_pca_umap_coordinates.tsv")
OUT_REPORT = Path("results/tables/tme_subset_random_forest_report.txt")
OUT_UMAP = Path("results/figures/tme_subset_umap.png")
OUT_PCA = Path("results/figures/tme_subset_pca.png")
OUT_CM = Path("results/figures/tme_subset_confusion_matrix.png")

KEEP_LABELS = ["TUM", "STR", "LYM", "NORM"]

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
    plt.figure(figsize=(8, 6))

    for label in KEEP_LABELS:
        sub = df[df["label"] == label]
        plt.scatter(
            sub[x],
            sub[y],
            s=12,
            alpha=0.75,
            label=label,
        )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(markerscale=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, labels, out_file):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of patches")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("TME subset confusion matrix")

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")
    df = df[df["label"].isin(KEEP_LABELS)].copy()

    X = df[FEATURE_COLUMNS]
    y = df["label"]

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

    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    OUT_UMAP.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_COORDS, sep="\t", index=False)

    plot_embedding(
        df,
        "PC1",
        "PC2",
        OUT_PCA,
        "Tumour-Stroma-Immune subset PCA",
    )

    plot_embedding(
        df,
        "UMAP1",
        "UMAP2",
        OUT_UMAP,
        "Tumour-Stroma-Immune interface UMAP",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(OUT_REPORT, "w") as f:
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=KEEP_LABELS)
    plot_confusion_matrix(cm, KEEP_LABELS, OUT_CM)

    print(f"Subset size: {df.shape}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(report)
    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("\nSaved:")
    print(OUT_COORDS)
    print(OUT_PCA)
    print(OUT_UMAP)
    print(OUT_REPORT)
    print(OUT_CM)


if __name__ == "__main__":
    main()
