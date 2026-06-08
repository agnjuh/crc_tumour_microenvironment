from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)

FEATURE_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

OUT_REPORT = Path("results/tables/random_forest_integrated_report.txt")
OUT_CM = Path("results/figures/random_forest_integrated_confusion_matrix.png")

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

def main():
    df = pd.read_csv(FEATURE_FILE, sep="\t")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

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

    labels = sorted(y.unique())

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CM.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_REPORT, "w") as f:
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Random Forest confusion matrix: integrated morphology + nuclear features")
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=300)
    plt.close()

    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(report)
    print(f"\nSaved: {OUT_REPORT}")
    print(f"Saved: {OUT_CM}")

if __name__ == "__main__":
    main()
