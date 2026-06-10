from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


FEATURE_FILE = Path(
    "results/tables/crc_val_7k_integrated_features.tsv"
)

IMMUNE_FILE = Path(
    "results/tables/tumour_immune_proximity_scores.tsv"
)

OUT_TABLE = Path(
    "results/tables/immune_proximity_feature_importance.tsv"
)

OUT_FIGURE = Path(
    "results/figures/immune_proximity_feature_importance.png"
)


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

    features = pd.read_csv(
        FEATURE_FILE,
        sep="\t"
    )

    immune = pd.read_csv(
        IMMUNE_FILE,
        sep="\t"
    )

    df = features.merge(
        immune[
            [
                "image_id",
                "immune_proximity_score"
            ]
        ],
        on="image_id",
        how="inner"
    )

    df = df[df["label"] == "TUM"].copy()

    X = df[FEATURE_COLUMNS]
    y = df["immune_proximity_score"]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    OUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(
        OUT_TABLE,
        sep="\t",
        index=False
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Random Forest importance")
    plt.ylabel("Feature")

    plt.title(
        "Features associated with tumour immune proximity"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_FIGURE,
        dpi=300
    )

    plt.close()

    print("\nTop features:")

    print(
        importance.head(15).to_string(
            index=False
        )
    )

    print(f"\nSaved: {OUT_TABLE}")
    print(f"Saved: {OUT_FIGURE}")


if __name__ == "__main__":
    main()
