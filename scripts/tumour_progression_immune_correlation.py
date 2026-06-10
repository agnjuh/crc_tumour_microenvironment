from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


PROGRESSION_FILE = Path("results/tables/normal_tumour_progression_axis.tsv")
IMMUNE_FILE = Path("results/tables/tumour_immune_proximity_scores.tsv")

OUT_TABLE = Path("results/tables/tumour_progression_immune_correlation.tsv")
OUT_STATS = Path("results/tables/tumour_progression_immune_stats.tsv")
OUT_FIGURE = Path("results/figures/tumour_progression_immune_scatter.png")


def main():
    progression = pd.read_csv(PROGRESSION_FILE, sep="\t")
    immune = pd.read_csv(IMMUNE_FILE, sep="\t")

    df = progression.merge(
        immune[["image_id", "immune_proximity_score"]],
        on="image_id",
        how="inner",
    )

    df = df[df["label"] == "TUM"].copy()

    pearson_r, pearson_p = pearsonr(
        df["progression_score"],
        df["immune_proximity_score"],
    )

    spearman_rho, spearman_p = spearmanr(
        df["progression_score"],
        df["immune_proximity_score"],
    )

    stats = pd.DataFrame(
        {
            "metric": [
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
                "n_tumour_patches",
            ],
            "value": [
                pearson_r,
                pearson_p,
                spearman_rho,
                spearman_p,
                len(df),
            ],
        }
    )

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_TABLE, sep="\t", index=False)
    stats.to_csv(OUT_STATS, sep="\t", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["progression_score"],
        df["immune_proximity_score"],
        alpha=0.6,
        s=20,
    )
    plt.xlabel("Normal to tumour progression score")
    plt.ylabel("Immune-proximity score")
    plt.title("Tumour progression versus immune proximity")
    plt.tight_layout()
    plt.savefig(OUT_FIGURE, dpi=300)
    plt.close()

    print("Tumour progression versus immune proximity")
    print(f"Pearson r = {pearson_r:.4f} (p={pearson_p:.3e})")
    print(f"Spearman rho = {spearman_rho:.4f} (p={spearman_p:.3e})")
    print(f"n = {len(df)}")
    print(f"Saved: {OUT_TABLE}")
    print(f"Saved: {OUT_STATS}")
    print(f"Saved: {OUT_FIGURE}")


if __name__ == "__main__":
    main()
