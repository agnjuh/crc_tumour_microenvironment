from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

COORD_FILE = Path(
    "results/tables/crc_val_7k_integrated_pca_umap_coordinates.tsv"
)

OUT_FILE = Path(
    "results/figures/crc_val_7k_umap_patch_montage.png"
)

N_PATCHES = 150
THUMBNAIL_SIZE = 48

def main():

    df = pd.read_csv(COORD_FILE, sep="\t")

    rng = np.random.default_rng(42)

    if len(df) > N_PATCHES:
        df = df.sample(
            N_PATCHES,
            random_state=42
        ).reset_index(drop=True)

    x = df["UMAP1"].values
    y = df["UMAP2"].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)

    ax.set_title(
        "CRC-VAL-HE-7K Morphology Atlas",
        fontsize=16
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    for _, row in df.iterrows():

        img_path = row["path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(
                (THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            )

            arr = np.asarray(img)

            x0 = row["UMAP1"]
            y0 = row["UMAP2"]

            extent = [
                x0 - 0.4,
                x0 + 0.4,
                y0 - 0.4,
                y0 + 0.4,
            ]

            ax.imshow(
                arr,
                extent=extent,
                aspect="auto",
            )

        except Exception:
            continue

    plt.tight_layout()
    plt.savefig(
        OUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
