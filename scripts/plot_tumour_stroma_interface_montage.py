from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

INTERFACE_FILE = Path("results/tables/tumour_stroma_interface_patches.tsv")
OUT_FILE = Path("results/figures/tumour_stroma_interface_montage.png")

N_PATCHES = 64
N_COLS = 8

def main():
    df = pd.read_csv(INTERFACE_FILE, sep="\t").head(N_PATCHES)

    n_rows = math.ceil(len(df) / N_COLS)

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(12, 12)
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        img = Image.open(row["path"]).convert("RGB")

        ax.imshow(img)
        ax.set_title(
            f'{row["label"]}\nscore={row["interface_score"]:.3f}',
            fontsize=7
        )
        ax.axis("off")

    for ax in axes[len(df):]:
        ax.axis("off")

    plt.suptitle(
        "Most ambiguous tumour–stroma interface patches",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    plt.close()

    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
