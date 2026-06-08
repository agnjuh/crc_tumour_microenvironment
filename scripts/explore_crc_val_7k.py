from pathlib import Path
import random

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/raw/crc_val_7k/CRC-VAL-HE-7K")
OUT_TABLE = Path("results/tables/crc_val_7k_class_counts.tsv")
OUT_FIG = Path("results/figures/crc_val_7k_class_distribution.png")
OUT_GRID = Path("results/figures/crc_val_7k_example_patches.png")

CLASS_NAMES = {
    "ADI": "adipose",
    "BACK": "background",
    "DEB": "debris",
    "LYM": "lymphocytes",
    "MUC": "mucus",
    "MUS": "smooth muscle",
    "NORM": "normal mucosa",
    "STR": "cancer-associated stroma",
    "TUM": "tumour epithelium",
}

def main():
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths = []

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        files = sorted(class_dir.glob("*.tif"))

        rows.append({
            "label": label,
            "class_name": CLASS_NAMES.get(label, label),
            "n_images": len(files),
        })

        for f in files:
            image_paths.append((label, f))

    counts = pd.DataFrame(rows)
    counts.to_csv(OUT_TABLE, sep="\t", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(counts["label"], counts["n_images"])
    plt.xlabel("Tissue class")
    plt.ylabel("Number of image patches")
    plt.title("CRC-VAL-HE-7K class distribution")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    random.seed(42)
    selected = []

    for label in sorted(CLASS_NAMES):
        class_files = [p for lab, p in image_paths if lab == label]
        if class_files:
            selected.append((label, random.choice(class_files)))

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()

    for ax, (label, img_path) in zip(axes, selected):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{label}: {CLASS_NAMES.get(label, label)}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_GRID, dpi=300)
    plt.close()

    print(counts)
    print(f"\nSaved: {OUT_TABLE}")
    print(f"Saved: {OUT_FIG}")
    print(f"Saved: {OUT_GRID}")

if __name__ == "__main__":
    main()
