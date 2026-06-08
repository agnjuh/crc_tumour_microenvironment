from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy
from tqdm import tqdm

DATA_DIR = Path("data/raw/crc_val_7k/CRC-VAL-HE-7K")
OUT_FILE = Path("results/tables/crc_val_7k_morphology_features.tsv")

def image_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))

def edge_density(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(float))
    grad = np.sqrt(gx**2 + gy**2)
    return float(np.mean(grad > np.percentile(grad, 90)))

def compute_features(image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(float)

    gray = arr.mean(axis=2)

    features = {
        "image_id": image_path.stem,
        "filename": image_path.name,
        "label": image_path.parent.name,
        "path": str(image_path),
        "mean_r": arr[:, :, 0].mean(),
        "mean_g": arr[:, :, 1].mean(),
        "mean_b": arr[:, :, 2].mean(),
        "std_r": arr[:, :, 0].std(),
        "std_g": arr[:, :, 1].std(),
        "std_b": arr[:, :, 2].std(),
        "mean_intensity": gray.mean(),
        "std_intensity": gray.std(),
        "entropy": image_entropy(gray),
        "edge_density": edge_density(gray),
    }

    return features

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATA_DIR.glob("*/*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No .tif images found under {DATA_DIR}")

    rows = []

    for image_path in tqdm(image_paths, desc="Computing morphology features"):
        rows.append(compute_features(image_path))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"Saved features for {len(df)} images")
    print(f"Output: {OUT_FILE}")
    print(df.groupby("label").size())

if __name__ == "__main__":
    main()
