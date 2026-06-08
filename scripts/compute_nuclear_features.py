from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops

DATA_DIR = Path("data/raw/crc_val_7k/CRC-VAL-HE-7K")
OUT_FILE = Path("results/tables/crc_val_7k_nuclear_features.tsv")

MIN_OBJECT_SIZE = 20
PATCH_AREA = 224 * 224

def compute_nuclear_features(image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img)

    hed = rgb2hed(rgb)
    hematoxylin = hed[:, :, 0]

    hematoxylin = hematoxylin - hematoxylin.min()
    if hematoxylin.max() > 0:
        hematoxylin = hematoxylin / hematoxylin.max()

    try:
        thresh = threshold_otsu(hematoxylin)
        nuclei_mask = hematoxylin > thresh
    except ValueError:
        nuclei_mask = np.zeros_like(hematoxylin, dtype=bool)

    nuclei_mask = remove_small_objects(nuclei_mask, min_size=MIN_OBJECT_SIZE)
    nuclei_mask = remove_small_holes(nuclei_mask, area_threshold=MIN_OBJECT_SIZE)

    labelled = label(nuclei_mask)
    props = regionprops(labelled)

    areas = np.array([p.area for p in props], dtype=float)
    eccentricities = np.array([p.eccentricity for p in props], dtype=float)
    solidities = np.array([p.solidity for p in props], dtype=float)
    perimeters = np.array([p.perimeter for p in props], dtype=float)

    nuclei_count = len(props)

    if nuclei_count > 0:
        mean_area = areas.mean()
        median_area = np.median(areas)
        std_area = areas.std()
        mean_eccentricity = eccentricities.mean()
        mean_solidity = solidities.mean()
        mean_perimeter = perimeters.mean()
        nuclear_area_fraction = areas.sum() / PATCH_AREA
    else:
        mean_area = 0.0
        median_area = 0.0
        std_area = 0.0
        mean_eccentricity = 0.0
        mean_solidity = 0.0
        mean_perimeter = 0.0
        nuclear_area_fraction = 0.0

    return {
        "image_id": image_path.stem,
        "filename": image_path.name,
        "label": image_path.parent.name,
        "path": str(image_path),
        "nuclei_count": nuclei_count,
        "nuclei_density": nuclei_count / PATCH_AREA,
        "nuclear_area_fraction": nuclear_area_fraction,
        "mean_nucleus_area": mean_area,
        "median_nucleus_area": median_area,
        "std_nucleus_area": std_area,
        "mean_nucleus_eccentricity": mean_eccentricity,
        "mean_nucleus_solidity": mean_solidity,
        "mean_nucleus_perimeter": mean_perimeter,
    }

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(DATA_DIR.glob("*/*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No .tif images found under {DATA_DIR}")

    rows = []

    for image_path in tqdm(image_paths, desc="Computing nuclear features"):
        rows.append(compute_nuclear_features(image_path))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, sep="\t", index=False)

    print(f"Saved nuclear features for {len(df)} images")
    print(f"Output: {OUT_FILE}")
    print(df.groupby("label")["nuclei_count"].describe())

if __name__ == "__main__":
    main()
