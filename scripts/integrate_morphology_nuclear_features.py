from pathlib import Path

import pandas as pd

MORPH_FILE = Path("results/tables/crc_val_7k_morphology_features.tsv")
NUCLEAR_FILE = Path("results/tables/crc_val_7k_nuclear_features.tsv")
OUT_FILE = Path("results/tables/crc_val_7k_integrated_features.tsv")

morph = pd.read_csv(MORPH_FILE, sep="\t")
nuclear = pd.read_csv(NUCLEAR_FILE, sep="\t")

drop_cols = ["filename", "label", "path"]

nuclear_reduced = nuclear.drop(columns=drop_cols)

df = morph.merge(
    nuclear_reduced,
    on="image_id",
    how="inner"
)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, sep="\t", index=False)

print(f"Morphology features: {morph.shape}")
print(f"Nuclear features: {nuclear.shape}")
print(f"Integrated features: {df.shape}")
print(f"Saved: {OUT_FILE}")
