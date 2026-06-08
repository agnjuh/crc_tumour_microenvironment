from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

FEATURE_FILE = Path("results/tables/crc_val_7k_morphology_features.tsv")
OUT_FIG = Path("results/figures/random_forest_feature_importance.png")

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
]

df = pd.read_csv(FEATURE_FILE, sep="\t")

X = df[FEATURE_COLUMNS]
y = df["label"]

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X, y)

imp = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": rf.feature_importances_,
})

imp = imp.sort_values(
    "importance",
    ascending=True,
)

plt.figure(figsize=(8, 5))
plt.barh(
    imp["feature"],
    imp["importance"]
)
plt.xlabel("Importance")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)

print(imp)
print(f"\nSaved: {OUT_FIG}")
