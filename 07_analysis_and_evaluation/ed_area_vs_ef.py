# ============================================================
# FEATURE vs EF
# ED Area vs EF
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------

CSV_PATH = os.path.join(os.environ.get("ECHONET_ROOT", "./EchoNet_LV"), "outputs", "mask_features", "train_mask_features.csv")

SAVE_DIR = os.path.join(os.environ.get("ECHONET_ROOT", "./EchoNet_LV"), "outputs", "analysis")

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------
# CORRELATION
# ------------------------------------------------------------

r, p = pearsonr(
    df["Perimeter_Ratio"],
    df["EF"]
)

print()

print("Pearson r :", round(r,4))
print("P-value   :", p)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["Perimeter_Ratio"],
    df["EF"],
    alpha=0.5
)

plt.xlabel("Perimeter_Ratio")

plt.ylabel("Ground Truth EF")

plt.title("Perimeter_Ratio vs EF")

plt.text(

    0.03,

    0.97,

    f"Pearson r = {r:.3f}",

    transform=plt.gca().transAxes,

    verticalalignment="top",

    bbox=dict(facecolor="white")

)

plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    SAVE_DIR,
    "Perimeter_Ratio_vs_EF.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.show()

print()
print("Saved :", save_path)
