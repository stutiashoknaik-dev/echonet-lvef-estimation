# ============================================================
# ANALYSIS 2
# Residual Plot
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

CSV_PATH = os.path.join(os.environ.get("ECHONET_ROOT", "./EchoNet_LV"), "outputs", "r21d_multimodal", "Predicted_EF_Test.csv")

SAVE_DIR = os.path.join(os.environ.get("ECHONET_ROOT", "./EchoNet_LV"), "outputs", "analysis")

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

gt = df["GT_EF"]

pred = df["Predicted_EF"]

# ============================================================
# RESIDUALS
# ============================================================

df["Residual"] = pred - gt

print(df["Residual"].describe())

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9,6))

plt.scatter(
    gt,
    df["Residual"],
    color="royalblue",
    alpha=0.65,
    s=30,
    edgecolors="black"
)

# Zero Error Line

plt.axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

# Mean Residual Line

mean_residual = df["Residual"].mean()

plt.axhline(
    mean_residual,
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Mean Residual = {mean_residual:.2f}"
)

# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "Ground Truth EF (%)",
    fontsize=12
)

plt.ylabel(
    "Residual (Predicted - GT)",
    fontsize=12
)

plt.title(
    "Residual Plot",
    fontsize=15,
    fontweight="bold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(
    SAVE_DIR,
    "02_Residual_Plot.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("RESIDUAL ANALYSIS")
print("===================================")

print(f"Mean Residual : {mean_residual:.3f}")

print(f"Minimum       : {df['Residual'].min():.3f}")

print(f"Maximum       : {df['Residual'].max():.3f}")

print(f"Std Dev       : {df['Residual'].std():.3f}")

print("\nSaved Figure:")

print(save_path)
