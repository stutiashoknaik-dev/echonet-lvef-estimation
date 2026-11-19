# ============================================================
# ANALYSIS 1
# Ground Truth EF vs Predicted EF
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr

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
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nColumns Found:")
print(df.columns.tolist())

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "GT_EF",
    "Predicted_EF"
]

for col in required:

    if col not in df.columns:

        raise Exception(
            f"{col} not found."
        )

gt = df["GT_EF"].values

pred = df["Predicted_EF"].values

# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    gt,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        gt,
        pred
    )
)

r2 = r2_score(
    gt,
    pred
)

pearson_r, pearson_p = pearsonr(
    gt,
    pred
)

bias = np.mean(
    pred - gt
)

# ============================================================
# BEST FIT LINE
# ============================================================

m, c = np.polyfit(
    gt,
    pred,
    1
)

x = np.linspace(
    gt.min(),
    gt.max(),
    100
)

y = m*x + c

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8,8))

plt.scatter(

    gt,

    pred,

    color="royalblue",

    alpha=0.65,

    s=30,

    edgecolors="black"

)

# Perfect prediction

plt.plot(

    [0,100],

    [0,100],

    'r--',

    linewidth=2,

    label="Perfect Prediction"

)

# Regression line

plt.plot(

    x,

    y,

    color="green",

    linewidth=2,

    label="Regression Line"

)

# ============================================================
# METRIC BOX
# ============================================================

text = (

    f"MAE = {mae:.2f}\n"

    f"RMSE = {rmse:.2f}\n"

    f"R² = {r2:.3f}\n"

    f"Pearson r = {pearson_r:.3f}\n"

    f"Bias = {bias:.2f}"

)

plt.text(

    0.05,

    0.95,

    text,

    transform=plt.gca().transAxes,

    fontsize=11,

    verticalalignment="top",

    bbox=dict(

        facecolor="white",

        edgecolor="black"

    )

)

plt.xlabel(

    "Ground Truth EF (%)",

    fontsize=12

)

plt.ylabel(

    "Predicted EF (%)",

    fontsize=12

)

plt.title(

    "Ground Truth vs Predicted EF",

    fontsize=15,

    fontweight="bold"

)

plt.xlim(0,100)

plt.ylim(0,100)

plt.grid(True)

plt.legend()

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(

    SAVE_DIR,

    "01_GT_vs_Predicted_EF.png"

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

print("\n=======================================")
print("GROUND TRUTH vs PREDICTED EF")
print("=======================================")

print(f"Samples      : {len(df)}")
print(f"MAE          : {mae:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"R²           : {r2:.4f}")
print(f"Pearson r    : {pearson_r:.4f}")
print(f"P-value      : {pearson_p:.6f}")
print(f"Bias         : {bias:.4f}")

print("\nRegression Line")

print(f"Pred = {m:.3f} × GT + {c:.3f}")

print("\nFigure Saved")

print(save_path)
