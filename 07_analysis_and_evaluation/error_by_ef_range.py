# ============================================================
# ERROR ANALYSIS BY EF RANGE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

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

# ============================================================
# CREATE EF GROUPS
# ============================================================

bins = [0,40,50,60,70,100]

labels = [

    "<40",

    "40-50",

    "50-60",

    "60-70",

    ">70"

]

df["EF_Group"] = pd.cut(

    df["GT_EF"],

    bins=bins,

    labels=labels,

    include_lowest=True

)

# ============================================================
# ANALYSIS
# ============================================================

results = []

for group in labels:

    subset = df[

        df["EF_Group"] == group

    ]

    if len(subset)==0:

        continue

    gt = subset["GT_EF"]

    pred = subset["Predicted_EF"]

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

    bias = np.mean(

        pred - gt

    )

    results.append({

        "EF Range": group,

        "Patients": len(subset),

        "Mean GT EF": gt.mean(),

        "Mean Pred EF": pred.mean(),

        "MAE": mae,

        "RMSE": rmse,

        "Bias": bias

    })

# ============================================================
# RESULTS TABLE
# ============================================================

result_df = pd.DataFrame(results)

print("\n====================================")

print("ERROR BY EF RANGE")

print("====================================\n")

print(

    result_df.round(3)

)

# ============================================================
# SAVE CSV
# ============================================================

csv_save = os.path.join(

    SAVE_DIR,

    "Error_by_EF_Range.csv"

)

result_df.to_csv(

    csv_save,

    index=False

)

# ============================================================
# BAR PLOT
# ============================================================

plt.figure(figsize=(8,5))

bars = plt.bar(

    result_df["EF Range"],

    result_df["MAE"]

)

# Add values above bars

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x()+bar.get_width()/2,

        height+0.2,

        f"{height:.2f}",

        ha="center",

        fontsize=10

    )

plt.ylabel("Mean Absolute Error")

plt.xlabel("Ground Truth EF Range")

plt.title("MAE Across EF Ranges")

plt.grid(

    axis="y",

    alpha=0.3

)

plt.tight_layout()

plot_save = os.path.join(

    SAVE_DIR,

    "MAE_by_EF_Range.png"

)

plt.savefig(

    plot_save,

    dpi=300

)

plt.show()

print("\nSaved CSV :")

print(csv_save)

print("\nSaved Plot :")

print(plot_save)

print("\n====================================")
print("DONE")
print("====================================")
