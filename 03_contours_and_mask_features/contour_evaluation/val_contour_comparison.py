
# ============================================================
# FINAL GT vs PRED CONTOUR METRICS
# ============================================================
#
# USES:
# -----
# 1. ORIGINAL GT VolumeTracings
# 2. IMPROVED predicted contour extraction
#
# NO RESAMPLING
# NO FORCED POINT MATCHING
#
# METRICS:
# --------
# 1. Chamfer Distance
# 2. Hausdorff Distance
# 3. Mean Contour Distance (optional)
#
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# GT CSV
# ============================================================

GT_CSV = os.path.join(
    BASE_PATH,
    "outputs/val_contour_gt",
    "val_volume_tracings.csv"
)

# ============================================================
# PRED CSV
# ============================================================

PRED_CSV = os.path.join(
    BASE_PATH,
    "outputs/improved_validation_contours",
    "Improved_Predicted_Val_VolumeTracings.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_contour_metrics"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE FILES
# ============================================================

DETAIL_CSV = os.path.join(
    SAVE_DIR,
    "Contour_Metrics_Per_Frame.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "Final_Contour_Metrics.txt"
)

# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading contour CSV files...\n")

gt_df = pd.read_csv(GT_CSV)
pred_df = pd.read_csv(PRED_CSV)

print("GT CSV Loaded")
print("Predicted CSV Loaded")

# ============================================================
# RESULTS STORAGE
# ============================================================

results = []

# ============================================================
# UNIQUE GT FRAMES
# ============================================================

unique_frames = gt_df[
    ["FileName", "Frame"]
].drop_duplicates()

# ============================================================
# MAIN LOOP
# ============================================================

print("\nComputing contour metrics...\n")

for _, row in unique_frames.iterrows():

    video = row["FileName"]
    frame = row["Frame"]

    # --------------------------------------------------------
    # FILTER GT
    # --------------------------------------------------------

    gt_temp = gt_df[

        (gt_df["FileName"] == video)

        &

        (gt_df["Frame"] == frame)

    ]

    # --------------------------------------------------------
    # FILTER PRED
    # --------------------------------------------------------

    pred_temp = pred_df[

        (pred_df["FileName"] == video)

        &

        (pred_df["Frame"] == frame)

    ]

    if len(gt_temp) == 0 or len(pred_temp) == 0:
        continue

    # ========================================================
    # CREATE GT POINT ARRAY
    # ========================================================

    gt_left = gt_temp[
        ["X1", "Y1"]
    ].values

    gt_right = gt_temp[
        ["X2", "Y2"]
    ].values

    gt_points = np.vstack([

        gt_left,
        gt_right

    ])

    # ========================================================
    # CREATE PRED POINT ARRAY
    # ========================================================

    pred_left = pred_temp[
        ["X1", "Y1"]
    ].values

    pred_right = pred_temp[
        ["X2", "Y2"]
    ].values

    pred_points = np.vstack([

        pred_left,
        pred_right

    ])

    # ========================================================
    # DISTANCE MATRIX
    # ========================================================

    dist_matrix = cdist(
        gt_points,
        pred_points
    )

    # ========================================================
    # CHAMFER DISTANCE
    # ========================================================

    chamfer = (

        np.mean(np.min(dist_matrix, axis=1))

        +

        np.mean(np.min(dist_matrix, axis=0))

    ) / 2

    # ========================================================
    # HAUSDORFF DISTANCE
    # ========================================================
   # ========================================================
# HD95
# ========================================================

# GT -> Pred distances
    gt_to_pred = np.min(dist_matrix, axis=1)

# Pred -> GT distances
    pred_to_gt = np.min(dist_matrix, axis=0)

# Combine distances
    all_surface_distances = np.concatenate([
      gt_to_pred,
      pred_to_gt
])

# 95th percentile Hausdorff
    hd95 = np.percentile(
     all_surface_distances,95
)
    # ========================================================
    # OPTIONAL:
    # MEAN CONTOUR DISTANCE
    # ========================================================

    mean_distance = np.mean(

        np.min(dist_matrix, axis=1)

    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "FileName": video,
        "Frame": frame,

        "Chamfer_Distance": chamfer,

        "HD95": hd95,

        "Mean_Contour_Distance": mean_distance
    })

# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    DETAIL_CSV,
    index=False
)

# ============================================================
# COMPUTE FINAL AVERAGES
# ============================================================

avg_chamfer = results_df[
    "Chamfer_Distance"
].mean()

avg_hd95 = results_df[
    "HD95"
].mean()

avg_mean_distance = results_df[
    "Mean_Contour_Distance"
].mean()

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("FINAL CONTOUR METRICS")
print("======================================")

print(f"\nChamfer Distance : {avg_chamfer:.4f}")

print(f"HD95 : {avg_hd95:.4f}")

print(f"Mean Contour Distance : {avg_mean_distance:.4f}")

# ============================================================
# SAVE TXT SUMMARY
# ============================================================

with open(SUMMARY_TXT, "w") as f:

    f.write("======================================\n")
    f.write("FINAL CONTOUR METRICS\n")
    f.write("======================================\n\n")

    f.write(f"Chamfer Distance : {avg_chamfer:.4f}\n")
    f.write(f"HD95 : {avg_hd95:.4f}\n")
    f.write(f"Mean Contour Distance : "
            f"{avg_mean_distance:.4f}\n")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("FINAL CONTOUR EVALUATION COMPLETE")
print("======================================")

print(f"\nDetailed CSV Saved:\n{DETAIL_CSV}")

print(f"\nSummary TXT Saved:\n{SUMMARY_TXT}")
