
import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT PATHS
# ============================================================

MASK_DIR = os.path.join(
    BASE_PATH,
    "outputs/Test_predictions_edgeaware/masks"
)

GT_CSV = os.path.join(
    BASE_PATH,
    "data/VolumeTracings.csv"
)

PRED_CSV = os.path.join(
    BASE_PATH,
    "outputs/multifeature_cardiac_analysis/stable_cardiac_regions.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/mask_only_validation_data"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading CSV files...\n")

gt_df = pd.read_csv(GT_CSV)

pred_df = pd.read_csv(PRED_CSV)

print("GT CSV Loaded")
print("Prediction CSV Loaded")

# ============================================================
# CLEAN VIDEO NAME
# ============================================================

gt_df["Video"] = gt_df["FileName"].str.replace(
    ".avi",
    "",
    regex=False
)

# ============================================================
# COMPUTE GT ED/ES USING AREA
# ============================================================

frame_dict = {}

grouped = gt_df.groupby(
    ["Video", "Frame"]
)

for (video, frame), group in grouped:

    pts = group[["X1", "Y1"]].values

    x = pts[:, 0]
    y = pts[:, 1]

    area = 0.5 * abs(

        np.dot(x, np.roll(y, 1))

        -

        np.dot(y, np.roll(x, 1))
    )

    if video not in frame_dict:
        frame_dict[video] = []

    frame_dict[video].append(
        (frame, area)
    )

# ============================================================
# FINAL GT TABLE
# ============================================================

gt_results = []

for video, values in frame_dict.items():

    values = sorted(
        values,
        key=lambda x: x[1]
    )

    gt_es = values[0][0]

    gt_ed = values[-1][0]

    gt_results.append({

        "Video": video,

        "GT_ED": int(gt_ed),

        "GT_ES": int(gt_es)
    })

gt_final_df = pd.DataFrame(gt_results)

# ============================================================
# BEST PREDICTED CANDIDATES
# ============================================================

pred_best = pred_df.sort_values(
    "Score",
    ascending=False
).drop_duplicates(
    subset=["Video", "Type"]
)

# ============================================================
# SPLIT ED / ES
# ============================================================

pred_ed = pred_best[
    pred_best["Type"] == "ED"
][["Video", "Frame"]]

pred_ed.columns = [
    "Video",
    "Pred_ED"
]

pred_es = pred_best[
    pred_best["Type"] == "ES"
][["Video", "Frame"]]

pred_es.columns = [
    "Video",
    "Pred_ES"
]

# ============================================================
# MERGE
# ============================================================

merged = gt_final_df.merge(
    pred_ed,
    on="Video"
)

merged = merged.merge(
    pred_es,
    on="Video"
)

print(f"\nTotal Videos : {len(merged)}")

# ============================================================
# SAVE FRAME COMPARISON CSV
# ============================================================

csv_save_path = os.path.join(
    SAVE_DIR,
    "predicted_frames.csv"
)

merged = merged.sort_values("Video").reset_index(drop=True)

merged.to_csv(
    csv_save_path,
    index=False
)

print("\n======================================")
print("FRAME CSV SAVED")
print("======================================")
print(csv_save_path)
