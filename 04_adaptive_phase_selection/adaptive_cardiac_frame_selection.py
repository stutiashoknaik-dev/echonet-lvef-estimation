# ============================================================
# MASK-ONLY GT vs PREDICTED CARDIAC VALIDATION
# ============================================================

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
    "outputs/mask_only_validation"
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
# VIDEO LOOP
# ============================================================

for idx, row in merged.iterrows():

    video = row["Video"]

    print(f"Creating : {video}")

    gt_ed = row["GT_ED"]
    gt_es = row["GT_ES"]

    pred_ed = row["Pred_ED"]
    pred_es = row["Pred_ES"]

    # ========================================================
    # MASK PATHS
    # ========================================================

    gt_ed_mask_path = os.path.join(
        MASK_DIR,
        video,
        f"frame_{gt_ed}.png"
    )

    gt_es_mask_path = os.path.join(
        MASK_DIR,
        video,
        f"frame_{gt_es}.png"
    )

    pred_ed_mask_path = os.path.join(
        MASK_DIR,
        video,
        f"frame_{pred_ed}.png"
    )

    pred_es_mask_path = os.path.join(
        MASK_DIR,
        video,
        f"frame_{pred_es}.png"
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    all_paths = [

        gt_ed_mask_path,
        gt_es_mask_path,

        pred_ed_mask_path,
        pred_es_mask_path
    ]

    if not all(os.path.exists(p) for p in all_paths):
        continue

    # ========================================================
    # LOAD MASKS
    # ========================================================

    gt_ed_mask = cv2.imread(
        gt_ed_mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    gt_es_mask = cv2.imread(
        gt_es_mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_ed_mask = cv2.imread(
        pred_ed_mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_es_mask = cv2.imread(
        pred_es_mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    # ========================================================
    # CONVERT TO WHITE MASKS
    # ========================================================

    gt_ed_mask = (gt_ed_mask > 0).astype(np.uint8) * 255
    gt_es_mask = (gt_es_mask > 0).astype(np.uint8) * 255

    pred_ed_mask = (pred_ed_mask > 0).astype(np.uint8) * 255
    pred_es_mask = (pred_es_mask > 0).astype(np.uint8) * 255

    # ========================================================
    # CONVERT TO 3-CHANNEL
    # ========================================================

    gt_ed_mask = cv2.cvtColor(
        gt_ed_mask,
        cv2.COLOR_GRAY2BGR
    )

    gt_es_mask = cv2.cvtColor(
        gt_es_mask,
        cv2.COLOR_GRAY2BGR
    )

    pred_ed_mask = cv2.cvtColor(
        pred_ed_mask,
        cv2.COLOR_GRAY2BGR
    )

    pred_es_mask = cv2.cvtColor(
        pred_es_mask,
        cv2.COLOR_GRAY2BGR
    )

    # ========================================================
    # ADD LABELS
    # ========================================================

    labels = [

        (gt_ed_mask, f"GT ED : {gt_ed}"),
        (pred_ed_mask, f"PRED ED : {pred_ed}"),

        (gt_es_mask, f"GT ES : {gt_es}"),
        (pred_es_mask, f"PRED ES : {pred_es}")
    ]

    for image, text in labels:

        cv2.putText(

            image,

            text,

            (5, 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            1
        )

    # ========================================================
    # CREATE PANEL
    # ========================================================

    top_row = np.hstack([
        gt_ed_mask,
        pred_ed_mask
    ])

    bottom_row = np.hstack([
        gt_es_mask,
        pred_es_mask
    ])

    panel = np.vstack([
        top_row,
        bottom_row
    ])

    # ========================================================
    # SAVE
    # ========================================================

    save_path = os.path.join(
        SAVE_DIR,
        f"{video}_mask_comparison.png"
    )

    cv2.imwrite(
        save_path,
        panel
    )

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("MASK-ONLY VALIDATION COMPLETE")
print("======================================")

print("\n✅ White mask comparison panels saved")

print(f"\nSaved Directory:\n{SAVE_DIR}")
