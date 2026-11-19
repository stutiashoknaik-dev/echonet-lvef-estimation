# ============================================================
# GENERATE 15 VISUAL REPRESENTATIVE COMPARISONS
# ============================================================
#
# OUTPUT IMAGE FORMAT:
#
# ------------------------------------------------------------
# | GT ED | PRED ED | GT ES | PRED ES |
# ------------------------------------------------------------
#
# Categories:
#   - 5 Best
#   - 5 Average
#   - 5 Difficult
#
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
# INPUT CSVs
# ============================================================

ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/final_gt_vs_pred_metrics/ED_metrics.csv"
)

ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/final_gt_vs_pred_metrics/ES_metrics.csv"
)

# ============================================================
# GT MASK DIRECTORY
# ============================================================

GT_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_GT_test_masks"
)

# ============================================================
# PRED DIRECTORY
# ============================================================

PRED_DIR = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_visual_comparisons"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD METRICS
# ============================================================

print("\nLoading metric CSVs...\n")

ed_df = pd.read_csv(ED_CSV)
es_df = pd.read_csv(ES_CSV)

# ============================================================
# MERGE
# ============================================================

merged = ed_df.merge(
    es_df,
    on="Video",
    suffixes=("_ED", "_ES")
)

# ============================================================
# COMBINED DICE
# ============================================================

merged["Combined_Dice"] = (

    merged["Dice_ED"]

    +

    merged["Dice_ES"]

) / 2.0

# ============================================================
# SORT
# ============================================================

sorted_df = merged.sort_values(
    "Combined_Dice",
    ascending=False
).reset_index(drop=True)

# ============================================================
# BEST
# ============================================================

best_cases = sorted_df.head(5).copy()

best_cases["Category"] = "Best"

# ============================================================
# DIFFICULT
# ============================================================

difficult_cases = sorted_df.tail(5).copy()

difficult_cases["Category"] = "Difficult"

# ============================================================
# AVERAGE
# ============================================================

mean_dice = sorted_df["Combined_Dice"].mean()

sorted_df["Distance"] = np.abs(
    sorted_df["Combined_Dice"] - mean_dice
)

average_cases = sorted_df.sort_values(
    "Distance"
).head(5).copy()

average_cases["Category"] = "Average"

# ============================================================
# FINAL SELECTION
# ============================================================

final_cases = pd.concat([

    best_cases,
    average_cases,
    difficult_cases

])

final_cases = final_cases.drop_duplicates(
    subset=["Video"]
)

# ============================================================
# MAIN VISUALIZATION LOOP
# ============================================================

print("\nGenerating visual comparisons...\n")

for idx, row in final_cases.iterrows():

    video = row["Video"]

    category = row["Category"]

    print(f"Processing : {video}")

    # ========================================================
    # GT PATHS
    # ========================================================

    gt_video_dir = os.path.join(
        GT_DIR,
        video
    )

    gt_ed_file = [

        f for f in os.listdir(gt_video_dir)

        if f.startswith("ED")

    ][0]

    gt_es_file = [

        f for f in os.listdir(gt_video_dir)

        if f.startswith("ES")

    ][0]

    gt_ed_path = os.path.join(
        gt_video_dir,
        gt_ed_file
    )

    gt_es_path = os.path.join(
        gt_video_dir,
        gt_es_file
    )

    # ========================================================
    # PRED PATHS
    # ========================================================

    pred_ed_dir = os.path.join(
        PRED_DIR,
        video,
        "ED"
    )

    pred_es_dir = os.path.join(
        PRED_DIR,
        video,
        "ES"
    )

    pred_ed_file = [

        f for f in os.listdir(pred_ed_dir)

        if "mask" in f.lower()

    ][0]

    pred_es_file = [

        f for f in os.listdir(pred_es_dir)

        if "mask" in f.lower()

    ][0]

    pred_ed_path = os.path.join(
        pred_ed_dir,
        pred_ed_file
    )

    pred_es_path = os.path.join(
        pred_es_dir,
        pred_es_file
    )

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    gt_ed = cv2.imread(
        gt_ed_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_ed = cv2.imread(
        pred_ed_path,
        cv2.IMREAD_GRAYSCALE
    )

    gt_es = cv2.imread(
        gt_es_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_es = cv2.imread(
        pred_es_path,
        cv2.IMREAD_GRAYSCALE
    )

    # ========================================================
    # RESIZE FOR DISPLAY
    # ========================================================

    size = (256, 256)

    gt_ed = cv2.resize(gt_ed, size)
    pred_ed = cv2.resize(pred_ed, size)

    gt_es = cv2.resize(gt_es, size)
    pred_es = cv2.resize(pred_es, size)

    # ========================================================
    # CONVERT TO BGR
    # ========================================================

    gt_ed = cv2.cvtColor(gt_ed, cv2.COLOR_GRAY2BGR)
    pred_ed = cv2.cvtColor(pred_ed, cv2.COLOR_GRAY2BGR)

    gt_es = cv2.cvtColor(gt_es, cv2.COLOR_GRAY2BGR)
    pred_es = cv2.cvtColor(pred_es, cv2.COLOR_GRAY2BGR)

    # ========================================================
    # LABELS
    # ========================================================

    cv2.putText(
        gt_ed,
        "GT ED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        pred_ed,
        "PRED ED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        gt_es,
        "GT ES",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        pred_es,
        "PRED ES",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # ========================================================
    # COMBINE
    # ========================================================

    top_row = np.hstack([

        gt_ed,
        pred_ed

    ])

    bottom_row = np.hstack([

        gt_es,
        pred_es

    ])

    final_image = np.vstack([

        top_row,
        bottom_row

    ])

    # ========================================================
    # TITLE
    # ========================================================
    # ========================================================
# CREATE BOTTOM INFO PANEL
# ========================================================

    info_panel = np.zeros(
      (80, final_image.shape[1], 3),
       dtype=np.uint8
    )

# ========================================================
# TEXT
# ========================================================

    cv2.putText(

     info_panel,

     f"Category : {category}",

     (20, 30),

     cv2.FONT_HERSHEY_SIMPLEX,

      0.8,

    (255,255,255),

    2
)

    cv2.putText(

    info_panel,

    f"Video : {video}",

    (20, 65),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (255,255,255),

    2
)

# ========================================================
# APPEND BELOW IMAGE
# ========================================================

    final_image = np.vstack([

    final_image,

    info_panel

])
    # ========================================================
    # SAVE
    # ========================================================

    save_path = os.path.join(

        SAVE_DIR,

        f"{category}_{video}.png"
    )

    cv2.imwrite(
        save_path,
        final_image
    )

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("VISUAL COMPARISON GENERATION COMPLETE")
print("======================================")

print(f"\nSaved Directory:\n{SAVE_DIR}")
