
# ============================================================
# IMPROVED TEST CONTOUR EXTRACTION
# SEPARATE ED / ES CSV GENERATION
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
# INPUTS
# ============================================================

MASK_DIR = os.path.join(
    BASE_PATH,
    "outputs/Test_predictions_edgeaware/masks"
)

FRAME_CSV = os.path.join(
    BASE_PATH,
    "outputs/mask_only_validation1",
    "predicted_frames_ed_es_70_30.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE CSV FILES
# ============================================================

ED_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ED_Test_VolumeTracings_70_30.csv"
)

ES_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ES_Test_VolumeTracings_70_30.csv"
)

# ============================================================
# STORAGE
# ============================================================

print("\nLoading Predicted Frame CSV...\n")

frame_df = pd.read_csv(FRAME_CSV)

print(f"Videos Found : {len(frame_df)}")

ed_results = []
es_results = []

# ============================================================
# LOOP THROUGH ALL VIDEOS
# ============================================================

for _, row in frame_df.iterrows():

    video_name = str(row["Video"]).strip()

    print(f"\nProcessing : {video_name}")

    phase_frames = [

        ("ED", int(row["Pred_ED"])),

        ("ES", int(row["Pred_ES"]))

    ]

    for phase_name, frame_num in phase_frames:

        mask_path = os.path.join(

            MASK_DIR,

            video_name,

            f"frame_{frame_num}.png"

        )

        if not os.path.exists(mask_path):

            print(f"Missing : {mask_path}")

            continue

        mask = cv2.imread(

            mask_path,

            cv2.IMREAD_GRAYSCALE

        )

        if mask is None:
            continue

        # =====================================================
        # BINARY MASK
        # =====================================================

        _, binary = cv2.threshold(

            mask,

            127,

            255,

            cv2.THRESH_BINARY

        )

        # =====================================================
        # FIND CONTOURS
        # =====================================================

        contours, _ = cv2.findContours(

            binary,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_NONE

        )

        if len(contours) == 0:
            continue

        contour = max(

            contours,

            key=cv2.contourArea

        )

        contour = contour.squeeze()

        if len(contour.shape) != 2:
            continue

        n = len(contour)

        half = n // 2

        left_pts = contour[:half]

        right_pts = contour[half:]

        right_pts = right_pts[::-1]

        n_points = min(

            len(left_pts),

            len(right_pts)

        )

        if n_points < 5:
            continue

        left_pts = left_pts[:n_points]

        right_pts = right_pts[:n_points]

        for i in range(n_points):

            x1, y1 = left_pts[i]

            x2, y2 = right_pts[i]

            row_dict = {

                "FileName": video_name + ".avi",

                "X1": float(x1),
                "Y1": float(y1),

                "X2": float(x2),
                "Y2": float(y2),

                "Frame": int(frame_num)

            }

            if phase_name == "ED":

                ed_results.append(row_dict)

            else:

                es_results.append(row_dict)
# ============================================================
# CREATE DATAFRAMES
# ============================================================

ed_df = pd.DataFrame(ed_results)
es_df = pd.DataFrame(es_results)

# ============================================================
# SAVE CSV FILES
# ============================================================

ed_df.to_csv(
    ED_SAVE_CSV,
    index=False
)

es_df.to_csv(
    ES_SAVE_CSV,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("SEPARATE ED / ES CONTOUR EXTRACTION COMPLETE")
print("======================================")

print(f"\nED Points Extracted : {len(ed_df)}")
print(f"ES Points Extracted : {len(es_df)}")

print(f"\nED CSV Saved:\n{ED_SAVE_CSV}")
print(f"\nES CSV Saved:\n{ES_SAVE_CSV}")
