# ============================================================
# IMPROVED PREDICTED VOLUMETRACING EXTRACTION
# ============================================================
#
# GOAL:
# -----
# Extract EchoNet-style contour points from
# predicted validation masks
#
# WITHOUT modifying GT format
#
# IMPORTANT FIXES:
# ----------------
# 1. Preserve OpenCV contour order
# 2. NO sorting by Y
# 3. Split contour using traversal order
# 4. Reverse one wall for anatomical consistency
#
# OUTPUT FORMAT:
# --------------
# FileName | X1 | Y1 | X2 | Y2 | Frame
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
# INPUT VALIDATION MASK DIRECTORY
# ============================================================

VAL_MASK_DIR = os.path.join(
    BASE_PATH,
    "data/masks/val"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/improved_validation_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE CSV
# ============================================================

SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Improved_Predicted_Val_VolumeTracings.csv"
)

# ============================================================
# STORAGE
# ============================================================

results = []

# ============================================================
# VIDEO LOOP
# ============================================================

videos = os.listdir(VAL_MASK_DIR)

print("\nStarting improved contour extraction...\n")

for video in videos:

    print(f"Processing : {video}")

    video_dir = os.path.join(
        VAL_MASK_DIR,
        video
    )

    if not os.path.isdir(video_dir):
        continue

    # ========================================================
    # MASK FILES
    # ========================================================

    mask_files = [

        f for f in os.listdir(video_dir)

        if f.endswith(".png")

    ]

    # ========================================================
    # MASK LOOP
    # ========================================================

    for mask_file in mask_files:

        mask_path = os.path.join(
            video_dir,
            mask_file
        )

        # ====================================================
        # EXTRACT FRAME NUMBER
        # ====================================================

        try:

            frame_num = int(
                mask_file.split("_")[1]
            )

        except:
            continue

        # ====================================================
        # LOAD MASK
        # ====================================================

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        # ====================================================
        # BINARY MASK
        # ====================================================

        _, binary = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # ====================================================
        # FIND CONTOURS
        # ====================================================

        contours, _ = cv2.findContours(

            binary,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        # ====================================================
        # LARGEST CONTOUR
        # ====================================================

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour = contour.squeeze()

        if len(contour.shape) != 2:
            continue

        # ====================================================
        # IMPORTANT:
        # KEEP ORIGINAL OPENCV ORDER
        # DO NOT SORT
        # ====================================================

        # ====================================================
        # SPLIT CONTOUR INTO 2 WALLS
        # ====================================================

        n = len(contour)

        half = n // 2

        left_pts = contour[:half]

        right_pts = contour[half:]

        # ====================================================
        # REVERSE RIGHT WALL
        # ====================================================

        right_pts = right_pts[::-1]

        # ====================================================
        # MATCH POINT COUNTS
        # ====================================================

        n_points = min(
            len(left_pts),
            len(right_pts)
        )

        if n_points < 5:
            continue

        left_pts = left_pts[:n_points]

        right_pts = right_pts[:n_points]

        # ====================================================
        # STORE POINTS
        # ====================================================

        for i in range(n_points):

            x1, y1 = left_pts[i]

            x2, y2 = right_pts[i]

            results.append({

                "FileName": video + ".avi",

                "X1": float(x1),
                "Y1": float(y1),

                "X2": float(x2),
                "Y2": float(y2),

                "Frame": int(frame_num)
            })

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    SAVE_CSV,
    index=False
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("IMPROVED CONTOUR EXTRACTION COMPLETE")
print("======================================")

print(f"\nTotal Points Extracted : {len(df)}")

print(f"\nSaved CSV:\n{SAVE_CSV}")
