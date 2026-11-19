# ============================================================
# RESAMPLE GT + PRED CONTOURS TO FIXED 100 POINTS
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT CSV FILES
# ============================================================

GT_CSV = os.path.join(
    BASE_PATH,
    "outputs/val_contour_gt",
    "val_volume_tracings.csv"
)

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
    "outputs/resampled_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE PATHS
# ============================================================

GT_SAVE = os.path.join(
    SAVE_DIR,
    "GT_Resampled_100.csv"
)

PRED_SAVE = os.path.join(
    SAVE_DIR,
    "PRED_Resampled_100.csv"
)

# ============================================================
# NUMBER OF RESAMPLED POINTS
# ============================================================

NUM_POINTS = 100

# ============================================================
# RESAMPLING FUNCTION
# ============================================================

def resample_contour(points, num_points=100):

    """
    points:
        Nx2 array
    """

    if len(points) < 2:
        return None

    # --------------------------------------------------------
    # DISTANCES BETWEEN POINTS
    # --------------------------------------------------------

    distances = np.sqrt(
        np.sum(
            np.diff(points, axis=0) ** 2,
            axis=1
        )
    )

    distances = np.insert(
        np.cumsum(distances),
        0,
        0
    )

    # --------------------------------------------------------
    # NORMALIZED DISTANCE
    # --------------------------------------------------------

    if distances[-1] == 0:
        return None

    normalized = distances / distances[-1]

    # --------------------------------------------------------
    # INTERPOLATION
    # --------------------------------------------------------

    interp_x = interp1d(
        normalized,
        points[:, 0],
        kind='linear'
    )

    interp_y = interp1d(
        normalized,
        points[:, 1],
        kind='linear'
    )

    # --------------------------------------------------------
    # NEW SAMPLE POSITIONS
    # --------------------------------------------------------

    alpha = np.linspace(
        0,
        1,
        num_points
    )

    x_new = interp_x(alpha)
    y_new = interp_y(alpha)

    resampled = np.stack(
        [x_new, y_new],
        axis=1
    )

    return resampled

# ============================================================
# PROCESS FUNCTION
# ============================================================

def process_csv(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    results = []

    # --------------------------------------------------------
    # GROUP BY VIDEO + FRAME
    # --------------------------------------------------------

    grouped = df.groupby(
        ["FileName", "Frame"]
    )

    for (video, frame), group in grouped:

        # ----------------------------------------------------
        # LEFT SIDE
        # ----------------------------------------------------

        left_points = group[
            ["X1", "Y1"]
        ].values

        # ----------------------------------------------------
        # RIGHT SIDE
        # ----------------------------------------------------

        right_points = group[
            ["X2", "Y2"]
        ].values

        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        left_resampled = resample_contour(
            left_points,
            NUM_POINTS
        )

        right_resampled = resample_contour(
            right_points,
            NUM_POINTS
        )

        if left_resampled is None:
            continue

        if right_resampled is None:
            continue

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        for i in range(NUM_POINTS):

            x1, y1 = left_resampled[i]
            x2, y2 = right_resampled[i]

            results.append({

                "FileName": video,

                "X1": float(x1),
                "Y1": float(y1),

                "X2": float(x2),
                "Y2": float(y2),

                "Frame": int(frame)
            })

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    final_df = pd.DataFrame(results)

    final_df.to_csv(
        output_csv,
        index=False
    )

    print(f"\nSaved : {output_csv}")
    print(f"Total Points : {len(final_df)}")

# ============================================================
# RUN GT
# ============================================================

print("\n======================================")
print("RESAMPLING GT CONTOURS")
print("======================================")

process_csv(
    GT_CSV,
    GT_SAVE
)

# ============================================================
# RUN PRED
# ============================================================

print("\n======================================")
print("RESAMPLING PRED CONTOURS")
print("======================================")

process_csv(
    PRED_CSV,
    PRED_SAVE
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("CONTOUR RESAMPLING COMPLETE")
print("======================================")
