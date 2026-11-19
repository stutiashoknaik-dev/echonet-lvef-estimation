# ============================================================
# RESAMPLE TEST PREDICTED CONTOURS TO FIXED 100 POINTS
# Compatible with FusionDataset
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
# INPUT FILES
# ============================================================

ED_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "test_predicted_contours",
    "Predicted_ED_Test_VolumeTracings.csv"
)

ES_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "test_predicted_contours",
    "Predicted_ES_Test_VolumeTracings.csv"
)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "test_predicted_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

ED_SAVE = os.path.join(
    SAVE_DIR,
    "Predicted_ED_Test_Contours_100Pts.csv"
)

ES_SAVE = os.path.join(
    SAVE_DIR,
    "Predicted_ES_Test_Contours_100Pts.csv"
)

# ============================================================
# CONFIG
# ============================================================

NUM_POINTS = 100

# ============================================================
# RESAMPLING FUNCTION
# ============================================================

def resample_contour(points, num_points=100):

    if len(points) < 2:
        return None

    # Remove duplicate consecutive points
    diff = np.diff(points, axis=0)
    keep = np.any(diff != 0, axis=1)

    keep = np.insert(keep, 0, True)

    points = points[keep]

    if len(points) < 2:
        return None

    # Arc length

    dist = np.sqrt(
        np.sum(
            np.diff(points, axis=0) ** 2,
            axis=1
        )
    )

    dist = np.insert(
        np.cumsum(dist),
        0,
        0
    )

    if dist[-1] == 0:
        return None

    dist = dist / dist[-1]

    interp_x = interp1d(
        dist,
        points[:, 0],
        kind="linear"
    )

    interp_y = interp1d(
        dist,
        points[:, 1],
        kind="linear"
    )

    alpha = np.linspace(
        0,
        1,
        num_points
    )

    x = interp_x(alpha)
    y = interp_y(alpha)

    return np.stack(
        [x, y],
        axis=1
    )

# ============================================================
# PROCESS CSV
# ============================================================

def process_csv(input_csv, output_csv):

    print("\nLoading:")
    print(input_csv)

    df = pd.read_csv(input_csv)

    results = []

    groups = list(
        df.groupby(
            ["FileName", "Frame"]
        )
    )

    print("\nContours Found :", len(groups))

    for idx, ((video, frame), group) in enumerate(groups):

        if idx % 250 == 0:
            print(f"{idx}/{len(groups)}")

        left = group[
            ["X1", "Y1"]
        ].values.astype(np.float32)

        right = group[
            ["X2", "Y2"]
        ].values.astype(np.float32)

        left = resample_contour(
            left,
            NUM_POINTS
        )

        right = resample_contour(
            right,
            NUM_POINTS
        )

        if left is None:
            continue

        if right is None:
            continue

        for pid in range(NUM_POINTS):

            results.append({

                "FileName": video,

                "PointID": pid,

                "X1": float(left[pid, 0]),

                "Y1": float(left[pid, 1]),

                "X2": float(right[pid, 0]),

                "Y2": float(right[pid, 1]),

                "Frame": int(frame)

            })

    final_df = pd.DataFrame(results)

    final_df.to_csv(
        output_csv,
        index=False
    )

    print("\nSaved:")
    print(output_csv)

    print("Rows :", len(final_df))

    print("Columns:")
    print(final_df.columns.tolist())

    print("\nRows Per Video (Should be 100):")

    print(
        final_df.groupby(
            "FileName"
        ).size().value_counts().head()
    )

# ============================================================
# RUN ED
# ============================================================

print("\n========================================")
print("RESAMPLING ED TEST CONTOURS")
print("========================================")

process_csv(
    ED_CSV,
    ED_SAVE
)

# ============================================================
# RUN ES
# ============================================================

print("\n========================================")
print("RESAMPLING ES TEST CONTOURS")
print("========================================")

process_csv(
    ES_CSV,
    ES_SAVE
)

# ============================================================
# FINAL
# ============================================================

print("\n========================================")
print("TEST CONTOUR RESAMPLING COMPLETE")
print("========================================")

print("\nSaved Files:")

print(ED_SAVE)

print(ES_SAVE)
