# ============================================================
# GENERATE FRAMEWISE LV AREA + SMOOTHED LV AREA
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT
# ============================================================

MASK_ROOT = os.path.join(
    BASE_PATH,
    "outputs",
    "Test_predictions_edgeaware/masks"
)

# ============================================================
# OUTPUT
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "smoothed_area_analysis"
)

os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    "smoothed_framewise_lv_area.csv"
)

RAW_OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    "framewise_lv_area.csv"
)

# ============================================================
# SMOOTHING PARAMETERS
# ============================================================

WINDOW_LENGTH = 11
POLYORDER = 2

# ============================================================
# STORAGE
# ============================================================

results = []

# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted([
    d for d in os.listdir(MASK_ROOT)
    if os.path.isdir(os.path.join(MASK_ROOT, d))
])

print("Total videos:", len(videos))

# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video in videos:

    video_dir = os.path.join(
        MASK_ROOT,
        video
    )

    frame_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    def frame_number(filename):
        digits = "".join(
            c for c in os.path.splitext(filename)[0]
            if c.isdigit()
        )
        return int(digits) if digits else -1

    frame_files = sorted(
        frame_files,
        key=frame_number
    )

    areas = []
    frames = []

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    for frame_file in frame_files:

        frame_path = os.path.join(
            video_dir,
            frame_file
        )

        mask = cv2.imread(
            frame_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        # ----------------------------------------------------
        # BINARY MASK
        # ----------------------------------------------------

        binary = mask > 0

        # ----------------------------------------------------
        # LV AREA = NUMBER OF MASK PIXELS
        # ----------------------------------------------------

        area = int(np.sum(binary))

        frame = frame_number(frame_file)

        frames.append(frame)
        areas.append(area)

    if len(areas) == 0:
        continue

    # ========================================================
    # SMOOTH AREA
    # ========================================================

    area_array = np.asarray(
        areas,
        dtype=np.float32
    )

    # Savitzky-Golay requires an odd window
    # smaller than number of frames

    if len(area_array) >= 5:

        window = min(
            WINDOW_LENGTH,
            len(area_array)
        )

        if window % 2 == 0:
            window -= 1

        if window >= 5:

            smoothed_area = savgol_filter(
                area_array,
                window_length=window,
                polyorder=min(
                    POLYORDER,
                    window - 1
                )
            )

        else:
            smoothed_area = area_array.copy()

    else:

        smoothed_area = area_array.copy()

    # ========================================================
    # STORE
    # ========================================================

    for frame, raw_area, smooth_area in zip(
        frames,
        areas,
        smoothed_area
    ):

        results.append({

            "Video": video,

            "Frame": int(frame),

            "LV_Area": float(raw_area),

            "Smoothed_Area": float(smooth_area)
        })

# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(results)

df = df.sort_values(
    ["Video", "Frame"]
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

df[
    ["Video", "Frame", "LV_Area"]
].to_csv(
    RAW_OUTPUT_CSV,
    index=False
)

print()
print("============================================")
print("SMOOTHED LV AREA ANALYSIS COMPLETE")
print("============================================")
print("Saved:")
print(OUTPUT_CSV)
print(RAW_OUTPUT_CSV)
print()
print("Rows:", len(df))
