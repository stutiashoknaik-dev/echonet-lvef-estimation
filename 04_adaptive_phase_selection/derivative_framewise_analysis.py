# ============================================================
# DERIVATIVE-BASED CARDIAC PHASE ANALYSIS
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT
# ============================================================

AREA_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "smoothed_area_analysis",
    "smoothed_framewise_lv_area.csv"
)

# ============================================================
# OUTPUT
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "derivative_phase_detection"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    "derivative_framewise_analysis.csv"
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(AREA_CSV)

print("Loaded:", AREA_CSV)

# ============================================================
# STORAGE
# ============================================================

results = []

# ============================================================
# PROCESS VIDEO BY VIDEO
# ============================================================

for video in df["Video"].unique():

    temp = df[
        df["Video"] == video
    ].copy()

    temp = temp.sort_values(
        "Frame"
    )

    frames = temp["Frame"].to_numpy()

    area = temp["Smoothed_Area"].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # FIRST DERIVATIVE
    # --------------------------------------------------------

    if len(area) >= 2:

        derivative = np.gradient(
            area
        )

    else:

        derivative = np.zeros_like(
            area
        )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    for frame, d in zip(
        frames,
        derivative
    ):

        results.append({

            "Video": video,

            "Frame": int(frame),

            "Derivative": float(d)
        })

# ============================================================
# SAVE
# ============================================================

out_df = pd.DataFrame(
    results
)

out_df = out_df.sort_values(
    ["Video", "Frame"]
)

out_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("============================================")
print("DERIVATIVE ANALYSIS COMPLETE")
print("============================================")
print("Saved:")
print(OUTPUT_CSV)
print("Rows:", len(out_df))
