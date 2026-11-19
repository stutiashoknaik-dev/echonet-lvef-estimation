# ============================================================
# MULTI-FEATURE CARDIAC STATE ANALYSIS
# ============================================================

import os
import pandas as pd
import numpy as np

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT CSV FILES
# ============================================================

AREA_CSV = os.path.join(
    BASE_PATH,
    "outputs/smoothed_area_analysis/smoothed_framewise_lv_area.csv"
)

DERIVATIVE_CSV = os.path.join(
    BASE_PATH,
    "outputs/derivative_phase_detection/derivative_framewise_analysis.csv"
)

OPTICALFLOW_CSV = os.path.join(
    BASE_PATH,
    "outputs/opticalflow_phase_detection/opticalflow_framewise_analysis.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/multifeature_cardiac_analysis"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE FILES
# ============================================================

FRAMEWISE_SAVE = os.path.join(
    SAVE_DIR,"multifeature_framewise_analysis.csv"
)

STATE_SAVE = os.path.join(
    SAVE_DIR,
    "stable_cardiac_regions.csv"
)

# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading CSV files...\n")

area_df = pd.read_csv(AREA_CSV)

derivative_df = pd.read_csv(DERIVATIVE_CSV)

motion_df = pd.read_csv(OPTICALFLOW_CSV)

print("Area CSV Loaded")
print("Derivative CSV Loaded")
print("Optical Flow CSV Loaded")

# ============================================================
# MERGE ALL FEATURES
# ============================================================

merged = area_df.merge(

    derivative_df[
        ["Video", "Frame", "Derivative"]
    ],

    on=["Video", "Frame"]
)

merged = merged.merge(

    motion_df[
        ["Video", "Frame", "Motion_Magnitude"]
    ],

    on=["Video", "Frame"]
)

# ============================================================
# STORAGE
# ============================================================

final_results = []

region_results = []

# ============================================================
# VIDEO LOOP
# ============================================================

videos = merged["Video"].unique()

for video in videos:

    print(f"Processing : {video}")

    temp = merged[
        merged["Video"] == video
    ].copy()

    temp = temp.sort_values("Frame")

    # ========================================================
    # FEATURES
    # ========================================================

    frames = temp["Frame"].values

    area = temp["Smoothed_Area"].values

    derivative = temp["Derivative"].values

    motion = temp["Motion_Magnitude"].values

    # ========================================================
    # NORMALIZATION
    # ========================================================

    area_norm = (
        area - np.min(area)
    ) / (
        np.max(area) -
        np.min(area) + 1e-8
    )

    motion_norm = (
        motion - np.min(motion)
    ) / (
        np.max(motion) -
        np.min(motion) + 1e-8
    )

    derivative_norm = (
        np.abs(derivative) -
        np.min(np.abs(derivative))
    ) / (
        np.max(np.abs(derivative)) -
        np.min(np.abs(derivative)) + 1e-8
    )

    # ========================================================
    # STABILITY SCORE
    # ========================================================

    stability = 1 - (
        0.5 * motion_norm +
        0.5 * derivative_norm
    )

    # ========================================================
    # ED SCORE
    # ========================================================

    ed_score = (
        0.7 * area_norm +
        0.3 * stability
    )

    # ========================================================
    # ES SCORE
    # ========================================================

    es_score = (
        0.7 * (1 - area_norm) +
        0.3 * stability
    )

    # ========================================================
    # TOP ED REGION
    # ========================================================

    top_ed_idx = np.argsort(ed_score)[-5:]

    # ========================================================
    # TOP ES REGION
    # ========================================================

    top_es_idx = np.argsort(es_score)[-5:]

    # ========================================================
    # STORE FRAMEWISE
    # ========================================================

    for i in range(len(frames)):

        final_results.append({

            "Video": video,

            "Frame": int(frames[i]),

            "Area": float(area[i]),

            "Derivative": float(derivative[i]),

            "Motion": float(motion[i]),

            "Stability": float(stability[i]),

            "ED_Score": float(ed_score[i]),

            "ES_Score": float(es_score[i])
        })

    # ========================================================
    # STORE ED REGIONS
    # ========================================================

    for idx in top_ed_idx:

        region_results.append({

            "Video": video,

            "Type": "ED",

            "Frame": int(frames[idx]),

            "Score": float(ed_score[idx])
        })

    # ========================================================
    # STORE ES REGIONS
    # ========================================================

    for idx in top_es_idx:

        region_results.append({

            "Video": video,

            "Type": "ES",

            "Frame": int(frames[idx]),

            "Score": float(es_score[idx])
        })

# ============================================================
# SAVE FRAMEWISE CSV
# ============================================================

final_df = pd.DataFrame(
    final_results
)

final_df.to_csv(
    FRAMEWISE_SAVE,
    index=False
)

# ============================================================
# SAVE REGIONS CSV
# ============================================================

region_df = pd.DataFrame(
    region_results
)

region_df.to_csv(
    STATE_SAVE,
    index=False
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("MULTI-FEATURE CARDIAC ANALYSIS COMPLETE")
print("======================================")

print(f"\nTotal Videos : {len(videos)}")

print("\n✅ Framewise feature analysis saved")
print("✅ Stable cardiac candidate regions saved")

print("\nSaved Files:")

print(FRAMEWISE_SAVE)
print(STATE_SAVE)
