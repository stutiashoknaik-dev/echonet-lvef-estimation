# ============================================================
# MODULE 2
# ANALYZE UNIFORM SAMPLING OVER ENTIRE TEST SET
# ============================================================
#
# PURPOSE
# -------
# Evaluate how well the current R(2+1)D uniform 32-frame
# sampling captures the Adaptive ED and ES phases
# across the complete test dataset.
#
# OUTPUTS
# --------
# 1. Per-video phase distance CSV
# 2. Summary statistics
# 3. ED histogram
# 4. ES histogram
# 5. ED vs ES Boxplot
# 6. Adaptive vs Uniform Scatter Plot
#
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUTS
# ============================================================

FRAMEWISE_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "multifeature_cardiac_analysis",
    "multifeature_framewise_analysis.csv"
)

FRAME_ROOT = os.path.join(
    BASE_PATH,
    "data",
    "frames",
    "test"
)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "uniform_sampling_dataset_analysis"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ============================================================
# OUTPUT FILES
# ============================================================

RESULT_CSV = os.path.join(
    SAVE_DIR,
    "phase_distance_results.csv"
)

SUMMARY_CSV = os.path.join(
    SAVE_DIR,
    "summary_statistics.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

NUM_SAMPLED_FRAMES = 32

# ============================================================
# LOAD FRAMEWISE ANALYSIS
# ============================================================

print("\n========================================")
print("Loading Framewise Analysis")
print("========================================")

framewise_df = pd.read_csv(
    FRAMEWISE_CSV
)

videos = sorted(
    framewise_df["Video"].unique()
)

print(f"\nTotal Test Videos : {len(videos)}")

print(f"Total Frame Records : {len(framewise_df)}")

# ============================================================
# RESULT STORAGE
# ============================================================

results = []

# ============================================================
# PROCESS EVERY VIDEO
# ============================================================

print("\n========================================")
print("Processing Test Videos")
print("========================================")

for video in tqdm(videos):

    # --------------------------------------------------------
    # CURRENT VIDEO
    # --------------------------------------------------------

    temp = framewise_df[
        framewise_df["Video"] == video
    ].copy()

    temp = temp.sort_values(
        "Frame"
    )

    frames = temp["Frame"].values

    ed_score = temp["ED_Score"].values

    es_score = temp["ES_Score"].values

    # --------------------------------------------------------
    # FRAME DIRECTORY
    # --------------------------------------------------------

    frame_folder = os.path.join(
        FRAME_ROOT,
        video
    )

    if not os.path.exists(frame_folder):
        continue

    frame_files = sorted(

        [
            f
            for f in os.listdir(frame_folder)
            if f.endswith(".png")
        ],

        key=lambda x: int(
            x.replace("frame_", "")
             .replace(".png", "")
        )

    )

    total_frames = len(frame_files)

    if total_frames == 0:
        continue

    # --------------------------------------------------------
    # EXACT R(2+1)D UNIFORM SAMPLING
    # --------------------------------------------------------

    sampled_frames = np.linspace(

        0,

        total_frames - 1,

        NUM_SAMPLED_FRAMES,

        dtype=int

    )

    # --------------------------------------------------------
    # ADAPTIVE ED / ES
    # --------------------------------------------------------

    ed_index = np.argmax(ed_score)

    es_index = np.argmax(es_score)

    adaptive_ed = int(
        frames[ed_index]
    )

    adaptive_es = int(
        frames[es_index]
    )

        # ========================================================
    # FIND NEAREST UNIFORM SAMPLED FRAME
    # ========================================================

    nearest_ed = sampled_frames[
        np.argmin(
            np.abs(
                sampled_frames - adaptive_ed
            )
        )
    ]

    nearest_es = sampled_frames[
        np.argmin(
            np.abs(
                sampled_frames - adaptive_es
            )
        )
    ]

    # ========================================================
    # COMPUTE DISTANCES
    #
    # Euclidean Distance in 1D
    # = Absolute Frame Difference
    # ========================================================

    ed_distance = int(
        abs(
            adaptive_ed - nearest_ed
        )
    )

    es_distance = int(
        abs(
            adaptive_es - nearest_es
        )
    )

    # ========================================================
    # PHASE CAPTURE
    #
    # We store multiple tolerances.
    # The decision will be made later.
    # ========================================================

    ed_capture_2 = ed_distance <= 2
    ed_capture_5 = ed_distance <= 5
    ed_capture_10 = ed_distance <= 10

    es_capture_2 = es_distance <= 2
    es_capture_5 = es_distance <= 5
    es_capture_10 = es_distance <= 10

    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({

        "Video": video,

        "Total_Frames": total_frames,

        # -----------------------------
        # Adaptive Phases
        # -----------------------------

        "Adaptive_ED": adaptive_ed,

        "Adaptive_ES": adaptive_es,

        # -----------------------------
        # Uniform Sampling
        # -----------------------------

        "Nearest_Uniform_ED": int(nearest_ed),

        "Nearest_Uniform_ES": int(nearest_es),

        # -----------------------------
        # Distances
        # -----------------------------

        "ED_Distance": ed_distance,

        "ES_Distance": es_distance,

        # -----------------------------
        # Capture Flags
        # -----------------------------

        "ED_<=2": ed_capture_2,

        "ED_<=5": ed_capture_5,

        "ED_<=10": ed_capture_10,

        "ES_<=2": es_capture_2,

        "ES_<=5": es_capture_5,

        "ES_<=10": es_capture_10

    })

    # ============================================================
# PART 3
# SAVE RESULTS + SUMMARY STATISTICS
# ============================================================

print("\n========================================")
print("Generating Summary Statistics")
print("========================================")

# ------------------------------------------------------------
# RESULTS DATAFRAME
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# SORT BY VIDEO
# ------------------------------------------------------------

results_df = results_df.sort_values(
    "Video"
)

# ------------------------------------------------------------
# SAVE DETAILED CSV
# ------------------------------------------------------------

results_df.to_csv(
    RESULT_CSV,
    index=False
)

print(f"\nDetailed Results Saved :")
print(RESULT_CSV)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = {}

# ------------------------------------------------------------
# ED DISTANCE
# ------------------------------------------------------------

summary["ED_Mean"] = results_df["ED_Distance"].mean()

summary["ED_Median"] = results_df["ED_Distance"].median()

summary["ED_STD"] = results_df["ED_Distance"].std()

summary["ED_Min"] = results_df["ED_Distance"].min()

summary["ED_Max"] = results_df["ED_Distance"].max()

summary["ED_Q1"] = results_df["ED_Distance"].quantile(0.25)

summary["ED_Q3"] = results_df["ED_Distance"].quantile(0.75)

# ------------------------------------------------------------
# ES DISTANCE
# ------------------------------------------------------------

summary["ES_Mean"] = results_df["ES_Distance"].mean()

summary["ES_Median"] = results_df["ES_Distance"].median()

summary["ES_STD"] = results_df["ES_Distance"].std()

summary["ES_Min"] = results_df["ES_Distance"].min()

summary["ES_Max"] = results_df["ES_Distance"].max()

summary["ES_Q1"] = results_df["ES_Distance"].quantile(0.25)

summary["ES_Q3"] = results_df["ES_Distance"].quantile(0.75)

# ============================================================
# CAPTURE RATES
# ============================================================

summary["ED_Capture_2(%)"] = (
    results_df["ED_<=2"].mean() * 100
)

summary["ED_Capture_5(%)"] = (
    results_df["ED_<=5"].mean() * 100
)

summary["ED_Capture_10(%)"] = (
    results_df["ED_<=10"].mean() * 100
)

summary["ES_Capture_2(%)"] = (
    results_df["ES_<=2"].mean() * 100
)

summary["ES_Capture_5(%)"] = (
    results_df["ES_<=5"].mean() * 100
)

summary["ES_Capture_10(%)"] = (
    results_df["ES_<=10"].mean() * 100
)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(

    summary,

    index=[0]

)

summary_df.to_csv(

    SUMMARY_CSV,

    index=False

)

print(f"\nSummary Saved :")
print(SUMMARY_CSV)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("SUMMARY")
print("========================================")

print("\nED DISTANCE")

print("------------------------------")

print(f"Mean     : {summary['ED_Mean']:.2f}")

print(f"Median   : {summary['ED_Median']:.2f}")

print(f"Std      : {summary['ED_STD']:.2f}")

print(f"Minimum  : {summary['ED_Min']}")

print(f"Maximum  : {summary['ED_Max']}")

print(f"Q1       : {summary['ED_Q1']:.2f}")

print(f"Q3       : {summary['ED_Q3']:.2f}")

print("\nES DISTANCE")

print("------------------------------")

print(f"Mean     : {summary['ES_Mean']:.2f}")

print(f"Median   : {summary['ES_Median']:.2f}")

print(f"Std      : {summary['ES_STD']:.2f}")

print(f"Minimum  : {summary['ES_Min']}")

print(f"Maximum  : {summary['ES_Max']}")

print(f"Q1       : {summary['ES_Q1']:.2f}")

print(f"Q3       : {summary['ES_Q3']:.2f}")

print("\nCAPTURE RATE")

print("------------------------------")

print(f"ED ≤2 Frames  : {summary['ED_Capture_2(%)']:.2f}%")

print(f"ED ≤5 Frames  : {summary['ED_Capture_5(%)']:.2f}%")

print(f"ED ≤10 Frames : {summary['ED_Capture_10(%)']:.2f}%")

print()

print(f"ES ≤2 Frames  : {summary['ES_Capture_2(%)']:.2f}%")

print(f"ES ≤5 Frames  : {summary['ES_Capture_5(%)']:.2f}%")

print(f"ES ≤10 Frames : {summary['ES_Capture_10(%)']:.2f}%")

print("\n========================================")
print("PART 3 COMPLETED")
print("========================================")
