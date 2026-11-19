# ============================================================
# VISUALIZE UNIFORM SAMPLING vs ADAPTIVE PHASES
# ============================================================
#
# PURPOSE
# -------
# Visualize whether the 32 uniformly sampled frames used by the
# R(2+1)D model overlap with the adaptive ED and ES phases.
#
# This is ONLY a visualization experiment.
#
# No retraining.
# No model inference.
#
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT CSV
# ============================================================

FRAMEWISE_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "multifeature_cardiac_analysis",
    "multifeature_framewise_analysis.csv"
)

# ============================================================
# FRAME DIRECTORY
# ============================================================

FRAME_ROOT = os.path.join(
    BASE_PATH,
    "data",
    "frames",
    "test"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "uniform_vs_adaptive_visualization"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ============================================================
# VIDEO TO VISUALIZE
# ============================================================

VIDEO_NAME = "0X1A3D565B371DC573"

NUM_SAMPLED_FRAMES = 32

# ============================================================
# LOAD FRAMEWISE ANALYSIS
# ============================================================

print("\nLoading framewise analysis...")

framewise_df = pd.read_csv(
    FRAMEWISE_CSV
)

video_df = framewise_df[
    framewise_df["Video"] == VIDEO_NAME
].copy()

video_df = video_df.sort_values(
    "Frame"
)

print(f"Frames in CSV : {len(video_df)}")

# ============================================================
# LOAD FRAME FOLDER
# ============================================================

frame_folder = os.path.join(
    FRAME_ROOT,
    VIDEO_NAME
)

if not os.path.exists(frame_folder):

    raise FileNotFoundError(
        f"Missing folder:\n{frame_folder}"
    )

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

print(f"Extracted Frames : {len(frame_files)}")

# ============================================================
# REPRODUCE EXACT UNIFORM SAMPLING
# ============================================================

sampled_frames = np.linspace(

    0,

    len(frame_files)-1,

    NUM_SAMPLED_FRAMES,

    dtype=int

)

print("\nUniform Sampled Frames\n")

print(sampled_frames)

# ============================================================
# LOAD SIGNALS
# ============================================================

frames = video_df["Frame"].values

ed_score = video_df["ED_Score"].values

es_score = video_df["ES_Score"].values

print("\nSignals Loaded")

print(f"ED Score : {len(ed_score)}")

print(f"ES Score : {len(es_score)}")

# ============================================================
# ADAPTIVE ED & ES FRAME DETECTION
# ============================================================

print("\n========================================")
print("Adaptive Phase Detection")
print("========================================")

# ------------------------------------------------------------
# Adaptive ED Frame
# ------------------------------------------------------------

ed_index = np.argmax(ed_score)

ed_frame = int(frames[ed_index])

ed_value = float(ed_score[ed_index])

# ------------------------------------------------------------
# Adaptive ES Frame
# ------------------------------------------------------------

es_index = np.argmax(es_score)

es_frame = int(frames[es_index])

es_value = float(es_score[es_index])

print(f"\nAdaptive ED Frame : {ed_frame}")
print(f"ED Score          : {ed_value:.4f}")

print(f"\nAdaptive ES Frame : {es_frame}")
print(f"ES Score          : {es_value:.4f}")

# ============================================================
# FIND NEAREST UNIFORM SAMPLED FRAME
# ============================================================

nearest_ed = sampled_frames[
    np.argmin(
        np.abs(sampled_frames - ed_frame)
    )
]

nearest_es = sampled_frames[
    np.argmin(
        np.abs(sampled_frames - es_frame)
    )
]

ed_distance = abs(
    nearest_ed - ed_frame
)

es_distance = abs(
    nearest_es - es_frame
)

print("\n========================================")
print("Nearest Uniform Sample")
print("========================================")

print(f"\nNearest Sample to ED : {nearest_ed}")
print(f"Distance from ED     : {ed_distance} frame(s)")

print(f"\nNearest Sample to ES : {nearest_es}")
print(f"Distance from ES     : {es_distance} frame(s)")

# ============================================================
# SIMPLE PHASE COVERAGE CHECK
# ============================================================

TOLERANCE = 2

ed_captured = ed_distance <= TOLERANCE

es_captured = es_distance <= TOLERANCE

print("\n========================================")
print("Phase Coverage")
print("========================================")

print(f"\nED Captured : {ed_captured}")

print(f"ES Captured : {es_captured}")



# ============================================================
# VISUALIZATION
# ============================================================

print("\nGenerating Visualization...")

# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize(signal):

    signal = np.asarray(signal)

    return (

        signal - signal.min()

    ) / (

        signal.max() - signal.min() + 1e-8

    )

ed_plot = normalize(ed_score)

es_plot = normalize(es_score)

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(

    3,

    1,

    figsize=(18,10),

    sharex=True

)

# ============================================================
# TIMELINE
# ============================================================

timeline = axes[0]

timeline.set_title(

    f"{VIDEO_NAME}\nUniform Sampling vs Adaptive Phases",

    fontsize=15,

    weight="bold"

)

timeline.hlines(

    1,

    0,

    len(frames)-1,

    color="black",

    linewidth=2

)

# Uniform samples

for s in sampled_frames:

    timeline.vlines(

        s,

        0.8,

        1.2,

        color="green",

        linewidth=2

    )

# Adaptive ED

timeline.scatter(

    ed_frame,

    1,

    color="red",

    s=120,

    label="Adaptive ED",

    zorder=5

)

# Adaptive ES

timeline.scatter(

    es_frame,

    1,

    color="blue",

    s=120,

    label="Adaptive ES",

    zorder=5

)

timeline.set_ylim(

    0.6,

    1.4

)

timeline.set_yticks([])

timeline.legend(

    loc="upper right"

)

# ============================================================
# ED SCORE
# ============================================================

axes[1].plot(

    frames,

    ed_plot,

    color="red",

    linewidth=2,

    label="ED Score"

)

axes[1].scatter(

    ed_frame,

    ed_plot[ed_index],

    color="black",

    s=80,

    zorder=5

)

for s in sampled_frames:

    axes[1].axvline(

        s,

        color="green",

        alpha=0.25

    )

axes[1].set_ylabel("Normalized ED")

axes[1].legend()

# ============================================================
# ES SCORE
# ============================================================

axes[2].plot(

    frames,

    es_plot,

    color="blue",

    linewidth=2,

    label="ES Score"

)

axes[2].scatter(

    es_frame,

    es_plot[es_index],

    color="black",

    s=80,

    zorder=5

)

for s in sampled_frames:

    axes[2].axvline(

        s,

        color="green",

        alpha=0.25

    )

axes[2].set_ylabel("Normalized ES")

axes[2].set_xlabel("Frame Number")

axes[2].legend()

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

save_path = os.path.join(

    SAVE_DIR,

    f"{VIDEO_NAME}_uniform_vs_adaptive.png"

)

plt.savefig(

    save_path,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

print("\nVisualization Saved")

print(save_path)
