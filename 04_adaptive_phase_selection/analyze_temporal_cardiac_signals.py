
"""
analyze_temporal_cardiac_signals.py

Module 1
--------
Visualizes physiological signals extracted from each cardiac video.

This module:
1. Loads multifeature_framewise_analysis.csv
2. Loads stable_cardiac_regions.csv
3. Plots:
   - Area
   - Derivative
   - Motion
   - Stability
   - ED Score
   - ES Score
4. Marks adaptive ED / ES frames
5. Saves publication-quality figures

Author: ChatGPT + Stuti
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION
# ==========================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FRAMEWISE_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "multifeature_cardiac_analysis",
    "multifeature_framewise_analysis.csv"
)

REGION_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "multifeature_cardiac_analysis",
    "stable_cardiac_regions.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "temporal_signal_analysis"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

frame_df = pd.read_csv(FRAMEWISE_CSV)
region_df = pd.read_csv(REGION_CSV)

videos = sorted(frame_df["Video"].unique())

print(f"Videos found : {len(videos)}")

# ==========================
# FUNCTION
# ==========================

def plot_video(video_name):

    temp = frame_df[
        frame_df["Video"] == video_name
    ].sort_values("Frame")

    if len(temp) == 0:
        print(f"No data for {video_name}")
        return

    frames = temp["Frame"]

    ed_frames = region_df[
        (region_df["Video"] == video_name) &
        (region_df["Type"] == "ED")
    ]["Frame"].tolist()

    es_frames = region_df[
        (region_df["Video"] == video_name) &
        (region_df["Type"] == "ES")
    ]["Frame"].tolist()

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16,18),
        sharex=True
    )

    signals = [
        ("Area","Area"),
        ("Derivative","Derivative"),
        ("Motion","Motion"),
        ("Stability","Stability"),
        ("ED_Score","ED Score"),
        ("ES_Score","ES Score")
    ]

    for ax,(col,title) in zip(axes,signals):

        ax.plot(
            frames,
            temp[col],
            linewidth=2
        )

        for x in ed_frames:
            ax.axvline(
                x,
                color="green",
                linestyle="--",
                alpha=0.8
            )

        for x in es_frames:
            ax.axvline(
                x,
                color="red",
                linestyle="--",
                alpha=0.8
            )

        ax.set_ylabel(title)
        ax.grid(True,alpha=0.3)

    axes[-1].set_xlabel("Frame Number")

    fig.suptitle(
        f"Temporal Cardiac Signal Analysis\n{video_name}",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    save_path = os.path.join(
        SAVE_DIR,
        f"{video_name}_signals.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved : {save_path}")

# ==========================
# MAIN
# ==========================

print("\nGenerating plots...\n")

for video in videos:
    plot_video(video)

print("\nDone.")
