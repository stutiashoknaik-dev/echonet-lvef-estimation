# ============================================================
# ANALYZE UNIFORM SAMPLING COVERAGE
# ============================================================
#
# PURPOSE
# -------
# This script evaluates whether the current R(2+1)D uniform
# 32-frame sampling strategy actually captures the important
# physiological cardiac phases identified by the adaptive
# temporal analysis.
#
# The script DOES NOT train any model.
#
# It only compares:
#
#   1. Uniformly sampled frames (32 frames)
#   2. High ED score frames
#   3. High ES score frames
#
# and computes their overlap.
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
# INPUTS
# ============================================================

FRAMEWISE_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "multifeature_cardiac_analysis",
    "multifeature_framewise_analysis.csv"
)

FRAME_FOLDER = os.path.join(
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
    "uniform_sampling_analysis"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# OUTPUT FILES
# ============================================================

TIMELINE_DIR = os.path.join(
    SAVE_DIR,
    "timeline_visualizations"
)

os.makedirs(TIMELINE_DIR, exist_ok=True)

COVERAGE_CSV = os.path.join(
    SAVE_DIR,
    "coverage_analysis.csv"
)

SUMMARY_CSV = os.path.join(
    SAVE_DIR,
    "summary_statistics.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

NUM_UNIFORM_FRAMES = 32

# ============================================================
# IMPORTANT REGION PERCENTAGE
# ============================================================

TOP_PERCENT = 10

MATCH_TOLERANCE = 2



# ============================================================
# LOAD FRAMEWISE INFORMATION
# ============================================================

print("\n========================================")
print("Loading temporal feature information...")
print("========================================")

framewise_df = pd.read_csv(FRAMEWISE_CSV)

print(f"\nTotal Records : {len(framewise_df)}")

videos = sorted(framewise_df["Video"].unique())

print(f"Total Videos  : {len(videos)}")

print("\nTemporal feature file loaded successfully.")

# ============================================================
# STORAGE VARIABLES
# ============================================================

coverage_results = []

summary_results = []

# ============================================================
# HELPER FUNCTION
# ============================================================

def get_uniform_sampled_frames(total_frames, num_samples=32):
    """
    Simulate the same uniform sampling strategy used
    by the R(2+1)D model.

    Example

    210 frames

    ↓

    [0,6,13,20,...,209]
    """

    if total_frames <= num_samples:

        return np.arange(total_frames)

    sampled = np.linspace(
        0,
        total_frames - 1,
        num_samples
    )

    sampled = np.round(sampled).astype(int)

    sampled = np.unique(sampled)

    return sampled


# ============================================================
# HELPER FUNCTION
# ============================================================

# ============================================================
# IMPORTANT REGIONS
# ============================================================

def get_important_frames(
    scores,
    frames,
    top_percent=10
):
    """
    Select all frames whose score belongs to the
    highest 'top_percent' of the distribution.
    """

    scores = np.asarray(scores)
    frames = np.asarray(frames)

    threshold = np.percentile(
        scores,
        100 - top_percent
    )

    important = frames[
        scores >= threshold
    ]

    return np.sort(important)


# ============================================================
# HELPER FUNCTION
# ============================================================

def compute_overlap(selected_frames,
                    important_frames,
                    tolerance=2):
    """
    Counts how many important frames
    are covered by uniform sampling.

    A frame is considered covered if

    |sampled - important| <= tolerance
    """

    matched = []

    for frame in important_frames:

        if np.any(
            np.abs(selected_frames - frame) <= tolerance
        ):

            matched.append(frame)

    return matched
# ============================================================
# MAIN VIDEO LOOP
# ============================================================

print("\n========================================")
print("Analyzing Uniform Sampling Coverage...")
print("========================================")

for video in videos:

    print(f"\nProcessing : {video}")

    # ========================================================
    # VIDEO DATA
    # ========================================================

    video_df = framewise_df[
        framewise_df["Video"] == video
    ].copy()

    video_df = video_df.sort_values("Frame")

    frames = video_df["Frame"].values

    ed_scores = video_df["ED_Score"].values

    es_scores = video_df["ES_Score"].values

    # ========================================================
    # TOTAL FRAMES
    # ========================================================

    frame_dir = os.path.join(
        FRAME_FOLDER,
        video
    )

    if not os.path.exists(frame_dir):

        print("Frame folder not found.")

        continue

    image_files = sorted([

        f for f in os.listdir(frame_dir)

        if f.endswith(".png")

    ])

    total_frames = len(image_files)

    # ========================================================
    # SIMULATE R(2+1)D UNIFORM SAMPLING
    # ========================================================

    uniform_frames = get_uniform_sampled_frames(
        total_frames,
        NUM_UNIFORM_FRAMES
    )

    # ========================================================
    # TOP ED FRAMES
    # ========================================================

    top_ed_frames = get_important_frames(
    ed_scores,
    frames,
    TOP_PERCENT
)

    # ========================================================
    # TOP ES FRAMES
    # ========================================================

    top_es_frames = get_important_frames(
    es_scores,
    frames,
    TOP_PERCENT
)

    # ========================================================
    # COMPUTE OVERLAP
    # ========================================================

    matched_ed = compute_overlap(

        uniform_frames,

        top_ed_frames,

        MATCH_TOLERANCE

    )

    matched_es = compute_overlap(

        uniform_frames,

        top_es_frames,

        MATCH_TOLERANCE

    )

    # ========================================================
    # COVERAGE
    # ========================================================

    ed_coverage = (

        len(matched_ed)

        /

        len(top_ed_frames)

    ) * 100

    es_coverage = (

        len(matched_es)

        /

        len(top_es_frames)

    ) * 100

    overall_coverage = (

        ed_coverage +

        es_coverage

    ) / 2

    # ========================================================
    # STORE RESULTS
    # ========================================================

    coverage_results.append({

        "Video": video,

        "Total_Frames": total_frames,

        "Uniform_Frames": len(uniform_frames),

        "Top_ED_Frames": ",".join(map(str, top_ed_frames)),

        "Top_ES_Frames": ",".join(map(str, top_es_frames)),

        "Matched_ED": ",".join(map(str, matched_ed)),

        "Matched_ES": ",".join(map(str, matched_es)),

        "ED_Coverage": round(ed_coverage,2),

        "ES_Coverage": round(es_coverage,2),

        "Overall_Coverage": round(overall_coverage,2),

        "Important_ED_Frames": len(top_ed_frames),

        "Important_ES_Frames": len(top_es_frames),

        "Matched_ED_Count": len(matched_ed),

        "Matched_ES_Count": len(matched_es),

    })

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_results.append(

        overall_coverage

    )

    print(

        f"Coverage : "

        f"ED={ed_coverage:.1f}% "

        f"ES={es_coverage:.1f}% "

        f"Overall={overall_coverage:.1f}%"

    )

# ============================================================
# TIMELINE VISUALIZATION
# ============================================================

print("\nGenerating Timeline Visualizations...")

for video in videos:

    video_df = framewise_df[
        framewise_df["Video"] == video
    ].copy()

    video_df = video_df.sort_values("Frame")

    frames = video_df["Frame"].values

    area = video_df["Area"].values

    stability = video_df["Stability"].values

    ed_scores = video_df["ED_Score"].values

    es_scores = video_df["ES_Score"].values

    # --------------------------------------------------------
    # Normalize Signals
    # --------------------------------------------------------

    def normalize(x):

        x = np.asarray(x)

        return (

            x - x.min()

        ) / (

            x.max() - x.min() + 1e-8

        )

    area = normalize(area)

    stability = normalize(stability)

    ed_scores = normalize(ed_scores)

    es_scores = normalize(es_scores)

    importance = (

        0.20 * area +

        0.30 * stability +

        0.25 * ed_scores +

        0.25 * es_scores

    )

    importance = normalize(importance)

    # --------------------------------------------------------
    # Uniform Frames
    # --------------------------------------------------------

    total_frames = len(frames)

    uniform_frames = get_uniform_sampled_frames(

        total_frames,

        NUM_UNIFORM_FRAMES

    )

    # --------------------------------------------------------
    # Important Frames
    # --------------------------------------------------------

    top_ed_frames = get_important_frames(

        ed_scores,

        frames,

        TOP_PERCENT

    )

    top_es_frames = get_important_frames(

        es_scores,

        frames,

        TOP_PERCENT

    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    plt.figure(

        figsize=(18,8)

    )

    plt.plot(

        frames,

        importance,

        linewidth=3,

        label="Importance Score"

    )

    plt.plot(

        frames,

        ed_scores,

        linewidth=2,

        alpha=0.8,

        label="ED Score"

    )

    plt.plot(

        frames,

        es_scores,

        linewidth=2,

        alpha=0.8,

        label="ES Score"

    )

    # --------------------------------------------------------
    # Uniform Sample Lines
    # --------------------------------------------------------

    for f in uniform_frames:

        plt.axvline(

            f,

            color="blue",

            alpha=0.15

        )

    # --------------------------------------------------------
    # ED Regions
    # --------------------------------------------------------

    for f in top_ed_frames:

        plt.axvspan(

            f-0.5,

            f+0.5,

            color="green",

            alpha=0.35

        )

    # --------------------------------------------------------
    # ES Regions
    # --------------------------------------------------------

    for f in top_es_frames:

        plt.axvspan(

            f-0.5,

            f+0.5,

            color="red",

            alpha=0.35

        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    matched_ed = compute_overlap(

        uniform_frames,

        top_ed_frames,

        MATCH_TOLERANCE

    )

    matched_es = compute_overlap(

        uniform_frames,

        top_es_frames,

        MATCH_TOLERANCE

    )

    ed_cov = (

        len(matched_ed)

        /

        len(top_ed_frames)

    ) * 100

    es_cov = (

        len(matched_es)

        /

        len(top_es_frames)

    ) * 100

    overall = (

        ed_cov +

        es_cov

    ) / 2

    stats = (

        f"Frames : {total_frames}\n"

        f"ED Coverage : {ed_cov:.1f}%\n"

        f"ES Coverage : {es_cov:.1f}%\n"

        f"Overall : {overall:.1f}%"

    )

    plt.text(

        0.99,

        0.98,

        stats,

        transform=plt.gca().transAxes,

        ha="right",

        va="top",

        fontsize=11,

        bbox=dict(

            facecolor="white",

            edgecolor="black"

        )

    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.title(

        f"{video}\nUniform Sampling vs Important Cardiac Regions",

        fontsize=15,

        fontweight="bold"

    )

    plt.xlabel("Frame Number")

    plt.ylabel("Normalized Score")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(

        TIMELINE_DIR,

        f"{video}_timeline.png"

    )

    plt.savefig(

        save_path,

        dpi=300

    )

    plt.close()

print("\nTimeline Visualizations Saved.")
