# ============================================================
# FINAL PHASEWISE CONTOUR EVALUATION
# Genuine HD95 = 95th percentile of symmetric nearest-point
# contour distances
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT FILES
# These should contain the FINAL reference and predicted
# phase-specific contours.
# Coordinates are expected to be in the same 112 x 112
# image coordinate system.
# ============================================================

GT_ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_contour_gt_phasewise",
    "test_volume_tracings_ED.csv"
)

GT_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_contour_gt_phasewise",
    "test_volume_tracings_ES.csv"
)

PRED_ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_anatomical",
    "Predicted_ED_Test_VolumeTracings_Anatomical.csv"
)

PRED_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_anatomical",
    "Predicted_ES_Test_VolumeTracings_Anatomical.csv"
)

# ============================================================
# OUTPUT
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_phasewise_contour_metrics_hd95"
)

os.makedirs(SAVE_DIR, exist_ok=True)

DETAIL_CSV = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Metrics_HD95.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Summary_HD95.txt"
)

# ============================================================
# LOAD DATA
# ============================================================

gt_ed_df = pd.read_csv(GT_ED_CSV)
gt_es_df = pd.read_csv(GT_ES_CSV)
pred_ed_df = pd.read_csv(PRED_ED_CSV)
pred_es_df = pd.read_csv(PRED_ES_CSV)

for df in [gt_ed_df, gt_es_df, pred_ed_df, pred_es_df]:
    df["FileName"] = (
        df["FileName"]
        .astype(str)
        .str.strip()
    )

print("GT ED:", len(gt_ed_df))
print("GT ES:", len(gt_es_df))
print("Pred ED:", len(pred_ed_df))
print("Pred ES:", len(pred_es_df))

gt_ed_v = set(gt_ed_df["FileName"].unique())
gt_es_v = set(gt_es_df["FileName"].unique())

pred_ed_v = set(pred_ed_df["FileName"].unique())
pred_es_v = set(pred_es_df["FileName"].unique())

print("GT ED videos:", len(gt_ed_v))
print("GT ES videos:", len(gt_es_v))
print("Pred ED videos:", len(pred_ed_v))
print("Pred ES videos:", len(pred_es_v))

print("\nMissing Pred ED:")
print(sorted(gt_ed_v - pred_ed_v))

print("\nMissing Pred ES:")
print(sorted(gt_es_v - pred_es_v))

print("\nED intersection:", len(gt_ed_v & pred_ed_v))
print("ES intersection:", len(gt_es_v & pred_es_v))

print("\nVideos with BOTH ED and ES valid:",
      len((gt_ed_v & pred_ed_v) & (gt_es_v & pred_es_v)))


# ============================================================
# METRICS
# ============================================================

def compute_metrics(gt_points, pred_points):
    """
    Compute:
      - symmetric Chamfer distance
      - genuine symmetric HD95
      - mean GT-to-predicted contour distance

    Distances are reported in the coordinate units of the input
    contour CSVs. For the present pipeline, these correspond to
    pixels in the resized 112 x 112 image coordinate system.
    """

    gt_points = np.asarray(gt_points, dtype=np.float64)
    pred_points = np.asarray(pred_points, dtype=np.float64)

    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.nan, np.nan, np.nan

    # Pairwise Euclidean distances
    dist_matrix = cdist(gt_points, pred_points)

    # --------------------------------------------------------
    # Symmetric Chamfer distance
    # --------------------------------------------------------

    gt_to_pred = np.min(dist_matrix, axis=1)
    pred_to_gt = np.min(dist_matrix, axis=0)

    chamfer = (
        np.mean(gt_to_pred) +
        np.mean(pred_to_gt)
    ) / 2.0

    # --------------------------------------------------------
    # Genuine HD95
    #
    # Combine nearest-neighbour distances in both directions
    # and take the 95th percentile.
    # --------------------------------------------------------

    symmetric_distances = np.concatenate(
        [gt_to_pred, pred_to_gt]
    )

    hd95 = np.percentile(
        symmetric_distances,
        95
    )

    # --------------------------------------------------------
    # Mean contour distance
    # --------------------------------------------------------

    mean_distance = np.mean(gt_to_pred)

    return chamfer, hd95, mean_distance


# ============================================================
# PHASEWISE COMPARISON
# ============================================================

results = []


def compare_phase(phase_name, gt_df, pred_df):

    gt_videos = set(gt_df["FileName"].unique())
    pred_videos = set(pred_df["FileName"].unique())

    common_videos = sorted(
        gt_videos.intersection(pred_videos)
    )

    print("\n======================================")
    print(f"{phase_name} CONTOUR EVALUATION")
    print("======================================")
    print(f"GT videos       : {len(gt_videos)}")
    print(f"Predicted videos: {len(pred_videos)}")
    print(f"Matched videos  : {len(common_videos)}")

    for video in common_videos:

        gt_video = gt_df[
            gt_df["FileName"] == video
        ]

        pred_video = pred_df[
            pred_df["FileName"] == video
        ]

        if len(gt_video) == 0 or len(pred_video) == 0:
            continue

        # ----------------------------------------------------
        # Reference contour points
        # ----------------------------------------------------

        gt_points = np.vstack([
            gt_video[["X1", "Y1"]].values,
            gt_video[["X2", "Y2"]].values
        ])

        # ----------------------------------------------------
        # Predicted contour points
        # ----------------------------------------------------

        pred_points = np.vstack([
            pred_video[["X1", "Y1"]].values,
            pred_video[["X2", "Y2"]].values
        ])

        if len(gt_points) == 0 or len(pred_points) == 0:
            continue

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        chamfer, hd95, mean_distance = compute_metrics(
            gt_points,
            pred_points
        )

        results.append({
            "FileName": video,
            "Phase": phase_name,
            "Chamfer_Distance": chamfer,
            "HD95": hd95,
            "Mean_Contour_Distance": mean_distance
        })


# ============================================================
# RUN ED AND ES
# ============================================================

compare_phase(
    "ED",
    gt_ed_df,
    pred_ed_df
)

compare_phase(
    "ES",
    gt_es_df,
    pred_es_df
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    DETAIL_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("FINAL CONTOUR RESULTS")
print("======================================")

for phase in ["ED", "ES"]:

    phase_df = results_df[
        results_df["Phase"] == phase
    ]

    print(f"\n{phase}")
    print(f"Valid videos              : {len(phase_df)}")
    print(
        f"Chamfer Distance          : "
        f"{phase_df['Chamfer_Distance'].mean():.4f}"
    )
    print(
        f"HD95                      : "
        f"{phase_df['HD95'].mean():.4f}"
    )
    print(
        f"Mean Contour Distance     : "
        f"{phase_df['Mean_Contour_Distance'].mean():.4f}"
    )


print("\n======================================")
print("OVERALL")
print("======================================")

print(
    f"Overall valid comparisons : "
    f"{len(results_df)}"
)

print(
    f"Overall Chamfer Distance  : "
    f"{results_df['Chamfer_Distance'].mean():.4f}"
)

print(
    f"Overall HD95              : "
    f"{results_df['HD95'].mean():.4f}"
)

print(
    f"Overall Mean Contour Dist. : "
    f"{results_df['Mean_Contour_Distance'].mean():.4f}"
)


# ============================================================
# SAVE SUMMARY TEXT
# ============================================================

with open(SUMMARY_TXT, "w") as f:

    f.write(
        "FINAL PHASEWISE CONTOUR EVALUATION\n"
    )
    f.write(
        "HD95 = 95th percentile of symmetric "
        "nearest-point contour distances\n\n"
    )

    for phase in ["ED", "ES"]:

        phase_df = results_df[
            results_df["Phase"] == phase
        ]

        f.write(
            f"{phase} RESULTS\n"
        )

        f.write(
            f"Valid videos: "
            f"{len(phase_df)}\n"
        )

        f.write(
            f"Chamfer Distance: "
            f"{phase_df['Chamfer_Distance'].mean():.4f}\n"
        )

        f.write(
            f"HD95: "
            f"{phase_df['HD95'].mean():.4f}\n"
        )

        f.write(
            f"Mean Contour Distance: "
            f"{phase_df['Mean_Contour_Distance'].mean():.4f}\n\n"
        )

    f.write("OVERALL RESULTS\n")
    f.write(
        f"Valid comparisons: "
        f"{len(results_df)}\n"
    )
    f.write(
        f"Chamfer Distance: "
        f"{results_df['Chamfer_Distance'].mean():.4f}\n"
    )
    f.write(
        f"HD95: "
        f"{results_df['HD95'].mean():.4f}\n"
    )
    f.write(
        f"Mean Contour Distance: "
        f"{results_df['Mean_Contour_Distance'].mean():.4f}\n"
    )

print("\nSaved detailed results to:")
print(DETAIL_CSV)

print("\nSaved summary to:")
print(SUMMARY_TXT)
