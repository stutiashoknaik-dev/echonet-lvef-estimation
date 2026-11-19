# ============================================================
# PHASEWISE ED / ES CONTOUR COMPARISON
# GT ED vs Pred ED (Apex-Based)
# GT ES vs Pred ES (Apex-Based)
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
# INPUT CSV FILES
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
    "outputs/test_predicted_contours_apex_based",
    "Predicted_ED_Test_VolumeTracings_ApexBased.csv"
)

PRED_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_apex_based",
    "Predicted_ES_Test_VolumeTracings_ApexBased.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_phasewise_contour_metrics_apex_based"
)

os.makedirs(SAVE_DIR, exist_ok=True)

DETAIL_CSV = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Metrics_ApexBased.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Summary_ApexBased.txt"
)

# ============================================================
# LOAD CSVs
# ============================================================

print("\nLoading CSV files...\n")

gt_ed_df = pd.read_csv(GT_ED_CSV)
gt_es_df = pd.read_csv(GT_ES_CSV)
pred_ed_df = pd.read_csv(PRED_ED_CSV)
pred_es_df = pd.read_csv(PRED_ES_CSV)

for df in [gt_ed_df, gt_es_df, pred_ed_df, pred_es_df]:
    df["FileName"] = df["FileName"].astype(str).str.strip()

print("GT ED Loaded")
print("GT ES Loaded")
print("Pred ED Loaded")
print("Pred ES Loaded")

# ============================================================
# METRIC FUNCTION
# ============================================================

def compute_metrics(gt_points, pred_points):
    dist_matrix = cdist(gt_points, pred_points)

    chamfer = (
        np.mean(np.min(dist_matrix, axis=1)) +
        np.mean(np.min(dist_matrix, axis=0))
    ) / 2.0

    gt_to_pred = np.min(dist_matrix, axis=1)
    pred_to_gt = np.min(dist_matrix, axis=0)

    hd95 = np.percentile(
        np.concatenate([gt_to_pred, pred_to_gt]),
        95
    )

    mean_distance = np.mean(gt_to_pred)

    return chamfer, hd95, mean_distance

# ============================================================
# PHASEWISE COMPARISON
# ============================================================

results = []

def compare_phase(phase_name, gt_df, pred_df):
    videos = sorted(gt_df["FileName"].unique())

    print("\n======================================")
    print(f"{phase_name} COMPARISON")
    print("======================================")
    print(f"Videos: {len(videos)}")

    for video in videos:
        gt_video = gt_df[gt_df["FileName"] == video]
        pred_video = pred_df[pred_df["FileName"] == video]

        if len(gt_video) == 0 or len(pred_video) == 0:
            continue

        gt_points = np.vstack([
            gt_video[["X1", "Y1"]].values,
            gt_video[["X2", "Y2"]].values
        ])

        pred_points = np.vstack([
            pred_video[["X1", "Y1"]].values,
            pred_video[["X2", "Y2"]].values
        ])

        if len(gt_points) == 0 or len(pred_points) == 0:
            continue

        chamfer, hd95, mean_distance = compute_metrics(gt_points, pred_points)

        results.append({
            "FileName": video,
            "Phase": phase_name,
            "Chamfer_Distance": chamfer,
            "HD95": hd95,
            "Mean_Contour_Distance": mean_distance
        })

# Run both phases
compare_phase("ED", gt_ed_df, pred_ed_df)
compare_phase("ES", gt_es_df, pred_es_df)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv(DETAIL_CSV, index=False)

# ============================================================
# SUMMARY
# ============================================================

def print_phase_summary(phase):
    phase_df = results_df[results_df["Phase"] == phase]
    print("\n======================================")
    print(f"{phase} RESULTS")
    print("======================================")
    print(f"\nChamfer Distance : {phase_df['Chamfer_Distance'].mean():.4f}")
    print(f"HD95 : {phase_df['HD95'].mean():.4f}")
    print(f"Mean Contour Distance : {phase_df['Mean_Contour_Distance'].mean():.4f}")

print_phase_summary("ED")
print_phase_summary("ES")

print("\n======================================")
print("OVERALL RESULTS")
print("======================================")
print(f"\nChamfer Distance : {results_df['Chamfer_Distance'].mean():.4f}")
print(f"HD95 : {results_df['HD95'].mean():.4f}")
print(f"Mean Contour Distance : {results_df['Mean_Contour_Distance'].mean():.4f}")

with open(SUMMARY_TXT, "w") as f:
    f.write("======================================\n")
    f.write("PHASEWISE CONTOUR SUMMARY\n")
    f.write("======================================\n\n")

    for phase in ["ED", "ES"]:
        phase_df = results_df[results_df["Phase"] == phase]
        f.write(f"{phase} RESULTS\n")
        f.write(f"Chamfer Distance : {phase_df['Chamfer_Distance'].mean():.4f}\n")
        f.write(f"HD95 : {phase_df['HD95'].mean():.4f}\n")
        f.write(f"Mean Contour Distance : {phase_df['Mean_Contour_Distance'].mean():.4f}\n\n")

    f.write("OVERALL RESULTS\n")
    f.write(f"Chamfer Distance : {results_df['Chamfer_Distance'].mean():.4f}\n")
    f.write(f"HD95 : {results_df['HD95'].mean():.4f}\n")
    f.write(f"Mean Contour Distance : {results_df['Mean_Contour_Distance'].mean():.4f}\n")

print("\n======================================")
print("PHASEWISE CONTOUR EVALUATION COMPLETE")
print("======================================")
print(f"\nDetailed CSV Saved:\n{DETAIL_CSV}")
print(f"\nSummary TXT Saved:\n{SUMMARY_TXT}")
