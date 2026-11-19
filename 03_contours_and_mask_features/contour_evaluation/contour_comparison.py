# # ============================================================
# # FINAL PHASEWISE CONTOUR METRICS
# # GT ED/ES vs PRED ED/ES
# # ============================================================

# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cdist

# # ============================================================
# # BASE PATH
# # ============================================================

# BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# # ============================================================
# # GT CSV
# # ============================================================

# GT_CSV = os.path.join(
#     BASE_PATH,
#     "outputs/test_contour_gt",
#     "test_volume_tracings.csv"
# )

# # ============================================================
# # PREDICTED CSVs
# # ============================================================

# PRED_ED_CSV = os.path.join(
#     BASE_PATH,
#     "outputs/test_predicted_contours",
#     "Predicted_ED_Test_VolumeTracings.csv"
# )

# PRED_ES_CSV = os.path.join(
#     BASE_PATH,
#     "outputs/test_predicted_contours",
#     "Predicted_ES_Test_VolumeTracings.csv"
# )

# # ============================================================
# # SAVE DIRECTORY
# # ============================================================

# SAVE_DIR = os.path.join(
#     BASE_PATH,
#     "outputs/final_phasewise_contour_metrics"
# )

# os.makedirs(SAVE_DIR, exist_ok=True)

# # ============================================================
# # LOAD CSV FILES
# # ============================================================

# print("\nLoading CSV files...\n")

# gt_df = pd.read_csv(GT_CSV)

# pred_ed_df = pd.read_csv(PRED_ED_CSV)
# pred_es_df = pd.read_csv(PRED_ES_CSV)

# # ============================================================
# # CLEAN FILENAMES
# # ============================================================

# for df in [gt_df, pred_ed_df, pred_es_df]:

#     df["FileName"] = (
#         df["FileName"]
#         .astype(str)
#         .str.strip()
#     )

# # ============================================================
# # UNIQUE VIDEOS
# # ============================================================

# videos = gt_df["FileName"].unique()

# print(f"Total Videos : {len(videos)}")

# # ============================================================
# # STORAGE
# # ============================================================

# all_results = []

# # ============================================================
# # METRIC FUNCTION
# # ============================================================

# def compute_metrics(gt_points, pred_points):

#     dist_matrix = cdist(
#         gt_points,
#         pred_points
#     )

#     # --------------------------------------------------------
#     # Chamfer
#     # --------------------------------------------------------

#     chamfer = (

#         np.mean(np.min(dist_matrix, axis=1))

#         +

#         np.mean(np.min(dist_matrix, axis=0))

#     ) / 2

#     # --------------------------------------------------------
#     # HD95
#     # --------------------------------------------------------

#     gt_to_pred = np.min(
#         dist_matrix,
#         axis=1
#     )

#     pred_to_gt = np.min(
#         dist_matrix,
#         axis=0
#     )

#     hd95 = np.percentile(

#         np.concatenate([

#             gt_to_pred,
#             pred_to_gt

#         ]),

#         95
#     )

#     # --------------------------------------------------------
#     # Mean Contour Distance
#     # --------------------------------------------------------

#     mean_distance = np.mean(
#         gt_to_pred
#     )

#     return chamfer, hd95, mean_distance

# # ============================================================
# # MAIN LOOP
# # ============================================================

# print("\nComputing phasewise contour metrics...\n")

# for video in videos:

#     # ========================================================
#     # GT VIDEO
#     # ========================================================

#     gt_video = gt_df[
#         gt_df["FileName"] == video
#     ]

#     if len(gt_video) == 0:
#         continue

#     # ========================================================
#     # GT ED / ES
#     # ========================================================

#     frames = gt_video["Frame"].unique()

#     gt_ed_frame = np.min(frames)
#     gt_es_frame = np.max(frames)

#     gt_ed = gt_video[
#         gt_video["Frame"] == gt_ed_frame
#     ]

#     gt_es = gt_video[
#         gt_video["Frame"] == gt_es_frame
#     ]

#     # ========================================================
#     # PRED ED
#     # ========================================================

#     pred_ed = pred_ed_df[
#         pred_ed_df["FileName"] == video
#     ]

#     # ========================================================
#     # PRED ES
#     # ========================================================

#     pred_es = pred_es_df[
#         pred_es_df["FileName"] == video
#     ]

#     # ========================================================
#     # PHASE LOOP
#     # ========================================================

#     phases = [

#         ("ED", gt_ed, pred_ed),

#         ("ES", gt_es, pred_es)

#     ]

#     for phase_name, gt_temp, pred_temp in phases:

#         if len(gt_temp) == 0 or len(pred_temp) == 0:
#             continue

#         # ----------------------------------------------------
#         # GT POINTS
#         # ----------------------------------------------------

#         gt_points = np.vstack([

#             gt_temp[["X1","Y1"]].values,

#             gt_temp[["X2","Y2"]].values

#         ])

#         # ----------------------------------------------------
#         # PRED POINTS
#         # ----------------------------------------------------

#         pred_points = np.vstack([

#             pred_temp[["X1","Y1"]].values,

#             pred_temp[["X2","Y2"]].values

#         ])

#         if len(gt_points) == 0 or len(pred_points) == 0:
#             continue

#         # ----------------------------------------------------
#         # COMPUTE METRICS
#         # ----------------------------------------------------

#         chamfer, hd95, mean_distance = compute_metrics(

#             gt_points,
#             pred_points
#         )

#         # ----------------------------------------------------
#         # STORE
#         # ----------------------------------------------------

#         all_results.append({

#             "FileName": video,

#             "Phase": phase_name,

#             "Chamfer_Distance": chamfer,

#             "HD95": hd95,

#             "Mean_Contour_Distance": mean_distance
#         })

# # ============================================================
# # SAVE CSV
# # ============================================================

# results_df = pd.DataFrame(all_results)

# csv_path = os.path.join(
#     SAVE_DIR,
#     "Final_Phasewise_Contour_Metrics.csv"
# )

# results_df.to_csv(
#     csv_path,
#     index=False
# )

# # ============================================================
# # PRINT PHASEWISE RESULTS
# # ============================================================

# for phase in ["ED", "ES"]:

#     phase_df = results_df[
#         results_df["Phase"] == phase
#     ]

#     print("\n======================================")
#     print(f"{phase} RESULTS")
#     print("======================================")

#     print(f"\nChamfer Distance : "
#           f"{phase_df['Chamfer_Distance'].mean():.4f}")

#     print(f"HD95 : "
#           f"{phase_df['HD95'].mean():.4f}")

#     print(f"Mean Contour Distance : "
#           f"{phase_df['Mean_Contour_Distance'].mean():.4f}")

# # ============================================================
# # OVERALL
# # ============================================================

# print("\n======================================")
# print("OVERALL RESULTS")
# print("======================================")

# print(f"\nChamfer Distance : "
#       f"{results_df['Chamfer_Distance'].mean():.4f}")

# print(f"HD95 : "
#       f"{results_df['HD95'].mean():.4f}")

# print(f"Mean Contour Distance : "
#       f"{results_df['Mean_Contour_Distance'].mean():.4f}")

# # ============================================================
# # FINAL
# # ============================================================

# print("\n======================================")
# print("FINAL PHASEWISE CONTOUR EVALUATION COMPLETE")
# print("======================================")

# print(f"\nCSV Saved:\n{csv_path}")




# ============================================================
# PHASEWISE ED / ES CONTOUR COMPARISON
# GT ED vs Pred ED
# GT ES vs Pred ES
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
    "outputs/test_predicted_contours",
    "Predicted_ED_Test_VolumeTracings.csv"
)

PRED_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours",
    "Predicted_ES_Test_VolumeTracings.csv"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_phasewise_contour_metrics"
)

os.makedirs(SAVE_DIR, exist_ok=True)

DETAIL_CSV = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Metrics.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "Phasewise_Contour_Summary.txt"
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

    print(f"\n======================================")
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

        chamfer, hd95, mean_distance = compute_metrics(
            gt_points, pred_points
        )

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
