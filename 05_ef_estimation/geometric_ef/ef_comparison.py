# ============================================================
# UPDATED FULL EF + CONTOUR COMPARISON PIPELINE
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ------------------------------------------------------------
# GT CONTOURS
# ------------------------------------------------------------

GT_ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_contour_gt_phasewise/test_volume_tracings_ED.csv"
)

GT_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_contour_gt_phasewise/test_volume_tracings_ES.csv"
)

# ------------------------------------------------------------
# PREDICTED CONTOURS
# ------------------------------------------------------------

PRED_ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_apex_based/",
    "Predicted_ED_Test_VolumeTracings_ApexBased.csv"
)

PRED_ES_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_apex_based/",
    "Predicted_ES_Test_VolumeTracings_ApexBased.csv"
)

# ------------------------------------------------------------
# PREDICTED EF
# ------------------------------------------------------------

PRED_EF_CSV = os.path.join(
    BASE_PATH,
     "outputs/final_hybrid_ef/",
    "Hybrid_Mask_Contour_EF.csv"
)

# ------------------------------------------------------------
# FILELIST
# ------------------------------------------------------------

FILELIST_CSV = os.path.join(
    BASE_PATH,
    "data/FileList.csv"
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

SAVE_PATH = os.path.join(
    BASE_PATH,
    "outputs/final_comparison_results_updated"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# NORMALIZE VIDEO NAME
# ============================================================

def normalize_name(x):

    x = str(x)

    x = os.path.basename(x)

    x = os.path.splitext(x)[0]

    return x.strip()

# ============================================================
# FIND NAME COLUMN
# ============================================================

def find_name_column(df):

    possible = [
        "FileName",
        "Filename",
        "Video",
        "video",
        "Name"
    ]

    for col in possible:

        if col in df.columns:
            return col

    return None

# ============================================================
# BUILD CONTOUR
# ============================================================

def build_contour(df_group):

    # --------------------------------------------------------
    # SORT FOR STABILITY
    # --------------------------------------------------------

    if "Y1" in df_group.columns:

        df_group = df_group.sort_values(
            by=["Y1", "X1"]
        )

    points = []

    # --------------------------------------------------------
    # LEFT WALL
    # --------------------------------------------------------

    for _, row in df_group.iterrows():

        points.append([
            row["X1"],
            row["Y1"]
        ])

    # --------------------------------------------------------
    # RIGHT WALL REVERSED
    # --------------------------------------------------------

    reverse_rows = df_group.iloc[::-1]

    for _, row in reverse_rows.iterrows():

        points.append([
            row["X2"],
            row["Y2"]
        ])

    contour = np.array(
        points,
        dtype=np.float32
    )

    return contour

# ============================================================
# CONTOUR GEOMETRY
# ============================================================

def contour_geometry(points):

    if len(points) < 3:
        return 0, 0

    contour = points.reshape((-1,1,2))

    area = abs(
        cv2.contourArea(contour)
    )

    perimeter = cv2.arcLength(
        contour,
        closed=True
    )

    return area, perimeter

# ============================================================
# CONTOUR METRICS
# ============================================================

def compute_metrics(pred_pts, gt_pts):

    if len(pred_pts) == 0:
        return None

    if len(gt_pts) == 0:
        return None

    distances = cdist(
        pred_pts,
        gt_pts
    )

    d1 = np.min(
        distances,
        axis=1
    )

    d2 = np.min(
        distances,
        axis=0
    )

    all_d = np.concatenate([d1, d2])

    # ========================================================
    # CHAMFER
    # ========================================================

    chamfer = (
        np.mean(d1)
        +
        np.mean(d2)
    ) / 2

    # ========================================================
    # HD95
    # ========================================================

    hd95 = np.percentile(
        all_d,
        95
    )

    # ========================================================
    # MEAN CONTOUR DISTANCE
    # ========================================================

    mean_distance = np.mean(all_d)

    # ========================================================
    # AREA + PERIMETER
    # ========================================================

    pred_area, pred_peri = contour_geometry(
        pred_pts
    )

    gt_area, gt_peri = contour_geometry(
        gt_pts
    )

    area_diff = abs(
        pred_area - gt_area
    )

    peri_diff = abs(
        pred_peri - gt_peri
    )

    return {

        "Chamfer": chamfer,

        "HD95": hd95,

        "MeanContourDistance": mean_distance,

        "PredArea": pred_area,

        "GTArea": gt_area,

        "AreaDifference": area_diff,

        "PredPerimeter": pred_peri,

        "GTPerimeter": gt_peri,

        "PerimeterDifference": peri_diff
    }

# ============================================================
# LOAD CONTOURS
# ============================================================

def load_contours(csv_path):

    df = pd.read_csv(csv_path)

    name_col = find_name_column(df)

    df[name_col] = df[name_col].apply(
        normalize_name
    )

    groups = {}

    for name, g in df.groupby(name_col):

        groups[name] = g.reset_index(drop=True)

    return groups

# ============================================================
# PHASEWISE COMPARISON
# ============================================================

def compare_phase(
    gt_csv,
    pred_csv,
    phase_name
):

    gt_groups = load_contours(gt_csv)

    pred_groups = load_contours(pred_csv)

    common = sorted(
        set(gt_groups.keys())
        &
        set(pred_groups.keys())
    )

    results = []

    print(f"\n====================================")
    print(f"{phase_name} COMPARISON")
    print(f"====================================")

    print(f"Videos : {len(common)}")

    for video in common:

        gt_pts = build_contour(
            gt_groups[video]
        )

        pred_pts = build_contour(
            pred_groups[video]
        )

        metrics = compute_metrics(
            pred_pts,
            gt_pts
        )

        if metrics is None:
            continue

        metrics["Video"] = video

        metrics["Phase"] = phase_name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    save_csv = os.path.join(
        SAVE_PATH,
        f"{phase_name}_Contour_Comparison.csv"
    )

    results_df.to_csv(
        save_csv,
        index=False
    )

    print(f"\n{phase_name} comparison saved!")

    return results_df

# ============================================================
# EF COMPARISON
# ============================================================

def compare_ef():

    pred_df = pd.read_csv(
        PRED_EF_CSV
    )

    file_df = pd.read_csv(
        FILELIST_CSV
    )

    # --------------------------------------------------------
    # GT EF COLUMN
    # --------------------------------------------------------

    ef_col = None

    possible_ef_cols = [
        "EF",
        "ef",
        "EjectionFraction"
    ]

    for col in possible_ef_cols:

        if col in file_df.columns:

            ef_col = col

            break

    if ef_col is None:

        print("\nNo GT EF column found!")

        return

    # --------------------------------------------------------
    # VIDEO COLUMN
    # --------------------------------------------------------

    video_col = find_name_column(file_df)

    file_df = file_df[
        [video_col, ef_col]
    ]

    file_df.columns = [
        "Video",
        "GT_EF"
    ]

    file_df["Video"] = file_df[
        "Video"
    ].apply(normalize_name)

    pred_df["Video"] = pred_df[
        "Video"
    ].apply(normalize_name)

    # ========================================================
    # MERGE
    # ========================================================

    merged = pred_df.merge(
        file_df,
        on="Video",
        how="inner"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    merged = merged.dropna()

    merged = merged[
        np.isfinite(
            merged["Estimated_EF (%)"]
        )
    ]

    if len(merged) == 0:

        print("\nNo matching EF samples!")

        return

    y_true = merged["GT_EF"].values

    y_pred = merged[
        "Estimated_EF (%)"
    ].values

    # ========================================================
    # METRICS
    # ========================================================

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            /
            np.clip(
                np.abs(y_true),
                1e-7,
                None
            )
        )
    ) * 100

    bias = np.mean(
        y_pred - y_true
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    corr, pval = pearsonr(
        y_true,
        y_pred
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    ef_results = pd.DataFrame([{

        "Samples": len(merged),

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "Bias": bias,

        "Pearson_r": corr,

        "Pearson_p": pval,

        "R2": r2
    }])

    ef_metrics_csv = os.path.join(
        SAVE_PATH,
        "EF_Comparison_Metrics.csv"
    )

    ef_results.to_csv(
        ef_metrics_csv,
        index=False
    )

    merged_csv = os.path.join(
        SAVE_PATH,
        "EF_Pred_vs_GT.csv"
    )

    merged.to_csv(
        merged_csv,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n====================================")
    print("EF COMPARISON COMPLETED")
    print("====================================")

    print(f"Samples   : {len(merged)}")
    print(f"MAE       : {mae:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"MAPE      : {mape:.4f}")
    print(f"Bias      : {bias:.4f}")
    print(f"Pearson r : {corr:.4f}")
    print(f"R2 Score  : {r2:.4f}")

# ============================================================
# RUN ED
# ============================================================

print("\nRunning ED comparison...")

ed_results = compare_phase(
    GT_ED_CSV,
    PRED_ED_CSV,
    "ED"
)

# ============================================================
# RUN ES
# ============================================================

print("\nRunning ES comparison...")

es_results = compare_phase(
    GT_ES_CSV,
    PRED_ES_CSV,
    "ES"
)

# ============================================================
# COMBINE
# ============================================================

combined = pd.concat([
    ed_results,
    es_results
])

combined_csv = os.path.join(
    SAVE_PATH,
    "Combined_Contour_Comparison.csv"
)

combined.to_csv(
    combined_csv,
    index=False
)

print("\nCombined contour comparison saved!")

# ============================================================
# EF COMPARISON
# ============================================================

print("\nRunning EF comparison...")

compare_ef()

print("\nDONE.")
