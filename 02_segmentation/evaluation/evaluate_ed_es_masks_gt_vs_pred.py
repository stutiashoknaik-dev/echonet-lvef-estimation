
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# PATHS
# ============================================================

GT_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_GT_test_masks"
)

PRED_DIR = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/final_gt_vs_pred_metrics"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def dice_score(gt, pred):

    intersection = np.sum(gt * pred)

    return (
        2.0 * intersection
    ) / (
        np.sum(gt) + np.sum(pred) + 1e-8
    )


def iou_score(gt, pred):

    intersection = np.sum(gt * pred)

    union = np.sum((gt + pred) > 0)

    return intersection / (union + 1e-8)


def mae_score(gt, pred):

    return np.mean(
        np.abs(gt - pred)
    )


def hausdorff95(gt, pred):

    gt_pts = np.argwhere(gt > 0)
    pred_pts = np.argwhere(pred > 0)

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan

    hd1 = directed_hausdorff(
        gt_pts,
        pred_pts
    )[0]

    hd2 = directed_hausdorff(
        pred_pts,
        gt_pts
    )[0]

    return max(hd1, hd2)

# ============================================================
# STORAGE
# ============================================================

ed_results = []
es_results = []

# ============================================================
# VIDEO LIST
# ============================================================

videos = os.listdir(GT_DIR)

print("\nStarting GT vs Pred evaluation...\n")

# ============================================================
# MAIN LOOP
# ============================================================

for video in videos:

    gt_video_dir = os.path.join(
        GT_DIR,
        video
    )

    pred_video_dir = os.path.join(
        PRED_DIR,
        video
    )

    if not os.path.exists(pred_video_dir):
        continue

    # ========================================================
    # ED
    # ========================================================

    gt_ed_files = [

        f for f in os.listdir(gt_video_dir)

        if f.startswith("ED")
    ]

    pred_ed_dir = os.path.join(
        pred_video_dir,
        "ED"
    )

    if (
        len(gt_ed_files) == 0
        or
        not os.path.exists(pred_ed_dir)
    ):
        continue

    pred_ed_files = [

        f for f in os.listdir(pred_ed_dir)

        if "mask" in f.lower()
    ]

    if len(pred_ed_files) == 0:
        continue

    gt_ed_path = os.path.join(
        gt_video_dir,
        gt_ed_files[0]
    )

    pred_ed_path = os.path.join(
        pred_ed_dir,
        pred_ed_files[0]
    )

    gt_ed = cv2.imread(
        gt_ed_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_ed = cv2.imread(
        pred_ed_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gt_ed is None or pred_ed is None:
        continue

    gt_ed = (
        gt_ed > 0
    ).astype(np.float32)

    pred_ed = (
        pred_ed > 0
    ).astype(np.float32)

    ed_results.append({

        "Video": video,

        "Dice": dice_score(
            gt_ed,
            pred_ed
        ),

        "IoU": iou_score(
            gt_ed,
            pred_ed
        ),

        "MAE": mae_score(
            gt_ed,
            pred_ed
        ),

        "HD95": hausdorff95(
            gt_ed,
            pred_ed
        )
    })

    # ========================================================
    # ES
    # ========================================================

    gt_es_files = [

        f for f in os.listdir(gt_video_dir)

        if f.startswith("ES")
    ]

    pred_es_dir = os.path.join(
        pred_video_dir,
        "ES"
    )

    if (
        len(gt_es_files) == 0
        or
        not os.path.exists(pred_es_dir)
    ):
        continue

    pred_es_files = [

        f for f in os.listdir(pred_es_dir)

        if "mask" in f.lower()
    ]

    if len(pred_es_files) == 0:
        continue

    gt_es_path = os.path.join(
        gt_video_dir,
        gt_es_files[0]
    )

    pred_es_path = os.path.join(
        pred_es_dir,
        pred_es_files[0]
    )

    gt_es = cv2.imread(
        gt_es_path,
        cv2.IMREAD_GRAYSCALE
    )

    pred_es = cv2.imread(
        pred_es_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gt_es is None or pred_es is None:
        continue

    gt_es = (
        gt_es > 0
    ).astype(np.float32)

    pred_es = (
        pred_es > 0
    ).astype(np.float32)

    es_results.append({

        "Video": video,

        "Dice": dice_score(
            gt_es,
            pred_es
        ),

        "IoU": iou_score(
            gt_es,
            pred_es
        ),

        "MAE": mae_score(
            gt_es,
            pred_es
        ),

        "HD95": hausdorff95(
            gt_es,
            pred_es
        )
    })

# ============================================================
# DATAFRAMES
# ============================================================

ed_df = pd.DataFrame(ed_results)
es_df = pd.DataFrame(es_results)

# ============================================================
# SAVE CSV
# ============================================================

ed_csv = os.path.join(
    SAVE_DIR,
    "ED_metrics.csv"
)

es_csv = os.path.join(
    SAVE_DIR,
    "ES_metrics.csv"
)

ed_df.to_csv(
    ed_csv,
    index=False
)

es_df.to_csv(
    es_csv,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("FINAL GT vs PRED RESULTS")
print("======================================")

print("\n---------- ED RESULTS ----------\n")

print(
    f"Dice : {ed_df['Dice'].mean():.4f}"
)

print(
    f"IoU  : {ed_df['IoU'].mean():.4f}"
)

print(
    f"MAE  : {ed_df['MAE'].mean():.4f}"
)

print(
    f"HD95 : {ed_df['HD95'].mean():.4f}"
)

print("\n---------- ES RESULTS ----------\n")

print(
    f"Dice : {es_df['Dice'].mean():.4f}"
)

print(
    f"IoU  : {es_df['IoU'].mean():.4f}"
)

print(
    f"MAE  : {es_df['MAE'].mean():.4f}"
)

print(
    f"HD95 : {es_df['HD95'].mean():.4f}"
)

# ============================================================
# SAVE FINAL SUMMARY TXT
# ============================================================

summary_txt = os.path.join(
    SAVE_DIR,
    "final_results_summary.txt"
)

with open(summary_txt, "w") as f:

    f.write("======================================\n")
    f.write("FINAL GT vs PRED RESULTS\n")
    f.write("======================================\n\n")

    f.write("---------- ED RESULTS ----------\n\n")

    f.write(
        f"Dice : {ed_df['Dice'].mean():.4f}\n"
    )

    f.write(
        f"IoU  : {ed_df['IoU'].mean():.4f}\n"
    )

    f.write(
        f"MAE  : {ed_df['MAE'].mean():.4f}\n"
    )

    f.write(
        f"HD95 : {ed_df['HD95'].mean():.4f}\n\n"
    )

    f.write("---------- ES RESULTS ----------\n\n")

    f.write(
        f"Dice : {es_df['Dice'].mean():.4f}\n"
    )

    f.write(
        f"IoU  : {es_df['IoU'].mean():.4f}\n"
    )

    f.write(
        f"MAE  : {es_df['MAE'].mean():.4f}\n"
    )

    f.write(
        f"HD95 : {es_df['HD95'].mean():.4f}\n"
    )

print("\n✅ Summary TXT saved")
print(f"\nSaved TXT:\n{summary_txt}")

print("\n✅ ED metrics saved")
print("✅ ES metrics saved")
