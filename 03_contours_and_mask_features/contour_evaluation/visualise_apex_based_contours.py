# ============================================================
# APEX-BASED GT vs PRED VISUAL COMPARISON
# NO COORDINATE REORDERING
# NO RESAMPLING
# USES STORED ORDER AS-IS
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# GT CSVs
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

# ============================================================
# PRED CSVs - APEX BASED
# ============================================================

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
    "outputs/apex_based_visual_comparison"
)

os.makedirs(SAVE_DIR, exist_ok=True)

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
# CONFIG
# ============================================================

SCALE = 4
NUM_SAMPLES_PER_PHASE = 15  # change to 10 if you want more
RANDOM_STATE = 42

# ============================================================
# HELPERS
# ============================================================

def build_contour_array(df_slice):
    """
    Keep the CSV order exactly as it is.
    Do not sort, do not reorder.
    Returns left wall and right wall as stored.
    """
    left_pts = df_slice[["X1", "Y1"]].values.astype(np.int32)
    right_pts = df_slice[["X2", "Y2"]].values.astype(np.int32)

    return left_pts, right_pts

def draw_phase_comparison(gt_df, pred_df, phase_name, save_dir, num_samples=5):
    """
    Randomly sample videos and draw GT vs Pred side by side.
    """
    gt_videos = sorted(gt_df["FileName"].unique())
    common_videos = sorted(list(set(gt_videos) & set(pred_df["FileName"].unique())))

    if len(common_videos) == 0:
        print(f"No common videos found for {phase_name}")
        return

    sample_n = min(num_samples, len(common_videos))
    sampled_videos = pd.Series(common_videos).sample(
        n=sample_n,
        random_state=RANDOM_STATE
    ).tolist()

    print(f"\n{phase_name}: visualizing {len(sampled_videos)} samples")

    for idx, video in enumerate(sampled_videos, start=1):

        gt_temp = gt_df[gt_df["FileName"] == video]
        pred_temp = pred_df[pred_df["FileName"] == video]

        if len(gt_temp) == 0 or len(pred_temp) == 0:
            continue

        # Create blank canvases
        gt_img = np.zeros((600, 600, 3), dtype=np.uint8)
        pred_img = np.zeros((600, 600, 3), dtype=np.uint8)

        # Build contours in stored order
        gt_left, gt_right = build_contour_array(gt_temp)
        pred_left, pred_right = build_contour_array(pred_temp)

        # Scale coordinates for visibility
        gt_left = gt_left * SCALE
        gt_right = gt_right * SCALE
        pred_left = pred_left * SCALE
        pred_right = pred_right * SCALE

        # Draw GT contours as-is
        if len(gt_left) > 1:
            cv2.polylines(gt_img, [gt_left], False, (0, 255, 0), 2)
        if len(gt_right) > 1:
            cv2.polylines(gt_img, [gt_right], False, (0, 255, 0), 2)

        # Draw Pred contours as-is
        if len(pred_left) > 1:
            cv2.polylines(pred_img, [pred_left], False, (0, 0, 255), 2)
        if len(pred_right) > 1:
            cv2.polylines(pred_img, [pred_right], False, (0, 0, 255), 2)

        # Add points for clarity
        for p in gt_left:
            cv2.circle(gt_img, tuple(p), 2, (0, 255, 0), -1)
        for p in gt_right:
            cv2.circle(gt_img, tuple(p), 2, (0, 255, 0), -1)

        for p in pred_left:
            cv2.circle(pred_img, tuple(p), 2, (0, 0, 255), -1)
        for p in pred_right:
            cv2.circle(pred_img, tuple(p), 2, (0, 0, 255), -1)

        # Labels
        cv2.putText(
            gt_img, "GT CONTOUR", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        cv2.putText(
            pred_img, "PRED CONTOUR", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # Combine
        combined = np.hstack([gt_img, pred_img])

        # Bottom panel
        panel = np.zeros((80, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            panel,
            f"{phase_name} | {video}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
        cv2.putText(
            panel,
            f"Sample {idx}/{sample_n}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        final = np.vstack([combined, panel])

        save_path = os.path.join(
            save_dir,
            f"{phase_name}_{idx}_{video.replace('.avi','')}.png"
        )
        cv2.imwrite(save_path, final)

# ============================================================
# RUN FOR ED AND ES
# ============================================================

draw_phase_comparison(
    gt_ed_df,
    pred_ed_df,
    phase_name="ED",
    save_dir=SAVE_DIR,
    num_samples=NUM_SAMPLES_PER_PHASE
)

draw_phase_comparison(
    gt_es_df,
    pred_es_df,
    phase_name="ES",
    save_dir=SAVE_DIR,
    num_samples=NUM_SAMPLES_PER_PHASE
)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("APEX-BASED VISUAL COMPARISON COMPLETE")
print("======================================")
print(f"\nSaved Directory:\n{SAVE_DIR}")
