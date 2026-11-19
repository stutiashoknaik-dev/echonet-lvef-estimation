# Evaluation of the Attention U-Net (attention_best_model.keras) on the validation split:
# Dice and true HD95 with overlay images.

import os
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FRAME_DIR = os.path.join(BASE_PATH, "data/frames/val")
MASK_DIR  = os.path.join(BASE_PATH, "data/masks/val")

# ------------------------------------------------------------
# ATTENTION MODEL
# ------------------------------------------------------------

MODEL_PATH = "attention_best_model.keras"

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/attention_evaluation_true_hd95"
)

OVERLAY_DIR = os.path.join(
    SAVE_DIR,
    "overlays"
)

os.makedirs(OVERLAY_DIR, exist_ok=True)

IMG_SIZE = 112

# ============================================================
# DICE
# ============================================================

def dice_coef(y_true, y_pred):

    smooth = 1e-6

    intersection = np.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Attention U-Net Model Loaded")

# ============================================================
# TRUE HD95
# ============================================================

def hd95(gt, pred):

    gt_pts = np.argwhere(gt > 0)
    pred_pts = np.argwhere(pred > 0)

    # --------------------------------------------------------
    # EMPTY CHECK
    # --------------------------------------------------------

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 999

    # --------------------------------------------------------
    # DISTANCE MATRIX
    # --------------------------------------------------------

    distances = cdist(gt_pts, pred_pts)

    # --------------------------------------------------------
    # GT → PRED
    # --------------------------------------------------------

    gt_to_pred = np.min(distances, axis=1)

    # --------------------------------------------------------
    # PRED → GT
    # --------------------------------------------------------

    pred_to_gt = np.min(distances, axis=0)

    # --------------------------------------------------------
    # COMBINE DISTANCES
    # --------------------------------------------------------

    all_distances = np.concatenate([
        gt_to_pred,
        pred_to_gt
    ])

    # --------------------------------------------------------
    # TRUE HD95
    # --------------------------------------------------------

    return np.percentile(all_distances, 95)

# ============================================================
# OVERLAY FUNCTION
# ============================================================

def create_overlay(img, gt, pred):

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --------------------------------------------------------
    # TRUE POSITIVE = GREEN
    # --------------------------------------------------------

    overlay[(gt == 1) & (pred == 1)] = [0,255,0]

    # --------------------------------------------------------
    # FALSE NEGATIVE = RED
    # --------------------------------------------------------

    overlay[(gt == 1) & (pred == 0)] = [0,0,255]

    # --------------------------------------------------------
    # FALSE POSITIVE = BLUE
    # --------------------------------------------------------

    overlay[(gt == 0) & (pred == 1)] = [255,0,0]

    return overlay

# ============================================================
# EVALUATION
# ============================================================

dice_scores = []
hd_scores = []

sample_count = 0

print("\nStarting evaluation...\n")

for video in os.listdir(FRAME_DIR):

    v_frame = os.path.join(FRAME_DIR, video)
    v_mask  = os.path.join(MASK_DIR, video)

    if not os.path.exists(v_mask):
        continue

    for file in os.listdir(v_frame):

        img_path = os.path.join(v_frame, file)

        mask_path = os.path.join(
            v_mask,
            file.replace(".png", "_mask.png")
        )

        # ----------------------------------------------------
        # CHECK MASK
        # ----------------------------------------------------

        if not os.path.exists(mask_path):

            print("Missing mask:", mask_path)

            continue

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        img_norm = img.astype(np.float32) / 255.0

        x = np.expand_dims(
            img_norm,
            axis=(0,-1)
        )

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        gt = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        gt = (gt > 0).astype(np.uint8)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        pred = model.predict(
            x,
            verbose=0
        )[0,:,:,0]

        pred = (pred > 0.5).astype(np.uint8)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        d = dice_coef(gt, pred)
        h = hd95(gt, pred)

        dice_scores.append(d)
        hd_scores.append(h)

        # ----------------------------------------------------
        # SAVE OVERLAY
        # ----------------------------------------------------

        overlay = create_overlay(
            img,
            gt,
            pred
        )

        save_name = f"{video}_{file}"

        cv2.imwrite(
            os.path.join(OVERLAY_DIR, save_name),
            overlay
        )

        sample_count += 1

# ============================================================
# FINAL RESULTS
# ============================================================

final_dice = np.mean(dice_scores)
final_hd95 = np.mean(hd_scores)

print("\n================================================")
print("ATTENTION U-NET TRUE HD95 RESULTS")
print("================================================")

print("Total Samples :", sample_count)

print("Dice Score :", final_dice)
print("TRUE HD95  :", final_hd95)

# ============================================================
# SAVE TXT REPORT
# ============================================================

report_path = os.path.join(
    SAVE_DIR,
    "results.txt"
)

with open(report_path, "w") as f:

    f.write("=====================================\n")
    f.write("ATTENTION U-NET TRUE HD95 RESULTS\n")
    f.write("=====================================\n\n")

    f.write(f"Total Samples Evaluated : {sample_count}\n\n")

    f.write(f"Average Dice Score : {final_dice:.4f}\n")
    f.write(f"Average TRUE HD95  : {final_hd95:.4f}\n")

print("\n✅ Results saved")
print("✅ Overlay images saved")
