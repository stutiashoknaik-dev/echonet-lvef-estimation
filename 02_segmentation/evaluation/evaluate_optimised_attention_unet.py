# ============================================================
# OPTIMIZED ATTENTION U-NET EVALUATION
# DICE + IOU + TRUE HD95 + FULL OVERLAYS
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf

from scipy.spatial.distance import cdist
from tensorflow.keras.models import load_model

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

IMG_SIZE = 112

# ============================================================
# VALIDATION PATHS
# ============================================================

FRAME_DIR = os.path.join(
    BASE_PATH,
    "data/frames/val"
)

MASK_DIR = os.path.join(
    BASE_PATH,
    "data/masks/val"
)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "attention_final_model.keras"

# ============================================================
# SAVE PATHS
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/optimized_attention_full_eval"
)

OVERLAY_DIR = os.path.join(
    SAVE_DIR,
    "overlays"
)

PRED_DIR = os.path.join(
    SAVE_DIR,
    "predictions"
)

os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ============================================================
# DICE
# ============================================================

def dice_score(y_true, y_pred):

    smooth = 1e-6

    intersection = np.sum(
        y_true * y_pred
    )

    return (

        2. * intersection + smooth

    ) / (

        np.sum(y_true) +
        np.sum(y_pred) +
        smooth
    )

# ============================================================
# IOU
# ============================================================

def iou_score(y_true, y_pred):

    smooth = 1e-6

    intersection = np.sum(
        y_true * y_pred
    )

    union = (

        np.sum(y_true) +
        np.sum(y_pred) -
        intersection
    )

    return (

        intersection + smooth

    ) / (

        union + smooth
    )

# ============================================================
# TRUE HD95
# ============================================================

def true_hd95(gt, pred):

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

    distances = cdist(
        gt_pts,
        pred_pts
    )

    # --------------------------------------------------------
    # GT → PRED
    # --------------------------------------------------------

    gt_to_pred = np.min(
        distances,
        axis=1
    )

    # --------------------------------------------------------
    # PRED → GT
    # --------------------------------------------------------

    pred_to_gt = np.min(
        distances,
        axis=0
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    all_distances = np.concatenate([

        gt_to_pred,
        pred_to_gt

    ])

    # --------------------------------------------------------
    # TRUE HD95
    # --------------------------------------------------------

    return np.percentile(
        all_distances,
        95
    )

# ============================================================
# CUSTOM OBJECTS
# ============================================================

def dice_coef(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (

        2. * intersection + smooth

    ) / (

        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) +
        smooth
    )

# ------------------------------------------------------------

def iou_metric(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (

        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) -
        intersection
    )

    return (

        intersection + smooth

    ) / (

        union + smooth
    )

# ------------------------------------------------------------

def combined_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(bce)

    smooth = 1e-6

    intersection = tf.reduce_sum(

        y_true * y_pred,

        axis=[1,2,3]
    )

    union = (

        tf.reduce_sum(y_true, axis=[1,2,3]) +

        tf.reduce_sum(y_pred, axis=[1,2,3])

    )

    dice = (

        2. * intersection + smooth

    ) / (

        union + smooth
    )

    dice_loss = 1 - dice

    iou = (

        intersection + smooth

    ) / (

        union - intersection + smooth
    )

    iou_loss = 1 - iou

    total_loss = (

        bce +

        tf.reduce_mean(dice_loss) +

        0.5 * tf.reduce_mean(iou_loss)

    )

    return total_loss

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Optimized Attention U-Net...\n")

model = load_model(

    MODEL_PATH,

    custom_objects={

        "combined_loss": combined_loss,
        "dice_coef": dice_coef,
        "iou_metric": iou_metric
    },

    compile=False
)

print("✅ Model Loaded")

# ============================================================
# OVERLAY FUNCTION
# ============================================================

def create_overlay(img, gt, pred):

    # ========================================================
    # BASE IMAGE
    # ========================================================

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    # ========================================================
    # TRUE POSITIVE = GREEN
    # ========================================================

    overlay[(gt == 1) & (pred == 1)] = [0,255,0]

    # ========================================================
    # FALSE NEGATIVE = RED
    # GT PRESENT BUT PRED MISSING
    # ========================================================

    overlay[(gt == 1) & (pred == 0)] = [0,0,255]

    # ========================================================
    # FALSE POSITIVE = BLUE
    # PRED PRESENT BUT GT MISSING
    # ========================================================

    overlay[(gt == 0) & (pred == 1)] = [255,0,0]

    # ========================================================
    # PREDICTED CONTOUR BOUNDARY = RED LINE
    # ========================================================

    pred_mask = (pred * 255).astype(np.uint8)

    contours, _ = cv2.findContours(

        pred_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(

        overlay,

        contours,

        -1,

        (0,0,255),

        1
    )

    return overlay

# ============================================================
# EVALUATION
# ============================================================

dice_scores = []
iou_scores = []
hd95_scores = []

sample_count = 0

print("\nStarting Evaluation...\n")

# ============================================================
# LOOP
# ============================================================

for video in os.listdir(FRAME_DIR):

    v_frame = os.path.join(
        FRAME_DIR,
        video
    )

    v_mask = os.path.join(
        MASK_DIR,
        video
    )

    if not os.path.exists(v_mask):
        continue

    # ========================================================
    # FILE LOOP
    # ========================================================

    for file in os.listdir(v_frame):

        img_path = os.path.join(
            v_frame,
            file
        )

        mask_path = os.path.join(

            v_mask,

            file.replace(
                ".png",
                "_mask.png"
            )
        )

        # ====================================================
        # CHECK MASK
        # ====================================================

        if not os.path.exists(mask_path):

            print("Missing mask:", mask_path)

            continue

        # ====================================================
        # IMAGE
        # ====================================================

        img = cv2.imread(

            img_path,

            cv2.IMREAD_GRAYSCALE
        )

        original_img = img.copy()

        img_norm = img.astype(
            np.float32
        ) / 255.0

        x = np.expand_dims(
            img_norm,
            axis=(0,-1)
        )

        # ====================================================
        # GT MASK
        # ====================================================

        gt = cv2.imread(

            mask_path,

            cv2.IMREAD_GRAYSCALE
        )

        gt = (
            gt > 0
        ).astype(np.uint8)

        # ====================================================
        # PREDICTION
        # ====================================================

        pred = model.predict(

            x,

            verbose=0
        )[0,:,:,0]

        pred = (
            pred > 0.5
        ).astype(np.uint8)

        # ====================================================
        # METRICS
        # ====================================================

        d = dice_score(
            gt,
            pred
        )

        iou = iou_score(
            gt,
            pred
        )

        hd = true_hd95(
            gt,
            pred
        )

        dice_scores.append(d)
        iou_scores.append(iou)
        hd95_scores.append(hd)

        # ====================================================
        # SAVE PREDICTION MASK
        # ====================================================

        pred_save = (pred * 255).astype(np.uint8)

        save_name = f"{video}_{file}"

        cv2.imwrite(

            os.path.join(
                PRED_DIR,
                save_name
            ),

            pred_save
        )

        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        overlay = create_overlay(

            original_img,

            gt,

            pred
        )

        cv2.imwrite(

            os.path.join(
                OVERLAY_DIR,
                save_name
            ),

            overlay
        )

        sample_count += 1

# ============================================================
# FINAL RESULTS
# ============================================================

final_dice = np.mean(dice_scores)
final_iou = np.mean(iou_scores)
final_hd95 = np.mean(hd95_scores)

# ============================================================
# SAVE RESULTS
# ============================================================

result_path = os.path.join(
    SAVE_DIR,
    "results.txt"
)

with open(result_path, "w") as f:

    f.write("========================================\n")
    f.write("OPTIMIZED ATTENTION U-NET RESULTS\n")
    f.write("========================================\n\n")

    f.write(f"Total Samples : {sample_count}\n\n")

    f.write(f"Dice Score : {final_dice:.4f}\n")
    f.write(f"IoU Score  : {final_iou:.4f}\n")
    f.write(f"TRUE HD95  : {final_hd95:.4f}\n")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n======================================")
print("FINAL EVALUATION RESULTS")
print("======================================")

print(f"Total Samples : {sample_count}")

print(f"\nDice Score : {final_dice:.4f}")
print(f"IoU Score  : {final_iou:.4f}")
print(f"TRUE HD95  : {final_hd95:.4f}")

print("\n✅ Results saved")
print("✅ Overlay images saved")
print("✅ Prediction masks saved")
