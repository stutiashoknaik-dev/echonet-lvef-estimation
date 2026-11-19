# ============================================================
# EDGE-AWARE FINE-TUNED MODEL EVALUATION
# WITH SOFT BOUNDARY REFINEMENT
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
# MODEL PATH
# ============================================================

MODEL_PATH = "attention_unet_edge_finetuned.keras"

# ============================================================
# VALIDATION DATA
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
# SAVE PATHS
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/edge_fine_tuned_eval"
)

MASK_SAVE = os.path.join(
    SAVE_DIR,
    "masks"
)

OVERLAY_SAVE = os.path.join(
    SAVE_DIR,
    "overlays"
)

os.makedirs(MASK_SAVE, exist_ok=True)
os.makedirs(OVERLAY_SAVE, exist_ok=True)

# ============================================================
# DICE METRIC
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

# ============================================================
# IOU METRIC
# ============================================================

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

# ============================================================
# EDGE LOSS
# ============================================================

def edge_loss(y_true, y_pred):

    gt_edges = tf.image.sobel_edges(y_true)

    pred_edges = tf.image.sobel_edges(y_pred)

    loss = tf.reduce_mean(
        tf.abs(gt_edges - pred_edges)
    )

    return loss

# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(y_true, y_pred):

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

    return 1 - tf.reduce_mean(dice)

# ============================================================
# IOU LOSS
# ============================================================

def iou_loss(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1,2,3]
    )

    union = (
        tf.reduce_sum(y_true, axis=[1,2,3]) +
        tf.reduce_sum(y_pred, axis=[1,2,3]) -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return 1 - tf.reduce_mean(iou)

# ============================================================
# FINAL LOSS
# ============================================================

def combined_boundary_loss(y_true, y_pred):

    dl = dice_loss(y_true, y_pred)

    il = iou_loss(y_true, y_pred)

    el = edge_loss(y_true, y_pred)

    total = (

        0.5 * dl +

        0.3 * il +

        0.2 * el

    )

    return total

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Edge-Aware Fine-Tuned Model...\n")

model = load_model(

    MODEL_PATH,

    custom_objects={

        "combined_boundary_loss": combined_boundary_loss,
        "dice_coef": dice_coef,
        "iou_metric": iou_metric

    },

    compile=False
)

print("✅ Model Loaded")

# ============================================================
# TRUE DICE
# ============================================================

def dice_score(gt, pred):

    smooth = 1e-6

    intersection = np.sum(gt * pred)

    return (
        2 * intersection + smooth
    ) / (
        np.sum(gt) +
        np.sum(pred) +
        smooth
    )

# ============================================================
# TRUE IOU
# ============================================================

def iou_score(gt, pred):

    smooth = 1e-6

    intersection = np.sum(gt * pred)

    union = (
        np.sum(gt) +
        np.sum(pred) -
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

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 999

    distances = cdist(
        gt_pts,
        pred_pts
    )

    d1 = np.min(distances, axis=1)
    d2 = np.min(distances, axis=0)

    return np.percentile(
        np.concatenate([d1, d2]),
        95
    )

# ============================================================
# SOFT REFINEMENT
# ============================================================

def refine_mask(mask):

    mask = (mask > 0).astype(np.uint8)

    # ========================================================
    # VERY LIGHT MORPH CLOSE
    # ========================================================

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3,3)
    )

    refined = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel
    )

    # ========================================================
    # VERY LIGHT BLUR
    # ========================================================

    blurred = cv2.GaussianBlur(

        refined.astype(np.float32),

        (3,3),

        0.3
    )

    refined = (blurred > 0.5).astype(np.uint8)

    # ========================================================
    # KEEP LARGEST COMPONENT
    # ========================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined,
        connectivity=8
    )

    if num_labels > 1:

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        clean = np.zeros_like(refined)

        clean[labels == largest] = 1

        refined = clean

    return refined.astype(np.uint8)

# ============================================================
# OVERLAY
# ============================================================

def create_overlay(img, gt, pred):

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
    # ========================================================

    overlay[(gt == 1) & (pred == 0)] = [0,0,255]

    # ========================================================
    # FALSE POSITIVE = BLUE
    # ========================================================

    overlay[(gt == 0) & (pred == 1)] = [255,0,0]

    # ========================================================
    # CONTOUR LINE
    # ========================================================

    contours, _ = cv2.findContours(

        (pred * 255).astype(np.uint8),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_NONE
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
# EVALUATION LOOP
# ============================================================

dice_scores = []
iou_scores = []
hd95_scores = []

total = 0

print("\nStarting Evaluation...\n")

# ============================================================
# VIDEO LOOP
# ============================================================

for video in os.listdir(FRAME_DIR):

    video_frame = os.path.join(
        FRAME_DIR,
        video
    )

    video_mask = os.path.join(
        MASK_DIR,
        video
    )

    if not os.path.exists(video_mask):
        continue

    # ========================================================
    # FRAME LOOP
    # ========================================================

    for file in os.listdir(video_frame):

        img_path = os.path.join(
            video_frame,
            file
        )

        mask_path = os.path.join(

            video_mask,

            file.replace(
                ".png",
                "_mask.png"
            )
        )

        if not os.path.exists(mask_path):
            continue

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        original = img.copy()

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        img = img.astype(np.float32) / 255.0

        x = np.expand_dims(
            img,
            axis=(0,-1)
        )

        # ====================================================
        # LOAD GT
        # ====================================================

        gt = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        gt = cv2.resize(
            gt,
            (IMG_SIZE, IMG_SIZE)
        )

        gt = (gt > 0).astype(np.uint8)

        # ====================================================
        # PREDICT
        # ====================================================

        pred = model.predict(
            x,
            verbose=0
        )[0,:,:,0]

        pred = (
            pred > 0.5
        ).astype(np.uint8)

        # ====================================================
        # SOFT REFINEMENT
        # ====================================================

        pred = refine_mask(pred)

        # ====================================================
        # METRICS
        # ====================================================

        d = dice_score(gt, pred)
        i = iou_score(gt, pred)
        h = true_hd95(gt, pred)

        dice_scores.append(d)
        iou_scores.append(i)
        hd95_scores.append(h)

        # ====================================================
        # SAVE MASK
        # ====================================================

        save_name = f"{video}_{file}"

        cv2.imwrite(

            os.path.join(
                MASK_SAVE,
                save_name
            ),

            pred * 255
        )

        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = create_overlay(
            original,
            gt,
            pred
        )

        cv2.imwrite(

            os.path.join(
                OVERLAY_SAVE,
                save_name
            ),

            overlay
        )

        total += 1

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
    f.write("EDGE-AWARE FINE-TUNED RESULTS\n")
    f.write("========================================\n\n")

    f.write(f"Total Samples : {total}\n\n")

    f.write(f"Dice Score : {final_dice:.4f}\n")
    f.write(f"IoU Score  : {final_iou:.4f}\n")
    f.write(f"TRUE HD95  : {final_hd95:.4f}\n")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n======================================")
print("EDGE-AWARE FINE-TUNED RESULTS")
print("======================================")

print(f"Total Samples : {total}")

print(f"\nDice Score : {final_dice:.4f}")
print(f"IoU Score  : {final_iou:.4f}")
print(f"TRUE HD95  : {final_hd95:.4f}")

print("\n✅ Masks saved")
print("✅ Overlays saved")
print("✅ Results saved")
