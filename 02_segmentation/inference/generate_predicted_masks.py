# ============================================================
# GENERATE PREDICTED MASKS WITH THE EDGE-AWARE FINE-TUNED ATTENTION U-NET
# Usage: python generate_predicted_masks.py [test | val | val_full]
# Writes per-frame masks and overlays to outputs/<Split>_predictions_edgeaware/
# ============================================================
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

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
# TEST FRAME DIRECTORY
# ============================================================

# Which split to run: python generate_predicted_masks.py [test | val | val_full]
# (replaces three copies of this script that differed only in these two folders)
SPLITS = {
    "test":     ("data/frames/test",      "outputs/Test_predictions_edgeaware"),
    "val":      ("data/frames/val",       "outputs/Val_predictions_edgeaware"),
    "val_full": ("data/frames_full/val",  "outputs/Val_predictions_full_edgeaware"),
}
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
if SPLIT not in SPLITS:
    raise SystemExit(f"Unknown split {SPLIT!r}; choose one of {list(SPLITS)}")

FRAME_DIR = os.path.join(BASE_PATH, SPLITS[SPLIT][0])

# ============================================================
# SAVE DIRECTORIES
# ============================================================

SAVE_DIR = os.path.join(BASE_PATH, SPLITS[SPLIT][1])

MASK_SAVE_DIR = os.path.join(
    SAVE_DIR,
    "masks"
)

OVERLAY_SAVE_DIR = os.path.join(
    SAVE_DIR,
    "overlays"
)

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(OVERLAY_SAVE_DIR, exist_ok=True)

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
# EDGE-AWARE LOSS
# ============================================================

def edge_loss(y_true, y_pred):

    gt_edges = tf.image.sobel_edges(y_true)

    pred_edges = tf.image.sobel_edges(y_pred)

    loss = tf.reduce_mean(
        tf.abs(gt_edges - pred_edges)
    )

    return loss

# ============================================================
# FINAL COMBINED LOSS
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
# CREATE OVERLAY
# ============================================================

def create_overlay(img, pred):

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    # ========================================================
    # PREDICTED REGION = GREEN
    # ========================================================

    overlay[pred == 1] = [0,255,0]

    # ========================================================
    # DRAW CONTOUR
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
# MAIN LOOP
# ============================================================

total_frames = 0

print("\nGenerating masks for ALL test frames...\n")

# ============================================================
# VIDEO LOOP
# ============================================================

for video in os.listdir(FRAME_DIR):

    video_path = os.path.join(
        FRAME_DIR,
        video
    )

    if not os.path.isdir(video_path):
        continue

    print(f"\nProcessing Video : {video}")

    # ========================================================
    # SAVE VIDEO FOLDERS
    # ========================================================

    save_mask_video = os.path.join(
        MASK_SAVE_DIR,
        video
    )

    save_overlay_video = os.path.join(
        OVERLAY_SAVE_DIR,
        video
    )

    os.makedirs(save_mask_video, exist_ok=True)
    os.makedirs(save_overlay_video, exist_ok=True)

    # ========================================================
    # FRAME LOOP
    # ========================================================

    for file in sorted(os.listdir(video_path)):

        frame_path = os.path.join(
            video_path,
            file
        )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        img = cv2.imread(

            frame_path,

            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        # ====================================================
        # RESIZE
        # ====================================================

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        original = img.copy()

        # ====================================================
        # NORMALIZE
        # ====================================================

        img_norm = img.astype(
            np.float32
        ) / 255.0

        x = np.expand_dims(
            img_norm,
            axis=(0,-1)
        )

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
        # SOFT REFINEMENT
        # ====================================================

        pred = refine_mask(pred)

        # ====================================================
        # SAVE MASK
        # ====================================================

        save_mask = (
            pred * 255
        ).astype(np.uint8)

        cv2.imwrite(

            os.path.join(
                save_mask_video,
                file
            ),

            save_mask
        )

        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = create_overlay(
            original,
            pred
        )

        cv2.imwrite(

            os.path.join(
                save_overlay_video,
                file
            ),

            overlay
        )

        total_frames += 1

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("EDGE-AWARE TEST MASK GENERATION COMPLETE")
print("======================================")

print(f"\nTotal Frames Processed : {total_frames}")

print("\n✅ Masks saved")
print("✅ Overlays saved")
