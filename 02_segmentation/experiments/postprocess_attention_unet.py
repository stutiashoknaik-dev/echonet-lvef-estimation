# ============================================================
# # LIGHT POST-PROCESSING FOR OPTIMIZED ATTENTION U-NET
# # FOCUS:
# # - Smooth boundaries
# # - Reduce jagged edges
# # - Preserve LV anatomy
# # - Minimal distortion
# # ============================================================

import os
import cv2
import numpy as np

from scipy.spatial.distance import cdist

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT PREDICTIONS
# ============================================================

PRED_DIR = os.path.join(
    BASE_PATH,
    "outputs/optimized_attention_full_eval/predictions"
)

# ============================================================
# GT MASKS
# ============================================================

GT_DIR = os.path.join(
    BASE_PATH,
    "data/masks/val"
)

# ============================================================
# ORIGINAL FRAMES
# ============================================================

FRAME_DIR = os.path.join(
    BASE_PATH,
    "data/frames/val"
)

# ============================================================
# SAVE PATHS
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/optimized_attention_post_processing"
)

POST_MASK_DIR = os.path.join(
    SAVE_DIR,
    "post_processed_masks"
)

OVERLAY_DIR = os.path.join(
    SAVE_DIR,
    "overlays"
)

os.makedirs(POST_MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ============================================================
# METRICS
# ============================================================

def dice_score(gt, pred):

    smooth = 1e-6

    intersection = np.sum(gt * pred)

    return (

        2.0 * intersection + smooth

    ) / (

        np.sum(gt) +
        np.sum(pred) +
        smooth
    )

# ------------------------------------------------------------

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

# ------------------------------------------------------------

def true_hd95(gt, pred):

    gt_pts = np.argwhere(gt > 0)
    pred_pts = np.argwhere(pred > 0)

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 999

    distances = cdist(
        gt_pts,
        pred_pts
    )

    gt_to_pred = np.min(
        distances,
        axis=1
    )

    pred_to_gt = np.min(
        distances,
        axis=0
    )

    all_distances = np.concatenate([

        gt_to_pred,
        pred_to_gt

    ])

    return np.percentile(
        all_distances,
        95
    )

# ============================================================
# OVERLAY FUNCTION
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
    # CONTOUR BOUNDARY
    # ========================================================

    pred_uint8 = (pred * 255).astype(np.uint8)

    contours, _ = cv2.findContours(

        pred_uint8,

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
# LIGHT POST-PROCESSING
# ============================================================

def post_process(mask):

    mask = (mask > 0).astype(np.uint8)

    # ========================================================
    # STEP 1:
    # SMALL MORPHOLOGICAL CLOSING
    # removes tiny gaps/jagged edges
    # ========================================================

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3,3)
    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel
    )

    # ========================================================
    # STEP 2:
    # LIGHT GAUSSIAN SMOOTHING
    # ========================================================

    blurred = cv2.GaussianBlur(

        mask.astype(np.float32),

        (3,3),

        0
    )

    # ========================================================
    # STEP 3:
    # RE-THRESHOLD
    # ========================================================

    mask = (blurred > 0.5).astype(np.uint8)

    # ========================================================
    # STEP 4:
    # KEEP LARGEST COMPONENT ONLY
    # ========================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        final_mask = np.zeros_like(mask)

        final_mask[labels == largest] = 1

        mask = final_mask

    # ========================================================
    # STEP 5:
    # VERY LIGHT CONTOUR SMOOTHING
    # ========================================================

    contour_mask = np.zeros_like(mask)

    contours, _ = cv2.findContours(

        (mask * 255).astype(np.uint8),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = 0.002 * cv2.arcLength(
            largest_contour,
            True
        )

        smooth_contour = cv2.approxPolyDP(

            largest_contour,

            epsilon,

            True
        )

        cv2.drawContours(

            contour_mask,

            [smooth_contour],

            -1,

            1,

            thickness=-1
        )

        mask = contour_mask

    return mask.astype(np.uint8)

# ============================================================
# MAIN LOOP
# ============================================================

dice_scores = []
iou_scores = []
hd95_scores = []

total_samples = 0

print("\nStarting Light Post-Processing...\n")

# ============================================================
# LOOP THROUGH PREDICTIONS
# ============================================================

for file in os.listdir(PRED_DIR):

    pred_path = os.path.join(
        PRED_DIR,
        file
    )

    # ========================================================
    # EXTRACT VIDEO + FRAME NAME
    # ========================================================

    split_idx = file.find("_ED_")

    if split_idx == -1:
        split_idx = file.find("_ES_")

    if split_idx == -1:
        continue

    video_name = file[:split_idx]

    frame_name = file[len(video_name)+1:]

    # ========================================================
    # PATHS
    # ========================================================

    gt_path = os.path.join(

        GT_DIR,
        video_name,

        frame_name.replace(
            ".png",
            "_mask.png"
        )
    )

    frame_path = os.path.join(

        FRAME_DIR,
        video_name,
        frame_name
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(gt_path):
        continue

    if not os.path.exists(frame_path):
        continue

    # ========================================================
    # LOAD
    # ========================================================

    pred = cv2.imread(

        pred_path,

        cv2.IMREAD_GRAYSCALE
    )

    gt = cv2.imread(

        gt_path,

        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.imread(

        frame_path,

        cv2.IMREAD_GRAYSCALE
    )

    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # ========================================================
    # POST-PROCESS
    # ========================================================

    post_mask = post_process(pred)

    # ========================================================
    # METRICS
    # ========================================================

    dice = dice_score(
        gt,
        post_mask
    )

    iou = iou_score(
        gt,
        post_mask
    )

    hd95 = true_hd95(
        gt,
        post_mask
    )

    dice_scores.append(dice)
    iou_scores.append(iou)
    hd95_scores.append(hd95)

    # ========================================================
    # SAVE MASK
    # ========================================================

    save_mask = (post_mask * 255).astype(np.uint8)

    cv2.imwrite(

        os.path.join(
            POST_MASK_DIR,
            file
        ),

        save_mask
    )

    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = create_overlay(

        img,
        gt,
        post_mask
    )

    cv2.imwrite(

        os.path.join(
            OVERLAY_DIR,
            file
        ),

        overlay
    )

    total_samples += 1

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
    f.write("OPTIMIZED ATTENTION POST-PROCESSING\n")
    f.write("========================================\n\n")

    f.write(f"Total Samples : {total_samples}\n\n")

    f.write(f"Dice Score : {final_dice:.4f}\n")
    f.write(f"IoU Score  : {final_iou:.4f}\n")
    f.write(f"TRUE HD95  : {final_hd95:.4f}\n")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n======================================")
print("POST-PROCESSING RESULTS")
print("======================================")

print(f"Total Samples : {total_samples}")

print(f"\nDice Score : {final_dice:.4f}")
print(f"IoU Score  : {final_iou:.4f}")
print(f"TRUE HD95  : {final_hd95:.4f}")

print("\n✅ Post-processed masks saved")
print("✅ Overlay images saved")
print("✅ Results saved")
