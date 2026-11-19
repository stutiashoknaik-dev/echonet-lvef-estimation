# ============================================================
# FINAL MASK GENERATION (CORRECT + STABLE)
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FRAME_PATH = os.path.join(BASE_PATH, "data/frames")
MASK_PATH  = os.path.join(BASE_PATH, "data/masks")
CSV_PATH   = os.path.join(BASE_PATH, "data/VolumeTracings.csv")

IMG_SIZE = 112

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)
print("CSV Loaded")

# ============================================================
# FIXED FRAME ID EXTRACTOR (VERY IMPORTANT)
# ============================================================

def get_frame_id(img_name):
    parts = img_name.replace(".png", "").split("_")

    for p in parts:
        if p.isdigit():
            return int(p)

    return None


# ============================================================
# CREATE MASK FROM GT
# ============================================================

def create_mask(group, shape):

    pts1 = group[["X1", "Y1"]].values.astype(np.int32)
    pts2 = group[["X2", "Y2"]].values.astype(np.int32)

    # Proper polygon (NO triangle issue)
    pts = np.concatenate([pts1, pts2[::-1]], axis=0)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask


# ============================================================
# APPLY AUGMENTATION
# ============================================================

def apply_augmentation(mask, aug_type):

    if aug_type == "flip":
        return cv2.flip(mask, 1)

    elif aug_type == "rot":
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1)
        return cv2.warpAffine(mask, M, (w, h))

    return mask


# ============================================================
# MAIN MASK GENERATION
# ============================================================

def generate_masks(split):

    print(f"\nProcessing {split}...")

    FRAME_DIR = os.path.join(FRAME_PATH, split)
    SAVE_DIR  = os.path.join(MASK_PATH, split)

    for video in os.listdir(FRAME_DIR):

        video_dir = os.path.join(FRAME_DIR, video)
        save_dir  = os.path.join(SAVE_DIR, video)

        os.makedirs(save_dir, exist_ok=True)

        # get CSV rows for this video
        video_csv = df[df["FileName"] == video + ".avi"]

        if video_csv.empty:
            continue

        grouped = video_csv.groupby("Frame")

        for img_name in os.listdir(video_dir):

            img_path = os.path.join(video_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 🔥 FIXED FRAME ID EXTRACTION
            frame_id = get_frame_id(img_name)

            if frame_id is None:
                continue

            if frame_id not in grouped.groups:
                continue

            group = grouped.get_group(frame_id)

            # create base mask
            mask = create_mask(group, (IMG_SIZE, IMG_SIZE))

            # apply augmentation if needed
            if "_flip" in img_name:
                mask = apply_augmentation(mask, "flip")

            elif "_rot" in img_name:
                mask = apply_augmentation(mask, "rot")

            # save mask
            save_name = img_name.replace(".png", "_mask.png")
            cv2.imwrite(os.path.join(save_dir, save_name), mask)

    print(f"{split} done!")


# ============================================================
# RUN
# ============================================================


generate_masks("train")
generate_masks("val")

print("\n✅ ALL MASKS GENERATED SUCCESSFULLY")
