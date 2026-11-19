# ============================================================
# GENERATE MASK FEATURES
# TRAIN + VAL

# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

from skimage.measure import regionprops

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

TRAIN_MASK_DIR = os.path.join(
    BASE_PATH,
    "data",
    "masks",
    "train"
)

VAL_MASK_DIR = os.path.join(
    BASE_PATH,
    "data",
    "masks",
    "val"
)

FILELIST_CSV = os.path.join(
    BASE_PATH,
    "data",
    "FileList.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features"
)

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_SAVE = os.path.join(
    SAVE_DIR,
    "train_mask_features.csv"
)

VAL_SAVE = os.path.join(
    SAVE_DIR,
    "val_mask_features.csv"
)

# ============================================================
# FILELIST
# ============================================================

filelist = pd.read_csv(FILELIST_CSV)

filelist["FileName"] = (
    filelist["FileName"]
    .astype(str)
    .str.replace(".avi", "", regex=False)
    .str.strip()
)

ef_map = dict(
    zip(
        filelist["FileName"],
        filelist["EF"]
    )
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mask_features(mask_path):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        return None

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    area = cv2.countNonZero(binary)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    props = regionprops(
        (binary > 0).astype(np.uint8)
    )

    if len(props) == 0:
        return None

    p = props[0]

    major_axis = float(
        p.major_axis_length
    )

    minor_axis = float(
        p.minor_axis_length
    )

    eccentricity = float(
        p.eccentricity
    )

    solidity = float(
        p.solidity
    )

    extent = float(
        p.extent
    )

    equiv_diameter = float(
        p.equivalent_diameter
    )

    convex_area = float(
        p.area_convex
    )

    aspect_ratio = (
        major_axis /
        (minor_axis + 1e-6)
    )

    return {

        "Area": float(area),

        "Perimeter": float(perimeter),

        "MajorAxis": major_axis,

        "MinorAxis": minor_axis,

        "AspectRatio": aspect_ratio,

        "Extent": extent,

        "Solidity": solidity,

        "Eccentricity": eccentricity,

        "EquivalentDiameter": equiv_diameter,

        "ConvexArea": convex_area
    }

# ============================================================
# FIND TRUE ED / ES MASKS
# IGNORE AUGMENTATIONS
# ============================================================

def get_mask_paths(folder):

    ed_mask = None
    es_mask = None

    for file in os.listdir(folder):

        lower = file.lower()

        if not lower.endswith(".png"):
            continue

        if "flip" in lower:
            continue

        if "rot" in lower:
            continue

        if lower.startswith("ed_"):
            ed_mask = os.path.join(folder, file)

        elif lower.startswith("es_"):
            es_mask = os.path.join(folder, file)

    return ed_mask, es_mask

# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(mask_root):

    results = []

    folders = os.listdir(mask_root)

    print(
        f"\nVideos Found : {len(folders)}"
    )

    for idx, video_id in enumerate(folders):

        if idx % 250 == 0:
            print(
                f"{idx}/{len(folders)}"
            )

        video_folder = os.path.join(
            mask_root,
            video_id
        )

        if not os.path.isdir(video_folder):
            continue

        if video_id not in ef_map:
            continue

        ef = ef_map[video_id]

        ed_mask, es_mask = get_mask_paths(
            video_folder
        )

        if ed_mask is None:
            continue

        if es_mask is None:
            continue

        ed_feat = extract_mask_features(
            ed_mask
        )

        es_feat = extract_mask_features(
            es_mask
        )

        if ed_feat is None:
            continue

        if es_feat is None:
            continue

        row = {

            "FileName": video_id,

            "EF": float(ef)
        }

        # -----------------------------
        # ED FEATURES
        # -----------------------------

        for k, v in ed_feat.items():

            row[
                f"ED_{k}"
            ] = v

        # -----------------------------
        # ES FEATURES
        # -----------------------------

        for k, v in es_feat.items():

            row[
                f"ES_{k}"
            ] = v

        # -----------------------------
        # DELTA FEATURES
        # -----------------------------

        row["Area_Change"] = (
            ed_feat["Area"]
            -
            es_feat["Area"]
        )

        row["Perimeter_Change"] = (
            ed_feat["Perimeter"]
            -
            es_feat["Perimeter"]
        )

        row["Area_Ratio"] = (
            ed_feat["Area"]
            /
            (es_feat["Area"] + 1e-6)
        )

        row["Perimeter_Ratio"] = (
            ed_feat["Perimeter"]
            /
            (es_feat["Perimeter"] + 1e-6)
        )

        results.append(row)

    return pd.DataFrame(results)

# ============================================================
# TRAIN
# ============================================================

print("\nGenerating TRAIN Features...")

train_df = process_split(
    TRAIN_MASK_DIR
)

train_df.to_csv(
    TRAIN_SAVE,
    index=False
)

# ============================================================
# VAL
# ============================================================

print("\nGenerating VAL Features...")

val_df = process_split(
    VAL_MASK_DIR
)

val_df.to_csv(
    VAL_SAVE,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n====================================")
print("MASK FEATURE GENERATION COMPLETE")
print("====================================")

print(
    "\nTrain Samples :",
    len(train_df)
)

print(
    "Val Samples   :",
    len(val_df)
)

print("\nSaved Files:")

print(TRAIN_SAVE)

print(VAL_SAVE)
