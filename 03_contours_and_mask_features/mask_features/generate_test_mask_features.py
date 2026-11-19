# ============================================================
# GENERATE TEST MASK FEATURES
# FROM PREDICTED ED / ES MASKS
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

TEST_MASK_ROOT = os.path.join(
    BASE_PATH,
    "outputs",
    "adaptive_phase_selection"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features"
)

os.makedirs(SAVE_DIR, exist_ok=True)

TEST_SAVE = os.path.join(
    SAVE_DIR,
    "test_mask_features.csv"
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mask_features(mask_path):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

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

    contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)

    props = regionprops(
        (binary > 0).astype(np.uint8)
    )

    if len(props) == 0:
        return None

    p = props[0]

    major_axis = float(p.major_axis_length)
    minor_axis = float(p.minor_axis_length)

    aspect_ratio = major_axis / (minor_axis + 1e-6)

    return {

        "Area": float(area),

        "Perimeter": float(perimeter),

        "MajorAxis": major_axis,

        "MinorAxis": minor_axis,

        "AspectRatio": aspect_ratio,

        "Extent": float(p.extent),

        "Solidity": float(p.solidity),

        "Eccentricity": float(p.eccentricity),

        "EquivalentDiameter": float(
            p.equivalent_diameter
        ),

        "ConvexArea": float(
            p.area_convex
        )
    }

# ============================================================
# FIND ED / ES MASKS
# ============================================================
def get_mask_paths(video_folder):

    ed_folder = os.path.join(video_folder, "ED")
    es_folder = os.path.join(video_folder, "ES")

    ed_mask = None
    es_mask = None

    # --------------------------
    # ED MASK
    # --------------------------

    if os.path.exists(ed_folder):

        for file in os.listdir(ed_folder):

            lower = file.lower()

            if (
                lower.startswith("ed_mask")
                and lower.endswith(".png")
            ):
                ed_mask = os.path.join(
                    ed_folder,
                    file
                )
                break

    # --------------------------
    # ES MASK
    # --------------------------

    if os.path.exists(es_folder):

        for file in os.listdir(es_folder):

            lower = file.lower()

            if (
                lower.startswith("es_mask")
                and lower.endswith(".png")
            ):
                es_mask = os.path.join(
                    es_folder,
                    file
                )
                break

    return ed_mask, es_mask

# ============================================================
# PROCESS TEST VIDEOS
# ============================================================

results = []

video_folders = sorted(os.listdir(TEST_MASK_ROOT))

print("\nVideos Found :", len(video_folders))

for idx, video in enumerate(video_folders):

    if idx % 250 == 0:
        print(f"{idx}/{len(video_folders)}")

    video_folder = os.path.join(
        TEST_MASK_ROOT,
        video
    )

    if not os.path.isdir(video_folder):
        continue

    ed_mask, es_mask = get_mask_paths(
        video_folder
    )

    if ed_mask is None or es_mask is None:
        continue

    ed_feat = extract_mask_features(
        ed_mask
    )

    es_feat = extract_mask_features(
        es_mask
    )

    if ed_feat is None or es_feat is None:
        continue

    row = {

        "FileName": video
    }

    # ED Features
    for k, v in ed_feat.items():
        row[f"ED_{k}"] = v

    # ES Features
    for k, v in es_feat.items():
        row[f"ES_{k}"] = v

    # Derived Features

    row["Area_Change"] = (
        ed_feat["Area"]
        - es_feat["Area"]
    )

    row["Perimeter_Change"] = (
        ed_feat["Perimeter"]
        - es_feat["Perimeter"]
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

# ============================================================
# SAVE CSV
# ============================================================

test_df = pd.DataFrame(results)

test_df.to_csv(
    TEST_SAVE,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n====================================")
print("TEST MASK FEATURE GENERATION COMPLETE")
print("====================================")

print("\nSamples :", len(test_df))

print("\nSaved To:")

print(TEST_SAVE)

print("\nColumns:")

print(test_df.columns.tolist())
