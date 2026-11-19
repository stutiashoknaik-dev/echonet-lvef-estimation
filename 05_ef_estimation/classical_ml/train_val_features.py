import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
FILELIST_CSV = os.path.join(BASE_PATH, "data", "FileList.csv")
MASK_ROOT = os.path.join(BASE_PATH, "data", "masks")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs", "ef_features")
CHECK_DIR = os.path.join(BASE_PATH, "outputs", "dataset_checks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECK_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_name(x):
    """
    Normalize video/folder names for matching.
    """
    x = str(x).strip()
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return x.lower()

def normalize_split(x):
    """
    Normalize split names to TRAIN / VAL / TEST.
    """
    x = str(x).strip().lower()
    if x in ["train", "training"]:
        return "TRAIN"
    if x in ["val", "valid", "validation"]:
        return "VAL"
    if x in ["test"]:
        return "TEST"
    return x.upper()

def largest_connected_component(binary_mask):
    """
    Keep only the largest connected white region.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:
        return binary_mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(binary_mask)
    cleaned[labels == largest_label] = 1
    return cleaned

def find_original_mask(video_dir, phase_prefix):
    """
    Find only the original ED/ES mask:
    - starts with ED or ES
    - ends with _mask.png
    - ignores flip/rot augmented masks
    """
    if not os.path.exists(video_dir):
        return None

    candidates = []
    phase_prefix = phase_prefix.lower()

    for fname in os.listdir(video_dir):
        low = fname.lower()

        if not low.endswith("_mask.png"):
            continue
        if not low.startswith(phase_prefix):
            continue
        if "flip" in low or "rot" in low:
            continue

        candidates.append(fname)

    if not candidates:
        return None

    candidates.sort()
    return os.path.join(video_dir, candidates[0])

def contour_features_from_mask(mask_path):
    """
    Extract geometric features from one binary mask.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Binary mask
    mask = (img > 127).astype(np.uint8)

    # Keep only the main LV component
    mask = largest_connected_component(mask)

    # Area in pixels
    area = int(np.sum(mask))

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            "Area": area,
            "Perimeter": 0.0,
            "BBox_W": 0,
            "BBox_H": 0,
            "BBox_Area": 0.0,
            "Extent": 0.0,
            "Solidity": 0.0,
            "Circularity": 0.0,
            "Centroid_X": 0.0,
            "Centroid_Y": 0.0,
            "Orientation": 0.0,
            "MajorAxis": 0.0,
            "MinorAxis": 0.0,
            "AspectRatio": 0.0,
            "Eccentricity": 0.0,
            "ContourPoints": 0,
        }

    cnt = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(w * h)
    extent = float(area / bbox_area) if bbox_area > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0

    moments = cv2.moments(cnt)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0.0, 0.0

    orientation = 0.0
    major_axis = 0.0
    minor_axis = 0.0

    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        _, axes, angle = cv2.fitEllipse(cnt)
        major_axis = float(max(axes))
        minor_axis = float(min(axes))
        orientation = float(angle)
    else:
        major_axis = float(max(w, h))
        minor_axis = float(min(w, h))

    aspect_ratio = float(major_axis / minor_axis) if minor_axis > 0 else 0.0
    eccentricity = (
        float(np.sqrt(1.0 - (minor_axis / major_axis) ** 2))
        if major_axis > 0 and minor_axis > 0 and major_axis >= minor_axis
        else 0.0
    )

    return {
        "Area": area,
        "Perimeter": perimeter,
        "BBox_W": int(w),
        "BBox_H": int(h),
        "BBox_Area": bbox_area,
        "Extent": extent,
        "Solidity": solidity,
        "Circularity": circularity,
        "Centroid_X": cx,
        "Centroid_Y": cy,
        "Orientation": orientation,
        "MajorAxis": major_axis,
        "MinorAxis": minor_axis,
        "AspectRatio": aspect_ratio,
        "Eccentricity": eccentricity,
        "ContourPoints": int(len(cnt)),
    }

# ============================================================
# LOAD FILELIST
# ============================================================

filelist_df = pd.read_csv(FILELIST_CSV)

required_cols = ["FileName", "Split", "EF"]
missing_cols = [c for c in required_cols if c not in filelist_df.columns]
if missing_cols:
    raise ValueError(f"FileList.csv is missing columns: {missing_cols}")

filelist_df["FileName_norm"] = filelist_df["FileName"].apply(normalize_name)
filelist_df["Split"] = filelist_df["Split"].apply(normalize_split)
filelist_df["EF"] = pd.to_numeric(filelist_df["EF"], errors="coerce")

# Fast lookup for EF labels
ef_lookup = (
    filelist_df.dropna(subset=["EF"])
    .drop_duplicates(subset=["FileName_norm"])
    .set_index("FileName_norm")["EF"]
    .to_dict()
)

# ============================================================
# BUILD FOLDER LOOKUP
# ============================================================

def get_folder_lookup(split_name):
    """
    Map normalized folder name -> actual folder name
    """
    split_dir = os.path.join(MASK_ROOT, split_name.lower())
    lookup = {}

    if not os.path.exists(split_dir):
        return lookup

    for folder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder)
        if os.path.isdir(folder_path):
            lookup[normalize_name(folder)] = folder

    return lookup

# ============================================================
# FEATURE EXTRACTION FOR ONE VIDEO
# ============================================================

def build_row_for_video(video_name, split_name, folder_lookup):
    """
    Create one row per video using original ED and ES masks.
    """
    video_key = normalize_name(video_name)

    if video_key not in folder_lookup:
        return None

    actual_folder = folder_lookup[video_key]
    video_dir = os.path.join(MASK_ROOT, split_name.lower(), actual_folder)

    ed_mask_path = find_original_mask(video_dir, "ED")
    es_mask_path = find_original_mask(video_dir, "ES")

    if ed_mask_path is None or es_mask_path is None:
        return None

    if video_key not in ef_lookup:
        return None

    ed_feat = contour_features_from_mask(ed_mask_path)
    es_feat = contour_features_from_mask(es_mask_path)

    if ed_feat is None or es_feat is None:
        return None

    gt_ef = float(ef_lookup[video_key])

    row = {
        "FileName": video_name,
        "VideoKey": video_key,
        "Split": split_name.upper(),

        # ED features
        "ED_Area": ed_feat["Area"],
        "ED_Perimeter": ed_feat["Perimeter"],
        "ED_BBox_W": ed_feat["BBox_W"],
        "ED_BBox_H": ed_feat["BBox_H"],
        "ED_BBox_Area": ed_feat["BBox_Area"],
        "ED_Extent": ed_feat["Extent"],
        "ED_Solidity": ed_feat["Solidity"],
        "ED_Circularity": ed_feat["Circularity"],
        "ED_Centroid_X": ed_feat["Centroid_X"],
        "ED_Centroid_Y": ed_feat["Centroid_Y"],
        "ED_Orientation": ed_feat["Orientation"],
        "ED_MajorAxis": ed_feat["MajorAxis"],
        "ED_MinorAxis": ed_feat["MinorAxis"],
        "ED_AspectRatio": ed_feat["AspectRatio"],
        "ED_Eccentricity": ed_feat["Eccentricity"],
        "ED_ContourPoints": ed_feat["ContourPoints"],

        # ES features
        "ES_Area": es_feat["Area"],
        "ES_Perimeter": es_feat["Perimeter"],
        "ES_BBox_W": es_feat["BBox_W"],
        "ES_BBox_H": es_feat["BBox_H"],
        "ES_BBox_Area": es_feat["BBox_Area"],
        "ES_Extent": es_feat["Extent"],
        "ES_Solidity": es_feat["Solidity"],
        "ES_Circularity": es_feat["Circularity"],
        "ES_Centroid_X": es_feat["Centroid_X"],
        "ES_Centroid_Y": es_feat["Centroid_Y"],
        "ES_Orientation": es_feat["Orientation"],
        "ES_MajorAxis": es_feat["MajorAxis"],
        "ES_MinorAxis": es_feat["MinorAxis"],
        "ES_AspectRatio": es_feat["AspectRatio"],
        "ES_Eccentricity": es_feat["Eccentricity"],
        "ES_ContourPoints": es_feat["ContourPoints"],

        # Combined features
        "Area_Change": ed_feat["Area"] - es_feat["Area"],
        "Area_Ratio": (es_feat["Area"] / ed_feat["Area"]) if ed_feat["Area"] > 0 else 0.0,
        "Perimeter_Change": ed_feat["Perimeter"] - es_feat["Perimeter"],
        "Perimeter_Ratio": (es_feat["Perimeter"] / ed_feat["Perimeter"]) if ed_feat["Perimeter"] > 0 else 0.0,
        "Axis_Ratio_Change": ed_feat["AspectRatio"] - es_feat["AspectRatio"],

        # Label
        "GT_EF": gt_ef,

        # Traceability
        "ED_MaskFile": os.path.basename(ed_mask_path),
        "ES_MaskFile": os.path.basename(es_mask_path),
    }

    return row

# ============================================================
# PROCESS TRAIN / VAL ONLY
# ============================================================

for split_name in ["TRAIN", "VAL"]:
    split_df = filelist_df[filelist_df["Split"] == split_name].copy()

    folder_lookup = get_folder_lookup(split_name)

    train_or_val_names = set(split_df["FileName_norm"].tolist())
    folder_names = set(folder_lookup.keys())

    common_videos = sorted(train_or_val_names & folder_names)
    missing_in_mask = sorted(train_or_val_names - folder_names)
    extra_in_mask = sorted(folder_names - train_or_val_names)

    # Save split report
    report_df = pd.DataFrame({
        "CommonVideos": pd.Series(common_videos),
        "MissingInMaskFolder": pd.Series(missing_in_mask),
        "ExtraInMaskFolder": pd.Series(extra_in_mask),
    })
    report_path = os.path.join(CHECK_DIR, f"{split_name.lower()}_video_check.csv")
    report_df.to_csv(report_path, index=False)

    print(f"\nProcessing {split_name} split")
    print(f"Videos in FileList : {len(train_or_val_names)}")
    print(f"Videos in Mask Dir : {len(folder_names)}")
    print(f"Common usable      : {len(common_videos)}")
    print(f"Missing in masks   : {len(missing_in_mask)}")

    rows = []
    skipped_no_masks = []
    skipped_no_label = []

    for video_key in common_videos:
        video_name = split_df.loc[split_df["FileName_norm"] == video_key, "FileName"].iloc[0]

        row = build_row_for_video(video_name, split_name, folder_lookup)

        if row is None:
            # Determine why it failed
            if video_key not in ef_lookup:
                skipped_no_label.append(video_name)
            else:
                skipped_no_masks.append(video_name)
            continue

        rows.append(row)

    out_df = pd.DataFrame(rows)

    out_csv = os.path.join(OUTPUT_DIR, f"{split_name.lower()}_ef_features.csv")
    out_df.to_csv(out_csv, index=False)

    # Save skip lists
    pd.DataFrame({"Skipped_NoMaskOrContour": skipped_no_masks}).to_csv(
        os.path.join(CHECK_DIR, f"{split_name.lower()}_skipped_no_masks.csv"),
        index=False
    )
    pd.DataFrame({"Skipped_NoLabel": skipped_no_label}).to_csv(
        os.path.join(CHECK_DIR, f"{split_name.lower()}_skipped_no_label.csv"),
        index=False
    )

    print(f"Saved: {out_csv}")
    print(f"Rows created: {len(out_df)}")

print("\n====================================")
print("FEATURE EXTRACTION COMPLETED")
print("====================================")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Check reports: {CHECK_DIR}")
