import os
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FILELIST_CSV = os.path.join(BASE_PATH, "data", "FileList.csv")
TEST_ROOT = os.path.join(BASE_PATH, "outputs", "adaptive_phase_selection")
MODEL_PATH = os.path.join(BASE_PATH, "outputs", "ef_model", "xgboost_ef_model.pkl")

SAVE_DIR = os.path.join(BASE_PATH, "outputs", "ef_test_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def normalize_name(x):
    x = str(x).strip()
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return x.lower()

def normalize_split(x):
    x = str(x).strip().lower()
    if x in ["train", "training"]:
        return "TRAIN"
    if x in ["val", "valid", "validation"]:
        return "VAL"
    if x in ["test"]:
        return "TEST"
    return x.upper()

def largest_connected_component(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(binary_mask)
    cleaned[labels == largest_label] = 1
    return cleaned

def find_mask_file(phase_dir, phase_prefix):
    """
    Pick only the mask file, not the frame file.
    Example:
        ED_mask_213.png
        ES_mask_48.png
    """
    if not os.path.exists(phase_dir):
        return None

    candidates = []
    phase_prefix = phase_prefix.lower()

    for fname in os.listdir(phase_dir):
        low = fname.lower()

        if not low.endswith(".png"):
            continue

        if "mask" not in low:
            continue

        if not low.startswith(phase_prefix):
            continue

        if "frame" in low:
            continue

        if "flip" in low or "rot" in low:
            continue

        candidates.append(fname)

    if not candidates:
        return None

    candidates.sort()
    return os.path.join(phase_dir, candidates[0])

def contour_features_from_mask(mask_path):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    mask = (img > 127).astype(np.uint8)
    mask = largest_connected_component(mask)

    area = int(np.sum(mask))
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

def build_folder_lookup(root_dir):
    lookup = {}
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            lookup[normalize_name(folder)] = folder
    return lookup

# ============================================================
# LOAD FILELIST
# ============================================================

filelist_df = pd.read_csv(FILELIST_CSV)
filelist_df["FileName_norm"] = filelist_df["FileName"].apply(normalize_name)
filelist_df["Split"] = filelist_df["Split"].apply(normalize_split)
filelist_df["EF"] = pd.to_numeric(filelist_df["EF"], errors="coerce")

test_df = filelist_df[filelist_df["Split"] == "TEST"].copy()

ef_lookup = (
    test_df.dropna(subset=["EF"])
    .drop_duplicates(subset=["FileName_norm"])
    .set_index("FileName_norm")["EF"]
    .to_dict()
)

folder_lookup = build_folder_lookup(TEST_ROOT)

common_videos = sorted(set(test_df["FileName_norm"]) & set(folder_lookup.keys()))

print("\n====================================")
print("TEST DATA CHECK")
print("====================================")
print(f"Videos in FileList TEST : {len(test_df)}")
print(f"Folders in TEST_ROOT    : {len(folder_lookup)}")
print(f"Common usable           : {len(common_videos)}")
print(f"Missing in folder       : {len(set(test_df['FileName_norm']) - set(folder_lookup.keys()))}")

# ============================================================
# BUILD TEST FEATURE CSV
# ============================================================

rows = []
skipped = []

for video_key in common_videos:
    video_name = test_df.loc[test_df["FileName_norm"] == video_key, "FileName"].iloc[0]
    actual_folder = folder_lookup[video_key]
    video_dir = os.path.join(TEST_ROOT, actual_folder)

    ed_dir = os.path.join(video_dir, "ED")
    es_dir = os.path.join(video_dir, "ES")

    ed_mask_path = find_mask_file(ed_dir, "ED")
    es_mask_path = find_mask_file(es_dir, "ES")

    if ed_mask_path is None or es_mask_path is None:
        skipped.append(video_name)
        print("Skipped (mask missing):", video_name)
        continue

    ed_feat = contour_features_from_mask(ed_mask_path)
    es_feat = contour_features_from_mask(es_mask_path)

    if ed_feat is None or es_feat is None:
        skipped.append(video_name)
        print("Skipped (feature error):", video_name)
        continue

    if video_key not in ef_lookup:
        skipped.append(video_name)
        print("Skipped (EF missing):", video_name)
        continue

    gt_ef = float(ef_lookup[video_key])

    row = {
        "FileName": video_name,
        "VideoKey": video_key,
        "Split": "TEST",

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

        "Area_Change": ed_feat["Area"] - es_feat["Area"],
        "Area_Ratio": (es_feat["Area"] / ed_feat["Area"]) if ed_feat["Area"] > 0 else 0.0,
        "Perimeter_Change": ed_feat["Perimeter"] - es_feat["Perimeter"],
        "Perimeter_Ratio": (es_feat["Perimeter"] / ed_feat["Perimeter"]) if ed_feat["Perimeter"] > 0 else 0.0,
        "Axis_Ratio_Change": ed_feat["AspectRatio"] - es_feat["AspectRatio"],

        "GT_EF": gt_ef,
        "ED_MaskFile": os.path.basename(ed_mask_path),
        "ES_MaskFile": os.path.basename(es_mask_path),
    }

    rows.append(row)

test_feat_df = pd.DataFrame(rows)
TEST_FEATURE_CSV = os.path.join(SAVE_DIR, "test_ef_features.csv")
test_feat_df.to_csv(TEST_FEATURE_CSV, index=False)

print(f"\nSaved test features: {TEST_FEATURE_CSV}")
print(f"Rows created: {len(test_feat_df)}")
print(f"Skipped: {len(skipped)}")

if len(test_feat_df) == 0:
    raise ValueError("No test features were generated. Check mask file matching.")

# ============================================================
# LOAD MODEL AND PREDICT
# ============================================================

model = joblib.load(MODEL_PATH)

DROP_COLUMNS = [
    "FileName",
    "VideoKey",
    "Split",
    "GT_EF",
    "ED_MaskFile",
    "ES_MaskFile",
]

X_test = test_feat_df.drop(columns=DROP_COLUMNS)
y_test = test_feat_df["GT_EF"].values
y_pred = model.predict(X_test)

pred_df = pd.DataFrame({
    "FileName": test_feat_df["FileName"],
    "GT_EF": y_test,
    "Predicted_EF": y_pred,
    "Absolute_Error": np.abs(y_test - y_pred)
})

PRED_EF_CSV = os.path.join(SAVE_DIR, "test_ef_predictions.csv")
pred_df.to_csv(PRED_EF_CSV, index=False)

# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

try:
    pearson_r, p_value = pearsonr(y_test, y_pred)
except Exception:
    pearson_r, p_value = np.nan, np.nan

metrics_df = pd.DataFrame([{
    "Samples": len(y_test),
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Pearson_r": pearson_r,
    "Pearson_p": p_value
}])

EF_METRICS_CSV = os.path.join(SAVE_DIR, "test_ef_metrics.csv")
metrics_df.to_csv(EF_METRICS_CSV, index=False)

print("\n====================================")
print("TEST EF RESULTS")
print("====================================")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R2 Score  : {r2:.4f}")
print(f"Pearson r : {pearson_r:.4f}")
print(f"Samples   : {len(y_test)}")

print("\nSaved files:")
print(TEST_FEATURE_CSV)
print(PRED_EF_CSV)
print(EF_METRICS_CSV)
