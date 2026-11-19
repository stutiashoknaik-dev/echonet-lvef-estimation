# ============================================================
# FINAL HYBRID EF CALCULATION
# MASK AREA + CONTOUR LENGTH
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ------------------------------------------------------------
# PHASE SELECTION MASKS
# ------------------------------------------------------------

PHASE_PATH = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

# ------------------------------------------------------------
# CONTOUR CSV FILES
# ------------------------------------------------------------

ED_CSV = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_apex_based",
    "Predicted_ED_Test_VolumeTracings_ApexBased.csv"
)

ES_CSV = os.path.join(
    BASE_PATH,
   "outputs/test_predicted_contours_apex_based",
    "Predicted_ES_Test_VolumeTracings_ApexBased.csv"
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

SAVE_PATH = os.path.join(
    BASE_PATH,
    "outputs/final_hybrid_ef"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading contour CSV files...\n")

ed_df = pd.read_csv(ED_CSV)
es_df = pd.read_csv(ES_CSV)

print("ED CSV Loaded")
print("ES CSV Loaded")

# ============================================================
# NORMALIZE VIDEO NAME
# ============================================================

def normalize_name(x):

    x = str(x)

    x = os.path.basename(x)

    x = os.path.splitext(x)[0]

    return x.strip()

# ============================================================
# FIND VIDEO COLUMN
# ============================================================

def find_video_column(df):

    possible = [
        "FileName",
        "Filename",
        "Video",
        "video"
    ]

    for col in possible:

        if col in df.columns:
            return col

    return None

# ============================================================
# COMPUTE AREA FROM MASK
# ============================================================

def compute_mask_area(mask_path):

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

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest
    )

    return float(area)

# ============================================================
# BUILD CONTOUR
# ============================================================

def build_contour(group):

    left_pts = []

    right_pts = []

    # --------------------------------------------------------
    # LEFT WALL
    # --------------------------------------------------------

    for _, row in group.iterrows():

        left_pts.append([
            row["X1"],
            row["Y1"]
        ])

    # --------------------------------------------------------
    # RIGHT WALL
    # --------------------------------------------------------

    for _, row in group.iterrows():

        right_pts.append([
            row["X2"],
            row["Y2"]
        ])

    # --------------------------------------------------------
    # REVERSE RIGHT WALL
    # --------------------------------------------------------

    right_pts = right_pts[::-1]

    # --------------------------------------------------------
    # COMPLETE CONTOUR
    # --------------------------------------------------------

    contour = np.array(
        left_pts + right_pts,
        dtype=np.float32
    )

    return contour

# ============================================================
# COMPUTE LV LENGTH FROM CONTOUR
# ============================================================

def compute_lv_length(contour):

    # --------------------------------------------------------
    # APEX
    # --------------------------------------------------------

    apex_idx = np.argmin(
        contour[:,1]
    )

    apex = contour[apex_idx]

    # --------------------------------------------------------
    # BASE REGION
    # --------------------------------------------------------

    y_max = np.max(
        contour[:,1]
    )

    base_region = contour[
        contour[:,1] > (y_max - 5)
    ]

    if len(base_region) < 2:
        return None

    # --------------------------------------------------------
    # LEFT + RIGHT BASE
    # --------------------------------------------------------

    left_base = base_region[
        np.argmin(base_region[:,0])
    ]

    right_base = base_region[
        np.argmax(base_region[:,0])
    ]

    # --------------------------------------------------------
    # BASAL MIDPOINT
    # --------------------------------------------------------

    basal_mid = np.array([

        (left_base[0] + right_base[0]) / 2,

        (left_base[1] + right_base[1]) / 2

    ])

    # --------------------------------------------------------
    # LV LENGTH
    # --------------------------------------------------------

    length = np.linalg.norm(
        apex - basal_mid
    )

    return float(length)

# ============================================================
# VIDEO COLUMNS
# ============================================================

ed_video_col = find_video_column(ed_df)
es_video_col = find_video_column(es_df)

# ============================================================
# NORMALIZE VIDEO NAMES
# ============================================================

ed_df[ed_video_col] = ed_df[
    ed_video_col
].apply(normalize_name)

es_df[es_video_col] = es_df[
    es_video_col
].apply(normalize_name)

# ============================================================
# GROUP VIDEOS
# ============================================================

ed_groups = dict(
    tuple(
        ed_df.groupby(ed_video_col)
    )
)

es_groups = dict(
    tuple(
        es_df.groupby(es_video_col)
    )
)

# ============================================================
# COMMON VIDEOS
# ============================================================

common_videos = sorted(
    set(ed_groups.keys())
    &
    set(es_groups.keys())
)

print(f"\nCommon Videos : {len(common_videos)}")

# ============================================================
# RESULTS
# ============================================================

results = []

# ============================================================
# LOOP THROUGH VIDEOS
# ============================================================

for video in common_videos:

    # ========================================================
    # CONTOUR GROUPS
    # ========================================================

    ed_group = ed_groups[video]
    es_group = es_groups[video]

    # ========================================================
    # BUILD CONTOURS
    # ========================================================

    ed_contour = build_contour(
        ed_group
    )

    es_contour = build_contour(
        es_group
    )

    # ========================================================
    # COMPUTE LENGTHS
    # ========================================================

    ED_length = compute_lv_length(
        ed_contour
    )

    ES_length = compute_lv_length(
        es_contour
    )

    if ED_length is None:
        continue

    if ES_length is None:
        continue

    if ED_length <= 0:
        continue

    if ES_length <= 0:
        continue

    # ========================================================
    # MASK PATHS
    # ========================================================

    video_folder = os.path.join(
        PHASE_PATH,
        video
    )

    ed_folder = os.path.join(
        video_folder,
        "ED"
    )

    es_folder = os.path.join(
        video_folder,
        "ES"
    )

    if not os.path.exists(ed_folder):
        continue

    if not os.path.exists(es_folder):
        continue

    # ========================================================
    # GET MASK FILES
    # ========================================================

    ed_masks = sorted([
        f for f in os.listdir(ed_folder)
        if "mask" in f.lower()
    ])

    es_masks = sorted([
        f for f in os.listdir(es_folder)
        if "mask" in f.lower()
    ])

    if len(ed_masks) == 0:
        continue

    if len(es_masks) == 0:
        continue

    ed_mask_path = os.path.join(
        ed_folder,
        ed_masks[0]
    )

    es_mask_path = os.path.join(
        es_folder,
        es_masks[0]
    )

    # ========================================================
    # AREA FROM MASKS
    # ========================================================

    ED_area = compute_mask_area(
        ed_mask_path
    )

    ES_area = compute_mask_area(
        es_mask_path
    )

    if ED_area is None:
        continue

    if ES_area is None:
        continue

    # ========================================================
    # AREA + LENGTH VOLUME
    # ========================================================

    EDV = 0.85 * (
        (ED_area ** 2)
        /
        ED_length
    )

    ESV = 0.85 * (
        (ES_area ** 2)
        /
        ES_length
    )

    # ========================================================
    # EF
    # ========================================================

    if EDV <= 0:
        continue

    EF = (
        (EDV - ESV)
        /
        EDV
    ) * 100

    # ========================================================
    # CLAMP
    # ========================================================

    EF = max(
        0,
        min(100, EF)
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Video": video,

        # ----------------------------------------------------
        # ED
        # ----------------------------------------------------

        "ED_Area": ED_area,
        "ED_Length": ED_length,
        "Estimated_EDV": EDV,

        # ----------------------------------------------------
        # ES
        # ----------------------------------------------------

        "ES_Area": ES_area,
        "ES_Length": ES_length,
        "Estimated_ESV": ESV,

        # ----------------------------------------------------
        # EF
        # ----------------------------------------------------

        "Estimated_EF (%)": EF
    })

    print(
        f"{video} --> EF = {EF:.2f}%"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

save_csv = os.path.join(
    SAVE_PATH,
    "Hybrid_Mask_Contour_EF.csv"
)

results_df.to_csv(
    save_csv,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n====================================")
print("HYBRID MASK + CONTOUR EF COMPLETED")
print("====================================")

print(f"\nTotal Videos : {len(results_df)}")

print(f"\nSaved to:\n{save_csv}")
