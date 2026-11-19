# ============================================================
# ADVANCED ANATOMICAL CONTOUR EXTRACTION
# OPEN LV WALL TRACING (VOLUMETRACINGS-LIKE)
# APEX + BASAL ENDPOINT METHOD
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

ROOT_DIR = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_anatomical"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE CSVs
# ============================================================

ED_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ED_Test_VolumeTracings_Anatomical.csv"
)

ES_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ES_Test_VolumeTracings_Anatomical.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

RESAMPLE_POINTS = 100
MIN_CONTOUR_AREA = 100

# ============================================================
# STORAGE
# ============================================================

ed_results = []
es_results = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_frame_num(mask_file):

    try:

        return int(
            mask_file.split("_")[-1]
            .replace(".png", "")
        )

    except:

        return None

# ============================================================

def keep_largest_component(binary):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    cleaned = np.zeros_like(binary)

    cleaned[labels == largest_label] = 255

    return cleaned

# ============================================================

def resample_path(points, n_points=100):

    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return None

    deltas = np.diff(points, axis=0)

    dist = np.sqrt(
        (deltas ** 2).sum(axis=1)
    )

    cumulative = np.concatenate([
        [0],
        np.cumsum(dist)
    ])

    if cumulative[-1] == 0:
        return None

    target = np.linspace(
        0,
        cumulative[-1],
        n_points
    )

    x_new = np.interp(
        target,
        cumulative,
        points[:, 0]
    )

    y_new = np.interp(
        target,
        cumulative,
        points[:, 1]
    )

    return np.stack([x_new, y_new], axis=1)

# ============================================================

def find_apex(contour):

    y_min = np.min(contour[:, 1])

    candidates = contour[
        contour[:, 1] == y_min
    ]

    center_x = np.mean(contour[:, 0])

    distances = np.abs(
        candidates[:, 0] - center_x
    )

    apex = candidates[
        np.argmin(distances)
    ]

    return apex

# ============================================================

def find_basal_points(contour):

    y_max = np.max(contour[:, 1])

    threshold = y_max - 15

    basal_region = contour[
        contour[:, 1] >= threshold
    ]

    if len(basal_region) < 2:
        return None, None

    left_base = basal_region[
        np.argmin(basal_region[:, 0])
    ]

    right_base = basal_region[
        np.argmax(basal_region[:, 0])
    ]

    return left_base, right_base

# ============================================================

def contour_index(contour, point):

    distances = np.linalg.norm(
        contour - point,
        axis=1
    )

    return np.argmin(distances)

# ============================================================

def extract_path(contour, start_idx, end_idx, direction):

    n = len(contour)

    path = []

    idx = start_idx

    while True:

        path.append(contour[idx])

        if idx == end_idx:
            break

        idx = (idx + direction) % n

        if len(path) > n + 5:
            break

    return np.array(path)

# ============================================================

def choose_best_path(path1, path2, prefer="left"):

    mean_x1 = np.mean(path1[:, 0])
    mean_x2 = np.mean(path2[:, 0])

    if prefer == "left":

        if mean_x1 < mean_x2:
            return path1

        else:
            return path2

    else:

        if mean_x1 > mean_x2:
            return path1

        else:
            return path2

# ============================================================

def extract_anatomical_traces(binary):

    # --------------------------------------------------------
    # CLEAN COMPONENTS
    # --------------------------------------------------------

    binary = keep_largest_component(binary)

    # --------------------------------------------------------
    # LIGHT MORPHOLOGY
    # --------------------------------------------------------

    kernel = np.ones((3,3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # FIND CONTOUR
    # --------------------------------------------------------

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None, None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None, None

    contour = contour.squeeze()

    if len(contour.shape) != 2:
        return None, None

    # --------------------------------------------------------
    # FIND ANATOMICAL LANDMARKS
    # --------------------------------------------------------

    apex = find_apex(contour)

    left_base, right_base = find_basal_points(contour)

    if left_base is None or right_base is None:
        return None, None

    apex_idx = contour_index(contour, apex)

    left_idx = contour_index(contour, left_base)

    right_idx = contour_index(contour, right_base)

    # --------------------------------------------------------
    # TRACE LEFT WALL
    # --------------------------------------------------------

    left_forward = extract_path(
        contour,
        apex_idx,
        left_idx,
        1
    )

    left_backward = extract_path(
        contour,
        apex_idx,
        left_idx,
        -1
    )

    left_wall = choose_best_path(
        left_forward,
        left_backward,
        prefer="left"
    )

    # --------------------------------------------------------
    # TRACE RIGHT WALL
    # --------------------------------------------------------

    right_forward = extract_path(
        contour,
        apex_idx,
        right_idx,
        1
    )

    right_backward = extract_path(
        contour,
        apex_idx,
        right_idx,
        -1
    )

    right_wall = choose_best_path(
        right_forward,
        right_backward,
        prefer="right"
    )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    left_wall = resample_path(
        left_wall,
        RESAMPLE_POINTS
    )

    right_wall = resample_path(
        right_wall,
        RESAMPLE_POINTS
    )

    if left_wall is None or right_wall is None:
        return None, None

    return left_wall, right_wall

# ============================================================
# MAIN PROCESSING
# ============================================================

video_folders = os.listdir(ROOT_DIR)

print(f"\nTotal Videos Found : {len(video_folders)}")

for video_name in video_folders:

    print(f"\nProcessing : {video_name}")

    video_path = os.path.join(
        ROOT_DIR,
        video_name
    )

    if not os.path.isdir(video_path):
        continue

    phase_dirs = [

        ("ED", os.path.join(video_path, "ED")),

        ("ES", os.path.join(video_path, "ES"))

    ]

    for phase_name, phase_path in phase_dirs:

        if not os.path.exists(phase_path):
            continue

        mask_files = [

            f for f in os.listdir(phase_path)

            if (
                f.endswith(".png")
                and "mask" in f.lower()
            )

        ]

        for mask_file in mask_files:

            frame_num = parse_frame_num(mask_file)

            if frame_num is None:
                continue

            mask_path = os.path.join(
                phase_path,
                mask_file
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            # ------------------------------------------------
            # BINARY
            # ------------------------------------------------

            _, binary = cv2.threshold(

                mask,

                127,

                255,

                cv2.THRESH_BINARY
            )

            # ------------------------------------------------
            # EXTRACT ANATOMICAL WALLS
            # ------------------------------------------------

            left_wall, right_wall = extract_anatomical_traces(
                binary
            )

            if left_wall is None or right_wall is None:
                continue

            n_points = min(
                len(left_wall),
                len(right_wall)
            )

            for i in range(n_points):

                x1, y1 = left_wall[i]
                x2, y2 = right_wall[i]

                row = {

                    "FileName": video_name + ".avi",

                    "X1": float(x1),
                    "Y1": float(y1),

                    "X2": float(x2),
                    "Y2": float(y2),

                    "Frame": int(frame_num)
                }

                if phase_name == "ED":

                    ed_results.append(row)

                else:

                    es_results.append(row)

# ============================================================
# SAVE CSVs
# ============================================================

ed_df = pd.DataFrame(ed_results)
es_df = pd.DataFrame(es_results)

ed_df.to_csv(
    ED_SAVE_CSV,
    index=False
)

es_df.to_csv(
    ES_SAVE_CSV,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("ANATOMICAL OPEN-WALL EXTRACTION COMPLETE")
print("======================================")

print(f"\nED Points Extracted : {len(ed_df)}")
print(f"ES Points Extracted : {len(es_df)}")

print(f"\nED CSV Saved:\n{ED_SAVE_CSV}")
print(f"\nES CSV Saved:\n{ES_SAVE_CSV}")
