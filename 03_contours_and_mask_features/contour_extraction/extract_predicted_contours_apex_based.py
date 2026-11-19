# ============================================================
# IMPROVED TEST CONTOUR EXTRACTION
# APEX-BASED LEFT/RIGHT WALL SPLIT
# SEPARATE ED / ES CSV GENERATION
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours_apex_based"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE CSV FILES
# ============================================================

ED_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ED_Test_VolumeTracings_ApexBased.csv"
)

ES_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ES_Test_VolumeTracings_ApexBased.csv"
)

# ============================================================
# PARAMETERS
# ============================================================

RESAMPLE_POINTS = 100

# ============================================================
# STORAGE
# ============================================================

ed_results = []
es_results = []

# ============================================================
# HELPERS
# ============================================================

def parse_frame_num(mask_file):
    try:
        return int(mask_file.split("_")[-1].replace(".png", ""))
    except:
        return None

def cyclic_path(contour, start_idx, end_idx, step):
    """
    Traverse contour cyclically from start_idx to end_idx (inclusive).
    """
    n = len(contour)
    path = []
    idx = start_idx

    while True:
        path.append(contour[idx])
        if idx == end_idx:
            break
        idx = (idx + step) % n
        if len(path) > n + 5:
            break

    return np.array(path)

def resample_path(points, num_points=100):
    """
    Resample a polyline to a fixed number of points using arc length.
    """
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return None

    deltas = np.diff(points, axis=0)
    dist = np.sqrt((deltas ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(dist)])

    if s[-1] == 0:
        return np.repeat(points[:1], num_points, axis=0)

    new_s = np.linspace(0, s[-1], num_points)
    x_new = np.interp(new_s, s, points[:, 0])
    y_new = np.interp(new_s, s, points[:, 1])

    return np.stack([x_new, y_new], axis=1)

def select_apex_index(contour):
    """
    Apex = point with minimum Y.
    If multiple, choose the one closest to the centroid X.
    """
    y_min = np.min(contour[:, 1])
    candidates = np.where(contour[:, 1] == y_min)[0]

    if len(candidates) == 1:
        return int(candidates[0])

    centroid_x = np.mean(contour[:, 0])
    chosen = candidates[np.argmin(np.abs(contour[candidates, 0] - centroid_x))]
    return int(chosen)

def select_base_index(contour, centroid_x, side="left"):
    """
    Pick a base point near the bottom of the contour on the requested side.
    """
    if side == "left":
        side_mask = contour[:, 0] < centroid_x
    else:
        side_mask = contour[:, 0] >= centroid_x

    candidates = np.where(side_mask)[0]

    if len(candidates) == 0:
        # fallback to globally lowest point
        y_max = np.max(contour[:, 1])
        fallback = np.where(contour[:, 1] == y_max)[0]
        return int(fallback[0])

    y_vals = contour[candidates, 1]
    idx = candidates[np.argmax(y_vals)]
    return int(idx)

def choose_wall_path(contour, apex_idx, base_idx, centroid_x, prefer="left"):
    """
    There are two cyclic paths from apex to base.
    Choose the one whose mean x lies on the expected side.
    """
    forward = cyclic_path(contour, apex_idx, base_idx, step=1)
    backward = cyclic_path(contour, apex_idx, base_idx, step=-1)

    f_mean_x = np.mean(forward[:, 0])
    b_mean_x = np.mean(backward[:, 0])

    if prefer == "left":
        # prefer the path with smaller mean x
        if f_mean_x <= b_mean_x:
            return forward if f_mean_x <= centroid_x else backward
        else:
            return backward if b_mean_x <= centroid_x else forward

    else:
        # prefer the path with larger mean x
        if f_mean_x >= b_mean_x:
            return forward if f_mean_x >= centroid_x else backward
        else:
            return backward if b_mean_x >= centroid_x else forward

def extract_wall_traces_from_mask(mask):
    """
    Extract apex-based left/right wall traces from a binary LV mask.
    Returns left_trace, right_trace or (None, None) if failed.
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()

    if len(contour.shape) != 2 or len(contour) < 10:
        return None, None

    centroid_x = np.mean(contour[:, 0])

    apex_idx = select_apex_index(contour)
    left_base_idx = select_base_index(contour, centroid_x, side="left")
    right_base_idx = select_base_index(contour, centroid_x, side="right")

    left_path = choose_wall_path(contour, apex_idx, left_base_idx, centroid_x, prefer="left")
    right_path = choose_wall_path(contour, apex_idx, right_base_idx, centroid_x, prefer="right")

    if left_path is None or right_path is None:
        return None, None

    left_path = resample_path(left_path, RESAMPLE_POINTS)
    right_path = resample_path(right_path, RESAMPLE_POINTS)

    if left_path is None or right_path is None:
        return None, None

    return left_path, right_path

def process_phase_folder(video_name, phase_name, phase_path, target_list):
    """
    Process ED or ES folder of one video and append rows to target_list.
    """
    if not os.path.exists(phase_path):
        return

    mask_files = [
        f for f in os.listdir(phase_path)
        if f.endswith(".png") and "mask" in f.lower()
    ]

    for mask_file in mask_files:
        frame_num = parse_frame_num(mask_file)
        if frame_num is None:
            continue

        mask_path = os.path.join(phase_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        left_trace, right_trace = extract_wall_traces_from_mask(binary)

        if left_trace is None or right_trace is None:
            continue

        # Ensure equal lengths
        n_points = min(len(left_trace), len(right_trace))
        if n_points < 5:
            continue

        left_trace = left_trace[:n_points]
        right_trace = right_trace[:n_points]

        for i in range(n_points):
            x1, y1 = left_trace[i]
            x2, y2 = right_trace[i]

            target_list.append({
                "FileName": video_name + ".avi",
                "X1": float(x1),
                "Y1": float(y1),
                "X2": float(x2),
                "Y2": float(y2),
                "Frame": int(frame_num)
            })

# ============================================================
# GET VIDEO FOLDERS
# ============================================================

video_folders = os.listdir(ROOT_DIR)
print(f"\nTotal Videos Found : {len(video_folders)}")

# ============================================================
# MAIN LOOP
# ============================================================

for video_name in video_folders:

    video_path = os.path.join(ROOT_DIR, video_name)

    if not os.path.isdir(video_path):
        continue

    print(f"\nProcessing : {video_name}")

    ed_path = os.path.join(video_path, "ED")
    es_path = os.path.join(video_path, "ES")

    process_phase_folder(video_name, "ED", ed_path, ed_results)
    process_phase_folder(video_name, "ES", es_path, es_results)

# ============================================================
# CREATE DATAFRAMES
# ============================================================

ed_df = pd.DataFrame(ed_results)
es_df = pd.DataFrame(es_results)

# ============================================================
# SAVE CSVs
# ============================================================

ed_df.to_csv(ED_SAVE_CSV, index=False)
es_df.to_csv(ES_SAVE_CSV, index=False)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("APEX-BASED ED / ES CONTOUR EXTRACTION COMPLETE")
print("======================================")

print(f"\nED Points Extracted : {len(ed_df)}")
print(f"ES Points Extracted : {len(es_df)}")

print(f"\nED CSV Saved:\n{ED_SAVE_CSV}")
print(f"\nES CSV Saved:\n{ES_SAVE_CSV}")
