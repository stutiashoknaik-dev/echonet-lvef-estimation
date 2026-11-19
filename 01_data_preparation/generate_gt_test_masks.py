# ============================================================
# FINAL CORRECT GT TEST MASK GENERATION
# ============================================================
#
# GOAL:
# -----
# Generate TRUE GT masks ONLY for:
#   - TEST videos
#   - GT ED frame
#   - GT ES frame
#
# IMPORTANT:
# ----------
# Uses correct EchoNet contour reconstruction:
#
#   (X1,Y1) + reversed(X2,Y2)
#
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
# PATHS
# ============================================================

frames_path = os.path.join(
    BASE_PATH,
    "data/frames"
)

csv_path = os.path.join(
    BASE_PATH,
    "data/VolumeTracings.csv"
)

save_path = os.path.join(
    BASE_PATH,
    "outputs/final_GT_test_masks"
)

os.makedirs(save_path, exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading VolumeTracings.csv...\n")

vt = pd.read_csv(csv_path)

print("CSV Loaded Successfully")

# ============================================================
# CLEAN VIDEO NAMES
# ============================================================

vt["FileName"] = vt["FileName"].str.strip()

# ============================================================
# REMOVE .avi
# IMPORTANT FIX
# ============================================================

vt["Video"] = vt["FileName"].str.replace(
    ".avi",
    "",
    regex=False
)

# ============================================================
# COMPUTE GT ED / ES USING AREA
# ============================================================

print("\nComputing GT ED/ES frames...\n")

frame_dict = {}

grouped = vt.groupby(
    ["Video", "Frame"]
)

for (video_name, frame_id), group in grouped:

    x1 = group["X1"].values
    y1 = group["Y1"].values

    x2 = group["X2"].values
    y2 = group["Y2"].values

    # ========================================================
    # CLOSED POLYGON
    # ========================================================

    x = np.concatenate([
        x1,
        x2[::-1]
    ])

    y = np.concatenate([
        y1,
        y2[::-1]
    ])

    # ========================================================
    # AREA
    # ========================================================

    area = 0.5 * abs(

        np.dot(x, np.roll(y, 1))

        -

        np.dot(y, np.roll(x, 1))
    )

    if video_name not in frame_dict:
        frame_dict[video_name] = []

    frame_dict[video_name].append(
        (frame_id, area)
    )

# ============================================================
# FINAL GT FRAMES
# ============================================================

gt_frames = {}

for video_name, values in frame_dict.items():

    values = sorted(
        values,
        key=lambda x: x[1]
    )

    es = values[0][0]
    ed = values[-1][0]

    gt_frames[video_name] = {

        "ED": int(ed),
        "ES": int(es)
    }

print("GT ED/ES computation complete")

# ============================================================
# SAVE GT FRAME CSV
# ============================================================

gt_csv_path = os.path.join(
    save_path,
    "gt_ed_es_frames.csv"
)

gt_csv_data = []

for video_name in gt_frames:

    gt_csv_data.append({

        "Video": video_name,

        "GT_ED_Frame": gt_frames[video_name]["ED"],

        "GT_ES_Frame": gt_frames[video_name]["ES"]
    })

gt_frame_df = pd.DataFrame(gt_csv_data)

gt_frame_df.to_csv(
    gt_csv_path,
    index=False
)

print("\nGT frame CSV saved")

# ============================================================
# GENERATE ONLY TEST GT MASKS
# ============================================================

print("\nGenerating TEST GT masks...\n")

count = 0

for video_name in gt_frames:

    # ========================================================
    # TEST FRAME DIRECTORY
    # ========================================================

    frame_dir = os.path.join(
        frames_path,
        "test",
        video_name
    )

    # ========================================================
    # SKIP NON-TEST VIDEOS
    # ========================================================

    if not os.path.exists(frame_dir):
        continue

    print(f"Processing : {video_name}")

    ed_frame = gt_frames[video_name]["ED"]

    es_frame = gt_frames[video_name]["ES"]

    target_frames = [ed_frame, es_frame]

    # ========================================================
    # VIDEO CSV DATA
    # ========================================================

    video_df = vt[
        vt["Video"] == video_name
    ]

    # ========================================================
    # SAVE DIR
    # ========================================================

    video_save_dir = os.path.join(
        save_path,
        video_name
    )

    os.makedirs(
        video_save_dir,
        exist_ok=True
    )

    # ========================================================
    # PROCESS ED / ES
    # ========================================================

    for frame_id in target_frames:

        # ====================================================
        # FRAME FILE
        # ====================================================

        frame_file = os.path.join(
            frame_dir,
            f"frame_{frame_id}.png"
        )

        if not os.path.exists(frame_file):
            continue

        # ====================================================
        # READ FRAME
        # ====================================================

        img = cv2.imread(
            frame_file,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        h, w = img.shape

        # ====================================================
        # EMPTY MASK
        # ====================================================

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # ====================================================
        # CURRENT FRAME DATA
        # ====================================================

        frame_df = video_df[
            video_df["Frame"] == frame_id
        ]

        if len(frame_df) == 0:
            continue

        # ====================================================
        # CONTOUR POINTS
        # ====================================================

        x1 = frame_df["X1"].values.astype(int)
        y1 = frame_df["Y1"].values.astype(int)

        x2 = frame_df["X2"].values.astype(int)
        y2 = frame_df["Y2"].values.astype(int)

        # ====================================================
        # CLOSED POLYGON
        # ====================================================

        x = np.concatenate([
            x1,
            x2[::-1]
        ])

        y = np.concatenate([
            y1,
            y2[::-1]
        ])

        points = np.vstack(
            (x, y)
        ).T.astype(np.int32)

        # ====================================================
        # FILL MASK
        # ====================================================

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        # ====================================================
        # PHASE
        # ====================================================

        phase = (
            "ED"
            if frame_id == ed_frame
            else "ES"
        )

        # ====================================================
        # SAVE MASK
        # ====================================================

        save_file = os.path.join(
            video_save_dir,
            f"{phase}_frame_{frame_id}.png"
        )

        cv2.imwrite(
            save_file,
            mask
        )

        count += 1

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("FINAL TEST GT MASK GENERATION DONE")
print("======================================")

print(f"\nTotal GT Masks Generated : {count}")

print(f"\nSaved Directory :\n{save_path}")

print(f"\nGT Frame CSV :\n{gt_csv_path}")
