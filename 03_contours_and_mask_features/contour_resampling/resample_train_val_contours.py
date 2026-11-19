# ============================================================
# TRAIN / VAL CONTOUR RESAMPLING
# ECHONET VOLUMETRACINGS
# 21 POINTS -> 100 POINTS
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FILELIST_CSV = os.path.join(
    BASE_PATH,
    "data",
    "FileList.csv"
)

VOLUME_CSV = os.path.join(
    BASE_PATH,
    "data",
    "VolumeTracings.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "train_val_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_ED_CSV = os.path.join(
    SAVE_DIR,
    "Train_ED_Contours_100Pts.csv"
)

TRAIN_ES_CSV = os.path.join(
    SAVE_DIR,
    "Train_ES_Contours_100Pts.csv"
)

VAL_ED_CSV = os.path.join(
    SAVE_DIR,
    "Val_ED_Contours_100Pts.csv"
)

VAL_ES_CSV = os.path.join(
    SAVE_DIR,
    "Val_ES_Contours_100Pts.csv"
)

# ============================================================
# SETTINGS
# ============================================================

RESAMPLE_POINTS = 100

# ============================================================
# RESAMPLING FUNCTION
# ============================================================

def resample_wall(x, y, num_points=100):

    pts = np.column_stack([x, y]).astype(np.float32)

    if len(pts) < 2:
        return None

    delta = np.diff(pts, axis=0)

    dist = np.sqrt(
        np.sum(delta ** 2, axis=1)
    )

    arc = np.concatenate([
        [0],
        np.cumsum(dist)
    ])

    if arc[-1] == 0:
        return np.repeat(
            pts[:1],
            num_points,
            axis=0
        )

    new_arc = np.linspace(
        0,
        arc[-1],
        num_points
    )

    x_new = np.interp(
        new_arc,
        arc,
        pts[:, 0]
    )

    y_new = np.interp(
        new_arc,
        arc,
        pts[:, 1]
    )

    return np.column_stack([
        x_new,
        y_new
    ])

# ============================================================
# POLYGON AREA
# ============================================================

def compute_lv_area(df_frame):

    x = np.concatenate([
        df_frame["X1"].values,
        df_frame["X2"].values[::-1]
    ])

    y = np.concatenate([
        df_frame["Y1"].values,
        df_frame["Y2"].values[::-1]
    ])

    area = 0.5 * abs(
        np.dot(x, np.roll(y, 1))
        -
        np.dot(y, np.roll(x, 1))
    )

    return area

# ============================================================
# LOAD CSVs
# ============================================================

print("\nLoading CSVs...")

filelist = pd.read_csv(FILELIST_CSV)
vt = pd.read_csv(VOLUME_CSV)

# ============================================================
# FIX FILENAME MISMATCH
# ============================================================

filelist["FileName"] = (
    filelist["FileName"]
    .astype(str)
    .str.strip()
    .str.replace(".avi", "", regex=False)
)

vt["FileName"] = (
    vt["FileName"]
    .astype(str)
    .str.strip()
    .str.replace(".avi", "", regex=False)
)

filelist["Split"] = (
    filelist["Split"]
    .astype(str)
    .str.upper()
    .str.strip()
)

print("FileList rows      :", len(filelist))
print("VolumeTracing rows :", len(vt))

# ============================================================
# TRAIN / VAL SPLITS
# ============================================================

train_files = set(
    filelist[
        filelist["Split"] == "TRAIN"
    ]["FileName"]
)

val_files = set(
    filelist[
        filelist["Split"] == "VAL"
    ]["FileName"]
)

print("\nTrain Videos :", len(train_files))
print("Val Videos   :", len(val_files))

# ============================================================
# STORAGE
# ============================================================

train_ed = []
train_es = []

val_ed = []
val_es = []

# ============================================================
# GROUP VIDEOS
# ============================================================

groups = vt.groupby("FileName")

total_videos = len(groups)

print("\nProcessing Videos...\n")

# ============================================================
# MAIN LOOP
# ============================================================

for idx, (video, df_video) in enumerate(groups):

    if idx % 250 == 0:
        print(f"{idx}/{total_videos}")

    frames = sorted(
        df_video["Frame"].unique()
    )

    if len(frames) != 2:
        continue

    # --------------------------------------------------------
    # FIND ED / ES USING AREA
    # --------------------------------------------------------

    frame_areas = {}

    for frame in frames:

        df_frame = df_video[
            df_video["Frame"] == frame
        ]

        area = compute_lv_area(df_frame)

        frame_areas[frame] = area

    ed_frame = max(
        frame_areas,
        key=frame_areas.get
    )

    es_frame = min(
        frame_areas,
        key=frame_areas.get
    )

    # --------------------------------------------------------
    # PROCESS ED + ES
    # --------------------------------------------------------

    for phase_name, phase_frame in [
        ("ED", ed_frame),
        ("ES", es_frame)
    ]:

        df_phase = df_video[
            df_video["Frame"] == phase_frame
        ]

        left = resample_wall(
            df_phase["X1"].values,
            df_phase["Y1"].values,
            RESAMPLE_POINTS
        )

        right = resample_wall(
            df_phase["X2"].values,
            df_phase["Y2"].values,
            RESAMPLE_POINTS
        )

        if left is None or right is None:
            continue

        rows = []

        for point_id in range(
            RESAMPLE_POINTS
        ):

            rows.append({

                "FileName": video,

                "Frame": int(
                    phase_frame
                ),

                "PointID": point_id,

                "X1": float(
                    left[point_id, 0]
                ),

                "Y1": float(
                    left[point_id, 1]
                ),

                "X2": float(
                    right[point_id, 0]
                ),

                "Y2": float(
                    right[point_id, 1]
                )
            })

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        if video in train_files:

            if phase_name == "ED":
                train_ed.extend(rows)

            else:
                train_es.extend(rows)

        # ----------------------------------------------------
        # VAL
        # ----------------------------------------------------

        elif video in val_files:

            if phase_name == "ED":
                val_ed.extend(rows)

            else:
                val_es.extend(rows)

# ============================================================
# DEBUG
# ============================================================

print("\n===================================")
print("DEBUG COUNTS")
print("===================================")

print("Train ED :", len(train_ed))
print("Train ES :", len(train_es))
print("Val ED   :", len(val_ed))
print("Val ES   :", len(val_es))

if len(train_ed) > 0:

    print("\nFirst Train ED Row:")
    print(train_ed[0])

# ============================================================
# SAVE CSVs
# ============================================================

pd.DataFrame(train_ed).to_csv(
    TRAIN_ED_CSV,
    index=False
)

pd.DataFrame(train_es).to_csv(
    TRAIN_ES_CSV,
    index=False
)

pd.DataFrame(val_ed).to_csv(
    VAL_ED_CSV,
    index=False
)

pd.DataFrame(val_es).to_csv(
    VAL_ES_CSV,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n===================================")
print("CONTOUR RESAMPLING COMPLETE")
print("===================================")

print("\nTrain ED :", len(train_ed))
print("Train ES :", len(train_es))

print("Val ED   :", len(val_ed))
print("Val ES   :", len(val_es))

print("\nSaved To:")
print(SAVE_DIR)
