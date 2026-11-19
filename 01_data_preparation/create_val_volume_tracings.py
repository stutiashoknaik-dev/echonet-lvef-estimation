# ============================================================
# CREATE VALIDATION-ONLY GROUND TRUTH CSV
# SAME FORMAT AS VolumeTracings.csv
# ============================================================

import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ORIGINAL GT CSV
CSV_PATH = os.path.join(
    BASE_PATH,
    "data/VolumeTracings.csv"
)

# VALIDATION FRAMES
VAL_FRAME_PATH = os.path.join(
    BASE_PATH,
    "data/frames/val"
)

# SAVE PATH
SAVE_PATH = os.path.join(
    BASE_PATH,
    "outputs/val_contour_gt"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# LOAD GT CSV
# ============================================================

print("Loading VolumeTracings.csv...")

df = pd.read_csv(CSV_PATH)

print("CSV Loaded!")

# ============================================================
# GET VALIDATION VIDEOS
# ============================================================

val_videos = os.listdir(VAL_FRAME_PATH)

# ADD .avi EXTENSION
val_videos = [
    v + ".avi"
    for v in val_videos
]

print(f"Validation Videos Found : {len(val_videos)}")

# ============================================================
# FILTER ONLY VALIDATION DATA
# ============================================================

val_df = df[
    df["FileName"].isin(val_videos)
]

# ============================================================
# SORT
# ============================================================

val_df = val_df.sort_values(
    by=["FileName", "Frame"]
)

# ============================================================
# SAVE CSV
# ============================================================

save_csv = os.path.join(
    SAVE_PATH,
    "val_volume_tracings.csv"
)

val_df.to_csv(
    save_csv,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("VALIDATION GT CSV CREATED")
print("======================================")

print(f"Total Rows : {len(val_df)}")

print(f"\nSaved to:\n{save_csv}")

print("\nDONE!")
