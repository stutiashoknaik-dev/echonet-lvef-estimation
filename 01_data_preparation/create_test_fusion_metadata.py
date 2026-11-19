# ============================================================
# CREATE TEST FUSION METADATA
# ============================================================

import os
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

TEST_MASK_FEATURES = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "test_mask_features.csv"
)

TEST_VIDEO_DIR = os.path.join(
    BASE_PATH,
    "data",
    "videos_split",
    "test"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "fusion_metadata"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

SAVE_CSV = os.path.join(
    SAVE_DIR,
    "fusion_test_metadata.csv"
)

# ============================================================
# LOAD FILELIST
# ============================================================

print("\nLoading FileList...")

filelist = pd.read_csv(FILELIST_CSV)

filelist["FileName"] = (

    filelist["FileName"]

    .astype(str)

    .str.replace(".avi", "", regex=False)

    .str.strip()

)

# ============================================================
# LOAD TEST MASK FEATURES
# ============================================================

print("Loading Test Mask Features...")

mask_df = pd.read_csv(TEST_MASK_FEATURES)

mask_df["FileName"] = (

    mask_df["FileName"]

    .astype(str)

    .str.replace(".avi", "", regex=False)

    .str.strip()

)

print(
    "Mask Feature Samples :",
    len(mask_df)
)

# ============================================================
# EF LOOKUP
# ============================================================

ef_lookup = dict(

    zip(

        filelist["FileName"],

        filelist["EF"]

    )

)

# ============================================================
# CREATE METADATA
# ============================================================

results = []

missing = 0

for idx, video in enumerate(mask_df["FileName"]):

    if idx % 250 == 0:

        print(
            f"{idx}/{len(mask_df)}"
        )

    video_path = os.path.join(

        TEST_VIDEO_DIR,

        video + ".avi"

    )

    if not os.path.exists(video_path):

        missing += 1

        continue

    if video not in ef_lookup:

        missing += 1

        continue

    results.append({

        "FileName": video,

        "VideoPath": video_path,

        "EF": float(

            ef_lookup[video]

        )

    })

# ============================================================
# SAVE
# ============================================================

metadata = pd.DataFrame(results)

metadata = metadata.sort_values(

    "FileName"

).reset_index(

    drop=True

)

metadata.to_csv(

    SAVE_CSV,

    index=False

)

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("TEST FUSION METADATA CREATED")
print("===================================")

print()

print(
    "Total Metadata :",
    len(metadata)
)

print(
    "Missing Videos :",
    missing
)

print()

print("Saved To:")

print(
    SAVE_CSV
)

print()

print("Columns:")

print(
    metadata.columns.tolist()
)

print()

print(metadata.head())
