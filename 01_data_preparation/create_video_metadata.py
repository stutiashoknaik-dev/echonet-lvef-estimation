import os
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

FILELIST_CSV = os.path.join(BASE_PATH, "data", "FileList.csv")

TRAIN_DIR = os.path.join(BASE_PATH, "data", "videos_split", "train")
VAL_DIR   = os.path.join(BASE_PATH, "data", "videos_split", "val")
TEST_DIR  = os.path.join(BASE_PATH, "data", "videos_split", "test")

OUTPUT_DIR = os.path.join(BASE_PATH, "outputs", "video_metadata")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# HELPER
# ==========================================================

def normalize_name(name):
    return os.path.splitext(str(name))[0].lower()

# ==========================================================
# LOAD FILELIST
# ==========================================================

df = pd.read_csv(FILELIST_CSV)

print("\nColumns Found:")
print(df.columns.tolist())

required_cols = ["FileName", "EF", "Split"]

for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Missing column: {col}")

df["VideoKey"] = df["FileName"].apply(normalize_name)

# ==========================================================
# CREATE METADATA
# ==========================================================

def build_metadata(split_name, video_dir):

    split_df = df[df["Split"].str.upper() == split_name.upper()].copy()

    rows = []

    missing = []

    video_files = {
        normalize_name(f): os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(".avi")
    }

    for _, row in split_df.iterrows():

        video_key = row["VideoKey"]

        if video_key not in video_files:
            missing.append(row["FileName"])
            continue

        rows.append({
            "FileName": row["FileName"],
            "VideoPath": video_files[video_key],
            "EF": row["EF"]
        })

    metadata_df = pd.DataFrame(rows)

    save_csv = os.path.join(
        OUTPUT_DIR,
        f"{split_name.lower()}_metadata.csv"
    )

    metadata_df.to_csv(save_csv, index=False)

    print("\n====================================")
    print(split_name)
    print("====================================")

    print("Expected Videos :", len(split_df))
    print("Found Videos    :", len(metadata_df))
    print("Missing Videos  :", len(missing))

    print("Saved Metadata:")
    print(save_csv)

    if len(missing) > 0:

        missing_csv = os.path.join(
            OUTPUT_DIR,
            f"{split_name.lower()}_missing.csv"
        )

        pd.DataFrame(
            {"MissingVideo": missing}
        ).to_csv(missing_csv, index=False)

        print("Missing Report:")
        print(missing_csv)

# ==========================================================
# RUN
# ==========================================================

build_metadata("TRAIN", TRAIN_DIR)
build_metadata("VAL", VAL_DIR)
build_metadata("TEST", TEST_DIR)

print("\n====================================")
print("METADATA CREATION COMPLETED")
print("====================================")
print("Output Folder:")
print(OUTPUT_DIR)
