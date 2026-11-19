import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

TRAIN_VIDEO_DIR = os.path.join(
    BASE_PATH,
    "data/videos_split/train"
)

VAL_VIDEO_DIR = os.path.join(
    BASE_PATH,
    "data/videos_split/val"
)

TRAIN_MASK_CSV = os.path.join(
    BASE_PATH,
    "outputs/mask_features/train_mask_features.csv"
)

VAL_MASK_CSV = os.path.join(
    BASE_PATH,
    "outputs/mask_features/val_mask_features.csv"
)

TRAIN_ED_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs/train_val_contours/Train_ED_Contours_100Pts.csv"
)

TRAIN_ES_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs/train_val_contours/Train_ES_Contours_100Pts.csv"
)

VAL_ED_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs/train_val_contours/Val_ED_Contours_100Pts.csv"
)

VAL_ES_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs/train_val_contours/Val_ES_Contours_100Pts.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/fusion_metadata"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading Files...")

train_mask = pd.read_csv(TRAIN_MASK_CSV)
val_mask = pd.read_csv(VAL_MASK_CSV)

train_ed = pd.read_csv(TRAIN_ED_CONTOUR)
train_es = pd.read_csv(TRAIN_ES_CONTOUR)

val_ed = pd.read_csv(VAL_ED_CONTOUR)
val_es = pd.read_csv(VAL_ES_CONTOUR)

# ============================================================
# UNIQUE VIDEOS
# ============================================================

train_mask_set = set(train_mask["FileName"])

val_mask_set = set(val_mask["FileName"])

train_ed_set = set(
    train_ed["FileName"].str.replace(
        ".avi",
        "",
        regex=False
    )
)

train_es_set = set(
    train_es["FileName"].str.replace(
        ".avi",
        "",
        regex=False
    )
)

val_ed_set = set(
    val_ed["FileName"].str.replace(
        ".avi",
        "",
        regex=False
    )
)

val_es_set = set(
    val_es["FileName"].str.replace(
        ".avi",
        "",
        regex=False
    )
)

# ============================================================
# TRAIN VIDEOS
# ============================================================

train_video_set = set()

for file in os.listdir(TRAIN_VIDEO_DIR):

    if file.endswith(".avi"):

        train_video_set.add(
            file.replace(".avi", "")
        )

# ============================================================
# VAL VIDEOS
# ============================================================

val_video_set = set()

for file in os.listdir(VAL_VIDEO_DIR):

    if file.endswith(".avi"):

        val_video_set.add(
            file.replace(".avi", "")
        )

# ============================================================
# COMMON TRAIN
# ============================================================

train_common = (
    train_video_set
    &
    train_mask_set
    &
    train_ed_set
    &
    train_es_set
)

# ============================================================
# COMMON VAL
# ============================================================

val_common = (
    val_video_set
    &
    val_mask_set
    &
    val_ed_set
    &
    val_es_set
)

print("\n===================================")
print("COMMON SAMPLES")
print("===================================")

print(
    "Train :",
    len(train_common)
)

print(
    "Val   :",
    len(val_common)
)

# ============================================================
# TRAIN METADATA
# ============================================================

train_rows = []

for video in sorted(train_common):

    video_path = os.path.join(
        TRAIN_VIDEO_DIR,
        video + ".avi"
    )

    ef = train_mask.loc[
        train_mask["FileName"] == video,
        "EF"
    ].iloc[0]

    train_rows.append({

        "FileName": video,

        "VideoPath": video_path,

        "EF": ef
    })

# ============================================================
# VAL METADATA
# ============================================================

val_rows = []

for video in sorted(val_common):

    video_path = os.path.join(
        VAL_VIDEO_DIR,
        video + ".avi"
    )

    ef = val_mask.loc[
        val_mask["FileName"] == video,
        "EF"
    ].iloc[0]

    val_rows.append({

        "FileName": video,

        "VideoPath": video_path,

        "EF": ef
    })

# ============================================================
# SAVE
# ============================================================

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)

TRAIN_SAVE = os.path.join(
    SAVE_DIR,
    "fusion_train_metadata.csv"
)

VAL_SAVE = os.path.join(
    SAVE_DIR,
    "fusion_val_metadata.csv"
)

train_df.to_csv(
    TRAIN_SAVE,
    index=False
)

val_df.to_csv(
    VAL_SAVE,
    index=False
)

print("\n===================================")
print("FUSION METADATA CREATED")
print("===================================")

print("\nTrain Samples :", len(train_df))
print("Val Samples   :", len(val_df))

print("\nSaved:")

print(TRAIN_SAVE)
print(VAL_SAVE)
