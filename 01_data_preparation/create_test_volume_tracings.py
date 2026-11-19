# ============================================================
# CREATE TEST GT WITH ED / ES PHASE LABELS
# FileList -> test split videos
# VolumeTracings -> test contour rows
# Phase inference:
#   ED = largest LV area
#   ES = smallest LV area
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

VOLTRACINGS_CSV = os.path.join(
    BASE_PATH,
    "data/VolumeTracings.csv"
)

FILELIST_CSV = os.path.join(
    BASE_PATH,
    "data/FileList.csv"
)

# Optional fallback if FileList split column is not available
TEST_FRAME_PATH = os.path.join(
    BASE_PATH,
    "data/frames/test"
)

SAVE_PATH = os.path.join(
    BASE_PATH,
    "outputs/test_contour_gt_phasewise"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def normalize_video_name(name):
    name = str(name).strip()
    if not name.lower().endswith(".avi"):
        name += ".avi"
    return name

def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def contour_area(group):
    """
    Rebuild a closed contour from VolumeTracings rows and compute area.
    GT rows are (X1,Y1) left wall and (X2,Y2) right wall.
    """
    x1 = group["X1"].astype(float).values
    y1 = group["Y1"].astype(float).values
    x2 = group["X2"].astype(float).values
    y2 = group["Y2"].astype(float).values

    if len(x1) < 3 or len(x2) < 3:
        return np.nan

    # closed polygon: left wall + reversed right wall
    x = np.concatenate([x1, x2[::-1]])
    y = np.concatenate([y1, y2[::-1]])

    if len(x) < 3:
        return np.nan

    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )
    return float(area)

# ============================================================
# LOAD FILELIST
# ============================================================

print("Loading FileList.csv...")
filelist_df = pd.read_csv(FILELIST_CSV)
print("FileList loaded.")

video_col = find_column(filelist_df.columns, ["FileName", "Filename", "Video", "VideoName", "Name"])
split_col = find_column(filelist_df.columns, ["Split", "Subset", "Set", "Group"])

if video_col is not None and split_col is not None:
    test_videos = filelist_df[
        filelist_df[split_col].astype(str).str.lower().str.contains("test", na=False)
    ][video_col].dropna().astype(str).map(normalize_video_name).unique().tolist()
    source_used = "FileList.csv"
else:
    # fallback: use test frames folder
    test_videos = [
        normalize_video_name(v)
        for v in os.listdir(TEST_FRAME_PATH)
        if os.path.isdir(os.path.join(TEST_FRAME_PATH, v))
    ]
    source_used = "data/frames/test fallback"

print(f"Test videos found from {source_used}: {len(test_videos)}")

# ============================================================
# LOAD VOLTRACINGS
# ============================================================

print("Loading VolumeTracings.csv...")
df = pd.read_csv(VOLTRACINGS_CSV)
print("VolumeTracings loaded.")

df["FileName"] = df["FileName"].astype(str).str.strip()

# ============================================================
# FILTER TEST ONLY
# ============================================================

test_df = df[df["FileName"].isin(test_videos)].copy()
test_df = test_df.sort_values(by=["FileName", "Frame"]).reset_index(drop=True)

print(f"Test contour rows: {len(test_df)}")
print(f"Unique test videos in GT: {test_df['FileName'].nunique()}")

# ============================================================
# LABEL ED / ES PER VIDEO
# ============================================================

phase_rows = []
ed_rows = []
es_rows = []
summary_rows = []

print("\nInferring ED / ES from GT contour area...\n")

for video, video_df in test_df.groupby("FileName"):
    areas = []

    for frame_id, grp in video_df.groupby("Frame"):
        area = contour_area(grp)
        if np.isnan(area):
            continue
        areas.append((frame_id, area))

    if len(areas) == 0:
        continue

    ed_frame = max(areas, key=lambda t: t[1])[0]
    es_frame = min(areas, key=lambda t: t[1])[0]

    tmp = video_df.copy()
    tmp["Phase"] = ""

    tmp.loc[tmp["Frame"] == ed_frame, "Phase"] = "ED"
    tmp.loc[tmp["Frame"] == es_frame, "Phase"] = "ES"

    phase_rows.append(tmp)
    ed_rows.append(tmp[tmp["Phase"] == "ED"])
    es_rows.append(tmp[tmp["Phase"] == "ES"])

    summary_rows.append({
        "FileName": video,
        "ED_Frame": int(ed_frame),
        "ES_Frame": int(es_frame),
        "ED_Area": float(max(areas, key=lambda t: t[1])[1]),
        "ES_Area": float(min(areas, key=lambda t: t[1])[1]),
        "Num_Traced_Frames": int(len(areas))
    })

# ============================================================
# CONCATENATE
# ============================================================

phase_df = pd.concat(phase_rows, ignore_index=True) if phase_rows else pd.DataFrame()
ed_df = pd.concat(ed_rows, ignore_index=True) if ed_rows else pd.DataFrame()
es_df = pd.concat(es_rows, ignore_index=True) if es_rows else pd.DataFrame()
summary_df = pd.DataFrame(summary_rows)

# ============================================================
# SAVE FILES
# ============================================================

PHASE_CSV = os.path.join(SAVE_PATH, "test_volume_tracings_with_phase.csv")
ED_CSV = os.path.join(SAVE_PATH, "test_volume_tracings_ED.csv")
ES_CSV = os.path.join(SAVE_PATH, "test_volume_tracings_ES.csv")
SUMMARY_CSV = os.path.join(SAVE_PATH, "test_phase_inference_summary.csv")

phase_df.to_csv(PHASE_CSV, index=False)
ed_df.to_csv(ED_CSV, index=False)
es_df.to_csv(ES_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

# ============================================================
# PRINT
# ============================================================

print("\n======================================")
print("TEST GT ED / ES LABELING COMPLETE")
print("======================================")
print(f"\nCombined phasewise CSV:\n{PHASE_CSV}")
print(f"\nED CSV:\n{ED_CSV}")
print(f"\nES CSV:\n{ES_CSV}")
print(f"\nSummary CSV:\n{SUMMARY_CSV}")
print(f"\nVideos processed: {summary_df['FileName'].nunique() if len(summary_df) else 0}")
print(f"ED/ES labeled rows: {len(phase_df)}")
