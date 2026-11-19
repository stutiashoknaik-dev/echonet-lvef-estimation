# ============================================================
# BUILD ANCHOR LOOKUP TABLES
# ------------------------------------------------------------
# Run this ONCE before training. It produces three small CSVs:
#   train_anchor_lookup.csv, val_anchor_lookup.csv, test_anchor_lookup.csv
# each with columns: FileName, ED_Frame, ES_Frame, TotalFrames
#
# TRAIN / VAL source: VolumeTracings.csv (ground truth). Same area-
# ranking logic your own adaptive_cardiac_frame_selection.py already
# uses -- largest-area frame = ED, smallest-area frame = ES.
#
# TEST source: your existing Predicted_ED_Test_Contours_100Pts.csv /
# Predicted_ES_Test_Contours_100Pts.csv, which already carry the
# adaptively-selected Frame number per video. No recomputation needed.
#
# TotalFrames is read once per video via cv2 (fast -- just header
# metadata, not full decode).
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# COLAB PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

FILELIST_CSV = os.path.join(BASE_PATH, "FileList.csv")
VOLUME_CSV = os.path.join(BASE_PATH, "VolumeTracings.csv")

TEST_ED_CSV = os.path.join(BASE_PATH, "outputs/test_predicted_contours/Predicted_ED_Test_Contours_100Pts.csv")
TEST_ES_CSV = os.path.join(BASE_PATH, "outputs/test_predicted_contours/Predicted_ES_Test_Contours_100Pts.csv")

VIDEO_ROOT = os.path.join(BASE_PATH, "videos_split")

SAVE_DIR = os.path.join(BASE_PATH, "outputs/anchor_lookup")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def resolve_video_path(filename, split):
    name = filename if filename.lower().endswith(".avi") else filename + ".avi"
    return os.path.join(VIDEO_ROOT, split, name)


def build_train_val_lookup(split):
    print(f"\nBuilding {split} anchor lookup from ground truth...")

    filelist = pd.read_csv(FILELIST_CSV)
    filelist["FileName"] = filelist["FileName"].astype(str).str.replace(".avi", "", regex=False).str.strip()
    split_videos = set(filelist[filelist["Split"].str.upper() == split.upper()]["FileName"])

    vol = pd.read_csv(VOLUME_CSV)
    vol["Video"] = vol["FileName"].astype(str).str.replace(".avi", "", regex=False).str.strip()
    vol = vol[vol["Video"].isin(split_videos)]

    rows = []
    grouped = vol.groupby(["Video", "Frame"])
    per_video_areas = {}

    for (video, frame), group in grouped:
        pts = group[["X1", "Y1"]].values
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        per_video_areas.setdefault(video, []).append((int(frame), area))

    for video, values in per_video_areas.items():
        values_sorted = sorted(values, key=lambda v: v[1])
        es_frame = values_sorted[0][0]
        ed_frame = values_sorted[-1][0]

        video_path = resolve_video_path(video, split)
        total_frames = get_total_frames(video_path) if os.path.exists(video_path) else max(ed_frame, es_frame) + 1

        rows.append({
            "FileName": video,
            "ED_Frame": ed_frame,
            "ES_Frame": es_frame,
            "TotalFrames": total_frames,
        })

    df = pd.DataFrame(rows)
    save_path = os.path.join(SAVE_DIR, f"{split.lower()}_anchor_lookup.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} videos -> {save_path}")
    return df


def build_test_lookup():
    print("\nBuilding test anchor lookup from adaptive-selection predictions...")

    ed_df = pd.read_csv(TEST_ED_CSV)
    es_df = pd.read_csv(TEST_ES_CSV)

    ed_frames = ed_df.groupby("FileName")["Frame"].first()
    es_frames = es_df.groupby("FileName")["Frame"].first()

    rows = []
    common_videos = set(ed_frames.index) & set(es_frames.index)

    for video in sorted(common_videos):
        video_name = video if video.lower().endswith(".avi") else video
        clean_name = video_name.replace(".avi", "")

        video_path = resolve_video_path(clean_name, "test")
        total_frames = get_total_frames(video_path) if os.path.exists(video_path) else max(
            int(ed_frames[video]), int(es_frames[video])
        ) + 1

        rows.append({
            "FileName": clean_name,
            "ED_Frame": int(ed_frames[video]),
            "ES_Frame": int(es_frames[video]),
            "TotalFrames": total_frames,
        })

    df = pd.DataFrame(rows)
    save_path = os.path.join(SAVE_DIR, "test_anchor_lookup.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} videos -> {save_path}")
    return df


if __name__ == "__main__":
    train_df = build_train_val_lookup("train")
    val_df = build_train_val_lookup("val")
    test_df = build_test_lookup()

    print("\n====================================")
    print("ANCHOR LOOKUP BUILD COMPLETE")
    print("====================================")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
