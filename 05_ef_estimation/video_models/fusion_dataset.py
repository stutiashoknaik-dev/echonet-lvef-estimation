import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

# ============================================================
# CONFIG
# ============================================================

NUM_FRAMES = 32
IMG_SIZE = 112

def _clean_name(name):
    return str(name).replace(".avi", "").strip()

# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        mask_feature_csv,
        ed_contour_csv,
        es_contour_csv,
        framewise_csv=None,
        use_adaptive=True
    ):

        print("\nLoading Dataset...")

        self.use_adaptive = use_adaptive

        self.meta = pd.read_csv(metadata_csv)
        self.mask_df = pd.read_csv(mask_feature_csv)
        self.ed_df = pd.read_csv(ed_contour_csv)
        self.es_df = pd.read_csv(es_contour_csv)

        if framewise_csv and os.path.exists(framewise_csv):
            self.framewise_df = pd.read_csv(framewise_csv)
            self.framewise_df["CleanName"] = self.framewise_df["FileName"].apply(_clean_name)
            self.framewise_lookup = {
                k: g for k, g in self.framewise_df.groupby("CleanName")
            }
        else:
            self.framewise_df = None
            self.framewise_lookup = {}

        # ----------------------------------------------------
        # CLEAN FILENAMES FOR ROBUST LOOKUPS
        # ----------------------------------------------------

        self.meta["CleanName"] = self.meta["FileName"].apply(_clean_name)
        self.mask_df["CleanName"] = self.mask_df["FileName"].apply(_clean_name)
        self.ed_df["CleanName"] = self.ed_df["FileName"].apply(_clean_name)
        self.es_df["CleanName"] = self.es_df["FileName"].apply(_clean_name)

        # ----------------------------------------------------
        # MASK FEATURES LOOKUP
        # ----------------------------------------------------

        self.mask_lookup = {
            row["CleanName"]: row
            for _, row in self.mask_df.iterrows()
        }

        # ----------------------------------------------------
        # CONTOURS LOOKUP
        # ----------------------------------------------------

        self.ed_lookup = {
            k: g
            for k, g in self.ed_df.groupby("CleanName")
        }

        self.es_lookup = {
            k: g
            for k, g in self.es_df.groupby("CleanName")
        }

        print(f"Samples: {len(self.meta)}")

    # ========================================================
    # ADAPTIVE CARDIAC FRAME SELECTION
    # ========================================================

    def adaptive_frame_selection(self, video_name, total_frames):
        clean_vname = _clean_name(video_name)

        # Fallback if total_frames <= 0
        if total_frames <= 0:
            return np.zeros(NUM_FRAMES, dtype=np.int64)

        if not self.use_adaptive or clean_vname not in self.framewise_lookup:
            return np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=np.int64)

        try:
            fw_group = self.framewise_lookup[clean_vname]
            if "Area" in fw_group.columns and "Frame" in fw_group.columns:
                fw_sorted = fw_group.sort_values("Area")
                es_frame = int(fw_sorted.iloc[0]["Frame"])
                ed_frame = int(fw_sorted.iloc[-1]["Frame"])

                start_f = max(0, min(ed_frame, es_frame) - 2)
                end_f = min(total_frames - 1, max(ed_frame, es_frame) + 2)

                if end_f > start_f:
                    indices = np.linspace(start_f, end_f, NUM_FRAMES).astype(np.int64)
                    return np.sort(indices)
        except Exception:
            pass

        # Default fallback to uniform sampling
        return np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=np.int64)

    # ========================================================
    # VIDEO SAMPLING
    # ========================================================

    def sample_video(self, video_path, video_name):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0 or not cap.isOpened():
            if cap.isOpened():
                cap.release()
            # Return dummy zero tensor if video cannot be opened
            return torch.zeros((NUM_FRAMES, 1, IMG_SIZE, IMG_SIZE), dtype=torch.float32)

        frame_ids = self.adaptive_frame_selection(video_name, total_frames)

        frames = []
        current = 0
        target = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if target >= len(frame_ids):
                break

            while target < len(frame_ids) and current == frame_ids[target]:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                gray = gray.astype(np.float32) / 255.0

                frames.append(gray)
                target += 1

            current += 1

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)]

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())

        frames = frames[:NUM_FRAMES]
        arr = np.stack(frames, axis=0)         # (32, 112, 112)
        arr = np.nan_to_num(arr, nan=0.0)
        tensor = torch.from_numpy(arr).unsqueeze(1) # (32, 1, 112, 112)

        return tensor

    # ========================================================
    # CONTOURS
    # ========================================================

    def get_contours(self, video_name):

        clean_vname = _clean_name(video_name)

        if clean_vname not in self.ed_lookup or clean_vname not in self.es_lookup:
            return torch.zeros(800, dtype=torch.float32)

        ed = self.ed_lookup[clean_vname].sort_values("PointID")
        es = self.es_lookup[clean_vname].sort_values("PointID")

        ed_feat = np.concatenate([
            ed["X1"].values,
            ed["Y1"].values,
            ed["X2"].values,
            ed["Y2"].values
        ])

        es_feat = np.concatenate([
            es["X1"].values,
            es["Y1"].values,
            es["X2"].values,
            es["Y2"].values
        ])

        contour = np.concatenate([ed_feat, es_feat]).astype(np.float32)
        contour = contour / 112.0
        contour = np.nan_to_num(contour, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(contour, dtype=torch.float32)

    # ========================================================
    # MASK FEATURES
    # ========================================================

    def get_mask_features(self, video_name):

        clean_vname = _clean_name(video_name)

        if clean_vname not in self.mask_lookup:
            return torch.zeros(24, dtype=torch.float32)

        row = self.mask_lookup[clean_vname]

        feature_cols = [
            c for c in row.index
            if c not in ["FileName", "CleanName", "EF"]
        ]

        feats = row[feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(feats, dtype=torch.float32)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.meta)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        video_name = row["FileName"]
        video_path = row["VideoPath"]

        ef = np.float32(row["EF"]) if "EF" in row else np.float32(0.0)

        video = self.sample_video(video_path, video_name)
        contour = self.get_contours(video_name)
        mask_feat = self.get_mask_features(video_name)

        ef_tensor = torch.tensor(ef, dtype=torch.float32)

        return video, contour, mask_feat, ef_tensor
