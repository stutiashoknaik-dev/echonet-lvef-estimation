import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from anchor_frame_sampler import build_anchor_sequence

MORPH_FEATURE_NAMES = [
    "Area", "Perimeter", "MajorAxis", "MinorAxis",
    "AspectRatio", "Extent", "Solidity", "Eccentricity",
    "EquivalentDiameter", "ConvexArea",
]


class AnchorFusionEFDataset(Dataset):
    """
    split: "train" | "val" | "test" -- determines which contour/mask
           source files are used (GT for train/val, predicted for test).

    CHANGES FROM v1: jitter completely removed. Contour points are
    used exactly as stored in the CSV for every split, always.
    """

    def __init__(self, split, base_path, num_frames=32, img_size=112,
                 morph_mean=None, morph_std=None):
        self.split = split
        self.base_path = base_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.morph_mean = morph_mean
        self.morph_std = morph_std

        self.video_root = os.path.join(base_path, "videos_split", split)

        anchor_path = os.path.join(base_path, "outputs/anchor_lookup", f"{split}_anchor_lookup.csv")
        self.anchors = pd.read_csv(anchor_path).set_index("FileName")

        meta_path = os.path.join(base_path, "outputs/video_metadata", f"{split}_metadata.csv")
        meta_df = pd.read_csv(meta_path)
        video_col = next(c for c in ["VideoPath", "FileName", "Video", "Path"] if c in meta_df.columns)
        ef_col = next(c for c in ["EF", "GT_EF", "EjectionFraction"] if c in meta_df.columns)
        meta_df["clean_name"] = meta_df[video_col].astype(str).apply(
            lambda s: os.path.splitext(os.path.basename(s))[0]
        )
        self.ef_lookup = meta_df.set_index("clean_name")[ef_col].to_dict()

        if split == "test":
            ed_contour_path = os.path.join(base_path, "outputs/test_predicted_contours/Predicted_ED_Test_Contours_100Pts.csv")
            es_contour_path = os.path.join(base_path, "outputs/test_predicted_contours/Predicted_ES_Test_Contours_100Pts.csv")
            mask_feat_path = os.path.join(base_path, "outputs/mask_features/test_mask_features.csv")
        else:
            split_cap = split.capitalize()
            ed_contour_path = os.path.join(base_path, f"outputs/train_val_contours/{split_cap}_ED_Contours_100Pts.csv")
            es_contour_path = os.path.join(base_path, f"outputs/train_val_contours/{split_cap}_ES_Contours_100Pts.csv")
            mask_feat_path = os.path.join(base_path, f"outputs/mask_features/{split}_mask_features.csv")

        self.ed_contours = self._load_contours(ed_contour_path)
        self.es_contours = self._load_contours(es_contour_path)
        self.mask_features = pd.read_csv(mask_feat_path)
        self.mask_features["clean_name"] = self.mask_features["FileName"].astype(str).apply(
            lambda s: os.path.splitext(os.path.basename(s))[0]
        )
        self.mask_features = self.mask_features.set_index("clean_name")

        valid_videos = (
            set(self.anchors.index)
            & set(self.ef_lookup.keys())
            & set(self.ed_contours.keys())
            & set(self.es_contours.keys())
            & set(self.mask_features.index)
        )
        self.video_ids = sorted(valid_videos)
        print(f"[{split}] {len(self.video_ids)} videos with complete anchor+contour+mask+EF data")

    def _load_contours(self, path):
        df = pd.read_csv(path)
        df["clean_name"] = df["FileName"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
        out = {}
        for name, group in df.groupby("clean_name"):
            group = group.sort_values("PointID")
            pts = group[["X1", "Y1", "X2", "Y2"]].values.astype(np.float32)
            if pts.shape[0] == 100:
                out[name] = pts
        return out

    def __len__(self):
        return len(self.video_ids)

    def _resolve_video_path(self, name):
        cand = os.path.join(self.video_root, name + ".avi")
        if os.path.exists(cand):
            return cand
        return os.path.join(self.video_root, name)

    def _read_selected_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        frames = {}
        needed = set(frame_indices.tolist())
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in needed:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                frames[idx] = gray
            idx += 1
        cap.release()

        out = []
        last_valid = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for i in frame_indices.tolist():
            if i in frames:
                last_valid = frames[i]
            out.append(last_valid)
        return np.stack(out, axis=0)

    def _extract_morph_vector(self, row, prefix):
        v = np.array([row[f"{prefix}_{k}"] for k in MORPH_FEATURE_NAMES], dtype=np.float32)
        if self.morph_mean is not None and self.morph_std is not None:
            v = (v - self.morph_mean) / self.morph_std
        return v

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        anchor_row = self.anchors.loc[video_id]
        ed_frame = int(anchor_row["ED_Frame"])
        es_frame = int(anchor_row["ES_Frame"])
        total_frames = int(anchor_row["TotalFrames"])

        frame_indices, ed_pos, es_pos = build_anchor_sequence(
            total_frames, ed_frame, es_frame, num_frames=self.num_frames
        )

        video_path = self._resolve_video_path(video_id)
        raw_frames = self._read_selected_frames(video_path, frame_indices)

        video_arr = raw_frames.astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(video_arr[:, None, :, :])

        role_flag = np.zeros((self.num_frames, 3), dtype=np.float32)
        role_flag[:, 2] = 1.0
        role_flag[ed_pos] = [1.0, 0.0, 0.0]
        role_flag[es_pos] = [0.0, 1.0, 0.0]
        role_flag_tensor = torch.from_numpy(role_flag)

        # ---- NO JITTER: contour points used exactly as stored ----
        ed_contour = self.ed_contours[video_id]
        es_contour = self.es_contours[video_id]
        ed_contour_norm = (ed_contour / self.img_size).astype(np.float32).flatten()
        es_contour_norm = (es_contour / self.img_size).astype(np.float32).flatten()

        mask_row = self.mask_features.loc[video_id]
        ed_morph = self._extract_morph_vector(mask_row, "ED")
        es_morph = self._extract_morph_vector(mask_row, "ES")
        global_delta = np.array([
            mask_row["Area_Change"], mask_row["Perimeter_Change"],
            mask_row["Area_Ratio"], mask_row["Perimeter_Ratio"],
        ], dtype=np.float32)

        ef = float(self.ef_lookup[video_id])

        return {
            "video": video_tensor,
            "role_flag": role_flag_tensor,
            "ed_pos": torch.tensor(ed_pos, dtype=torch.long),
            "es_pos": torch.tensor(es_pos, dtype=torch.long),
            "ed_contour": torch.from_numpy(ed_contour_norm),
            "es_contour": torch.from_numpy(es_contour_norm),
            "ed_morph": torch.from_numpy(ed_morph),
            "es_morph": torch.from_numpy(es_morph),
            "global_delta": torch.from_numpy(global_delta),
            "ef": torch.tensor(ef, dtype=torch.float32),
            "video_id": video_id,
        }


def compute_morph_norm_stats(base_path, max_samples=300):
    mask_feat_path = os.path.join(base_path, "outputs/mask_features/train_mask_features.csv")
    df = pd.read_csv(mask_feat_path)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    ed_vals = df[[f"ED_{k}" for k in MORPH_FEATURE_NAMES]].values
    es_vals = df[[f"ES_{k}" for k in MORPH_FEATURE_NAMES]].values
    all_vals = np.concatenate([ed_vals, es_vals], axis=0).astype(np.float32)

    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def get_train_ef_values(base_path):
    """Used by train_anchor_fusion.py to build the LDS loss weighting."""
    meta_path = os.path.join(base_path, "outputs/video_metadata/train_metadata.csv")
    df = pd.read_csv(meta_path)
    ef_col = next(c for c in ["EF", "GT_EF", "EjectionFraction"] if c in df.columns)
    return df[ef_col].dropna().values
