# ============================================================
# TEST INFERENCE + EVALUATION IN ONE SCRIPT
# DenseNet121 + BiLSTM
# Inference: test_metadata.csv only
# Evaluation: FileList.csv GT EF
# MATCHES TRAINING ARCHITECTURE (BatchNorm1d)
# ============================================================

import os
import cv2
import math
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

COLAB_PATH_1 = "/content/drive/MyDrive/EchoNet-Dynamic"
COLAB_PATH_2 = "/content/drive/MyDrive/EchoNet_LV"
COLAB_LOCAL = "/content/EchoNet-Dynamic"
DEFAULT_BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
LOCAL_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(COLAB_PATH_1):
    BASE_PATH = COLAB_PATH_1
elif os.path.exists(COLAB_PATH_2):
    BASE_PATH = COLAB_PATH_2
elif os.path.exists(COLAB_LOCAL):
    BASE_PATH = COLAB_LOCAL
elif os.path.exists(DEFAULT_BASE_PATH):
    BASE_PATH = DEFAULT_BASE_PATH
else:
    BASE_PATH = LOCAL_BASE_PATH

print(f"[Colab / Local Auto-Detect] Using BASE_PATH: {BASE_PATH}")

TEST_CSV = os.path.join(BASE_PATH, "outputs/video_metadata/test_metadata.csv")
FILELIST_CSV = os.path.join(BASE_PATH, "data/FileList.csv")
MODEL_PATH = os.path.join(BASE_PATH, "outputs/video_ef_model_cpu_full_ft", "best_model.pth")

SAVE_DIR = os.path.join(BASE_PATH, "outputs/test_inference_eval_results")
os.makedirs(SAVE_DIR, exist_ok=True)

PRED_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
MERGED_CSV = os.path.join(SAVE_DIR, "test_pred_vs_gt.csv")
METRICS_CSV = os.path.join(SAVE_DIR, "test_metrics.csv")

NUM_FRAMES = 32
IMG_SIZE = 112
BATCH_SIZE = 8 if torch.cuda.is_available() else 2
NUM_WORKERS = 2 if torch.cuda.is_available() else 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CPU SETTINGS
# ============================================================

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)

# ============================================================
# HELPERS
# ============================================================

def normalize_name(x):
    x = str(x).strip()
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return x

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resolve_video_path(raw_value, video_root):
    raw_value = str(raw_value).strip()

    candidates = [raw_value]
    if not raw_value.lower().endswith(".avi"):
        candidates.append(raw_value + ".avi")

    # direct paths
    for c in candidates:
        if os.path.exists(c):
            return c

    # under common roots
    possible_roots = [
        os.path.join(BASE_PATH, "data", "videos_split", "test"),
        os.path.join(BASE_PATH, "data", "videos_split"),
        os.path.join(BASE_PATH, "data", "videos"),
        video_root,
    ]

    for c in candidates:
        base = os.path.basename(c)
        for root in possible_roots:
            p = os.path.join(root, base)
            if os.path.exists(p):
                return p

    return None

def sample_frame_indices(total_frames, num_frames):
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)

    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).astype(np.int64)

    idx = np.arange(total_frames, dtype=np.int64)
    pad = np.full(num_frames - total_frames, idx[-1], dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr, pval = pearsonr(y_true, y_pred)
    else:
        corr, pval = float("nan"), float("nan")

    bias = float(np.mean(y_pred - y_true))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(corr),
        "Pearson_p": float(pval),
        "Bias": bias,
    }

# ============================================================
# DATASET
# ============================================================

class EchoVideoTestDataset(Dataset):
    def __init__(self, csv_path, num_frames=32, img_size=112):
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.img_size = img_size

        df = pd.read_csv(csv_path)

        video_col = find_column(df, [
            "VideoPath", "video_path", "FileName", "Filename", "Video", "video", "Path"
        ])

        if video_col is None:
            raise ValueError(f"No video path column found in {csv_path}")

        samples = []
        missing = 0
        skipped = 0

        for _, row in df.iterrows():
            raw_video = row[video_col]

            if pd.isna(raw_video):
                skipped += 1
                continue

            video_path = resolve_video_path(raw_video, BASE_PATH)
            if video_path is None:
                missing += 1
                continue

            samples.append({
                "video_id": normalize_name(raw_video),
                "video_path": video_path,
            })

        self.samples = samples

        print(f"\n[{os.path.basename(csv_path)}]")
        print(f"Kept samples   : {len(self.samples)}")
        print(f"Missing videos : {missing}")
        print(f"Skipped rows   : {skipped}")

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return np.zeros((self.num_frames, 1, self.img_size, self.img_size), dtype=np.float32)

        indices = sample_frame_indices(total_frames, self.num_frames)

        frames = []
        last_frame = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret or frame is None:
                if last_frame is None:
                    gray = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    gray = last_frame.copy()
            else:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame

                gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                gray = gray.astype(np.float32) / 255.0
                last_frame = gray.copy()

            frames.append(gray)

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((self.img_size, self.img_size), dtype=np.float32)]

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        arr = np.stack(frames[:self.num_frames], axis=0)  # (T, H, W)
        arr = arr[:, None, :, :]                          # (T, 1, H, W)
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video = self._load_video(item["video_path"])

        return {
            "video": torch.from_numpy(video),   # (T,1,H,W)
            "video_id": item["video_id"],
        }

# ============================================================
# MODEL
# ============================================================

def build_densenet121_grayscale(pretrained=True):
    try:
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
    except Exception:
        backbone = models.densenet121(weights=None)

    old_conv = backbone.features.conv0
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            new_conv.weight.copy_(old_conv.weight[:, :1, :, :])

    backbone.features.conv0 = new_conv
    return backbone

class DenseNetBiLSTMRegressor(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.3, pretrained=True):
        super().__init__()

        self.backbone = build_densenet121_grayscale(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # MUST MATCH TRAINING
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = self.pool(feats).flatten(1)  # (B*T, 1024)

        feats = feats.view(b, t, -1)         # (B, T, 1024)

        lstm_out, _ = self.lstm(feats)       # (B, T, 512)
        pooled = torch.mean(lstm_out, dim=1) # temporal mean pooling

        out = self.regressor(pooled).squeeze(1)
        return out

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):
    model = DenseNetBiLSTMRegressor(
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        pretrained=False
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(model, loader):
    rows = []

    for batch in loader:
        videos = batch["video"].to(DEVICE)
        video_ids = batch["video_id"]

        preds = model(videos).detach().cpu().numpy()
        preds = np.clip(preds, 0, 100)

        for vid, pred in zip(video_ids, preds):
            rows.append({
                "Video": vid,
                "Pred_EF": float(pred)
            })

    return pd.DataFrame(rows)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_against_gt(pred_df, filelist_csv):
    file_df = pd.read_csv(filelist_csv)

    ef_col = find_column(file_df, ["EF", "ef", "EjectionFraction"])
    video_col = find_column(file_df, ["FileName", "Filename", "Video", "video", "Name"])

    if ef_col is None:
        raise ValueError("No EF column found in FileList.csv")

    if video_col is None:
        raise ValueError("No video/name column found in FileList.csv")

    file_df = file_df[[video_col, ef_col]].copy()
    file_df.columns = ["Video", "GT_EF"]

    file_df["Video"] = file_df["Video"].apply(normalize_name)
    pred_df["Video"] = pred_df["Video"].apply(normalize_name)

    merged = pred_df.merge(file_df, on="Video", how="inner")
    merged = merged.dropna()
    merged = merged[np.isfinite(merged["Pred_EF"])]

    if len(merged) == 0:
        raise ValueError("No matching videos found for evaluation.")

    y_true = merged["GT_EF"].values
    y_pred = merged["Pred_EF"].values

    metrics = regression_metrics(y_true, y_pred)

    merged.to_csv(MERGED_CSV, index=False)

    metrics_df = pd.DataFrame([{
        "Samples": len(merged),
        **metrics
    }])
    metrics_df.to_csv(METRICS_CSV, index=False)

    return merged, metrics

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n====================================")
    print("TEST INFERENCE + EVALUATION")
    print("DenseNet121 + BiLSTM")
    print("====================================\n")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found: {TEST_CSV}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(FILELIST_CSV):
        raise FileNotFoundError(f"FileList.csv not found: {FILELIST_CSV}")

    test_ds = EchoVideoTestDataset(
        TEST_CSV,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    model = load_model(MODEL_PATH)

    pred_df = run_inference(model, test_loader)
    pred_df.to_csv(PRED_CSV, index=False)

    print(f"\nPredictions saved to:\n{PRED_CSV}")
    print(f"Total predicted videos: {len(pred_df)}")

    merged, metrics = evaluate_against_gt(pred_df, FILELIST_CSV)

    print("\n====================================")
    print("EVALUATION RESULTS")
    print("====================================")
    print(f"Samples   : {len(merged)}")
    print(f"MAE       : {metrics['MAE']:.4f}")
    print(f"RMSE      : {metrics['RMSE']:.4f}")
    print(f"R2        : {metrics['R2']:.4f}")
    print(f"Pearson r : {metrics['Pearson_r']:.4f}")
    print(f"Pearson p : {metrics['Pearson_p']:.4f}")
    print(f"Bias      : {metrics['Bias']:.4f}")

    print("\nSaved:")
    print(MERGED_CSV)
    print(METRICS_CSV)

if __name__ == "__main__":
    main()
