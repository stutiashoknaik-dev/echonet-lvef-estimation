# ============================================================
# FULL CPU TRAINING SCRIPT
# DenseNet121 + BiLSTM for EF Prediction
# ============================================================

import os
import math
import random
import warnings
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models

warnings.filterwarnings("ignore")

# ============================================================
# BASIC CPU SETTINGS
# ============================================================

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

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

TRAIN_CSV = os.path.join(BASE_PATH, "outputs/video_metadata/train_metadata.csv")
VAL_CSV   = os.path.join(BASE_PATH, "outputs/video_metadata/val_metadata.csv")

VIDEO_ROOT = os.path.join(BASE_PATH, "videos_split")

SAVE_DIR = os.path.join(BASE_PATH, "densenet_bilstm")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_FRAMES = 32
IMG_SIZE = 112
BATCH_SIZE = 8 if torch.cuda.is_available() else 4
EPOCHS = 30
PATIENCE = 10
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 5.0
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
RESUME = False

RESUME_PATH = os.path.join(SAVE_DIR, "latest_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {DEVICE}")

if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))

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
# HELPER FUNCTIONS
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

    # direct check
    for c in candidates:
        if os.path.exists(c):
            return c

    # check under video_root
    for c in candidates:
        base = os.path.basename(c)
        p = os.path.join(video_root, base)
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

class EchoVideoEFDataset(Dataset):
    def __init__(self, csv_path, video_root, num_frames=32, img_size=112):
        self.csv_path = csv_path
        self.video_root = video_root
        self.num_frames = num_frames
        self.img_size = img_size

        df = pd.read_csv(csv_path)

        video_col = find_column(df, [
            "VideoPath", "video_path", "FileName", "Filename", "Video", "video", "Path"
        ])
        ef_col = find_column(df, [
            "EF", "ef", "GT_EF", "Gt_EF", "EjectionFraction", "Ejection_Fraction"
        ])

        if video_col is None:
            raise ValueError(f"No video/path column found in {csv_path}")
        if ef_col is None:
            raise ValueError(f"No EF column found in {csv_path}")

        samples = []
        missing = 0
        skipped = 0

        for _, row in df.iterrows():
            raw_video = row[video_col]
            ef = row[ef_col]

            if pd.isna(raw_video) or pd.isna(ef):
                skipped += 1
                continue

            video_path = resolve_video_path(raw_video, video_root)
            if video_path is None:
                missing += 1
                continue

            samples.append({
                "video_id": normalize_name(raw_video),
                "video_path": video_path,
                "ef": float(ef),
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
        ef = np.float32(item["ef"])

        return {
            "video": torch.from_numpy(video),                # (T,1,H,W)
            "ef": torch.tensor(ef, dtype=torch.float32),     # scalar
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
        try:
            backbone = models.densenet121(pretrained=pretrained)
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
        """
        x: (B, T, 1, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feats = self.backbone.features(x)
        feats = F.relu(feats, inplace=True)
        feats = self.pool(feats).flatten(1)   # (B*T, 1024)

        feats = feats.view(b, t, -1)          # (B, T, 1024)

        lstm_out, _ = self.lstm(feats)        # (B, T, 512)
        pooled = torch.mean(lstm_out, dim=1)  # temporal mean pooling

        out = self.regressor(pooled).squeeze(1)
        return out

# ============================================================
# TRAIN / EVAL
# ============================================================

def make_loader(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type=="cuda"),
    drop_last=drop_last
  )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n = 0

    for batch in loader:
        videos = batch["video"].to(device)
        targets = batch["ef"].to(device)

        optimizer.zero_grad(set_to_none=True)

        preds = model(videos)
        loss = criterion(preds, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        bs = videos.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, save_predictions_csv=None):
    model.eval()
    running_loss = 0.0
    n = 0

    all_gt = []
    all_pred = []
    all_vid = []

    for batch in loader:
        videos = batch["video"].to(device)
        targets = batch["ef"].to(device)

        preds = model(videos)
        loss = criterion(preds, targets)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        preds_np = np.clip(preds_np, 0, 100)

        running_loss += loss.item() * videos.size(0)
        n += videos.size(0)

        all_gt.extend(targets_np.tolist())
        all_pred.extend(preds_np.tolist())
        all_vid.extend(batch["video_id"])

    metrics = regression_metrics(all_gt, all_pred)
    avg_loss = running_loss / max(n, 1)

    pred_df = pd.DataFrame({
        "Video": all_vid,
        "GT_EF": all_gt,
        "Pred_EF": all_pred
    })

    if save_predictions_csv is not None:
        pred_df.to_csv(save_predictions_csv, index=False)

    return avg_loss, metrics, pred_df

# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(path, model, optimizer, epoch, best_val_mae, history):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": best_val_mae,
        "history": history,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_mae = ckpt.get("best_val_mae", float("inf"))
    history = ckpt.get("history", [])
    return start_epoch, best_val_mae, history

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n====================================")
    print("CPU FULL FINE-TUNING TRAINING")
    print("DenseNet121 + BiLSTM")
    print("====================================\n")

    train_ds = EchoVideoEFDataset(TRAIN_CSV, VIDEO_ROOT, num_frames=NUM_FRAMES, img_size=IMG_SIZE)
    val_ds   = EchoVideoEFDataset(VAL_CSV, VIDEO_ROOT, num_frames=NUM_FRAMES, img_size=IMG_SIZE)


    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True,drop_last=True)
    val_loader   = make_loader(val_ds, BATCH_SIZE, shuffle=False,drop_last=False)


    model = DenseNetBiLSTMRegressor(
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        pretrained=True
    ).to(DEVICE)

    # Full fine-tuning, but with a smaller LR for the backbone
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.lstm.parameters()) + list(model.regressor.parameters()), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.HuberLoss(delta=1.0) if hasattr(nn, "HuberLoss") else nn.SmoothL1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    start_epoch = 1
    best_val_mae = float("inf")
    history = []
    patience_counter = 0

    latest_ckpt = os.path.join(SAVE_DIR, "latest_model.pth")
    best_ckpt = os.path.join(SAVE_DIR, "best_model.pth")
    history_csv = os.path.join(SAVE_DIR, "training_history.csv")
    val_pred_csv = os.path.join(SAVE_DIR, "validation_predictions.csv")

    if RESUME and os.path.exists(RESUME_PATH):
        print(f"Resuming from checkpoint: {RESUME_PATH}")
        start_epoch, best_val_mae, history = load_checkpoint(RESUME_PATH, model, optimizer)
        print(f"Resumed at epoch {start_epoch}, best val MAE = {best_val_mae:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        val_loss, val_metrics, val_pred_df = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            save_predictions_csv=None
        )

        scheduler.step(val_metrics["MAE"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Val_Pearson_r": val_metrics["Pearson_r"],
            "Val_Pearson_p": val_metrics["Pearson_p"],
            "Val_Bias": val_metrics["Bias"],
            "LR": current_lr
        }
        history.append(row)

        pd.DataFrame(history).to_csv(history_csv, index=False)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val MAE    : {val_metrics['MAE']:.4f}")
        print(f"Val RMSE   : {val_metrics['RMSE']:.4f}")
        print(f"Val R2     : {val_metrics['R2']:.4f}")
        print(f"Val Pears. : {val_metrics['Pearson_r']:.4f}")
        print(f"Val Bias   : {val_metrics['Bias']:.4f}")
        print(f"LR         : {current_lr:.2e}")

        # Save latest
        save_checkpoint(latest_ckpt, model, optimizer, epoch, best_val_mae, history)

        # Save best
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            patience_counter = 0

            save_checkpoint(best_ckpt, model, optimizer, epoch, best_val_mae, history)
            val_pred_df.to_csv(val_pred_csv, index=False)

            print(f"New best model saved. Best Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break


    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")
    print(f"Saved best model : {best_ckpt}")
    print(f"Saved latest model: {latest_ckpt}")
    print(f"History CSV      : {history_csv}")
    print(f"Val preds CSV    : {val_pred_csv}")

if __name__ == "__main__":
    main()
