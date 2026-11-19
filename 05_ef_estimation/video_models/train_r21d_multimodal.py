# ============================================================
# TRAIN R(2+1)D MULTIMODAL EF MODEL
# Video + Contours + Mask Features
# CPU TRAINING
# ============================================================

import os
import sys
import json
import math
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# LOCAL IMPORTS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from fusion_dataset import FusionDataset
from r21d_multimodal_model import R21DMultimodalEF

# ============================================================
# CPU SETTINGS
# ============================================================

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ============================================================
# CONFIG & COLAB PATH RESOLUTION
# ============================================================

COLAB_PATH_1 = "/content/drive/MyDrive/EchoNet-Dynamic"
# COLAB_PATH_2 = "/content/drive/MyDrive/EchoNet_LV"
COLAB_LOCAL = "/content/EchoNet-Dynamic"
DEFAULT_BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
LOCAL_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(COLAB_PATH_1):
    BASE_PATH = COLAB_PATH_1
# elif os.path.exists(COLAB_PATH_2):
#     BASE_PATH = COLAB_PATH_2
elif os.path.exists(COLAB_LOCAL):
    BASE_PATH = COLAB_LOCAL
elif os.path.exists(DEFAULT_BASE_PATH):
    BASE_PATH = DEFAULT_BASE_PATH
else:
    BASE_PATH = LOCAL_BASE_PATH

print(f"[Colab / Local Auto-Detect] Using BASE_PATH: {BASE_PATH}")

TRAIN_META = os.path.join(
    BASE_PATH,
    "outputs",
    "fusion_metadata",
    "fusion_train_metadata.csv"
)

VAL_META = os.path.join(
    BASE_PATH,
    "outputs",
    "fusion_metadata",
    "fusion_val_metadata.csv"
)

TRAIN_MASK_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "train_mask_features.csv"
)

VAL_MASK_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "val_mask_features.csv"
)

TRAIN_ED_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs",
    "train_val_contours",
    "Train_ED_Contours_100Pts.csv"
)

TRAIN_ES_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs",
    "train_val_contours",
    "Train_ES_Contours_100Pts.csv"
)

VAL_ED_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs",
    "train_val_contours",
    "Val_ED_Contours_100Pts.csv"
)

VAL_ES_CONTOUR = os.path.join(
    BASE_PATH,
    "outputs",
    "train_val_contours",
    "Val_ES_Contours_100Pts.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "r21d_multimodal_adaptive"
)

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 10
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 5.0
NUM_WORKERS = 2
RESUME = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_CKPT = os.path.join(SAVE_DIR, "best_model.pth")
LATEST_CKPT = os.path.join(SAVE_DIR, "latest_model.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
VAL_PRED_CSV = os.path.join(SAVE_DIR, "validation_predictions.csv")
NORM_STATS_JSON = os.path.join(SAVE_DIR, "mask_norm_stats.json")

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

def compute_mask_stats(mask_csv):
    df = pd.read_csv(mask_csv)
    feature_cols = [c for c in df.columns if c not in ["FileName", "EF"]]
    arr = df[feature_cols].to_numpy(dtype=np.float32)

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0

    return feature_cols, mean.astype(np.float32), std.astype(np.float32)

def prepare_batch(batch, mask_mean, mask_std):
    videos, contours, masks, ef = batch

    videos = videos.to(DEVICE, non_blocking=True).float()
    contours = contours.to(DEVICE, non_blocking=True).float()
    masks = masks.to(DEVICE, non_blocking=True).float()
    ef = ef.to(DEVICE, non_blocking=True).float()

    mask_mean = mask_mean.to(DEVICE)
    mask_std = mask_std.to(DEVICE)

    masks = (masks - mask_mean) / mask_std

    return videos, contours, masks, ef

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_mae, history, mask_mean, mask_std):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_mae": best_val_mae,
        "history": history,
        "mask_mean": mask_mean.cpu(),
        "mask_std": mask_std.cpu(),
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_mae = ckpt.get("best_val_mae", float("inf"))
    history = ckpt.get("history", [])
    mask_mean = ckpt.get("mask_mean", None)
    mask_std = ckpt.get("mask_std", None)
    return start_epoch, best_val_mae, history, mask_mean, mask_std

@torch.no_grad()
def evaluate(model, loader, criterion, mask_mean, mask_std, save_csv_path=None):
    model.eval()

    all_gt = []
    all_pred = []
    all_vid = []
    running_loss = 0.0
    n = 0

    for batch in loader:
        videos, contours, masks, ef = prepare_batch(batch, mask_mean, mask_std)

        preds = model(videos, contours, masks)
        loss = criterion(preds, ef)

        preds_np = np.clip(preds.detach().cpu().numpy(), 0, 100)
        ef_np = ef.detach().cpu().numpy()

        running_loss += loss.item() * videos.size(0)
        n += videos.size(0)

        all_gt.extend(ef_np.tolist())
        all_pred.extend(preds_np.tolist())

        if hasattr(loader.dataset, "meta"):
            all_vid.extend(loader.dataset.meta.iloc[len(all_vid):len(all_vid) + len(preds_np)]["FileName"].tolist())
        else:
            all_vid.extend([""] * len(preds_np))

    metrics = regression_metrics(all_gt, all_pred)
    avg_loss = running_loss / max(n, 1)

    pred_df = pd.DataFrame({
        "FileName": all_vid,
        "GT_EF": all_gt,
        "Pred_EF": all_pred
    })

    if save_csv_path is not None:
        pred_df.to_csv(save_csv_path, index=False)

    return avg_loss, metrics, pred_df

def train_one_epoch(model, loader, optimizer, criterion, mask_mean, mask_std):
    model.train()
    running_loss = 0.0
    n = 0

    for batch in loader:
        videos, contours, masks, ef = prepare_batch(batch, mask_mean, mask_std)

        optimizer.zero_grad(set_to_none=True)

        preds = model(videos, contours, masks)
        loss = criterion(preds, ef)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        bs = videos.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n====================================")
    print("R(2+1)D MULTIMODAL TRAINING")
    print("Video + Contours + Mask Features")
    print("====================================\n")

    # --------------------------------------------------------
    # MASK NORMALIZATION STATS
    # --------------------------------------------------------

    mask_cols, mask_mean_np, mask_std_np = compute_mask_stats(TRAIN_MASK_CSV)

    with open(NORM_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mask_feature_columns": mask_cols,
                "mask_mean": mask_mean_np.tolist(),
                "mask_std": mask_std_np.tolist(),
                "contour_scale": 112.0
            },
            f,
            indent=2
        )

    mask_mean = torch.tensor(mask_mean_np, dtype=torch.float32)
    mask_std = torch.tensor(mask_std_np, dtype=torch.float32)

    print("Mask stats computed from training split.")
    print(f"Mask features : {len(mask_cols)}")
    print(f"Saved norm stats to: {NORM_STATS_JSON}")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = FusionDataset(
        metadata_csv=TRAIN_META,
        mask_feature_csv=TRAIN_MASK_CSV,
        ed_contour_csv=TRAIN_ED_CONTOUR,
        es_contour_csv=TRAIN_ES_CONTOUR
    )

    val_dataset = FusionDataset(
        metadata_csv=VAL_META,
        mask_feature_csv=VAL_MASK_CSV,
        ed_contour_csv=VAL_ED_CONTOUR,
        es_contour_csv=VAL_ES_CONTOUR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type=="cuda"),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type=="cuda"),
        drop_last=False
    )

    print(f"\nTrain Samples : {len(train_dataset)}")
    print(f"Val Samples   : {len(val_dataset)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = R21DMultimodalEF().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.video_encoder.parameters(), "lr": LR_BACKBONE},
            {
                "params": list(model.contour_encoder.parameters()) +
                          list(model.mask_encoder.parameters()) +
                          list(model.fusion_head.parameters()),
                "lr": LR_HEAD
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.HuberLoss(delta=1.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 1
    best_val_mae = float("inf")
    history = []

    if RESUME and os.path.exists(LATEST_CKPT):
        print(f"\nResuming from checkpoint: {LATEST_CKPT}")
        start_epoch, best_val_mae, history, saved_mean, saved_std = load_checkpoint(
            LATEST_CKPT, model, optimizer, scheduler
        )
        if saved_mean is not None and saved_std is not None:
            mask_mean.copy_(saved_mean)
            mask_std.copy_(saved_std)
        print(f"Resumed at epoch {start_epoch}, best val MAE = {best_val_mae:.4f}")

    # --------------------------------------------------------
    # SANITY FORWARD PASS
    # --------------------------------------------------------

    print("\nRunning forward-pass sanity test...")

    batch = next(iter(train_loader))
    videos, contours, masks, ef = prepare_batch(batch, mask_mean, mask_std)

    with torch.no_grad():
        preds = model(videos, contours, masks)

    print("Video Input   :", videos.shape)
    print("Contour Input :", contours.shape)
    print("Mask Input    :", masks.shape)
    print("Pred Shape    :", preds.shape)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            mask_mean,
            mask_std
        )

        val_loss, val_metrics, val_pred_df = evaluate(
            model,
            val_loader,
            criterion,
            mask_mean,
            mask_std,
            save_csv_path=None
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

        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val MAE    : {val_metrics['MAE']:.4f}")
        print(f"Val RMSE   : {val_metrics['RMSE']:.4f}")
        print(f"Val R2     : {val_metrics['R2']:.4f}")
        print(f"Val Pears. : {val_metrics['Pearson_r']:.4f}")
        print(f"Val Bias   : {val_metrics['Bias']:.4f}")
        print(f"LR         : {current_lr:.2e}")

        # Save latest every epoch
        save_checkpoint(
            LATEST_CKPT,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_mae,
            history,
            mask_mean,
            mask_std
        )

        # Save best
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            patience_counter = 0

            save_checkpoint(
                BEST_CKPT,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_mae,
                history,
                mask_mean,
                mask_std
            )

            val_pred_df.to_csv(VAL_PRED_CSV, index=False)

            print(f"New best model saved. Best Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # FINAL LOAD BEST AND REPORT
    # --------------------------------------------------------

    print("\nLoading best checkpoint for final validation report...")
    if os.path.exists(BEST_CKPT):
        ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Best checkpoint loaded from epoch {ckpt.get('epoch', 'N/A')}")
    else:
        print("Best checkpoint not found. Using current model weights.")

    final_val_loss, final_val_metrics, final_val_pred_df = evaluate(
        model,
        val_loader,
        criterion,
        mask_mean,
        mask_std,
        save_csv_path=VAL_PRED_CSV
    )

    summary_df = pd.DataFrame([{
        "Best_Val_MAE": best_val_mae,
        "Final_Val_Loss": final_val_loss,
        "Final_Val_MAE": final_val_metrics["MAE"],
        "Final_Val_RMSE": final_val_metrics["RMSE"],
        "Final_Val_R2": final_val_metrics["R2"],
        "Final_Val_Pearson_r": final_val_metrics["Pearson_r"],
        "Final_Val_Pearson_p": final_val_metrics["Pearson_p"],
        "Final_Val_Bias": final_val_metrics["Bias"],
        "Train_Samples": len(train_dataset),
        "Val_Samples": len(val_dataset),
        "Batch_Size": BATCH_SIZE,
        "Epochs_Ran": len(history),
        "Device": "CPU"
    }])

    summary_csv = os.path.join(SAVE_DIR, "final_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")
    print(f"Saved best model : {BEST_CKPT}")
    print(f"Saved latest model: {LATEST_CKPT}")
    print(f"History CSV      : {HISTORY_CSV}")
    print(f"Val preds CSV    : {VAL_PRED_CSV}")
    print(f"Summary CSV      : {summary_csv}")

if __name__ == "__main__":
    main()
