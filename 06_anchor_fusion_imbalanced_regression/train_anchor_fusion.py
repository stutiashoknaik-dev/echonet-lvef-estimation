# ============================================================
# TRAIN v2: Anchor-Fusion DenseNet-BiLSTM
# ------------------------------------------------------------
# CHANGES FROM v1:
#   - Modality dropout: REMOVED (model no longer accepts the param)
#   - Contour jitter: REMOVED (dataset no longer applies it, any split)
#   - Regressor: BatchNorm1d -> LayerNorm (see anchor_fusion_model.py)
#   - Loss: Huber -> LDS-weighted Huber (see lds_loss.py), addresses
#     the severe/hyperdynamic EF underperformance found in v1 results
#   - Scheduler: patience 2 -> 4, so single noisy epochs don't
#     immediately halve the learning rate
# Resume-from-checkpoint and per-epoch checkpointing are unchanged.
# ============================================================

import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from anchor_fusion_dataset import AnchorFusionEFDataset, compute_morph_norm_stats, get_train_ef_values
from anchor_fusion_model import AnchorFusionDenseNetBiLSTM
from lds_loss import compute_lds_weights, LDSWeightedHuberLoss

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

OLD_CKPT = os.path.join(BASE_PATH, "outputs/video_ef_model_cpu_full_ft/best_model.pth")
SAVE_DIR = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2")
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_CKPT = os.path.join(SAVE_DIR, "best_model.pth")
LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
VAL_PRED_CSV = os.path.join(SAVE_DIR, "validation_predictions.csv")

NUM_FRAMES = 32
BATCH_SIZE = 4
EPOCHS = 40               # bump this later once you see the trend, same as before
PATIENCE = 12              # raised from 6 -- give the more-stable setup more room

LR_BACKBONE = 5e-6
LR_NEW_MODULES = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 5.0

SCHEDULER_PATIENCE = 4        # was 2 -- don't react to single noisy epochs
LDS_SIGMA = 2.0                # Gaussian smoothing width for LDS, in EF points

NUM_WORKERS = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True
GRAD_ACCUM_STEPS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(42)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "Pearson_r": float(corr), "Bias": bias}


def bucket_metrics(y_true, y_pred):
    """Same clinical-bucket breakdown used in the diagnosis -- kept as a
    permanent part of evaluation now, not a one-off diagnostic."""
    df = pd.DataFrame({"gt": y_true, "pred": y_pred})
    df["abs_error"] = (df["pred"] - df["gt"]).abs()
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["bucket"] = pd.cut(df["gt"], bins=bins, labels=labels)
    return df.groupby("bucket", observed=True)["abs_error"].mean().to_dict()


def move_batch(batch, device):
    keys = ["video", "role_flag", "ed_pos", "es_pos", "ed_contour", "es_contour",
             "ed_morph", "es_morph", "global_delta", "ef"]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        b = move_batch(batch, DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
                          b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"],
                          b["global_delta"])
            loss = criterion(preds, b["ef"]) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = b["video"].size(0)
        running_loss += loss.item() * GRAD_ACCUM_STEPS * bs
        n += bs

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, save_csv=None):
    model.eval()
    running_loss, n = 0.0, 0
    all_gt, all_pred, all_vid = [], [], []

    for batch in loader:
        b = move_batch(batch, DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
                          b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"],
                          b["global_delta"])
            loss = criterion(preds, b["ef"])

        preds_np = np.clip(preds.float().cpu().numpy(), 0, 100)
        running_loss += loss.item() * b["video"].size(0)
        n += b["video"].size(0)

        all_gt.extend(b["ef"].cpu().numpy().tolist())
        all_pred.extend(preds_np.tolist())
        all_vid.extend(batch["video_id"])

    metrics = regression_metrics(all_gt, all_pred)
    buckets = bucket_metrics(all_gt, all_pred)
    avg_loss = running_loss / max(n, 1)
    pred_df = pd.DataFrame({"Video": all_vid, "GT_EF": all_gt, "Pred_EF": all_pred})
    if save_csv:
        pred_df.to_csv(save_csv, index=False)
    return avg_loss, metrics, buckets, pred_df


def main():
    print("\n====================================")
    print("TRAINING v2: Anchor-Fusion DenseNet-BiLSTM")
    print("Changes: no jitter, no modality dropout, LayerNorm regressor, LDS-weighted loss")
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | Epochs: {EPOCHS}")
    print("====================================\n")

    morph_mean, morph_std = compute_morph_norm_stats(BASE_PATH)
    np.save(os.path.join(SAVE_DIR, "morph_mean.npy"), morph_mean)
    np.save(os.path.join(SAVE_DIR, "morph_std.npy"), morph_std)

    # ---- LDS weighting, computed once from train split only ----
    train_ef_values = get_train_ef_values(BASE_PATH)
    bin_edges, bin_weights = compute_lds_weights(train_ef_values, sigma=LDS_SIGMA)
    np.save(os.path.join(SAVE_DIR, "lds_bin_edges.npy"), bin_edges)
    np.save(os.path.join(SAVE_DIR, "lds_bin_weights.npy"), bin_weights)
    print(f"LDS weights computed from {len(train_ef_values)} training EF values "
          f"(sigma={LDS_SIGMA}). Weight range: {bin_weights.min():.2f} - {bin_weights.max():.2f}")

    train_ds = AnchorFusionEFDataset("train", BASE_PATH, num_frames=NUM_FRAMES,
                                      morph_mean=morph_mean, morph_std=morph_std)
    val_ds = AnchorFusionEFDataset("val", BASE_PATH, num_frames=NUM_FRAMES,
                                    morph_mean=morph_mean, morph_std=morph_std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )

    model = AnchorFusionDenseNetBiLSTM().to(DEVICE)  # no modality_dropout_p param anymore

    new_module_params = (
        list(model.cbam.parameters()) + list(model.contour_encoder.parameters())
        + list(model.morph_encoder.parameters()) + list(model.global_delta_encoder.parameters())
        + list(model.lstm.parameters()) + list(model.temporal_pool.parameters())
        + list(model.regressor.parameters())
    )
    optimizer = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": LR_BACKBONE},
         {"params": new_module_params, "lr": LR_NEW_MODULES}],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = LDSWeightedHuberLoss(bin_edges, bin_weights, delta=1.0).to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 1
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    if os.path.exists(LAST_CKPT):
        print(f"Found existing checkpoint at {LAST_CKPT} -- RESUMING.")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if USE_AMP and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(HISTORY_CSV):
            history = pd.read_csv(HISTORY_CSV).to_dict("records")
        print(f"Resuming from epoch {start_epoch}, best_val_mae so far = {best_val_mae:.4f}")
    else:
        print("No resume checkpoint found -- fresh start (v2 run).")
        if os.path.exists(OLD_CKPT):
            model.load_pretrained_backbone(OLD_CKPT, device=DEVICE)
            print(f"Loaded backbone weights from {OLD_CKPT}")
        else:
            print("WARNING: old checkpoint not found -- training backbone from ImageNet init only.")

    if start_epoch > EPOCHS:
        print(f"start_epoch ({start_epoch}) already exceeds EPOCHS ({EPOCHS}). Raise EPOCHS to continue.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_metrics, val_buckets, val_pred_df = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["MAE"])

        row = {"Epoch": epoch, "Train_Loss": train_loss, "Val_Loss": val_loss, **val_metrics,
               "LR": optimizer.param_groups[0]["lr"], **{f"MAE_{k}": v for k, v in val_buckets.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        bucket_str = " | ".join(f"{k}: {v:.2f}" for k, v in val_buckets.items())
        print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_metrics['MAE']:.4f} "
              f"| Val RMSE: {val_metrics['RMSE']:.4f} | Val R2: {val_metrics['R2']:.4f} "
              f"| Pearson r: {val_metrics['Pearson_r']:.4f}")
        print(f"  Bucket MAE -> {bucket_str}")

        is_best = val_metrics["MAE"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["MAE"]
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "best_val_mae": best_val_mae}, BEST_CKPT)
            val_pred_df.to_csv(VAL_PRED_CSV, index=False)
            print(f"New best model saved. Best Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "patience_counter": patience_counter,
        }, LAST_CKPT)

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n====================================")
    print("TRAINING COMPLETE (this run)")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    print(f"Best model: {BEST_CKPT}")
    print("====================================")


if __name__ == "__main__":
    main()
