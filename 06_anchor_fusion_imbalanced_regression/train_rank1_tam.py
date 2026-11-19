# ============================================================
# RANK 1: TAM-style Multi-Head Temporal Attention
# ------------------------------------------------------------
# Reuses the EXISTING stage2_fused_cache (from
# extract_cached_fused_sequences.py) -- no new caching step needed.
# DenseNet backbone is never touched. Only change from v2: the
# single-head TemporalAttentionPool is replaced with a multi-head
# version (4 heads by default), following the Temporal Attention
# Module design validated on cardiac ultrasound/MRI.
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fds_module_v3 import FDS
from lds_loss import compute_lds_weights, LDSWeightedHuberLoss

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_fused_cache")
V2_CKPT = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")

SAVE_DIR = os.path.join(BASE_PATH, "outputs/rank1_tam")
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_CKPT = os.path.join(SAVE_DIR, "best_head.pth")
LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
TEST_PRED_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
TEST_METRICS_CSV = os.path.join(SAVE_DIR, "test_metrics.csv")
TEST_BUCKET_CSV = os.path.join(SAVE_DIR, "test_bucket_metrics.csv")

FUSED_DIM = 1107
LSTM_HIDDEN = 256
GLOBAL_EMBED_DIM = 16
NUM_ATTENTION_HEADS = 4

EPOCHS = 50
PATIENCE = 12
BATCH_SIZE = 16
LR = 3e-5
SAMPLER_MAX_WEIGHT_RATIO = 3.0
FDS_NUM_BINS = 20
FDS_SIGMA = 3.0
FDS_START_SMOOTH_EPOCH = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadTemporalAttentionPool(nn.Module):
    def __init__(self, input_dim, num_heads=NUM_ATTENTION_HEADS, hidden_dim=128):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(input_dim * num_heads, input_dim)

    def forward(self, x):
        pooled_heads = []
        for head in self.heads:
            weights = torch.softmax(head(x), dim=1)
            pooled_heads.append(torch.sum(x * weights, dim=1))
        combined = torch.cat(pooled_heads, dim=-1)
        return self.output_proj(combined)


class TAMHead(nn.Module):
    def __init__(self, fused_dim=FUSED_DIM, lstm_hidden=LSTM_HIDDEN, global_embed_dim=GLOBAL_EMBED_DIM):
        super().__init__()
        self.lstm = nn.LSTM(fused_dim, lstm_hidden, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.3)
        self.temporal_pool = MultiHeadTemporalAttentionPool(lstm_hidden * 2)
        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, global_embed_dim), nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + global_embed_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, fused, global_delta, fds=None, ef=None, epoch=None):
        lstm_out, _ = self.lstm(fused)
        pooled = self.temporal_pool(lstm_out)
        global_embed = self.global_delta_encoder(global_delta)
        final_feat = torch.cat([pooled, global_embed], dim=-1)

        if fds is not None and self.training and ef is not None:
            fds.collect(final_feat, ef)
            if epoch is not None:
                final_feat = fds.smooth(final_feat, ef, epoch)

        return self.regressor(final_feat).squeeze(1)

    def load_from_v2_checkpoint(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        own_state = self.state_dict()
        loaded = 0
        for key in own_state:
            if key in state and own_state[key].shape == state[key].shape:
                own_state[key] = state[key]
                loaded += 1
        self.load_state_dict(own_state)
        print(f"Warm-started {loaded}/{len(own_state)} parameter tensors from v2 checkpoint "
              f"(temporal_pool stays new/random, since it's now multi-head).")


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson_r": corr, "Bias": float(np.mean(y_pred - y_true))}


def bucket_metrics(y_true, y_pred, return_dict=False):
    df = pd.DataFrame({"gt": y_true, "pred": y_pred})
    df["abs_error"] = (df["pred"] - df["gt"]).abs()
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["bucket"] = pd.cut(df["gt"], bins=bins, labels=labels)
    if return_dict:
        return df.groupby("bucket", observed=True)["abs_error"].mean().to_dict()
    counts = df.groupby("bucket", observed=True).size()
    maes = df.groupby("bucket", observed=True)["abs_error"].mean()
    return pd.DataFrame({"n": counts, "MAE": maes}).reset_index()


def load_cache(split):
    d = torch.load(os.path.join(CACHE_DIR, f"{split}_fused.pt"))
    return d["fused"], d["global_delta"], d["ef"], d["video_ids"]


def build_capped_sampler(ef_values, bin_edges, bin_weights, max_ratio):
    ef_values = np.asarray(ef_values)
    bin_idx = np.digitize(ef_values, bin_edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, len(bin_weights) - 1)
    w = bin_weights[bin_idx].copy()
    cap = w.mean() * max_ratio
    w = np.clip(w, None, cap)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)


def main():
    print("\n==== RANK 1: TAM Multi-Head Temporal Attention ====")
    print(f"Device: {DEVICE}\n")

    train_fused, train_gd, train_ef, _ = load_cache("train")
    val_fused, val_gd, val_ef, _ = load_cache("val")
    print(f"Cached fused sequences -- train: {train_fused.shape}, val: {val_fused.shape}")

    bin_edges, bin_weights = compute_lds_weights(train_ef.numpy(), sigma=2.0)
    sampler = build_capped_sampler(train_ef.numpy(), bin_edges, bin_weights, SAMPLER_MAX_WEIGHT_RATIO)
    train_loader = DataLoader(TensorDataset(train_fused, train_gd, train_ef), batch_size=BATCH_SIZE, sampler=sampler)
    val_fused_dev, val_gd_dev = val_fused.to(DEVICE), val_gd.to(DEVICE)

    head = TAMHead().to(DEVICE)
    fds = FDS(feature_dim=LSTM_HIDDEN * 2 + GLOBAL_EMBED_DIM, num_bins=FDS_NUM_BINS, sigma=FDS_SIGMA,
              start_update_epoch=1, start_smooth_epoch=FDS_START_SMOOTH_EPOCH).to(DEVICE)
    criterion = LDSWeightedHuberLoss(bin_edges, bin_weights, delta=1.0).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    start_epoch = 1
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    if os.path.exists(LAST_CKPT):
        print(f"Resuming from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        head.load_state_dict(ckpt["head_state_dict"])
        fds.load_state_dict(ckpt["fds_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(HISTORY_CSV):
            history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        print("No resume checkpoint -- fresh start, warm-starting from v2.")
        head.load_from_v2_checkpoint(V2_CKPT, DEVICE)

    if start_epoch > EPOCHS:
        print("Already past configured EPOCHS.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        head.train()
        running_loss, n = 0.0, 0
        for fused, gd, ef in train_loader:
            fused, gd, ef = fused.to(DEVICE), gd.to(DEVICE), ef.to(DEVICE)
            preds = head(fused, gd, fds=fds, ef=ef, epoch=epoch)
            loss = criterion(preds, ef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * fused.size(0)
            n += fused.size(0)
        fds.update_last_epoch_stats(epoch)

        head.eval()
        with torch.no_grad():
            val_pred = head(val_fused_dev, val_gd_dev).cpu().numpy()
        m = regression_metrics(val_ef.numpy(), val_pred)
        buckets = bucket_metrics(val_ef.numpy(), val_pred, return_dict=True)
        scheduler.step(m["MAE"])

        row = {"Epoch": epoch, "Train_Loss": running_loss / n, **m,
               "LR": optimizer.param_groups[0]["lr"], **{f"MAE_{k}": v for k, v in buckets.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        print(f"Epoch {epoch}/{EPOCHS} | Val MAE: {m['MAE']:.3f} | Pearson r: {m['Pearson_r']:.3f} | " +
              " | ".join(f"{k}:{v:.2f}" for k, v in buckets.items()))

        is_best = m["MAE"] < best_val_mae
        if is_best:
            best_val_mae = m["MAE"]
            patience_counter = 0
            torch.save({"head_state_dict": head.state_dict(), "fds_state_dict": fds.state_dict(),
                        "epoch": epoch, "best_val_mae": best_val_mae}, BEST_CKPT)
            print(f"  New best. Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1

        torch.save({
            "head_state_dict": head.state_dict(), "fds_state_dict": fds.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "best_val_mae": best_val_mae, "patience_counter": patience_counter,
        }, LAST_CKPT)

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n==== FINAL TEST EVALUATION ====")
    test_fused, test_gd, test_ef, test_ids = load_cache("test")
    best = torch.load(BEST_CKPT, map_location=DEVICE)
    head.load_state_dict(best["head_state_dict"])
    head.eval()
    with torch.no_grad():
        test_pred = head(test_fused.to(DEVICE), test_gd.to(DEVICE)).cpu().numpy()
    test_pred = np.clip(test_pred, 0, 100)

    test_metrics = regression_metrics(test_ef.numpy(), test_pred)
    test_buckets = bucket_metrics(test_ef.numpy(), test_pred)
    print(test_metrics)
    print(test_buckets.to_string(index=False))

    pd.DataFrame({"Video": test_ids, "GT_EF": test_ef.numpy(), "Pred_EF": test_pred}).to_csv(TEST_PRED_CSV, index=False)
    pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_CSV, index=False)
    test_buckets.to_csv(TEST_BUCKET_CSV, index=False)
    print(f"\nSaved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
