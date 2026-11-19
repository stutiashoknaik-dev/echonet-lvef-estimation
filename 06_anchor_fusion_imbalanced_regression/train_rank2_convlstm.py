# ============================================================
# RANK 2: Attention-Boosted Bidirectional ConvLSTM
# ------------------------------------------------------------
# Runs on the spatial-map cache from extract_cached_spatial_maps.py.
# DenseNet backbone still never touched -- but unlike TAM, this
# keeps each frame's spatial grid intact through the temporal step,
# so the recurrent unit can track WHERE the LV boundary is moving,
# not just THAT something changed frame to frame.
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
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_spatial_cache")
V2_CKPT = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")

SAVE_DIR = os.path.join(BASE_PATH, "outputs/rank2_convlstm")
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_CKPT = os.path.join(SAVE_DIR, "best_head.pth")
LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
TEST_PRED_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
TEST_METRICS_CSV = os.path.join(SAVE_DIR, "test_metrics.csv")
TEST_BUCKET_CSV = os.path.join(SAVE_DIR, "test_bucket_metrics.csv")

CNN_CHANNELS = 1024
SPATIAL_H, SPATIAL_W = 3, 3
NON_SPATIAL_DIM = 83
CONVLSTM_HIDDEN = 128
GLOBAL_EMBED_DIM = 16

EPOCHS = 40
PATIENCE = 10
BATCH_SIZE = 8
LR = 3e-5
SAMPLER_MAX_WEIGHT_RATIO = 3.0
FDS_NUM_BINS = 20
FDS_SIGMA = 3.0
FDS_START_SMOOTH_EPOCH = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                               kernel_size=kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class BidirectionalConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, spatial_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.spatial_size = spatial_size
        self.forward_cell = ConvLSTMCell(in_channels, hidden_channels)
        self.backward_cell = ConvLSTMCell(in_channels, hidden_channels)

    def _run_direction(self, cell, seq):
        b, t, c, h, w = seq.shape
        h_t = torch.zeros(b, self.hidden_channels, h, w, device=seq.device, dtype=seq.dtype)
        c_t = torch.zeros(b, self.hidden_channels, h, w, device=seq.device, dtype=seq.dtype)
        outputs = []
        for step in range(t):
            h_t, c_t = cell(seq[:, step], h_t, c_t)
            outputs.append(h_t)
        return torch.stack(outputs, dim=1)

    def forward(self, seq):
        fwd_out = self._run_direction(self.forward_cell, seq)
        bwd_out = self._run_direction(self.backward_cell, torch.flip(seq, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        return torch.cat([fwd_out, bwd_out], dim=2)


class SpatialTemporalAttentionPool(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(channels, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        pooled_per_frame = x.mean(dim=[3, 4])
        weights = torch.softmax(self.attn(pooled_per_frame), dim=1)
        return torch.sum(pooled_per_frame * weights, dim=1)


class ConvLSTMHead(nn.Module):
    def __init__(self, cnn_channels=CNN_CHANNELS, non_spatial_dim=NON_SPATIAL_DIM,
                 hidden_channels=CONVLSTM_HIDDEN, global_embed_dim=GLOBAL_EMBED_DIM):
        super().__init__()
        self.hidden_channels = hidden_channels
        in_channels = cnn_channels + non_spatial_dim

        self.convlstm = BidirectionalConvLSTM(in_channels, hidden_channels, (SPATIAL_H, SPATIAL_W))
        self.temporal_pool = SpatialTemporalAttentionPool(hidden_channels * 2)
        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, global_embed_dim), nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + global_embed_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, spatial_cnn, non_spatial, global_delta, fds=None, ef=None, epoch=None):
        b, t, c, h, w = spatial_cnn.shape
        non_spatial_broadcast = non_spatial.view(b, t, -1, 1, 1).expand(-1, -1, -1, h, w)
        seq = torch.cat([spatial_cnn, non_spatial_broadcast], dim=2)

        convlstm_out = self.convlstm(seq)
        pooled = self.temporal_pool(convlstm_out)

        global_embed = self.global_delta_encoder(global_delta)
        final_feat = torch.cat([pooled, global_embed], dim=-1)

        if fds is not None and self.training and ef is not None:
            fds.collect(final_feat, ef)
            if epoch is not None:
                final_feat = fds.smooth(final_feat, ef, epoch)

        return self.regressor(final_feat).squeeze(1)


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
    d = torch.load(os.path.join(CACHE_DIR, f"{split}_spatial.pt"))
    return d["spatial_cnn"], d["non_spatial"], d["global_delta"], d["ef"], d["video_ids"]


def build_capped_sampler(ef_values, bin_edges, bin_weights, max_ratio):
    ef_values = np.asarray(ef_values)
    bin_idx = np.digitize(ef_values, bin_edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, len(bin_weights) - 1)
    w = bin_weights[bin_idx].copy()
    cap = w.mean() * max_ratio
    w = np.clip(w, None, cap)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)


def main():
    print("\n==== RANK 2: Attention-Boosted Bidirectional ConvLSTM ====")
    print(f"Device: {DEVICE}\n")

    train_sp, train_ns, train_gd, train_ef, _ = load_cache("train")
    val_sp, val_ns, val_gd, val_ef, _ = load_cache("val")
    print(f"Cached spatial maps -- train: {train_sp.shape}, val: {val_sp.shape}")

    bin_edges, bin_weights = compute_lds_weights(train_ef.numpy(), sigma=2.0)
    sampler = build_capped_sampler(train_ef.numpy(), bin_edges, bin_weights, SAMPLER_MAX_WEIGHT_RATIO)
    train_loader = DataLoader(TensorDataset(train_sp, train_ns, train_gd, train_ef),
                               batch_size=BATCH_SIZE, sampler=sampler)

    head = ConvLSTMHead().to(DEVICE)
    fds = FDS(feature_dim=CONVLSTM_HIDDEN * 2 + GLOBAL_EMBED_DIM, num_bins=FDS_NUM_BINS, sigma=FDS_SIGMA,
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
        print("No resume checkpoint -- fresh start. ConvLSTM is architecturally new (no matching")
        print("shapes in v2's checkpoint), so it trains from random init -- no warm start available.")

    if start_epoch > EPOCHS:
        print("Already past configured EPOCHS.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        head.train()
        running_loss, n = 0.0, 0
        for sp, ns, gd, ef in train_loader:
            sp, ns, gd, ef = sp.to(DEVICE), ns.to(DEVICE), gd.to(DEVICE), ef.to(DEVICE)
            preds = head(sp, ns, gd, fds=fds, ef=ef, epoch=epoch)
            loss = criterion(preds, ef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * sp.size(0)
            n += sp.size(0)
        fds.update_last_epoch_stats(epoch)

        head.eval()
        val_preds = []
        with torch.no_grad():
            val_loader = DataLoader(TensorDataset(val_sp, val_ns, val_gd), batch_size=BATCH_SIZE, shuffle=False)
            for sp_b, ns_b, gd_b in val_loader:
                pred_b = head(sp_b.to(DEVICE), ns_b.to(DEVICE), gd_b.to(DEVICE))
                val_preds.append(pred_b.cpu())
        val_pred = torch.cat(val_preds, dim=0).numpy()
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
    test_sp, test_ns, test_gd, test_ef, test_ids = load_cache("test")
    best = torch.load(BEST_CKPT, map_location=DEVICE)
    head.load_state_dict(best["head_state_dict"])
    head.eval()

    test_preds = []
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(test_sp, test_ns, test_gd), batch_size=BATCH_SIZE, shuffle=False)
        for sp_b, ns_b, gd_b in test_loader:
            pred_b = head(sp_b.to(DEVICE), ns_b.to(DEVICE), gd_b.to(DEVICE))
            test_preds.append(pred_b.cpu())
    test_pred = np.clip(torch.cat(test_preds, dim=0).numpy(), 0, 100)

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
