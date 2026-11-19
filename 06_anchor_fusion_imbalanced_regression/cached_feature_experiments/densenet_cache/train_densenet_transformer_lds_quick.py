# ============================================================
# QUICK RUN: DENSENET FEATURES + TRANSFORMER + EXACT V2 LDS
# ============================================================
import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_fused_cache")
SAVE_DIR = os.path.join(BASE_PATH, "outputs/densenet_transformer_lds_quick")
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_CKPT = os.path.join(SAVE_DIR, "best_model.pth")
LAST_CKPT = os.path.join(SAVE_DIR, "last_checkpoint.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
TEST_PRED_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
TEST_METRICS_CSV = os.path.join(SAVE_DIR, "test_metrics.csv")
TEST_BUCKET_CSV = os.path.join(SAVE_DIR, "test_bucket_metrics.csv")

SEQ_LEN = 32
INPUT_DIM = 1107
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
FF_DIM = 512
DROPOUT = 0.20
GLOBAL_EMBED_DIM = 16

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Exact settings from the supplied lds_loss.py
LDS_NUM_BINS = 100
LDS_EF_RANGE = (0, 100)
LDS_SIGMA = 2.0
HUBER_DELTA = 1.0

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------- Exact LDS from supplied lds_loss.py ----------
def compute_lds_weights(ef_values, num_bins=100, ef_range=(0, 100), sigma=2.0):
    ef_values = np.asarray(ef_values, dtype=np.float64)
    hist, bin_edges = np.histogram(
        ef_values, bins=num_bins, range=ef_range
    )
    hist = hist.astype(np.float64)

    bin_width = (ef_range[1] - ef_range[0]) / num_bins
    kernel_radius = max(1, int(3 * sigma / bin_width))
    x = np.arange(-kernel_radius, kernel_radius + 1) * bin_width

    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    smoothed = np.convolve(hist, kernel, mode="same")
    smoothed = np.clip(smoothed, 1.0, None)

    weights = 1.0 / np.sqrt(smoothed)
    weights = weights / weights.mean()

    return bin_edges, weights

class LDSWeightedHuberLoss(nn.Module):
    def __init__(self, bin_edges, bin_weights, delta=1.0):
        super().__init__()
        self.register_buffer(
            "bin_edges_t",
            torch.tensor(bin_edges, dtype=torch.float32),
            persistent=False,
        )
        self.bin_edges = bin_edges
        self.bin_weights = bin_weights
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, targets):
        per_sample_loss = self.huber(preds, targets)

        ef_np = targets.detach().cpu().numpy()
        bin_idx = np.digitize(ef_np, self.bin_edges[1:-1])
        bin_idx = np.clip(bin_idx, 0, len(self.bin_weights) - 1)

        w = self.bin_weights[bin_idx]
        weights = torch.tensor(
            w, dtype=targets.dtype, device=targets.device
        )
        return (per_sample_loss * weights).mean()

def load_cache(split):
    path = os.path.join(CACHE_DIR, f"{split}_fused.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache: {path}")

    d = torch.load(path, map_location="cpu")
    fused = d["fused"].float()
    global_delta = d["global_delta"].float()
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    expected = (SEQ_LEN, INPUT_DIM)
    if tuple(fused.shape[1:]) != expected:
        raise ValueError(
            f"{split}: expected [N,{expected[0]},{expected[1]}], got {tuple(fused.shape)}"
        )
    if global_delta.shape[1] != 4:
        raise ValueError(f"{split}: expected global_delta [N,4], got {tuple(global_delta.shape)}")

    return fused, global_delta, ef, video_ids

class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)

class DenseNetTransformerLDS(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(INPUT_DIM, D_MODEL),
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, SEQ_LEN, D_MODEL)
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS
        )

        self.temporal_pool = TemporalAttentionPool(D_MODEL)

        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, GLOBAL_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(D_MODEL + GLOBAL_EMBED_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, fused, global_delta):
        x = self.input_projection(fused)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        pooled = self.temporal_pool(x)
        global_embed = self.global_delta_encoder(global_delta)
        combined = torch.cat([pooled, global_embed], dim=-1)
        return self.regressor(combined).squeeze(1)

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    corr = (
        pearsonr(y_true, y_pred)[0]
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(corr),
        "Bias": float(np.mean(y_pred - y_true)),
    }

def bucket_metrics(y_true, y_pred):
    df = pd.DataFrame({"gt": y_true, "pred": y_pred})
    df["abs_error"] = (df["pred"] - df["gt"]).abs()
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["bucket"] = pd.cut(df["gt"], bins=bins, labels=labels)
    counts = df.groupby("bucket", observed=True).size()
    maes = df.groupby("bucket", observed=True)["abs_error"].mean()
    return pd.DataFrame({
        "bucket": counts.index,
        "n": counts.values,
        "MAE": maes.values,
    })

@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []
    for fused, gd, _ in loader:
        fused = fused.to(DEVICE, non_blocking=True)
        gd = gd.to(DEVICE, non_blocking=True)
        preds.append(model(fused, gd).cpu())
    return torch.cat(preds, dim=0).numpy()

def main():
    seed_everything(SEED)

    print("\n============================================================")
    print(" QUICK RUN: DENSENET FEATURES + TRANSFORMER + EXACT LDS")
    print("============================================================")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Train + validation only at first; test is intentionally held back.
    train_fused, train_gd, train_ef, _ = load_cache("train")
    val_fused, val_gd, val_ef, _ = load_cache("val")

    print("\nCached feature shapes:")
    print("Train:", train_fused.shape)
    print("Val:", val_fused.shape)

    # Exact LDS: fitted from TRAIN EF labels only.
    bin_edges, bin_weights = compute_lds_weights(
        train_ef.numpy(),
        num_bins=LDS_NUM_BINS,
        ef_range=LDS_EF_RANGE,
        sigma=LDS_SIGMA,
    )

    print("\nLDS configuration:")
    print(f"  num_bins = {LDS_NUM_BINS}")
    print(f"  EF range = {LDS_EF_RANGE}")
    print(f"  sigma = {LDS_SIGMA}")
    print(f"  Huber delta = {HUBER_DELTA}")
    print(
        f"  weight range = [{bin_weights.min():.4f}, "
        f"{bin_weights.max():.4f}], mean={bin_weights.mean():.4f}"
    )

    train_loader = DataLoader(
        TensorDataset(train_fused, train_gd, train_ef),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(val_fused, val_gd, val_ef),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DenseNetTransformerLDS().to(DEVICE)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\nTrainable Transformer parameters: {trainable_params:,}")

    criterion = LDSWeightedHuberLoss(
        bin_edges,
        bin_weights,
        delta=HUBER_DELTA,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    start_epoch = 1
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    if os.path.exists(LAST_CKPT):
        print(f"\nResuming from {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(HISTORY_CSV):
            history = pd.read_csv(HISTORY_CSV).to_dict("records")
    else:
        print("\nFresh run.")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n = 0

        for fused, gd, ef in train_loader:
            fused = fused.to(DEVICE, non_blocking=True)
            gd = gd.to(DEVICE, non_blocking=True)
            ef = ef.to(DEVICE, non_blocking=True)

            pred = model(fused, gd)
            loss = criterion(pred, ef)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = fused.size(0)
            running_loss += loss.item() * bs
            n += bs

        val_pred = predict_loader(model, val_loader)
        val_metrics = regression_metrics(val_ef.numpy(), val_pred)
        val_buckets = bucket_metrics(val_ef.numpy(), val_pred)

        scheduler.step(val_metrics["MAE"])

        row = {
            "Epoch": epoch,
            "Train_LDS_Huber_Loss": running_loss / n,
            **val_metrics,
            "LR": optimizer.param_groups[0]["lr"],
        }
        for _, r in val_buckets.iterrows():
            row[f"MAE_{r['bucket']}"] = float(r["MAE"])

        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Val MAE: {val_metrics['MAE']:.4f} | "
            f"R2: {val_metrics['R2']:.4f} | "
            f"Pearson: {val_metrics['Pearson_r']:.4f} | "
            + " | ".join(
                f"{r['bucket']}:{r['MAE']:.2f}"
                for _, r in val_buckets.iterrows()
            )
        )

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_mae": best_val_mae,
                    "lds_config": {
                        "num_bins": LDS_NUM_BINS,
                        "ef_range": LDS_EF_RANGE,
                        "sigma": LDS_SIGMA,
                        "huber_delta": HUBER_DELTA,
                    },
                    "model_config": {
                        "seq_len": SEQ_LEN,
                        "input_dim": INPUT_DIM,
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "ff_dim": FF_DIM,
                    },
                },
                BEST_CKPT,
            )

            print(f"  New best! Val MAE = {best_val_mae:.4f}")
        else:
            patience_counter += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mae": best_val_mae,
                "patience_counter": patience_counter,
            },
            LAST_CKPT,
        )

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # Final test: load test only now.
    print("\n============================================================")
    print(" FINAL TEST EVALUATION")
    print("============================================================")

    test_fused, test_gd, test_ef, test_ids = load_cache("test")

    test_loader = DataLoader(
        TensorDataset(test_fused, test_gd, test_ef),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    best = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(best["model_state_dict"])

    test_pred = np.clip(
        predict_loader(model, test_loader),
        0,
        100,
    )

    y_test = test_ef.numpy()
    test_metrics = regression_metrics(y_test, test_pred)
    test_buckets = bucket_metrics(y_test, test_pred)

    print("\nDenseNet + Transformer + LDS TEST metrics:")
    print(test_metrics)

    print("\nTEST bucket breakdown:")
    print(test_buckets.to_string(index=False))

    pd.DataFrame({
        "Video": test_ids,
        "GT_EF": y_test,
        "Pred_EF": test_pred,
    }).to_csv(TEST_PRED_CSV, index=False)

    pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_CSV, index=False)
    test_buckets.to_csv(TEST_BUCKET_CSV, index=False)

    print(f"\nSaved results to: {SAVE_DIR}")
    print("\nLDS weights were fitted from TRAIN EF labels only.")
    print("TEST was used only for final evaluation.")

if __name__ == "__main__":
    main()
