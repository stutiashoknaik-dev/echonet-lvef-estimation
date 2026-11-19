
# ============================================================
# QUICK SCREEN: DENSENET CACHED FEATURES + DUAL-SCALE TCN
# ============================================================
# Controlled experiment against the established DenseNet+TCN.
#
# SAME:
#   train/val/test caches
#   32 x 1107 fused representation
#   global_delta
#   SmoothL1 regression loss
#   optimizer family
#   early stopping
#
# ONLY MAIN ARCHITECTURAL CHANGE:
#   instead of ONE TCN pathway,
#   use TWO temporal pathways:
#
#     LOCAL branch  : dilations [1,2]
#     LONG branch   : dilations [4,8]
#
#   Each branch -> temporal attention
#   Then gated fusion -> global_delta -> EF
#
# NO LDS / FDS / BMC.
# TEST is not loaded in this training script.
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "stage2_fused_cache"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "densenet_dual_tcn_quick"
)

os.makedirs(SAVE_DIR, exist_ok=True)

BEST_CKPT = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

LAST_CKPT = os.path.join(
    SAVE_DIR,
    "last_checkpoint.pth"
)

HISTORY_CSV = os.path.join(
    SAVE_DIR,
    "training_history.csv"
)

# Established DenseNet cache configuration.
SEQ_LEN = 32
INPUT_DIM = 1107

# Projection.
PROJ_DIM = 256

# Two temporal scales.
LOCAL_DILATIONS = [1, 2]
LONG_DILATIONS = [4, 8]

TCN_CHANNELS = 256
KERNEL_SIZE = 3
DROPOUT = 0.20

ATTN_HIDDEN = 128
GLOBAL_DIM = 16

BATCH_SIZE = 64
EPOCHS = 70
PATIENCE = 12

LR = 3e-4
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(split):
    path = os.path.join(
        CACHE_DIR,
        f"{split}_fused.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache:\n{path}"
        )

    d = torch.load(
        path,
        map_location="cpu"
    )

    fused = d["fused"].float()
    global_delta = d["global_delta"].float()
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    if fused.ndim != 3:
        raise ValueError(
            f"Expected fused [N,T,D], got {tuple(fused.shape)}"
        )

    if fused.shape[1] != SEQ_LEN:
        raise ValueError(
            f"Expected T={SEQ_LEN}, got {fused.shape[1]}"
        )

    if fused.shape[2] != INPUT_DIM:
        raise ValueError(
            f"Expected D={INPUT_DIM}, got {fused.shape[2]}"
        )

    if global_delta.shape[1] != 4:
        raise ValueError(
            f"Expected global_delta [N,4], got {tuple(global_delta.shape)}"
        )

    return fused, global_delta, ef, video_ids


class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        if self.size == 0:
            return x
        return x[:, :, :-self.size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.GELU()

    def forward(self, x):
        z = self.conv1(x)
        z = self.chomp1(z)
        z = self.norm1(z)
        z = self.act(z)
        z = self.drop1(z)

        z = self.conv2(z)
        z = self.chomp2(z)
        z = self.norm2(z)
        z = self.act(z)
        z = self.drop2(z)

        return self.act(
            z + self.residual(x)
        )


class TCNBranch(nn.Module):
    def __init__(self, input_dim, dilations):
        super().__init__()

        blocks = []
        in_ch = input_dim

        for dilation in dilations:
            blocks.append(
                TemporalBlock(
                    in_ch,
                    TCN_CHANNELS,
                    KERNEL_SIZE,
                    dilation,
                    DROPOUT,
                )
            )
            in_ch = TCN_CHANNELS

        self.network = nn.Sequential(
            *blocks
        )

    def forward(self, x):
        # [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)
        x = self.network(x)
        # [B,D,T] -> [B,T,D]
        return x.transpose(1, 2)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        w = torch.softmax(
            self.score(x),
            dim=1
        )
        return (x * w).sum(dim=1)


class GatedDualScaleFusion(nn.Module):
    """
    Learned gate:
        gate = sigmoid(W[local,long])
        fused = gate*local + (1-gate)*long

    This lets the model choose whether each sample/time-derived
    representation should rely more on short- or long-range
    temporal structure.
    """

    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                dim * 2,
                dim
            ),
            nn.GELU(),
            nn.Linear(
                dim,
                dim
            ),
            nn.Sigmoid()
        )

    def forward(self, local, long):
        gate = self.gate(
            torch.cat(
                [local, long],
                dim=-1
            )
        )

        return (
            gate * local
            + (1.0 - gate) * long
        )


class DenseNetDualTCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                PROJ_DIM
            )
        )

        # Local temporal pathway.
        self.local_tcn = TCNBranch(
            PROJ_DIM,
            LOCAL_DILATIONS
        )

        # Long-range temporal pathway.
        self.long_tcn = TCNBranch(
            PROJ_DIM,
            LONG_DILATIONS
        )

        self.local_pool = TemporalAttentionPool(
            TCN_CHANNELS,
            ATTN_HIDDEN
        )

        self.long_pool = TemporalAttentionPool(
            TCN_CHANNELS,
            ATTN_HIDDEN
        )

        self.temporal_fusion = GatedDualScaleFusion(
            TCN_CHANNELS
        )

        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(
                4,
                GLOBAL_DIM
            ),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                TCN_CHANNELS + GLOBAL_DIM,
                256
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(
                256,
                64
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, fused, global_delta):
        x = self.input_projection(fused)

        local_seq = self.local_tcn(x)
        long_seq = self.long_tcn(x)

        local_feat = self.local_pool(local_seq)
        long_feat = self.long_pool(long_seq)

        temporal_feat = self.temporal_fusion(
            local_feat,
            long_feat
        )

        global_feat = self.global_delta_encoder(
            global_delta
        )

        return self.regressor(
            torch.cat(
                [
                    temporal_feat,
                    global_feat
                ],
                dim=1
            )
        ).squeeze(1)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    corr = (
        pearsonr(y_true, y_pred)[0]
        if (
            np.std(y_true) > 0
            and np.std(y_pred) > 0
        )
        else np.nan
    )

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
        "Pearson_r": float(corr),
        "Bias": float(
            np.mean(y_pred - y_true)
        ),
    }


def bucket_metrics(y_true, y_pred):
    df = pd.DataFrame(
        {
            "GT_EF": y_true,
            "Pred_EF": y_pred
        }
    )

    df["Abs_Error"] = (
        df["Pred_EF"] -
        df["GT_EF"]
    ).abs()

    df["Bucket"] = pd.cut(
        df["GT_EF"],
        bins=[
            0,
            30,
            40,
            50,
            60,
            70,
            100
        ],
        labels=[
            "<30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            "70+"
        ]
    )

    return (
        df.groupby(
            "Bucket",
            observed=True
        )
        .agg(
            n=("Abs_Error", "size"),
            MAE=("Abs_Error", "mean")
        )
        .reset_index()
    )


@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions = []

    for batch in loader:
        # Validation loader contains:
        # (fused, global_delta, ef)
        # We only need fused + global_delta for prediction.
        fused = batch[0]
        global_delta = batch[1]

        predictions.append(
            model(
                fused.to(
                    DEVICE,
                    non_blocking=True
                ),
                global_delta.to(
                    DEVICE,
                    non_blocking=True
                )
            ).cpu()
        )

    return torch.cat(
        predictions,
        dim=0
    ).numpy()


def main():
    seed_all(SEED)

    print(
        "\n============================================================"
    )
    print(
        " QUICK SCREEN: DENSENET + DUAL-SCALE TCN"
    )
    print(
        "============================================================"
    )

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_x, train_g, train_y, _ = load_cache("train")
    val_x, val_g, val_y, _ = load_cache("val")

    print(
        "Train:",
        train_x.shape
    )
    print(
        "Val:",
        val_x.shape
    )

    train_loader = DataLoader(
        TensorDataset(
            train_x,
            train_g,
            train_y
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        TensorDataset(
            val_x,
            val_g,
            val_y
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DenseNetDualTCN(
        train_x.shape[-1]
    ).to(DEVICE)

    print(
        "Trainable head parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    print(
        "Local dilations:",
        LOCAL_DILATIONS
    )
    print(
        "Long dilations:",
        LONG_DILATIONS
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss(
        beta=1.0
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        for fused, global_delta, ef in train_loader:

            fused = fused.to(
                DEVICE,
                non_blocking=True
            )

            global_delta = global_delta.to(
                DEVICE,
                non_blocking=True
            )

            ef = ef.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(
                fused,
                global_delta
            )

            loss = criterion(
                pred,
                ef
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            bs = fused.size(0)

            running_loss += (
                loss.item() * bs
            )

            sample_count += bs

        val_pred = predict(
            model,
            val_loader
        )

        vm = regression_metrics(
            val_y.numpy(),
            val_pred
        )

        vb = bucket_metrics(
            val_y.numpy(),
            val_pred
        )

        scheduler.step(
            vm["MAE"]
        )

        row = {
            "Epoch": epoch,
            "Train_Loss":
                running_loss /
                sample_count,
            **vm
        }

        history.append(row)

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_CSV,
            index=False
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Val MAE: {vm['MAE']:.4f} | "
            f"RMSE: {vm['RMSE']:.4f} | "
            f"R2: {vm['R2']:.4f} | "
            f"Pearson: {vm['Pearson_r']:.4f} | "
            +
            " | ".join(
                f"{r['Bucket']}:{r['MAE']:.2f}"
                for _, r in vb.iterrows()
            )
        )

        if vm["MAE"] < best_val_mae:

            best_val_mae = vm["MAE"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "best_val_mae":
                        best_val_mae,
                    "input_dim":
                        train_x.shape[-1],
                    "seq_len":
                        train_x.shape[1],
                    "local_dilations":
                        LOCAL_DILATIONS,
                    "long_dilations":
                        LONG_DILATIONS,
                },
                BEST_CKPT
            )

            print(
                f"  New best! "
                f"Val MAE = {best_val_mae:.4f}"
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                "Early stopping triggered."
            )
            break

    print(
        "\n============================================================"
    )
    print(
        "DUAL-SCALE TCN SCREEN COMPLETE"
    )
    print(
        "============================================================"
    )

    print(
        "Best Val MAE:",
        best_val_mae
    )

    print(
        "Checkpoint:",
        BEST_CKPT
    )


if __name__ == "__main__":
    main()
