
# ============================================================
# QUICK SCREEN: 3D CNN CACHE + TCN
# ============================================================
# Same cached 3D-CNN representation:
#   [N, 8, 339]
#
# ONLY the temporal head changes:
#   3D-CNN cache -> projection -> TCN -> attention pooling
#                 -> global-delta encoder -> EF regressor
#
# No LDS / FDS / BMC / resampling.
# Test is used only for final evaluation.
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

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE = os.path.join(
    BASE,
    "outputs",
    "stage2_3dcnn_cache"
)
SAVE = os.path.join(
    BASE,
    "outputs",
    "3dcnn_tcn_quick"
)

os.makedirs(SAVE, exist_ok=True)

BEST_CKPT = os.path.join(
    SAVE,
    "best_model.pth"
)

HISTORY_CSV = os.path.join(
    SAVE,
    "training_history.csv"
)

TEST_PRED_CSV = os.path.join(
    SAVE,
    "test_predictions.csv"
)

TEST_METRICS_CSV = os.path.join(
    SAVE,
    "test_metrics.csv"
)

TEST_BUCKET_CSV = os.path.join(
    SAVE,
    "test_bucket_metrics.csv"
)

# Same TCN settings as your successful DenseNet+TCN experiment.
TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]
DROPOUT = 0.20

GLOBAL_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

BATCH_SIZE = 64
EPOCHS = 70
PATIENCE = 10

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
        CACHE,
        f"{split}_fused.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache: {path}"
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
            f"{split}: fused must be [N,T,D], "
            f"got {tuple(fused.shape)}"
        )

    if global_delta.shape[1] != 4:
        raise ValueError(
            f"{split}: expected global_delta [N,4], "
            f"got {tuple(global_delta.shape)}"
        )

    return fused, global_delta, ef, video_ids


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    # Exact naming/structure style used in the successful
    # DenseNet+TCN experiment.
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

        self.chomp1 = Chomp1d(
            padding
        )

        self.norm1 = nn.BatchNorm1d(
            out_channels
        )

        self.drop1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp2 = Chomp1d(
            padding
        )

        self.norm2 = nn.BatchNorm1d(
            out_channels
        )

        self.drop2 = nn.Dropout(
            dropout
        )

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.drop2(out)

        res = self.residual(x)

        return self.activation(
            out + res
        )


class TCNEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        blocks = []
        in_channels = input_dim

        for dilation in TCN_DILATIONS:
            blocks.append(
                TemporalBlock(
                    in_channels,
                    TCN_CHANNELS,
                    TCN_KERNEL_SIZE,
                    dilation,
                    DROPOUT,
                )
            )
            in_channels = TCN_CHANNELS

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

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):
        weights = torch.softmax(
            self.score(x),
            dim=1,
        )

        return (
            x * weights
        ).sum(dim=1)


class R3DTCNEF(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                PROJ_DIM,
            ),
        )

        self.tcn = TCNEncoder(
            PROJ_DIM
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                TCN_CHANNELS,
                ATTN_HIDDEN,
            )
        )

        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(
                4,
                GLOBAL_DIM,
            ),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                TCN_CHANNELS + GLOBAL_DIM,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(
                256,
                64,
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, fused, global_delta):
        x = self.input_projection(
            fused
        )

        x = self.tcn(x)

        x = self.temporal_pool(
            x
        )

        g = self.global_delta_encoder(
            global_delta
        )

        return self.regressor(
            torch.cat(
                [x, g],
                dim=1,
            )
        ).squeeze(1)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float32
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float32
    )

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
            np.mean(
                y_pred - y_true
            )
        ),
    }


def bucket_metrics(y_true, y_pred):
    df = pd.DataFrame(
        {
            "GT_EF": y_true,
            "Pred_EF": y_pred,
        }
    )

    df["Abs_Error"] = (
        df["Pred_EF"] -
        df["GT_EF"]
    ).abs()

    df["Bucket"] = pd.cut(
        df["GT_EF"],
        bins=[
            0, 30, 40, 50, 60, 70, 100
        ],
        labels=[
            "<30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            "70+",
        ],
    )

    return (
        df.groupby(
            "Bucket",
            observed=True
        )
        .agg(
            n=("Abs_Error", "size"),
            MAE=("Abs_Error", "mean"),
        )
        .reset_index()
    )


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []

    for batch in loader:
        fused = batch[0].to(
            DEVICE,
            non_blocking=True
        )
        gd = batch[1].to(
            DEVICE,
            non_blocking=True
        )

        preds.append(
            model(
                fused,
                gd
            ).cpu()
        )

    return torch.cat(
        preds,
        dim=0
    ).numpy()


def main():

    seed_all(SEED)

    print(
        "\n============================================================"
    )
    print(
        " QUICK SCREEN: 3D CNN CACHE + TCN"
    )
    print(
        "============================================================"
    )

    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_fused, train_gd, train_ef, _ = (
        load_cache("train")
    )

    val_fused, val_gd, val_ef, _ = (
        load_cache("val")
    )

    print(
        "\nCache shapes:"
    )
    print(
        "Train:",
        train_fused.shape
    )
    print(
        "Val:",
        val_fused.shape
    )

    input_dim = train_fused.shape[-1]
    seq_len = train_fused.shape[1]

    print(
        "Input dim:",
        input_dim
    )
    print(
        "Temporal tokens:",
        seq_len
    )

    train_loader = DataLoader(
        TensorDataset(
            train_fused,
            train_gd,
            train_ef
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        TensorDataset(
            val_fused,
            val_gd,
            val_ef
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    model = R3DTCNEF(
        input_dim
    ).to(DEVICE)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable TCN/head parameters:",
        f"{trainable:,}"
    )

    criterion = nn.SmoothL1Loss(
        beta=1.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
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

        for fused, gd, ef in train_loader:

            fused = fused.to(
                DEVICE,
                non_blocking=True
            )
            gd = gd.to(
                DEVICE,
                non_blocking=True
            )
            ef = ef.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(
                fused,
                gd
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

        val_pred = predict_loader(
            model,
            val_loader
        )

        vm = regression_metrics(
            val_ef.numpy(),
            val_pred
        )

        vb = bucket_metrics(
            val_ef.numpy(),
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

        for _, r in vb.iterrows():
            row[
                f"MAE_{r['Bucket']}"
            ] = float(
                r["MAE"]
            )

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
                        input_dim,
                    "seq_len":
                        seq_len,
                    "tcn_channels":
                        TCN_CHANNELS,
                    "kernel_size":
                        TCN_KERNEL_SIZE,
                    "dilations":
                        TCN_DILATIONS,
                },
                BEST_CKPT
            )

            print(
                f"  New best! "
                f"Val MAE = "
                f"{best_val_mae:.4f}"
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
        "TRAINING COMPLETE"
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
