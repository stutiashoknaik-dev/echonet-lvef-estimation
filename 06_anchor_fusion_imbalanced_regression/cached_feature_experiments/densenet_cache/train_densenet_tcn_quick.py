
# ============================================================
# QUICK RUN: FROZEN DENSENET FEATURES + TCN
# ============================================================
# Clean architecture experiment.
#
# Input:
#   train_fused.pt / val_fused.pt / test_fused.pt
#   Shape: [N, 32, 1107]
#
# Architecture:
#   Frozen DenseNet-derived per-frame features
#       ↓
#   Linear projection
#       ↓
#   Temporal Convolutional Network (dilated Conv1D)
#       ↓
#   Temporal attention pooling
#       ↓
#   Global-delta encoder
#       ↓
#   EF regressor
#
# IMPORTANT:
#   - NO LDS
#   - NO FDS
#   - NO BMC
#   - NO protected loss
#   - NO resampling
#   - Test is loaded only for final evaluation.
# ============================================================

import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

CACHE_DIR = os.path.join(
    BASE_PATH,
    "outputs/stage2_fused_cache"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/densenet_tcn_quick"
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

TEST_PRED_CSV = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

TEST_METRICS_CSV = os.path.join(
    SAVE_DIR,
    "test_metrics.csv"
)

TEST_BUCKET_CSV = os.path.join(
    SAVE_DIR,
    "test_bucket_metrics.csv"
)


# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 32
INPUT_DIM = 1107

TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]

DROPOUT = 0.20

GLOBAL_EMBED_DIM = 16

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10

LR = 3e-4
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def load_cache(split):
    path = os.path.join(
        CACHE_DIR,
        f"{split}_fused.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cached feature file:\n{path}\n\n"
            "Expected train_fused.pt, val_fused.pt, test_fused.pt."
        )

    d = torch.load(
        path,
        map_location="cpu"
    )

    fused = d["fused"].float()
    global_delta = d["global_delta"].float()
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    if tuple(fused.shape[1:]) != (
        SEQ_LEN,
        INPUT_DIM
    ):
        raise ValueError(
            f"{split}: expected [N,{SEQ_LEN},{INPUT_DIM}], "
            f"got {tuple(fused.shape)}"
        )

    if global_delta.ndim != 2 or global_delta.shape[1] != 4:
        raise ValueError(
            f"{split}: expected global_delta [N,4], "
            f"got {tuple(global_delta.shape)}"
        )

    return (
        fused,
        global_delta,
        ef,
        video_ids
    )


# ============================================================
# TCN BLOCKS
# ============================================================

class Chomp1d(nn.Module):
    """
    Removes right-side padding so the temporal output keeps the
    original sequence length.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
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
            dilation=dilation
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
            dilation=dilation
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
                kernel_size=1
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

    def __init__(
        self,
        input_dim=INPUT_DIM,
        channels=TCN_CHANNELS,
        kernel_size=TCN_KERNEL_SIZE,
        dilations=TCN_DILATIONS,
        dropout=DROPOUT
    ):
        super().__init__()

        blocks = []

        in_channels = input_dim

        for dilation in dilations:
            blocks.append(
                TemporalBlock(
                    in_channels,
                    channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            in_channels = channels

        self.network = nn.Sequential(
            *blocks
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(
            1,
            2
        )  # [B, D, T]

        x = self.network(
            x
        )

        return x.transpose(
            1,
            2
        )  # [B, T, C]


# ============================================================
# ATTENTION POOL
# ============================================================

class TemporalAttentionPool(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, x):

        weights = torch.softmax(
            self.score(x),
            dim=1
        )

        return torch.sum(
            x * weights,
            dim=1
        )


# ============================================================
# MODEL
# ============================================================

class DenseNetTCNEF(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(
                INPUT_DIM,
                TCN_CHANNELS
            )
        )

        self.tcn = TCNEncoder(
            input_dim=TCN_CHANNELS,
            channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            dropout=DROPOUT
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                TCN_CHANNELS
            )
        )

        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(
                4,
                GLOBAL_EMBED_DIM
            ),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                TCN_CHANNELS +
                GLOBAL_EMBED_DIM,
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

    def forward(
        self,
        fused,
        global_delta
    ):

        x = self.input_projection(
            fused
        )

        x = self.tcn(
            x
        )

        pooled = self.temporal_pool(
            x
        )

        global_embed = (
            self.global_delta_encoder(
                global_delta
            )
        )

        combined = torch.cat(
            [
                pooled,
                global_embed
            ],
            dim=-1
        )

        return self.regressor(
            combined
        ).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):
    y_true = np.asarray(
        y_true,
        dtype=np.float32
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float32
    )

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        corr = pearsonr(
            y_true,
            y_pred
        )[0]
    else:
        corr = float("nan")

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
        "Pearson_r": float(
            corr
        ),
        "Bias": float(
            np.mean(
                y_pred -
                y_true
            )
        )
    }


def bucket_metrics(
    y_true,
    y_pred
):

    df = pd.DataFrame({
        "gt": y_true,
        "pred": y_pred
    })

    df["abs_error"] = (
        df["pred"] -
        df["gt"]
    ).abs()

    bins = [
        0,
        30,
        40,
        50,
        60,
        70,
        100
    ]

    labels = [
        "<30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70+"
    ]

    df["bucket"] = pd.cut(
        df["gt"],
        bins=bins,
        labels=labels
    )

    counts = (
        df.groupby(
            "bucket",
            observed=True
        )
        .size()
    )

    maes = (
        df.groupby(
            "bucket",
            observed=True
        )["abs_error"]
        .mean()
    )

    return pd.DataFrame({
        "bucket": counts.index,
        "n": counts.values,
        "MAE": maes.values
    })


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_loader(
    model,
    loader
):

    model.eval()

    preds = []

    for fused, gd, _ in loader:

        fused = fused.to(
            DEVICE,
            non_blocking=True
        )

        gd = gd.to(
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


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print(
        "\n============================================================"
    )

    print(
        " QUICK RUN: FROZEN DENSENET FEATURES + TCN"
    )

    print(
        "============================================================"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load train + validation.
    # Test is intentionally held back until the final stage.
    # --------------------------------------------------------

    (
        train_fused,
        train_gd,
        train_ef,
        train_ids
    ) = load_cache("train")

    (
        val_fused,
        val_gd,
        val_ef,
        val_ids
    ) = load_cache("val")

    print(
        "\nCached feature shapes:"
    )

    print(
        "Train:",
        train_fused.shape
    )

    print(
        "Val:",
        val_fused.shape
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            train_fused,
            train_gd,
            train_ef
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
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
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DenseNetTCNEF().to(
        DEVICE
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable TCN parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Clean regression objective.
    # No LDS/FDS/BMC/resampling.
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss(
        beta=1.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    start_epoch = 1
    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    if os.path.exists(
        LAST_CKPT
    ):

        print(
            f"\nResuming from {LAST_CKPT}"
        )

        ckpt = torch.load(
            LAST_CKPT,
            map_location=DEVICE
        )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler_state_dict"]
        )

        start_epoch = (
            ckpt["epoch"] + 1
        )

        best_val_mae = (
            ckpt["best_val_mae"]
        )

        patience_counter = (
            ckpt.get(
                "patience_counter",
                0
            )
        )

        if os.path.exists(
            HISTORY_CSV
        ):
            history = pd.read_csv(
                HISTORY_CSV
            ).to_dict(
                "records"
            )

    else:

        print(
            "\nFresh run."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        n = 0

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
                loss.item() *
                bs
            )

            n += bs

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_pred = predict_loader(
            model,
            val_loader
        )

        val_metrics = regression_metrics(
            val_ef.numpy(),
            val_pred
        )

        val_buckets = bucket_metrics(
            val_ef.numpy(),
            val_pred
        )

        scheduler.step(
            val_metrics["MAE"]
        )

        row = {
            "Epoch": epoch,
            "Train_Loss":
                running_loss / n,
            **val_metrics,
            "LR":
                optimizer.param_groups[0]["lr"]
        }

        for _, r in val_buckets.iterrows():

            row[
                f"MAE_{r['bucket']}"
            ] = float(
                r["MAE"]
            )

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_CSV,
            index=False
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Val MAE: "
            f"{val_metrics['MAE']:.4f} | "
            f"R2: "
            f"{val_metrics['R2']:.4f} | "
            f"Pearson: "
            f"{val_metrics['Pearson_r']:.4f} | "
            +
            " | ".join(
                f"{r['bucket']}:{r['MAE']:.2f}"
                for _, r in val_buckets.iterrows()
            )
        )

        if (
            val_metrics["MAE"] <
            best_val_mae
        ):

            best_val_mae = (
                val_metrics["MAE"]
            )

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "best_val_mae":
                        best_val_mae,
                    "model_config": {
                        "seq_len":
                            SEQ_LEN,
                        "input_dim":
                            INPUT_DIM,
                        "tcn_channels":
                            TCN_CHANNELS,
                        "kernel_size":
                            TCN_KERNEL_SIZE,
                        "dilations":
                            TCN_DILATIONS
                    }
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

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "epoch":
                    epoch,
                "best_val_mae":
                    best_val_mae,
                "patience_counter":
                    patience_counter
            },
            LAST_CKPT
        )

        if (
            patience_counter >=
            PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        " FINAL TEST EVALUATION"
    )

    print(
        "============================================================"
    )

    (
        test_fused,
        test_gd,
        test_ef,
        test_ids
    ) = load_cache("test")

    test_loader = DataLoader(
        TensorDataset(
            test_fused,
            test_gd,
            test_ef
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    best = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        best["model_state_dict"]
    )

    test_pred = predict_loader(
        model,
        test_loader
    )

    test_pred = np.clip(
        test_pred,
        0,
        100
    )

    y_test = test_ef.numpy()

    test_metrics = regression_metrics(
        y_test,
        test_pred
    )

    test_buckets = bucket_metrics(
        y_test,
        test_pred
    )

    print(
        "\nDenseNet + TCN TEST metrics:"
    )

    print(
        test_metrics
    )

    print(
        "\nTEST bucket breakdown:"
    )

    print(
        test_buckets.to_string(
            index=False
        )
    )

    pd.DataFrame(
        {
            "Video": test_ids,
            "GT_EF": y_test,
            "Pred_EF": test_pred
        }
    ).to_csv(
        TEST_PRED_CSV,
        index=False
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        TEST_METRICS_CSV,
        index=False
    )

    test_buckets.to_csv(
        TEST_BUCKET_CSV,
        index=False
    )

    print(
        f"\nSaved results to:"
        f"\n{SAVE_DIR}"
    )


if __name__ == "__main__":
    main()
