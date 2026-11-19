
# ============================================================
# FINAL TEST: DENSENET CACHED FEATURES + DUAL-SCALE TCN
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
EPOCHS = 50
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

    for fused, global_delta in loader:
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



@torch.no_grad()
def main():

    print(
        "\n============================================================"
    )
    print(
        " FINAL TEST: DENSENET + DUAL-SCALE TCN"
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

    test_x, test_g, test_y, test_ids = load_cache(
        "test"
    )

    print(
        "TEST cache:"
    )
    print(
        "  fused:",
        test_x.shape
    )
    print(
        "  global_delta:",
        test_g.shape
    )
    print(
        "  EF:",
        test_y.shape
    )
    print(
        "  videos:",
        len(test_ids)
    )

    if not os.path.exists(BEST_CKPT):
        raise FileNotFoundError(
            f"Missing checkpoint:\n{BEST_CKPT}"
        )

    model = DenseNetDualTCN(
        test_x.shape[-1]
    ).to(DEVICE)

    checkpoint = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print(
        "\nCheckpoint loaded successfully."
    )

    print(
        "Best epoch:",
        checkpoint.get("epoch", "?")
    )

    print(
        "Best Val MAE:",
        checkpoint.get(
            "best_val_mae",
            "?"
        )
    )

    loader = DataLoader(
        TensorDataset(
            test_x,
            test_g
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    pred = predict(
        model,
        loader
    )

    pred = np.clip(
        pred,
        0,
        100
    )

    y = test_y.numpy()

    metrics = regression_metrics(
        y,
        pred
    )

    bucket = bucket_metrics(
        y,
        pred
    )

    print(
        "\n============================================================"
    )
    print(
        "DENSENET + DUAL-SCALE TCN TEST RESULTS"
    )
    print(
        "============================================================"
    )

    print(
        metrics
    )

    print(
        "\nBucket breakdown:"
    )

    print(
        bucket.to_string(
            index=False
        )
    )

    pd.DataFrame(
        {
            "Video": test_ids,
            "GT_EF": y,
            "Pred_EF": pred,
        }
    ).to_csv(
        os.path.join(
            SAVE_DIR,
            "test_predictions.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        os.path.join(
            SAVE_DIR,
            "test_metrics.csv"
        ),
        index=False
    )

    bucket.to_csv(
        os.path.join(
            SAVE_DIR,
            "test_bucket_metrics.csv"
        ),
        index=False
    )

    print(
        "\nSaved to:",
        SAVE_DIR
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
