
# ============================================================
# FINAL TEST: 3D CNN CACHE + TCN
# ============================================================
# EXACT SAME ARCHITECTURE / STATE-DICT NAMES AS TRAIN SCRIPT.
#
# Test set is used only for final evaluation.
# ============================================================

import os
import math
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

BEST_CKPT = os.path.join(
    SAVE,
    "best_model.pth"
)

BATCH_SIZE = 64

TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]
DROPOUT = 0.20

GLOBAL_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    return (
        d["fused"].float(),
        d["global_delta"].float(),
        d["ef"].float(),
        d["video_ids"]
    )


class Chomp1d(nn.Module):

    def __init__(
        self,
        chomp_size
    ):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self,x):
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
                1
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.GELU()

    def forward(self,x):

        z = self.conv1(x)
        z = self.chomp1(z)
        z = self.norm1(z)
        z = self.activation(z)
        z = self.drop1(z)

        z = self.conv2(z)
        z = self.chomp2(z)
        z = self.norm2(z)
        z = self.activation(z)
        z = self.drop2(z)

        return self.activation(
            z + self.residual(x)
        )


class TCNEncoder(nn.Module):

    def __init__(
        self,
        input_dim
    ):
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
                    DROPOUT
                )
            )
            in_channels = TCN_CHANNELS

        self.network = nn.Sequential(
            *blocks
        )

    def forward(self,x):
        x = x.transpose(1,2)
        x = self.network(x)
        return x.transpose(1,2)


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

    def forward(self,x):

        w = torch.softmax(
            self.score(x),
            dim=1
        )

        return (
            x * w
        ).sum(dim=1)


class R3DTCNEF(nn.Module):

    def __init__(
        self,
        input_dim
    ):
        super().__init__()

        # EXACT SAME names as training script.
        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                PROJ_DIM
            )
        )

        self.tcn = TCNEncoder(
            PROJ_DIM
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                TCN_CHANNELS,
                ATTN_HIDDEN
            )
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
                TCN_CHANNELS +
                GLOBAL_DIM,
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
        x = self.tcn(x)
        x = self.temporal_pool(x)
        g = self.global_delta_encoder(
            global_delta
        )

        return self.regressor(
            torch.cat(
                [x,g],
                dim=1
            )
        ).squeeze(1)


def regression_metrics(
    y_true,
    y_pred
):
    corr = (
        pearsonr(
            y_true,
            y_pred
        )[0]
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
                y_pred -
                y_true
            )
        )
    }


def bucket_metrics(
    y_true,
    y_pred
):

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
            0, 30, 40, 50, 60, 70, 100
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
            n=("Abs_Error","size"),
            MAE=("Abs_Error","mean")
        )
        .reset_index()
    )


@torch.no_grad()
def main():

    print(
        "\n============================================================"
    )
    print(
        " FINAL TEST: 3D CNN CACHE + TCN"
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

    if not os.path.exists(
        BEST_CKPT
    ):
        raise FileNotFoundError(
            f"Missing checkpoint:\n{BEST_CKPT}\n"
            "Run the TCN training first."
        )

    (
        test_fused,
        test_gd,
        test_ef,
        test_ids
    ) = load_cache(
        "test"
    )

    print(
        "\nTEST cache:"
    )
    print(
        "  fused:",
        test_fused.shape
    )
    print(
        "  global_delta:",
        test_gd.shape
    )
    print(
        "  EF:",
        test_ef.shape
    )
    print(
        "  videos:",
        len(test_ids)
    )

    model = R3DTCNEF(
        test_fused.shape[-1]
    ).to(DEVICE)

    checkpoint = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    model.eval()

    print(
        "\nTCN checkpoint loaded successfully."
    )

    print(
        "Best epoch:",
        checkpoint.get(
            "epoch",
            "?"
        )
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
            test_fused,
            test_gd
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    predictions = []

    for fused, gd in loader:

        predictions.append(
            model(
                fused.to(
                    DEVICE,
                    non_blocking=True
                ),
                gd.to(
                    DEVICE,
                    non_blocking=True
                )
            ).cpu()
        )

    pred = np.clip(
        torch.cat(
            predictions,
            dim=0
        ).numpy(),
        0,
        100
    )

    y = test_ef.numpy()

    tm = regression_metrics(
        y,
        pred
    )

    tb = bucket_metrics(
        y,
        pred
    )

    print(
        "\n============================================================"
    )
    print(
        "3D CNN + TCN TEST RESULTS"
    )
    print(
        "============================================================"
    )

    print(
        tm
    )

    print(
        "\nBucket breakdown:"
    )

    print(
        tb.to_string(
            index=False
        )
    )

    pd.DataFrame(
        {
            "Video": test_ids,
            "GT_EF": y,
            "Pred_EF": pred
        }
    ).to_csv(
        os.path.join(
            SAVE,
            "test_predictions.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [tm]
    ).to_csv(
        os.path.join(
            SAVE,
            "test_metrics.csv"
        ),
        index=False
    )

    tb.to_csv(
        os.path.join(
            SAVE,
            "test_bucket_metrics.csv"
        ),
        index=False
    )

    print(
        "\nSaved to:",
        SAVE
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
