
# ============================================================
# TEST ONLY: FROZEN DENSENET FEATURES + TCN
# ============================================================
# Uses the exact same model definition and variable names as
# train_densenet_tcn_quick.py.
#
# NO training / tuning / validation selection occurs here.
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

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

CACHE_DIR = os.path.join(
    BASE_PATH,
    "outputs/stage2_fused_cache"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/densenet_tcn_quick"
)

BEST_CKPT = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

TEST_CACHE = os.path.join(
    CACHE_DIR,
    "test_fused.pt"
)

OUT_PRED = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

OUT_METRICS = os.path.join(
    SAVE_DIR,
    "test_metrics.csv"
)

OUT_BUCKETS = os.path.join(
    SAVE_DIR,
    "test_bucket_metrics.csv"
)

SEQ_LEN = 32
INPUT_DIM = 1107

TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]

DROPOUT = 0.20
GLOBAL_EMBED_DIM = 16
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# DATA / METRICS / PREDICTION
# ============================================================

def load_test_cache():

    if not os.path.exists(
        TEST_CACHE
    ):
        raise FileNotFoundError(
            f"Missing test cache:\n{TEST_CACHE}"
        )

    d = torch.load(
        TEST_CACHE,
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
            f"Expected [N,{SEQ_LEN},{INPUT_DIM}], "
            f"got {tuple(fused.shape)}"
        )

    return (
        fused,
        global_delta,
        ef,
        video_ids
    )


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

    corr = (
        pearsonr(
            y_true,
            y_pred
        )[0]
        if (
            len(y_true) > 1
            and np.std(y_true) > 0
            and np.std(y_pred) > 0
        )
        else float("nan")
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


def bucket_metrics(
    y_true,
    y_pred
):

    df = pd.DataFrame({
        "GT_EF": y_true,
        "Pred_EF": y_pred
    })

    df["Abs_Error"] = (
        df["Pred_EF"] -
        df["GT_EF"]
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

    df["Bucket"] = pd.cut(
        df["GT_EF"],
        bins=bins,
        labels=labels
    )

    counts = (
        df.groupby(
            "Bucket",
            observed=True
        ).size()
    )

    maes = (
        df.groupby(
            "Bucket",
            observed=True
        )["Abs_Error"].mean()
    )

    return pd.DataFrame({
        "bucket": counts.index,
        "n": counts.values,
        "MAE": maes.values
    })


@torch.no_grad()
def predict_loader(
    model,
    loader
):

    model.eval()
    preds = []

    for fused, gd in loader:

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


def main():

    print(
        "\n============================================================"
    )
    print(
        " FINAL TEST: DENSENET + TCN"
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

    (
        test_fused,
        test_gd,
        test_ef,
        test_ids
    ) = load_test_cache()

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
        "  EF labels:",
        test_ef.shape
    )

    print(
        "  videos:",
        len(test_ids)
    )

    if not os.path.exists(
        BEST_CKPT
    ):
        raise FileNotFoundError(
            f"Best TCN checkpoint not found:\n{BEST_CKPT}\n"
            "Run train_densenet_tcn_quick.py first."
        )

    model = DenseNetTCNEF().to(
        DEVICE
    )

    checkpoint = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    # Exact architecture/name match with training script.
    model.load_state_dict(
        state,
        strict=True
    )

    model.eval()

    print(
        "\nTCN checkpoint loaded successfully."
    )

    print(
        "TCN best epoch:",
        checkpoint.get(
            "epoch",
            "?"
        )
    )

    print(
        "TCN best Val MAE:",
        checkpoint.get(
            "best_val_mae",
            "?"
        )
    )

    test_loader = DataLoader(
        TensorDataset(
            test_fused,
            test_gd
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
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

    metrics = regression_metrics(
        y_test,
        test_pred
    )

    buckets = bucket_metrics(
        y_test,
        test_pred
    )

    print(
        "\n============================================================"
    )

    print(
        " TCN TEST RESULTS"
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
        buckets.to_string(
            index=False
        )
    )

    pd.DataFrame({
        "Video": test_ids,
        "GT_EF": y_test,
        "Pred_EF": test_pred
    }).to_csv(
        OUT_PRED,
        index=False
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        OUT_METRICS,
        index=False
    )

    buckets.to_csv(
        OUT_BUCKETS,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUT_PRED
    )

    print(
        OUT_METRICS
    )

    print(
        OUT_BUCKETS
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
