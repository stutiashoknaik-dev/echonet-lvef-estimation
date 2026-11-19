
# ============================================================
# FINAL TEST: 3D CNN CACHE + TEMPORAL ATTENTION / MLP
# ============================================================
# EXACTLY MATCHES train_3dcnn_attention_quick_fixed.py
#
# Model:
#   frozen 3D-CNN cached feature sequence
#        -> input_projection
#        -> temporal_pool
#        -> global_delta_encoder
#        -> regressor
#
# TEST IS USED ONLY FOR FINAL EVALUATION.
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE = (
    "/content/drive/MyDrive/EchoNet-Dynamic"
)

CACHE = os.path.join(
    BASE,
    "outputs",
    "stage2_3dcnn_cache"
)

SAVE = os.path.join(
    BASE,
    "outputs",
    "3dcnn_attention_quick"
)

BEST_CKPT = os.path.join(
    SAVE,
    "best_model.pth"
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


# ============================================================
# CONFIG -- MUST MATCH TRAINING
# ============================================================

BATCH_SIZE = 64

GLOBAL_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CACHE
# ============================================================

def load_cache(
    split
):

    path = os.path.join(
        CACHE,
        f"{split}_fused.pt"
    )

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"Missing cache:\n{path}"
        )

    d = torch.load(
        path,
        map_location="cpu"
    )

    fused = d["fused"].float()
    global_delta = (
        d["global_delta"]
        .float()
    )
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    return (
        fused,
        global_delta,
        ef,
        video_ids
    )


# ============================================================
# EXACT TRAINING CLASS NAMES
# ============================================================

class TemporalAttentionPool(
    nn.Module
):

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

    def forward(
        self,
        x
    ):

        scores = self.score(
            x
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        return (
            x * weights
        ).sum(
            dim=1
        )


class R3DAttentionMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim
    ):
        super().__init__()

        # EXACT names used by training script.
        self.input_projection = (
            nn.Sequential(
                nn.LayerNorm(
                    input_dim
                ),
                nn.Linear(
                    input_dim,
                    PROJ_DIM
                )
            )
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                PROJ_DIM,
                ATTN_HIDDEN
            )
        )

        self.global_delta_encoder = (
            nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(
                    4,
                    GLOBAL_DIM
                ),
                nn.ReLU(inplace=True)
            )
        )

        self.regressor = (
            nn.Sequential(
                nn.Linear(
                    PROJ_DIM +
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
        )

    def forward(
        self,
        fused,
        global_delta
    ):

        x = self.input_projection(
            fused
        )

        x = self.temporal_pool(
            x
        )

        g = self.global_delta_encoder(
            global_delta
        )

        combined = torch.cat(
            [
                x,
                g
            ],
            dim=1
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


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader
):

    model.eval()

    out = []

    for batch in loader:

        fused = batch[0].to(
            DEVICE,
            non_blocking=True
        )

        global_delta = batch[1].to(
            DEVICE,
            non_blocking=True
        )

        out.append(
            model(
                fused,
                global_delta
            ).cpu()
        )

    return torch.cat(
        out,
        dim=0
    ).numpy()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n============================================================"
    )

    print(
        " FINAL TEST: 3D CNN CACHE + ATTENTION"
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
            f"Best checkpoint not found:\n"
            f"{BEST_CKPT}\n"
            "Run the training script first."
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

    input_dim = (
        test_fused.shape[-1]
    )

    seq_len = (
        test_fused.shape[1]
    )

    print(
        "  input_dim:",
        input_dim
    )

    print(
        "  temporal_tokens:",
        seq_len
    )

    # --------------------------------------------------------
    # Model MUST match training exactly.
    # --------------------------------------------------------

    model = (
        R3DAttentionMLP(
            input_dim
        )
        .to(DEVICE)
    )

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
        "\nCheckpoint loaded successfully."
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

    test_pred = predict(
        model,
        loader
    )

    test_pred = np.clip(
        test_pred,
        0,
        100
    )

    y_test = (
        test_ef.numpy()
    )

    test_metrics = regression_metrics(
        y_test,
        test_pred
    )

    test_buckets = bucket_metrics(
        y_test,
        test_pred
    )

    print(
        "\n============================================================"
    )

    print(
        "3D CNN + ATTENTION TEST RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        test_metrics
    )

    print(
        "\nBucket breakdown:"
    )

    print(
        test_buckets.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
        "\nSaved:"
    )

    print(
        TEST_PRED_CSV
    )

    print(
        TEST_METRICS_CSV
    )

    print(
        TEST_BUCKET_CSV
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
