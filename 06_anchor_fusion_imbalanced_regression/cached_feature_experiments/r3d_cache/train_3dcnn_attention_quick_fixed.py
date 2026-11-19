
# ============================================================
# QUICK SCREEN: 3D CNN CACHE + TEMPORAL ATTENTION / MLP
# ============================================================
# Uses the already-created 3D-CNN cache:
#   train_fused.pt
#   val_fused.pt
#   test_fused.pt
#
# The R3D-18 backbone is ALREADY FROZEN inside the cache.
# This script trains only:
#   fused 8x339 features -> projection
#                      -> temporal attention pooling
#                      -> global-delta branch
#                      -> EF regressor
#
# NO LDS / FDS / BMC / resampling.
#
# Train/val/test remain separated:
#   train -> optimize parameters
#   val   -> model selection / early stopping
#   test  -> final evaluation only
# ============================================================

import os
import math
import random

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

# IMPORTANT FIX:
# Create the output directory BEFORE any checkpoint save.
os.makedirs(
    SAVE,
    exist_ok=True
)

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


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64

EPOCHS = 50
PATIENCE = 10

LR = 3e-4
WEIGHT_DECAY = 1e-4

GLOBAL_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CACHE LOADING
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
            f"Missing cache file:\n{path}\n\n"
            "Run extract_3dcnn_fused_cache.py first."
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

    if fused.ndim != 3:
        raise ValueError(
            f"{split}: expected fused "
            f"[N,T,D], got {tuple(fused.shape)}"
        )

    if global_delta.ndim != 2:
        raise ValueError(
            f"{split}: expected global_delta [N,4], "
            f"got {tuple(global_delta.shape)}"
        )

    if (
        fused.shape[0]
        != global_delta.shape[0]
        or fused.shape[0]
        != ef.shape[0]
        or fused.shape[0]
        != len(video_ids)
    ):
        raise ValueError(
            f"{split}: sample counts do not match"
        )

    return (
        fused,
        global_delta,
        ef,
        video_ids
    )


# ============================================================
# TEMPORAL ATTENTION
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
        """
        x: [B,T,D]

        returns:
            pooled: [B,D]
        """

        scores = self.score(
            x
        )  # [B,T,1]

        weights = torch.softmax(
            scores,
            dim=1
        )

        return (
            x * weights
        ).sum(
            dim=1
        )


# ============================================================
# MODEL
# ============================================================

class R3DAttentionMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim
    ):
        super().__init__()

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
                y_pred - y_true
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
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_loader(
    model,
    loader
):

    model.eval()

    predictions = []

    for batch in loader:

        # Validation/test loaders contain:
        #   (fused, global_delta, EF)
        # or
        #   (fused, global_delta)
        #
        # Only first two are needed for prediction.
        fused = batch[0]
        global_delta = batch[1]

        fused = fused.to(
            DEVICE,
            non_blocking=True
        )

        global_delta = global_delta.to(
            DEVICE,
            non_blocking=True
        )

        pred = model(
            fused,
            global_delta
        )

        predictions.append(
            pred.cpu()
        )

    return torch.cat(
        predictions,
        dim=0
    ).numpy()


# ============================================================
# MAIN
# ============================================================

def main():

    seed_all(
        SEED
    )

    print(
        "\n============================================================"
    )

    print(
        " QUICK SCREEN: 3D CNN CACHE + TEMPORAL ATTENTION"
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

    print(
        "Cache:",
        CACHE
    )

    print(
        "Output:",
        SAVE
    )

    # --------------------------------------------------------
    # Load split-specific caches.
    # --------------------------------------------------------

    (
        train_fused,
        train_gd,
        train_ef,
        _
    ) = load_cache(
        "train"
    )

    (
        val_fused,
        val_gd,
        val_ef,
        _
    ) = load_cache(
        "val"
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

    input_dim = (
        train_fused.shape[-1]
    )

    seq_len = (
        train_fused.shape[1]
    )

    print(
        f"Input dimension: {input_dim}"
    )

    print(
        f"Temporal tokens: {seq_len}"
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
        pin_memory=(
            DEVICE.type == "cuda"
        )
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
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        R3DAttentionMLP(
            input_dim
        )
        .to(DEVICE)
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable head parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    criterion = (
        nn.SmoothL1Loss(
            beta=1.0
        )
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

    best_val_mae = float(
        "inf"
    )

    patience_counter = 0

    history = []

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

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
                loss.item() *
                bs
            )

            sample_count += bs

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

        history_row = {
            "Epoch": epoch,
            "Train_Loss":
                running_loss /
                sample_count,
            **val_metrics
        }

        for _, r in (
            val_buckets.iterrows()
        ):

            history_row[
                f"MAE_{r['Bucket']}"
            ] = float(
                r["MAE"]
            )

        history.append(
            history_row
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
                f"{r['Bucket']}:{r['MAE']:.2f}"
                for _, r
                in val_buckets.iterrows()
            )
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["MAE"] <
            best_val_mae
        ):

            best_val_mae = (
                val_metrics["MAE"]
            )

            patience_counter = 0

            # Directory already exists above.
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
                    "projection_dim":
                        PROJ_DIM,
                    "attention_hidden":
                        ATTN_HIDDEN,
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

        if (
            patience_counter >=
            PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final held-out TEST
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
    ) = load_cache(
        "test"
    )

    test_loader = DataLoader(
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

    if not os.path.exists(
        BEST_CKPT
    ):
        raise RuntimeError(
            "Best checkpoint was not created."
        )

    best = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        best[
            "model_state_dict"
        ],
        strict=True
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
        "\n3D CNN + Attention TEST:"
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

    # --------------------------------------------------------
    # Save test results
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
        "\nSaved results to:"
    )

    print(
        SAVE
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
