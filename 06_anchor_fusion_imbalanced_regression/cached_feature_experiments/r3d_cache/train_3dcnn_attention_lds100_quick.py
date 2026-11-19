
# ============================================================
# QUICK SCREEN: 3D CNN CACHE + TEMPORAL ATTENTION + LDS(100)
# ============================================================
# Same 3D-CNN cached representation and SAME attention model
# as the plain 3D-CNN + Attention experiment.
#
# ONLY CHANGE:
#   SmoothL1Loss -> LDS-weighted Huber
#
# LDS:
#   100 bins
#   EF range 0-100
#   sigma = 2.0
#   Huber delta = 1.0
#
# LDS weights are computed from TRAIN EF LABELS ONLY.
# Validation is used only for model selection.
# Test is used only for final evaluation.
#
# No FDS / BMC / resampling.
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
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

CACHE = os.path.join(
    BASE,
    "outputs",
    "stage2_3dcnn_cache"
)

SAVE = os.path.join(
    BASE,
    "outputs",
    "3dcnn_attention_lds100_quick"
)

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

# ---- SAME LDS STYLE AS YOUR v2 EXPERIMENT ----
LDS_NUM_BINS = 100
LDS_EF_MIN = 0.0
LDS_EF_MAX = 100.0
LDS_SIGMA = 2.0
HUBER_DELTA = 1.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CACHE LOADING
# ============================================================

def load_cache(split):

    path = os.path.join(
        CACHE,
        f"{split}_fused.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache:\n{path}\n"
            "Run extract_3dcnn_fused_cache.py first."
        )

    d = torch.load(
        path,
        map_location="cpu"
    )

    fused = d["fused"].float()
    global_delta = d["global_delta"].float()
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    return (
        fused,
        global_delta,
        ef,
        video_ids
    )


# ============================================================
# LDS WEIGHTS
# ============================================================

def compute_lds_weights(
    ef_values,
    num_bins=100,
    ef_min=0.0,
    ef_max=100.0,
    sigma=2.0
):
    """
    Fit LDS weights from TRAIN EF labels only.

    Histogram -> Gaussian smoothing -> inverse-sqrt density.

    The resulting weights are normalized to mean 1.
    """

    y = np.asarray(
        ef_values,
        dtype=np.float64
    )

    hist, edges = np.histogram(
        y,
        bins=num_bins,
        range=(ef_min, ef_max)
    )

    hist = hist.astype(
        np.float64
    )

    # EF range per bin.
    bin_width = (
        ef_max - ef_min
    ) / num_bins

    # Approximately +/- 3 sigma support.
    radius = max(
        1,
        int(
            np.ceil(
                3.0 *
                sigma /
                bin_width
            )
        )
    )

    x = (
        np.arange(
            -radius,
            radius + 1
        ) *
        bin_width
    )

    kernel = np.exp(
        -0.5 *
        (
            x /
            sigma
        ) ** 2
    )

    kernel /= kernel.sum()

    smoothed = np.convolve(
        hist,
        kernel,
        mode="same"
    )

    smoothed = np.clip(
        smoothed,
        1.0,
        None
    )

    weights = (
        1.0 /
        np.sqrt(
            smoothed
        )
    )

    weights /= weights.mean()

    return edges, weights


class LDSWeightedHuberLoss(
    nn.Module
):

    def __init__(
        self,
        bin_edges,
        bin_weights,
        delta=1.0
    ):
        super().__init__()

        # Registered as buffers so they move with the module.
        self.register_buffer(
            "bin_edges",
            torch.as_tensor(
                bin_edges,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "bin_weights",
            torch.as_tensor(
                bin_weights,
                dtype=torch.float32
            )
        )

        self.delta = delta

    def forward(
        self,
        pred,
        target
    ):
        abs_err = (
            pred -
            target
        ).abs()

        if self.delta > 0:

            huber = torch.where(
                abs_err <= self.delta,
                0.5 *
                abs_err.pow(2),
                self.delta *
                (
                    abs_err -
                    0.5 *
                    self.delta
                )
            )

        else:

            huber = abs_err

        # Map each target EF to its LDS bin.
        idx = torch.bucketize(
            target,
            self.bin_edges[1:-1]
        )

        idx = torch.clamp(
            idx,
            0,
            self.bin_weights.numel() - 1
        )

        weights = (
            self.bin_weights[idx]
        )

        return (
            huber *
            weights
        ).mean()


# ============================================================
# ATTENTION MODEL -- EXACT SAME ARCHITECTURE AS PLAIN RUN
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

    def forward(self, x):

        scores = self.score(
            x
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        return (
            x *
            weights
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

        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                PROJ_DIM
            )
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                PROJ_DIM,
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

        return self.regressor(
            torch.cat(
                [
                    x,
                    g
                ],
                dim=1
            )
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


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict_loader(
    model,
    loader
):

    model.eval()
    preds = []

    for batch in loader:

        fused = batch[0].to(
            DEVICE,
            non_blocking=True
        )

        global_delta = batch[1].to(
            DEVICE,
            non_blocking=True
        )

        preds.append(
            model(
                fused,
                global_delta
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

    seed_all(
        SEED
    )

    print(
        "\n============================================================"
    )

    print(
        " QUICK SCREEN: 3D CNN + ATTENTION + LDS(100)"
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
        "\nLDS configuration:"
    )

    print(
        f"  bins = {LDS_NUM_BINS}"
    )

    print(
        f"  EF range = "
        f"({LDS_EF_MIN}, {LDS_EF_MAX})"
    )

    print(
        f"  sigma = {LDS_SIGMA}"
    )

    print(
        f"  Huber delta = {HUBER_DELTA}"
    )

    # --------------------------------------------------------
    # TRAIN / VAL
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

    # --------------------------------------------------------
    # IMPORTANT:
    # LDS FIT FROM TRAIN ONLY
    # --------------------------------------------------------

    bin_edges, bin_weights = (
        compute_lds_weights(
            train_ef.numpy(),
            num_bins=LDS_NUM_BINS,
            ef_min=LDS_EF_MIN,
            ef_max=LDS_EF_MAX,
            sigma=LDS_SIGMA
        )
    )

    print(
        "\nLDS weights:"
    )

    print(
        f"  min = {bin_weights.min():.4f}"
    )

    print(
        f"  max = {bin_weights.max():.4f}"
    )

    print(
        f"  mean = {bin_weights.mean():.4f}"
    )

    # --------------------------------------------------------
    # DATALOADERS
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
    # MODEL
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
        f"\nTrainable head parameters: "
        f"{trainable_params:,}"
    )

    criterion = (
        LDSWeightedHuberLoss(
            bin_edges,
            bin_weights,
            delta=HUBER_DELTA
        )
        .to(DEVICE)
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

    best_val_mae = float(
        "inf"
    )

    patience_counter = 0
    history = []

    # --------------------------------------------------------
    # TRAIN
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
        # VALIDATION
        # ----------------------------------------------------

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
            "Train_LDS_Huber_Loss":
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
            f"Val MAE: {vm['MAE']:.4f} | "
            f"R2: {vm['R2']:.4f} | "
            f"Pearson: {vm['Pearson_r']:.4f} | "
            +
            " | ".join(
                f"{r['Bucket']}:{r['MAE']:.2f}"
                for _, r
                in vb.iterrows()
            )
        )

        if (
            vm["MAE"] <
            best_val_mae
        ):

            best_val_mae = (
                vm["MAE"]
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
                    "input_dim":
                        input_dim,
                    "seq_len":
                        seq_len,
                    "lds_num_bins":
                        LDS_NUM_BINS,
                    "lds_sigma":
                        LDS_SIGMA,
                    "huber_delta":
                        HUBER_DELTA,
                    "lds_bin_edges":
                        torch.tensor(
                            bin_edges,
                            dtype=torch.float32
                        ),
                    "lds_bin_weights":
                        torch.tensor(
                            bin_weights,
                            dtype=torch.float32
                        ),
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
        f"Best Val MAE: "
        f"{best_val_mae:.4f}"
    )

    print(
        f"Checkpoint:\n{BEST_CKPT}"
    )


if __name__ == "__main__":
    main()
