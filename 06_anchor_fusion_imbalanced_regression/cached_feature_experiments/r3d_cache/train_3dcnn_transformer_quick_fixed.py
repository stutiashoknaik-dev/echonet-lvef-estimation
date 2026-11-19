
# ============================================================
# LAST 3D-CNN SCREEN: CACHED R3D-18 + TRANSFORMER
# ============================================================
# Same cache as previous 3D-CNN experiments:
#   train: [7459, 8, 339]
#   val:   [1288, 8, 339]
#   test:  [1274, 8, 339]
#
# ONLY CHANGE:
#   temporal head = Transformer
#
# No LDS / FDS / BMC / resampling.
# Test set is used only for final evaluation.
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

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

CACHE = os.path.join(
    BASE,
    "outputs",
    "stage2_3dcnn_cache"
)

SAVE = os.path.join(
    BASE,
    "outputs",
    "3dcnn_transformer_quick"
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

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
FF_DIM = 512
DROPOUT = 0.20

GLOBAL_DIM = 16
ATTN_HIDDEN = 128

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

        scores = self.score(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        return (
            x * weights
        ).sum(
            dim=1
        )


class R3DTransformerEF(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                D_MODEL
            )
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                D_MODEL
            )
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_LAYERS
            )
        )

        self.temporal_pool = (
            TemporalAttentionPool(
                D_MODEL,
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
                D_MODEL + GLOBAL_DIM,
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

        x = (
            x +
            self.pos_embedding[
                :, :x.size(1)
            ]
        )

        x = self.transformer(
            x
        )

        x = self.temporal_pool(
            x
        )

        g = self.global_delta_encoder(
            global_delta
        )

        return self.regressor(
            torch.cat(
                [x, g],
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
                y_pred - y_true
            )
        )
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
            n=("Abs_Error", "size"),
            MAE=("Abs_Error", "mean")
        )
        .reset_index()
    )


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
        " LAST 3D-CNN SCREEN: CACHE + TRANSFORMER"
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

    train_x, train_g, train_y, _ = load_cache(
        "train"
    )

    val_x, val_g, val_y, _ = load_cache(
        "val"
    )

    print(
        "\nCache shapes:"
    )
    print(
        "Train:",
        train_x.shape
    )
    print(
        "Val:",
        val_x.shape
    )

    input_dim = train_x.shape[-1]
    seq_len = train_x.shape[1]

    train_loader = DataLoader(
        TensorDataset(
            train_x,
            train_g,
            train_y
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
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
        pin_memory=(DEVICE.type == "cuda")
    )

    model = R3DTransformerEF(
        input_dim,
        seq_len
    ).to(DEVICE)

    print(
        "Trainable Transformer/head parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

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

    best_val_mae = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        count = 0

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
            count += bs

        val_pred = predict_loader(
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

        history.append({
            "Epoch": epoch,
            "Train_Loss":
                running_loss / count,
            **vm
        })

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
            bad_epochs = 0

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
                    "d_model":
                        D_MODEL,
                    "nhead":
                        NHEAD,
                    "num_layers":
                        NUM_LAYERS,
                },
                BEST_CKPT
            )

            print(
                f"  New best! "
                f"Val MAE = {best_val_mae:.4f}"
            )

        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:

            print(
                "Early stopping triggered."
            )
            break

    print(
        "\n============================================================"
    )
    print(
        "TRANSFORMER SCREEN COMPLETE"
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
