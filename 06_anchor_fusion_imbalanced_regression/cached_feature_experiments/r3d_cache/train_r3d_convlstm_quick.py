
# ============================================================
# QUICK SCREEN: R3D-18 SPATIAL CACHE + CONVLSTM
# ============================================================
# Input:
#   spatial: [N,8,256,4,4]
#   shape:   [N,2*shape_dim]
#   global:  [N,4]
#
# Architecture:
#   frozen R3D spatial maps
#       ↓
#   ConvLSTM
#       ↓
#   temporal attention over ConvLSTM states
#       ↓
#   ED/ES shape embedding
#       +
#   global delta
#       ↓
#   EF regressor
#
# This is deliberately different from:
#   R3D + Attention
#   R3D + TCN
#   R3D + Transformer
#
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
    r2_score
)

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE = os.path.join(
    BASE,
    "outputs",
    "stage2_r3d_spatial_cache"
)
SAVE = os.path.join(
    BASE,
    "outputs",
    "r3d_convlstm_quick"
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

BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10

LR = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.25

SHAPE_PROJ = 128
GLOBAL_DIM = 16
ATTN_HIDDEN = 128

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
        f"{split}_spatial.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache:\n{path}\n"
            "Run extract_r3d_spatial_cache_convlstm.py first."
        )

    d = torch.load(
        path,
        map_location="cpu"
    )

    spatial = d["spatial"].float()
    shape = d["shape"].float()
    global_delta = d["global_delta"].float()
    ef = d["ef"].float()
    video_ids = d["video_ids"]

    return (
        spatial,
        shape,
        global_delta,
        ef,
        video_ids
    )


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3
    ):
        super().__init__()

        padding = kernel_size // 2

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x,
        h,
        c
    ):

        combined = torch.cat(
            [x, h],
            dim=1
        )

        gates = self.conv(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = (
            f * c +
            i * g
        )

        h_next = (
            o *
            torch.tanh(c_next)
        )

        return h_next, c_next


class ConvLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.25
    ):
        super().__init__()

        self.hidden_dims = [
            hidden_dim
        ] * num_layers

        cells = []

        for layer in range(
            num_layers
        ):

            in_dim = (
                input_dim
                if layer == 0
                else hidden_dim
            )

            cells.append(
                ConvLSTMCell(
                    in_dim,
                    hidden_dim,
                    kernel_size=3
                )
            )

        self.cells = nn.ModuleList(
            cells
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):
        """
        x: [B,T,C,H,W]
        returns:
            sequence of final-layer hidden maps
            [B,T,HIDDEN,H,W]
        """

        b, t, _, h, w = x.shape

        layer_input = x
        last_sequence = None

        for layer, cell in enumerate(
            self.cells
        ):

            hidden = torch.zeros(
                b,
                self.hidden_dims[layer],
                h,
                w,
                device=x.device,
                dtype=x.dtype
            )

            cell_state = torch.zeros_like(
                hidden
            )

            outputs = []

            for step in range(t):

                hidden, cell_state = (
                    cell(
                        layer_input[:, step],
                        hidden,
                        cell_state
                    )
                )

                outputs.append(
                    hidden
                )

            layer_output = torch.stack(
                outputs,
                dim=1
            )

            if layer < len(
                self.cells
            ) - 1:

                layer_output = (
                    self.dropout(
                        layer_output
                    )
                )

            layer_input = layer_output
            last_sequence = layer_output

        return last_sequence


class TemporalAttentionPool(
    nn.Module
):

    def __init__(
        self,
        dim,
        hidden=128
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                dim,
                hidden
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1
            )
        )

    def forward(
        self,
        x
    ):

        weights = torch.softmax(
            self.score(x),
            dim=1
        )

        return (
            x * weights
        ).sum(
            dim=1
        )


class R3DConvLSTMRegressor(
    nn.Module
):

    def __init__(
        self,
        spatial_channels,
        shape_dim
    ):
        super().__init__()

        self.convlstm = ConvLSTM(
            spatial_channels,
            HIDDEN_DIM,
            NUM_LAYERS,
            DROPOUT
        )

        # Spatial pooling after ConvLSTM.
        self.temporal_pool = (
            TemporalAttentionPool(
                HIDDEN_DIM,
                ATTN_HIDDEN
            )
        )

        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(
                shape_dim
            ),
            nn.Linear(
                shape_dim,
                SHAPE_PROJ
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10)
        )

        self.global_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(
                4,
                GLOBAL_DIM
            ),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                HIDDEN_DIM +
                SHAPE_PROJ +
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
        spatial,
        shape,
        global_delta
    ):

        # spatial: [B,T,C,H,W]
        seq = self.convlstm(
            spatial
        )

        # Spatial average -> [B,T,HIDDEN]
        seq = seq.mean(
            dim=(-1, -2)
        )

        temporal = (
            self.temporal_pool(
                seq
            )
        )

        shape = self.shape_encoder(
            shape
        )

        global_delta = (
            self.global_encoder(
                global_delta
            )
        )

        fused = torch.cat(
            [
                temporal,
                shape,
                global_delta
            ],
            dim=1
        )

        return self.regressor(
            fused
        ).squeeze(1)


def metrics(y, p):

    corr = (
        pearsonr(y, p)[0]
        if np.std(y) > 0
        and np.std(p) > 0
        else np.nan
    )

    return {
        "MAE": float(
            mean_absolute_error(y, p)
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(y, p)
            )
        ),
        "R2": float(
            r2_score(y, p)
        ),
        "Pearson_r": float(corr),
        "Bias": float(
            np.mean(p-y)
        )
    }


def buckets(y, p):

    df = pd.DataFrame({
        "GT_EF": y,
        "Pred_EF": p
    })

    df["Abs_Error"] = (
        df["Pred_EF"] -
        df["GT_EF"]
    ).abs()

    df["Bucket"] = pd.cut(
        df["GT_EF"],
        bins=[
            0,30,40,50,60,70,100
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
def predict(
    model,
    loader
):

    model.eval()
    out = []

    for spatial, shape, gd in loader:

        pred = model(
            spatial.to(
                DEVICE,
                non_blocking=True
            ),
            shape.to(
                DEVICE,
                non_blocking=True
            ),
            gd.to(
                DEVICE,
                non_blocking=True
            )
        )

        out.append(
            pred.cpu()
        )

    return torch.cat(
        out
    ).numpy()


def main():

    seed_all(SEED)

    print(
        "\n============================================================"
    )
    print(
        " QUICK SCREEN: R3D-18 SPATIAL CACHE + CONVLSTM"
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

    train_x, train_s, train_g, train_y, _ = (
        load_cache("train")
    )

    val_x, val_s, val_g, val_y, _ = (
        load_cache("val")
    )

    print(
        "\nCache:"
    )

    print(
        "Train spatial:",
        train_x.shape
    )

    print(
        "Val spatial:",
        val_x.shape
    )

    print(
        "Shape:",
        train_s.shape
    )

    model = R3DConvLSTMRegressor(
        spatial_channels=train_x.shape[2],
        shape_dim=train_s.shape[1]
    ).to(DEVICE)

    print(
        "Trainable ConvLSTM/head parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    train_loader = DataLoader(
        TensorDataset(
            train_x,
            train_s,
            train_g,
            train_y
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type=="cuda")
    )

    val_loader = DataLoader(
        TensorDataset(
            val_x,
            val_s,
            val_g,
            val_y
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type=="cuda")
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
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

    best = float("inf")
    bad = 0
    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()
        total = 0.0
        count = 0

        for spatial, shape, gd, ef in train_loader:

            spatial = spatial.to(
                DEVICE,
                non_blocking=True
            )

            shape = shape.to(
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
                spatial,
                shape,
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
                5.0
            )

            optimizer.step()

            total += (
                loss.item()
                * spatial.size(0)
            )

            count += (
                spatial.size(0)
            )

        val_pred = predict(
            model,
            DataLoader(
                TensorDataset(
                    val_x,
                    val_s,
                    val_g
                ),
                batch_size=BATCH_SIZE,
                shuffle=False
            )
        )

        vm = metrics(
            val_y.numpy(),
            val_pred
        )

        vb = buckets(
            val_y.numpy(),
            val_pred
        )

        scheduler.step(
            vm["MAE"]
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

        history.append(
            {
                "Epoch": epoch,
                "Train_Loss": total/count,
                **vm
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_CSV,
            index=False
        )

        if vm["MAE"] < best:

            best = vm["MAE"]
            bad = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "best_val_mae":
                        best,
                    "spatial_channels":
                        train_x.shape[2],
                    "shape_dim":
                        train_s.shape[1]
                },
                BEST_CKPT
            )

            print(
                f"  New best! Val MAE = {best:.4f}"
            )

        else:

            bad += 1

        if bad >= PATIENCE:

            print(
                "Early stopping triggered."
            )
            break

    print(
        "\n============================================================"
    )
    print(
        "CONVLSTM SCREEN COMPLETE"
    )
    print(
        "============================================================"
    )

    print(
        "Best Val MAE:",
        best
    )

    print(
        "Checkpoint:",
        BEST_CKPT
    )


if __name__ == "__main__":
    main()
