
# ============================================================
# FINAL TEST: 3D CNN CACHE + TRANSFORMER
# ============================================================
# Exact same architecture and state-dict names as training.
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

BEST_CKPT = os.path.join(
    SAVE,
    "best_model.pth"
)

BATCH_SIZE = 64

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
FF_DIM = 512
DROPOUT = 0.20
GLOBAL_DIM = 16
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
        ).sum(1)


class R3DTransformerEF(nn.Module):

    def __init__(
        self,
        input_dim,
        seq_len
    ):
        super().__init__()

        # EXACT same names as training script.
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
                [x,g],
                dim=1
            )
        ).squeeze(1)


def metrics(
    y,
    p
):

    corr = (
        pearsonr(
            y,
            p
        )[0]
        if (
            np.std(y) > 0
            and np.std(p) > 0
        )
        else np.nan
    )

    return {
        "MAE": float(
            mean_absolute_error(
                y,
                p
            )
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    y,
                    p
                )
            )
        ),
        "R2": float(
            r2_score(
                y,
                p
            )
        ),
        "Pearson_r": float(corr),
        "Bias": float(
            np.mean(
                p-y
            )
        )
    }


def buckets(
    y,
    p
):

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
def main():

    print(
        "\n============================================================"
    )
    print(
        " FINAL TEST: 3D CNN CACHE + TRANSFORMER"
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
            f"Missing checkpoint: {BEST_CKPT}"
        )

    x,g,y,ids = load_cache(
        "test"
    )

    print(
        "\nTEST cache:"
    )
    print(
        "  fused:",
        x.shape
    )
    print(
        "  global_delta:",
        g.shape
    )
    print(
        "  EF:",
        y.shape
    )
    print(
        "  videos:",
        len(ids)
    )

    model = (
        R3DTransformerEF(
            x.shape[-1],
            x.shape[1]
        )
        .to(DEVICE)
    )

    ck = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )

    model.eval()

    print(
        "\nTransformer checkpoint loaded successfully."
    )
    print(
        "Best epoch:",
        ck.get("epoch","?")
    )
    print(
        "Best Val MAE:",
        ck.get("best_val_mae","?")
    )

    loader = DataLoader(
        TensorDataset(
            x,
            g
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    preds=[]

    for xb,gb in loader:

        preds.append(
            model(
                xb.to(
                    DEVICE,
                    non_blocking=True
                ),
                gb.to(
                    DEVICE,
                    non_blocking=True
                )
            ).cpu()
        )

    p=np.clip(
        torch.cat(
            preds
        ).numpy(),
        0,
        100
    )

    yy=y.numpy()

    tm=metrics(
        yy,
        p
    )

    tb=buckets(
        yy,
        p
    )

    print(
        "\n============================================================"
    )
    print(
        "3D CNN + TRANSFORMER TEST RESULTS"
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

    pd.DataFrame({
        "Video":ids,
        "GT_EF":yy,
        "Pred_EF":p
    }).to_csv(
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
