
# ============================================================
# FINAL TEST: R3D-18 SPATIAL CACHE + CONVLSTM
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

BEST_CKPT = os.path.join(
    SAVE,
    "best_model.pth"
)

BATCH_SIZE = 16

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.25

SHAPE_PROJ = 128
GLOBAL_DIM = 16
ATTN_HIDDEN = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_cache(split):

    d = torch.load(
        os.path.join(
            CACHE,
            f"{split}_spatial.pt"
        ),
        map_location="cpu"
    )

    return (
        d["spatial"].float(),
        d["shape"].float(),
        d["global_delta"].float(),
        d["ef"].float(),
        d["video_ids"]
    )


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=kernel_size//2
        )

    def forward(self,x,h,c):

        z = self.conv(
            torch.cat([x,h],1)
        )

        i,f,o,g = torch.chunk(
            z,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f*c + i*g
        h = o*torch.tanh(c)

        return h,c


class ConvLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layers,
        dropout
    ):
        super().__init__()

        cells = []

        for l in range(layers):

            cells.append(
                ConvLSTMCell(
                    input_dim if l==0 else hidden_dim,
                    hidden_dim
                )
            )

        self.cells = nn.ModuleList(
            cells
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layers = layers
        self.hidden_dim = hidden_dim

    def forward(self,x):

        b,t,_,h,w = x.shape

        layer_input = x

        for layer,cell in enumerate(
            self.cells
        ):

            hidden = torch.zeros(
                b,
                self.hidden_dim,
                h,
                w,
                device=x.device,
                dtype=x.dtype
            )

            state = torch.zeros_like(
                hidden
            )

            outputs=[]

            for step in range(t):

                hidden,state = cell(
                    layer_input[:,step],
                    hidden,
                    state
                )

                outputs.append(
                    hidden
                )

            layer_input = torch.stack(
                outputs,
                dim=1
            )

            if layer < self.layers-1:
                layer_input = self.dropout(
                    layer_input
                )

        return layer_input


class AttentionPool(nn.Module):

    def __init__(
        self,
        d,
        hidden=128
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(d,hidden),
            nn.Tanh(),
            nn.Linear(hidden,1)
        )

    def forward(self,x):

        w = torch.softmax(
            self.score(x),
            1
        )

        return (
            x*w
        ).sum(1)


class R3DConvLSTMRegressor(nn.Module):

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

        self.temporal_pool = AttentionPool(
            HIDDEN_DIM,
            ATTN_HIDDEN
        )

        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(shape_dim),
            nn.Linear(shape_dim,SHAPE_PROJ),
            nn.ReLU(inplace=True),
            nn.Dropout(.10)
        )

        self.global_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4,GLOBAL_DIM),
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
            nn.Dropout(.25),

            nn.Linear(256,64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(.15),

            nn.Linear(64,1)
        )

    def forward(
        self,
        spatial,
        shape,
        gd
    ):

        x = self.convlstm(
            spatial
        )

        x = x.mean(
            dim=(-1,-2)
        )

        x = self.temporal_pool(
            x
        )

        shape = self.shape_encoder(
            shape
        )

        gd = self.global_encoder(
            gd
        )

        return self.regressor(
            torch.cat(
                [x,shape,gd],
                1
            )
        ).squeeze(1)


def metrics(y,p):

    c = (
        pearsonr(y,p)[0]
        if np.std(y)>0
        and np.std(p)>0
        else np.nan
    )

    return {
        "MAE":float(
            mean_absolute_error(y,p)
        ),
        "RMSE":float(
            math.sqrt(
                mean_squared_error(y,p)
            )
        ),
        "R2":float(
            r2_score(y,p)
        ),
        "Pearson_r":float(c),
        "Bias":float(
            np.mean(p-y)
        )
    }


def buckets(y,p):

    d = pd.DataFrame({
        "GT_EF":y,
        "Pred_EF":p
    })

    d["Abs_Error"] = (
        d["Pred_EF"] -
        d["GT_EF"]
    ).abs()

    d["Bucket"] = pd.cut(
        d["GT_EF"],
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
        d.groupby(
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
        " FINAL TEST: R3D-18 SPATIAL CACHE + CONVLSTM"
    )

    print(
        "============================================================"
    )

    print(
        "Device:",
        DEVICE
    )

    x,s,g,y,ids = load_cache(
        "test"
    )

    print(
        "TEST spatial:",
        x.shape
    )

    print(
        "TEST shape:",
        s.shape
    )

    print(
        "TEST global:",
        g.shape
    )

    print(
        "TEST EF:",
        y.shape
    )

    model = R3DConvLSTMRegressor(
        x.shape[2],
        s.shape[1]
    ).to(DEVICE)

    ckpt = torch.load(
        BEST_CKPT,
        map_location=DEVICE
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )

    model.eval()

    print(
        "ConvLSTM checkpoint loaded."
    )

    print(
        "Best epoch:",
        ckpt.get("epoch","?")
    )

    print(
        "Best Val MAE:",
        ckpt.get("best_val_mae","?")
    )

    loader = DataLoader(
        TensorDataset(x,s,g),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    pred=[]

    for xb,sb,gb in loader:

        pred.append(
            model(
                xb.to(DEVICE),
                sb.to(DEVICE),
                gb.to(DEVICE)
            ).cpu()
        )

    p = np.clip(
        torch.cat(pred).numpy(),
        0,
        100
    )

    yy = y.numpy()

    m = metrics(
        yy,p
    )

    b = buckets(
        yy,p
    )

    print(
        "\n============================================================"
    )

    print(
        "3D CNN + CONVLSTM TEST RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        m
    )

    print(
        "\nBucket breakdown:"
    )

    print(
        b.to_string(
            index=False
        )
    )

    out = pd.DataFrame({
        "Video":ids,
        "GT_EF":yy,
        "Pred_EF":p
    })

    out.to_csv(
        os.path.join(
            SAVE,
            "test_predictions.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [m]
    ).to_csv(
        os.path.join(
            SAVE,
            "test_metrics.csv"
        ),
        index=False
    )

    b.to_csv(
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


if __name__=="__main__":
    main()
