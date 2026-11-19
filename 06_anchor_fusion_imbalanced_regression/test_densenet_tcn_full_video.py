
# ============================================================
# FINAL TEST: FULL-VIDEO DenseNet + TCN
# ============================================================
# Loads ONLY best_model.pth from the full-video training run.
# Evaluates the untouched TEST split.
#
# Same V2 fusion:
#   DenseNet/CBAM + role + ED/ES contour/morphology
#   -> 32 x 1107
#   -> TCN + temporal attention
#   -> global delta
#   -> EF
#
# Test is used only here for final evaluation.
# ============================================================

import os
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from anchor_fusion_dataset_v3 import (
    AnchorFusionEFDataset,
    compute_morph_norm_stats,
)

from anchor_fusion_model_v3 import (
    AnchorFusionDenseNetBiLSTM,
)

warnings.filterwarnings("ignore")


BASE_PATH = (
    "/content/drive/MyDrive/EchoNet-Dynamic"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "densenet_tcn_full_video"
)

BEST_CKPT = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

METRICS_CSV = os.path.join(
    SAVE_DIR,
    "test_metrics.csv"
)

BUCKET_CSV = os.path.join(
    SAVE_DIR,
    "test_bucket_metrics.csv"
)

MORPH_MEAN_PATH = os.path.join(
    SAVE_DIR,
    "morph_mean.npy"
)

MORPH_STD_PATH = os.path.join(
    SAVE_DIR,
    "morph_std.npy"
)

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2

TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]
TCN_DROPOUT = 0.20

GLOBAL_EMBED_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
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
        return x[:,:,:-self.chomp_size].contiguous()


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
        cin = input_dim

        for dilation in TCN_DILATIONS:

            blocks.append(
                TemporalBlock(
                    cin,
                    TCN_CHANNELS,
                    TCN_KERNEL_SIZE,
                    dilation,
                    TCN_DROPOUT
                )
            )

            cin = TCN_CHANNELS

        self.network = nn.Sequential(
            *blocks
        )

    def forward(self,x):

        x = x.transpose(
            1,
            2
        )

        x = self.network(
            x
        )

        return x.transpose(
            1,
            2
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
        ).sum(
            dim=1
        )


class DenseNetTCNFullVideo(nn.Module):

    def __init__(self):

        super().__init__()

        self.base = (
            AnchorFusionDenseNetBiLSTM()
        )

        self.fused_dim = 1107

        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                self.fused_dim
            ),
            nn.Linear(
                self.fused_dim,
                PROJ_DIM
            )
        )

        self.tcn = (
            TCNEncoder(
                PROJ_DIM
            )
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

    def extract_fused_sequence(
        self,
        video,
        role_flag,
        ed_pos,
        es_pos,
        ed_contour,
        es_contour,
        ed_morph,
        es_morph
    ):

        b,t = (
            video.shape[0],
            video.shape[1]
        )

        cnn_feats = (
            self.base
            ._extract_cnn_features(
                video
            )
        )

        ed_embed = torch.cat(
            [
                self.base.contour_encoder(
                    ed_contour
                ),
                self.base.morph_encoder(
                    ed_morph
                ),
            ],
            dim=-1
        )

        es_embed = torch.cat(
            [
                self.base.contour_encoder(
                    es_contour
                ),
                self.base.morph_encoder(
                    es_morph
                ),
            ],
            dim=-1
        )

        shape_seq = torch.zeros(
            b,
            t,
            self.base.shape_embed_dim,
            device=video.device,
            dtype=cnn_feats.dtype
        )

        idx = torch.arange(
            b,
            device=video.device
        )

        shape_seq[
            idx,
            ed_pos
        ] = ed_embed

        shape_seq[
            idx,
            es_pos
        ] = es_embed

        fused = torch.cat(
            [
                cnn_feats,
                role_flag,
                shape_seq
            ],
            dim=-1
        )

        if fused.shape[-1] != 1107:
            raise RuntimeError(
                f"Expected fused dimension 1107, "
                f"got {fused.shape[-1]}"
            )

        return fused

    def forward(
        self,
        video,
        role_flag,
        ed_pos,
        es_pos,
        ed_contour,
        es_contour,
        ed_morph,
        es_morph,
        global_delta
    ):

        fused = (
            self.extract_fused_sequence(
                video,
                role_flag,
                ed_pos,
                es_pos,
                ed_contour,
                es_contour,
                ed_morph,
                es_morph,
            )
        )

        x = self.input_projection(
            fused
        )

        x = self.tcn(
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
        pearsonr(y,p)[0]
        if (
            len(y) > 1
            and np.std(y) > 0
            and np.std(p) > 0
        )
        else np.nan
    )

    return {
        "MAE": float(
            mean_absolute_error(
                y,p
            )
        ),
        "RMSE": float(
            math.sqrt(
                mean_squared_error(
                    y,p
                )
            )
        ),
        "R2": float(
            r2_score(
                y,p
            )
        ),
        "Pearson_r": float(
            corr
        ),
        "Bias": float(
            np.mean(p-y)
        )
    }


def buckets(
    y,
    p
):

    df = pd.DataFrame(
        {
            "GT_EF": y,
            "Pred_EF": p
        }
    )

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


def move_batch(batch):

    keys = [
        "video",
        "role_flag",
        "ed_pos",
        "es_pos",
        "ed_contour",
        "es_contour",
        "ed_morph",
        "es_morph",
        "global_delta",
        "ef",
    ]

    return {
        k: batch[k].to(
            DEVICE,
            non_blocking=True
        )
        for k in keys
    }


@torch.no_grad()
def main():

    print(
        "\n============================================================"
    )
    print(
        " FINAL TEST: FULL VIDEO DenseNet + TCN"
    )
    print(
        "============================================================"
    )

    print(
        "Device:",
        DEVICE
    )

    if not os.path.exists(
        BEST_CKPT
    ):
        raise FileNotFoundError(
            f"Best checkpoint not found:\n{BEST_CKPT}"
        )

    if (
        os.path.exists(
            MORPH_MEAN_PATH
        )
        and
        os.path.exists(
            MORPH_STD_PATH
        )
    ):

        morph_mean = np.load(
            MORPH_MEAN_PATH
        )

        morph_std = np.load(
            MORPH_STD_PATH
        )

        print(
            "Loaded saved train-only "
            "morphology stats."
        )

    else:

        print(
            "Saved morphology stats not found."
        )

        print(
            "Recomputing from TRAIN only..."
        )

        morph_mean, morph_std = (
            compute_morph_norm_stats(
                BASE_PATH
            )
        )

    test_ds = (
        AnchorFusionEFDataset(
            split="test",
            base_path=BASE_PATH,
            num_frames=NUM_FRAMES,
            morph_mean=morph_mean,
            morph_std=morph_std,
        )
    )

    print(
        "Test videos:",
        len(test_ds)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    model = (
        DenseNetTCNFullVideo()
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
        "Best checkpoint loaded."
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

    all_gt = []
    all_pred = []
    all_vid = []

    for batch in test_loader:

        b = move_batch(
            batch
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            pred = model(
                b["video"],
                b["role_flag"],
                b["ed_pos"],
                b["es_pos"],
                b["ed_contour"],
                b["es_contour"],
                b["ed_morph"],
                b["es_morph"],
                b["global_delta"]
            )

        pred_np = np.clip(
            pred.float()
            .cpu()
            .numpy(),
            0,
            100
        )

        all_gt.extend(
            b["ef"]
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            pred_np.tolist()
        )

        all_vid.extend(
            batch["video_id"]
        )

    y = np.asarray(
        all_gt,
        dtype=np.float32
    )

    p = np.asarray(
        all_pred,
        dtype=np.float32
    )

    tm = metrics(
        y,p
    )

    tb = buckets(
        y,p
    )

    print(
        "\n--- Test Set Metrics ---"
    )

    print(
        tm
    )

    print(
        "\n--- Test Set Bucket Breakdown ---"
    )

    print(
        tb.to_string(
            index=False
        )
    )

    pd.DataFrame(
        {
            "Video":
                all_vid,
            "GT_EF":
                y,
            "Pred_EF":
                p
        }
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    pd.DataFrame(
        [tm]
    ).to_csv(
        METRICS_CSV,
        index=False
    )

    tb.to_csv(
        BUCKET_CSV,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_CSV
    )
    print(
        METRICS_CSV
    )
    print(
        BUCKET_CSV
    )

    print(
        "\nTEST SET USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()
