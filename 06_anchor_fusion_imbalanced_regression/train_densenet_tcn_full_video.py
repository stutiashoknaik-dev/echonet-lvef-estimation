
# ============================================================
# FULL VIDEO TRAINING: DenseNet + SAME V2 MULTIMODAL FUSION + TCN
# ============================================================
# FINAL / EXPENSIVE RUN
#
# V2 fusion is preserved:
#   video
#      ↓
#   DenseNet + existing CBAM backbone
#      +
#   ED/ES contour encoder
#      +
#   ED/ES morphology encoder
#      +
#   role flag
#      ↓
#   32 x 1107 fused sequence
#      ↓
#   TCN (replaces v2 BiLSTM)
#      ↓
#   temporal attention
#      +
#   global-delta encoder
#      ↓
#   EF regression
#
# This is NOT the cached quick run.
# The DenseNet visual branch is allowed to fine-tune.
#
# Resume behavior:
#   last_checkpoint.pth
#       -> resumes latest epoch
#       -> restores model
#       -> restores optimizer
#       -> restores scheduler
#       -> restores AMP scaler
#       -> restores best Val MAE
#       -> restores patience counter
#       -> restores history
#
# best_model.pth:
#   always contains the best validation-MAE model.
#
# Test is NOT touched during training.
# ============================================================

import os
import math
import random
import warnings
from copy import deepcopy

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


# ============================================================
# PATHS
# ============================================================

BASE_PATH = (
    "/content/drive/MyDrive/EchoNet-Dynamic"
)

V2_CKPT = os.path.join(
    BASE_PATH,
    "outputs",
    "anchor_fusion_densenet_bilstm_v2",
    "best_model.pth",
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "densenet_tcn_full_video",
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

BEST_CKPT = os.path.join(
    SAVE_DIR,
    "best_model.pth",
)

LAST_CKPT = os.path.join(
    SAVE_DIR,
    "last_checkpoint.pth",
)

HISTORY_CSV = os.path.join(
    SAVE_DIR,
    "training_history.csv",
)

MORPH_MEAN_PATH = os.path.join(
    SAVE_DIR,
    "morph_mean.npy",
)

MORPH_STD_PATH = os.path.join(
    SAVE_DIR,
    "morph_std.npy",
)


# ============================================================
# DATA / TRAINING CONFIG
# ============================================================

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2

# Gradient accumulation keeps memory closer to batch=4
# while giving a larger effective optimization batch.
GRAD_ACCUM_STEPS = 2

EPOCHS = 80
PATIENCE = 12

# Differential learning rates:
# visual backbone is updated conservatively;
# new TCN/fusion head learns faster.
BACKBONE_LR = 1e-5
SHAPE_LR = 3e-5
HEAD_LR = 3e-4

WEIGHT_DECAY = 1e-4

TCN_CHANNELS = 256
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]
TCN_DROPOUT = 0.20

GLOBAL_EMBED_DIM = 16
PROJ_DIM = 256
ATTN_HIDDEN = 128

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python":
            random.getstate(),
        "numpy":
            np.random.get_state(),
        "torch":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state
):
    if not state:
        return

    try:
        random.setstate(
            state["python"]
        )
        np.random.set_state(
            state["numpy"]
        )
        torch.set_rng_state(
            state["torch"]
        )

        if (
            torch.cuda.is_available()
            and "cuda" in state
        ):
            torch.cuda.set_rng_state_all(
                state["cuda"]
            )
    except Exception as exc:
        print(
            "Warning: RNG state could not "
            f"be fully restored: {exc}"
        )


# ============================================================
# TCN
# ============================================================

class Chomp1d(
    nn.Module
):

    def __init__(
        self,
        chomp_size
    ):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(
        self,
        x
    ):
        if self.chomp_size == 0:
            return x

        return x[
            :,
            :,
            :-self.chomp_size
        ].contiguous()


class TemporalBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
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
            dilation=dilation,
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
            dilation=dilation,
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
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x
    ):

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

        return self.activation(
            out +
            self.residual(x)
        )


class TCNEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim,
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
                    TCN_DROPOUT,
                )
            )

            in_channels = TCN_CHANNELS

        self.network = nn.Sequential(
            *blocks
        )

    def forward(
        self,
        x
    ):
        # [B,T,D] -> [B,D,T]
        x = x.transpose(
            1,
            2
        )

        x = self.network(
            x
        )

        # [B,D,T] -> [B,T,D]
        return x.transpose(
            1,
            2
        )


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

        weights = torch.softmax(
            self.score(x),
            dim=1
        )

        return (
            x *
            weights
        ).sum(
            dim=1
        )


# ============================================================
# FULL MODEL
# ============================================================

class DenseNetTCNFullVideo(
    nn.Module
):

    def __init__(
        self
    ):
        super().__init__()

        # Existing v2 model contains:
        #   DenseNet + CBAM backbone
        #   contour encoder
        #   morphology encoder
        #   original LSTM/attention/regressor
        #
        # We reuse the existing submodules for the visual /
        # shape branches, but replace the temporal head.
        self.base = (
            AnchorFusionDenseNetBiLSTM()
        )

        # Expected fused dimension from your established cache:
        # CNN features + role flag + shape embedding = 1107.
        #
        # We intentionally infer it from the actual base model
        # rather than hard-coding an internal CNN feature size.
        self.fused_dim = (
            self._infer_fused_dim()
        )

        self.input_projection = nn.Sequential(
            nn.LayerNorm(
                self.fused_dim
            ),
            nn.Linear(
                self.fused_dim,
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

        # The original v2 temporal LSTM/attention/regressor are
        # NOT used by this model.
        for module_name in [
            "lstm",
            "temporal_pool",
            "global_delta_encoder",
            "regressor",
        ]:
            if hasattr(
                self.base,
                module_name
            ):
                for p in getattr(
                    self.base,
                    module_name
                ).parameters():
                    p.requires_grad = False

    def _infer_fused_dim(
        self
    ):

        # The established production cache is 1107 dimensions.
        # Prefer the actual model's stored shape when possible.
        #
        # Your supplied fusion script confirms:
        # fused = cnn_feats + role_flag + shape_seq
        # and the resulting representation is (B,T,1107).
        #
        # We calculate this from the model's declared dimensions.
        #
        # CNN output width:
        #   1107 - role_flag_width - shape_embed_dim.
        # The role flag is determined from the dataset and is 3.
        role_width = 3

        if not hasattr(
            self.base,
            "shape_embed_dim"
        ):
            raise AttributeError(
                "v2 model does not expose "
                "shape_embed_dim."
            )

        shape_dim = int(
            self.base.shape_embed_dim
        )

        fused_dim = (
            1107
        )

        if (
            fused_dim -
            role_width -
            shape_dim
            <= 0
        ):
            raise ValueError(
                "Unexpected v2 fusion dimensions: "
                f"shape_dim={shape_dim}"
            )

        return fused_dim

    def extract_fused_sequence(
        self,
        video,
        role_flag,
        ed_pos,
        es_pos,
        ed_contour,
        es_contour,
        ed_morph,
        es_morph,
    ):
        """
        Exactly the same fusion logic as the cached 1107
        production sequence, but computed live from the video.

        This follows your supplied cache code:
            cnn_feats
            +
            role_flag
            +
            ED/ES shape sequence.
        """

        b, t = (
            video.shape[0],
            video.shape[1]
        )

        device = (
            video.device
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
            device=device,
            dtype=cnn_feats.dtype
        )

        batch_idx = torch.arange(
            b,
            device=device
        )

        shape_seq[
            batch_idx,
            ed_pos
        ] = ed_embed

        shape_seq[
            batch_idx,
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

        # Safety check against the established 1107 representation.
        if fused.shape[-1] != self.fused_dim:
            raise RuntimeError(
                "Fusion dimension mismatch. "
                f"Expected {self.fused_dim}, "
                f"got {fused.shape[-1]}."
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
        global_delta,
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

        pred = self.regressor(
            torch.cat(
                [
                    x,
                    g
                ],
                dim=1
            )
        )

        return pred.squeeze(1)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_v2_into_model(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    base_state = (
        model.base.state_dict()
    )

    compatible = {
        k: v
        for k, v in state.items()
        if (
            k in base_state
            and
            base_state[k].shape == v.shape
        )
    }

    missing_before = (
        len(base_state) -
        len(compatible)
    )

    model.base.load_state_dict(
        compatible,
        strict=False
    )

    print(
        f"Loaded {len(compatible)}/"
        f"{len(base_state)} compatible "
        "parameter tensors from v2."
    )

    print(
        f"Skipped {missing_before} tensors "
        "(new TCN head and any incompatible "
        "v2-only tensors are expected)."
    )


# ============================================================
# BATCH TRANSFER
# ============================================================

def move_batch(
    batch
):

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

    result = {}

    for k in keys:

        result[k] = batch[k].to(
            DEVICE,
            non_blocking=True
        )

    return result


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
            len(y_true) > 1
            and np.std(y_true) > 0
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
        ),
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
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_validation(
    model,
    loader
):

    model.eval()

    all_gt = []
    all_pred = []

    for batch in loader:

        b = move_batch(
            batch
        )

        with torch.autocast(
            device_type=(
                DEVICE.type
            ),
            enabled=USE_AMP,
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
                b["global_delta"],
            )

        all_gt.extend(
            b["ef"]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            pred.float()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    y = np.asarray(
        all_gt,
        dtype=np.float32
    )

    p = np.clip(
        np.asarray(
            all_pred,
            dtype=np.float32
        ),
        0,
        100
    )

    return (
        y,
        p
    )


# ============================================================
# OPTIMIZER PARAMETER GROUPS
# ============================================================

def build_optimizer(
    model
):

    backbone_params = []
    shape_params = []
    head_params = []

    # Existing DenseNet/visual + existing shape encoders.
    for name, p in model.base.named_parameters():

        if not p.requires_grad:
            continue

        if (
            name.startswith(
                "contour_encoder."
            )
            or
            name.startswith(
                "morph_encoder."
            )
        ):

            shape_params.append(p)

        else:

            backbone_params.append(p)

    # New TCN/projection/attention/global/regressor.
    for p in model.input_projection.parameters():
        head_params.append(p)

    for p in model.tcn.parameters():
        head_params.append(p)

    for p in model.temporal_pool.parameters():
        head_params.append(p)

    for p in model.global_delta_encoder.parameters():
        head_params.append(p)

    for p in model.regressor.parameters():
        head_params.append(p)

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR,
            },
            {
                "params": shape_params,
                "lr": SHAPE_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print(
        "\n============================================================"
    )
    print(
        " FULL VIDEO TRAINING: DenseNet + TCN"
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
        "AMP:",
        USE_AMP
    )

    print(
        "Frames:",
        NUM_FRAMES
    )

    print(
        "Batch:",
        BATCH_SIZE,
        "| Grad accumulation:",
        GRAD_ACCUM_STEPS
    )

    print(
        "Epochs:",
        EPOCHS,
        "| Patience:",
        PATIENCE
    )

    print(
        "Backbone LR:",
        BACKBONE_LR,
        "| Shape LR:",
        SHAPE_LR,
        "| Head LR:",
        HEAD_LR
    )

    # --------------------------------------------------------
    # Morphology normalization.
    #
    # Prefer the saved stats from this training output folder.
    # Otherwise compute them from TRAIN only.
    # --------------------------------------------------------

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
            "\nLoaded saved train-only "
            "morphology normalization stats."
        )

    else:

        print(
            "\nComputing morphology normalization "
            "statistics from TRAIN only..."
        )

        morph_mean, morph_std = (
            compute_morph_norm_stats(
                BASE_PATH
            )
        )

        np.save(
            MORPH_MEAN_PATH,
            morph_mean
        )

        np.save(
            MORPH_STD_PATH,
            morph_std
        )

        print(
            "Saved morphology stats."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds = (
        AnchorFusionEFDataset(
            split="train",
            base_path=BASE_PATH,
            num_frames=NUM_FRAMES,
            morph_mean=morph_mean,
            morph_std=morph_std,
        )
    )

    val_ds = (
        AnchorFusionEFDataset(
            split="val",
            base_path=BASE_PATH,
            num_frames=NUM_FRAMES,
            morph_mean=morph_mean,
            morph_std=morph_std,
        )
    )

    print(
        f"\nTrain videos: {len(train_ds)}"
    )

    print(
        f"Val videos: {len(val_ds)}"
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_ds,
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

    # --------------------------------------------------------
    # Model / initialization selection
    # --------------------------------------------------------

    model = (
        DenseNetTCNFullVideo()
        .to(DEVICE)
    )

    # Priority:
    #   1. last_checkpoint.pth -> exact resume
    #   2. best_model.pth      -> restore best weights
    #                            with fresh optimizer state
    #   3. v2 checkpoint       -> fresh initialization
    if os.path.exists(LAST_CKPT):

        resume_mode = "last"

        print(
            "\nLatest checkpoint found:"
        )
        print(
            LAST_CKPT
        )

    elif os.path.exists(BEST_CKPT):

        resume_mode = "best"

        print(
            "\nNo latest checkpoint found, "
            "but best checkpoint exists:"
        )
        print(
            BEST_CKPT
        )

        checkpoint = torch.load(
            BEST_CKPT,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

    else:

        resume_mode = "fresh"

        if not os.path.exists(
            V2_CKPT
        ):
            raise FileNotFoundError(
                "v2 checkpoint not found:\n"
                f"{V2_CKPT}"
            )

        print(
            "\nNo DenseNet+TCN checkpoint found."
        )

        print(
            "Loading v2 checkpoint as initialization..."
        )

        load_v2_into_model(
            model,
            V2_CKPT,
            DEVICE
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    print(
        "\nOptimizer learning rates:"
    )
    print(
        "  DenseNet/backbone:",
        optimizer.param_groups[0]["lr"]
    )
    print(
        "  contour/morphology:",
        optimizer.param_groups[1]["lr"]
    )
    print(
        "  TCN/fusion head:",
        optimizer.param_groups[2]["lr"]
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
        )
    )

    criterion = (
        nn.SmoothL1Loss(
            beta=1.0
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP
        )
    )

    start_epoch = 1
    best_val_mae = float(
        "inf"
    )
    patience_counter = 0
    history = []

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if resume_mode == "last":

        print(
            "\nResuming exact training state from:"
        )
        print(
            LAST_CKPT
        )

        checkpoint = torch.load(
            LAST_CKPT,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        if (
            USE_AMP
            and "scaler_state_dict" in checkpoint
        ):

            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_mae = checkpoint[
            "best_val_mae"
        ]

        patience_counter = checkpoint.get(
            "patience_counter",
            0
        )

        if os.path.exists(
            HISTORY_CSV
        ):

            history = (
                pd.read_csv(
                    HISTORY_CSV
                )
                .to_dict(
                    "records"
                )
            )

        restore_rng_state(
            checkpoint.get(
                "rng_state"
            )
        )

        print(
            f"Resumed after epoch "
            f"{checkpoint['epoch']}."
        )

        print(
            f"Best Val MAE so far: "
            f"{best_val_mae:.4f}"
        )

    elif resume_mode == "best":

        checkpoint = torch.load(
            BEST_CKPT,
            map_location=DEVICE
        )

        # Best model weights were already loaded above.
        start_epoch = (
            checkpoint.get("epoch", 0) + 1
        )

        best_val_mae = checkpoint.get(
            "best_val_mae",
            float("inf")
        )

        if os.path.exists(
            HISTORY_CSV
        ):

            history = (
                pd.read_csv(
                    HISTORY_CSV
                )
                .to_dict(
                    "records"
                )
            )

        print(
            f"Restored best model from epoch "
            f"{checkpoint.get('epoch', '?')}."
        )

        print(
            f"Best Val MAE: "
            f"{best_val_mae:.4f}"
        )

        print(
            "Optimizer/scheduler/scaler start fresh "
            "because the latest training state was unavailable."
        )

    else:

        print(
            "\nStarting fresh from v2 initialization."
        )

    if start_epoch > EPOCHS:

        print(
            "\nTraining is already complete "
            f"(next epoch {start_epoch} > {EPOCHS})."
        )

        return

    # --------------------------------------------------------
    # Parameter summary
    # --------------------------------------------------------

    total_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{total_trainable:,}"
    )

    print(
        "\nTraining will use:"
    )

    print(
        "  Existing DenseNet/shape branches "
        "with conservative LR"
    )

    print(
        "  New TCN + fusion regression head "
        "with higher LR"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            b = move_batch(
                batch
            )

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
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
                    b["global_delta"],
                )

                raw_loss = criterion(
                    pred,
                    b["ef"]
                )

                loss = (
                    raw_loss /
                    GRAD_ACCUM_STEPS
                )

            if USE_AMP:

                scaler.scale(
                    loss
                ).backward()

            else:

                loss.backward()

            should_step = (
                step %
                GRAD_ACCUM_STEPS == 0
                or
                step == len(
                    train_loader
                )
            )

            if should_step:

                if USE_AMP:

                    scaler.unscale_(
                        optimizer
                    )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                if USE_AMP:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            bs = (
                b["video"]
                .size(0)
            )

            running_loss += (
                raw_loss.item() *
                bs
            )

            sample_count += bs

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        y_val, p_val = (
            evaluate_validation(
                model,
                val_loader
            )
        )

        val_metrics = (
            regression_metrics(
                y_val,
                p_val
            )
        )

        val_buckets = (
            bucket_metrics(
                y_val,
                p_val
            )
        )

        scheduler.step(
            val_metrics["MAE"]
        )

        row = {
            "Epoch": epoch,
            "Train_Loss":
                running_loss /
                sample_count,
            "LR_Backbone":
                optimizer.param_groups[0]["lr"],
            "LR_Shape":
                optimizer.param_groups[1]["lr"],
            "LR_Head":
                optimizer.param_groups[2]["lr"],
            **val_metrics
        }

        for _, r in val_buckets.iterrows():

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
            f"\nEpoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: "
            f"{row['Train_Loss']:.4f} | "
            f"Val MAE: "
            f"{val_metrics['MAE']:.4f} | "
            f"RMSE: "
            f"{val_metrics['RMSE']:.4f} | "
            f"R2: "
            f"{val_metrics['R2']:.4f} | "
            f"Pearson: "
            f"{val_metrics['Pearson_r']:.4f} | "
            f"Bias: "
            f"{val_metrics['Bias']:.4f}"
        )

        print(
            "  Buckets: "
            +
            " | ".join(
                f"{r['Bucket']}="
                f"{r['MAE']:.2f}"
                for _, r
                in val_buckets.iterrows()
            )
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["MAE"]
            < best_val_mae
        ):

            best_val_mae = (
                val_metrics["MAE"]
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
                    "patience_counter":
                        patience_counter,
                    "config": {
                        "num_frames":
                            NUM_FRAMES,
                        "batch_size":
                            BATCH_SIZE,
                        "grad_accum_steps":
                            GRAD_ACCUM_STEPS,
                        "epochs":
                            EPOCHS,
                        "backbone_lr":
                            BACKBONE_LR,
                        "shape_lr":
                            SHAPE_LR,
                        "head_lr":
                            HEAD_LR,
                        "tcn_channels":
                            TCN_CHANNELS,
                        "tcn_kernel":
                            TCN_KERNEL_SIZE,
                        "tcn_dilations":
                            TCN_DILATIONS,
                    },
                },
                BEST_CKPT
            )

            print(
                "  >>> NEW BEST MODEL <<<"
            )

            print(
                f"  Best Val MAE = "
                f"{best_val_mae:.4f}"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        last_payload = {
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "scaler_state_dict":
                (
                    scaler.state_dict()
                    if USE_AMP
                    else {}
                ),
            "epoch":
                epoch,
            "best_val_mae":
                best_val_mae,
            "patience_counter":
                patience_counter,
            "config": {
                "num_frames":
                    NUM_FRAMES,
                "batch_size":
                    BATCH_SIZE,
                "grad_accum_steps":
                    GRAD_ACCUM_STEPS,
                "epochs":
                    EPOCHS,
                "backbone_lr":
                    BACKBONE_LR,
                "head_lr":
                    HEAD_LR,
            },
            "rng_state":
                capture_rng_state(),
        }

        torch.save(
            last_payload,
            LAST_CKPT
        )

        print(
            f"  Last checkpoint saved "
            f"at epoch {epoch}."
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        "\n============================================================"
    )

    print(
        "FULL TRAINING FINISHED"
    )

    print(
        "============================================================"
    )

    print(
        f"Best Val MAE: "
        f"{best_val_mae:.4f}"
    )

    print(
        f"Best checkpoint:\n"
        f"{BEST_CKPT}"
    )

    print(
        f"Latest checkpoint:\n"
        f"{LAST_CKPT}"
    )

    print(
        f"History:\n"
        f"{HISTORY_CSV}"
    )

    print(
        "\nTEST WAS NOT USED DURING TRAINING."
    )


if __name__ == "__main__":
    main()
