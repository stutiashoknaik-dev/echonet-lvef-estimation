
# ============================================================
# 3D-CNN SPATIAL FEATURE CACHE FOR CONVLSTM
# ============================================================
# One expensive extraction step.
#
# R3D-18:
#   video [B,32,1,112,112]
#       ↓
#   frozen pretrained R3D-18
#       ↓
#   layer3 feature maps
#       ↓
#   spatial pooling to 4x4 ONLY
#       ↓
#   [B, 8, 256, 4, 4]
#
# Separately cache:
#   ED/ES contour + morphology embeddings
#   global_delta
#   EF
#   video IDs
#
# TRAIN morphology normalization is computed from TRAIN only.
# TEST is never used to fit anything.
# ============================================================

import os
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)

from anchor_fusion_dataset import (
    AnchorFusionEFDataset,
    compute_morph_norm_stats,
)

from anchor_fusion_model_v3 import (
    AnchorFusionDenseNetBiLSTM,
)

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

V2_CKPT = os.path.join(
    BASE,
    "outputs",
    "anchor_fusion_densenet_bilstm_v2",
    "best_model.pth"
)

CACHE_DIR = os.path.join(
    BASE,
    "outputs",
    "stage2_r3d_spatial_cache"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

NUM_FRAMES = 32
IMG_SIZE = 112
BATCH_SIZE = 4
NUM_WORKERS = 2

SPATIAL_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FrozenR3D18Layer3Map(nn.Module):

    def __init__(self):
        super().__init__()

        net = r3d_18(
            weights=R3D_18_Weights.DEFAULT
        )

        self.stem = net.stem
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, video):

        # [B,T,1,H,W]
        if video.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], got {tuple(video.shape)}"
            )

        x = video.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = x.repeat(
            1, 3, 1, 1, 1
        )

        mean = torch.tensor(
            [0.43216, 0.394666, 0.37645],
            device=x.device,
            dtype=x.dtype
        ).view(
            1, 3, 1, 1, 1
        )

        std = torch.tensor(
            [0.22803, 0.22145, 0.216989],
            device=x.device,
            dtype=x.dtype
        ).view(
            1, 3, 1, 1, 1
        )

        x = (
            x - mean
        ) / std

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # [B,256,T',H',W']
        # Preserve temporal and spatial structure.
        b, c, t, h, w = x.shape

        x = x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = x.view(
            b * t,
            c,
            h,
            w
        )

        x = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (SPATIAL_SIZE, SPATIAL_SIZE)
        )

        x = x.view(
            b,
            t,
            c,
            SPATIAL_SIZE,
            SPATIAL_SIZE
        )

        return x


def move_batch(batch):

    keys = [
        "video",
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
def extract_split(
    cnn,
    contour_encoder,
    morph_encoder,
    split,
    morph_mean,
    morph_std,
):

    print(
        f"\n============================================================"
    )

    print(
        f"Extracting spatial R3D cache: {split}"
    )

    print(
        "============================================================"
    )

    ds = AnchorFusionEFDataset(
        split,
        BASE,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        morph_mean=morph_mean,
        morph_std=morph_std
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    all_spatial = []
    all_shape = []
    all_global = []
    all_ef = []
    all_ids = []

    start = time.time()

    for i, batch in enumerate(
        loader,
        start=1
    ):

        b = move_batch(
            batch
        )

        spatial = cnn(
            b["video"]
        )

        ed_embed = torch.cat(
            [
                contour_encoder(
                    b["ed_contour"]
                ),
                morph_encoder(
                    b["ed_morph"]
                ),
            ],
            dim=-1
        )

        es_embed = torch.cat(
            [
                contour_encoder(
                    b["es_contour"]
                ),
                morph_encoder(
                    b["es_morph"]
                ),
            ],
            dim=-1
        )

        # Store ED/ES embeddings separately.
        shape = torch.cat(
            [
                ed_embed,
                es_embed
            ],
            dim=-1
        )

        all_spatial.append(
            spatial.half().cpu()
        )

        all_shape.append(
            shape.float().cpu()
        )

        all_global.append(
            b["global_delta"].float().cpu()
        )

        all_ef.append(
            b["ef"].float().cpu()
        )

        all_ids.extend(
            batch["video_id"]
        )

        if (
            i == 1
            or i % 100 == 0
            or i == len(loader)
        ):
            print(
                f"  processed "
                f"{i}/{len(loader)}"
            )

    spatial = torch.cat(
        all_spatial,
        dim=0
    )

    shape = torch.cat(
        all_shape,
        dim=0
    )

    global_delta = torch.cat(
        all_global,
        dim=0
    )

    ef = torch.cat(
        all_ef,
        dim=0
    )

    if not (
        spatial.shape[0]
        == shape.shape[0]
        == global_delta.shape[0]
        == ef.shape[0]
        == len(all_ids)
    ):
        raise RuntimeError(
            "Cached sample counts do not match."
        )

    out_path = os.path.join(
        CACHE_DIR,
        f"{split}_spatial.pt"
    )

    torch.save(
        {
            "spatial":
                spatial,
            "shape":
                shape,
            "global_delta":
                global_delta,
            "ef":
                ef,
            "video_ids":
                all_ids,
            "backbone":
                "r3d_18_layer3",
            "spatial_size":
                SPATIAL_SIZE,
            "feature_channels":
                int(spatial.shape[2]),
            "temporal_tokens":
                int(spatial.shape[1]),
        },
        out_path
    )

    elapsed = (
        time.time() - start
    )

    print(
        "\nSaved:",
        out_path
    )

    print(
        "  spatial:",
        spatial.shape,
        "(float16)"
    )

    print(
        "  shape:",
        shape.shape
    )

    print(
        "  global:",
        global_delta.shape
    )

    print(
        "  EF:",
        ef.shape
    )

    print(
        "  videos:",
        len(all_ids)
    )

    print(
        f"  elapsed: {elapsed/60:.2f} min"
    )


def main():

    print(
        "\n============================================================"
    )

    print(
        " R3D-18 SPATIAL CACHE FOR CONVLSTM"
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

    morph_mean, morph_std = (
        compute_morph_norm_stats(
            BASE
        )
    )

    print(
        "Morphology normalization: TRAIN ONLY"
    )

    # Load the already-trained v2 shape encoders.
    base = (
        AnchorFusionDenseNetBiLSTM()
        .to(DEVICE)
    )

    ckpt = torch.load(
        V2_CKPT,
        map_location=DEVICE
    )

    state = ckpt.get(
        "model_state_dict",
        ckpt
    )

    base.load_state_dict(
        state,
        strict=False
    )

    base.eval()

    for p in base.parameters():
        p.requires_grad = False

    cnn = (
        FrozenR3D18Layer3Map()
        .to(DEVICE)
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        extract_split(
            cnn,
            base.contour_encoder,
            base.morph_encoder,
            split,
            morph_mean,
            morph_std
        )

    print(
        "\n============================================================"
    )

    print(
        "SPATIAL R3D CACHE COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        CACHE_DIR
    )

    print(
        "\nNext: ConvLSTM + attention + global-delta regression."
    )


if __name__ == "__main__":
    main()
