
# ============================================================
# 3D CNN FEATURE CACHE -- FULL TRAIN / VAL / TEST
# ============================================================
# Same screening philosophy as the successful DenseNet cache:
#
#   RAW VIDEO
#      ↓
#   frozen pretrained R3D-18 (true 3D CNN)
#      ↓
#   preserve temporal feature sequence
#      ↓
#   + role flags
#   + ED/ES contour encodings
#   + ED/ES morphology encodings
#      ↓
#   cached fused sequence
#      +
#   global_delta
#      +
#   EF
#
# No training occurs here.
# The expensive 3D CNN is run once per video/split.
# Later we can train TCN / Transformer / BiLSTM heads quickly.
#
# IMPORTANT:
#   R3D-18 is cut at layer3, not layer4.
#   This keeps ~8 temporal tokens for a 32-frame input instead
#   of collapsing to ~4 tokens.
# ============================================================

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from anchor_fusion_dataset import (
    AnchorFusionEFDataset,
    compute_morph_norm_stats,
)

# Reuse the already trained v2 contour/morphology branches.
from anchor_fusion_model_v3 import AnchorFusionDenseNetBiLSTM


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BASE = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")

V2_CKPT = os.path.join(
    BASE,
    "outputs/anchor_fusion_densenet_bilstm_v2",
    "best_model.pth",
)

CACHE_DIR = os.path.join(
    BASE,
    "outputs/stage2_3dcnn_cache",
)

NUM_FRAMES = 32
IMG_SIZE = 112
BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# FROZEN R3D-18, CUT AT LAYER3
# ------------------------------------------------------------

class FrozenR3D18Layer3(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = r3d_18(
            weights=R3D_18_Weights.DEFAULT
        )

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, video):
        # video: [B,T,1,H,W]
        if video.ndim != 5 or video.shape[2] != 1:
            raise ValueError(
                f"Expected [B,T,1,H,W], got {tuple(video.shape)}"
            )

        # [B,T,1,H,W] -> [B,1,T,H,W] -> RGB
        x = video.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        x = x.repeat(1, 3, 1, 1, 1)

        mean = torch.tensor(
            [0.43216, 0.394666, 0.37645],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1, 1)

        std = torch.tensor(
            [0.22803, 0.22145, 0.216989],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1, 1)

        x = (x - mean) / std

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # [B,256,T',H',W'] -> [B,T',256]
        x = x.mean(dim=(3, 4))
        x = x.transpose(1, 2).contiguous()

        return x


# ------------------------------------------------------------
# LOAD V2 SHAPE BRANCHES
# ------------------------------------------------------------

def load_shape_encoders():
    if not os.path.exists(V2_CKPT):
        raise FileNotFoundError(
            f"v2 checkpoint not found:\n{V2_CKPT}"
        )

    model = AnchorFusionDenseNetBiLSTM().to(DEVICE)

    ckpt = torch.load(
        V2_CKPT,
        map_location=DEVICE
    )

    state = ckpt.get(
        "model_state_dict",
        ckpt
    )

    model.load_state_dict(
        state,
        strict=False
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    model.contour_encoder.eval()
    model.morph_encoder.eval()

    return (
        model.contour_encoder,
        model.morph_encoder,
        int(model.shape_embed_dim),
    )


# ------------------------------------------------------------
# RESAMPLE 32-STEP ROLE FLAGS
# ------------------------------------------------------------

def resample_role_flag(role_flag, target_t):
    x = role_flag.float().permute(0, 2, 1)
    x = torch.nn.functional.interpolate(
        x,
        size=target_t,
        mode="nearest",
    )
    return x.permute(0, 2, 1)


# ------------------------------------------------------------
# MAP ED / ES POSITIONS TO CNN TEMPORAL POSITIONS
# ------------------------------------------------------------

def map_position(pos, source_t, target_t):
    mapped = torch.floor(
        pos.float() * target_t / source_t
    ).long()
    return torch.clamp(
        mapped,
        0,
        target_t - 1,
    )


# ------------------------------------------------------------
# MULTIMODAL FUSION
# ------------------------------------------------------------

@torch.no_grad()
def make_fused_sequence(
    cnn,
    contour_encoder,
    morph_encoder,
    shape_dim,
    batch,
):
    video = batch["video"]
    role_flag = batch["role_flag"]
    ed_pos = batch["ed_pos"]
    es_pos = batch["es_pos"]

    ed_contour = batch["ed_contour"]
    es_contour = batch["es_contour"]
    ed_morph = batch["ed_morph"]
    es_morph = batch["es_morph"]

    cnn_feat = cnn(video)
    # [B,T',256]

    b, target_t, _ = cnn_feat.shape

    role = resample_role_flag(
        role_flag,
        target_t
    ).to(
        device=cnn_feat.device,
        dtype=cnn_feat.dtype,
    )

    ed_embed = torch.cat(
        [
            contour_encoder(ed_contour),
            morph_encoder(ed_morph),
        ],
        dim=-1,
    )

    es_embed = torch.cat(
        [
            contour_encoder(es_contour),
            morph_encoder(es_morph),
        ],
        dim=-1,
    )

    shape_seq = torch.zeros(
        b,
        target_t,
        shape_dim,
        device=cnn_feat.device,
        dtype=cnn_feat.dtype,
    )

    ed_target = map_position(
        ed_pos,
        NUM_FRAMES,
        target_t,
    )

    es_target = map_position(
        es_pos,
        NUM_FRAMES,
        target_t,
    )

    idx = torch.arange(
        b,
        device=cnn_feat.device,
    )

    shape_seq[idx, ed_target] = ed_embed
    shape_seq[idx, es_target] = es_embed

    return torch.cat(
        [
            cnn_feat,
            role,
            shape_seq,
        ],
        dim=-1,
    )


# ------------------------------------------------------------
# EXTRACT ONE SPLIT
# ------------------------------------------------------------

@torch.no_grad()
def extract_split(
    cnn,
    contour_encoder,
    morph_encoder,
    shape_dim,
    split,
    morph_mean,
    morph_std,
):
    print(
        f"\nExtracting 3D CNN cache for split: {split}"
    )

    ds = AnchorFusionEFDataset(
        split,
        BASE,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        morph_mean=morph_mean,
        morph_std=morph_std,
    )

    print(
        f"[{split}] {len(ds)} videos"
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    fused_all = []
    global_all = []
    ef_all = []
    ids_all = []

    start = time.time()

    for i, batch in enumerate(
        loader,
        start=1
    ):
        # Only tensor fields are moved.
        mb = {
            k: v.to(
                DEVICE,
                non_blocking=True,
            )
            if torch.is_tensor(v)
            else v
            for k, v in batch.items()
        }

        fused = make_fused_sequence(
            cnn,
            contour_encoder,
            morph_encoder,
            shape_dim,
            mb,
        )

        fused_all.append(
            fused.float().cpu()
        )

        global_all.append(
            mb["global_delta"].float().cpu()
        )

        ef_all.append(
            mb["ef"].float().cpu()
        )

        ids_all.extend(
            batch["video_id"]
        )

        if (
            i == 1
            or i % 100 == 0
            or i == len(loader)
        ):
            print(
                f"  processed {i}/{len(loader)}"
            )

    fused = torch.cat(
        fused_all,
        dim=0
    )

    global_delta = torch.cat(
        global_all,
        dim=0
    )

    ef = torch.cat(
        ef_all,
        dim=0
    )

    if not (
        fused.shape[0]
        == global_delta.shape[0]
        == ef.shape[0]
        == len(ids_all)
    ):
        raise RuntimeError(
            "Cached sample counts do not match."
        )

    out_path = os.path.join(
        CACHE_DIR,
        f"{split}_fused.pt"
    )

    torch.save(
        {
            "fused": fused,
            "global_delta": global_delta,
            "ef": ef,
            "video_ids": ids_all,
            "backbone": "r3d_18_layer3",
            "num_frames": NUM_FRAMES,
            "cnn_feature_dim": 256,
            "shape_embed_dim": shape_dim,
            "fusion_dim": int(fused.shape[-1]),
            "temporal_tokens": int(fused.shape[1]),
        },
        out_path
    )

    elapsed = (
        time.time() - start
    )

    print(
        f"Saved: {out_path}"
    )
    print(
        f"  fused: {tuple(fused.shape)}"
    )
    print(
        f"  global_delta: {tuple(global_delta.shape)}"
    )
    print(
        f"  EF: {tuple(ef.shape)}"
    )
    print(
        f"  videos: {len(ids_all)}"
    )
    print(
        f"  elapsed: {elapsed/60:.2f} min"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    print(
        "\n============================================================"
    )
    print(
        " 3D CNN CACHE: R3D-18 + SAME MULTIMODAL FUSION"
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
        "Frames:",
        NUM_FRAMES
    )

    print(
        "Backbone: pretrained R3D-18"
    )

    print(
        "Backbone: FROZEN"
    )

    print(
        "Cut point: layer3 "
        "(keeps more temporal information than layer4)"
    )

    # Train-only morphology normalization.
    morph_mean, morph_std = (
        compute_morph_norm_stats(
            BASE
        )
    )

    print(
        "Morphology normalization: TRAIN ONLY"
    )

    cnn = (
        FrozenR3D18Layer3()
        .to(DEVICE)
    )

    trainable = sum(
        p.numel()
        for p in cnn.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in cnn.parameters()
    )

    print(
        f"Trainable R3D parameters: "
        f"{trainable}/{total}"
    )

    contour_encoder, morph_encoder, shape_dim = (
        load_shape_encoders()
    )

    print(
        f"Shape embedding dimension: {shape_dim}"
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        extract_split(
            cnn,
            contour_encoder,
            morph_encoder,
            shape_dim,
            split,
            morph_mean,
            morph_std,
        )

    print(
        "\n============================================================"
    )
    print(
        "3D CNN CACHE COMPLETE"
    )
    print(
        "============================================================"
    )

    print(
        f"Cache directory:\n{CACHE_DIR}"
    )

    print(
        "\nNext heads can now be trained cheaply:"
    )
    print(
        "  1. 3D CNN + temporal attention/MLP"
    )
    print(
        "  2. 3D CNN + TCN"
    )
    print(
        "  3. 3D CNN + Transformer"
    )
    print(
        "  4. 3D CNN + BiLSTM"
    )


if __name__ == "__main__":
    main()
