# ============================================================
# EXTRACT CACHED SPATIAL MAPS (for ConvLSTM)
# ------------------------------------------------------------
# Different from extract_cached_fused_sequences.py: that one
# flattens each frame's CNN feature into a single vector. ConvLSTM
# needs the SPATIAL GRID kept intact (so it can track WHERE the LV
# boundary is moving, not just THAT something changed).
#
# Caches two things per split:
#   spatial_cnn:  (N, 32, 1024, 3, 3) -- CNN+CBAM output, un-flattened
#   non_spatial:  (N, 32, 83) -- role_flag(3) + anchor shape embedding(80),
#                 kept flat since these aren't inherently spatial;
#                 broadcast across the 3x3 grid at train time instead.
#
# WARNING: this cache is much larger than the flat one -- roughly
# 1.15 MB per video (vs a few KB for the flat cache). For ~7,000
# training videos, budget approximately 8 GB of Drive space for the
# train split alone. Check available space before running this on
# a smaller/free-tier Drive account.
# ============================================================

import os
import torch
from torch.utils.data import DataLoader

from anchor_fusion_dataset_v3 import AnchorFusionEFDataset, compute_morph_norm_stats
from anchor_fusion_model_v3 import AnchorFusionDenseNetBiLSTM

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
V2_CKPT = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_spatial_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_spatial_and_nonspatial(model, video, role_flag, ed_pos, es_pos,
                                    ed_contour, es_contour, ed_morph, es_morph):
    b, t, c, h, w = video.shape
    x = video.view(b * t, c, h, w)

    feats = model.backbone.features(x)
    feats = model.cbam(feats)
    feats = torch.nn.functional.relu(feats, inplace=True)
    _, C, Hp, Wp = feats.shape
    spatial_cnn = feats.view(b, t, C, Hp, Wp)

    ed_embed = torch.cat([model.contour_encoder(ed_contour), model.morph_encoder(ed_morph)], dim=-1)
    es_embed = torch.cat([model.contour_encoder(es_contour), model.morph_encoder(es_morph)], dim=-1)
    shape_seq = torch.zeros(b, t, model.shape_embed_dim, device=video.device, dtype=spatial_cnn.dtype)
    batch_idx = torch.arange(b, device=video.device)
    shape_seq[batch_idx, ed_pos] = ed_embed
    shape_seq[batch_idx, es_pos] = es_embed
    non_spatial = torch.cat([role_flag, shape_seq], dim=-1)

    return spatial_cnn, non_spatial


def move_batch(batch, device):
    keys = ["video", "role_flag", "ed_pos", "es_pos", "ed_contour", "es_contour",
             "ed_morph", "es_morph", "global_delta", "ef"]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def extract_split(model, split, morph_mean, morph_std):
    print(f"\nExtracting cached spatial maps for split: {split}")
    ds = AnchorFusionEFDataset(split, BASE_PATH, num_frames=NUM_FRAMES, morph_mean=morph_mean, morph_std=morph_std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    all_spatial, all_nonspatial, all_global_delta, all_ef, all_ids = [], [], [], [], []
    for batch in loader:
        b = move_batch(batch, DEVICE)
        spatial_cnn, non_spatial = extract_spatial_and_nonspatial(
            model, b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
            b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"],
        )
        all_spatial.append(spatial_cnn.cpu())
        all_nonspatial.append(non_spatial.cpu())
        all_global_delta.append(b["global_delta"].cpu())
        all_ef.append(b["ef"].cpu())
        all_ids.extend(batch["video_id"])

    spatial_all = torch.cat(all_spatial, dim=0)
    nonspatial_all = torch.cat(all_nonspatial, dim=0)
    global_delta_all = torch.cat(all_global_delta, dim=0)
    ef_all = torch.cat(all_ef, dim=0)

    save_path = os.path.join(CACHE_DIR, f"{split}_spatial.pt")
    torch.save({
        "spatial_cnn": spatial_all, "non_spatial": nonspatial_all,
        "global_delta": global_delta_all, "ef": ef_all, "video_ids": all_ids,
    }, save_path)

    size_gb = spatial_all.numel() * 4 / 1e9
    print(f"Saved {spatial_all.shape[0]} sequences, spatial shape {tuple(spatial_all.shape[1:])}, "
          f"~{size_gb:.2f} GB, to {save_path}")


def main():
    print(f"Device: {DEVICE}")
    morph_mean, morph_std = compute_morph_norm_stats(BASE_PATH)

    model = AnchorFusionDenseNetBiLSTM().to(DEVICE)
    ckpt = torch.load(V2_CKPT, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    for split in ["train", "val", "test"]:
        extract_split(model, split, morph_mean, morph_std)

    print("\nDone. ConvLSTM training can now run on these spatial-map caches.")


if __name__ == "__main__":
    main()
