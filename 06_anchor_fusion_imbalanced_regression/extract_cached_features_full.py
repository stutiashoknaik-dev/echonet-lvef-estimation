# ============================================================
# EXTRACT CACHED FEATURES -- FULL VERSION (train + val + test)
# ------------------------------------------------------------
# Same idea as the diagnostic version, but also caches the TEST
# split, since the full production runs need a real held-out test
# evaluation at the end, not just validation numbers.
#
# If you already ran the diagnostic version, train/val caches may
# already exist -- this script still works fine, it just re-writes
# them (harmless) and adds the missing test cache.
# ============================================================

import os
import torch
from torch.utils.data import DataLoader

from anchor_fusion_dataset_v3 import AnchorFusionEFDataset, compute_morph_norm_stats
from anchor_fusion_model_v3 import AnchorFusionDenseNetBiLSTM

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
V2_CKPT = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_feature_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_pooled_features(model, video, role_flag, ed_pos, es_pos,
                             ed_contour, es_contour, ed_morph, es_morph, global_delta):
    b, t = video.shape[0], video.shape[1]
    device = video.device
    cnn_feats = model._extract_cnn_features(video)
    ed_embed = torch.cat([model.contour_encoder(ed_contour), model.morph_encoder(ed_morph)], dim=-1)
    es_embed = torch.cat([model.contour_encoder(es_contour), model.morph_encoder(es_morph)], dim=-1)
    shape_seq = torch.zeros(b, t, model.shape_embed_dim, device=device, dtype=cnn_feats.dtype)
    batch_idx = torch.arange(b, device=device)
    shape_seq[batch_idx, ed_pos] = ed_embed
    shape_seq[batch_idx, es_pos] = es_embed
    fused = torch.cat([cnn_feats, role_flag, shape_seq], dim=-1)
    lstm_out, _ = model.lstm(fused)
    pooled = model.temporal_pool(lstm_out)
    global_embed = model.global_delta_encoder(global_delta)
    return torch.cat([pooled, global_embed], dim=-1)


def move_batch(batch, device):
    keys = ["video", "role_flag", "ed_pos", "es_pos", "ed_contour", "es_contour",
             "ed_morph", "es_morph", "global_delta", "ef"]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def extract_split(model, split, morph_mean, morph_std):
    print(f"\nExtracting cached features for split: {split}")
    ds = AnchorFusionEFDataset(split, BASE_PATH, num_frames=NUM_FRAMES,
                                morph_mean=morph_mean, morph_std=morph_std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    all_feats, all_ef, all_ids = [], [], []
    for batch in loader:
        b = move_batch(batch, DEVICE)
        feat = extract_pooled_features(
            model, b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
            b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"], b["global_delta"],
        )
        all_feats.append(feat.cpu())
        all_ef.append(b["ef"].cpu())
        all_ids.extend(batch["video_id"])

    feats = torch.cat(all_feats, dim=0)
    efs = torch.cat(all_ef, dim=0)
    save_path = os.path.join(CACHE_DIR, f"{split}_features.pt")
    torch.save({"features": feats, "ef": efs, "video_ids": all_ids}, save_path)
    print(f"Saved {feats.shape[0]} cached feature vectors ({feats.shape[1]}-dim) to {save_path}")


def main():
    print(f"Device: {DEVICE}")
    morph_mean, morph_std = compute_morph_norm_stats(BASE_PATH)

    model = AnchorFusionDenseNetBiLSTM().to(DEVICE)
    ckpt = torch.load(V2_CKPT, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded v2 checkpoint. Missing (expected: fds.*): {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()

    for split in ["train", "val", "test"]:
        extract_split(model, split, morph_mean, morph_std)

    print("\nDone. All three splits cached -- both full production scripts can now run entirely on these.")


if __name__ == "__main__":
    main()
