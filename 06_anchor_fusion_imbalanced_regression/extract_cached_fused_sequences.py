# ============================================================
# EXTRACT CACHED FUSED SEQUENCES (before the LSTM)
# ------------------------------------------------------------
# One step earlier than the other caching script: instead of caching
# the FINAL 528-dim vector (after LSTM+temporal attention), this
# caches the (32, 1107) fused per-timestep sequence -- CNN feature +
# role flag + anchor shape embedding -- exactly what the LSTM
# receives as input.
#
# Why: for the FDS escalation path, we want to let the LSTM and
# temporal attention adapt, while keeping the CNN backbone (the
# fragile, valuable part) completely untouched. Caching one step
# earlier than before makes that possible without ever re-running
# the CNN.
# ============================================================

import os
import torch
from torch.utils.data import DataLoader

from anchor_fusion_dataset_v3 import AnchorFusionEFDataset, compute_morph_norm_stats
from anchor_fusion_model_v3 import AnchorFusionDenseNetBiLSTM

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
V2_CKPT = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")
CACHE_DIR = os.path.join(BASE_PATH, "outputs/stage2_fused_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_fused_sequence(model, video, role_flag, ed_pos, es_pos,
                            ed_contour, es_contour, ed_morph, es_morph):
    """Everything up to (but not including) the LSTM."""
    b, t = video.shape[0], video.shape[1]
    device = video.device
    cnn_feats = model._extract_cnn_features(video)
    ed_embed = torch.cat([model.contour_encoder(ed_contour), model.morph_encoder(ed_morph)], dim=-1)
    es_embed = torch.cat([model.contour_encoder(es_contour), model.morph_encoder(es_morph)], dim=-1)
    shape_seq = torch.zeros(b, t, model.shape_embed_dim, device=device, dtype=cnn_feats.dtype)
    batch_idx = torch.arange(b, device=device)
    shape_seq[batch_idx, ed_pos] = ed_embed
    shape_seq[batch_idx, es_pos] = es_embed
    fused = torch.cat([cnn_feats, role_flag, shape_seq], dim=-1)  # (B, T, 1107)
    return fused


def move_batch(batch, device):
    keys = ["video", "role_flag", "ed_pos", "es_pos", "ed_contour", "es_contour",
             "ed_morph", "es_morph", "global_delta", "ef"]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def extract_split(model, split, morph_mean, morph_std):
    print(f"\nExtracting cached fused sequences for split: {split}")
    ds = AnchorFusionEFDataset(split, BASE_PATH, num_frames=NUM_FRAMES, morph_mean=morph_mean, morph_std=morph_std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    all_fused, all_global_delta, all_ef, all_ids = [], [], [], []
    for batch in loader:
        b = move_batch(batch, DEVICE)
        fused = extract_fused_sequence(model, b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
                                        b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"])
        all_fused.append(fused.cpu())
        all_global_delta.append(b["global_delta"].cpu())
        all_ef.append(b["ef"].cpu())
        all_ids.extend(batch["video_id"])

    fused_all = torch.cat(all_fused, dim=0)
    global_delta_all = torch.cat(all_global_delta, dim=0)
    ef_all = torch.cat(all_ef, dim=0)

    save_path = os.path.join(CACHE_DIR, f"{split}_fused.pt")
    torch.save({"fused": fused_all, "global_delta": global_delta_all, "ef": ef_all, "video_ids": all_ids}, save_path)
    print(f"Saved {fused_all.shape[0]} sequences, shape {tuple(fused_all.shape[1:])}, to {save_path}")


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

    print("\nDone. LSTM + temporal attention can now be retrained without ever re-touching the CNN backbone.")


if __name__ == "__main__":
    main()
