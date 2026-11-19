# ============================================================
# TEST v2: Anchor-Fusion DenseNet-BiLSTM
# Evaluates the best saved v2 model on the test split.
# Same clipping and metrics as your own test_anchor_fusion.py,
# now also reports the clinical bucket breakdown.
# ============================================================

import os
import math
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from anchor_fusion_dataset import AnchorFusionEFDataset, compute_morph_norm_stats
from anchor_fusion_model import AnchorFusionDenseNetBiLSTM

warnings.filterwarnings("ignore")

BASE_PATH = os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
SAVE_DIR = os.path.join(BASE_PATH, "outputs/anchor_fusion_densenet_bilstm_v2")
BEST_CKPT = os.path.join(SAVE_DIR, "best_model.pth")
OUTPUT_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
METRICS_CSV = os.path.join(SAVE_DIR, "test_metrics.csv")
BUCKET_CSV = os.path.join(SAVE_DIR, "test_bucket_metrics.csv")

NUM_FRAMES = 32
BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "Pearson_r": float(corr), "Bias": bias}


def bucket_metrics(y_true, y_pred):
    df = pd.DataFrame({"gt": y_true, "pred": y_pred})
    df["abs_error"] = (df["pred"] - df["gt"]).abs()
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["bucket"] = pd.cut(df["gt"], bins=bins, labels=labels)
    counts = df.groupby("bucket", observed=True).size()
    maes = df.groupby("bucket", observed=True)["abs_error"].mean()
    return pd.DataFrame({"n": counts, "MAE": maes}).reset_index().rename(columns={"index": "bucket"})


def move_batch(batch, device):
    keys = ["video", "role_flag", "ed_pos", "es_pos", "ed_contour", "es_contour",
            "ed_morph", "es_morph", "global_delta", "ef"]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


@torch.no_grad()
def evaluate_test(model, loader):
    model.eval()
    all_gt, all_pred, all_vid = [], [], []

    iterator = tqdm(loader, desc="Testing") if HAS_TQDM else loader
    for batch in iterator:
        b = move_batch(batch, DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(b["video"], b["role_flag"], b["ed_pos"], b["es_pos"],
                          b["ed_contour"], b["es_contour"], b["ed_morph"], b["es_morph"],
                          b["global_delta"])
        preds_np = np.clip(preds.float().cpu().numpy(), 0, 100)
        all_gt.extend(b["ef"].cpu().numpy().tolist())
        all_pred.extend(preds_np.tolist())
        all_vid.extend(batch["video_id"])

    return np.array(all_gt), np.array(all_pred), all_vid


def main():
    print("\n====================================")
    print("TESTING v2: Anchor-Fusion DenseNet-BiLSTM")
    print(f"Device: {DEVICE} | AMP: {USE_AMP}")
    print("====================================\n")

    morph_mean_path = os.path.join(SAVE_DIR, "morph_mean.npy")
    morph_std_path = os.path.join(SAVE_DIR, "morph_std.npy")
    if os.path.exists(morph_mean_path) and os.path.exists(morph_std_path):
        morph_mean = np.load(morph_mean_path)
        morph_std = np.load(morph_std_path)
        print("Loaded saved morph norm stats.")
    else:
        print("Morph norm stats not found; recomputing from training data...")
        morph_mean, morph_std = compute_morph_norm_stats(BASE_PATH)

    test_ds = AnchorFusionEFDataset(
        split="test", base_path=BASE_PATH, num_frames=NUM_FRAMES,
        morph_mean=morph_mean, morph_std=morph_std,
    )
    print(f"Test set size: {len(test_ds)} videos")

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    if not os.path.exists(BEST_CKPT):
        raise FileNotFoundError(f"Best checkpoint not found at {BEST_CKPT}. Run training first.")

    model = AnchorFusionDenseNetBiLSTM().to(DEVICE)  # no modality_dropout_p param anymore
    ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    best_val_mae = ckpt.get("best_val_mae", "?")
    print(f"Loaded best model from epoch {epoch} (Val MAE = {best_val_mae})")

    y_true, y_pred, video_ids = evaluate_test(model, test_loader)

    metrics = regression_metrics(y_true, y_pred)
    buckets = bucket_metrics(y_true, y_pred)

    print("\n--- Test Set Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\n--- Test Set Bucket Breakdown ---")
    print(buckets.to_string(index=False))

    pred_df = pd.DataFrame({"Video": video_ids, "GT_EF": y_true, "Pred_EF": y_pred})
    pred_df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
    buckets.to_csv(BUCKET_CSV, index=False)

    print(f"\nPredictions saved to {OUTPUT_CSV}")
    print(f"Metrics saved to {METRICS_CSV}")
    print(f"Bucket breakdown saved to {BUCKET_CSV}")

    print("\n====================================")
    print("TESTING COMPLETE")
    print("====================================")


if __name__ == "__main__":
    main()
