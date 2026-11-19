# ============================================================
# TEST R(2+1)D MULTIMODAL EF MODEL
# Video + Contours + Mask Features
# ============================================================

import os
import json
import time
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from fusion_dataset import FusionDataset
from r21d_multimodal_model import R21DMultimodalEF

# ============================================================
# CONFIG
# ============================================================

DEFAULT_BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
COLAB_PATH_1 = "/content/drive/MyDrive/EchoNet-Dynamic"
COLAB_PATH_2 = "/content/drive/MyDrive/EchoNet_LV"
COLAB_LOCAL = "/content/EchoNet-Dynamic"
DEFAULT_BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
LOCAL_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(COLAB_PATH_1):
    BASE_PATH = COLAB_PATH_1
elif os.path.exists(COLAB_PATH_2):
    BASE_PATH = COLAB_PATH_2
elif os.path.exists(COLAB_LOCAL):
    BASE_PATH = COLAB_LOCAL
elif os.path.exists(DEFAULT_BASE_PATH):
    BASE_PATH = DEFAULT_BASE_PATH
else:
    BASE_PATH = LOCAL_BASE_PATH

print(f"[Colab / Local Auto-Detect] Using BASE_PATH: {BASE_PATH}")

TEST_META = os.path.join(BASE_PATH, "outputs", "fusion_metadata", "fusion_test_metadata.csv")
TEST_MASK = os.path.join(BASE_PATH, "outputs", "mask_features", "test_mask_features.csv")
TEST_ED   = os.path.join(BASE_PATH, "outputs", "test_predicted_contours", "Predicted_ED_Test_Contours_100Pts.csv")
TEST_ES   = os.path.join(BASE_PATH, "outputs", "test_predicted_contours", "Predicted_ES_Test_Contours_100Pts.csv")

CHECKPOINT = os.path.join(BASE_PATH, "outputs", "r21d_multimodal", "best_model.pth")
NORM_JSON  = os.path.join(BASE_PATH, "outputs", "r21d_multimodal", "mask_norm_stats.json")
SAVE_DIR   = os.path.join(BASE_PATH, "outputs", "test_predictions")

os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_CSV = os.path.join(SAVE_DIR, "Predicted_EF_Test.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr, pval = pearsonr(y_true, y_pred)
    else:
        corr, pval = float("nan"), float("nan")

    bias = float(np.mean(y_pred - y_true))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(corr),
        "Pearson_p": float(pval),
        "Bias": bias,
    }

# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def main():
    print("\n===================================")
    print("R(2+1)D MULTIMODAL TEST INFERENCE")
    print("===================================")

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found at: {CHECKPOINT}")
        return

    # Load Normalization Stats
    if os.path.exists(NORM_JSON):
        with open(NORM_JSON, "r") as f:
            norm = json.load(f)
        mask_mean = torch.tensor(norm["mask_mean"], dtype=torch.float32)
        mask_std = torch.tensor(norm["mask_std"], dtype=torch.float32)
        print("Mask normalization loaded from JSON.")
    else:
        mask_mean = torch.zeros(24, dtype=torch.float32)
        mask_std = torch.ones(24, dtype=torch.float32)
        print("Warning: NORM_JSON not found. Using default zero-mean unit-variance.")

    test_dataset = FusionDataset(
        metadata_csv=TEST_META,
        mask_feature_csv=TEST_MASK,
        ed_contour_csv=TEST_ED,
        es_contour_csv=TEST_ES,
        use_adaptive=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print(f"Test Samples : {len(test_dataset)}")
    print(f"Total Batches : {len(test_loader)}")

    model = R21DMultimodalEF().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.HuberLoss(delta=1.0)
    predictions = []
    all_gt = []
    all_pred = []
    running_loss = 0.0
    total = 0

    mask_mean_device = mask_mean.to(DEVICE)
    mask_std_device = mask_std.to(DEVICE)

    with torch.no_grad():
        for i, (videos, contours, masks, gt_ef) in enumerate(test_loader):
            videos = videos.to(DEVICE).float()
            contours = contours.to(DEVICE).float()
            masks = masks.to(DEVICE).float()
            gt_ef = gt_ef.to(DEVICE).float()

            masks = (masks - mask_mean_device) / mask_std_device

            preds = model(videos, contours, masks)
            loss = criterion(preds, gt_ef)

            running_loss += loss.item() * videos.size(0)
            total += videos.size(0)

            preds_np = np.clip(preds.cpu().numpy(), 0, 100)
            gt_np = gt_ef.cpu().numpy()

            all_gt.extend(gt_np.tolist())
            all_pred.extend(preds_np.tolist())

            batch_start = i * BATCH_SIZE
            for j in range(len(preds_np)):
                idx = batch_start + j
                if idx < len(test_dataset.meta):
                    vname = test_dataset.meta.iloc[idx]["FileName"]
                else:
                    vname = f"Sample_{idx}"

                predictions.append({
                    "FileName": vname,
                    "GT_EF": float(gt_np[j]),
                    "Predicted_EF": round(float(preds_np[j]), 2),
                    "Absolute_Error": round(abs(float(preds_np[j]) - float(gt_np[j])), 2)
                })

    metrics = regression_metrics(all_gt, all_pred)
    print("\n===================================")
    print("TEST METRICS")
    print("===================================")
    print(f"Test Loss  : {running_loss / max(total, 1):.4f}")
    print(f"Test MAE   : {metrics['MAE']:.4f}")
    print(f"Test RMSE  : {metrics['RMSE']:.4f}")
    print(f"Test R2    : {metrics['R2']:.4f}")
    print(f"Pearson r  : {metrics['Pearson_r']:.4f}")
    print(f"Bias       : {metrics['Bias']:.4f}")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(SAVE_CSV, index=False)
    print(f"\nPredictions saved to: {SAVE_CSV}")

if __name__ == "__main__":
    main()
