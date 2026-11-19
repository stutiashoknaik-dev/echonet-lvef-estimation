# ============================================================
# DEMO PIPELINE: Echocardiogram EF Prediction
# Demonstrates R(2+1)D Multimodal model, Adaptive Frame Selection & Fusion Dataset
# ============================================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add scripts directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from r21d_multimodal_model import R21DMultimodalEF
from densenet_bilstm_model import DenseNetBiLSTM

def main():
    print("=" * 60)
    print("  ECHOCARDIOGRAM EF PREDICTION RESEARCH PROJECT DEMO")
    print("=" * 60)

    device = torch.device("cpu")
    batch_size = 4
    num_frames = 32
    img_size = 112

    print("\n[1] Initializing Synthetic Batch (Simulating EchoNet-Dynamic Video Pipeline)...")
    videos = torch.randn(batch_size, num_frames, 1, img_size, img_size, dtype=torch.float32)
    contours = torch.randn(batch_size, 800, dtype=torch.float32) / 112.0  # ED 400 + ES 400 normalized coords
    masks = torch.randn(batch_size, 24, dtype=torch.float32)               # 24 Morphological mask features
    gt_ef = torch.tensor([55.4, 42.1, 68.0, 31.5], dtype=torch.float32)    # Ground Truth EF %

    print(f"  • Video Tensor Shape   : {videos.shape}  (Batch, 32 Frames, 1 Ch, 112x112)")
    print(f"  • Contour Tensor Shape : {contours.shape}  (Batch, 800 Normalized Point Coords)")
    print(f"  • Mask Tensor Shape    : {masks.shape}    (Batch, 24 Morphological Features)")
    print(f"  • Target EF Tensor     : {gt_ef.numpy().tolist()}")

    print("\n[2] Instantiating Multimodal Model (R(2+1)D + Contours MLP + Mask MLP)...")
    multimodal_model = R21DMultimodalEF(pretrained=False).to(device)
    multimodal_model.eval()

    print("\n[3] Running Forward Pass through Multimodal Fusion Network...")
    with torch.no_grad():
        preds = multimodal_model(videos, contours, masks)
        preds_clipped = torch.clamp(preds, 0.0, 100.0)

    print("\n[4] Output Predictions & Loss Assessment:")
    criterion = nn.HuberLoss(delta=1.0)
    loss = criterion(preds_clipped, gt_ef)
    mae = torch.mean(torch.abs(preds_clipped - gt_ef))

    for i in range(batch_size):
        print(f"  Sample {i+1}: GT EF = {gt_ef[i].item():.1f}% | Predicted EF = {preds_clipped[i].item():.2f}% | Abs Error = {abs(preds_clipped[i].item() - gt_ef[i].item()):.2f}%")

    print(f"\n  • Batch Loss (Huber) : {loss.item():.4f}")
    print(f"  • Batch MAE          : {mae.item():.4f}% EF")

    print("\n[5] Instantiating DenseNet121 + BiLSTM Video Model...")
    densenet_bilstm = DenseNetBiLSTM().to(device)
    densenet_bilstm.eval()

    with torch.no_grad():
        bilstm_preds = densenet_bilstm(videos)
        bilstm_preds_clipped = torch.clamp(bilstm_preds, 0.0, 100.0)

    print(f"  • DenseNet121+BiLSTM Output Predictions: {[round(float(p), 2) for p in bilstm_preds_clipped.view(-1).tolist()]}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETED SUCCESSFULLY — ALL PIPELINES OPERATIONAL!")
    print("=" * 60)

if __name__ == "__main__":
    main()
