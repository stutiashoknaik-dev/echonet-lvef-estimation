# ============================================================
# LABEL DISTRIBUTION SMOOTHING (LDS) WEIGHTED LOSS
# Yang et al., "Delving into Deep Imbalanced Regression", ICML 2021
# ------------------------------------------------------------
# Base loss (Huber) is unchanged. This wraps it with a per-sample
# weight so rare EF values (severe <30%, hyperdynamic 70%+) get
# proportionally more gradient signal instead of being drowned out
# by the dense 50-65% band.
#
# Steps:
#   1. Bin the training EF distribution into a histogram.
#   2. Smooth it with a Gaussian kernel (this is the "LDS" part --
#      lets nearby EF values, e.g. 43 and 44, share density
#      instead of being treated as unrelated bins).
#   3. Weight = 1 / sqrt(smoothed_density) for each sample, based
#      on which bin its true EF falls into.
#   4. Multiply the per-sample Huber loss by this weight before
#      averaging.
# ============================================================

import numpy as np
import torch
import torch.nn as nn


def compute_lds_weights(ef_values, num_bins=100, ef_range=(0, 100), sigma=2.0):
    """
    ef_values: 1D array/list of EF values from the TRAINING split only.
    num_bins: histogram resolution.
    sigma: Gaussian smoothing width, in EF-percentage-points. Larger
           sigma = more sharing between nearby EF values.

    Returns:
        bin_edges: np.array, length num_bins+1
        bin_weights: np.array, length num_bins -- weight to apply to
                     any sample whose EF falls in that bin.
    """
    ef_values = np.asarray(ef_values, dtype=np.float64)
    hist, bin_edges = np.histogram(ef_values, bins=num_bins, range=ef_range)
    hist = hist.astype(np.float64)

    # Gaussian smoothing kernel over bins
    bin_width = (ef_range[1] - ef_range[0]) / num_bins
    kernel_radius = max(1, int(3 * sigma / bin_width))
    x = np.arange(-kernel_radius, kernel_radius + 1) * bin_width
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    smoothed = np.convolve(hist, kernel, mode="same")
    smoothed = np.clip(smoothed, 1.0, None)  # avoid div-by-zero for empty bins

    weights = 1.0 / np.sqrt(smoothed)
    weights = weights / weights.mean()  # normalize so average weight ~1 (keeps overall loss scale stable)

    return bin_edges, weights


def ef_to_weight(ef_batch, bin_edges, bin_weights):
    """
    ef_batch: torch.Tensor (B,) of true EF values for the current batch.
    Returns: torch.Tensor (B,) of per-sample weights.
    """
    ef_np = ef_batch.detach().cpu().numpy()
    bin_idx = np.digitize(ef_np, bin_edges[1:-1])  # maps into [0, num_bins-1]
    bin_idx = np.clip(bin_idx, 0, len(bin_weights) - 1)
    w = bin_weights[bin_idx]
    return torch.tensor(w, dtype=ef_batch.dtype, device=ef_batch.device)


class LDSWeightedHuberLoss(nn.Module):
    """
    Drop-in replacement for nn.HuberLoss. Same delta parameter, but
    reweights each sample's contribution by its LDS weight before
    averaging.
    """

    def __init__(self, bin_edges, bin_weights, delta=1.0):
        super().__init__()
        self.register_buffer("bin_edges_t", torch.tensor(bin_edges, dtype=torch.float32), persistent=False)
        self.bin_edges = bin_edges
        self.bin_weights = bin_weights
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, targets):
        per_sample_loss = self.huber(preds, targets)  # (B,)
        weights = ef_to_weight(targets, self.bin_edges, self.bin_weights)
        return (per_sample_loss * weights).mean()


if __name__ == "__main__":
    # quick sanity check with a synthetic imbalanced distribution
    np.random.seed(0)
    ef_train = np.concatenate([
        np.random.normal(58, 6, 900),   # dense middle band
        np.random.normal(20, 5, 60),    # rare severe cases
        np.random.normal(75, 4, 40),    # rare hyperdynamic cases
    ])
    bin_edges, bin_weights = compute_lds_weights(ef_train, sigma=2.0)

    loss_fn = LDSWeightedHuberLoss(bin_edges, bin_weights, delta=1.0)
    preds = torch.tensor([55.0, 20.0, 75.0], dtype=torch.float32)
    targets = torch.tensor([56.0, 25.0, 70.0], dtype=torch.float32)
    print("Weights for EF=56 (common), EF=25 (rare), EF=70 (rare):",
          ef_to_weight(targets, bin_edges, bin_weights))
    print("Weighted loss:", loss_fn(preds, targets).item())
