# ============================================================
# FEATURE DISTRIBUTION SMOOTHING (FDS)
# Yang, Zha, Chen, Wang, Katabi -- "Delving into Deep Imbalanced
# Regression", ICML 2021 (same paper that introduced LDS).
# Official reference implementation: github.com/YyzHarry/imbalanced-regression
# ------------------------------------------------------------
# WHAT THIS FIXES THAT LDS DOESN'T:
# LDS reweights the LOSS so rare EF values count for more.
# It does nothing about whether the model's internal FEATURE
# representation for those rare cases is actually well-formed --
# with few examples, the features for e.g. severe-EF hearts can
# still end up blurred toward the dominant mid-range hearts'
# feature statistics, purely because the network sees so few of
# them relative to everything else.
#
# FDS fixes this directly: it tracks the mean/variance of the
# model's own pooled feature vector, SEPARATELY for each EF bin,
# smooths those statistics across neighboring bins (same Gaussian-
# kernel idea as LDS, applied to features instead of loss weights),
# and recalibrates each sample's feature vector using the smoothed
# statistics for its bin, right before the final regression layer.
#
# HOW IT RUNS (matches the official design):
#   - During training, every batch's pooled features + EF labels
#     are appended to a running buffer (very cheap, just concatenation).
#   - At the END of each epoch, per-bin mean/variance are computed
#     from that epoch's collected features, blended into a running
#     estimate with momentum, then Gaussian-smoothed across bins.
#   - Only once training has passed `start_smooth_epoch` (giving the
#     network some epochs to first learn reasonable raw features)
#     does the calibration actually get APPLIED during forward passes.
# ============================================================


# ============================================================
# FEATURE DISTRIBUTION SMOOTHING (FDS) — CORRECTED VERSION
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDS(nn.Module):
    def __init__(self, feature_dim, ef_range=(0, 100), num_bins=50, sigma=2.0,
                 momentum=0.9, start_update_epoch=1, start_smooth_epoch=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.ef_range = ef_range
        self.num_bins = num_bins
        self.sigma = sigma
        self.momentum = momentum
        self.start_update_epoch = start_update_epoch
        self.start_smooth_epoch = start_smooth_epoch

        # Registered buffers → automatically moved to model's device
        self.register_buffer("running_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("running_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("smoothed_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("smoothed_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("num_samples_tracked", torch.zeros(num_bins))

        self._epoch_features = []
        self._epoch_labels = []

    def _bin_idx(self, labels):
        lo, hi = self.ef_range
        bin_width = (hi - lo) / self.num_bins
        idx = ((labels - lo) / bin_width).long()
        return idx.clamp(0, self.num_bins - 1)

    def collect(self, features, labels):
        """Call once per training batch. Cheap -- just buffers on CPU."""
        self._epoch_features.append(features.detach().cpu())
        self._epoch_labels.append(labels.detach().cpu())

    def update_last_epoch_stats(self, epoch):
        """Call once at the END of every training epoch."""
        if epoch < self.start_update_epoch or len(self._epoch_features) == 0:
            self._epoch_features = []
            self._epoch_labels = []
            return

        feats = torch.cat(self._epoch_features, dim=0)   # CPU
        labels = torch.cat(self._epoch_labels, dim=0)    # CPU
        bin_idx = self._bin_idx(labels)

        # Clone running stats on their own device (GPU if model is on GPU)
        new_mean = self.running_mean.clone()
        new_var = self.running_var.clone()
        device = self.running_mean.device

        for b in range(self.num_bins):
            mask = bin_idx == b
            n = int(mask.sum().item())
            if n > 1:
                bin_feats = feats[mask]  # CPU
                # Move computed statistics to the same device as the buffers
                m = bin_feats.mean(dim=0).to(device)
                v = bin_feats.var(dim=0, unbiased=False).to(device)
                new_mean[b] = self.momentum * self.running_mean[b] + (1 - self.momentum) * m
                new_var[b] = self.momentum * self.running_var[b] + (1 - self.momentum) * v
                self.num_samples_tracked[b] += n

        # In-place copy preserves buffer registration and device
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self._smooth_running_stats()

        # Clear epoch buffers
        self._epoch_features = []
        self._epoch_labels = []

    def _smooth_running_stats(self):
        # Ensure kernel is created on the same device as the running stats
        device = self.running_mean.device
        bin_width = (self.ef_range[1] - self.ef_range[0]) / self.num_bins
        kernel_radius = max(1, int(3 * self.sigma / bin_width))
        x = torch.arange(-kernel_radius, kernel_radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x * bin_width / self.sigma) ** 2)
        kernel = kernel / kernel.sum()

        def smooth_1d(stats):
            # stats: (num_bins, feature_dim) -> smooth along the bin dimension
            padded = F.pad(stats.T.unsqueeze(0), (kernel_radius, kernel_radius), mode="replicate")
            k = kernel.view(1, 1, -1).repeat(stats.shape[1], 1, 1)  # now on same device
            out = F.conv1d(padded, k, groups=stats.shape[1])
            return out.squeeze(0).T

        self.smoothed_mean = smooth_1d(self.running_mean)
        self.smoothed_var = smooth_1d(self.running_var)

    def smooth(self, features, labels, epoch):
        """Call during forward(), training mode only. No-op before start_smooth_epoch."""
        if epoch < self.start_smooth_epoch:
            return features

        bin_idx = self._bin_idx(labels)
        eps = 1e-6
        device = features.device
        run_mean = self.running_mean[bin_idx].to(device)
        run_var = self.running_var[bin_idx].to(device)
        sm_mean = self.smoothed_mean[bin_idx].to(device)
        sm_var = self.smoothed_var[bin_idx].to(device)

        calibrated = (features - run_mean) / torch.sqrt(run_var + eps) * torch.sqrt(sm_var + eps) + sm_mean
        return calibrated
