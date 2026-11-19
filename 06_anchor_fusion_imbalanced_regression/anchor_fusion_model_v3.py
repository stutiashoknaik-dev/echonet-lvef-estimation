import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from fds_module_v3 import FDS

# ============================================================
# CBAM, TemporalAttentionPool, backbone builder -- UNCHANGED from v2.
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class TemporalAttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x, return_weights=False):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled


def build_densenet121_grayscale(pretrained=True):
    try:
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
    except Exception:
        backbone = models.densenet121(weights=None)

    old_conv = backbone.features.conv0
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            new_conv.weight.copy_(old_conv.weight[:, :1, :, :])
    backbone.features.conv0 = new_conv
    return backbone


# ============================================================
# FULL MODEL -- v3
# CHANGES FROM v2:
#   - Adds an FDS module operating on the final pooled feature
#     vector (BiLSTM + temporal attention output, concatenated
#     with the global delta features) -- exactly where the
#     official FDS design places it: right before the regressor.
#   - forward() now optionally accepts `ef_labels` and
#     `current_epoch`. When training and both are provided:
#       (a) the pooled feature is calibrated via FDS.smooth()
#           (a no-op before start_smooth_epoch)
#       (b) the RAW (pre-calibration) pooled feature is handed to
#           FDS.collect() so end-of-epoch statistics reflect what
#           the network actually produced, not an already-smoothed
#           version of itself.
#   - Everything else (backbone, CBAM, anchor injection, BiLSTM,
#     temporal attention, LayerNorm regressor) is UNCHANGED from v2.
# ============================================================

class AnchorFusionDenseNetBiLSTM(nn.Module):
    def __init__(self, cnn_feature_dim=1024, contour_embed_dim=64, morph_embed_dim=16,
                 global_embed_dim=16, lstm_hidden=256, lstm_layers=2, dropout=0.3,
                 pretrained=True, cbam_reduction=16,
                 fds_num_bins=50, fds_sigma=2.0, fds_momentum=0.9,
                 fds_start_update_epoch=1, fds_start_smooth_epoch=2):
        super().__init__()

        self.backbone = build_densenet121_grayscale(pretrained=pretrained)
        self.cnn_feature_dim = cnn_feature_dim
        self.cbam = CBAM(channels=cnn_feature_dim, reduction=cbam_reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.contour_encoder = nn.Sequential(
            nn.LayerNorm(400),
            nn.Linear(400, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, contour_embed_dim), nn.ReLU(inplace=True),
        )
        self.morph_encoder = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, 32), nn.ReLU(inplace=True),
            nn.Linear(32, morph_embed_dim), nn.ReLU(inplace=True),
        )
        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, global_embed_dim), nn.ReLU(inplace=True),
        )

        shape_embed_dim = contour_embed_dim + morph_embed_dim
        self.shape_embed_dim = shape_embed_dim

        fused_input_dim = cnn_feature_dim + 3 + shape_embed_dim

        self.lstm = nn.LSTM(
            input_size=fused_input_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.temporal_pool = TemporalAttentionPool(input_dim=lstm_hidden * 2)

        pooled_feature_dim = lstm_hidden * 2 + global_embed_dim

        # ---- NEW: FDS operates on this final pooled feature vector ----
        self.fds = FDS(
            feature_dim=pooled_feature_dim, num_bins=fds_num_bins, sigma=fds_sigma,
            momentum=fds_momentum, start_update_epoch=fds_start_update_epoch,
            start_smooth_epoch=fds_start_smooth_epoch,
        )

        self.regressor = nn.Sequential(
            nn.Linear(pooled_feature_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def _extract_cnn_features(self, video):
        b, t, c, h, w = video.shape
        x = video.view(b * t, c, h, w)
        feats = self.backbone.features(x)
        feats = self.cbam(feats)
        feats = F.relu(feats, inplace=True)
        feats = self.pool(feats).flatten(1)
        return feats.view(b, t, self.cnn_feature_dim)

    def forward(self, video, role_flag, ed_pos, es_pos,
                ed_contour, es_contour, ed_morph, es_morph, global_delta,
                ef_labels=None, current_epoch=None):
        b, t = video.shape[0], video.shape[1]
        device = video.device

        cnn_feats = self._extract_cnn_features(video)

        ed_embed = torch.cat([self.contour_encoder(ed_contour), self.morph_encoder(ed_morph)], dim=-1)
        es_embed = torch.cat([self.contour_encoder(es_contour), self.morph_encoder(es_morph)], dim=-1)

        shape_seq = torch.zeros(b, t, self.shape_embed_dim, device=device, dtype=cnn_feats.dtype)
        batch_idx = torch.arange(b, device=device)
        shape_seq[batch_idx, ed_pos] = ed_embed
        shape_seq[batch_idx, es_pos] = es_embed

        fused = torch.cat([cnn_feats, role_flag, shape_seq], dim=-1)

        lstm_out, _ = self.lstm(fused)
        pooled = self.temporal_pool(lstm_out)

        global_embed = self.global_delta_encoder(global_delta)
        final_feat = torch.cat([pooled, global_embed], dim=-1)  # (B, pooled_feature_dim)

        # ---- FDS: collect raw stats, then (maybe) calibrate ----
        if self.training and ef_labels is not None:
            self.fds.collect(final_feat, ef_labels)
            if current_epoch is not None:
                final_feat = self.fds.smooth(final_feat, ef_labels, current_epoch)

        ef = self.regressor(final_feat).squeeze(1)
        return ef

    def load_pretrained_backbone(self, old_ckpt_path, device="cpu", strict_backbone=True):
        ckpt = torch.load(old_ckpt_path, map_location=device)
        old_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_state = {k.replace("backbone.", ""): v for k, v in old_state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=strict_backbone)
        print(f"[load_pretrained_backbone] missing={len(missing)} unexpected={len(unexpected)}")
        return missing, unexpected


if __name__ == "__main__":
    model = AnchorFusionDenseNetBiLSTM(fds_start_smooth_epoch=1)  # smooth immediately for this quick test
    model.train()
    B, T = 3, 8
    video = torch.randn(B, T, 1, 112, 112)
    role_flag = torch.zeros(B, T, 3); role_flag[:, :, 2] = 1.0
    ed_pos = torch.tensor([2, 3, 1]); es_pos = torch.tensor([5, 6, 4])
    ed_contour = torch.randn(B, 400); es_contour = torch.randn(B, 400)
    ed_morph = torch.randn(B, 10); es_morph = torch.randn(B, 10)
    global_delta = torch.randn(B, 4)
    ef_labels = torch.tensor([55.0, 22.0, 78.0])

    out = model(video, role_flag, ed_pos, es_pos, ed_contour, es_contour, ed_morph, es_morph,
                global_delta, ef_labels=ef_labels, current_epoch=1)
    print("EF output shape:", out.shape)

    loss = out.sum()
    loss.backward()
    print("Backward pass OK. Backbone grad exists:", model.backbone.features.conv0.weight.grad is not None)

    # confirm FDS accumulated the batch
    print("FDS epoch buffer size after one batch:", len(model.fds._epoch_features))
    model.fds.update_last_epoch_stats(epoch=1)
    print("FDS epoch buffer size after update_last_epoch_stats:", len(model.fds._epoch_features))
