import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


# ==========================================================
# DENSENET121 + BiLSTM FOR EF PREDICTION
# ==========================================================

class DenseNetBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # PRETRAINED DENSENET121
        # --------------------------------------------------

        self.densenet = densenet121(weights="DEFAULT")

        # Convert RGB -> Grayscale while preserving pretrained weights
        old_conv = self.densenet.features.conv0

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(
                dim=1,
                keepdim=True
            )

        self.densenet.features.conv0 = new_conv

        # Feature extractor only
        self.feature_extractor = self.densenet.features

        self.feature_dim = 1024

        # --------------------------------------------------
        # BiLSTM
        # --------------------------------------------------

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # --------------------------------------------------
        # REGRESSION HEAD
        # --------------------------------------------------

        self.regressor = nn.Sequential(

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        # Input:
        # (B, 32, 1, 112, 112)

        B, T, C, H, W = x.shape

        # Merge batch and temporal dimension
        x = x.view(B * T, C, H, W)

        # --------------------------------------------------
        # DenseNet Feature Extraction
        # --------------------------------------------------

        x = self.feature_extractor(x)

        x = F.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(B, T, self.feature_dim)

        # --------------------------------------------------
        # BiLSTM
        # --------------------------------------------------

        lstm_out, _ = self.bilstm(x)

        # --------------------------------------------------
        # Temporal Mean Pooling
        # Uses information from ALL 32 frames
        # --------------------------------------------------

        x = torch.mean(lstm_out, dim=1)

        # --------------------------------------------------
        # EF Prediction
        # --------------------------------------------------

        ef = self.regressor(x)

        return ef


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model = DenseNetBiLSTM()

    dummy = torch.randn(
        4,
        32,
        1,
        112,
        112
    )

    output = model(dummy)

    print("\nModel Created Successfully")

    print("Input Shape :", dummy.shape)

    print("Output Shape:", output.shape)
