import torch
import torch.nn as nn

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights
)

# ============================================================
# MULTIMODAL EF MODEL
# ============================================================

class R21DMultimodalEF(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # ====================================================
        # VIDEO BRANCH
        # ====================================================

        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None

        self.video_encoder = r2plus1d_18(
            weights=weights
        )

        old_conv = self.video_encoder.stem[0]

        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():

            new_conv.weight[:] = (
                old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        self.video_encoder.stem[0] = new_conv

        self.video_encoder.fc = nn.Identity()

        self.video_feature_dim = 512

        # ====================================================
        # CONTOUR BRANCH
        # ====================================================

        self.contour_encoder = nn.Sequential(

            nn.LayerNorm(800),

            nn.Linear(800, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU()
        )

        self.contour_feature_dim = 128

        # ====================================================
        # MASK BRANCH
        # ====================================================

        self.mask_encoder = nn.Sequential(

            nn.LayerNorm(24),

            nn.Linear(24, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),

            nn.ReLU()
        )

        self.mask_feature_dim = 32

        # ====================================================
        # FUSION
        # ====================================================

        fusion_dim = (

            self.video_feature_dim +

            self.contour_feature_dim +

            self.mask_feature_dim
        )

        self.fusion_head = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                1
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        videos,
        contours,
        masks
    ):

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        videos = videos.permute(
            0,
            2,
            1,
            3,
            4
        )

        video_feat = self.video_encoder(
            videos
        )

        # ----------------------------------------------------
        # CONTOUR
        # ----------------------------------------------------

        contour_feat = (
            self.contour_encoder(
                contours
            )
        )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask_feat = (
            self.mask_encoder(
                masks
            )
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fused = torch.cat(

            [
                video_feat,
                contour_feat,
                mask_feat
            ],

            dim=1
        )

        ef = self.fusion_head(
            fused
        )

        ef = ef.squeeze(1)

        return ef


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = R21DMultimodalEF()

    videos = torch.randn(
        4,
        32,
        1,
        112,
        112
    )

    contours = torch.randn(
        4,
        800
    )

    masks = torch.randn(
        4,
        24
    )

    preds = model(

        videos,

        contours,

        masks
    )

    print("\n===================================")
    print("MODEL TEST")
    print("===================================")

    print(
        "Video Input :",
        videos.shape
    )

    print(
        "Contour Input :",
        contours.shape
    )

    print(
        "Mask Input :",
        masks.shape
    )

    print(
        "Prediction Shape :",
        preds.shape
    )
