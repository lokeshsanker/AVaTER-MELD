"""
Video branch model: per-frame MobileNetV2 features -> BiGRU over the frame
sequence -> attention pooling -> classifier.

This replaces flat concatenation of frame features (which discards temporal
order and forces a fixed frame count) with a sequence model that can handle
variable-length sequences and actually models how the expression evolves
across frames.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import CNN_FEATURE_DIM, GRU_HIDDEN, NUM_CLASSES


class FrameEncoder(nn.Module):
    """Pretrained MobileNetV2 backbone, frozen or fine-tunable, producing a
    CNN_FEATURE_DIM-dim embedding per frame."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        backbone = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, H, W)
        f = self.features(x)
        f = self.pool(f).flatten(1)  # (B, CNN_FEATURE_DIM)
        return f


class AttentionPool(nn.Module):
    """Learned attention pooling over the time dimension of a BiGRU's output."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        # seq: (B, T, H)
        scores = self.attn(seq).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=1)        # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # (B, H)
        return pooled, weights


class VideoEmotionModel(nn.Module):
    """
    Full video branch: frame sequence (B, T, 3, H, W) -> emotion logits (B, NUM_CLASSES).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()
        self.frame_encoder = FrameEncoder(pretrained=True, freeze_backbone=freeze_backbone)
        self.gru = nn.GRU(
            input_size=CNN_FEATURE_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = AttentionPool(hidden_dim=2 * GRU_HIDDEN)
        self.classifier = nn.Sequential(
            nn.Linear(2 * GRU_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames):
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        feats = self.frame_encoder(frames_flat)          # (B*T, CNN_FEATURE_DIM)
        feats = feats.view(B, T, -1)                      # (B, T, CNN_FEATURE_DIM)

        seq_out, _ = self.gru(feats)                       # (B, T, 2*GRU_HIDDEN)
        pooled, attn_weights = self.pool(seq_out)           # (B, 2*GRU_HIDDEN)

        logits = self.classifier(pooled)                    # (B, num_classes)
        return logits, attn_weights

    @torch.no_grad()
    def predict_proba(self, frames):
        self.eval()
        logits, _ = self.forward(frames)
        return torch.softmax(logits, dim=-1)
